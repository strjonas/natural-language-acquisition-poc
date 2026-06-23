from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .env import Action, HomeostaticSocialGrid, SilentTeacher, SituatedTeacher
from .observations import (
    TEACHER_UTTERANCES,
    observation_vector,
    observation_vector_size,
    teacher_utterance_index,
)
from .qlearning import EpisodeStats


class RecurrentActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size
        self.input = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)
        self.consequence = nn.Linear(hidden_size + action_size, hidden_size)
        self.next_needs = nn.Linear(hidden_size, 4)
        self.reward = nn.Linear(hidden_size, 1)
        self.teacher_utterance = nn.Linear(hidden_size, len(TEACHER_UTTERANCES) + 1)

    def __call__(self, observations: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.hidden_states(observations)
        logits = self.policy(hidden)
        values = self.value(hidden).squeeze(-1)
        return logits, values

    def hidden_states(self, observations: mx.array) -> mx.array:
        x = nn.relu(self.input(observations))
        return self.rnn(x)

    def predict_consequences(
        self, observations: mx.array, actions: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        hidden = self.hidden_states(observations)
        action_features = mx.eye(self.action_size)[actions]
        x = mx.concatenate([hidden, action_features], axis=-1)
        x = nn.relu(self.consequence(x))
        next_needs = mx.sigmoid(self.next_needs(x))
        rewards = self.reward(x).squeeze(-1)
        utterance_logits = self.teacher_utterance(x)
        return next_needs, rewards, utterance_logits


@dataclass(frozen=True)
class RecurrentConfig:
    condition: str = "grounded_teacher"
    episodes: int = 500
    eval_episodes: int = 20
    seed: int = 1
    hidden_size: int = 128
    learning_rate: float = 1e-3
    discount: float = 0.99
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    next_needs_weight: float = 1.0
    reward_prediction_weight: float = 0.2
    utterance_prediction_weight: float = 0.2
    viability_reward_weight: float = 0.05
    normalize_advantages: bool = True
    max_steps: int = 120
    randomize_world: bool = True
    include_object_kinds: bool = False
    log_every: int = 0


@dataclass(frozen=True)
class TrainResult:
    condition: str
    train_stats: EpisodeStats
    eval_stats: EpisodeStats


def train_condition(config: RecurrentConfig) -> TrainResult:
    rng = np.random.default_rng(config.seed)
    mx.random.seed(config.seed)

    teacher = SituatedTeacher() if config.condition == "grounded_teacher" else SilentTeacher()
    include_language = config.condition == "grounded_teacher"
    input_size = observation_vector_size(
        include_language=include_language,
        include_object_kinds=config.include_object_kinds,
    )
    model = RecurrentActorCritic(input_size, config.hidden_size, len(Action))
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    env = HomeostaticSocialGrid(
        seed=config.seed,
        max_steps=config.max_steps,
        teacher=teacher,
        randomize_world=config.randomize_world,
    )

    train_stats: list[EpisodeStats] = []
    for episode in range(config.episodes):
        trajectory, stats = collect_episode(
            env,
            model,
            seed=config.seed + episode,
            rng=rng,
            include_language=include_language,
            include_object_kinds=config.include_object_kinds,
            viability_reward_weight=config.viability_reward_weight,
            train=True,
        )

        def loss_fn(
            observations: mx.array,
            action_masks: mx.array,
            actions: mx.array,
            returns: mx.array,
            reward_targets: mx.array,
            next_needs: mx.array,
            teacher_utterances: mx.array,
            entropy_weight: float,
            value_weight: float,
            next_needs_weight: float,
            reward_prediction_weight: float,
            utterance_prediction_weight: float,
            normalize_advantages: bool,
        ) -> mx.array:
            return _episode_loss(
                model,
                observations,
                action_masks,
                actions,
                returns,
                reward_targets,
                next_needs,
                teacher_utterances,
                entropy_weight,
                value_weight,
                next_needs_weight,
                reward_prediction_weight,
                utterance_prediction_weight,
                normalize_advantages,
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(
            trajectory.observations,
            trajectory.action_masks,
            trajectory.actions,
            trajectory.returns(config.discount),
            trajectory.reward_targets,
            trajectory.next_needs,
            trajectory.teacher_utterances,
            config.entropy_weight,
            config.value_weight,
            config.next_needs_weight,
            config.reward_prediction_weight,
            config.utterance_prediction_weight,
            config.normalize_advantages,
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        train_stats.append(stats)
        if config.log_every and (episode + 1) % config.log_every == 0:
            summary = average_stats(train_stats[-config.log_every :])
            print(
                "train,"
                f"{episode + 1},"
                f"{summary.total_reward:.4f},"
                f"{summary.steps:.2f},"
                f"{summary.mean_viability:.4f},"
                f"{summary.resource_uses:.2f},"
                f"{summary.teacher_utterances:.2f}"
            )

    eval_stats = [
        collect_episode(
            env,
            model,
            seed=config.seed + config.episodes + idx,
            rng=rng,
            include_language=include_language,
            include_object_kinds=config.include_object_kinds,
            viability_reward_weight=config.viability_reward_weight,
            train=False,
        )[1]
        for idx in range(config.eval_episodes)
    ]
    window = train_stats[-min(50, len(train_stats)) :]
    return TrainResult(
        condition=config.condition,
        train_stats=average_stats(window),
        eval_stats=average_stats(eval_stats),
    )


@dataclass(frozen=True)
class Trajectory:
    observations: mx.array
    action_masks: mx.array
    actions: mx.array
    rewards: tuple[float, ...]
    reward_targets: mx.array
    next_needs: mx.array
    teacher_utterances: mx.array

    def returns(self, discount: float) -> mx.array:
        running = 0.0
        values: list[float] = []
        for reward in reversed(self.rewards):
            running = reward + discount * running
            values.append(running)
        values.reverse()
        return mx.array(values, dtype=mx.float32)


def collect_episode(
    env: HomeostaticSocialGrid,
    model: RecurrentActorCritic,
    *,
    seed: int,
    rng: np.random.Generator,
    include_language: bool,
    include_object_kinds: bool,
    viability_reward_weight: float,
    train: bool,
) -> tuple[Trajectory, EpisodeStats]:
    observation = env.reset(seed=seed)
    obs_vectors: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    reward_targets: list[float] = []
    next_needs: list[list[float]] = []
    teacher_utterance_targets: list[int] = []

    total_reward = 0.0
    viability_sum = observation.needs.mean_viability()
    min_viability = observation.needs.viability()
    resource_uses = 0
    danger_hits = 0
    teacher_utterances = 0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        vector = observation_vector(
            observation,
            width=env.width,
            height=env.height,
            include_language=include_language,
            include_object_kinds=include_object_kinds,
        )
        obs_vectors.append(vector)
        mask = action_mask(observation)
        action_masks.append(mask)

        action_index = choose_action(
            model,
            np.stack(obs_vectors),
            np.stack(action_masks),
            rng=rng,
            sample=train,
        )
        action = tuple(Action)[action_index]
        observation, reward, terminated, truncated, info = env.step(action)
        shaped_reward = float(reward) + viability_reward_weight * float(
            info["mean_viability"]
        )

        actions.append(action_index)
        rewards.append(shaped_reward)
        reward_targets.append(float(reward))
        next_needs.append(
            [
                observation.needs.food,
                observation.needs.water,
                observation.needs.energy,
                observation.needs.safety,
            ]
        )
        teacher_utterance_targets.append(
            teacher_utterance_index(observation.teacher_utterance)
        )
        total_reward += float(reward)
        steps += 1
        viability_sum += observation.needs.mean_viability()
        min_viability = min(min_viability, observation.needs.viability())

        event = info["event"]
        if event in {"consumed_water", "consumed_food", "rested_shelter"}:
            resource_uses += 1
        if event == "hit_danger":
            danger_hits += 1
        if observation.teacher_utterance:
            teacher_utterances += 1

    trajectory = Trajectory(
        observations=mx.array(np.stack(obs_vectors), dtype=mx.float32),
        action_masks=mx.array(np.stack(action_masks), dtype=mx.float32),
        actions=mx.array(actions, dtype=mx.int32),
        rewards=tuple(rewards),
        reward_targets=mx.array(reward_targets, dtype=mx.float32),
        next_needs=mx.array(next_needs, dtype=mx.float32),
        teacher_utterances=mx.array(teacher_utterance_targets, dtype=mx.int32),
    )
    stats = EpisodeStats(
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        mean_viability=viability_sum / (steps + 1),
        min_viability=min_viability,
        resource_uses=resource_uses,
        danger_hits=danger_hits,
        teacher_utterances=teacher_utterances,
    )
    return trajectory, stats


def choose_action(
    model: RecurrentActorCritic,
    observation_sequence: np.ndarray,
    action_masks: np.ndarray,
    *,
    rng: np.random.Generator,
    sample: bool,
) -> int:
    logits, _values = model(mx.array(observation_sequence, dtype=mx.float32))
    last_logits = np.asarray(logits[-1])
    last_logits = np.where(action_masks[-1] > 0.0, last_logits, -1e9)
    probs = _softmax_np(last_logits)
    if sample:
        return int(rng.choice(len(probs), p=probs))
    return int(np.argmax(probs))


def action_mask(observation) -> np.ndarray:
    mask = np.ones(len(Action), dtype=np.float32)
    if observation.object_ahead is None:
        mask[tuple(Action).index(Action.POINT)] = 0.0
        mask[tuple(Action).index(Action.ASK)] = 0.0
        mask[tuple(Action).index(Action.CONSUME)] = 0.0
    return mask


def average_stats(stats: list[EpisodeStats]) -> EpisodeStats:
    return EpisodeStats(
        total_reward=mean(stat.total_reward for stat in stats),
        steps=mean(stat.steps for stat in stats),
        terminated=mean(float(stat.terminated) for stat in stats),
        truncated=mean(float(stat.truncated) for stat in stats),
        mean_viability=mean(stat.mean_viability for stat in stats),
        min_viability=mean(stat.min_viability for stat in stats),
        resource_uses=mean(stat.resource_uses for stat in stats),
        danger_hits=mean(stat.danger_hits for stat in stats),
        teacher_utterances=mean(stat.teacher_utterances for stat in stats),
    )


def _episode_loss(
    model: RecurrentActorCritic,
    observations: mx.array,
    action_masks: mx.array,
    actions: mx.array,
    returns: mx.array,
    reward_targets: mx.array,
    next_needs: mx.array,
    teacher_utterances: mx.array,
    entropy_weight: float,
    value_weight: float,
    next_needs_weight: float,
    reward_prediction_weight: float,
    utterance_prediction_weight: float,
    normalize_advantages: bool,
) -> mx.array:
    logits, values = model(observations)
    logits = mx.where(action_masks > 0.0, logits, mx.full(logits.shape, -1e9))
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    action_log_probs = mx.take_along_axis(
        log_probs, actions[:, None], axis=-1
    ).squeeze(-1)
    probs = mx.softmax(logits, axis=-1)
    entropy_terms = mx.where(
        action_masks > 0.0,
        probs * log_probs,
        mx.zeros(logits.shape),
    )
    entropy = -mx.sum(entropy_terms, axis=-1)
    advantages = returns - values
    if normalize_advantages:
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)
    policy_loss = -mx.mean(action_log_probs * mx.stop_gradient(advantages))
    value_loss = mx.mean((returns - values) ** 2)
    entropy_loss = -mx.mean(entropy)
    predicted_needs, predicted_rewards, utterance_logits = model.predict_consequences(
        observations, actions
    )
    next_needs_loss = mx.mean((predicted_needs - next_needs) ** 2)
    reward_prediction_loss = mx.mean((predicted_rewards - reward_targets) ** 2)
    utterance_loss = _cross_entropy(utterance_logits, teacher_utterances)
    return (
        policy_loss
        + value_weight * value_loss
        + entropy_weight * entropy_loss
        + next_needs_weight * next_needs_loss
        + reward_prediction_weight * reward_prediction_loss
        + utterance_prediction_weight * utterance_loss
    )


def _cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target_log_probs = mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(
        -1
    )
    return -mx.mean(target_log_probs)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def main() -> None:
    args = _parse_args()
    print(
        "condition,total_reward,steps,mean_viability,min_viability,resource_uses,danger_hits,teacher_utterances"
    )
    for condition in args.conditions:
        config = RecurrentConfig(
            condition=condition,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            entropy_weight=args.entropy_weight,
            viability_reward_weight=args.viability_reward_weight,
            next_needs_weight=args.next_needs_weight,
            reward_prediction_weight=args.reward_prediction_weight,
            utterance_prediction_weight=args.utterance_prediction_weight,
            randomize_world=args.randomize_world,
            include_object_kinds=args.include_object_kinds,
            max_steps=args.max_steps,
            log_every=args.log_every,
        )
        result = train_condition(config)
        stats = result.eval_stats
        print(
            ",".join(
                [
                    condition,
                    f"{stats.total_reward:.4f}",
                    f"{stats.steps:.2f}",
                    f"{stats.mean_viability:.4f}",
                    f"{stats.min_viability:.4f}",
                    f"{stats.resource_uses:.2f}",
                    f"{stats.danger_hits:.2f}",
                    f"{stats.teacher_utterances:.2f}",
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--viability-reward-weight", type=float, default=0.05)
    parser.add_argument("--next-needs-weight", type=float, default=1.0)
    parser.add_argument("--reward-prediction-weight", type=float, default=0.2)
    parser.add_argument("--utterance-prediction-weight", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument(
        "--fixed-world",
        action="store_false",
        dest="randomize_world",
        help="Disable randomized object placement across episodes.",
    )
    parser.add_argument(
        "--include-object-kinds",
        action="store_true",
        help="Expose object kinds directly to the learner.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["silent_teacher", "grounded_teacher"],
        choices=["silent_teacher", "grounded_teacher"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
