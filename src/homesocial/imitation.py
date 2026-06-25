from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .agents import TeacherFollowingAgent
from .env import (
    BODY_DYNAMICS_MODES,
    DETERMINISTIC_BODY,
    DIAGNOSTIC_MODES,
    STANDARD_MODE,
    Action,
    HomeostaticSocialGrid,
)
from .observations import (
    EXACT_INTEROCEPTION,
    INTEROCEPTION_MODES,
    observation_vector,
    observation_vector_size,
    teacher_utterance_index,
)
from .qlearning import EpisodeStats
from .recurrent_ac import (
    RecurrentActorCritic,
    RecurrentConfig,
    action_mask,
    average_stats,
    collect_episode,
)
from .teachers import TEACHER_MODES, build_teacher, masks_language, normalize_teacher_mode


@dataclass(frozen=True)
class ExpertEpisode:
    observations: mx.array
    next_observations: mx.array
    action_masks: mx.array
    actions: mx.array
    reward_targets: mx.array
    next_needs: mx.array
    teacher_utterances: mx.array
    stats: EpisodeStats


@dataclass(frozen=True)
class ExpertDataset:
    episodes: tuple[ExpertEpisode, ...]
    config: RecurrentConfig


@dataclass(frozen=True)
class ExpertBatch:
    observations: mx.array
    next_observations: mx.array
    action_masks: mx.array
    step_masks: mx.array
    actions: mx.array
    reward_targets: mx.array
    next_needs: mx.array
    teacher_utterances: mx.array


@dataclass(frozen=True)
class BCResult:
    episodes: int
    epochs: int
    final_loss: float
    expert_stats: EpisodeStats
    checkpoint_path: str | None


def collect_expert_episodes(
    config: RecurrentConfig,
    n: int,
    seed: int,
    noise: float = 0.05,
) -> ExpertDataset:
    rng = np.random.default_rng(seed)
    teacher_mode = normalize_teacher_mode(config.condition)
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=seed,
        max_steps=config.max_steps,
        teacher=build_teacher(teacher_mode, seed=seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    episodes = tuple(
        _collect_expert_episode(
            env,
            seed=seed + index,
            rng=rng,
            config=config,
            noise=noise,
            mask_language=masks_language(teacher_mode),
        )
        for index in range(n)
    )
    return ExpertDataset(episodes=episodes, config=config)


def train_bc(
    model_config: RecurrentConfig,
    dataset: ExpertDataset,
    checkpoint_path: str | None = None,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    seed: int | None = None,
) -> BCResult:
    if not dataset.episodes:
        raise ValueError("Cannot train behavior cloning on an empty dataset.")

    train_seed = model_config.seed if seed is None else seed
    rng = np.random.default_rng(train_seed)
    mx.random.seed(train_seed)

    input_size = observation_vector_size(
        include_language=model_config.include_language_channel,
        include_object_kinds=model_config.include_object_kinds,
        body_dynamics_mode=model_config.body_dynamics_mode,
    )
    model = RecurrentActorCritic(input_size, model_config.hidden_size, len(Action))
    optimizer = optim.Adam(learning_rate=model_config.learning_rate)
    episodes = list(dataset.episodes)
    final_loss = 0.0

    def loss_fn(
        observations: mx.array,
        next_observations: mx.array,
        action_masks: mx.array,
        step_masks: mx.array,
        actions: mx.array,
        reward_targets: mx.array,
        next_needs: mx.array,
        teacher_utterances: mx.array,
    ) -> mx.array:
        return _bc_loss(
            model,
            observations,
            next_observations,
            action_masks,
            step_masks,
            actions,
            reward_targets,
            next_needs,
            teacher_utterances,
            model_config.observation_prediction_weight,
            model_config.next_needs_weight,
            model_config.reward_prediction_weight,
            model_config.utterance_prediction_weight,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(episodes)
        for start in range(0, len(episodes), max(1, batch_size)):
            batch = pad_expert_episodes(episodes[start : start + max(1, batch_size)])
            loss, grads = loss_and_grad(
                batch.observations,
                batch.next_observations,
                batch.action_masks,
                batch.step_masks,
                batch.actions,
                batch.reward_targets,
                batch.next_needs,
                batch.teacher_utterances,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            final_loss = float(loss)

    if checkpoint_path is not None:
        save_checkpoint(model, model_config, checkpoint_path)

    return BCResult(
        episodes=len(dataset.episodes),
        epochs=max(1, epochs),
        final_loss=final_loss,
        expert_stats=average_stats([episode.stats for episode in dataset.episodes]),
        checkpoint_path=checkpoint_path,
    )


def evaluate_checkpoint(
    path: str,
    teacher_mode: str,
    seeds: list[int],
) -> list[EpisodeStats]:
    model, config = load_checkpoint(path)
    return evaluate_model(model, config, teacher_mode=teacher_mode, seeds=seeds)


def evaluate_model(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    teacher_mode: str,
    seeds: list[int],
) -> list[EpisodeStats]:
    normalized_mode = normalize_teacher_mode(teacher_mode)
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=config.seed,
        max_steps=config.max_steps,
        teacher=build_teacher(normalized_mode, seed=config.seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    rng = np.random.default_rng(config.seed + 100_000)
    return [
        collect_episode(
            env,
            model,
            seed=seed,
            rng=rng,
            include_language=config.include_language_channel,
            mask_language=masks_language(normalized_mode),
            include_object_kinds=config.include_object_kinds,
            interoception_mode=config.interoception_mode,
            body_dynamics_mode=config.body_dynamics_mode,
            viability_reward_weight=config.viability_reward_weight,
            train=False,
        )[1]
        for seed in seeds
    ]


def save_checkpoint(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    path: str,
) -> None:
    weights_path = Path(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))
    metadata_path = _metadata_path(weights_path)
    metadata_path.write_text(json.dumps(asdict(config), indent=2) + "\n")


def load_checkpoint(path: str) -> tuple[RecurrentActorCritic, RecurrentConfig]:
    weights_path = Path(path)
    config_data = json.loads(_metadata_path(weights_path).read_text())
    config = RecurrentConfig(**config_data)
    input_size = observation_vector_size(
        include_language=config.include_language_channel,
        include_object_kinds=config.include_object_kinds,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    model = RecurrentActorCritic(input_size, config.hidden_size, len(Action))
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())
    return model, config


def pad_expert_episodes(episodes: list[ExpertEpisode]) -> ExpertBatch:
    if not episodes:
        raise ValueError("Cannot pad an empty expert episode batch.")

    batch_size = len(episodes)
    max_length = max(episode.actions.shape[0] for episode in episodes)
    input_size = episodes[0].observations.shape[-1]
    action_size = episodes[0].action_masks.shape[-1]

    observations = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    next_observations = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    action_masks = np.zeros((batch_size, max_length, action_size), dtype=np.float32)
    step_masks = np.zeros((batch_size, max_length), dtype=np.float32)
    actions = np.zeros((batch_size, max_length), dtype=np.int32)
    reward_targets = np.zeros((batch_size, max_length), dtype=np.float32)
    next_needs = np.zeros((batch_size, max_length, 4), dtype=np.float32)
    teacher_utterances = np.zeros((batch_size, max_length), dtype=np.int32)

    for index, episode in enumerate(episodes):
        length = episode.actions.shape[0]
        observations[index, :length] = np.asarray(episode.observations)
        next_observations[index, :length] = np.asarray(episode.next_observations)
        action_masks[index, :length] = np.asarray(episode.action_masks)
        step_masks[index, :length] = 1.0
        actions[index, :length] = np.asarray(episode.actions)
        reward_targets[index, :length] = np.asarray(episode.reward_targets)
        next_needs[index, :length] = np.asarray(episode.next_needs)
        teacher_utterances[index, :length] = np.asarray(episode.teacher_utterances)

    return ExpertBatch(
        observations=mx.array(observations, dtype=mx.float32),
        next_observations=mx.array(next_observations, dtype=mx.float32),
        action_masks=mx.array(action_masks, dtype=mx.float32),
        step_masks=mx.array(step_masks, dtype=mx.float32),
        actions=mx.array(actions, dtype=mx.int32),
        reward_targets=mx.array(reward_targets, dtype=mx.float32),
        next_needs=mx.array(next_needs, dtype=mx.float32),
        teacher_utterances=mx.array(teacher_utterances, dtype=mx.int32),
    )


def _collect_expert_episode(
    env: HomeostaticSocialGrid,
    *,
    seed: int,
    rng: np.random.Generator,
    config: RecurrentConfig,
    noise: float,
    mask_language: bool,
) -> ExpertEpisode:
    agent = TeacherFollowingAgent()
    observation = env.reset(seed=seed)
    obs_vectors: list[np.ndarray] = []
    next_obs_vectors: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    actions: list[int] = []
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
        effective_mask_language = mask_language or not config.include_language_channel
        obs_vectors.append(
            observation_vector(
                observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                mask_language=effective_mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
        )
        mask = action_mask(observation)
        policy_observation = (
            replace(observation, teacher_utterance=None)
            if effective_mask_language
            else observation
        )
        action = agent.act(policy_observation)
        action_index = tuple(Action).index(action)
        if noise > 0.0 and rng.random() < noise:
            valid_actions = np.flatnonzero(mask > 0.0)
            action_index = int(rng.choice(valid_actions))
            action = tuple(Action)[action_index]
        if mask[action_index] <= 0.0:
            action = _fallback_valid_action(observation)
            action_index = tuple(Action).index(action)
        action_masks.append(mask)

        observation, reward, terminated, truncated, info = env.step(action)
        next_obs_vectors.append(
            observation_vector(
                observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                mask_language=effective_mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
        )

        actions.append(action_index)
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
            teacher_utterance_index(
                None
                if effective_mask_language
                else observation.teacher_utterance
            )
        )

        total_reward += float(reward)
        steps += 1
        viability_sum += observation.needs.mean_viability()
        min_viability = min(min_viability, observation.needs.viability())
        event = info["event"]
        if event in {"consumed_water", "consumed_food", "rested_shelter"}:
            resource_uses += 1
        if event in {"hit_danger", "consumed_danger", "rested_danger"}:
            danger_hits += 1
        if observation.teacher_utterance:
            teacher_utterances += 1

    return ExpertEpisode(
        observations=mx.array(np.stack(obs_vectors), dtype=mx.float32),
        next_observations=mx.array(np.stack(next_obs_vectors), dtype=mx.float32),
        action_masks=mx.array(np.stack(action_masks), dtype=mx.float32),
        actions=mx.array(actions, dtype=mx.int32),
        reward_targets=mx.array(reward_targets, dtype=mx.float32),
        next_needs=mx.array(next_needs, dtype=mx.float32),
        teacher_utterances=mx.array(teacher_utterance_targets, dtype=mx.int32),
        stats=EpisodeStats(
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            mean_viability=viability_sum / (steps + 1),
            min_viability=min_viability,
            resource_uses=resource_uses,
            danger_hits=danger_hits,
            teacher_utterances=teacher_utterances,
        ),
    )


def _bc_loss(
    model: RecurrentActorCritic,
    observations: mx.array,
    next_observations: mx.array,
    action_masks: mx.array,
    step_masks: mx.array,
    actions: mx.array,
    reward_targets: mx.array,
    next_needs: mx.array,
    teacher_utterances: mx.array,
    observation_prediction_weight: float,
    next_needs_weight: float,
    reward_prediction_weight: float,
    utterance_prediction_weight: float,
) -> mx.array:
    logits, _values = model(observations)
    logits = mx.where(action_masks > 0.0, logits, mx.full(logits.shape, -1e9))
    action_loss = _cross_entropy(logits, actions, step_masks)
    (
        predicted_observations,
        predicted_needs,
        predicted_rewards,
        utterance_logits,
    ) = model.predict_consequences(observations, actions)
    observation_prediction_loss = _masked_mean(
        mx.mean((predicted_observations - next_observations) ** 2, axis=-1),
        step_masks,
    )
    next_needs_loss = _masked_mean(
        mx.mean((predicted_needs - next_needs) ** 2, axis=-1),
        step_masks,
    )
    reward_prediction_loss = _masked_mean(
        (predicted_rewards - reward_targets) ** 2,
        step_masks,
    )
    utterance_loss = _cross_entropy(utterance_logits, teacher_utterances, step_masks)
    return (
        action_loss
        + observation_prediction_weight * observation_prediction_loss
        + next_needs_weight * next_needs_loss
        + reward_prediction_weight * reward_prediction_loss
        + utterance_prediction_weight * utterance_loss
    )


def _fallback_valid_action(observation) -> Action:
    if observation.object_ahead is not None:
        return Action.ASK
    if observation.last_event in {"bumped_wall", "blocked"}:
        return Action.TURN_RIGHT
    return Action.MOVE_FORWARD


def _cross_entropy(
    logits: mx.array,
    targets: mx.array,
    step_masks: mx.array,
) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    one_hot = mx.eye(log_probs.shape[-1])[targets]
    target_log_probs = mx.sum(log_probs * one_hot, axis=-1)
    return -_masked_mean(target_log_probs, step_masks)


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    return mx.sum(values * mask) / (mx.sum(mask) + 1e-8)


def _metadata_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(weights_path.suffix + ".json")


def main() -> None:
    args = _parse_args()
    config = RecurrentConfig(
        condition=args.condition,
        include_language_channel=args.include_language_channel,
        episodes=0,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        next_needs_weight=args.next_needs_weight,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        randomize_world=args.randomize_world,
        include_object_kinds=args.include_object_kinds,
        interoception_mode=args.interoception_mode,
        body_dynamics_mode=args.body_dynamics_mode,
        diagnostic_mode=args.diagnostic_mode,
        batch_size=args.batch_size,
    )
    dataset = collect_expert_episodes(
        config,
        n=args.expert_episodes,
        seed=args.seed,
        noise=args.noise,
    )
    result = train_bc(
        config,
        dataset,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        "phase,episodes,epochs,loss,mean_viability,resource_uses,danger_hits,teacher_utterances"
    )
    print(
        ",".join(
            [
                "bc_train",
                str(result.episodes),
                str(result.epochs),
                f"{result.final_loss:.4f}",
                f"{result.expert_stats.mean_viability:.4f}",
                f"{result.expert_stats.resource_uses:.2f}",
                f"{result.expert_stats.danger_hits:.2f}",
                f"{result.expert_stats.teacher_utterances:.2f}",
            ]
        )
    )
    seeds = [args.seed + args.expert_episodes + index for index in range(args.eval_episodes)]
    loaded_model, loaded_config = load_checkpoint(args.checkpoint)
    for mode in args.eval_teacher_modes:
        stats = average_stats(
            evaluate_model(
                loaded_model,
                loaded_config,
                teacher_mode=mode,
                seeds=seeds,
            )
        )
        print(
            ",".join(
                [
                    f"eval_{mode}",
                    str(args.eval_episodes),
                    "0",
                    "0.0000",
                    f"{stats.mean_viability:.4f}",
                    f"{stats.resource_uses:.2f}",
                    f"{stats.danger_hits:.2f}",
                    f"{stats.teacher_utterances:.2f}",
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--next-needs-weight", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--checkpoint", default="runs/bc_homegrid.weights.npz")
    parser.add_argument(
        "--condition",
        choices=TEACHER_MODES,
        default="grounded",
        help="Teacher mode used to collect expert trajectories.",
    )
    parser.add_argument(
        "--eval-teacher-modes",
        nargs="+",
        choices=TEACHER_MODES,
        default=["grounded", "masked", "shuffled", "wrong"],
    )
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
        "--no-language-channel",
        action="store_false",
        dest="include_language_channel",
        help="Remove the teacher-language channel from the learner input.",
    )
    parser.add_argument(
        "--diagnostic-mode",
        choices=DIAGNOSTIC_MODES,
        default=STANDARD_MODE,
    )
    parser.add_argument(
        "--interoception-mode",
        choices=INTEROCEPTION_MODES,
        default=EXACT_INTEROCEPTION,
    )
    parser.add_argument(
        "--body-dynamics-mode",
        choices=BODY_DYNAMICS_MODES,
        default=DETERMINISTIC_BODY,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
