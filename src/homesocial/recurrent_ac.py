from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

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
    TEACHER_UTTERANCES,
    observation_vector,
    observation_vector_size,
    teacher_utterance_index,
)
from .qlearning import EpisodeStats
from .teachers import TEACHER_MODES, build_teacher, masks_language, normalize_teacher_mode


class RecurrentActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.input = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.latent_refine = nn.Linear(hidden_size, hidden_size)
        self.latent_refine_norm = nn.LayerNorm(hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)
        self.transition = nn.Linear(hidden_size + action_size, hidden_size)
        self.transition_norm = nn.LayerNorm(hidden_size)
        self.transition_state = nn.Linear(hidden_size, hidden_size)
        self.next_observation = nn.Linear(hidden_size, input_size)
        self.next_needs = nn.Linear(hidden_size, 4)
        self.reward = nn.Linear(hidden_size, 1)
        self.teacher_utterance = nn.Linear(hidden_size, len(TEACHER_UTTERANCES) + 1)

    def __call__(self, observations: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.hidden_states(observations)
        logits = self.policy(hidden)
        values = self.value(hidden).squeeze(-1)
        return logits, values

    def hidden_states(self, observations: mx.array) -> mx.array:
        x = nn.relu(self.input_norm(self.input(observations)))
        x = x + nn.relu(self.encoder(x))
        hidden = self.rnn(x)
        return hidden + nn.relu(self.latent_refine_norm(self.latent_refine(hidden)))

    def predict_consequences(
        self, observations: mx.array, actions: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        hidden = self.hidden_states(observations)
        action_features = mx.eye(self.action_size)[actions]
        x = mx.concatenate([hidden, action_features], axis=-1)
        x = nn.relu(self.transition_norm(self.transition(x)))
        x = x + nn.relu(self.transition_state(x))
        next_observations = self.next_observation(x)
        next_needs = mx.sigmoid(self.next_needs(x))
        rewards = self.reward(x).squeeze(-1)
        utterance_logits = self.teacher_utterance(x)
        return next_observations, next_needs, rewards, utterance_logits


@dataclass(frozen=True)
class RecurrentConfig:
    condition: str = "grounded_teacher"
    include_language_channel: bool = True
    episodes: int = 500
    eval_episodes: int = 20
    seed: int = 1
    hidden_size: int = 128
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ppo_epochs: int = 3
    entropy_weight: float = 0.02
    value_weight: float = 0.5
    observation_prediction_weight: float = 0.1
    next_needs_weight: float = 1.0
    reward_prediction_weight: float = 0.2
    utterance_prediction_weight: float = 0.2
    viability_reward_weight: float = 0.05
    normalize_advantages: bool = True
    max_steps: int = 120
    width: int = 7
    height: int = 7
    randomize_world: bool = True
    include_object_kinds: bool = False
    interoception_mode: str = EXACT_INTEROCEPTION
    body_dynamics_mode: str = DETERMINISTIC_BODY
    renewable_resources: bool = False
    diagnostic_mode: str = STANDARD_MODE
    batch_size: int = 16
    log_every: int = 0
    init_from: str | None = None
    save_checkpoint: str | None = None


@dataclass(frozen=True)
class TrainResult:
    condition: str
    train_stats: EpisodeStats
    eval_stats: EpisodeStats


def train_condition(config: RecurrentConfig) -> TrainResult:
    rng = np.random.default_rng(config.seed)
    mx.random.seed(config.seed)

    teacher_mode = normalize_teacher_mode(config.condition)
    teacher = build_teacher(teacher_mode, seed=config.seed)
    mask_language = masks_language(teacher_mode)
    input_size = observation_vector_size(
        include_language=config.include_language_channel,
        include_object_kinds=config.include_object_kinds,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    model = RecurrentActorCritic(input_size, config.hidden_size, len(Action))
    if config.init_from is not None:
        model.load_weights(config.init_from)
        mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=config.seed,
        max_steps=config.max_steps,
        teacher=teacher,
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
        renewable_resources=config.renewable_resources,
    )

    train_stats: list[EpisodeStats] = []
    batch_size = max(1, config.batch_size)
    update_count = math.ceil(config.episodes / batch_size)
    for update in range(update_count):
        batch_start = update * batch_size
        batch_end = min(config.episodes, batch_start + batch_size)
        trajectories: list[Trajectory] = []
        for episode in range(batch_start, batch_end):
            trajectory, stats = collect_episode(
                env,
                model,
                seed=config.seed + episode,
                rng=rng,
                include_language=config.include_language_channel,
                mask_language=mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
                viability_reward_weight=config.viability_reward_weight,
                train=True,
            )
            trajectories.append(trajectory)
            train_stats.append(stats)

        batch = pad_trajectories(
            trajectories,
            discount=config.discount,
            gae_lambda=config.gae_lambda,
        )

        def loss_fn(
            observations: mx.array,
            next_observations: mx.array,
            action_masks: mx.array,
            step_masks: mx.array,
            actions: mx.array,
            advantages: mx.array,
            value_targets: mx.array,
            old_action_log_probs: mx.array,
            reward_targets: mx.array,
            next_needs: mx.array,
            teacher_utterances: mx.array,
            clip_ratio: float,
            entropy_weight: float,
            value_weight: float,
            observation_prediction_weight: float,
            next_needs_weight: float,
            reward_prediction_weight: float,
            utterance_prediction_weight: float,
            normalize_advantages: bool,
        ) -> mx.array:
            return _episode_loss(
                model,
                observations,
                next_observations,
                action_masks,
                step_masks,
                actions,
                advantages,
                value_targets,
                old_action_log_probs,
                reward_targets,
                next_needs,
                teacher_utterances,
                clip_ratio,
                entropy_weight,
                value_weight,
                observation_prediction_weight,
                next_needs_weight,
                reward_prediction_weight,
                utterance_prediction_weight,
                normalize_advantages,
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        ppo_epochs = max(1, config.ppo_epochs)
        for _ in range(ppo_epochs):
            loss, grads = loss_and_grad(
                batch.observations,
                batch.next_observations,
                batch.action_masks,
                batch.step_masks,
                batch.actions,
                batch.advantages,
                batch.value_targets,
                batch.old_action_log_probs,
                batch.reward_targets,
                batch.next_needs,
                batch.teacher_utterances,
                config.clip_ratio,
                config.entropy_weight,
                config.value_weight,
                config.observation_prediction_weight,
                config.next_needs_weight,
                config.reward_prediction_weight,
                config.utterance_prediction_weight,
                config.normalize_advantages,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
        if config.log_every and len(train_stats) >= config.log_every:
            logged_episodes = (len(train_stats) // config.log_every) * config.log_every
            previous_logged = (
                (len(train_stats) - len(trajectories)) // config.log_every
            ) * config.log_every
            if logged_episodes != previous_logged:
                summary = average_stats(train_stats[-config.log_every :])
                print(
                    "train,"
                    f"{logged_episodes},"
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
            include_language=config.include_language_channel,
            mask_language=mask_language,
            include_object_kinds=config.include_object_kinds,
            interoception_mode=config.interoception_mode,
            body_dynamics_mode=config.body_dynamics_mode,
            viability_reward_weight=config.viability_reward_weight,
            train=False,
        )[1]
        for idx in range(config.eval_episodes)
    ]
    if config.save_checkpoint is not None:
        save_checkpoint(model, config, config.save_checkpoint)
    window = train_stats[-min(50, len(train_stats)) :]
    return TrainResult(
        condition=config.condition,
        train_stats=average_stats(window),
        eval_stats=average_stats(eval_stats),
    )


@dataclass(frozen=True)
class Trajectory:
    observations: mx.array
    next_observations: mx.array
    action_masks: mx.array
    actions: mx.array
    rewards: tuple[float, ...]
    old_action_log_probs: mx.array
    values: mx.array
    bootstrap_value: float
    terminated: bool
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

    def generalized_advantages(
        self,
        *,
        discount: float,
        gae_lambda: float,
    ) -> tuple[mx.array, mx.array]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        running_advantage = 0.0

        for timestep in reversed(range(len(rewards))):
            if timestep == len(rewards) - 1:
                next_value = 0.0 if self.terminated else self.bootstrap_value
                nonterminal = 0.0 if self.terminated else 1.0
            else:
                next_value = values[timestep + 1]
                nonterminal = 1.0

            delta = rewards[timestep] + discount * next_value * nonterminal - values[timestep]
            running_advantage = (
                delta + discount * gae_lambda * nonterminal * running_advantage
            )
            advantages[timestep] = running_advantage

        value_targets = advantages + values
        return (
            mx.array(advantages, dtype=mx.float32),
            mx.array(value_targets, dtype=mx.float32),
        )


@dataclass(frozen=True)
class PaddedBatch:
    observations: mx.array
    next_observations: mx.array
    action_masks: mx.array
    step_masks: mx.array
    actions: mx.array
    advantages: mx.array
    value_targets: mx.array
    old_action_log_probs: mx.array
    reward_targets: mx.array
    next_needs: mx.array
    teacher_utterances: mx.array


def pad_trajectories(
    trajectories: list[Trajectory],
    *,
    discount: float,
    gae_lambda: float,
) -> PaddedBatch:
    if not trajectories:
        raise ValueError("Cannot pad an empty trajectory batch.")

    batch_size = len(trajectories)
    max_length = max(len(trajectory.rewards) for trajectory in trajectories)
    input_size = trajectories[0].observations.shape[-1]
    action_size = trajectories[0].action_masks.shape[-1]

    observations = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    next_observations = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    action_masks = np.zeros((batch_size, max_length, action_size), dtype=np.float32)
    step_masks = np.zeros((batch_size, max_length), dtype=np.float32)
    actions = np.zeros((batch_size, max_length), dtype=np.int32)
    advantages = np.zeros((batch_size, max_length), dtype=np.float32)
    value_targets = np.zeros((batch_size, max_length), dtype=np.float32)
    old_action_log_probs = np.zeros((batch_size, max_length), dtype=np.float32)
    reward_targets = np.zeros((batch_size, max_length), dtype=np.float32)
    next_needs = np.zeros((batch_size, max_length, 4), dtype=np.float32)
    teacher_utterances = np.zeros((batch_size, max_length), dtype=np.int32)

    for batch_index, trajectory in enumerate(trajectories):
        length = len(trajectory.rewards)
        trajectory_advantages, trajectory_value_targets = trajectory.generalized_advantages(
            discount=discount,
            gae_lambda=gae_lambda,
        )
        observations[batch_index, :length] = np.asarray(trajectory.observations)
        next_observations[batch_index, :length] = np.asarray(trajectory.next_observations)
        action_masks[batch_index, :length] = np.asarray(trajectory.action_masks)
        step_masks[batch_index, :length] = 1.0
        actions[batch_index, :length] = np.asarray(trajectory.actions)
        advantages[batch_index, :length] = np.asarray(trajectory_advantages)
        value_targets[batch_index, :length] = np.asarray(trajectory_value_targets)
        old_action_log_probs[batch_index, :length] = np.asarray(
            trajectory.old_action_log_probs
        )
        reward_targets[batch_index, :length] = np.asarray(trajectory.reward_targets)
        next_needs[batch_index, :length] = np.asarray(trajectory.next_needs)
        teacher_utterances[batch_index, :length] = np.asarray(
            trajectory.teacher_utterances
        )

    return PaddedBatch(
        observations=mx.array(observations, dtype=mx.float32),
        next_observations=mx.array(next_observations, dtype=mx.float32),
        action_masks=mx.array(action_masks, dtype=mx.float32),
        step_masks=mx.array(step_masks, dtype=mx.float32),
        actions=mx.array(actions, dtype=mx.int32),
        advantages=mx.array(advantages, dtype=mx.float32),
        value_targets=mx.array(value_targets, dtype=mx.float32),
        old_action_log_probs=mx.array(old_action_log_probs, dtype=mx.float32),
        reward_targets=mx.array(reward_targets, dtype=mx.float32),
        next_needs=mx.array(next_needs, dtype=mx.float32),
        teacher_utterances=mx.array(teacher_utterances, dtype=mx.int32),
    )


def collect_episode(
    env: HomeostaticSocialGrid,
    model: RecurrentActorCritic,
    *,
    seed: int,
    rng: np.random.Generator,
    include_language: bool,
    mask_language: bool = False,
    include_object_kinds: bool = False,
    interoception_mode: str = EXACT_INTEROCEPTION,
    body_dynamics_mode: str = DETERMINISTIC_BODY,
    viability_reward_weight: float,
    train: bool,
) -> tuple[Trajectory, EpisodeStats]:
    observation = env.reset(seed=seed)
    obs_vectors: list[np.ndarray] = []
    next_obs_vectors: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    old_action_log_probs: list[float] = []
    values: list[float] = []
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
            mask_language=mask_language,
            include_object_kinds=include_object_kinds,
            interoception_mode=interoception_mode,
            body_dynamics_mode=body_dynamics_mode,
        )
        obs_vectors.append(vector)
        mask = action_mask(observation)
        action_masks.append(mask)

        action_index, action_log_prob, value = choose_action_info(
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
        next_obs_vectors.append(
            observation_vector(
                observation,
                width=env.width,
                height=env.height,
                include_language=include_language,
                mask_language=mask_language,
                include_object_kinds=include_object_kinds,
                interoception_mode=interoception_mode,
                body_dynamics_mode=body_dynamics_mode,
            )
        )

        actions.append(action_index)
        old_action_log_probs.append(action_log_prob)
        values.append(value)
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
            teacher_utterance_index(
                None if mask_language or not include_language else observation.teacher_utterance
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

    bootstrap_value = 0.0
    if not terminated:
        final_vector = observation_vector(
            observation,
            width=env.width,
            height=env.height,
            include_language=include_language,
            mask_language=mask_language,
            include_object_kinds=include_object_kinds,
            interoception_mode=interoception_mode,
            body_dynamics_mode=body_dynamics_mode,
        )
        _, final_values = model(
            mx.array(np.concatenate([np.stack(obs_vectors), final_vector[None, :]]))
        )
        bootstrap_value = float(np.asarray(final_values[-1]))

    trajectory = Trajectory(
        observations=mx.array(np.stack(obs_vectors), dtype=mx.float32),
        next_observations=mx.array(np.stack(next_obs_vectors), dtype=mx.float32),
        action_masks=mx.array(np.stack(action_masks), dtype=mx.float32),
        actions=mx.array(actions, dtype=mx.int32),
        rewards=tuple(rewards),
        old_action_log_probs=mx.array(old_action_log_probs, dtype=mx.float32),
        values=mx.array(values, dtype=mx.float32),
        bootstrap_value=bootstrap_value,
        terminated=terminated,
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
    action_index, _log_prob, _value = choose_action_info(
        model,
        observation_sequence,
        action_masks,
        rng=rng,
        sample=sample,
    )
    return action_index


def choose_action_info(
    model: RecurrentActorCritic,
    observation_sequence: np.ndarray,
    action_masks: np.ndarray,
    *,
    rng: np.random.Generator,
    sample: bool,
) -> tuple[int, float, float]:
    logits, _values = model(mx.array(observation_sequence, dtype=mx.float32))
    last_logits = np.asarray(logits[-1])
    last_logits = np.where(action_masks[-1] > 0.0, last_logits, -1e9)
    probs = _softmax_np(last_logits)
    if sample:
        action_index = int(rng.choice(len(probs), p=probs))
    else:
        action_index = int(np.argmax(probs))
    action_log_prob = float(np.log(max(probs[action_index], 1e-8)))
    value = float(np.asarray(_values[-1]))
    return action_index, action_log_prob, value


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
    next_observations: mx.array,
    action_masks: mx.array,
    step_masks: mx.array,
    actions: mx.array,
    advantages: mx.array,
    value_targets: mx.array,
    old_action_log_probs: mx.array,
    reward_targets: mx.array,
    next_needs: mx.array,
    teacher_utterances: mx.array,
    clip_ratio: float,
    entropy_weight: float,
    value_weight: float,
    observation_prediction_weight: float,
    next_needs_weight: float,
    reward_prediction_weight: float,
    utterance_prediction_weight: float,
    normalize_advantages: bool,
) -> mx.array:
    logits, values = model(observations)
    logits = mx.where(action_masks > 0.0, logits, mx.full(logits.shape, -1e9))
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    action_log_probs = _select_log_probs(log_probs, actions)
    probs = mx.softmax(logits, axis=-1)
    entropy_terms = mx.where(
        action_masks > 0.0,
        probs * log_probs,
        mx.zeros(logits.shape),
    )
    entropy = -mx.sum(entropy_terms, axis=-1)
    if normalize_advantages:
        mean_advantage = _masked_mean(advantages, step_masks)
        centered = mx.where(step_masks > 0.0, advantages - mean_advantage, 0.0)
        variance = _masked_mean(centered**2, step_masks)
        advantages = centered / mx.sqrt(variance + 1e-8)
    ratios = mx.exp(action_log_probs - old_action_log_probs)
    clipped_ratios = mx.clip(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = mx.minimum(
        ratios * mx.stop_gradient(advantages),
        clipped_ratios * mx.stop_gradient(advantages),
    )
    policy_loss = -_masked_mean(surrogate, step_masks)
    value_loss = _masked_mean((value_targets - values) ** 2, step_masks)
    entropy_loss = -_masked_mean(entropy, step_masks)
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
        policy_loss
        + value_weight * value_loss
        + entropy_weight * entropy_loss
        + observation_prediction_weight * observation_prediction_loss
        + next_needs_weight * next_needs_loss
        + reward_prediction_weight * reward_prediction_loss
        + utterance_prediction_weight * utterance_loss
    )


def _cross_entropy(
    logits: mx.array,
    targets: mx.array,
    step_masks: mx.array,
) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target_log_probs = _select_log_probs(log_probs, targets)
    return -_masked_mean(target_log_probs, step_masks)


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    return mx.sum(values * mask) / (mx.sum(mask) + 1e-8)


def _select_log_probs(log_probs: mx.array, targets: mx.array) -> mx.array:
    one_hot = mx.eye(log_probs.shape[-1])[targets]
    return mx.sum(log_probs * one_hot, axis=-1)


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
            include_language_channel=args.include_language_channel,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            entropy_weight=args.entropy_weight,
            viability_reward_weight=args.viability_reward_weight,
            observation_prediction_weight=args.observation_prediction_weight,
            next_needs_weight=args.next_needs_weight,
            reward_prediction_weight=args.reward_prediction_weight,
            utterance_prediction_weight=args.utterance_prediction_weight,
            randomize_world=args.randomize_world,
            include_object_kinds=args.include_object_kinds,
            interoception_mode=args.interoception_mode,
            body_dynamics_mode=args.body_dynamics_mode,
            renewable_resources=args.renewable_resources,
            diagnostic_mode=args.diagnostic_mode,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
            log_every=args.log_every,
            init_from=args.init_from,
            save_checkpoint=args.save_checkpoint,
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--viability-reward-weight", type=float, default=0.05)
    parser.add_argument("--observation-prediction-weight", type=float, default=0.1)
    parser.add_argument("--next-needs-weight", type=float, default=1.0)
    parser.add_argument("--reward-prediction-weight", type=float, default=0.2)
    parser.add_argument("--utterance-prediction-weight", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--save-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
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
        help="Environment diagnostic mode. language_necessary hides exploitable object identity shortcuts.",
    )
    parser.add_argument(
        "--interoception-mode",
        choices=INTEROCEPTION_MODES,
        default=EXACT_INTEROCEPTION,
        help="Use exact need levels or hide them while preserving input shape.",
    )
    parser.add_argument(
        "--body-dynamics-mode",
        choices=BODY_DYNAMICS_MODES,
        default=DETERMINISTIC_BODY,
        help="Use deterministic metabolism or hidden stochastic nonlinear body dynamics.",
    )
    parser.add_argument(
        "--renewable-resources",
        action="store_true",
        help="Keep consumable resources in the world after use.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["silent", "grounded"],
        choices=[*TEACHER_MODES, "silent_teacher", "grounded_teacher"],
    )
    return parser.parse_args()


def save_checkpoint(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    path: str,
) -> None:
    weights_path = Path(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))
    _metadata_path(weights_path).write_text(json.dumps(asdict(config), indent=2) + "\n")


def _metadata_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(weights_path.suffix + ".json")


if __name__ == "__main__":
    main()
