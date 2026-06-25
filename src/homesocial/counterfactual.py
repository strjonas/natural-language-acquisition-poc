from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .agents import TeacherFollowingAgent
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint, save_checkpoint
from .observations import TEACHER_UTTERANCES, observation_vector, teacher_utterance_index
from .recurrent_ac import (
    RecurrentActorCritic,
    RecurrentConfig,
    action_mask,
    choose_action_info,
)
from .self_battery import RolloutPolicy
from .teachers import TEACHER_MODES, build_teacher, masks_language, normalize_teacher_mode


BranchSelection = Literal["probe", "all"]


@dataclass(frozen=True)
class CounterfactualSample:
    observations: mx.array
    action: int
    next_observation: mx.array
    next_needs: mx.array
    reward: float
    teacher_utterance: int


@dataclass(frozen=True)
class CounterfactualDataset:
    samples: tuple[CounterfactualSample, ...]
    config: RecurrentConfig


@dataclass(frozen=True)
class CounterfactualDecision:
    observations: mx.array
    actions: mx.array
    next_observations: mx.array
    next_needs: mx.array
    rewards: mx.array
    teacher_utterances: mx.array
    best_slot: int


@dataclass(frozen=True)
class CounterfactualDecisionDataset:
    decisions: tuple[CounterfactualDecision, ...]
    config: RecurrentConfig


@dataclass(frozen=True)
class CounterfactualBatch:
    observations: mx.array
    step_masks: mx.array
    lengths: mx.array
    actions: mx.array
    next_observations: mx.array
    next_needs: mx.array
    rewards: mx.array
    teacher_utterances: mx.array


@dataclass(frozen=True)
class CounterfactualDecisionBatch:
    observations: mx.array
    step_masks: mx.array
    lengths: mx.array
    actions: mx.array
    decision_masks: mx.array
    next_observations: mx.array
    next_needs: mx.array
    rewards: mx.array
    teacher_utterances: mx.array
    best_slots: mx.array


@dataclass(frozen=True)
class CounterfactualTrainResult:
    samples: int
    epochs: int
    final_loss: float
    checkpoint_path: str | None


def collect_counterfactual_branches(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    max_samples: int | None = None,
    rollout_policy: RolloutPolicy = "teacher",
    branch_selection: BranchSelection = "probe",
) -> CounterfactualDataset:
    normalized_mode = normalize_teacher_mode(teacher_mode)
    mask_language = masks_language(normalized_mode) or not config.include_language_channel
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=seed,
        max_steps=config.max_steps,
        teacher=build_teacher(normalized_mode, seed=seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    rng = np.random.default_rng(seed + 300_000)
    samples: list[CounterfactualSample] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        teacher_agent = TeacherFollowingAgent()
        obs_vectors: list[np.ndarray] = []
        action_masks: list[np.ndarray] = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            vector = observation_vector(
                observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                mask_language=mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
            obs_vectors.append(vector)
            mask = action_mask(observation)
            action_masks.append(mask)

            if _should_branch(observation, branch_selection):
                samples.extend(
                    _branch_samples(
                        env,
                        obs_vectors,
                        mask,
                        config=config,
                        mask_language=mask_language,
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    return CounterfactualDataset(
                        samples=tuple(samples[:max_samples]),
                        config=config,
                    )

            if rollout_policy == "teacher":
                policy_observation = observation
                if mask_language:
                    policy_observation = observation.__class__(
                        step_count=observation.step_count,
                        position=observation.position,
                        direction=observation.direction,
                        needs=observation.needs,
                        visible=observation.visible,
                        object_ahead=observation.object_ahead,
                        teacher_utterance=None,
                        last_event=observation.last_event,
                    )
                action = teacher_agent.act(policy_observation)
                action_index = tuple(Action).index(action)
                if mask[action_index] <= 0.0:
                    action = Action.MOVE_FORWARD
            else:
                action_index, _log_prob, _value = choose_action_info(
                    model,
                    np.stack(obs_vectors),
                    np.stack(action_masks),
                    rng=rng,
                    sample=False,
                )
                action = tuple(Action)[action_index]

            observation, _reward, terminated, truncated, _info = env.step(action)

    return CounterfactualDataset(samples=tuple(samples), config=config)


def collect_counterfactual_decisions(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    max_decisions: int | None = None,
    rollout_policy: RolloutPolicy = "teacher",
    branch_selection: BranchSelection = "probe",
) -> CounterfactualDecisionDataset:
    normalized_mode = normalize_teacher_mode(teacher_mode)
    mask_language = masks_language(normalized_mode) or not config.include_language_channel
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=seed,
        max_steps=config.max_steps,
        teacher=build_teacher(normalized_mode, seed=seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    rng = np.random.default_rng(seed + 350_000)
    decisions: list[CounterfactualDecision] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        teacher_agent = TeacherFollowingAgent()
        obs_vectors: list[np.ndarray] = []
        action_masks: list[np.ndarray] = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            vector = observation_vector(
                observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                mask_language=mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
            obs_vectors.append(vector)
            mask = action_mask(observation)
            action_masks.append(mask)

            if _should_branch(observation, branch_selection):
                decision = _branch_decision(
                    env,
                    obs_vectors,
                    mask,
                    config=config,
                    mask_language=mask_language,
                )
                if decision is not None:
                    decisions.append(decision)
                if max_decisions is not None and len(decisions) >= max_decisions:
                    return CounterfactualDecisionDataset(
                        decisions=tuple(decisions[:max_decisions]),
                        config=config,
                    )

            if rollout_policy == "teacher":
                policy_observation = observation
                if mask_language:
                    policy_observation = observation.__class__(
                        step_count=observation.step_count,
                        position=observation.position,
                        direction=observation.direction,
                        needs=observation.needs,
                        visible=observation.visible,
                        object_ahead=observation.object_ahead,
                        teacher_utterance=None,
                        last_event=observation.last_event,
                    )
                action = teacher_agent.act(policy_observation)
                action_index = tuple(Action).index(action)
                if mask[action_index] <= 0.0:
                    action = Action.MOVE_FORWARD
            else:
                action_index, _log_prob, _value = choose_action_info(
                    model,
                    np.stack(obs_vectors),
                    np.stack(action_masks),
                    rng=rng,
                    sample=False,
                )
                action = tuple(Action)[action_index]

            observation, _reward, terminated, truncated, _info = env.step(action)

    return CounterfactualDecisionDataset(decisions=tuple(decisions), config=config)


def train_counterfactual_consequences(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    dataset: CounterfactualDataset,
    *,
    checkpoint_path: str | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int | None = None,
    observation_prediction_weight: float = 0.05,
    next_needs_weight: float = 3.0,
    reward_prediction_weight: float = 1.0,
    utterance_prediction_weight: float = 0.2,
) -> CounterfactualTrainResult:
    if not dataset.samples:
        raise ValueError("Cannot train counterfactual consequences on an empty dataset.")

    train_seed = config.seed if seed is None else seed
    rng = np.random.default_rng(train_seed)
    mx.random.seed(train_seed)
    optimizer = optim.Adam(learning_rate=learning_rate)
    samples = list(dataset.samples)
    final_loss = 0.0

    def loss_fn(
        observations: mx.array,
        step_masks: mx.array,
        lengths: mx.array,
        actions: mx.array,
        next_observations: mx.array,
        next_needs: mx.array,
        rewards: mx.array,
        teacher_utterances: mx.array,
    ) -> mx.array:
        return _counterfactual_loss(
            model,
            observations,
            step_masks,
            lengths,
            actions,
            next_observations,
            next_needs,
            rewards,
            teacher_utterances,
            observation_prediction_weight,
            next_needs_weight,
            reward_prediction_weight,
            utterance_prediction_weight,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(samples)
        for start in range(0, len(samples), max(1, batch_size)):
            batch = pad_counterfactual_samples(samples[start : start + max(1, batch_size)])
            loss, grads = loss_and_grad(
                batch.observations,
                batch.step_masks,
                batch.lengths,
                batch.actions,
                batch.next_observations,
                batch.next_needs,
                batch.rewards,
                batch.teacher_utterances,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            final_loss = float(loss)

    if checkpoint_path is not None:
        save_checkpoint(model, config, checkpoint_path)

    return CounterfactualTrainResult(
        samples=len(dataset.samples),
        epochs=max(1, epochs),
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
    )


def train_counterfactual_ranked_consequences(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    dataset: CounterfactualDecisionDataset,
    *,
    checkpoint_path: str | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int | None = None,
    observation_prediction_weight: float = 0.05,
    next_needs_weight: float = 3.0,
    reward_prediction_weight: float = 1.0,
    utterance_prediction_weight: float = 0.2,
    rank_weight: float = 1.0,
) -> CounterfactualTrainResult:
    if not dataset.decisions:
        raise ValueError("Cannot train ranked consequences on an empty dataset.")

    train_seed = config.seed if seed is None else seed
    rng = np.random.default_rng(train_seed)
    mx.random.seed(train_seed)
    optimizer = optim.Adam(learning_rate=learning_rate)
    decisions = list(dataset.decisions)
    final_loss = 0.0

    def loss_fn(
        observations: mx.array,
        step_masks: mx.array,
        lengths: mx.array,
        actions: mx.array,
        decision_masks: mx.array,
        next_observations: mx.array,
        next_needs: mx.array,
        rewards: mx.array,
        teacher_utterances: mx.array,
        best_slots: mx.array,
    ) -> mx.array:
        return _ranked_counterfactual_loss(
            model,
            observations,
            step_masks,
            lengths,
            actions,
            decision_masks,
            next_observations,
            next_needs,
            rewards,
            teacher_utterances,
            best_slots,
            observation_prediction_weight,
            next_needs_weight,
            reward_prediction_weight,
            utterance_prediction_weight,
            rank_weight,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(decisions)
        for start in range(0, len(decisions), max(1, batch_size)):
            batch = pad_counterfactual_decisions(
                decisions[start : start + max(1, batch_size)]
            )
            loss, grads = loss_and_grad(
                batch.observations,
                batch.step_masks,
                batch.lengths,
                batch.actions,
                batch.decision_masks,
                batch.next_observations,
                batch.next_needs,
                batch.rewards,
                batch.teacher_utterances,
                batch.best_slots,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            final_loss = float(loss)

    if checkpoint_path is not None:
        save_checkpoint(model, config, checkpoint_path)

    return CounterfactualTrainResult(
        samples=sum(decision.actions.shape[0] for decision in dataset.decisions),
        epochs=max(1, epochs),
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
    )


def pad_counterfactual_samples(samples: list[CounterfactualSample]) -> CounterfactualBatch:
    if not samples:
        raise ValueError("Cannot pad an empty counterfactual batch.")

    batch_size = len(samples)
    max_length = max(sample.observations.shape[0] for sample in samples)
    input_size = samples[0].observations.shape[-1]

    observations = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    step_masks = np.zeros((batch_size, max_length), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int32)
    actions = np.zeros(batch_size, dtype=np.int32)
    next_observations = np.zeros((batch_size, input_size), dtype=np.float32)
    next_needs = np.zeros((batch_size, 4), dtype=np.float32)
    rewards = np.zeros(batch_size, dtype=np.float32)
    teacher_utterances = np.zeros(batch_size, dtype=np.int32)

    for index, sample in enumerate(samples):
        length = sample.observations.shape[0]
        observations[index, :length] = np.asarray(sample.observations)
        step_masks[index, :length] = 1.0
        lengths[index] = length
        actions[index] = sample.action
        next_observations[index] = np.asarray(sample.next_observation)
        next_needs[index] = np.asarray(sample.next_needs)
        rewards[index] = sample.reward
        teacher_utterances[index] = sample.teacher_utterance

    return CounterfactualBatch(
        observations=mx.array(observations, dtype=mx.float32),
        step_masks=mx.array(step_masks, dtype=mx.float32),
        lengths=mx.array(lengths, dtype=mx.int32),
        actions=mx.array(actions, dtype=mx.int32),
        next_observations=mx.array(next_observations, dtype=mx.float32),
        next_needs=mx.array(next_needs, dtype=mx.float32),
        rewards=mx.array(rewards, dtype=mx.float32),
        teacher_utterances=mx.array(teacher_utterances, dtype=mx.int32),
    )


def pad_counterfactual_decisions(
    decisions: list[CounterfactualDecision],
) -> CounterfactualDecisionBatch:
    if not decisions:
        raise ValueError("Cannot pad an empty counterfactual decision batch.")

    batch_size = len(decisions)
    max_length = max(decision.observations.shape[0] for decision in decisions)
    max_actions = max(decision.actions.shape[0] for decision in decisions)
    input_size = decisions[0].observations.shape[-1]

    observations = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    step_masks = np.zeros((batch_size, max_length), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int32)
    actions = np.zeros((batch_size, max_actions), dtype=np.int32)
    decision_masks = np.zeros((batch_size, max_actions), dtype=np.float32)
    next_observations = np.zeros(
        (batch_size, max_actions, input_size),
        dtype=np.float32,
    )
    next_needs = np.zeros((batch_size, max_actions, 4), dtype=np.float32)
    rewards = np.zeros((batch_size, max_actions), dtype=np.float32)
    teacher_utterances = np.zeros((batch_size, max_actions), dtype=np.int32)
    best_slots = np.zeros(batch_size, dtype=np.int32)

    for index, decision in enumerate(decisions):
        length = decision.observations.shape[0]
        action_count = decision.actions.shape[0]
        observations[index, :length] = np.asarray(decision.observations)
        step_masks[index, :length] = 1.0
        lengths[index] = length
        actions[index, :action_count] = np.asarray(decision.actions)
        decision_masks[index, :action_count] = 1.0
        next_observations[index, :action_count] = np.asarray(decision.next_observations)
        next_needs[index, :action_count] = np.asarray(decision.next_needs)
        rewards[index, :action_count] = np.asarray(decision.rewards)
        teacher_utterances[index, :action_count] = np.asarray(
            decision.teacher_utterances
        )
        best_slots[index] = decision.best_slot

    return CounterfactualDecisionBatch(
        observations=mx.array(observations, dtype=mx.float32),
        step_masks=mx.array(step_masks, dtype=mx.float32),
        lengths=mx.array(lengths, dtype=mx.int32),
        actions=mx.array(actions, dtype=mx.int32),
        decision_masks=mx.array(decision_masks, dtype=mx.float32),
        next_observations=mx.array(next_observations, dtype=mx.float32),
        next_needs=mx.array(next_needs, dtype=mx.float32),
        rewards=mx.array(rewards, dtype=mx.float32),
        teacher_utterances=mx.array(teacher_utterances, dtype=mx.int32),
        best_slots=mx.array(best_slots, dtype=mx.int32),
    )


def _should_branch(observation, branch_selection: BranchSelection) -> bool:
    if branch_selection == "all":
        return True
    if branch_selection == "probe":
        return observation.object_ahead is not None and observation.teacher_utterance is not None
    raise ValueError(f"Unknown branch selection: {branch_selection}")


def _branch_samples(
    env: HomeostaticSocialGrid,
    obs_vectors: list[np.ndarray],
    mask: np.ndarray,
    *,
    config: RecurrentConfig,
    mask_language: bool,
) -> list[CounterfactualSample]:
    samples: list[CounterfactualSample] = []
    valid_action_indices = [index for index, allowed in enumerate(mask) if allowed > 0.0]
    observations = mx.array(np.stack(obs_vectors), dtype=mx.float32)
    for action_index in valid_action_indices:
        env_copy = deepcopy(env)
        next_observation, reward, _terminated, _truncated, _info = env_copy.step(
            tuple(Action)[action_index]
        )
        next_observation_vector = observation_vector(
            next_observation,
            width=env.width,
            height=env.height,
            include_language=config.include_language_channel,
            mask_language=mask_language,
            include_object_kinds=config.include_object_kinds,
            interoception_mode=config.interoception_mode,
            body_dynamics_mode=config.body_dynamics_mode,
        )
        samples.append(
            CounterfactualSample(
                observations=observations,
                action=action_index,
                next_observation=mx.array(next_observation_vector, dtype=mx.float32),
                next_needs=mx.array(
                    [
                        next_observation.needs.food,
                        next_observation.needs.water,
                        next_observation.needs.energy,
                        next_observation.needs.safety,
                    ],
                    dtype=mx.float32,
                ),
                reward=float(reward),
                teacher_utterance=teacher_utterance_index(
                    None if mask_language else next_observation.teacher_utterance
                ),
            )
        )
    return samples


def _branch_decision(
    env: HomeostaticSocialGrid,
    obs_vectors: list[np.ndarray],
    mask: np.ndarray,
    *,
    config: RecurrentConfig,
    mask_language: bool,
) -> CounterfactualDecision | None:
    valid_action_indices = [index for index, allowed in enumerate(mask) if allowed > 0.0]
    if len(valid_action_indices) < 2:
        return None

    next_observation_vectors: list[np.ndarray] = []
    next_needs: list[list[float]] = []
    rewards: list[float] = []
    teacher_utterances: list[int] = []

    for action_index in valid_action_indices:
        env_copy = deepcopy(env)
        next_observation, reward, _terminated, _truncated, _info = env_copy.step(
            tuple(Action)[action_index]
        )
        next_observation_vectors.append(
            observation_vector(
                next_observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                mask_language=mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
        )
        next_needs.append(
            [
                next_observation.needs.food,
                next_observation.needs.water,
                next_observation.needs.energy,
                next_observation.needs.safety,
            ]
        )
        rewards.append(float(reward))
        teacher_utterances.append(
            teacher_utterance_index(
                None if mask_language else next_observation.teacher_utterance
            )
        )

    next_needs_array = np.asarray(next_needs, dtype=np.float32)
    actual_viability = np.mean(next_needs_array, axis=1)
    best_slot = int(np.argmax(actual_viability))

    return CounterfactualDecision(
        observations=mx.array(np.stack(obs_vectors), dtype=mx.float32),
        actions=mx.array(valid_action_indices, dtype=mx.int32),
        next_observations=mx.array(
            np.stack(next_observation_vectors),
            dtype=mx.float32,
        ),
        next_needs=mx.array(next_needs_array, dtype=mx.float32),
        rewards=mx.array(rewards, dtype=mx.float32),
        teacher_utterances=mx.array(teacher_utterances, dtype=mx.int32),
        best_slot=best_slot,
    )


def _counterfactual_loss(
    model: RecurrentActorCritic,
    observations: mx.array,
    step_masks: mx.array,
    lengths: mx.array,
    actions: mx.array,
    next_observations: mx.array,
    next_needs: mx.array,
    rewards: mx.array,
    teacher_utterances: mx.array,
    observation_prediction_weight: float,
    next_needs_weight: float,
    reward_prediction_weight: float,
    utterance_prediction_weight: float,
) -> mx.array:
    hidden = model.hidden_states(observations)
    hidden = mx.stop_gradient(_last_valid_hidden(hidden, lengths, step_masks))
    action_features = mx.eye(model.action_size)[actions]
    x = mx.concatenate([hidden, action_features], axis=-1)
    x = nn.relu(model.transition_norm(model.transition(x)))
    x = x + nn.relu(model.transition_state(x))
    predicted_observations = model.next_observation(x)
    predicted_needs = mx.sigmoid(model.next_needs(x))
    predicted_rewards = model.reward(x).squeeze(-1)
    utterance_logits = model.teacher_utterance(x)

    observation_loss = mx.mean((predicted_observations - next_observations) ** 2)
    needs_loss = mx.mean((predicted_needs - next_needs) ** 2)
    reward_loss = mx.mean((predicted_rewards - rewards) ** 2)
    utterance_loss = _cross_entropy(utterance_logits, teacher_utterances)
    return (
        observation_prediction_weight * observation_loss
        + next_needs_weight * needs_loss
        + reward_prediction_weight * reward_loss
        + utterance_prediction_weight * utterance_loss
    )


def _ranked_counterfactual_loss(
    model: RecurrentActorCritic,
    observations: mx.array,
    step_masks: mx.array,
    lengths: mx.array,
    actions: mx.array,
    decision_masks: mx.array,
    next_observations: mx.array,
    next_needs: mx.array,
    rewards: mx.array,
    teacher_utterances: mx.array,
    best_slots: mx.array,
    observation_prediction_weight: float,
    next_needs_weight: float,
    reward_prediction_weight: float,
    utterance_prediction_weight: float,
    rank_weight: float,
) -> mx.array:
    hidden = model.hidden_states(observations)
    hidden = mx.stop_gradient(_last_valid_hidden(hidden, lengths, step_masks))
    action_features = mx.eye(model.action_size)[actions]
    repeated_hidden = mx.broadcast_to(
        hidden[:, None, :],
        (hidden.shape[0], actions.shape[1], hidden.shape[-1]),
    )
    x = mx.concatenate([repeated_hidden, action_features], axis=-1)
    flat_x = mx.reshape(x, (-1, x.shape[-1]))
    flat_x = nn.relu(model.transition_norm(model.transition(flat_x)))
    flat_x = flat_x + nn.relu(model.transition_state(flat_x))
    predicted_observations = mx.reshape(
        model.next_observation(flat_x),
        next_observations.shape,
    )
    predicted_needs = mx.reshape(
        mx.sigmoid(model.next_needs(flat_x)),
        next_needs.shape,
    )
    predicted_rewards = mx.reshape(model.reward(flat_x).squeeze(-1), rewards.shape)
    utterance_logits = mx.reshape(
        model.teacher_utterance(flat_x),
        (*teacher_utterances.shape, len(TEACHER_UTTERANCES) + 1),
    )

    observation_loss = _masked_decision_mean(
        mx.mean((predicted_observations - next_observations) ** 2, axis=-1),
        decision_masks,
    )
    needs_loss = _masked_decision_mean(
        mx.mean((predicted_needs - next_needs) ** 2, axis=-1),
        decision_masks,
    )
    reward_loss = _masked_decision_mean(
        (predicted_rewards - rewards) ** 2,
        decision_masks,
    )
    utterance_loss = _masked_cross_entropy(
        utterance_logits,
        teacher_utterances,
        decision_masks,
    )
    predicted_viability = mx.mean(predicted_needs, axis=-1)
    rank_logits = mx.where(
        decision_masks > 0.0,
        predicted_viability * 50.0,
        mx.full(predicted_viability.shape, -1e9),
    )
    rank_loss = _cross_entropy(rank_logits, best_slots)
    return (
        observation_prediction_weight * observation_loss
        + next_needs_weight * needs_loss
        + reward_prediction_weight * reward_loss
        + utterance_prediction_weight * utterance_loss
        + rank_weight * rank_loss
    )


def _last_valid_hidden(
    hidden: mx.array,
    lengths: mx.array,
    step_masks: mx.array,
) -> mx.array:
    del step_masks
    max_length = hidden.shape[1]
    selectors = mx.eye(max_length)[lengths - 1]
    return mx.sum(hidden * selectors[..., None], axis=1)


def _cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    one_hot = mx.eye(logits.shape[-1])[targets]
    return -mx.mean(mx.sum(log_probs * one_hot, axis=-1))


def _masked_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    one_hot = mx.eye(logits.shape[-1])[targets]
    target_log_probs = mx.sum(log_probs * one_hot, axis=-1)
    return -_masked_decision_mean(target_log_probs, mask)


def _masked_decision_mean(values: mx.array, mask: mx.array) -> mx.array:
    return mx.sum(values * mask) / (mx.sum(mask) + 1e-8)


def main() -> None:
    args = _parse_args()
    model, config = load_checkpoint(args.checkpoint)
    if args.rank_weight > 0.0:
        decision_dataset = collect_counterfactual_decisions(
            model,
            config,
            episodes=args.episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            max_decisions=args.max_decisions,
            rollout_policy=args.rollout_policy,
            branch_selection=args.branch_selection,
        )
        result = train_counterfactual_ranked_consequences(
            model,
            config,
            decision_dataset,
            checkpoint_path=args.output_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            observation_prediction_weight=args.observation_prediction_weight,
            next_needs_weight=args.next_needs_weight,
            reward_prediction_weight=args.reward_prediction_weight,
            utterance_prediction_weight=args.utterance_prediction_weight,
            rank_weight=args.rank_weight,
        )
    else:
        dataset = collect_counterfactual_branches(
            model,
            config,
            episodes=args.episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            max_samples=args.max_samples,
            rollout_policy=args.rollout_policy,
            branch_selection=args.branch_selection,
        )
        result = train_counterfactual_consequences(
            model,
            config,
            dataset,
            checkpoint_path=args.output_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            observation_prediction_weight=args.observation_prediction_weight,
            next_needs_weight=args.next_needs_weight,
            reward_prediction_weight=args.reward_prediction_weight,
            utterance_prediction_weight=args.utterance_prediction_weight,
        )
    print("samples,epochs,loss,checkpoint")
    print(
        ",".join(
            [
                str(result.samples),
                str(result.epochs),
                f"{result.final_loss:.6f}",
                result.checkpoint_path or "",
            ]
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--teacher-mode", choices=TEACHER_MODES, default="grounded")
    parser.add_argument(
        "--rollout-policy",
        choices=["teacher", "model"],
        default="teacher",
    )
    parser.add_argument(
        "--branch-selection",
        choices=["probe", "all"],
        default="probe",
    )
    parser.add_argument("--observation-prediction-weight", type=float, default=0.05)
    parser.add_argument("--next-needs-weight", type=float, default=3.0)
    parser.add_argument("--reward-prediction-weight", type=float, default=1.0)
    parser.add_argument("--utterance-prediction-weight", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
