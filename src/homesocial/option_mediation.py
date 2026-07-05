from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .attribution import _needs_array
from .env import RESOURCE_ECOLOGIES, Action, HomeostaticSocialGrid
from .imitation import load_checkpoint
from .observations import observation_vector, observation_vector_size
from .option_counterfactual_language import (
    OPTION_NAMES,
    STATE_POLICIES,
    _option_action_index,
    _option_counterfactual_features,
    _state_agent,
)
from .option_feature_intervention import OPTION_FEATURE_INTERVENTIONS
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .teachers import build_teacher, masks_language, normalize_teacher_mode


OPTION_MEDIATION_BALANCE_TARGETS = ("none", "target_option")
OPTION_MEDIATION_FEATURE_MODES = ("latent_current", "delta")
OPTION_MEDIATION_TARGET_MODES = ("oracle", "self_model")
OPTION_MEDIATION_SLOTS = 2
OPTION_MEDIATION_VOCABULARY = 4


@dataclass(frozen=True)
class OptionMediationDataset:
    features: mx.array
    option_values: mx.array
    target_options: mx.array
    current_lowest: mx.array


@dataclass(frozen=True)
class OptionMediationSourceSample:
    history: tuple[np.ndarray, ...]
    option_actions: tuple[tuple[int, ...], ...]
    option_values: np.ndarray
    target_option: int
    current_lowest: float


@dataclass(frozen=True)
class OptionMediationSourceDataset:
    samples: tuple[OptionMediationSourceSample, ...]


@dataclass(frozen=True)
class TrainedOptionMediator:
    model: OptionMediationProtocol
    feature_mean: mx.array
    feature_std: mx.array


@dataclass(frozen=True)
class TrainedOptionReceiver:
    model: "OptionMediationReceiver"


@dataclass(frozen=True)
class OptionMediationResult:
    model_control: str
    intervention: str
    samples: int
    choice_accuracy: float
    mean_chosen_lowest: float
    mean_oracle_lowest: float
    mean_regret: float
    mean_chosen_delta: float
    mean_oracle_delta: float
    message_codes_used: int
    message_code_entropy: float
    dominant_message_code_fraction: float
    target_code_mutual_information: float
    choice_code_mutual_information: float
    message_patterns_used: int
    reused_message_pattern_fraction: float
    target_pattern_mutual_information: float
    choice_pattern_mutual_information: float
    target_counts: str


class OptionMediationProtocol(nn.Module):
    def __init__(
        self,
        input_size: int,
        option_count: int,
        *,
        hidden_size: int = 96,
        receiver_size: int = 96,
        slots: int = OPTION_MEDIATION_SLOTS,
        vocabulary_size: int = OPTION_MEDIATION_VOCABULARY,
        receiver_copies: int = 1,
    ) -> None:
        super().__init__()
        self.option_count = option_count
        self.slots = slots
        self.vocabulary_size = vocabulary_size
        self.receiver_copies = max(1, receiver_copies)
        self.sender = nn.Linear(input_size, hidden_size)
        self.sender_hidden = nn.Linear(hidden_size, hidden_size)
        self.tokens = [
            nn.Linear(hidden_size, vocabulary_size) for _ in range(slots)
        ]
        receiver_input = option_count * slots * vocabulary_size
        self.receiver = nn.Linear(receiver_input, receiver_size)
        self.receiver_hidden = nn.Linear(receiver_size, receiver_size)
        self.choice = nn.Linear(receiver_size, option_count)
        self.score = nn.Linear(receiver_size, option_count)
        self.code_target = nn.Linear(slots * vocabulary_size, 1)
        self.extra_receivers = [
            nn.Linear(receiver_input, receiver_size)
            for _copy in range(self.receiver_copies - 1)
        ]
        self.extra_receiver_hiddens = [
            nn.Linear(receiver_size, receiver_size)
            for _copy in range(self.receiver_copies - 1)
        ]
        self.extra_choices = [
            nn.Linear(receiver_size, option_count)
            for _copy in range(self.receiver_copies - 1)
        ]

    def message(
        self,
        features: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[list[mx.array], list[mx.array]]:
        batch_size, option_count, feature_size = features.shape
        flat = mx.reshape(features, (batch_size * option_count, feature_size))
        hidden = nn.relu(self.sender(flat))
        hidden = hidden + nn.relu(self.sender_hidden(hidden))
        messages: list[mx.array] = []
        probabilities: list[mx.array] = []
        for token in self.tokens:
            logits = token(hidden)
            probs = mx.softmax(logits / temperature, axis=-1)
            probabilities.append(mx.reshape(probs, (batch_size, option_count, -1)))
            if hard:
                hard_message = mx.eye(self.vocabulary_size)[
                    mx.argmax(probs, axis=-1)
                ]
                message = hard_message + probs - mx.stop_gradient(probs)
            else:
                message = probs
            messages.append(mx.reshape(message, (batch_size, option_count, -1)))
        return messages, probabilities

    def receive_hidden(
        self,
        messages: list[mx.array],
        *,
        receiver_index: int = 0,
    ) -> mx.array:
        flat_messages = [
            mx.reshape(message, (message.shape[0], -1))
            for message in messages
        ]
        inputs = mx.concatenate(flat_messages, axis=-1)
        if receiver_index == 0:
            hidden = nn.relu(self.receiver(inputs))
            return hidden + nn.relu(self.receiver_hidden(hidden))
        extra_index = receiver_index - 1
        hidden = nn.relu(self.extra_receivers[extra_index](inputs))
        return hidden + nn.relu(self.extra_receiver_hiddens[extra_index](hidden))

    def receive(
        self,
        messages: list[mx.array],
        *,
        receiver_index: int = 0,
    ) -> mx.array:
        hidden = self.receive_hidden(messages, receiver_index=receiver_index)
        if receiver_index == 0:
            return self.choice(hidden)
        return self.extra_choices[receiver_index - 1](hidden)

    def receive_all(self, messages: list[mx.array]) -> list[mx.array]:
        return [
            self.receive(messages, receiver_index=receiver_index)
            for receiver_index in range(self.receiver_copies)
        ]

    def score_messages(self, messages: list[mx.array]) -> mx.array:
        return self.score(self.receive_hidden(messages))

    def code_target_logits(self, messages: list[mx.array]) -> mx.array:
        batch_size, option_count = messages[0].shape[:2]
        branch_messages = [
            mx.reshape(message, (batch_size * option_count, -1))
            for message in messages
        ]
        inputs = mx.concatenate(branch_messages, axis=-1)
        logits = self.code_target(inputs)
        return mx.reshape(logits, (batch_size, option_count))

    def __call__(self, features: mx.array) -> tuple[mx.array, list[mx.array]]:
        messages, probabilities = self.message(features, hard=True)
        return self.receive(messages), probabilities


class OptionMediationReceiver(nn.Module):
    def __init__(
        self,
        option_count: int,
        *,
        receiver_size: int = 96,
        slots: int = OPTION_MEDIATION_SLOTS,
        vocabulary_size: int = OPTION_MEDIATION_VOCABULARY,
    ) -> None:
        super().__init__()
        receiver_input = option_count * slots * vocabulary_size
        self.receiver = nn.Linear(receiver_input, receiver_size)
        self.receiver_hidden = nn.Linear(receiver_size, receiver_size)
        self.choice = nn.Linear(receiver_size, option_count)

    def receive_hidden(self, messages: list[mx.array]) -> mx.array:
        flat_messages = [
            mx.reshape(message, (message.shape[0], -1))
            for message in messages
        ]
        inputs = mx.concatenate(flat_messages, axis=-1)
        hidden = nn.relu(self.receiver(inputs))
        return hidden + nn.relu(self.receiver_hidden(hidden))

    def receive(self, messages: list[mx.array]) -> mx.array:
        return self.choice(self.receive_hidden(messages))


def collect_option_mediation_dataset(
    base_model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    rollout_mode: str = "latent_current",
    state_policy: str = "cycle",
    balance_target: str = "target_option",
    max_states: int = 3000,
    min_value_gap: float = 0.005,
    feature_mode: str = "latent_current",
    option_action_noise: float = 0.0,
) -> OptionMediationDataset:
    source = collect_option_mediation_source(
        config,
        episodes=episodes,
        seed=seed,
        teacher_mode=teacher_mode,
        horizon=horizon,
        state_policy=state_policy,
        balance_target=balance_target,
        max_states=max_states,
        min_value_gap=min_value_gap,
        option_action_noise=option_action_noise,
    )
    return option_mediation_dataset_from_source(
        source,
        base_model,
        feature_mode=feature_mode,
        rollout_mode=rollout_mode,
    )


def collect_option_mediation_source(
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    state_policy: str = "cycle",
    balance_target: str = "target_option",
    max_states: int = 3000,
    min_value_gap: float = 0.005,
    option_action_noise: float = 0.0,
) -> OptionMediationSourceDataset:
    if state_policy not in STATE_POLICIES:
        raise ValueError(f"Unknown state policy: {state_policy}.")
    if balance_target not in OPTION_MEDIATION_BALANCE_TARGETS:
        raise ValueError(f"Unknown balance target: {balance_target}.")
    if not 0.0 <= option_action_noise <= 1.0:
        raise ValueError("Option action noise must be in [0, 1].")
    normalized_teacher = normalize_teacher_mode(teacher_mode)
    mask_language = (
        masks_language(normalized_teacher) or not config.include_language_channel
    )
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=seed,
        max_steps=config.max_steps,
        teacher=build_teacher(normalized_teacher, seed=seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
        renewable_resources=config.renewable_resources,
        resource_ecology=config.resource_ecology,
    )
    rng = np.random.default_rng(seed + 3_170_000)
    samples: list[OptionMediationSourceSample] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = _state_agent(state_policy, seed=seed, episode=episode)
        history: list[np.ndarray] = []
        terminated = False
        truncated = False
        while not terminated and not truncated and len(samples) < max_states:
            history.append(
                observation_vector(
                    observation,
                    width=env.width,
                    height=env.height,
                    include_language=config.include_language_channel,
                    mask_language=mask_language,
                    include_object_kinds=config.include_object_kinds,
                    interoception_mode=config.interoception_mode,
                    body_dynamics_mode=config.body_dynamics_mode,
                )
            )
            option_actions, values = _state_option_actions_and_values(
                env,
                observation,
                horizon=max(1, horizon),
                rng=rng,
                option_action_noise=option_action_noise,
            )
            ordered_values = np.sort(values)
            if (
                len(ordered_values) < 2
                or ordered_values[-1] - ordered_values[-2] >= min_value_gap
            ):
                samples.append(
                    OptionMediationSourceSample(
                        history=tuple(
                            np.asarray(item, dtype=np.float32) for item in history
                        ),
                        option_actions=tuple(
                            tuple(int(action) for action in actions)
                            for actions in option_actions
                        ),
                        option_values=values.astype(np.float32),
                        target_option=int(np.argmax(values)),
                        current_lowest=float(np.min(_needs_array(observation.needs))),
                    )
                )

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
            action = agent.act(policy_observation)
            action_index = tuple(Action).index(action)
            if action_mask(observation)[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)
        if len(samples) >= max_states:
            break

    if not samples:
        raise ValueError("No option mediation states were collected.")

    target_options = [sample.target_option for sample in samples]
    indices = np.arange(len(samples))
    if balance_target == "target_option":
        indices = _balanced_target_option_indices(target_options, rng)

    return OptionMediationSourceDataset(
        samples=tuple(samples[int(index)] for index in indices)
    )


def option_mediation_dataset_from_source(
    source: OptionMediationSourceDataset,
    base_model: RecurrentActorCritic,
    *,
    feature_mode: str = "latent_current",
    rollout_mode: str = "latent_current",
) -> OptionMediationDataset:
    if feature_mode not in OPTION_MEDIATION_FEATURE_MODES:
        raise ValueError(f"Unknown option mediation feature mode: {feature_mode}.")
    if rollout_mode != "latent_current":
        raise ValueError("Option mediation currently requires latent_current features.")
    grouped_features: list[np.ndarray] = []
    grouped_values: list[np.ndarray] = []
    target_options: list[int] = []
    current_lowest: list[float] = []

    for sample in source.samples:
        history = [np.asarray(item, dtype=np.float32) for item in sample.history]
        grouped_features.append(
            np.stack(
                [
                    _select_option_mediation_features(
                        _option_counterfactual_features(
                            base_model,
                            history,
                            option_actions=list(actions),
                            rollout_mode=rollout_mode,
                        ),
                        feature_mode=feature_mode,
                    )
                    for actions in sample.option_actions
                ]
            ).astype(np.float32)
        )
        grouped_values.append(np.asarray(sample.option_values, dtype=np.float32))
        target_options.append(sample.target_option)
        current_lowest.append(sample.current_lowest)

    return OptionMediationDataset(
        features=mx.array(
            np.stack(grouped_features),
            dtype=mx.float32,
        ),
        option_values=mx.array(
            np.stack(grouped_values),
            dtype=mx.float32,
        ),
        target_options=mx.array(
            target_options,
            dtype=mx.int32,
        ),
        current_lowest=mx.array(
            current_lowest,
            dtype=mx.float32,
        ),
    )


def intervene_option_mediation_features(
    dataset: OptionMediationDataset,
    *,
    intervention: str,
    seed: int = 1,
) -> OptionMediationDataset:
    if intervention not in OPTION_FEATURE_INTERVENTIONS:
        raise ValueError(f"Unknown option feature intervention: {intervention}.")
    features = np.asarray(dataset.features).copy()
    rng = np.random.default_rng(seed)
    blocks = _feature_blocks(features.shape[-1])
    if intervention.startswith("zero_"):
        block_name = intervention.removeprefix("zero_")
        if block_name not in blocks:
            raise ValueError(
                f"Intervention {intervention} is invalid for feature width "
                f"{features.shape[-1]}."
            )
        features[:, :, blocks[block_name]] = 0.0
    elif intervention.startswith("shuffle_"):
        block_name = intervention.removeprefix("shuffle_")
        if block_name not in blocks:
            raise ValueError(
                f"Intervention {intervention} is invalid for feature width "
                f"{features.shape[-1]}."
            )
        block = blocks[block_name]
        flat = features[:, :, block].reshape((-1, block.stop - block.start))
        rng.shuffle(flat)
        features[:, :, block] = flat.reshape(features[:, :, block].shape)
    elif intervention == "negate_delta":
        features[:, :, blocks["delta"]] *= -1.0
    return OptionMediationDataset(
        features=mx.array(features, dtype=mx.float32),
        option_values=dataset.option_values,
        target_options=dataset.target_options,
        current_lowest=dataset.current_lowest,
    )


def train_option_mediator(
    dataset: OptionMediationDataset,
    *,
    hidden_size: int = 96,
    receiver_size: int = 96,
    receiver_copies: int = 1,
    slots: int = OPTION_MEDIATION_SLOTS,
    vocabulary_size: int = OPTION_MEDIATION_VOCABULARY,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    balance_weight: float = 0.02,
    entropy_weight: float = 0.0,
    message_commitment_weight: float = 0.0,
    message_temperature: float = 0.6,
    soft_message_training: bool = False,
    score_targets: np.ndarray | None = None,
    score_pretrain_epochs: int = 0,
    score_pretrain_commitment_weight: float | None = None,
    frozen_receiver_epochs: int = 0,
    score_distillation_weight: float = 0.0,
    score_distillation_temperature: float = 1.0,
    score_rank_weight: float = 0.0,
    score_reconstruction_weight: float = 0.0,
    code_target_weight: float = 0.0,
    message_replay_weight: float = 0.0,
    score_rank_code_weight: float = 0.0,
    score_rank_code_slot: int = 0,
    seed: int = 1,
) -> TrainedOptionMediator:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    feature_mean = mx.mean(dataset.features, axis=(0, 1), keepdims=True)
    feature_std = mx.sqrt(
        mx.mean((dataset.features - feature_mean) ** 2, axis=(0, 1), keepdims=True)
        + 1e-6
    )
    features = (dataset.features - feature_mean) / feature_std
    sample_count = int(features.shape[0])
    option_count = int(features.shape[1])
    model = OptionMediationProtocol(
        int(features.shape[-1]),
        option_count,
        hidden_size=hidden_size,
        receiver_size=receiver_size,
        receiver_copies=receiver_copies,
        slots=slots,
        vocabulary_size=vocabulary_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    indices = np.arange(sample_count)
    pretrain_commitment_weight = (
        message_commitment_weight
        if score_pretrain_commitment_weight is None
        else score_pretrain_commitment_weight
    )
    normalized_score_targets: mx.array | None = None
    if (
        score_reconstruction_weight > 0.0
        or score_pretrain_epochs > 0
        or score_distillation_weight > 0.0
        or score_rank_weight > 0.0
        or score_rank_code_weight > 0.0
    ):
        if score_targets is None:
            raise ValueError("Score supervision requires score targets.")
        score_array = np.asarray(score_targets, dtype=np.float32)
        if score_array.shape != (sample_count, option_count):
            raise ValueError("Score targets must have shape (samples, options).")
        score_mean = np.mean(score_array)
        score_std = float(np.std(score_array) + 1e-6)
        normalized_score_targets = mx.array(
            (score_array - score_mean) / score_std,
            dtype=mx.float32,
        )
    score_rank_code_targets = mx.zeros((sample_count, option_count), dtype=mx.int32)
    if score_rank_code_weight > 0.0:
        if score_targets is None:
            raise ValueError("Rank-code supervision requires score targets.")
        if not 0 <= score_rank_code_slot < slots:
            raise ValueError("Rank-code slot must select an existing message slot.")
        rank_order = np.argsort(np.asarray(score_targets, dtype=np.float32), axis=1)
        rank_positions = np.empty_like(rank_order)
        rows = np.arange(sample_count)[:, None]
        rank_positions[rows, rank_order] = np.arange(option_count)[None, :]
        rank_code_targets = np.floor(
            rank_positions * vocabulary_size / max(1, option_count)
        ).astype(np.int32)
        rank_code_targets = np.clip(rank_code_targets, 0, vocabulary_size - 1)
        score_rank_code_targets = mx.array(rank_code_targets, dtype=mx.int32)

    def pairwise_score_rank_loss(
        predicted_scores: mx.array,
        batch_score_targets: mx.array,
    ) -> mx.array:
        predicted_diff = predicted_scores[:, :, None] - predicted_scores[:, None, :]
        target_diff = batch_score_targets[:, :, None] - batch_score_targets[:, None, :]
        target_abs = mx.abs(target_diff)
        direction = mx.where(target_diff >= 0.0, 1.0, -1.0)
        pair_weight = mx.where(target_abs > 1e-6, target_abs, 0.0)
        pair_loss = mx.logaddexp(mx.array(0.0), -direction * predicted_diff)
        return mx.sum(pair_loss * pair_weight) / (mx.sum(pair_weight) + 1e-6)

    def message_commitment_loss(probabilities: list[mx.array]) -> mx.array:
        commitment = mx.array(0.0)
        for probs in probabilities:
            flat_probs = mx.reshape(probs, (-1, probs.shape[-1]))
            hard_targets = mx.stop_gradient(
                mx.eye(vocabulary_size)[mx.argmax(flat_probs, axis=-1)]
            )
            commitment = commitment - mx.mean(
                mx.sum(hard_targets * mx.log(flat_probs + 1e-8), axis=-1)
            )
        return commitment / len(probabilities)

    def code_target_loss(messages: list[mx.array], batch_targets: mx.array) -> mx.array:
        logits = model.code_target_logits(messages)
        labels = mx.eye(option_count)[batch_targets]
        positive_weight = float(max(1, option_count - 1))
        weights = mx.where(labels > 0.0, positive_weight, 1.0)
        losses = mx.logaddexp(mx.array(0.0), logits) - labels * logits
        return mx.sum(weights * losses) / (mx.sum(weights) + 1e-6)

    def message_replay_loss(
        probabilities: list[mx.array],
        replay_targets: list[mx.array],
    ) -> mx.array:
        replay_loss = mx.array(0.0)
        for probs, targets in zip(probabilities, replay_targets):
            flat_probs = mx.reshape(probs, (-1, probs.shape[-1]))
            flat_targets = mx.reshape(targets, (-1,))
            target_one_hot = mx.eye(vocabulary_size)[flat_targets]
            replay_loss = replay_loss - mx.mean(
                mx.sum(target_one_hot * mx.log(flat_probs + 1e-8), axis=-1)
            )
        return replay_loss / len(probabilities)

    def score_rank_code_loss(
        probabilities: list[mx.array],
        rank_code_targets: mx.array,
    ) -> mx.array:
        probs = probabilities[score_rank_code_slot]
        flat_probs = mx.reshape(probs, (-1, probs.shape[-1]))
        flat_targets = mx.reshape(rank_code_targets, (-1,))
        target_one_hot = mx.eye(vocabulary_size)[flat_targets]
        return -mx.mean(mx.sum(target_one_hot * mx.log(flat_probs + 1e-8), axis=-1))

    def score_pretrain_loss_fn(
        batch_features: mx.array,
        batch_score_targets: mx.array,
        batch_targets: mx.array,
        batch_rank_code_targets: mx.array,
    ) -> mx.array:
        messages, probabilities = model.message(
            batch_features,
            temperature=message_temperature,
            hard=True,
        )
        predicted_scores = model.score_messages(messages)
        score_loss = mx.mean((predicted_scores - batch_score_targets) ** 2)
        rank_loss = pairwise_score_rank_loss(predicted_scores, batch_score_targets)
        uniform = 1.0 / vocabulary_size
        balance_loss = mx.array(0.0)
        for probs in probabilities:
            flat_probs = mx.reshape(probs, (-1, probs.shape[-1]))
            balance_loss = balance_loss + mx.sum(
                (mx.mean(flat_probs, axis=0) - uniform) ** 2
            )
        return (
            score_loss
            + score_rank_weight * rank_loss
            + balance_weight * balance_loss / len(probabilities)
            + pretrain_commitment_weight * message_commitment_loss(probabilities)
            + code_target_weight * code_target_loss(messages, batch_targets)
            + score_rank_code_weight
            * score_rank_code_loss(probabilities, batch_rank_code_targets)
        )

    def loss_fn(
        batch_features: mx.array,
        batch_targets: mx.array,
        batch_score_targets: mx.array,
        batch_replay_targets: list[mx.array],
        batch_rank_code_targets: mx.array,
    ) -> mx.array:
        messages, probabilities = model.message(
            batch_features,
            temperature=message_temperature,
            hard=not soft_message_training,
        )
        receiver_logits = model.receive_all(messages)
        choice_loss = mx.array(0.0)
        distillation_loss = mx.array(0.0)
        for logits in receiver_logits:
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            selected = mx.sum(
                log_probs * mx.eye(option_count)[batch_targets],
                axis=-1,
            )
            choice_loss = choice_loss - mx.mean(selected)
            if score_distillation_weight > 0.0:
                target_probs = mx.softmax(
                    batch_score_targets / max(1e-6, score_distillation_temperature),
                    axis=-1,
                )
                distillation_loss = distillation_loss - mx.mean(
                    mx.sum(target_probs * log_probs, axis=-1)
                )
        choice_loss = choice_loss / len(receiver_logits)
        distillation_loss = distillation_loss / len(receiver_logits)
        uniform = 1.0 / vocabulary_size
        balance_loss = mx.array(0.0)
        entropy = mx.array(0.0)
        for probs in probabilities:
            flat_probs = mx.reshape(probs, (-1, probs.shape[-1]))
            balance_loss = balance_loss + mx.sum(
                (mx.mean(flat_probs, axis=0) - uniform) ** 2
            )
            entropy = entropy - mx.mean(
                mx.sum(flat_probs * mx.log(flat_probs + 1e-8), axis=-1)
            )
        balance_loss = balance_loss / len(probabilities)
        entropy = entropy / len(probabilities)
        score_loss = mx.array(0.0)
        rank_loss = mx.array(0.0)
        if score_reconstruction_weight > 0.0 or score_rank_weight > 0.0:
            predicted_scores = model.score_messages(messages)
            if score_reconstruction_weight > 0.0:
                score_loss = mx.mean((predicted_scores - batch_score_targets) ** 2)
            if score_rank_weight > 0.0:
                rank_loss = pairwise_score_rank_loss(
                    predicted_scores,
                    batch_score_targets,
                )
        return (
            choice_loss
            + score_distillation_weight * distillation_loss
            + balance_weight * balance_loss
            + entropy_weight * entropy
            + message_commitment_weight * message_commitment_loss(probabilities)
            + score_reconstruction_weight * score_loss
            + score_rank_weight * rank_loss
            + code_target_weight * code_target_loss(messages, batch_targets)
            + message_replay_weight
            * message_replay_loss(probabilities, batch_replay_targets)
            + score_rank_code_weight
            * score_rank_code_loss(probabilities, batch_rank_code_targets)
        )

    def frozen_receiver_loss_fn(
        batch_messages: list[mx.array],
        batch_targets: mx.array,
        batch_score_targets: mx.array,
    ) -> mx.array:
        choice_loss = mx.array(0.0)
        distillation_loss = mx.array(0.0)
        for logits in model.receive_all(batch_messages):
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            selected = mx.sum(
                log_probs * mx.eye(option_count)[batch_targets],
                axis=-1,
            )
            choice_loss = choice_loss - mx.mean(selected)
            if score_distillation_weight > 0.0:
                target_probs = mx.softmax(
                    batch_score_targets / max(1e-6, score_distillation_temperature),
                    axis=-1,
                )
                distillation_loss = distillation_loss - mx.mean(
                    mx.sum(target_probs * log_probs, axis=-1)
                )
        choice_loss = choice_loss / model.receiver_copies
        distillation_loss = distillation_loss / model.receiver_copies
        return choice_loss + score_distillation_weight * distillation_loss

    score_pretrain_loss_and_grad = nn.value_and_grad(model, score_pretrain_loss_fn)
    for _epoch in range(max(0, score_pretrain_epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(
                indices[start : start + max(1, batch_size)],
                dtype=mx.int32,
            )
            loss, grads = score_pretrain_loss_and_grad(
                features[batch],
                normalized_score_targets[batch],
                dataset.target_options[batch],
                score_rank_code_targets[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    replay_targets = [
        mx.zeros((sample_count, option_count), dtype=mx.int32)
        for _slot in range(slots)
    ]
    if message_replay_weight > 0.0:
        _replay_messages, replay_probabilities = model.message(
            features,
            temperature=message_temperature,
            hard=True,
        )
        replay_targets = [
            mx.stop_gradient(mx.argmax(probabilities, axis=-1))
            for probabilities in replay_probabilities
        ]
        mx.eval(*replay_targets)

    frozen_receiver_loss_and_grad = nn.value_and_grad(model, frozen_receiver_loss_fn)
    if frozen_receiver_epochs > 0:
        frozen_messages, _probabilities = model.message(
            features,
            temperature=message_temperature,
            hard=True,
        )
        frozen_messages = [mx.stop_gradient(message) for message in frozen_messages]
        for _epoch in range(frozen_receiver_epochs):
            rng.shuffle(indices)
            for start in range(0, sample_count, max(1, batch_size)):
                batch = mx.array(
                    indices[start : start + max(1, batch_size)],
                    dtype=mx.int32,
                )
                batch_score_targets = (
                    mx.zeros((batch.shape[0], option_count), dtype=mx.float32)
                    if normalized_score_targets is None
                    else normalized_score_targets[batch]
                )
                loss, grads = frozen_receiver_loss_and_grad(
                    [message[batch] for message in frozen_messages],
                    dataset.target_options[batch],
                    batch_score_targets,
                )
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(
                indices[start : start + max(1, batch_size)],
                dtype=mx.int32,
            )
            batch_score_targets = (
                mx.zeros((batch.shape[0], option_count), dtype=mx.float32)
                if normalized_score_targets is None
                else normalized_score_targets[batch]
            )
            loss, grads = loss_and_grad(
                features[batch],
                dataset.target_options[batch],
                batch_score_targets,
                [targets[batch] for targets in replay_targets],
                score_rank_code_targets[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    return TrainedOptionMediator(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def train_option_receiver_for_sender(
    trained_sender: TrainedOptionMediator,
    dataset: OptionMediationDataset,
    *,
    receiver_size: int = 96,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    max_samples: int | None = None,
    seed: int = 1,
) -> TrainedOptionReceiver:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    features = (
        dataset.features - trained_sender.feature_mean
    ) / trained_sender.feature_std
    messages, _probabilities = trained_sender.model.message(features, hard=True)
    frozen_messages = [mx.stop_gradient(message) for message in messages]
    sample_count = int(features.shape[0])
    train_indices = np.arange(sample_count)
    if max_samples is not None and max_samples > 0 and max_samples < sample_count:
        train_indices = rng.choice(train_indices, size=max_samples, replace=False)
    option_count = int(dataset.features.shape[1])
    receiver = OptionMediationReceiver(
        option_count,
        receiver_size=receiver_size,
        slots=trained_sender.model.slots,
        vocabulary_size=trained_sender.model.vocabulary_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(batch_messages: list[mx.array], batch_targets: mx.array) -> mx.array:
        logits = receiver.receive(batch_messages)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(log_probs * mx.eye(option_count)[batch_targets], axis=-1)
        return -mx.mean(selected)

    loss_and_grad = nn.value_and_grad(receiver, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(train_indices)
        for start in range(0, len(train_indices), max(1, batch_size)):
            batch = mx.array(
                train_indices[start : start + max(1, batch_size)],
                dtype=mx.int32,
            )
            loss, grads = loss_and_grad(
                [message[batch] for message in frozen_messages],
                dataset.target_options[batch],
            )
            optimizer.update(receiver, grads)
            mx.eval(receiver.parameters(), optimizer.state, loss)

    return TrainedOptionReceiver(model=receiver)


def evaluate_option_mediator(
    trained: TrainedOptionMediator,
    dataset: OptionMediationDataset,
    *,
    model_control: str,
    intervention: str,
) -> OptionMediationResult:
    features = (dataset.features - trained.feature_mean) / trained.feature_std
    messages, _probabilities = trained.model.message(features, hard=True)
    logits = trained.model.receive(messages)
    return _option_mediation_result_from_messages(
        messages,
        logits,
        dataset,
        model_control=model_control,
        intervention=intervention,
    )


def evaluate_option_receiver_transfer(
    trained_sender: TrainedOptionMediator,
    trained_receiver: TrainedOptionReceiver,
    dataset: OptionMediationDataset,
    *,
    model_control: str,
    intervention: str,
) -> OptionMediationResult:
    features = (
        dataset.features - trained_sender.feature_mean
    ) / trained_sender.feature_std
    messages, _probabilities = trained_sender.model.message(features, hard=True)
    logits = trained_receiver.model.receive(messages)
    return _option_mediation_result_from_messages(
        messages,
        logits,
        dataset,
        model_control=model_control,
        intervention=intervention,
    )


def _option_mediation_result_from_messages(
    messages: list[mx.array],
    logits: mx.array,
    dataset: OptionMediationDataset,
    *,
    model_control: str,
    intervention: str,
) -> OptionMediationResult:
    choices = np.asarray(mx.argmax(logits, axis=-1), dtype=np.int32)
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    values = np.asarray(dataset.option_values, dtype=np.float32)
    current = np.asarray(dataset.current_lowest, dtype=np.float32)
    chosen_values = values[np.arange(values.shape[0]), choices]
    oracle_values = np.max(values, axis=1)
    codes = _message_codes(messages)
    code_entropy, dominant_code_fraction = _message_code_stats(codes)
    pattern_ids, patterns_used = _message_pattern_ids(codes)
    reused_pattern_fraction = _reused_pattern_fraction(pattern_ids)
    option_indices = np.arange(values.shape[1], dtype=np.int32)[None, :]
    return OptionMediationResult(
        model_control=model_control,
        intervention=intervention,
        samples=int(values.shape[0]),
        choice_accuracy=float(np.mean(choices == targets)),
        mean_chosen_lowest=float(np.mean(chosen_values)),
        mean_oracle_lowest=float(np.mean(oracle_values)),
        mean_regret=float(np.mean(oracle_values - chosen_values)),
        mean_chosen_delta=float(np.mean(chosen_values - current)),
        mean_oracle_delta=float(np.mean(oracle_values - current)),
        message_codes_used=int(len(np.unique(codes))),
        message_code_entropy=code_entropy,
        dominant_message_code_fraction=dominant_code_fraction,
        target_code_mutual_information=_binary_code_mutual_information(
            codes,
            option_indices == targets[:, None],
        ),
        choice_code_mutual_information=_binary_code_mutual_information(
            codes,
            option_indices == choices[:, None],
        ),
        message_patterns_used=patterns_used,
        reused_message_pattern_fraction=reused_pattern_fraction,
        target_pattern_mutual_information=_categorical_mutual_information(
            pattern_ids,
            targets,
        ),
        choice_pattern_mutual_information=_categorical_mutual_information(
            pattern_ids,
            choices,
        ),
        target_counts=target_count_string(dataset),
    )


def majority_option_result(
    train_dataset: OptionMediationDataset,
    eval_dataset: OptionMediationDataset,
    *,
    model_control: str = "target_majority",
) -> OptionMediationResult:
    train_targets = np.asarray(train_dataset.target_options, dtype=np.int32)
    choice = int(np.argmax(np.bincount(train_targets, minlength=len(OPTION_NAMES))))
    values = np.asarray(eval_dataset.option_values, dtype=np.float32)
    current = np.asarray(eval_dataset.current_lowest, dtype=np.float32)
    targets = np.asarray(eval_dataset.target_options, dtype=np.int32)
    choices = np.full(values.shape[0], choice, dtype=np.int32)
    chosen_values = values[np.arange(values.shape[0]), choices]
    oracle_values = np.max(values, axis=1)
    return OptionMediationResult(
        model_control=model_control,
        intervention="original",
        samples=int(values.shape[0]),
        choice_accuracy=float(np.mean(choices == targets)),
        mean_chosen_lowest=float(np.mean(chosen_values)),
        mean_oracle_lowest=float(np.mean(oracle_values)),
        mean_regret=float(np.mean(oracle_values - chosen_values)),
        mean_chosen_delta=float(np.mean(chosen_values - current)),
        mean_oracle_delta=float(np.mean(oracle_values - current)),
        message_codes_used=0,
        message_code_entropy=0.0,
        dominant_message_code_fraction=0.0,
        target_code_mutual_information=0.0,
        choice_code_mutual_information=0.0,
        message_patterns_used=0,
        reused_message_pattern_fraction=0.0,
        target_pattern_mutual_information=0.0,
        choice_pattern_mutual_information=0.0,
        target_counts=target_count_string(eval_dataset),
    )


def predicted_future_option_result(
    dataset: OptionMediationDataset,
    *,
    model_control: str = "predicted_future",
) -> OptionMediationResult:
    choices = predicted_future_option_choices(dataset)
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    values = np.asarray(dataset.option_values, dtype=np.float32)
    current = np.asarray(dataset.current_lowest, dtype=np.float32)
    chosen_values = values[np.arange(values.shape[0]), choices]
    oracle_values = np.max(values, axis=1)
    return OptionMediationResult(
        model_control=model_control,
        intervention="predicted_future",
        samples=int(values.shape[0]),
        choice_accuracy=float(np.mean(choices == targets)),
        mean_chosen_lowest=float(np.mean(chosen_values)),
        mean_oracle_lowest=float(np.mean(oracle_values)),
        mean_regret=float(np.mean(oracle_values - chosen_values)),
        mean_chosen_delta=float(np.mean(chosen_values - current)),
        mean_oracle_delta=float(np.mean(oracle_values - current)),
        message_codes_used=0,
        message_code_entropy=0.0,
        dominant_message_code_fraction=0.0,
        target_code_mutual_information=0.0,
        choice_code_mutual_information=0.0,
        message_patterns_used=0,
        reused_message_pattern_fraction=0.0,
        target_pattern_mutual_information=0.0,
        choice_pattern_mutual_information=0.0,
        target_counts=target_count_string(dataset),
    )


def predicted_future_option_choices(dataset: OptionMediationDataset) -> np.ndarray:
    predicted_scores = predicted_future_option_scores(dataset)
    return np.asarray(np.argmax(predicted_scores, axis=-1), dtype=np.int32)


def predicted_future_option_scores(dataset: OptionMediationDataset) -> np.ndarray:
    features = np.asarray(dataset.features, dtype=np.float32)
    if features.shape[-1] != 12:
        raise ValueError("Predicted-future control requires latent_current features.")
    predicted_future = features[:, :, 4:8]
    return np.min(predicted_future, axis=-1).astype(np.float32)


def option_mediation_dataset_with_targets(
    dataset: OptionMediationDataset,
    target_options: np.ndarray,
) -> OptionMediationDataset:
    targets = np.asarray(target_options, dtype=np.int32)
    if targets.shape[0] != int(dataset.features.shape[0]):
        raise ValueError("Replacement targets must match dataset sample count.")
    return OptionMediationDataset(
        features=dataset.features,
        option_values=dataset.option_values,
        target_options=mx.array(targets, dtype=mx.int32),
        current_lowest=dataset.current_lowest,
    )


def target_count_string(dataset: OptionMediationDataset) -> str:
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    return ";".join(
        f"{OPTION_NAMES[index]}={int(np.sum(targets == index))}"
        for index in range(len(OPTION_NAMES))
    )


def _select_option_mediation_features(
    features: np.ndarray,
    *,
    feature_mode: str,
) -> np.ndarray:
    if feature_mode == "latent_current":
        return features
    if feature_mode == "delta":
        return features[8:12]
    raise ValueError(f"Unknown option mediation feature mode: {feature_mode}.")


def _feature_blocks(width: int) -> dict[str, slice]:
    if width == 12:
        return {
            "current": slice(0, 4),
            "future": slice(4, 8),
            "delta": slice(8, 12),
        }
    if width == 4:
        return {"delta": slice(0, 4)}
    raise ValueError("Option mediation interventions require feature width 12 or 4.")


def _state_option_actions_and_values(
    env: HomeostaticSocialGrid,
    observation,
    *,
    horizon: int,
    rng: np.random.Generator | None = None,
    option_action_noise: float = 0.0,
) -> tuple[list[list[int]], np.ndarray]:
    option_actions: list[list[int]] = []
    values: list[float] = []
    for option_name in OPTION_NAMES:
        branch = deepcopy(env)
        branch_observation = observation
        actions: list[int] = []
        for _step in range(horizon):
            option_action, option_action_index = _option_action_index(
                option_name,
                branch_observation,
                rng=rng,
                option_action_noise=option_action_noise,
            )
            actions.append(option_action_index)
            branch_observation, _reward, terminated, truncated, _info = branch.step(
                option_action
            )
            if terminated or truncated:
                break
        option_actions.append(actions)
        values.append(float(np.min(_needs_array(branch_observation.needs))))
    return option_actions, np.array(values, dtype=np.float32)


def _state_option_features_and_values(
    base_model: RecurrentActorCritic,
    env: HomeostaticSocialGrid,
    observation,
    history: list[np.ndarray],
    *,
    horizon: int,
    rollout_mode: str,
    rng: np.random.Generator | None = None,
    option_action_noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    values: list[float] = []
    for option_name in OPTION_NAMES:
        branch = deepcopy(env)
        branch_observation = observation
        option_actions: list[int] = []
        for _step in range(horizon):
            option_action, option_action_index = _option_action_index(
                option_name,
                branch_observation,
                rng=rng,
                option_action_noise=option_action_noise,
            )
            option_actions.append(option_action_index)
            branch_observation, _reward, terminated, truncated, _info = branch.step(
                option_action
            )
            if terminated or truncated:
                break
        features.append(
            _option_counterfactual_features(
                base_model,
                history,
                option_actions=option_actions,
                rollout_mode=rollout_mode,
            )
        )
        values.append(float(np.min(_needs_array(branch_observation.needs))))
    return np.stack(features).astype(np.float32), np.array(values, dtype=np.float32)


def _balanced_target_option_indices(
    target_options: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    buckets = [
        np.array(
            [index for index, target in enumerate(target_options) if target == option],
            dtype=np.int64,
        )
        for option in range(len(OPTION_NAMES))
    ]
    non_empty = [bucket for bucket in buckets if len(bucket) > 0]
    if not non_empty:
        return np.arange(len(target_options))
    target = min(len(bucket) for bucket in non_empty)
    indices = np.concatenate(
        [rng.choice(bucket, size=target, replace=False) for bucket in non_empty]
    )
    rng.shuffle(indices)
    return indices


def _message_codes(messages: list[mx.array]) -> np.ndarray:
    code = np.zeros(messages[0].shape[:2], dtype=np.int64)
    multiplier = 1
    for message in messages:
        token = np.asarray(mx.argmax(message, axis=-1), dtype=np.int64)
        code += token * multiplier
        multiplier *= int(message.shape[-1])
    return code


def _message_code_stats(codes: np.ndarray) -> tuple[float, float]:
    counts = np.bincount(np.asarray(codes, dtype=np.int64).reshape(-1))
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0, 0.0
    probabilities = counts / np.sum(counts)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    dominant_fraction = float(np.max(probabilities))
    return entropy, dominant_fraction


def _message_pattern_ids(codes: np.ndarray) -> tuple[np.ndarray, int]:
    _patterns, inverse = np.unique(
        np.asarray(codes, dtype=np.int64),
        axis=0,
        return_inverse=True,
    )
    return inverse.astype(np.int64), int(len(_patterns))


def _reused_pattern_fraction(pattern_ids: np.ndarray) -> float:
    ids = np.asarray(pattern_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return 0.0
    counts = np.bincount(ids)
    return float(np.mean(counts[ids] > 1))


def _binary_code_mutual_information(
    codes: np.ndarray,
    positives: np.ndarray,
) -> float:
    flat_codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    flat_positives = np.asarray(positives, dtype=bool).reshape(-1)
    if flat_codes.size == 0:
        return 0.0
    positive_counts = np.bincount(flat_codes[flat_positives])
    negative_counts = np.bincount(flat_codes[~flat_positives])
    size = max(positive_counts.size, negative_counts.size)
    positive_counts = np.pad(positive_counts, (0, size - positive_counts.size))
    negative_counts = np.pad(negative_counts, (0, size - negative_counts.size))
    joint = np.stack([negative_counts, positive_counts], axis=1).astype(np.float64)
    total = float(np.sum(joint))
    if total == 0.0:
        return 0.0
    joint_probability = joint / total
    code_probability = np.sum(joint_probability, axis=1, keepdims=True)
    label_probability = np.sum(joint_probability, axis=0, keepdims=True)
    independent = code_probability * label_probability
    mask = joint_probability > 0.0
    return float(np.sum(joint_probability[mask] * np.log(joint_probability[mask] / independent[mask])))


def _categorical_mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.int64).reshape(-1)
    right_values = np.asarray(right, dtype=np.int64).reshape(-1)
    if left_values.size == 0 or left_values.shape != right_values.shape:
        return 0.0
    left_ids = np.unique(left_values, return_inverse=True)[1]
    right_ids = np.unique(right_values, return_inverse=True)[1]
    left_count = int(np.max(left_ids)) + 1
    right_count = int(np.max(right_ids)) + 1
    joint = np.bincount(
        left_ids * right_count + right_ids,
        minlength=left_count * right_count,
    ).reshape((left_count, right_count))
    total = float(np.sum(joint))
    if total == 0.0:
        return 0.0
    joint_probability = joint / total
    left_probability = np.sum(joint_probability, axis=1, keepdims=True)
    right_probability = np.sum(joint_probability, axis=0, keepdims=True)
    independent = left_probability * right_probability
    mask = joint_probability > 0.0
    return float(np.sum(joint_probability[mask] * np.log(joint_probability[mask] / independent[mask])))


def _row(result: OptionMediationResult) -> str:
    return ",".join(
        [
            result.model_control,
            result.intervention,
            str(result.samples),
            f"{result.choice_accuracy:.4f}",
            f"{result.mean_chosen_lowest:.6f}",
            f"{result.mean_oracle_lowest:.6f}",
            f"{result.mean_regret:.6f}",
            f"{result.mean_chosen_delta:.6f}",
            f"{result.mean_oracle_delta:.6f}",
            str(result.message_codes_used),
            f"{result.message_code_entropy:.6f}",
            f"{result.dominant_message_code_fraction:.6f}",
            f"{result.target_code_mutual_information:.6f}",
            f"{result.choice_code_mutual_information:.6f}",
            str(result.message_patterns_used),
            f"{result.reused_message_pattern_fraction:.6f}",
            f"{result.target_pattern_mutual_information:.6f}",
            f"{result.choice_pattern_mutual_information:.6f}",
            result.target_counts,
        ]
    )


def main() -> None:
    args = _parse_args()
    trained_base, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)

    print(
        "model_control,intervention,samples,choice_accuracy,mean_chosen_lowest,"
        "mean_oracle_lowest,mean_regret,mean_chosen_delta,mean_oracle_delta,"
        "message_codes_used,message_code_entropy,dominant_message_code_fraction,"
        "target_code_mutual_information,choice_code_mutual_information,"
        "message_patterns_used,reused_message_pattern_fraction,"
        "target_pattern_mutual_information,choice_pattern_mutual_information,"
        "target_counts"
    )
    train_source = collect_option_mediation_source(
        config,
        episodes=args.train_episodes,
        seed=args.seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.balance_target,
        max_states=args.max_train_states,
        min_value_gap=args.min_value_gap,
        option_action_noise=args.option_action_noise,
    )
    eval_source = collect_option_mediation_source(
        config,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.balance_target,
        max_states=args.max_eval_states,
        min_value_gap=args.min_value_gap,
        option_action_noise=args.option_action_noise,
    )
    for model_control, base_model in _base_models(
        trained_base,
        config,
        random_model_control=args.random_model_control,
        seed=args.seed,
    ):
        train_dataset = option_mediation_dataset_from_source(
            train_source,
            base_model,
            feature_mode=args.feature_mode,
            rollout_mode="latent_current",
        )
        eval_dataset = option_mediation_dataset_from_source(
            eval_source,
            base_model,
            feature_mode=args.feature_mode,
            rollout_mode="latent_current",
        )
        training_dataset = train_dataset
        if args.target_mode == "self_model":
            rank_train_dataset = (
                train_dataset
                if args.feature_mode == "latent_current"
                else option_mediation_dataset_from_source(
                    train_source,
                    base_model,
                    feature_mode="latent_current",
                    rollout_mode="latent_current",
                )
            )
            training_dataset = option_mediation_dataset_with_targets(
                train_dataset,
                predicted_future_option_choices(rank_train_dataset),
            )
        score_targets = None
        if (
            args.score_reconstruction_weight > 0.0
            or args.score_pretrain_epochs > 0
            or args.score_distillation_weight > 0.0
            or args.score_rank_weight > 0.0
            or args.score_rank_code_weight > 0.0
        ):
            rank_score_dataset = (
                train_dataset
                if args.feature_mode == "latent_current"
                else option_mediation_dataset_from_source(
                    train_source,
                    base_model,
                    feature_mode="latent_current",
                    rollout_mode="latent_current",
                )
            )
            score_targets = predicted_future_option_scores(rank_score_dataset)
        trained = train_option_mediator(
            training_dataset,
            hidden_size=args.hidden_size,
            receiver_size=args.receiver_size,
            receiver_copies=args.receiver_copies,
            slots=args.message_slots,
            vocabulary_size=args.message_vocabulary,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight,
            message_commitment_weight=args.message_commitment_weight,
            message_temperature=args.message_temperature,
            soft_message_training=args.soft_message_training,
            score_targets=score_targets,
            score_pretrain_epochs=args.score_pretrain_epochs,
            score_pretrain_commitment_weight=args.score_pretrain_commitment_weight,
            frozen_receiver_epochs=args.frozen_receiver_epochs,
            score_distillation_weight=args.score_distillation_weight,
            score_distillation_temperature=args.score_distillation_temperature,
            score_rank_weight=args.score_rank_weight,
            score_reconstruction_weight=args.score_reconstruction_weight,
            code_target_weight=args.code_target_weight,
            message_replay_weight=args.message_replay_weight,
            score_rank_code_weight=args.score_rank_code_weight,
            score_rank_code_slot=args.score_rank_code_slot,
            seed=args.seed,
        )
        for intervention in args.interventions:
            intervened = intervene_option_mediation_features(
                eval_dataset,
                intervention=intervention,
                seed=args.seed + 20_000,
            )
            print(
                _row(
                    evaluate_option_mediator(
                        trained,
                        intervened,
                        model_control=model_control,
                        intervention=intervention,
                    )
                )
            )
        if args.self_model_rank_control:
            rank_dataset = (
                eval_dataset
                if args.feature_mode == "latent_current"
                else option_mediation_dataset_from_source(
                    eval_source,
                    base_model,
                    feature_mode="latent_current",
                    rollout_mode="latent_current",
                )
            )
            print(
                _row(
                    predicted_future_option_result(
                        rank_dataset,
                        model_control=f"{model_control}_self_model",
                    )
                )
            )
        if model_control == "trained":
            print(_row(majority_option_result(train_dataset, eval_dataset)))


def _base_models(
    trained_base: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    random_model_control: bool,
    seed: int,
):
    yield "trained", trained_base
    if random_model_control:
        mx.random.seed(seed)
        yield (
            "random",
            RecurrentActorCritic(
                observation_vector_size(
                    include_language=config.include_language_channel,
                    include_object_kinds=config.include_object_kinds,
                    body_dynamics_mode=config.body_dynamics_mode,
                ),
                config.hidden_size,
                len(Action),
            ),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--max-train-states", type=int, default=12000)
    parser.add_argument("--max-eval-states", type=int, default=6000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--renewable-resources", action="store_true")
    parser.add_argument(
        "--resource-ecology",
        choices=RESOURCE_ECOLOGIES,
        default=None,
    )
    parser.add_argument(
        "--state-policy",
        choices=STATE_POLICIES,
        default="cycle",
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument(
        "--balance-target",
        choices=OPTION_MEDIATION_BALANCE_TARGETS,
        default="target_option",
    )
    parser.add_argument("--min-value-gap", type=float, default=0.005)
    parser.add_argument(
        "--option-action-noise",
        type=float,
        default=0.0,
        help="Probability of replacing a scripted option step with another valid body action.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=OPTION_MEDIATION_FEATURE_MODES,
        default="latent_current",
    )
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--receiver-copies", type=int, default=1)
    parser.add_argument("--message-slots", type=int, default=OPTION_MEDIATION_SLOTS)
    parser.add_argument(
        "--message-vocabulary",
        type=int,
        default=OPTION_MEDIATION_VOCABULARY,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--message-commitment-weight", type=float, default=0.0)
    parser.add_argument("--message-temperature", type=float, default=0.6)
    parser.add_argument("--soft-message-training", action="store_true")
    parser.add_argument("--score-pretrain-epochs", type=int, default=0)
    parser.add_argument("--score-pretrain-commitment-weight", type=float, default=None)
    parser.add_argument("--frozen-receiver-epochs", type=int, default=0)
    parser.add_argument("--score-distillation-weight", type=float, default=0.0)
    parser.add_argument("--score-distillation-temperature", type=float, default=1.0)
    parser.add_argument("--score-rank-weight", type=float, default=0.0)
    parser.add_argument("--score-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--code-target-weight", type=float, default=0.0)
    parser.add_argument("--message-replay-weight", type=float, default=0.0)
    parser.add_argument("--score-rank-code-weight", type=float, default=0.0)
    parser.add_argument("--score-rank-code-slot", type=int, default=0)
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=OPTION_FEATURE_INTERVENTIONS,
        default=["original", "shuffle_delta", "negate_delta"],
    )
    parser.add_argument("--random-model-control", action="store_true")
    parser.add_argument("--self-model-rank-control", action="store_true")
    parser.add_argument(
        "--target-mode",
        choices=OPTION_MEDIATION_TARGET_MODES,
        default="oracle",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
