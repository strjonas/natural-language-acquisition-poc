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
from .observations import observation_vector
from .option_counterfactual_language import (
    OPTION_NAMES,
    STATE_POLICIES,
    _option_action_index,
    _state_agent,
)
from .option_world_model import (
    OptionBranchDataset,
    collect_option_branch_dataset,
    _option_rollout_predictions,
)
from .qlearning import EpisodeStats
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .teachers import (
    TEACHER_MODES,
    build_teacher,
    masks_language,
    normalize_teacher_mode,
)


PARTNER_PROPOSAL_MODES = (
    "partial_body",
    "partial_food_water",
    "partial_energy_safety",
    "second_best",
    "worst",
)
SITUATED_INTERVENTIONS = (
    "original",
    "shuffle_delta",
    "reverse_delta_rank",
    "shuffle_outcome",
    "reverse_outcome_rank",
    "zero_outcome",
)
SITUATED_VALUE_MODES = ("final_lowest", "trajectory_min", "trajectory_mean")
SITUATED_OPTION_SETS = ("base", "extended")
ONLINE_SELF_MODEL_SOURCES = ("exact", "learned")
ONLINE_SELF_CALIBRATION_MODES = ("all", "values", "value_lcb", "value_knn_lcb")
EXTENDED_SITUATED_OPTION_NAMES = (
    "seek_lowest",
    "seek_food",
    "seek_water",
    "seek_shelter",
    "rest",
    "wait",
    "turn_left",
    "turn_right",
)


@dataclass(frozen=True)
class SituatedPartnerDataset:
    features: mx.array
    current_needs: mx.array
    final_needs: mx.array
    option_values: mx.array
    target_options: mx.array
    current_lowest: mx.array
    option_names: tuple[str, ...] = OPTION_NAMES


@dataclass(frozen=True)
class TrainedSituatedPartnerDialogue:
    model: "SituatedPartnerDialogue"
    feature_mean: mx.array
    feature_std: mx.array
    option_names: tuple[str, ...] = OPTION_NAMES


@dataclass(frozen=True)
class SituatedPartnerResult:
    model_control: str
    intervention: str
    samples: int
    proposal_accuracy: float
    final_accuracy: float
    changed_fraction: float
    mean_proposal_delta: float
    mean_chosen_delta: float
    mean_oracle_delta: float
    mean_regret: float
    target_counts: str


@dataclass(frozen=True)
class OnlinePartnerResult:
    model_control: str
    intervention: str
    episodes: int
    mean_total_reward: float
    mean_steps: float
    mean_viability: float
    mean_min_viability: float
    mean_resource_uses: float
    mean_danger_hits: float
    mean_teacher_utterances: float
    termination_rate: float
    truncation_rate: float
    mean_proposal_delta: float
    mean_chosen_delta: float
    mean_oracle_delta: float
    mean_regret: float
    override_rate: float


@dataclass(frozen=True)
class SituatedSelfModelRankSample:
    history: tuple[np.ndarray, ...]
    option_actions: tuple[np.ndarray, ...]
    option_final_needs: np.ndarray
    option_values: np.ndarray
    target_option: int


@dataclass(frozen=True)
class SituatedSelfModelCalibrator:
    final_scale: np.ndarray
    final_offset: np.ndarray
    final_rmse: np.ndarray
    value_scale: np.ndarray
    value_offset: np.ndarray
    value_rmse: np.ndarray
    value_reference_predictions: np.ndarray
    value_reference_errors: np.ndarray
    option_names: tuple[str, ...]


@dataclass(frozen=True)
class OnlineAdaptationRiskCalibrator:
    features: np.ndarray
    risks: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    option_names: tuple[str, ...]


class SituatedPartnerDialogue(nn.Module):
    def __init__(
        self,
        input_size: int,
        option_count: int,
        *,
        hidden_size: int = 96,
        receiver_size: int = 96,
        vocabulary_size: int = 4,
    ) -> None:
        super().__init__()
        self.option_count = option_count
        self.vocabulary_size = vocabulary_size
        self.reply_sender = nn.Linear(input_size + 1, hidden_size)
        self.reply_sender_hidden = nn.Linear(hidden_size, hidden_size)
        self.reply_token = nn.Linear(hidden_size, vocabulary_size)
        final_input = option_count * vocabulary_size + option_count
        self.final_receiver = nn.Linear(final_input, receiver_size)
        self.final_hidden = nn.Linear(receiver_size, receiver_size)
        self.final_choice = nn.Linear(receiver_size, option_count)

    def reply_message(
        self,
        features: mx.array,
        proposal_signal: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[mx.array, mx.array]:
        batch_size, option_count, feature_size = features.shape
        reply_features = mx.concatenate([features, proposal_signal[:, :, None]], axis=-1)
        flat = mx.reshape(reply_features, (batch_size * option_count, feature_size + 1))
        hidden = nn.relu(self.reply_sender(flat))
        hidden = hidden + nn.relu(self.reply_sender_hidden(hidden))
        logits = self.reply_token(hidden)
        probs = mx.softmax(logits / temperature, axis=-1)
        if hard:
            hard_message = mx.eye(self.vocabulary_size)[mx.argmax(probs, axis=-1)]
            message = hard_message + probs - mx.stop_gradient(probs)
        else:
            message = probs
        return (
            mx.reshape(message, (batch_size, option_count, self.vocabulary_size)),
            mx.reshape(probs, (batch_size, option_count, self.vocabulary_size)),
        )

    def final(self, reply_message: mx.array, proposal_signal: mx.array) -> mx.array:
        inputs = mx.concatenate(
            [
                mx.reshape(reply_message, (reply_message.shape[0], -1)),
                proposal_signal,
            ],
            axis=-1,
        )
        hidden = nn.relu(self.final_receiver(inputs))
        hidden = hidden + nn.relu(self.final_hidden(hidden))
        return self.final_choice(hidden)


def situated_partner_dataset_from_branches(
    branches: OptionBranchDataset,
    *,
    min_value_gap: float = 0.0,
    min_oracle_delta: float | None = None,
    value_mode: str = "final_lowest",
) -> SituatedPartnerDataset:
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    option_count = len(OPTION_NAMES)
    samples = list(branches.samples)
    grouped_features: list[np.ndarray] = []
    grouped_current: list[np.ndarray] = []
    grouped_final: list[np.ndarray] = []
    grouped_values: list[np.ndarray] = []
    targets: list[int] = []
    current_lowest: list[float] = []

    for start in range(0, len(samples) - option_count + 1, option_count):
        group = samples[start : start + option_count]
        labels = [sample.option_label for sample in group]
        if labels != list(range(option_count)):
            continue
        current = np.asarray(group[0].current_needs, dtype=np.float32)
        final_needs = np.stack(
            [np.asarray(sample.next_needs, dtype=np.float32)[-1] for sample in group]
        ).astype(np.float32)
        option_needs = [
            np.asarray(sample.next_needs, dtype=np.float32) for sample in group
        ]
        values = np.array(
            [_option_value(needs, value_mode=value_mode) for needs in option_needs],
            dtype=np.float32,
        )
        current_low = float(np.min(current))
        if float(np.max(values) - np.min(values)) < min_value_gap:
            continue
        if (
            min_oracle_delta is not None
            and float(np.max(values) - current_low) < min_oracle_delta
        ):
            continue
        option_ids = np.eye(option_count, dtype=np.float32)
        features = np.concatenate(
            [
                np.repeat(current[None, :], option_count, axis=0),
                final_needs,
                final_needs - current[None, :],
                option_ids,
            ],
            axis=1,
        )
        grouped_features.append(features.astype(np.float32))
        grouped_current.append(current)
        grouped_final.append(final_needs)
        grouped_values.append(values)
        targets.append(int(np.argmax(values)))
        current_lowest.append(current_low)

    if not grouped_features:
        raise ValueError("No complete situated partner states were collected.")

    return SituatedPartnerDataset(
        features=mx.array(np.stack(grouped_features), dtype=mx.float32),
        current_needs=mx.array(np.stack(grouped_current), dtype=mx.float32),
        final_needs=mx.array(np.stack(grouped_final), dtype=mx.float32),
        option_values=mx.array(np.stack(grouped_values), dtype=mx.float32),
        target_options=mx.array(targets, dtype=mx.int32),
        current_lowest=mx.array(current_lowest, dtype=mx.float32),
        option_names=OPTION_NAMES,
    )


def collect_situated_partner_dataset(
    config,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    state_policy: str = "cycle",
    max_states: int = 3000,
    option_action_noise: float = 0.0,
    option_set: str = "base",
    min_value_gap: float = 0.0,
    min_oracle_delta: float | None = None,
    value_mode: str = "final_lowest",
) -> SituatedPartnerDataset:
    if state_policy not in STATE_POLICIES:
        raise ValueError(f"Unknown state policy: {state_policy}.")
    if option_set not in SITUATED_OPTION_SETS:
        raise ValueError(f"Unknown situated option set: {option_set}.")
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    if not 0.0 <= option_action_noise <= 1.0:
        raise ValueError("Option action noise must be in [0, 1].")

    option_names = _situated_option_names(option_set)
    normalized_teacher = normalize_teacher_mode(teacher_mode)
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
    rng = np.random.default_rng(seed + 9_910_000)
    grouped_features: list[np.ndarray] = []
    grouped_current: list[np.ndarray] = []
    grouped_final: list[np.ndarray] = []
    grouped_values: list[np.ndarray] = []
    targets: list[int] = []
    current_lowest: list[float] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = _state_agent(state_policy, seed=seed, episode=episode)
        terminated = False
        truncated = False
        while not terminated and not truncated and len(grouped_features) < max_states:
            features, final_needs, values, current_low = _situated_option_features(
                env,
                observation,
                option_names=option_names,
                horizon=max(1, horizon),
                rng=rng,
                option_action_noise=option_action_noise,
                value_mode=value_mode,
            )
            if (
                float(np.max(values) - np.min(values)) >= min_value_gap
                and (
                    min_oracle_delta is None
                    or float(np.max(values) - current_low) >= min_oracle_delta
                )
            ):
                grouped_features.append(features)
                grouped_current.append(_needs_array(observation.needs).astype(np.float32))
                grouped_final.append(final_needs)
                grouped_values.append(values)
                targets.append(int(np.argmax(values)))
                current_lowest.append(current_low)

            action = agent.act(observation)
            observation, _reward, terminated, truncated, _info = env.step(action)
        if len(grouped_features) >= max_states:
            break

    if not grouped_features:
        raise ValueError("No situated partner states were collected.")

    return SituatedPartnerDataset(
        features=mx.array(np.stack(grouped_features), dtype=mx.float32),
        current_needs=mx.array(np.stack(grouped_current), dtype=mx.float32),
        final_needs=mx.array(np.stack(grouped_final), dtype=mx.float32),
        option_values=mx.array(np.stack(grouped_values), dtype=mx.float32),
        target_options=mx.array(targets, dtype=mx.int32),
        current_lowest=mx.array(current_lowest, dtype=mx.float32),
        option_names=option_names,
    )


def collect_situated_self_model_rank_samples(
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    state_policy: str = "cycle",
    max_samples: int = 3000,
    option_action_noise: float = 0.0,
    option_set: str = "base",
    value_mode: str = "final_lowest",
) -> tuple[SituatedSelfModelRankSample, ...]:
    if state_policy not in STATE_POLICIES:
        raise ValueError(f"Unknown state policy: {state_policy}.")
    if option_set not in SITUATED_OPTION_SETS:
        raise ValueError(f"Unknown situated option set: {option_set}.")
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    if not 0.0 <= option_action_noise <= 1.0:
        raise ValueError("Option action noise must be in [0, 1].")

    option_names = _situated_option_names(option_set)
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
    rng = np.random.default_rng(seed + 3_310_000)
    samples: list[SituatedSelfModelRankSample] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = _state_agent(state_policy, seed=seed, episode=episode)
        history = [
            _online_observation_vector(
                observation,
                env=env,
                config=config,
                mask_language=mask_language,
            )
        ]
        terminated = False
        truncated = False
        while not terminated and not truncated and len(samples) < max_samples:
            _features, final_needs, values, _current_low = _situated_option_features(
                env,
                observation,
                option_names=option_names,
                horizon=max(1, horizon),
                rng=rng,
                option_action_noise=option_action_noise,
                value_mode=value_mode,
            )
            action_sequences = _situated_option_action_sequences(
                env,
                observation,
                option_names=option_names,
                horizon=max(1, horizon),
                rng=rng,
                option_action_noise=option_action_noise,
            )
            samples.append(
                SituatedSelfModelRankSample(
                    history=tuple(item.copy() for item in history),
                    option_actions=tuple(action_sequences),
                    option_final_needs=final_needs.copy(),
                    option_values=values.copy(),
                    target_option=int(np.argmax(values)),
                )
            )

            action = agent.act(observation)
            observation, _reward, terminated, truncated, _info = env.step(action)
            history.append(
                _online_observation_vector(
                    observation,
                    env=env,
                    config=config,
                    mask_language=mask_language,
                )
            )
        if len(samples) >= max_samples:
            break

    if not samples:
        raise ValueError("No situated self-model rank samples were collected.")
    return tuple(samples)


def collect_online_adaptation_risk_samples(
    trained: TrainedSituatedPartnerDialogue,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    state_policy: str = "cycle",
    max_samples: int = 1500,
    partner_mode: str = "partial_body",
    option_action_noise: float = 0.0,
    value_mode: str = "final_lowest",
    online_self_model_source: str = "exact",
    online_self_model: RecurrentActorCritic | None = None,
    online_self_model_config: RecurrentConfig | None = None,
    online_self_model_calibrator: SituatedSelfModelCalibrator | None = None,
    online_self_calibration_mode: str = "all",
    online_self_calibration_uncertainty_scale: float = 1.0,
    online_self_calibration_knn: int = 16,
) -> OnlineAdaptationRiskCalibrator:
    if episodes <= 0:
        raise ValueError("Risk calibration episodes must be positive.")
    if max_samples <= 0:
        raise ValueError("Risk calibration max_samples must be positive.")
    if partner_mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {partner_mode}.")
    if state_policy not in STATE_POLICIES:
        raise ValueError(f"Unknown state policy: {state_policy}.")
    if not 0.0 <= option_action_noise <= 1.0:
        raise ValueError("Option action noise must be in [0, 1].")
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    if online_self_model_source not in ONLINE_SELF_MODEL_SOURCES:
        raise ValueError(f"Unknown online self-model source: {online_self_model_source}.")
    if online_self_calibration_mode not in ONLINE_SELF_CALIBRATION_MODES:
        raise ValueError(
            f"Unknown online self-calibration mode: {online_self_calibration_mode}."
        )
    if online_self_calibration_uncertainty_scale < 0.0:
        raise ValueError("online_self_calibration_uncertainty_scale must be non-negative.")
    if online_self_calibration_knn <= 0:
        raise ValueError("online_self_calibration_knn must be positive.")
    if online_self_model_source == "learned" and (
        online_self_model is None or online_self_model_config is None
    ):
        raise ValueError("learned online self-model source requires a model and config.")

    self_model_config = online_self_model_config or config
    normalized_teacher = normalize_teacher_mode(teacher_mode)
    mask_language = (
        masks_language(normalized_teacher)
        or not self_model_config.include_language_channel
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
    rng = np.random.default_rng(seed + 8_410_000)
    features: list[np.ndarray] = []
    risks: list[float] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = _state_agent(state_policy, seed=seed, episode=episode)
        observation_history = [
            _online_observation_vector(
                observation,
                env=env,
                config=self_model_config,
                mask_language=mask_language,
            )
        ]
        terminated = False
        truncated = False
        while not terminated and not truncated and len(features) < max_samples:
            exact_state_dataset = _online_state_dataset(
                env,
                observation,
                horizon=max(1, horizon),
                rng=rng,
                option_action_noise=option_action_noise,
                value_mode=value_mode,
                option_names=trained.option_names,
            )
            decision_dataset = exact_state_dataset
            if online_self_model_source == "learned":
                decision_dataset = _learned_online_state_dataset(
                    online_self_model,
                    self_model_config,
                    env,
                    observation,
                    observation_history,
                    option_names=trained.option_names,
                    horizon=max(1, horizon),
                    rng=rng,
                    option_action_noise=option_action_noise,
                    value_mode=value_mode,
                    calibrator=online_self_model_calibrator,
                    calibration_mode=online_self_calibration_mode,
                    uncertainty_scale=online_self_calibration_uncertainty_scale,
                    uncertainty_knn=online_self_calibration_knn,
                )
            proposal = int(_partner_proposals(decision_dataset, mode=partner_mode)[0])
            exact_values = np.asarray(
                exact_state_dataset.option_values,
                dtype=np.float32,
            )[0]
            current_lowest = float(np.asarray(exact_state_dataset.current_lowest)[0])
            for choice in range(len(trained.option_names)):
                features.append(
                    _online_risk_feature_vector(
                        decision_dataset,
                        proposal=proposal,
                        choice=choice,
                    )
                )
                risks.append(max(0.0, current_lowest - float(exact_values[choice])))
                if len(features) >= max_samples:
                    break

            action = agent.act(observation)
            observation, _reward, terminated, truncated, _info = env.step(action)
            observation_history.append(
                _online_observation_vector(
                    observation,
                    env=env,
                    config=self_model_config,
                    mask_language=mask_language,
                )
            )
        if len(features) >= max_samples:
            break

    if not features:
        raise ValueError("No online risk calibration samples were collected.")
    feature_array = np.stack(features).astype(np.float32)
    risk_array = np.asarray(risks, dtype=np.float32)
    feature_mean = np.mean(feature_array, axis=0).astype(np.float32)
    feature_std = np.sqrt(
        np.mean((feature_array - feature_mean[None, :]) ** 2, axis=0) + 1e-6
    ).astype(np.float32)
    return OnlineAdaptationRiskCalibrator(
        features=feature_array,
        risks=risk_array,
        feature_mean=feature_mean,
        feature_std=feature_std,
        option_names=trained.option_names,
    )


def train_situated_self_model_rank(
    model: RecurrentActorCritic,
    samples: tuple[SituatedSelfModelRankSample, ...],
    *,
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    temperature: float = 0.05,
    value_mode: str = "final_lowest",
    seed: int = 1,
) -> float:
    if epochs <= 0:
        return 0.0
    if not samples:
        raise ValueError("Cannot rank-finetune on empty situated samples.")
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    optimizer = optim.Adam(learning_rate=learning_rate)
    shuffled = list(samples)
    final_loss = 0.0

    def loss_fn(
        observations: mx.array,
        step_masks: mx.array,
        observation_lengths: mx.array,
        actions: mx.array,
        action_masks: mx.array,
        targets: mx.array,
    ) -> mx.array:
        scores = _situated_grouped_predicted_scores(
            model,
            observations,
            step_masks,
            observation_lengths,
            actions,
            action_masks,
            value_mode=value_mode,
        )
        logits = scores / max(temperature, 1e-6)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(log_probs * mx.eye(scores.shape[1])[targets], axis=-1)
        return -mx.mean(selected)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), max(1, batch_size)):
            batch = _pad_situated_rank_samples(
                shuffled[start : start + max(1, batch_size)]
            )
            loss, grads = loss_and_grad(*batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            final_loss = float(loss)
    return final_loss


def fit_situated_self_model_calibrator(
    model: RecurrentActorCritic,
    samples: tuple[SituatedSelfModelRankSample, ...],
    *,
    option_names: tuple[str, ...],
    value_mode: str,
    ridge: float = 1e-4,
    batch_size: int = 128,
) -> SituatedSelfModelCalibrator:
    if not samples:
        raise ValueError("Cannot fit calibrator on empty situated samples.")
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    if ridge < 0.0:
        raise ValueError("Calibrator ridge must be non-negative.")

    predicted_finals: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    for start in range(0, len(samples), max(1, batch_size)):
        batch = _pad_situated_rank_samples(
            list(samples[start : start + max(1, batch_size)])
        )
        final, values = _situated_grouped_predicted_final_and_scores(
            model,
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
            value_mode=value_mode,
        )
        predicted_finals.append(np.asarray(final, dtype=np.float32))
        predicted_values.append(np.asarray(values, dtype=np.float32))

    predicted_final = np.concatenate(predicted_finals, axis=0)
    predicted_value = np.concatenate(predicted_values, axis=0)
    target_final = np.stack([sample.option_final_needs for sample in samples])
    target_value = np.stack([sample.option_values for sample in samples])
    option_count = predicted_value.shape[1]
    if option_count != len(option_names):
        raise ValueError("Calibrator option names do not match sample option count.")

    final_scale = np.ones((option_count, 4), dtype=np.float32)
    final_offset = np.zeros((option_count, 4), dtype=np.float32)
    final_rmse = np.zeros((option_count, 4), dtype=np.float32)
    value_scale = np.ones(option_count, dtype=np.float32)
    value_offset = np.zeros(option_count, dtype=np.float32)
    value_rmse = np.zeros(option_count, dtype=np.float32)
    value_reference_predictions = predicted_value.astype(np.float32)
    value_reference_errors = np.zeros_like(value_reference_predictions, dtype=np.float32)
    for option_index in range(option_count):
        for need_index in range(4):
            scale, offset = _fit_affine_calibration(
                predicted_final[:, option_index, need_index],
                target_final[:, option_index, need_index],
                ridge=ridge,
            )
            final_scale[option_index, need_index] = scale
            final_offset[option_index, need_index] = offset
            calibrated_final = (
                scale * predicted_final[:, option_index, need_index] + offset
            )
            final_rmse[option_index, need_index] = float(
                np.sqrt(
                    np.mean(
                        (
                            calibrated_final
                            - target_final[:, option_index, need_index]
                        )
                        ** 2
                    )
                )
            )
        scale, offset = _fit_affine_calibration(
            predicted_value[:, option_index],
            target_value[:, option_index],
            ridge=ridge,
        )
        value_scale[option_index] = scale
        value_offset[option_index] = offset
        calibrated_value = scale * predicted_value[:, option_index] + offset
        value_rmse[option_index] = float(
            np.sqrt(
                np.mean((calibrated_value - target_value[:, option_index]) ** 2)
            )
        )
        value_reference_errors[:, option_index] = np.abs(
            calibrated_value - target_value[:, option_index]
        ).astype(np.float32)

    return SituatedSelfModelCalibrator(
        final_scale=final_scale,
        final_offset=final_offset,
        final_rmse=final_rmse,
        value_scale=value_scale,
        value_offset=value_offset,
        value_rmse=value_rmse,
        value_reference_predictions=value_reference_predictions,
        value_reference_errors=value_reference_errors,
        option_names=option_names,
    )


def _fit_affine_calibration(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> tuple[float, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    design = np.stack([predicted, np.ones_like(predicted)], axis=1)
    regularizer = ridge * np.eye(2, dtype=np.float64)
    regularizer[1, 1] = 0.0
    coeffs = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return float(coeffs[0]), float(coeffs[1])


def _calibrator_local_value_error(
    calibrator: SituatedSelfModelCalibrator,
    predicted_values: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("Local calibration neighbor count must be positive.")
    predicted_values = np.asarray(predicted_values, dtype=np.float32)
    errors = np.zeros_like(predicted_values, dtype=np.float32)
    for option_index in range(predicted_values.shape[0]):
        references = calibrator.value_reference_predictions[:, option_index]
        residuals = calibrator.value_reference_errors[:, option_index]
        neighbor_count = min(k, references.shape[0])
        distances = np.abs(references - predicted_values[option_index])
        nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        errors[option_index] = float(np.mean(residuals[nearest]))
    return errors


def train_situated_partner_dialogue(
    dataset: SituatedPartnerDataset,
    *,
    hidden_size: int = 96,
    receiver_size: int = 96,
    vocabulary_size: int = 4,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    partner_mode: str = "partial_body",
    target_weight: float = 0.0,
    balance_weight: float = 0.02,
    entropy_weight: float = 0.0,
    message_temperature: float = 0.6,
    seed: int = 1,
) -> TrainedSituatedPartnerDialogue:
    if partner_mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {partner_mode}.")
    if target_weight < 0.0:
        raise ValueError("target_weight must be non-negative.")

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
    model = SituatedPartnerDialogue(
        int(features.shape[-1]),
        option_count,
        hidden_size=hidden_size,
        receiver_size=receiver_size,
        vocabulary_size=vocabulary_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    indices = np.arange(sample_count)
    partner_proposals = mx.array(
        _partner_proposals(dataset, mode=partner_mode),
        dtype=mx.int32,
    )

    def outcome_regret(logits: mx.array, option_values: mx.array) -> mx.array:
        choice_probs = mx.softmax(logits, axis=-1)
        expected_value = mx.sum(choice_probs * option_values, axis=-1)
        return mx.max(option_values, axis=-1) - expected_value

    def ce_loss(logits: mx.array, targets: mx.array) -> mx.array:
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(log_probs * mx.eye(option_count)[targets], axis=-1)
        return -mx.mean(selected)

    def token_regularizer(probs: mx.array) -> mx.array:
        flat = mx.reshape(probs, (-1, probs.shape[-1]))
        uniform = 1.0 / vocabulary_size
        balance = mx.sum((mx.mean(flat, axis=0) - uniform) ** 2)
        entropy = -mx.mean(mx.sum(flat * mx.log(flat + 1e-8), axis=-1))
        return balance_weight * balance + entropy_weight * entropy

    def loss_fn(
        batch_features: mx.array,
        batch_values: mx.array,
        batch_partner_proposals: mx.array,
    ) -> mx.array:
        proposal_signal = mx.eye(option_count)[batch_partner_proposals]
        reply_message, reply_probs = model.reply_message(
            batch_features,
            proposal_signal,
            temperature=message_temperature,
            hard=True,
        )
        final_logits = model.final(reply_message, proposal_signal)
        targets = mx.argmax(batch_values, axis=-1)
        return (
            mx.mean(outcome_regret(final_logits, batch_values))
            + target_weight * ce_loss(final_logits, targets)
            + token_regularizer(reply_probs)
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(indices[start : start + max(1, batch_size)], dtype=mx.int32)
            loss, grads = loss_and_grad(
                features[batch],
                dataset.option_values[batch],
                partner_proposals[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    return TrainedSituatedPartnerDialogue(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        option_names=dataset.option_names,
    )


def evaluate_situated_partner_dialogue(
    trained: TrainedSituatedPartnerDialogue,
    dataset: SituatedPartnerDataset,
    *,
    model_control: str,
    intervention: str,
    partner_mode: str = "partial_body",
) -> SituatedPartnerResult:
    if partner_mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {partner_mode}.")
    features = (dataset.features - trained.feature_mean) / trained.feature_std
    proposals = _partner_proposals(dataset, mode=partner_mode)
    proposal_signal = mx.eye(trained.model.option_count)[
        mx.array(proposals, dtype=mx.int32)
    ]
    reply_message, _reply_probs = trained.model.reply_message(
        features,
        proposal_signal,
        hard=True,
    )
    final_logits = trained.model.final(reply_message, proposal_signal)
    choices = np.asarray(mx.argmax(final_logits, axis=-1), dtype=np.int32)
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    values = np.asarray(dataset.option_values, dtype=np.float32)
    current = np.asarray(dataset.current_lowest, dtype=np.float32)
    proposal_values = values[np.arange(values.shape[0]), proposals]
    chosen_values = values[np.arange(values.shape[0]), choices]
    oracle_values = np.max(values, axis=1)
    return SituatedPartnerResult(
        model_control=model_control,
        intervention=intervention,
        samples=int(values.shape[0]),
        proposal_accuracy=float(np.mean(proposals == targets)),
        final_accuracy=float(np.mean(choices == targets)),
        changed_fraction=float(np.mean(choices != proposals)),
        mean_proposal_delta=float(np.mean(proposal_values - current)),
        mean_chosen_delta=float(np.mean(chosen_values - current)),
        mean_oracle_delta=float(np.mean(oracle_values - current)),
        mean_regret=float(np.mean(oracle_values - chosen_values)),
        target_counts=target_count_string(dataset),
    )


def evaluate_online_partner_dialogue(
    trained: TrainedSituatedPartnerDialogue,
    config,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    partner_mode: str = "partial_body",
    model_control: str = "dialogue",
    intervention: str = "original",
    option_action_noise: float = 0.0,
    option_commit_steps: int = 1,
    value_mode: str = "final_lowest",
    online_adaptation_steps: int = 0,
    online_adaptation_learning_rate: float = 3e-4,
    online_adaptation_target_weight: float = 0.0,
    online_adaptation_kl_weight: float = 0.0,
    online_adaptation_min_value_gap: float = 0.0,
    online_adaptation_choice_guard: bool = False,
    online_adaptation_choice_guard_margin: float = 0.0,
    online_adaptation_local: bool = False,
    online_self_model_source: str = "exact",
    online_self_model: RecurrentActorCritic | None = None,
    online_self_model_config: RecurrentConfig | None = None,
    online_self_model_calibrator: SituatedSelfModelCalibrator | None = None,
    online_self_calibration_mode: str = "all",
    online_self_calibration_uncertainty_scale: float = 1.0,
    online_self_calibration_knn: int = 16,
    online_adaptation_risk_calibrator: OnlineAdaptationRiskCalibrator | None = None,
    online_adaptation_risk_threshold: float = 0.0,
    online_adaptation_risk_knn: int = 16,
) -> OnlinePartnerResult:
    if partner_mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {partner_mode}.")
    if model_control not in {"dialogue", "adaptive_dialogue", "partner", "oracle"}:
        raise ValueError(f"Unknown online model control: {model_control}.")
    if intervention not in SITUATED_INTERVENTIONS:
        raise ValueError(f"Unknown situated partner intervention: {intervention}.")
    if not 0.0 <= option_action_noise <= 1.0:
        raise ValueError("Option action noise must be in [0, 1].")
    if option_commit_steps <= 0:
        raise ValueError("option_commit_steps must be positive.")
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    if online_self_model_source not in ONLINE_SELF_MODEL_SOURCES:
        raise ValueError(f"Unknown online self-model source: {online_self_model_source}.")
    if online_self_calibration_mode not in ONLINE_SELF_CALIBRATION_MODES:
        raise ValueError(
            f"Unknown online self-calibration mode: {online_self_calibration_mode}."
        )
    if online_self_calibration_uncertainty_scale < 0.0:
        raise ValueError("online_self_calibration_uncertainty_scale must be non-negative.")
    if online_self_calibration_knn <= 0:
        raise ValueError("online_self_calibration_knn must be positive.")
    if online_adaptation_risk_threshold < 0.0:
        raise ValueError("online_adaptation_risk_threshold must be non-negative.")
    if online_adaptation_risk_knn <= 0:
        raise ValueError("online_adaptation_risk_knn must be positive.")
    if online_adaptation_steps < 0:
        raise ValueError("online_adaptation_steps must be non-negative.")
    if online_adaptation_learning_rate <= 0.0:
        raise ValueError("online_adaptation_learning_rate must be positive.")
    if online_adaptation_target_weight < 0.0:
        raise ValueError("online_adaptation_target_weight must be non-negative.")
    if online_adaptation_kl_weight < 0.0:
        raise ValueError("online_adaptation_kl_weight must be non-negative.")
    if online_adaptation_min_value_gap < 0.0:
        raise ValueError("online_adaptation_min_value_gap must be non-negative.")
    if online_adaptation_choice_guard_margin < 0.0:
        raise ValueError("online_adaptation_choice_guard_margin must be non-negative.")
    if model_control == "adaptive_dialogue" and online_adaptation_steps <= 0:
        raise ValueError("adaptive_dialogue requires online_adaptation_steps > 0.")
    if online_self_model_source == "learned" and (
        online_self_model is None or online_self_model_config is None
    ):
        raise ValueError("learned online self-model source requires a model and config.")
    if (
        online_adaptation_risk_calibrator is not None
        and online_adaptation_risk_calibrator.option_names != trained.option_names
    ):
        raise ValueError("Risk calibrator option names do not match online options.")

    normalized_teacher = normalize_teacher_mode(teacher_mode)
    self_model_config = online_self_model_config or config
    mask_language = (
        masks_language(normalized_teacher)
        or not self_model_config.include_language_channel
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
    rng = np.random.default_rng(seed + 8_110_000)
    stats: list[EpisodeStats] = []
    proposal_deltas: list[float] = []
    chosen_deltas: list[float] = []
    oracle_deltas: list[float] = []
    regrets: list[float] = []
    overrides: list[float] = []
    working_trained = (
        deepcopy(trained) if model_control == "adaptive_dialogue" else trained
    )
    adaptation_optimizer = (
        optim.Adam(learning_rate=online_adaptation_learning_rate)
        if model_control == "adaptive_dialogue"
        else None
    )

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        observation_history = [
            _online_observation_vector(
                observation,
                env=env,
                config=self_model_config,
                mask_language=mask_language,
            )
        ]
        terminated = False
        truncated = False
        total_reward = 0.0
        viability_sum = observation.needs.mean_viability()
        min_viability = observation.needs.viability()
        resource_uses = 0
        danger_hits = 0
        teacher_utterances = 0
        steps = 0

        while not terminated and not truncated:
            exact_state_dataset = _online_state_dataset(
                env,
                observation,
                horizon=max(1, horizon),
                rng=rng,
                option_action_noise=option_action_noise,
                value_mode=value_mode,
                option_names=trained.option_names,
            )
            decision_dataset = exact_state_dataset
            if (
                online_self_model_source == "learned"
                and model_control in {"dialogue", "adaptive_dialogue"}
            ):
                decision_dataset = _learned_online_state_dataset(
                    online_self_model,
                    self_model_config,
                    env,
                    observation,
                    observation_history,
                    option_names=trained.option_names,
                    horizon=max(1, horizon),
                    rng=rng,
                    option_action_noise=option_action_noise,
                    value_mode=value_mode,
                    calibrator=online_self_model_calibrator,
                    calibration_mode=online_self_calibration_mode,
                    uncertainty_scale=online_self_calibration_uncertainty_scale,
                    uncertainty_knn=online_self_calibration_knn,
                )
            proposal = int(_partner_proposals(decision_dataset, mode=partner_mode)[0])
            pre_adaptation_choice = None
            if model_control == "adaptive_dialogue" and (
                online_adaptation_choice_guard
                or online_adaptation_risk_calibrator is not None
            ):
                pre_adaptation_choice = _online_choice(
                    working_trained,
                    decision_dataset,
                    proposal=proposal,
                    model_control="dialogue",
                    intervention=intervention,
                    seed=seed + 20_000 + episode + steps,
                )
            adapted_trained = working_trained
            adapt_optimizer = adaptation_optimizer
            if model_control == "adaptive_dialogue" and online_adaptation_local:
                adapted_trained = deepcopy(working_trained)
                adapt_optimizer = optim.Adam(
                    learning_rate=online_adaptation_learning_rate
                )
            if model_control == "adaptive_dialogue":
                adapt_dataset = intervene_situated_partner_features(
                    decision_dataset,
                    intervention=intervention,
                    seed=seed + 20_000 + episode + steps,
                )
                _adapt_online_dialogue(
                    adapted_trained,
                    adapt_dataset,
                    proposal=proposal,
                    optimizer=adapt_optimizer,
                    steps=online_adaptation_steps,
                    target_weight=online_adaptation_target_weight,
                    kl_weight=online_adaptation_kl_weight,
                    min_value_gap=online_adaptation_min_value_gap,
                )
            choice = _online_choice(
                adapted_trained,
                decision_dataset,
                proposal=proposal,
                model_control=model_control,
                intervention=intervention,
                seed=seed + 20_000 + episode + steps,
            )
            if pre_adaptation_choice is not None:
                predicted_values = np.asarray(
                    decision_dataset.option_values,
                    dtype=np.float32,
                )[0]
                if (
                    online_adaptation_choice_guard
                    and (
                        predicted_values[choice]
                        < predicted_values[pre_adaptation_choice]
                        + online_adaptation_choice_guard_margin
                    )
                ):
                    choice = pre_adaptation_choice
                elif online_adaptation_risk_calibrator is not None:
                    risk = _online_risk_estimate(
                        online_adaptation_risk_calibrator,
                        _online_risk_feature_vector(
                            decision_dataset,
                            proposal=proposal,
                            choice=choice,
                        ),
                        k=online_adaptation_risk_knn,
                    )
                    if risk > online_adaptation_risk_threshold:
                        choice = pre_adaptation_choice
            values = np.asarray(exact_state_dataset.option_values, dtype=np.float32)[0]
            current_lowest = float(np.asarray(exact_state_dataset.current_lowest)[0])
            proposal_deltas.append(float(values[proposal] - current_lowest))
            chosen_deltas.append(float(values[choice] - current_lowest))
            oracle_deltas.append(float(np.max(values) - current_lowest))
            regrets.append(float(np.max(values) - values[choice]))
            overrides.append(1.0 if choice != proposal else 0.0)

            for _commit in range(option_commit_steps):
                action = _situated_option_action(
                    trained.option_names[choice],
                    observation,
                    rng=rng,
                    option_action_noise=option_action_noise,
                )
                observation, reward, terminated, truncated, info = env.step(action)
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
                observation_history.append(
                    _online_observation_vector(
                        observation,
                        env=env,
                        config=self_model_config,
                        mask_language=mask_language,
                    )
                )
                if terminated or truncated:
                    break

        stats.append(
            EpisodeStats(
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
        )

    return OnlinePartnerResult(
        model_control=model_control,
        intervention=intervention,
        episodes=episodes,
        mean_total_reward=float(np.mean([item.total_reward for item in stats])),
        mean_steps=float(np.mean([item.steps for item in stats])),
        mean_viability=float(np.mean([item.mean_viability for item in stats])),
        mean_min_viability=float(np.mean([item.min_viability for item in stats])),
        mean_resource_uses=float(np.mean([item.resource_uses for item in stats])),
        mean_danger_hits=float(np.mean([item.danger_hits for item in stats])),
        mean_teacher_utterances=float(
            np.mean([item.teacher_utterances for item in stats])
        ),
        termination_rate=float(np.mean([item.terminated for item in stats])),
        truncation_rate=float(np.mean([item.truncated for item in stats])),
        mean_proposal_delta=float(np.mean(proposal_deltas)),
        mean_chosen_delta=float(np.mean(chosen_deltas)),
        mean_oracle_delta=float(np.mean(oracle_deltas)),
        mean_regret=float(np.mean(regrets)),
        override_rate=float(np.mean(overrides)),
    )


def _online_state_dataset(
    env: HomeostaticSocialGrid,
    observation,
    *,
    horizon: int,
    rng: np.random.Generator,
    option_action_noise: float,
    value_mode: str,
    option_names: tuple[str, ...] = OPTION_NAMES,
) -> SituatedPartnerDataset:
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    current = _needs_array(observation.needs).astype(np.float32)
    features, final, values, current_low = _situated_option_features(
        env,
        observation,
        option_names=option_names,
        horizon=horizon,
        rng=rng,
        option_action_noise=option_action_noise,
        value_mode=value_mode,
    )
    return SituatedPartnerDataset(
        features=mx.array(features[None, :, :], dtype=mx.float32),
        current_needs=mx.array(current[None, :], dtype=mx.float32),
        final_needs=mx.array(final[None, :, :], dtype=mx.float32),
        option_values=mx.array(values[None, :], dtype=mx.float32),
        target_options=mx.array([int(np.argmax(values))], dtype=mx.int32),
        current_lowest=mx.array([current_low], dtype=mx.float32),
        option_names=option_names,
    )


def _learned_online_state_dataset(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    env: HomeostaticSocialGrid,
    observation,
    observation_history: list[np.ndarray],
    *,
    option_names: tuple[str, ...],
    horizon: int,
    rng: np.random.Generator,
    option_action_noise: float,
    value_mode: str,
    calibrator: SituatedSelfModelCalibrator | None = None,
    calibration_mode: str = "all",
    uncertainty_scale: float = 1.0,
    uncertainty_knn: int = 16,
) -> SituatedPartnerDataset:
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    if not observation_history:
        raise ValueError("Learned online state dataset requires observation history.")

    current = _needs_array(observation.needs).astype(np.float32)
    action_sequences = _situated_option_action_sequences(
        env,
        observation,
        option_names=option_names,
        horizon=max(1, horizon),
        rng=rng,
        option_action_noise=option_action_noise,
    )
    max_action_length = max(len(actions) for actions in action_sequences)
    observations = np.repeat(
        np.stack(observation_history, dtype=np.float32)[None, :, :],
        len(option_names),
        axis=0,
    )
    step_masks = np.ones(observations.shape[:2], dtype=np.float32)
    observation_lengths = np.full(
        len(option_names),
        observations.shape[1],
        dtype=np.int32,
    )
    actions = np.zeros((len(option_names), max_action_length), dtype=np.int32)
    action_masks = np.zeros((len(option_names), max_action_length), dtype=np.float32)
    for index, sequence in enumerate(action_sequences):
        actions[index, : len(sequence)] = sequence
        action_masks[index, : len(sequence)] = 1.0

    _predicted_observations, predicted_needs, _predicted_rewards = (
        _option_rollout_predictions(
            model,
            mx.array(observations, dtype=mx.float32),
            mx.array(step_masks, dtype=mx.float32),
            mx.array(observation_lengths, dtype=mx.int32),
            mx.array(actions, dtype=mx.int32),
            mx.array(action_masks, dtype=mx.float32),
        )
    )
    predicted = np.asarray(predicted_needs, dtype=np.float32)
    predicted_trajectories = [
        predicted[index, : len(action_sequences[index])]
        for index in range(len(option_names))
    ]
    final = np.stack(
        [trajectory[-1] for trajectory in predicted_trajectories],
        dtype=np.float32,
    )
    values = np.array(
        [
            _option_value(trajectory, value_mode=value_mode)
            for trajectory in predicted_trajectories
        ],
        dtype=np.float32,
    )
    if calibrator is not None:
        if calibrator.option_names != option_names:
            raise ValueError("Calibrator option names do not match online option names.")
        if calibration_mode not in ONLINE_SELF_CALIBRATION_MODES:
            raise ValueError(f"Unknown online self-calibration mode: {calibration_mode}.")
        raw_values = values.copy()
        if calibration_mode == "all":
            final = np.clip(
                final * calibrator.final_scale + calibrator.final_offset,
                0.0,
                1.0,
            ).astype(np.float32)
        values = np.clip(
            values * calibrator.value_scale + calibrator.value_offset,
            0.0,
            1.0,
        ).astype(np.float32)
        if calibration_mode == "value_lcb":
            values = np.clip(
                values - uncertainty_scale * calibrator.value_rmse,
                0.0,
                1.0,
            ).astype(np.float32)
        elif calibration_mode == "value_knn_lcb":
            values = np.clip(
                values
                - uncertainty_scale
                * _calibrator_local_value_error(
                    calibrator,
                    raw_values,
                    k=uncertainty_knn,
                ),
                0.0,
                1.0,
            ).astype(np.float32)
    features = np.concatenate(
        [
            np.repeat(current[None, :], len(option_names), axis=0),
            final,
            final - current[None, :],
            np.eye(len(option_names), dtype=np.float32),
        ],
        axis=1,
    )
    return SituatedPartnerDataset(
        features=mx.array(features[None, :, :], dtype=mx.float32),
        current_needs=mx.array(current[None, :], dtype=mx.float32),
        final_needs=mx.array(final[None, :, :], dtype=mx.float32),
        option_values=mx.array(values[None, :], dtype=mx.float32),
        target_options=mx.array([int(np.argmax(values))], dtype=mx.int32),
        current_lowest=mx.array([float(np.min(current))], dtype=mx.float32),
        option_names=option_names,
    )


def _pad_situated_rank_samples(
    samples: list[SituatedSelfModelRankSample],
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    if not samples:
        raise ValueError("Cannot pad empty situated rank samples.")
    batch_size = len(samples)
    option_count = len(samples[0].option_actions)
    max_observation_length = max(len(sample.history) for sample in samples)
    max_action_length = max(
        len(actions) for sample in samples for actions in sample.option_actions
    )
    input_size = samples[0].history[0].shape[-1]
    observations = np.zeros(
        (batch_size, max_observation_length, input_size),
        dtype=np.float32,
    )
    step_masks = np.zeros((batch_size, max_observation_length), dtype=np.float32)
    observation_lengths = np.zeros(batch_size, dtype=np.int32)
    actions = np.zeros(
        (batch_size, option_count, max_action_length),
        dtype=np.int32,
    )
    action_masks = np.zeros(
        (batch_size, option_count, max_action_length),
        dtype=np.float32,
    )
    targets = np.zeros(batch_size, dtype=np.int32)

    for index, sample in enumerate(samples):
        observation_length = len(sample.history)
        observations[index, :observation_length] = np.stack(sample.history)
        step_masks[index, :observation_length] = 1.0
        observation_lengths[index] = observation_length
        for option_index, option_actions in enumerate(sample.option_actions):
            action_length = len(option_actions)
            actions[index, option_index, :action_length] = option_actions
            action_masks[index, option_index, :action_length] = 1.0
        targets[index] = sample.target_option

    return (
        mx.array(observations, dtype=mx.float32),
        mx.array(step_masks, dtype=mx.float32),
        mx.array(observation_lengths, dtype=mx.int32),
        mx.array(actions, dtype=mx.int32),
        mx.array(action_masks, dtype=mx.float32),
        mx.array(targets, dtype=mx.int32),
    )


def _situated_grouped_predicted_scores(
    model: RecurrentActorCritic,
    observations: mx.array,
    step_masks: mx.array,
    observation_lengths: mx.array,
    actions: mx.array,
    action_masks: mx.array,
    *,
    value_mode: str,
) -> mx.array:
    _final_needs, scores = _situated_grouped_predicted_final_and_scores(
        model,
        observations,
        step_masks,
        observation_lengths,
        actions,
        action_masks,
        value_mode=value_mode,
    )
    return scores


def _situated_grouped_predicted_final_and_scores(
    model: RecurrentActorCritic,
    observations: mx.array,
    step_masks: mx.array,
    observation_lengths: mx.array,
    actions: mx.array,
    action_masks: mx.array,
    *,
    value_mode: str,
) -> tuple[mx.array, mx.array]:
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    hidden = model.hidden_states(observations)
    initial = _last_valid_hidden(hidden, observation_lengths, step_masks)
    option_finals = []
    option_scores = []
    for option_index in range(actions.shape[1]):
        state = initial
        final_needs = mx.sigmoid(model.next_needs(state))
        lowest_steps = []
        for step in range(actions.shape[2]):
            action_features = mx.eye(model.action_size)[actions[:, option_index, step]]
            x = mx.concatenate([state, action_features], axis=-1)
            x = nn.relu(model.transition_norm(model.transition(x)))
            next_state = x + nn.relu(model.transition_state(x))
            next_needs = mx.sigmoid(model.next_needs(next_state))
            mask = action_masks[:, option_index, step : step + 1]
            state = mx.where(mask > 0.0, next_state, state)
            final_needs = mx.where(mask > 0.0, next_needs, final_needs)
            lowest_steps.append(
                mx.where(
                    mask[:, 0] > 0.0,
                    mx.min(next_needs, axis=-1),
                    mx.ones((actions.shape[0],), dtype=mx.float32),
                )
            )
        option_finals.append(final_needs)
        if value_mode == "final_lowest":
            option_scores.append(mx.min(final_needs, axis=-1))
        else:
            lowest = mx.stack(lowest_steps, axis=1)
            masks = action_masks[:, option_index, :]
            if value_mode == "trajectory_min":
                option_scores.append(mx.min(lowest, axis=1))
            else:
                option_scores.append(
                    mx.sum(lowest * masks, axis=1)
                    / mx.maximum(mx.sum(masks, axis=1), 1.0)
                )
    return mx.stack(option_finals, axis=1), mx.stack(option_scores, axis=1)


def _last_valid_hidden(
    hidden: mx.array,
    lengths: mx.array,
    step_masks: mx.array,
) -> mx.array:
    del step_masks
    selectors = mx.eye(hidden.shape[1])[lengths - 1]
    return mx.sum(hidden * selectors[..., None], axis=1)


def _situated_option_features(
    env: HomeostaticSocialGrid,
    observation,
    *,
    option_names: tuple[str, ...],
    horizon: int,
    rng: np.random.Generator,
    option_action_noise: float,
    value_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    final_needs: list[np.ndarray] = []
    option_trajectories: list[np.ndarray] = []
    current = _needs_array(observation.needs).astype(np.float32)
    for option_name in option_names:
        branch = deepcopy(env)
        branch_observation = observation
        trajectory_needs: list[np.ndarray] = []
        for _step in range(max(1, horizon)):
            action = _situated_option_action(
                option_name,
                branch_observation,
                rng=rng,
                option_action_noise=option_action_noise,
            )
            branch_observation, _reward, terminated, truncated, _info = branch.step(
                action
            )
            trajectory_needs.append(
                _needs_array(branch_observation.needs).astype(np.float32)
            )
            if terminated or truncated:
                break
        if not trajectory_needs:
            trajectory_needs.append(
                _needs_array(branch_observation.needs).astype(np.float32)
            )
        option_trajectories.append(np.stack(trajectory_needs).astype(np.float32))
        final_needs.append(option_trajectories[-1][-1])
    final = np.stack(final_needs).astype(np.float32)
    values = np.array(
        [
            _option_value(trajectory, value_mode=value_mode)
            for trajectory in option_trajectories
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            np.repeat(current[None, :], len(option_names), axis=0),
            final,
            final - current[None, :],
            np.eye(len(option_names), dtype=np.float32),
        ],
        axis=1,
    )
    return features.astype(np.float32), final, values, float(np.min(current))


def _situated_option_action_sequences(
    env: HomeostaticSocialGrid,
    observation,
    *,
    option_names: tuple[str, ...],
    horizon: int,
    rng: np.random.Generator,
    option_action_noise: float,
) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    actions = tuple(Action)
    for option_name in option_names:
        branch = deepcopy(env)
        branch_observation = observation
        action_indices: list[int] = []
        for _step in range(max(1, horizon)):
            action = _situated_option_action(
                option_name,
                branch_observation,
                rng=rng,
                option_action_noise=option_action_noise,
            )
            action_indices.append(actions.index(action))
            branch_observation, _reward, terminated, truncated, _info = branch.step(
                action
            )
            if terminated or truncated:
                break
        sequences.append(np.asarray(action_indices, dtype=np.int32))
    return sequences


def _online_observation_vector(
    observation,
    *,
    env: HomeostaticSocialGrid,
    config: RecurrentConfig,
    mask_language: bool,
) -> np.ndarray:
    return observation_vector(
        observation,
        width=env.width,
        height=env.height,
        include_language=config.include_language_channel,
        mask_language=mask_language,
        include_object_kinds=config.include_object_kinds,
        interoception_mode=config.interoception_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    ).astype(np.float32)


def _situated_option_names(option_set: str) -> tuple[str, ...]:
    if option_set == "base":
        return OPTION_NAMES
    if option_set == "extended":
        return EXTENDED_SITUATED_OPTION_NAMES
    raise ValueError(f"Unknown situated option set: {option_set}.")


def _situated_option_action(
    option_name: str,
    observation,
    *,
    rng: np.random.Generator,
    option_action_noise: float,
) -> Action:
    if not 0.0 <= option_action_noise <= 1.0:
        raise ValueError("Option action noise must be in [0, 1].")
    if option_name in OPTION_NAMES:
        action, _action_index = _option_action_index(
            option_name,
            observation,
            rng=None,
            option_action_noise=0.0,
        )
        return _maybe_noisy_action(
            action,
            observation,
            rng=rng,
            option_action_noise=option_action_noise,
        )
    if option_name == "seek_lowest":
        needs = observation.needs
        need_values = {
            "seek_food": needs.food,
            "seek_water": needs.water,
            "seek_shelter": min(needs.energy, needs.safety),
        }
        target = min(need_values, key=need_values.get)
        return _situated_option_action(
            target,
            observation,
            rng=rng,
            option_action_noise=option_action_noise,
        )
    if option_name == "turn_left":
        return _maybe_noisy_action(
            Action.TURN_LEFT,
            observation,
            rng=rng,
            option_action_noise=option_action_noise,
        )
    if option_name == "turn_right":
        return _maybe_noisy_action(
            Action.TURN_RIGHT,
            observation,
            rng=rng,
            option_action_noise=option_action_noise,
        )
    raise ValueError(f"Unknown situated option: {option_name}.")


def _maybe_noisy_action(
    action: Action,
    observation,
    *,
    rng: np.random.Generator,
    option_action_noise: float,
) -> Action:
    if option_action_noise <= 0.0 or rng.random() >= option_action_noise:
        return action
    actions = tuple(Action)
    mask = action_mask(observation)
    action_index = actions.index(action)
    candidates = [
        index
        for index, candidate in enumerate(actions)
        if mask[index] > 0.0
        and index != action_index
        and candidate not in {Action.POINT, Action.ASK}
    ]
    if not candidates:
        return action
    return actions[int(rng.choice(candidates))]


def _online_choice(
    trained: TrainedSituatedPartnerDialogue,
    state_dataset: SituatedPartnerDataset,
    *,
    proposal: int,
    model_control: str,
    intervention: str,
    seed: int,
) -> int:
    values = np.asarray(state_dataset.option_values, dtype=np.float32)[0]
    if model_control == "partner":
        return proposal
    if model_control == "oracle":
        return int(np.argmax(values))

    intervened = intervene_situated_partner_features(
        state_dataset,
        intervention=intervention,
        seed=seed,
    )
    features = (intervened.features - trained.feature_mean) / trained.feature_std
    proposal_signal = mx.eye(trained.model.option_count)[
        mx.array([proposal], dtype=mx.int32)
    ]
    reply_message, _reply_probs = trained.model.reply_message(
        features,
        proposal_signal,
        hard=True,
    )
    final_logits = trained.model.final(reply_message, proposal_signal)
    return int(np.asarray(mx.argmax(final_logits, axis=-1))[0])


def _online_risk_feature_vector(
    state_dataset: SituatedPartnerDataset,
    *,
    proposal: int,
    choice: int,
) -> np.ndarray:
    values = np.asarray(state_dataset.option_values, dtype=np.float32)[0]
    current = float(np.asarray(state_dataset.current_lowest, dtype=np.float32)[0])
    option_count = values.shape[0]
    if not 0 <= proposal < option_count:
        raise ValueError("Risk feature proposal index is out of range.")
    if not 0 <= choice < option_count:
        raise ValueError("Risk feature choice index is out of range.")
    sorted_values = np.sort(values)
    best = float(sorted_values[-1])
    second = float(sorted_values[-2]) if option_count > 1 else best
    chosen = float(values[choice])
    proposed = float(values[proposal])
    predicted_target = int(np.argmax(values))
    denom = float(max(1, option_count - 1))
    choice_one_hot = np.eye(option_count, dtype=np.float32)[choice]
    proposal_one_hot = np.eye(option_count, dtype=np.float32)[proposal]
    scalar_features = np.asarray(
        [
            current,
            chosen,
            chosen - current,
            chosen - proposed,
            best - second,
            best - chosen,
            float(choice != proposal),
            float(choice == predicted_target),
            float(choice) / denom,
            float(proposal) / denom,
        ],
        dtype=np.float32,
    )
    return np.concatenate([scalar_features, choice_one_hot, proposal_one_hot]).astype(
        np.float32
    )


def _online_risk_estimate(
    calibrator: OnlineAdaptationRiskCalibrator,
    feature: np.ndarray,
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("Online risk neighbor count must be positive.")
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != calibrator.feature_mean.shape:
        raise ValueError("Online risk feature shape does not match calibrator.")
    references = (calibrator.features - calibrator.feature_mean[None, :]) / (
        calibrator.feature_std[None, :] + 1e-6
    )
    query = (feature - calibrator.feature_mean) / (calibrator.feature_std + 1e-6)
    distances = np.sum((references - query[None, :]) ** 2, axis=1)
    neighbor_count = min(k, calibrator.risks.shape[0])
    nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
    return float(np.mean(calibrator.risks[nearest]))


def _adapt_online_dialogue(
    trained: TrainedSituatedPartnerDialogue,
    state_dataset: SituatedPartnerDataset,
    *,
    proposal: int,
    optimizer: optim.Optimizer,
    steps: int,
    target_weight: float,
    kl_weight: float,
    min_value_gap: float,
) -> None:
    features = (state_dataset.features - trained.feature_mean) / trained.feature_std
    values = state_dataset.option_values
    value_span = float(np.max(np.asarray(values)) - np.min(np.asarray(values)))
    if value_span < min_value_gap:
        return
    option_count = trained.model.option_count
    proposal_signal = mx.eye(option_count)[mx.array([proposal], dtype=mx.int32)]
    targets = mx.argmax(values, axis=-1)
    prior_message, _prior_probs = trained.model.reply_message(
        features,
        proposal_signal,
        hard=True,
    )
    prior_logits = mx.stop_gradient(trained.model.final(prior_message, proposal_signal))
    prior_log_probs = prior_logits - mx.logsumexp(prior_logits, axis=-1, keepdims=True)
    prior_probs = mx.stop_gradient(mx.exp(prior_log_probs))

    def loss_fn() -> mx.array:
        reply_message, _reply_probs = trained.model.reply_message(
            features,
            proposal_signal,
            hard=True,
        )
        logits = trained.model.final(reply_message, proposal_signal)
        choice_probs = mx.softmax(logits, axis=-1)
        expected_value = mx.sum(choice_probs * values, axis=-1)
        regret = mx.max(values, axis=-1) - expected_value
        loss = mx.mean(regret)
        if target_weight <= 0.0:
            target_loss = mx.array(0.0)
        else:
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            selected = mx.sum(log_probs * mx.eye(option_count)[targets], axis=-1)
            target_loss = -mx.mean(selected)
        if kl_weight <= 0.0:
            kl_loss = mx.array(0.0)
        else:
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            kl_loss = mx.mean(
                mx.sum(prior_probs * (prior_log_probs - log_probs), axis=-1)
            )
        return loss + target_weight * target_loss + kl_weight * kl_loss

    loss_and_grad = nn.value_and_grad(trained.model, loss_fn)
    for _step in range(max(1, steps)):
        loss, grads = loss_and_grad()
        optimizer.update(trained.model, grads)
        mx.eval(trained.model.parameters(), optimizer.state, loss)


def intervene_situated_partner_features(
    dataset: SituatedPartnerDataset,
    *,
    intervention: str,
    seed: int = 1,
) -> SituatedPartnerDataset:
    if intervention not in SITUATED_INTERVENTIONS:
        raise ValueError(f"Unknown situated partner intervention: {intervention}.")
    if intervention == "original":
        return dataset
    features = np.asarray(dataset.features).copy()
    current_width = 4
    final_width = 4
    final_slice = slice(current_width, current_width + final_width)
    delta_start = current_width + final_width
    delta_slice = slice(delta_start, delta_start + 4)
    rng = np.random.default_rng(seed)
    if intervention == "shuffle_delta":
        flat = features[:, :, delta_slice].reshape((-1, 4))
        rng.shuffle(flat)
        features[:, :, delta_slice] = flat.reshape(features[:, :, delta_slice].shape)
    elif intervention == "reverse_delta_rank":
        values = np.asarray(dataset.option_values, dtype=np.float32)
        order = np.argsort(values, axis=1, kind="stable")
        reverse_order = order[:, ::-1]
        rows = np.arange(features.shape[0])[:, None]
        original = features[:, :, delta_slice].copy()
        features[rows, order, delta_slice] = original[rows, reverse_order]
    elif intervention == "shuffle_outcome":
        flat_final = features[:, :, final_slice].reshape((-1, 4))
        flat_delta = features[:, :, delta_slice].reshape((-1, 4))
        permutation = rng.permutation(flat_final.shape[0])
        features[:, :, final_slice] = flat_final[permutation].reshape(
            features[:, :, final_slice].shape
        )
        features[:, :, delta_slice] = flat_delta[permutation].reshape(
            features[:, :, delta_slice].shape
        )
    elif intervention == "reverse_outcome_rank":
        values = np.asarray(dataset.option_values, dtype=np.float32)
        order = np.argsort(values, axis=1, kind="stable")
        reverse_order = order[:, ::-1]
        rows = np.arange(features.shape[0])[:, None]
        original_final = features[:, :, final_slice].copy()
        original_delta = features[:, :, delta_slice].copy()
        features[rows, order, final_slice] = original_final[rows, reverse_order]
        features[rows, order, delta_slice] = original_delta[rows, reverse_order]
    elif intervention == "zero_outcome":
        features[:, :, final_slice] = 0.0
        features[:, :, delta_slice] = 0.0
    return SituatedPartnerDataset(
        features=mx.array(features, dtype=mx.float32),
        current_needs=dataset.current_needs,
        final_needs=dataset.final_needs,
        option_values=dataset.option_values,
        target_options=dataset.target_options,
        current_lowest=dataset.current_lowest,
        option_names=dataset.option_names,
    )


def _partner_proposals(
    dataset: SituatedPartnerDataset,
    *,
    mode: str,
) -> np.ndarray:
    if mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {mode}.")
    values = np.asarray(dataset.option_values, dtype=np.float32)
    final_needs = np.asarray(dataset.final_needs, dtype=np.float32)
    if mode == "partial_body":
        partial_scores = np.min(final_needs[:, :, :2], axis=2)
        return np.argmax(partial_scores, axis=1).astype(np.int32)
    if mode == "partial_food_water":
        partial_scores = np.mean(final_needs[:, :, :2], axis=2)
        return np.argmax(partial_scores, axis=1).astype(np.int32)
    if mode == "partial_energy_safety":
        partial_scores = np.mean(final_needs[:, :, 2:], axis=2)
        return np.argmax(partial_scores, axis=1).astype(np.int32)
    if mode == "second_best":
        order = np.argsort(values, axis=1, kind="stable")
        return order[:, -2].astype(np.int32)
    if mode == "worst":
        return np.argmin(values, axis=1).astype(np.int32)
    raise ValueError(f"Unknown partner proposal mode: {mode}.")


def _option_value(needs: np.ndarray, *, value_mode: str) -> float:
    if value_mode not in SITUATED_VALUE_MODES:
        raise ValueError(f"Unknown situated value mode: {value_mode}.")
    needs = np.asarray(needs, dtype=np.float32)
    if needs.ndim == 1:
        needs = needs[None, :]
    lowest = np.min(needs, axis=1)
    if value_mode == "final_lowest":
        return float(lowest[-1])
    if value_mode == "trajectory_min":
        return float(np.min(lowest))
    if value_mode == "trajectory_mean":
        return float(np.mean(lowest))
    raise ValueError(f"Unknown situated value mode: {value_mode}.")


def target_count_string(dataset: SituatedPartnerDataset) -> str:
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    return ";".join(
        f"{name}={int(np.sum(targets == index))}"
        for index, name in enumerate(dataset.option_names)
    )


def format_result(result: SituatedPartnerResult) -> str:
    return ",".join(
        [
            result.model_control,
            result.intervention,
            str(result.samples),
            f"{result.proposal_accuracy:.4f}",
            f"{result.final_accuracy:.4f}",
            f"{result.changed_fraction:.4f}",
            f"{result.mean_proposal_delta:.6f}",
            f"{result.mean_chosen_delta:.6f}",
            f"{result.mean_oracle_delta:.6f}",
            f"{result.mean_regret:.6f}",
            result.target_counts,
        ]
    )


def format_online_result(result: OnlinePartnerResult) -> str:
    return ",".join(
        [
            result.model_control,
            result.intervention,
            str(result.episodes),
            f"{result.mean_total_reward:.6f}",
            f"{result.mean_steps:.2f}",
            f"{result.mean_viability:.6f}",
            f"{result.mean_min_viability:.6f}",
            f"{result.mean_resource_uses:.2f}",
            f"{result.mean_danger_hits:.2f}",
            f"{result.mean_teacher_utterances:.2f}",
            f"{result.termination_rate:.4f}",
            f"{result.truncation_rate:.4f}",
            f"{result.mean_proposal_delta:.6f}",
            f"{result.mean_chosen_delta:.6f}",
            f"{result.mean_oracle_delta:.6f}",
            f"{result.mean_regret:.6f}",
            f"{result.override_rate:.4f}",
        ]
    )


def main() -> None:
    args = _parse_args()
    _model, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)
    if args.option_set == "base":
        train_branches = collect_option_branch_dataset(
            config,
            episodes=args.train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            state_policy=args.state_policy,
            max_samples=args.max_train_samples,
            option_action_noise=args.option_action_noise,
        )
        train_dataset = situated_partner_dataset_from_branches(
            train_branches,
            min_value_gap=args.min_value_gap,
            min_oracle_delta=args.min_oracle_delta,
            value_mode=args.value_mode,
        )
    else:
        train_dataset = collect_situated_partner_dataset(
            config,
            episodes=args.train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            state_policy=args.state_policy,
            max_states=args.max_train_samples,
            option_action_noise=args.option_action_noise,
            option_set=args.option_set,
            min_value_gap=args.min_value_gap,
            min_oracle_delta=args.min_oracle_delta,
            value_mode=args.value_mode,
        )
    trained = train_situated_partner_dialogue(
        train_dataset,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        vocabulary_size=args.vocabulary_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        partner_mode=args.train_partner_mode,
        target_weight=args.target_weight,
        balance_weight=args.balance_weight,
        entropy_weight=args.entropy_weight,
        message_temperature=args.message_temperature,
        seed=args.seed,
    )
    rank_samples = None
    if args.online_self_rank_finetune_epochs > 0:
        rank_samples = collect_situated_self_model_rank_samples(
            config,
            episodes=args.online_self_rank_finetune_episodes,
            seed=args.seed + 40_000,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            state_policy=args.state_policy,
            max_samples=args.online_self_rank_finetune_samples,
            option_action_noise=args.online_option_action_noise,
            option_set=args.option_set,
            value_mode=args.value_mode,
        )
        train_situated_self_model_rank(
            _model,
            rank_samples,
            epochs=args.online_self_rank_finetune_epochs,
            batch_size=args.online_self_rank_finetune_batch_size,
            learning_rate=args.online_self_rank_finetune_learning_rate,
            temperature=args.online_self_rank_finetune_temperature,
            value_mode=args.value_mode,
            seed=args.seed + 40_101,
        )
    calibrator = None
    if args.online_self_calibration:
        if rank_samples is None:
            rank_samples = collect_situated_self_model_rank_samples(
                config,
                episodes=args.online_self_calibration_episodes,
                seed=args.seed + 41_000,
                teacher_mode=args.teacher_mode,
                horizon=args.horizon,
                state_policy=args.state_policy,
                max_samples=args.online_self_calibration_samples,
                option_action_noise=args.online_option_action_noise,
                option_set=args.option_set,
                value_mode=args.value_mode,
            )
        calibrator = fit_situated_self_model_calibrator(
            _model,
            rank_samples,
            option_names=train_dataset.option_names,
            value_mode=args.value_mode,
            ridge=args.online_self_calibration_ridge,
            batch_size=args.online_self_calibration_batch_size,
        )
    risk_calibrator = None
    if args.online_risk_calibration:
        risk_calibrator = collect_online_adaptation_risk_samples(
            trained,
            config,
            episodes=args.online_risk_calibration_episodes,
            seed=args.seed + 42_000,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            state_policy=args.state_policy,
            max_samples=args.online_risk_calibration_samples,
            partner_mode=args.train_partner_mode,
            option_action_noise=args.online_option_action_noise,
            value_mode=args.value_mode,
            online_self_model_source=args.online_self_model_source,
            online_self_model=_model,
            online_self_model_config=config,
            online_self_model_calibrator=calibrator,
            online_self_calibration_mode=args.online_self_calibration_mode,
            online_self_calibration_uncertainty_scale=(
                args.online_self_calibration_uncertainty_scale
            ),
            online_self_calibration_knn=args.online_self_calibration_knn,
        )
    if args.report_mode == "online":
        print(
            "model_control,intervention,episodes,mean_total_reward,mean_steps,"
            "mean_viability,mean_min_viability,mean_resource_uses,"
            "mean_danger_hits,mean_teacher_utterances,termination_rate,"
            "truncation_rate,mean_proposal_delta,mean_chosen_delta,"
            "mean_oracle_delta,mean_regret,override_rate"
        )
        for control in args.online_controls:
            interventions = (
                args.interventions
                if control in {"dialogue", "adaptive_dialogue"}
                else ["original"]
            )
            for intervention in interventions:
                print(
                    format_online_result(
                        evaluate_online_partner_dialogue(
                            trained,
                            config,
                            episodes=args.online_eval_episodes,
                            seed=args.seed + 30_000,
                            teacher_mode=args.teacher_mode,
                            horizon=args.horizon,
                            partner_mode=args.train_partner_mode,
                            model_control=control,
                            intervention=intervention,
                            option_action_noise=args.online_option_action_noise,
                            option_commit_steps=args.option_commit_steps,
                            value_mode=args.value_mode,
                            online_adaptation_steps=args.online_adaptation_steps,
                            online_adaptation_learning_rate=(
                                args.online_adaptation_learning_rate
                            ),
                            online_adaptation_target_weight=(
                                args.online_adaptation_target_weight
                            ),
                            online_adaptation_kl_weight=(
                                args.online_adaptation_kl_weight
                            ),
                            online_adaptation_min_value_gap=(
                                args.online_adaptation_min_value_gap
                            ),
                            online_adaptation_choice_guard=(
                                args.online_adaptation_choice_guard
                            ),
                            online_adaptation_choice_guard_margin=(
                                args.online_adaptation_choice_guard_margin
                            ),
                            online_adaptation_local=args.online_adaptation_local,
                            online_self_model_source=args.online_self_model_source,
                            online_self_model=_model,
                            online_self_model_config=config,
                            online_self_model_calibrator=calibrator,
                            online_self_calibration_mode=(
                                args.online_self_calibration_mode
                            ),
                            online_self_calibration_uncertainty_scale=(
                                args.online_self_calibration_uncertainty_scale
                            ),
                            online_self_calibration_knn=(
                                args.online_self_calibration_knn
                            ),
                            online_adaptation_risk_calibrator=risk_calibrator,
                            online_adaptation_risk_threshold=(
                                args.online_risk_threshold
                            ),
                            online_adaptation_risk_knn=args.online_risk_knn,
                        )
                    )
                )
        return

    if args.option_set == "base":
        eval_branches = collect_option_branch_dataset(
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            state_policy=args.state_policy,
            max_samples=args.max_eval_samples,
            option_action_noise=args.option_action_noise,
        )
        eval_dataset = situated_partner_dataset_from_branches(
            eval_branches,
            min_value_gap=args.min_value_gap,
            min_oracle_delta=args.min_oracle_delta,
            value_mode=args.value_mode,
        )
    else:
        eval_dataset = collect_situated_partner_dataset(
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            state_policy=args.state_policy,
            max_states=args.max_eval_samples,
            option_action_noise=args.option_action_noise,
            option_set=args.option_set,
            min_value_gap=args.min_value_gap,
            min_oracle_delta=args.min_oracle_delta,
            value_mode=args.value_mode,
        )
    print(
        "model_control,intervention,samples,proposal_accuracy,final_accuracy,"
        "changed_fraction,mean_proposal_delta,mean_chosen_delta,"
        "mean_oracle_delta,mean_regret,target_counts"
    )
    for intervention in args.interventions:
        intervened = intervene_situated_partner_features(
            eval_dataset,
            intervention=intervention,
            seed=args.seed + 20_000,
        )
        for partner_mode in args.partner_modes:
            print(
                format_result(
                    evaluate_situated_partner_dialogue(
                        trained,
                        intervened,
                        model_control=f"situated_{partner_mode}",
                        intervention=intervention,
                        partner_mode=partner_mode,
                    )
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-eval-samples", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--teacher-mode", choices=TEACHER_MODES, default="grounded")
    parser.add_argument("--state-policy", choices=STATE_POLICIES, default="cycle")
    parser.add_argument("--renewable-resources", action="store_true")
    parser.add_argument("--resource-ecology", choices=RESOURCE_ECOLOGIES, default=None)
    parser.add_argument("--option-action-noise", type=float, default=0.0)
    parser.add_argument("--option-set", choices=SITUATED_OPTION_SETS, default="base")
    parser.add_argument("--min-value-gap", type=float, default=0.0)
    parser.add_argument("--min-oracle-delta", type=float, default=None)
    parser.add_argument(
        "--value-mode",
        choices=SITUATED_VALUE_MODES,
        default="final_lowest",
    )
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--vocabulary-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-weight", type=float, default=0.0)
    parser.add_argument(
        "--train-partner-mode",
        choices=PARTNER_PROPOSAL_MODES,
        default="partial_body",
    )
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--message-temperature", type=float, default=0.6)
    parser.add_argument("--report-mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--online-eval-episodes", type=int, default=60)
    parser.add_argument("--online-option-action-noise", type=float, default=0.0)
    parser.add_argument("--option-commit-steps", type=int, default=1)
    parser.add_argument("--online-adaptation-steps", type=int, default=0)
    parser.add_argument("--online-adaptation-learning-rate", type=float, default=3e-4)
    parser.add_argument("--online-adaptation-target-weight", type=float, default=0.0)
    parser.add_argument("--online-adaptation-kl-weight", type=float, default=0.0)
    parser.add_argument("--online-adaptation-min-value-gap", type=float, default=0.0)
    parser.add_argument("--online-adaptation-choice-guard", action="store_true")
    parser.add_argument(
        "--online-adaptation-choice-guard-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument("--online-adaptation-local", action="store_true")
    parser.add_argument(
        "--online-self-model-source",
        choices=ONLINE_SELF_MODEL_SOURCES,
        default="exact",
    )
    parser.add_argument("--online-self-rank-finetune-epochs", type=int, default=0)
    parser.add_argument("--online-self-rank-finetune-episodes", type=int, default=80)
    parser.add_argument("--online-self-rank-finetune-samples", type=int, default=1500)
    parser.add_argument("--online-self-rank-finetune-batch-size", type=int, default=64)
    parser.add_argument(
        "--online-self-rank-finetune-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--online-self-rank-finetune-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument("--online-self-calibration", action="store_true")
    parser.add_argument("--online-self-calibration-episodes", type=int, default=80)
    parser.add_argument("--online-self-calibration-samples", type=int, default=1500)
    parser.add_argument("--online-self-calibration-batch-size", type=int, default=128)
    parser.add_argument("--online-self-calibration-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--online-self-calibration-mode",
        choices=ONLINE_SELF_CALIBRATION_MODES,
        default="all",
    )
    parser.add_argument(
        "--online-self-calibration-uncertainty-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--online-self-calibration-knn", type=int, default=16)
    parser.add_argument("--online-risk-calibration", action="store_true")
    parser.add_argument("--online-risk-calibration-episodes", type=int, default=40)
    parser.add_argument("--online-risk-calibration-samples", type=int, default=1200)
    parser.add_argument("--online-risk-threshold", type=float, default=0.02)
    parser.add_argument("--online-risk-knn", type=int, default=16)
    parser.add_argument(
        "--online-controls",
        nargs="+",
        choices=("partner", "dialogue", "adaptive_dialogue", "oracle"),
        default=["partner", "dialogue", "oracle"],
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=SITUATED_INTERVENTIONS,
        default=["original", "shuffle_outcome", "reverse_outcome_rank"],
    )
    parser.add_argument(
        "--partner-modes",
        nargs="+",
        choices=PARTNER_PROPOSAL_MODES,
        default=[
            "partial_body",
            "partial_food_water",
            "partial_energy_safety",
            "second_best",
            "worst",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
