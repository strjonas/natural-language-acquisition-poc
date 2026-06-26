from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import _needs_array
from .emergent_language import (
    FEATURE_MODES,
    INTENT_LABELS,
    _balanced_intent_indices,
    _communication_features,
    _intent,
)
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint
from .interoception import HISTORY_MODES, _history_control
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .report_head import _need_label
from .teachers import build_teacher, masks_language, normalize_teacher_mode


STATE_MESSAGE_SLOTS = 3
STATE_MESSAGE_VOCABULARY = 4
SEVERITY_LABELS = ("critical", "low", "adequate")
TREND_LABELS = ("worsening", "steady", "improving")
SELF_STATE_FEATURE_MODES = FEATURE_MODES + ("self_estimate_delta",)
BALANCE_TARGETS = ("none", "intent", "trend")


@dataclass(frozen=True)
class SelfStateDataset:
    features: mx.array
    needs: mx.array
    low_flags: mx.array
    dominant_labels: mx.array
    severity_labels: mx.array
    trend_labels: mx.array


@dataclass(frozen=True)
class TrainedSelfStateCommunication:
    model: SelfStateCommunication
    feature_mean: mx.array
    feature_std: mx.array
    agreement_weight: float


@dataclass(frozen=True)
class SelfStateCommunicationResult:
    model_control: str
    feature_mode: str
    history_mode: str
    agreement_weight: float
    population_size: int
    samples_per_sender: int
    need_mse: float
    low_flag_accuracy: float
    dominant_accuracy: float
    severity_accuracy: float
    trend_accuracy: float
    exact_discrete_accuracy: float
    dominant_code_agreement: float
    dominant_code_distinctness: float
    message_codes_used: int
    dominant_codebook: str


class MultiSlotSender(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 96,
        slots: int = STATE_MESSAGE_SLOTS,
        vocabulary_size: int = STATE_MESSAGE_VOCABULARY,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.vocabulary_size = vocabulary_size
        self.sender = nn.Linear(input_size, hidden_size)
        self.sender_hidden = nn.Linear(hidden_size, hidden_size)
        self.tokens = [
            nn.Linear(hidden_size, vocabulary_size) for _ in range(slots)
        ]

    def __call__(
        self,
        features: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[list[mx.array], list[mx.array]]:
        hidden = nn.relu(self.sender(features))
        hidden = hidden + nn.relu(self.sender_hidden(hidden))
        messages: list[mx.array] = []
        probabilities: list[mx.array] = []
        for token in self.tokens:
            logits = token(hidden)
            probs = mx.softmax(logits / temperature, axis=-1)
            probabilities.append(probs)
            if hard:
                hard_message = mx.eye(self.vocabulary_size)[
                    mx.argmax(probs, axis=-1)
                ]
                messages.append(hard_message + probs - mx.stop_gradient(probs))
            else:
                messages.append(probs)
        return messages, probabilities


class SelfStateCommunication(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        population_size: int = 4,
        hidden_size: int = 96,
        receiver_size: int = 96,
        slots: int = STATE_MESSAGE_SLOTS,
        vocabulary_size: int = STATE_MESSAGE_VOCABULARY,
    ) -> None:
        super().__init__()
        self.population_size = population_size
        self.slots = slots
        self.vocabulary_size = vocabulary_size
        self.senders = [
            MultiSlotSender(
                input_size,
                hidden_size=hidden_size,
                slots=slots,
                vocabulary_size=vocabulary_size,
            )
            for _ in range(population_size)
        ]
        receiver_input = slots * vocabulary_size
        self.receiver = nn.Linear(receiver_input, receiver_size)
        self.receiver_hidden = nn.Linear(receiver_size, receiver_size)
        self.need = nn.Linear(receiver_size, 4)
        self.low_flags = nn.Linear(receiver_size, 8)
        self.dominant = nn.Linear(receiver_size, len(INTENT_LABELS))
        self.severity = nn.Linear(receiver_size, len(SEVERITY_LABELS))
        self.trend = nn.Linear(receiver_size, len(TREND_LABELS))

    def message(
        self,
        sender_index: int,
        features: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[list[mx.array], list[mx.array]]:
        return self.senders[sender_index](
            features,
            temperature=temperature,
            hard=hard,
        )

    def receive(
        self,
        messages: list[mx.array],
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        inputs = mx.concatenate(messages, axis=-1)
        hidden = nn.relu(self.receiver(inputs))
        hidden = hidden + nn.relu(self.receiver_hidden(hidden))
        low_logits = mx.reshape(self.low_flags(hidden), (-1, 4, 2))
        return (
            mx.sigmoid(self.need(hidden)),
            low_logits,
            self.dominant(hidden),
            self.severity(hidden),
            self.trend(hidden),
        )

    def __call__(
        self,
        sender_index: int,
        features: mx.array,
        *,
        temperature: float = 0.6,
    ) -> tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        list[mx.array],
    ]:
        messages, probabilities = self.message(
            sender_index,
            features,
            temperature=temperature,
            hard=True,
        )
        need, low, dominant, severity, trend = self.receive(messages)
        return need, low, dominant, severity, trend, probabilities


def collect_self_state_dataset(
    base_model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    history_mode: str = "full",
    feature_mode: str = "self_estimate",
    balance_intents: bool = False,
    balance_target: str | None = None,
    max_states: int = 3000,
) -> SelfStateDataset:
    if feature_mode not in SELF_STATE_FEATURE_MODES:
        raise ValueError(f"Unknown communication feature mode: {feature_mode}.")
    if history_mode not in HISTORY_MODES:
        raise ValueError(f"Unknown history mode: {history_mode}.")
    if balance_target is None:
        balance_target = "intent" if balance_intents else "none"
    if balance_target not in BALANCE_TARGETS:
        raise ValueError(f"Unknown balance target: {balance_target}.")
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
    )
    rng = np.random.default_rng(seed + 2_010_000)
    state_features: list[np.ndarray] = []
    state_needs: list[np.ndarray] = []
    low_flags: list[np.ndarray] = []
    dominant_labels: list[int] = []
    severity_labels: list[int] = []
    trend_labels: list[int] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        previous_needs: np.ndarray | None = None
        terminated = False
        truncated = False
        while not terminated and not truncated and len(state_features) < max_states:
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
            controlled = _history_control(history, history_mode, rng)
            feature = _self_state_features(
                base_model,
                controlled,
                feature_mode=feature_mode,
            )
            needs = _needs_array(observation.needs)
            state_features.append(feature)
            state_needs.append(needs)
            low_flags.append((needs < 0.60).astype(np.int32))
            dominant_labels.append(_intent(_need_label(needs)))
            severity_labels.append(_severity(float(np.min(needs))))
            trend_labels.append(_trend(previous_needs, needs))
            previous_needs = needs

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
            mask = action_mask(observation)
            if mask[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)
        if len(state_features) >= max_states:
            break

    indices = np.arange(len(state_features))
    if balance_target == "intent":
        indices = _balanced_intent_indices(dominant_labels, rng)
    elif balance_target == "trend":
        indices = _balanced_label_indices(trend_labels, rng)

    return SelfStateDataset(
        features=mx.array(
            np.stack([state_features[int(index)] for index in indices]),
            dtype=mx.float32,
        ),
        needs=mx.array(
            np.stack([state_needs[int(index)] for index in indices]),
            dtype=mx.float32,
        ),
        low_flags=mx.array(
            np.stack([low_flags[int(index)] for index in indices]),
            dtype=mx.int32,
        ),
        dominant_labels=mx.array(
            [dominant_labels[int(index)] for index in indices],
            dtype=mx.int32,
        ),
        severity_labels=mx.array(
            [severity_labels[int(index)] for index in indices],
            dtype=mx.int32,
        ),
        trend_labels=mx.array(
            [trend_labels[int(index)] for index in indices],
            dtype=mx.int32,
        ),
    )


def train_self_state_communication(
    dataset: SelfStateDataset,
    *,
    population_size: int = 4,
    hidden_size: int = 96,
    receiver_size: int = 96,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    agreement_weight: float = 0.0,
    balance_weight: float = 0.02,
    entropy_weight: float = 0.0,
    need_weight: float = 2.0,
    low_weight: float = 1.0,
    dominant_weight: float = 1.0,
    severity_weight: float = 0.5,
    trend_weight: float = 0.5,
    seed: int = 1,
) -> TrainedSelfStateCommunication:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    model = SelfStateCommunication(
        int(dataset.features.shape[-1]),
        population_size=population_size,
        hidden_size=hidden_size,
        receiver_size=receiver_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    feature_mean = mx.mean(dataset.features, axis=0, keepdims=True)
    feature_std = mx.sqrt(
        mx.mean((dataset.features - feature_mean) ** 2, axis=0, keepdims=True)
        + 1e-6
    )
    features = (dataset.features - feature_mean) / feature_std
    sample_count = int(features.shape[0])
    indices = np.arange(sample_count)

    def loss_fn(
        batch_features: mx.array,
        batch_needs: mx.array,
        batch_low_flags: mx.array,
        batch_dominant: mx.array,
        batch_severity: mx.array,
        batch_trend: mx.array,
    ) -> mx.array:
        prediction_loss = mx.array(0.0)
        balance_loss = mx.array(0.0)
        entropy = mx.array(0.0)
        all_probabilities: list[list[mx.array]] = []

        for sender_index in range(population_size):
            (
                need_prediction,
                low_logits,
                dominant_logits,
                severity_logits,
                trend_logits,
                probabilities,
            ) = model(sender_index, batch_features)

            prediction_loss = prediction_loss + need_weight * mx.mean(
                (need_prediction - batch_needs) ** 2
            )
            prediction_loss = prediction_loss + low_weight * _class_loss(
                mx.reshape(low_logits, (-1, 2)),
                mx.reshape(batch_low_flags, (-1,)),
                classes=2,
            )
            prediction_loss = prediction_loss + dominant_weight * _class_loss(
                dominant_logits,
                batch_dominant,
                classes=len(INTENT_LABELS),
            )
            prediction_loss = prediction_loss + severity_weight * _class_loss(
                severity_logits,
                batch_severity,
                classes=len(SEVERITY_LABELS),
            )
            prediction_loss = prediction_loss + trend_weight * _class_loss(
                trend_logits,
                batch_trend,
                classes=len(TREND_LABELS),
            )

            uniform = 1.0 / STATE_MESSAGE_VOCABULARY
            for probs in probabilities:
                balance_loss = balance_loss + mx.sum(
                    (mx.mean(probs, axis=0) - uniform) ** 2
                )
                entropy = entropy - mx.mean(
                    mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
                )
            all_probabilities.append(probabilities)

        prediction_loss = prediction_loss / population_size
        balance_loss = balance_loss / (population_size * STATE_MESSAGE_SLOTS)
        entropy = entropy / (population_size * STATE_MESSAGE_SLOTS)
        agreement_loss = _agreement_loss(all_probabilities)
        return (
            prediction_loss
            + balance_weight * balance_loss
            + entropy_weight * entropy
            + agreement_weight * agreement_loss
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(
                indices[start : start + max(1, batch_size)],
                dtype=mx.int32,
            )
            loss, grads = loss_and_grad(
                features[batch],
                dataset.needs[batch],
                dataset.low_flags[batch],
                dataset.dominant_labels[batch],
                dataset.severity_labels[batch],
                dataset.trend_labels[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    return TrainedSelfStateCommunication(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        agreement_weight=agreement_weight,
    )


def evaluate_self_state_communication(
    trained: TrainedSelfStateCommunication,
    dataset: SelfStateDataset,
    *,
    model_control: str,
    feature_mode: str,
    history_mode: str,
) -> SelfStateCommunicationResult:
    features = (dataset.features - trained.feature_mean) / trained.feature_std
    target_needs = np.asarray(dataset.needs)
    target_low = np.asarray(dataset.low_flags)
    target_dominant = np.asarray(dataset.dominant_labels)
    target_severity = np.asarray(dataset.severity_labels)
    target_trend = np.asarray(dataset.trend_labels)

    need_mse: list[float] = []
    low_accuracy: list[float] = []
    dominant_accuracy: list[float] = []
    severity_accuracy: list[float] = []
    trend_accuracy: list[float] = []
    exact_accuracy: list[float] = []
    code_rows: list[np.ndarray] = []
    codebooks: list[str] = []
    all_codes: list[np.ndarray] = []
    all_dominant: list[np.ndarray] = []

    for sender_index in range(trained.model.population_size):
        messages, _probabilities = trained.model.message(
            sender_index,
            features,
            hard=True,
        )
        (
            need_prediction,
            low_logits,
            dominant_logits,
            severity_logits,
            trend_logits,
        ) = trained.model.receive(messages)
        predicted_needs = np.asarray(need_prediction)
        predicted_low = np.asarray(mx.argmax(low_logits, axis=-1))
        predicted_dominant = np.asarray(mx.argmax(dominant_logits, axis=-1))
        predicted_severity = np.asarray(mx.argmax(severity_logits, axis=-1))
        predicted_trend = np.asarray(mx.argmax(trend_logits, axis=-1))
        codes = _message_codes(messages)

        need_mse.append(float(np.mean((predicted_needs - target_needs) ** 2)))
        low_accuracy.append(float(np.mean(predicted_low == target_low)))
        dominant_accuracy.append(
            float(np.mean(predicted_dominant == target_dominant))
        )
        severity_accuracy.append(
            float(np.mean(predicted_severity == target_severity))
        )
        trend_accuracy.append(float(np.mean(predicted_trend == target_trend)))
        exact_accuracy.append(
            float(
                np.mean(
                    (predicted_dominant == target_dominant)
                    & (predicted_severity == target_severity)
                    & (predicted_trend == target_trend)
                    & np.all(predicted_low == target_low, axis=1)
                )
            )
        )
        code_rows.append(_dominant_codes(codes, target_dominant))
        codebooks.append(f"s{sender_index}:{_codebook(codes, target_dominant)}")
        all_codes.append(codes)
        all_dominant.append(target_dominant)

    code_matrix = np.stack(code_rows)
    return SelfStateCommunicationResult(
        model_control=model_control,
        feature_mode=feature_mode,
        history_mode=history_mode,
        agreement_weight=trained.agreement_weight,
        population_size=trained.model.population_size,
        samples_per_sender=int(dataset.features.shape[0]),
        need_mse=float(np.mean(need_mse)),
        low_flag_accuracy=float(np.mean(low_accuracy)),
        dominant_accuracy=float(np.mean(dominant_accuracy)),
        severity_accuracy=float(np.mean(severity_accuracy)),
        trend_accuracy=float(np.mean(trend_accuracy)),
        exact_discrete_accuracy=float(np.mean(exact_accuracy)),
        dominant_code_agreement=_code_agreement(code_matrix),
        dominant_code_distinctness=_code_distinctness(code_matrix),
        message_codes_used=int(len(np.unique(np.concatenate(all_codes)))),
        dominant_codebook=";".join(codebooks),
    )


def _class_loss(logits: mx.array, labels: mx.array, *, classes: int) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    selected = mx.sum(log_probs * mx.eye(classes)[labels], axis=-1)
    return -mx.mean(selected)


def _agreement_loss(all_probabilities: list[list[mx.array]]) -> mx.array:
    if len(all_probabilities) < 2:
        return mx.array(0.0)
    loss = mx.array(0.0)
    comparisons = 0
    for left in range(len(all_probabilities)):
        for right in range(left + 1, len(all_probabilities)):
            for slot in range(len(all_probabilities[left])):
                loss = loss + mx.mean(
                    (all_probabilities[left][slot] - all_probabilities[right][slot])
                    ** 2
                )
            comparisons += len(all_probabilities[left])
    return loss / comparisons


def _severity(value: float) -> int:
    if value < 0.30:
        return 0
    if value < 0.60:
        return 1
    return 2


def _trend(previous: np.ndarray | None, current: np.ndarray) -> int:
    if previous is None:
        return 1
    delta = float(np.min(current) - np.min(previous))
    if delta < -0.015:
        return 0
    if delta > 0.015:
        return 2
    return 1


def _self_state_features(
    base_model: RecurrentActorCritic,
    history: list[np.ndarray],
    *,
    feature_mode: str,
) -> np.ndarray:
    if feature_mode != "self_estimate_delta":
        return _communication_features(
            base_model,
            history,
            feature_mode=feature_mode,
        )
    current = _communication_features(
        base_model,
        history,
        feature_mode="self_estimate",
    )
    if len(history) <= 1:
        previous = current
    else:
        previous = _communication_features(
            base_model,
            history[:-1],
            feature_mode="self_estimate",
        )
    return np.concatenate([current, current - previous]).astype(np.float32)


def _balanced_label_indices(
    labels: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int32)
    buckets = [np.flatnonzero(values == label) for label in np.unique(values)]
    if not buckets or any(bucket.size == 0 for bucket in buckets):
        return np.arange(len(values))
    target = max(1, min(bucket.size for bucket in buckets))
    balanced = [
        rng.choice(bucket, size=target, replace=bucket.size < target)
        for bucket in buckets
    ]
    result = np.concatenate(balanced)
    rng.shuffle(result)
    return result


def _message_codes(messages: list[mx.array]) -> np.ndarray:
    codes = np.zeros(int(messages[0].shape[0]), dtype=np.int32)
    for message in messages:
        codes *= STATE_MESSAGE_VOCABULARY
        codes += np.asarray(mx.argmax(message, axis=-1), dtype=np.int32)
    return codes


def _dominant_codes(codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.full(len(INTENT_LABELS), -1, dtype=np.int32)
    for label in range(len(INTENT_LABELS)):
        values = codes[labels == label]
        if len(values) == 0:
            continue
        result[label] = int(
            np.argmax(
                np.bincount(
                    values,
                    minlength=STATE_MESSAGE_VOCABULARY**STATE_MESSAGE_SLOTS,
                )
            )
        )
    return result


def _code_agreement(code_matrix: np.ndarray) -> float:
    agreements = 0
    comparisons = 0
    for left in range(code_matrix.shape[0]):
        for right in range(left + 1, code_matrix.shape[0]):
            for label in range(code_matrix.shape[1]):
                left_code = code_matrix[left, label]
                right_code = code_matrix[right, label]
                if left_code < 0 or right_code < 0:
                    continue
                agreements += int(left_code == right_code)
                comparisons += 1
    return agreements / max(1, comparisons)


def _code_distinctness(code_matrix: np.ndarray) -> float:
    distinctness: list[float] = []
    for row in code_matrix:
        valid = row[row >= 0]
        if len(valid) == 0:
            continue
        distinctness.append(len(np.unique(valid)) / len(valid))
    return float(np.mean(distinctness)) if distinctness else 0.0


def _codebook(codes: np.ndarray, labels: np.ndarray) -> str:
    entries: list[str] = []
    for code in sorted(int(value) for value in np.unique(codes)):
        values = labels[codes == code]
        label = int(np.argmax(np.bincount(values, minlength=len(INTENT_LABELS))))
        entries.append(f"{_render_code(code)}:{INTENT_LABELS[label]}")
    return "|".join(entries)


def _render_code(code: int) -> str:
    digits: list[str] = []
    value = code
    for _slot in range(STATE_MESSAGE_SLOTS):
        digits.append(str(value % STATE_MESSAGE_VOCABULARY))
        value //= STATE_MESSAGE_VOCABULARY
    return "".join(reversed(digits))


def _result_row(result: SelfStateCommunicationResult) -> str:
    return ",".join(
        [
            result.model_control,
            result.feature_mode,
            result.history_mode,
            f"{result.agreement_weight:.4f}",
            str(result.population_size),
            str(result.samples_per_sender),
            f"{result.need_mse:.6f}",
            f"{result.low_flag_accuracy:.4f}",
            f"{result.dominant_accuracy:.4f}",
            f"{result.severity_accuracy:.4f}",
            f"{result.trend_accuracy:.4f}",
            f"{result.exact_discrete_accuracy:.4f}",
            f"{result.dominant_code_agreement:.4f}",
            f"{result.dominant_code_distinctness:.4f}",
            str(result.message_codes_used),
            result.dominant_codebook,
        ]
    )


def main() -> None:
    args = _parse_args()
    trained_base, config = load_checkpoint(args.checkpoint)
    base_models = [("trained", trained_base)]
    if args.random_model_control:
        mx.random.seed(args.seed)
        base_models.append(
            (
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
        )

    print(
        "model_control,feature_mode,history_mode,agreement_weight,"
        "population_size,samples_per_sender,need_mse,low_flag_accuracy,"
        "dominant_accuracy,severity_accuracy,trend_accuracy,"
        "exact_discrete_accuracy,dominant_code_agreement,"
        "dominant_code_distinctness,message_codes_used,dominant_codebook"
    )
    for model_control, base_model in base_models:
        train_dataset = collect_self_state_dataset(
            base_model,
            config,
            episodes=args.train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            history_mode="full",
            feature_mode=args.feature_mode,
            balance_target=args.balance_target,
            max_states=args.max_train_states,
        )
        for agreement_weight in args.agreement_weights:
            trained = train_self_state_communication(
                train_dataset,
                population_size=args.population_size,
                hidden_size=args.hidden_size,
                receiver_size=args.receiver_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                agreement_weight=agreement_weight,
                balance_weight=args.balance_weight,
                entropy_weight=args.entropy_weight,
                need_weight=args.need_weight,
                low_weight=args.low_weight,
                dominant_weight=args.dominant_weight,
                severity_weight=args.severity_weight,
                trend_weight=args.trend_weight,
                seed=args.seed,
            )
            for history_mode in args.history_modes:
                eval_dataset = collect_self_state_dataset(
                    base_model,
                    config,
                    episodes=args.eval_episodes,
                    seed=args.seed + 10_000,
                    teacher_mode=args.teacher_mode,
                    history_mode=history_mode,
                    feature_mode=args.feature_mode,
                    balance_target=args.balance_target,
                    max_states=args.max_eval_states,
                )
                result = evaluate_self_state_communication(
                    trained,
                    eval_dataset,
                    model_control=model_control,
                    feature_mode=args.feature_mode,
                    history_mode=history_mode,
                )
                print(_result_row(result))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--train-episodes", type=int, default=260)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--max-train-states", type=int, default=4200)
    parser.add_argument("--max-eval-states", type=int, default=2000)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--need-weight", type=float, default=2.0)
    parser.add_argument("--low-weight", type=float, default=1.0)
    parser.add_argument("--dominant-weight", type=float, default=1.0)
    parser.add_argument("--severity-weight", type=float, default=0.5)
    parser.add_argument("--trend-weight", type=float, default=0.5)
    parser.add_argument(
        "--agreement-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.05],
    )
    parser.add_argument(
        "--history-modes",
        nargs="+",
        choices=HISTORY_MODES,
        default=["full", "latest"],
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument(
        "--feature-mode",
        choices=SELF_STATE_FEATURE_MODES,
        default="self_estimate",
    )
    parser.add_argument(
        "--balance-target",
        choices=BALANCE_TARGETS,
        default="intent",
    )
    parser.add_argument("--random-model-control", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
