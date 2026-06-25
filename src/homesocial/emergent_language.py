from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import _needs_array
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint
from .interoception import HISTORY_MODES, _history_control
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .report_head import _need_label
from .teachers import build_teacher, masks_language, normalize_teacher_mode


OBJECT_KINDS = ("food", "water", "shelter", "danger")
INTENT_LABELS = ("food", "water", "rest", "avoid")
MESSAGE_SLOTS = 2
MESSAGE_VOCABULARY = 3
FEATURE_MODES = ("self_estimate", "hidden")


@dataclass(frozen=True)
class CommunicationDataset:
    features: mx.array
    objects: mx.array
    decisions: mx.array
    need_labels: mx.array
    severity_labels: mx.array


@dataclass(frozen=True)
class TrainedCommunication:
    model: EmergentCommunication
    feature_mean: mx.array
    feature_std: mx.array


@dataclass(frozen=True)
class CommunicationResult:
    model_control: str
    feature_mode: str
    history_mode: str
    samples: int
    decision_accuracy: float
    balanced_accuracy: float
    use_precision: float
    use_recall: float
    pair_intent_purity: float
    slot1_need_nmi: float
    slot2_need_nmi: float
    slot1_severity_nmi: float
    slot2_severity_nmi: float
    shuffle_slot1_accuracy: float
    shuffle_slot2_accuracy: float
    shuffle_both_accuracy: float
    message_pairs_used: int
    codebook: str


@dataclass(frozen=True)
class CommunicationMediationResult:
    samples: int
    helpful_use_rate: float
    irrelevant_use_rate: float
    helpful_use_precision: float
    target_need_delta: float


class EmergentCommunication(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        receiver_size: int = 48,
        vocabulary_size: int = MESSAGE_VOCABULARY,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.sender = nn.Linear(input_size, hidden_size)
        self.sender_hidden = nn.Linear(hidden_size, hidden_size)
        self.token1 = nn.Linear(hidden_size, vocabulary_size)
        self.token2 = nn.Linear(hidden_size, vocabulary_size)
        receiver_input = 2 * vocabulary_size + len(OBJECT_KINDS)
        self.receiver = nn.Linear(receiver_input, receiver_size)
        self.receiver_hidden = nn.Linear(receiver_size, receiver_size)
        self.decision = nn.Linear(receiver_size, 2)

    def message(
        self,
        features: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        hidden = nn.relu(self.sender(features))
        hidden = hidden + nn.relu(self.sender_hidden(hidden))
        logits1 = self.token1(hidden)
        logits2 = self.token2(hidden)
        probs1 = mx.softmax(logits1 / temperature, axis=-1)
        probs2 = mx.softmax(logits2 / temperature, axis=-1)
        if not hard:
            return probs1, probs2, probs1, probs2
        hard1 = mx.eye(self.vocabulary_size)[mx.argmax(probs1, axis=-1)]
        hard2 = mx.eye(self.vocabulary_size)[mx.argmax(probs2, axis=-1)]
        straight1 = hard1 + probs1 - mx.stop_gradient(probs1)
        straight2 = hard2 + probs2 - mx.stop_gradient(probs2)
        return straight1, straight2, probs1, probs2

    def receive(
        self,
        message1: mx.array,
        message2: mx.array,
        objects: mx.array,
    ) -> mx.array:
        object_features = mx.eye(len(OBJECT_KINDS))[objects]
        inputs = mx.concatenate([message1, message2, object_features], axis=-1)
        hidden = nn.relu(self.receiver(inputs))
        hidden = hidden + nn.relu(self.receiver_hidden(hidden))
        return self.decision(hidden)

    def __call__(
        self,
        features: mx.array,
        objects: mx.array,
        *,
        temperature: float = 0.6,
    ) -> tuple[mx.array, mx.array, mx.array]:
        message1, message2, probs1, probs2 = self.message(
            features,
            temperature=temperature,
            hard=True,
        )
        return self.receive(message1, message2, objects), probs1, probs2


def collect_communication_dataset(
    base_model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    history_mode: str = "full",
    feature_mode: str = "self_estimate",
    balance_intents: bool = False,
    max_states: int = 3000,
) -> CommunicationDataset:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown communication feature mode: {feature_mode}.")
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
    rng = np.random.default_rng(seed + 1_410_000)
    state_features: list[np.ndarray] = []
    state_needs: list[int] = []
    state_severity: list[int] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
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
            hidden = _communication_features(
                base_model,
                controlled,
                feature_mode=feature_mode,
            )
            needs = _needs_array(observation.needs)
            state_features.append(hidden)
            state_needs.append(_need_label(needs))
            state_severity.append(_severity(float(np.min(needs))))

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
            mask = action_mask(observation)
            action_index = tuple(Action).index(action)
            if mask[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)
        if len(state_features) >= max_states:
            break

    state_indices = np.arange(len(state_features))
    if balance_intents:
        state_indices = _balanced_intent_indices(state_needs, rng)

    features: list[np.ndarray] = []
    objects: list[int] = []
    decisions: list[int] = []
    needs: list[int] = []
    severity: list[int] = []
    for state_index in state_indices:
        feature = state_features[int(state_index)]
        need = state_needs[int(state_index)]
        state_severity_value = state_severity[int(state_index)]
        for object_index in range(len(OBJECT_KINDS)):
            features.append(feature)
            objects.append(object_index)
            decisions.append(_helpful_decision(need, object_index))
            needs.append(need)
            severity.append(state_severity_value)

    return CommunicationDataset(
        features=mx.array(np.stack(features), dtype=mx.float32),
        objects=mx.array(objects, dtype=mx.int32),
        decisions=mx.array(decisions, dtype=mx.int32),
        need_labels=mx.array(needs, dtype=mx.int32),
        severity_labels=mx.array(severity, dtype=mx.int32),
    )


def train_communication(
    dataset: CommunicationDataset,
    *,
    hidden_size: int = 64,
    receiver_size: int = 48,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    balance_weight: float = 0.1,
    entropy_weight: float = 0.01,
    seed: int = 1,
) -> TrainedCommunication:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    model = EmergentCommunication(
        int(dataset.features.shape[-1]),
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
        batch_objects: mx.array,
        batch_decisions: mx.array,
    ) -> mx.array:
        logits, probs1, probs2 = model(batch_features, batch_objects)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(
            log_probs * mx.eye(2)[batch_decisions],
            axis=-1,
        )
        weights = mx.where(batch_decisions == 1, 3.0, 1.0)
        decision_loss = -mx.sum(selected * weights) / mx.sum(weights)
        uniform = 1.0 / MESSAGE_VOCABULARY
        balance_loss = mx.sum((mx.mean(probs1, axis=0) - uniform) ** 2)
        balance_loss += mx.sum((mx.mean(probs2, axis=0) - uniform) ** 2)
        entropy = -mx.mean(mx.sum(probs1 * mx.log(probs1 + 1e-8), axis=-1))
        entropy += -mx.mean(mx.sum(probs2 * mx.log(probs2 + 1e-8), axis=-1))
        return (
            decision_loss
            + balance_weight * balance_loss
            + entropy_weight * entropy
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
                dataset.objects[batch],
                dataset.decisions[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
    return TrainedCommunication(model, feature_mean, feature_std)


def evaluate_communication(
    trained: TrainedCommunication,
    dataset: CommunicationDataset,
    *,
    model_control: str,
    feature_mode: str,
    history_mode: str,
    seed: int,
) -> CommunicationResult:
    features = (dataset.features - trained.feature_mean) / trained.feature_std
    message1, message2, _soft1, _soft2 = trained.model.message(
        features,
        hard=True,
    )
    objects = dataset.objects
    logits = trained.model.receive(message1, message2, objects)
    predictions = np.asarray(mx.argmax(logits, axis=-1))
    targets = np.asarray(dataset.decisions)
    token1 = np.asarray(mx.argmax(message1, axis=-1))
    token2 = np.asarray(mx.argmax(message2, axis=-1))
    state_rows = np.asarray(dataset.objects) == 0
    state_need = np.asarray(dataset.need_labels)[state_rows]
    state_severity = np.asarray(dataset.severity_labels)[state_rows]
    state_token1 = token1[state_rows]
    state_token2 = token2[state_rows]
    intent = np.asarray([_intent(label) for label in state_need])
    pairs = state_token1 * MESSAGE_VOCABULARY + state_token2

    rng = np.random.default_rng(seed)
    shuffled1 = token1.copy()
    shuffled2 = token2.copy()
    rng.shuffle(shuffled1)
    rng.shuffle(shuffled2)
    slot1_accuracy = _intervention_accuracy(
        trained.model,
        shuffled1,
        token2,
        objects,
        targets,
    )
    slot2_accuracy = _intervention_accuracy(
        trained.model,
        token1,
        shuffled2,
        objects,
        targets,
    )
    both_accuracy = _intervention_accuracy(
        trained.model,
        shuffled1,
        shuffled2,
        objects,
        targets,
    )
    return CommunicationResult(
        model_control=model_control,
        feature_mode=feature_mode,
        history_mode=history_mode,
        samples=len(targets),
        decision_accuracy=float(np.mean(predictions == targets)),
        balanced_accuracy=_balanced_accuracy(targets, predictions),
        use_precision=_precision(targets, predictions),
        use_recall=_recall(targets, predictions),
        pair_intent_purity=_purity(pairs, intent),
        slot1_need_nmi=_normalized_mutual_information(state_token1, state_need),
        slot2_need_nmi=_normalized_mutual_information(state_token2, state_need),
        slot1_severity_nmi=_normalized_mutual_information(
            state_token1,
            state_severity,
        ),
        slot2_severity_nmi=_normalized_mutual_information(
            state_token2,
            state_severity,
        ),
        shuffle_slot1_accuracy=slot1_accuracy,
        shuffle_slot2_accuracy=slot2_accuracy,
        shuffle_both_accuracy=both_accuracy,
        message_pairs_used=int(len(np.unique(pairs))),
        codebook=_codebook(pairs, intent),
    )


def evaluate_communication_mediation(
    base_model: RecurrentActorCritic,
    config: RecurrentConfig,
    trained: TrainedCommunication,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    history_mode: str = "full",
    feature_mode: str = "self_estimate",
    max_samples: int = 1200,
) -> CommunicationMediationResult:
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
    rng = np.random.default_rng(seed + 1_510_000)
    helpful_trials = 0
    helpful_uses = 0
    irrelevant_trials = 0
    irrelevant_uses = 0
    target_deltas: list[float] = []
    samples = 0

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        terminated = False
        truncated = False
        while not terminated and not truncated and samples < max_samples:
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
            object_kind = (
                observation.object_ahead.kind
                if observation.object_ahead is not None
                else None
            )
            if object_kind in OBJECT_KINDS:
                controlled = _history_control(history, history_mode, rng)
                features = _communication_features(
                    base_model,
                    controlled,
                    feature_mode=feature_mode,
                )[None, :]
                normalized = (
                    mx.array(features, dtype=mx.float32) - trained.feature_mean
                ) / trained.feature_std
                message1, message2, _soft1, _soft2 = trained.model.message(
                    normalized,
                    hard=True,
                )
                object_index = OBJECT_KINDS.index(object_kind)
                logits = trained.model.receive(
                    message1,
                    message2,
                    mx.array([object_index], dtype=mx.int32),
                )
                use = int(np.asarray(mx.argmax(logits, axis=-1))[0]) == 1
                branch = deepcopy(env)
                action = (
                    Action.REST
                    if use and object_kind == "shelter"
                    else Action.CONSUME
                    if use
                    else Action.WAIT
                )
                before = _needs_array(observation.needs)
                next_observation, _reward, _terminated, _truncated, info = (
                    branch.step(action)
                )
                after = _needs_array(next_observation.needs)
                true_need = _need_label(before)
                helpful = _helpful_decision(true_need, object_index) == 1
                used = info["event"] in {
                    "consumed_food",
                    "consumed_water",
                    "rested_shelter",
                }
                if helpful:
                    helpful_trials += 1
                    helpful_uses += int(used)
                    target_deltas.append(
                        float(after[object_index] - before[object_index])
                    )
                else:
                    irrelevant_trials += 1
                    irrelevant_uses += int(used)
                samples += 1

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
            mask = action_mask(observation)
            action_index = tuple(Action).index(action)
            if mask[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)
        if samples >= max_samples:
            break

    return CommunicationMediationResult(
        samples=samples,
        helpful_use_rate=helpful_uses / max(1, helpful_trials),
        irrelevant_use_rate=irrelevant_uses / max(1, irrelevant_trials),
        helpful_use_precision=helpful_uses / max(
            1,
            helpful_uses + irrelevant_uses,
        ),
        target_need_delta=float(np.mean(target_deltas)) if target_deltas else 0.0,
    )


def _severity(value: float) -> int:
    if value < 0.30:
        return 0
    if value < 0.60:
        return 1
    return 2


def _communication_features(
    base_model: RecurrentActorCritic,
    history: list[np.ndarray],
    *,
    feature_mode: str,
) -> np.ndarray:
    observations = mx.array(np.stack(history), dtype=mx.float32)
    if feature_mode == "hidden":
        return np.asarray(
            base_model.hidden_states(observations)[-1],
            dtype=np.float32,
        )
    actions = np.zeros(len(history), dtype=np.int32)
    actions[-1] = tuple(Action).index(Action.WAIT)
    _next_observations, predicted_needs, _rewards, _utterances = (
        base_model.predict_consequences(
            observations,
            mx.array(actions, dtype=mx.int32),
        )
    )
    return np.asarray(predicted_needs[-1], dtype=np.float32)


def _helpful_decision(need_label: int, object_index: int) -> int:
    if need_label == 0:
        return int(object_index == 0)
    if need_label == 1:
        return int(object_index == 1)
    if need_label == 2:
        return int(object_index == 2)
    return 0


def _intent(need_label: int) -> int:
    return need_label if need_label < 3 else 3


def _balanced_intent_indices(
    need_labels: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    intents = np.asarray([_intent(label) for label in need_labels], dtype=np.int32)
    target = max(1, len(intents) // len(INTENT_LABELS))
    balanced: list[np.ndarray] = []
    for intent in range(len(INTENT_LABELS)):
        candidates = np.flatnonzero(intents == intent)
        if candidates.size == 0:
            return np.arange(len(intents))
        balanced.append(
            rng.choice(
                candidates,
                size=target,
                replace=candidates.size < target,
            )
        )
    result = np.concatenate(balanced)
    rng.shuffle(result)
    return result


def _intervention_accuracy(
    model: EmergentCommunication,
    token1: np.ndarray,
    token2: np.ndarray,
    objects: mx.array,
    targets: np.ndarray,
) -> float:
    message1 = mx.eye(MESSAGE_VOCABULARY)[mx.array(token1, dtype=mx.int32)]
    message2 = mx.eye(MESSAGE_VOCABULARY)[mx.array(token2, dtype=mx.int32)]
    predictions = np.asarray(
        mx.argmax(model.receive(message1, message2, objects), axis=-1)
    )
    return float(np.mean(predictions == targets))


def _balanced_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    negative = targets == 0
    positive = targets == 1
    negative_accuracy = np.mean(predictions[negative] == 0)
    positive_accuracy = np.mean(predictions[positive] == 1)
    return float((negative_accuracy + positive_accuracy) / 2)


def _precision(targets: np.ndarray, predictions: np.ndarray) -> float:
    predicted_positive = predictions == 1
    if not np.any(predicted_positive):
        return 0.0
    return float(np.mean(targets[predicted_positive] == 1))


def _recall(targets: np.ndarray, predictions: np.ndarray) -> float:
    positive = targets == 1
    if not np.any(positive):
        return 0.0
    return float(np.mean(predictions[positive] == 1))


def _purity(clusters: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    for cluster in np.unique(clusters):
        values = labels[clusters == cluster]
        correct += int(np.max(np.bincount(values)))
    return correct / len(labels)


def _codebook(pairs: np.ndarray, intents: np.ndarray) -> str:
    entries: list[str] = []
    for pair in sorted(int(value) for value in np.unique(pairs)):
        values = intents[pairs == pair]
        intent = int(np.argmax(np.bincount(values, minlength=len(INTENT_LABELS))))
        token1, token2 = divmod(pair, MESSAGE_VOCABULARY)
        entries.append(f"{token1}{token2}:{INTENT_LABELS[intent]}")
    return "|".join(entries)


def _normalized_mutual_information(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_values, first_inverse = np.unique(first, return_inverse=True)
    second_values, second_inverse = np.unique(second, return_inverse=True)
    joint = np.zeros((len(first_values), len(second_values)), dtype=np.float64)
    for left, right in zip(first_inverse, second_inverse):
        joint[left, right] += 1.0
    joint /= np.sum(joint)
    left_prob = np.sum(joint, axis=1)
    right_prob = np.sum(joint, axis=0)
    mutual_information = 0.0
    for left in range(joint.shape[0]):
        for right in range(joint.shape[1]):
            if joint[left, right] <= 0:
                continue
            mutual_information += joint[left, right] * np.log(
                joint[left, right] / (left_prob[left] * right_prob[right])
            )
    left_entropy = -np.sum(left_prob[left_prob > 0] * np.log(left_prob[left_prob > 0]))
    right_entropy = -np.sum(
        right_prob[right_prob > 0] * np.log(right_prob[right_prob > 0])
    )
    denominator = np.sqrt(left_entropy * right_entropy)
    return float(mutual_information / denominator) if denominator > 0 else 0.0


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
        "model_control,feature_mode,history_mode,samples,decision_accuracy,balanced_accuracy,"
        "use_precision,use_recall,pair_intent_purity,slot1_need_nmi,"
        "slot2_need_nmi,slot1_severity_nmi,slot2_severity_nmi,"
        "shuffle_slot1_accuracy,shuffle_slot2_accuracy,shuffle_both_accuracy,"
        "message_pairs_used,codebook,med_helpful_use,med_irrelevant_use,"
        "med_use_precision,med_target_need_delta"
    )
    for model_control, base_model in base_models:
        train_dataset = collect_communication_dataset(
            base_model,
            config,
            episodes=args.train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            history_mode="full",
            feature_mode=args.feature_mode,
            balance_intents=True,
            max_states=args.max_train_states,
        )
        trained = train_communication(
            train_dataset,
            hidden_size=args.hidden_size,
            receiver_size=args.receiver_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        for history_mode in args.history_modes:
            dataset = collect_communication_dataset(
                base_model,
                config,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                teacher_mode=args.teacher_mode,
                history_mode=history_mode,
                feature_mode=args.feature_mode,
                balance_intents=True,
                max_states=args.max_eval_states,
            )
            result = evaluate_communication(
                trained,
                dataset,
                model_control=model_control,
                feature_mode=args.feature_mode,
                history_mode=history_mode,
                seed=args.seed + 20_000,
            )
            mediation = evaluate_communication_mediation(
                base_model,
                config,
                trained,
                episodes=args.mediation_episodes,
                seed=args.seed + 30_000,
                teacher_mode=args.teacher_mode,
                history_mode=history_mode,
                feature_mode=args.feature_mode,
                max_samples=args.max_mediation_samples,
            )
            print(
                ",".join(
                    [
                        result.model_control,
                        result.feature_mode,
                        result.history_mode,
                        str(result.samples),
                        f"{result.decision_accuracy:.4f}",
                        f"{result.balanced_accuracy:.4f}",
                        f"{result.use_precision:.4f}",
                        f"{result.use_recall:.4f}",
                        f"{result.pair_intent_purity:.4f}",
                        f"{result.slot1_need_nmi:.4f}",
                        f"{result.slot2_need_nmi:.4f}",
                        f"{result.slot1_severity_nmi:.4f}",
                        f"{result.slot2_severity_nmi:.4f}",
                        f"{result.shuffle_slot1_accuracy:.4f}",
                        f"{result.shuffle_slot2_accuracy:.4f}",
                        f"{result.shuffle_both_accuracy:.4f}",
                        str(result.message_pairs_used),
                        result.codebook,
                        f"{mediation.helpful_use_rate:.4f}",
                        f"{mediation.irrelevant_use_rate:.4f}",
                        f"{mediation.helpful_use_precision:.4f}",
                        f"{mediation.target_need_delta:.4f}",
                    ]
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--train-episodes", type=int, default=180)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--mediation-episodes", type=int, default=160)
    parser.add_argument("--max-train-states", type=int, default=3000)
    parser.add_argument("--max-eval-states", type=int, default=1600)
    parser.add_argument("--max-mediation-samples", type=int, default=1200)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--receiver-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--history-modes",
        nargs="+",
        choices=HISTORY_MODES,
        default=["full", "latest", "shuffled", "reversed"],
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="self_estimate",
    )
    parser.add_argument("--random-model-control", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
