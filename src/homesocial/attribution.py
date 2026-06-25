from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .agents import TeacherFollowingAgent
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint
from .observations import observation_vector
from .recurrent_ac import (
    RecurrentActorCritic,
    RecurrentConfig,
    action_mask,
    choose_action_info,
)
from .self_battery import RolloutPolicy
from .teachers import TEACHER_MODES, build_teacher, masks_language, normalize_teacher_mode


CAUSE_LABELS = ("self_caused", "world_caused", "teacher_caused")
SELF_CAUSED = 0
WORLD_CAUSED = 1
TEACHER_CAUSED = 2


@dataclass(frozen=True)
class AttributionDataset:
    features: mx.array
    labels: mx.array


@dataclass(frozen=True)
class AttributionResult:
    train_samples: int
    eval_samples: int
    train_accuracy: float
    eval_accuracy: float
    per_class_accuracy: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class AttributionFeatureLayout:
    hidden: slice
    action: slice
    current_needs: slice
    observed_delta: slice
    predicted_delta: slice
    mismatch: slice
    observed_reward: slice
    predicted_reward: slice
    reward_mismatch: slice
    teacher_utterance_present: slice
    size: int


class AttributionProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, len(CAUSE_LABELS))

    def __call__(self, features: mx.array) -> mx.array:
        x = nn.relu(self.input(features))
        x = x + nn.relu(self.hidden(x))
        return self.output(x)


def attribution_feature_layout(hidden_size: int) -> AttributionFeatureLayout:
    start = 0

    def take(width: int) -> slice:
        nonlocal start
        result = slice(start, start + width)
        start += width
        return result

    return AttributionFeatureLayout(
        hidden=take(hidden_size),
        action=take(len(Action)),
        current_needs=take(4),
        observed_delta=take(4),
        predicted_delta=take(4),
        mismatch=take(4),
        observed_reward=take(1),
        predicted_reward=take(1),
        reward_mismatch=take(1),
        teacher_utterance_present=take(1),
        size=start,
    )


def collect_attribution_dataset(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    rollout_policy: RolloutPolicy = "teacher",
    max_samples: int | None = None,
) -> AttributionDataset:
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
    rng = np.random.default_rng(seed + 450_000)
    features: list[np.ndarray] = []
    labels: list[int] = []

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

            if observation.object_ahead is not None:
                _append_teacher_sample(
                    model,
                    config,
                    env,
                    obs_vectors,
                    mask_language=mask_language,
                    features=features,
                    labels=labels,
                )
                _append_self_world_samples(
                    model,
                    config,
                    env,
                    obs_vectors,
                    mask,
                    mask_language=mask_language,
                    features=features,
                    labels=labels,
                )
                if max_samples is not None and len(labels) >= max_samples:
                    return _build_dataset(features[:max_samples], labels[:max_samples])

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

    return _build_dataset(features, labels)


def train_and_evaluate_attribution(
    train_dataset: AttributionDataset,
    eval_dataset: AttributionDataset,
    *,
    hidden_size: int = 64,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 1,
) -> AttributionResult:
    if train_dataset.features.shape[0] == 0:
        raise ValueError("Cannot train attribution probe on an empty dataset.")
    if eval_dataset.features.shape[0] == 0:
        raise ValueError("Cannot evaluate attribution probe on an empty dataset.")

    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    probe = AttributionProbe(train_dataset.features.shape[-1], hidden_size=hidden_size)
    optimizer = optim.Adam(learning_rate=learning_rate)
    sample_count = int(train_dataset.features.shape[0])
    indices = np.arange(sample_count)
    feature_mean = mx.mean(train_dataset.features, axis=0, keepdims=True)
    feature_std = mx.sqrt(
        mx.mean((train_dataset.features - feature_mean) ** 2, axis=0, keepdims=True)
        + 1e-6
    )
    train_features = (train_dataset.features - feature_mean) / feature_std
    eval_features = (eval_dataset.features - feature_mean) / feature_std

    def loss_fn(features: mx.array, labels: mx.array) -> mx.array:
        logits = probe(features)
        return _cross_entropy(logits, labels)

    loss_and_grad = nn.value_and_grad(probe, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch_indices = indices[start : start + max(1, batch_size)]
            mx_indices = mx.array(batch_indices, dtype=mx.int32)
            batch_features = train_features[mx_indices]
            batch_labels = train_dataset.labels[mx_indices]
            loss, grads = loss_and_grad(batch_features, batch_labels)
            optimizer.update(probe, grads)
            mx.eval(probe.parameters(), optimizer.state, loss)

    train_predictions = _predict(probe, train_features)
    eval_predictions = _predict(probe, eval_features)
    eval_labels = np.asarray(eval_dataset.labels)
    confusion = _confusion_matrix(eval_labels, eval_predictions)
    per_class = tuple(
        _class_accuracy(eval_labels, eval_predictions, class_index)
        for class_index in range(len(CAUSE_LABELS))
    )
    return AttributionResult(
        train_samples=int(train_dataset.features.shape[0]),
        eval_samples=int(eval_dataset.features.shape[0]),
        train_accuracy=float(np.mean(train_predictions == np.asarray(train_dataset.labels))),
        eval_accuracy=float(np.mean(eval_predictions == eval_labels)),
        per_class_accuracy=per_class,
        confusion=tuple(tuple(int(value) for value in row) for row in confusion),
    )


def _append_teacher_sample(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    env: HomeostaticSocialGrid,
    obs_vectors: list[np.ndarray],
    *,
    mask_language: bool,
    features: list[np.ndarray],
    labels: list[int],
) -> None:
    mask = action_mask(env._observe(None, None))
    ask_index = tuple(Action).index(Action.ASK)
    point_index = tuple(Action).index(Action.POINT)
    action_index = ask_index if mask[ask_index] > 0.0 else point_index
    if mask[action_index] <= 0.0:
        return

    env_copy = _clone_env(env)
    next_observation, reward, _terminated, _truncated, _info = env_copy.step(
        tuple(Action)[action_index]
    )
    if next_observation.teacher_utterance is None:
        return
    features.append(
        _attribution_features(
            model,
            obs_vectors,
            action_index,
            current_needs=_needs_array(env.needs),
            observed_next_needs=_needs_array(next_observation.needs),
            observed_reward=float(reward),
            teacher_utterance_present=not mask_language,
        )
    )
    labels.append(TEACHER_CAUSED)


def _append_self_world_samples(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    env: HomeostaticSocialGrid,
    obs_vectors: list[np.ndarray],
    mask: np.ndarray,
    *,
    mask_language: bool,
    features: list[np.ndarray],
    labels: list[int],
) -> None:
    del config
    current_needs = _needs_array(env.needs)
    for action_index, allowed in enumerate(mask):
        if allowed <= 0.0:
            continue
        action = tuple(Action)[action_index]
        if action in {Action.ASK, Action.POINT, Action.WAIT}:
            continue
        env_copy = _clone_env(env)
        next_observation, reward, _terminated, _truncated, _info = env_copy.step(action)
        observed_next_needs = _needs_array(next_observation.needs)
        if not _meaningful_need_change(current_needs, observed_next_needs):
            continue

        features.append(
            _attribution_features(
                model,
                obs_vectors,
                action_index,
                current_needs=current_needs,
                observed_next_needs=observed_next_needs,
                observed_reward=float(reward),
                teacher_utterance_present=bool(
                    next_observation.teacher_utterance and not mask_language
                ),
            )
        )
        labels.append(SELF_CAUSED)

        wait_index = tuple(Action).index(Action.WAIT)
        world_reward = float(np.mean(observed_next_needs) - np.mean(current_needs))
        features.append(
            _attribution_features(
                model,
                obs_vectors,
                wait_index,
                current_needs=current_needs,
                observed_next_needs=observed_next_needs,
                observed_reward=world_reward,
                teacher_utterance_present=False,
            )
        )
        labels.append(WORLD_CAUSED)


def _attribution_features(
    model: RecurrentActorCritic,
    obs_vectors: list[np.ndarray],
    action_index: int,
    *,
    current_needs: np.ndarray,
    observed_next_needs: np.ndarray,
    observed_reward: float,
    teacher_utterance_present: bool,
) -> np.ndarray:
    observations = mx.array(np.stack(obs_vectors), dtype=mx.float32)
    hidden = np.asarray(model.hidden_states(observations)[-1])
    actions = np.zeros(len(obs_vectors), dtype=np.int32)
    actions[-1] = action_index
    _next_observations, predicted_needs, predicted_rewards, _utterances = (
        model.predict_consequences(observations, mx.array(actions, dtype=mx.int32))
    )
    predicted_next_needs = np.asarray(predicted_needs[-1])
    predicted_reward = float(np.asarray(predicted_rewards[-1]))
    action_features = np.zeros(len(Action), dtype=np.float32)
    action_features[action_index] = 1.0
    observed_delta = observed_next_needs - current_needs
    predicted_delta = predicted_next_needs - current_needs
    mismatch = observed_delta - predicted_delta
    scalar_features = np.asarray(
        [
            observed_reward,
            predicted_reward,
            observed_reward - predicted_reward,
            float(teacher_utterance_present),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            hidden.astype(np.float32),
            action_features,
            current_needs.astype(np.float32),
            observed_delta.astype(np.float32),
            predicted_delta.astype(np.float32),
            mismatch.astype(np.float32),
            scalar_features,
        ]
    )


def _build_dataset(features: list[np.ndarray], labels: list[int]) -> AttributionDataset:
    if not features:
        return AttributionDataset(
            features=mx.zeros((0, 1), dtype=mx.float32),
            labels=mx.zeros((0,), dtype=mx.int32),
        )
    return AttributionDataset(
        features=mx.array(np.stack(features), dtype=mx.float32),
        labels=mx.array(labels, dtype=mx.int32),
    )


def _clone_env(env: HomeostaticSocialGrid) -> HomeostaticSocialGrid:
    import copy

    return copy.deepcopy(env)


def _needs_array(needs) -> np.ndarray:
    return np.asarray([needs.food, needs.water, needs.energy, needs.safety], dtype=np.float32)


def _meaningful_need_change(before: np.ndarray, after: np.ndarray) -> bool:
    return bool(np.max(np.abs(after - before)) > 0.04)


def _cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    one_hot = mx.eye(logits.shape[-1])[labels]
    return -mx.mean(mx.sum(log_probs * one_hot, axis=-1))


def _predict(probe: AttributionProbe, features: mx.array) -> np.ndarray:
    logits = probe(features)
    return np.asarray(mx.argmax(logits, axis=-1))


def _confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    confusion = np.zeros((len(CAUSE_LABELS), len(CAUSE_LABELS)), dtype=np.int32)
    for label, prediction in zip(labels, predictions):
        confusion[int(label), int(prediction)] += 1
    return confusion


def _class_accuracy(labels: np.ndarray, predictions: np.ndarray, class_index: int) -> float:
    mask = labels == class_index
    if not np.any(mask):
        return float("nan")
    return float(np.mean(predictions[mask] == labels[mask]))


def main() -> None:
    args = _parse_args()
    model, config = load_checkpoint(args.checkpoint)
    train_dataset = collect_attribution_dataset(
        model,
        config,
        episodes=args.train_episodes,
        seed=args.seed,
        teacher_mode=args.teacher_mode,
        rollout_policy=args.rollout_policy,
        max_samples=args.max_train_samples,
    )
    print(
        "eval_teacher_mode,train_samples,eval_samples,train_accuracy,eval_accuracy,self_accuracy,world_accuracy,teacher_accuracy,confusion"
    )
    for mode in args.eval_teacher_modes:
        eval_dataset = collect_attribution_dataset(
            model,
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=mode,
            rollout_policy=args.rollout_policy,
            max_samples=args.max_eval_samples,
        )
        result = train_and_evaluate_attribution(
            train_dataset,
            eval_dataset,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        confusion = ";".join(
            "|".join(str(value) for value in row) for row in result.confusion
        )
        print(
            ",".join(
                [
                    mode,
                    str(result.train_samples),
                    str(result.eval_samples),
                    f"{result.train_accuracy:.4f}",
                    f"{result.eval_accuracy:.4f}",
                    f"{result.per_class_accuracy[SELF_CAUSED]:.4f}",
                    f"{result.per_class_accuracy[WORLD_CAUSED]:.4f}",
                    f"{result.per_class_accuracy[TEACHER_CAUSED]:.4f}",
                    confusion,
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--train-episodes", type=int, default=160)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-eval-samples", type=int, default=1200)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--teacher-mode", choices=TEACHER_MODES, default="grounded")
    parser.add_argument(
        "--eval-teacher-modes",
        nargs="+",
        choices=TEACHER_MODES,
        default=["grounded", "masked", "shuffled", "wrong"],
    )
    parser.add_argument(
        "--rollout-policy",
        choices=["teacher", "model"],
        default="teacher",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
