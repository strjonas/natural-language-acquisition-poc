from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import (
    CAUSE_LABELS,
    SELF_CAUSED,
    TEACHER_CAUSED,
    WORLD_CAUSED,
    _attribution_features,
    _clone_env,
    _needs_array,
)
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


NEED_LABELS = ("need_food", "need_water", "need_rest", "unsafe", "stable")
CONSEQUENCE_LABELS = (
    "food_up",
    "water_up",
    "energy_up",
    "safety_down",
    "no_major_change",
)


@dataclass(frozen=True)
class ReportDataset:
    features: mx.array
    need_labels: mx.array
    cause_labels: mx.array
    consequence_labels: mx.array


@dataclass(frozen=True)
class ReportResult:
    train_samples: int
    eval_samples: int
    need_accuracy: float
    cause_accuracy: float
    consequence_accuracy: float
    exact_match_accuracy: float
    rendered_examples: tuple[str, ...]


class SelfReportHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 96) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.need = nn.Linear(hidden_size, len(NEED_LABELS))
        self.cause = nn.Linear(hidden_size, len(CAUSE_LABELS))
        self.consequence = nn.Linear(hidden_size, len(CONSEQUENCE_LABELS))

    def __call__(self, features: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        x = nn.relu(self.input(features))
        x = x + nn.relu(self.hidden(x))
        return self.need(x), self.cause(x), self.consequence(x)


def collect_report_dataset(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    rollout_policy: RolloutPolicy = "teacher",
    max_samples: int | None = None,
) -> ReportDataset:
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
    )
    rng = np.random.default_rng(seed + 550_000)
    features: list[np.ndarray] = []
    need_labels: list[int] = []
    cause_labels: list[int] = []
    consequence_labels: list[int] = []

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
            )
            obs_vectors.append(vector)
            mask = action_mask(observation)
            action_masks.append(mask)

            if observation.object_ahead is not None:
                _append_teacher_report_sample(
                    model,
                    env,
                    obs_vectors,
                    mask_language=mask_language,
                    features=features,
                    need_labels=need_labels,
                    cause_labels=cause_labels,
                    consequence_labels=consequence_labels,
                )
                _append_self_world_report_samples(
                    model,
                    env,
                    obs_vectors,
                    mask,
                    mask_language=mask_language,
                    features=features,
                    need_labels=need_labels,
                    cause_labels=cause_labels,
                    consequence_labels=consequence_labels,
                )
                if max_samples is not None and len(cause_labels) >= max_samples:
                    return _build_report_dataset(
                        features[:max_samples],
                        need_labels[:max_samples],
                        cause_labels[:max_samples],
                        consequence_labels[:max_samples],
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

    return _build_report_dataset(
        features,
        need_labels,
        cause_labels,
        consequence_labels,
    )


def train_and_evaluate_report_head(
    train_dataset: ReportDataset,
    eval_dataset: ReportDataset,
    *,
    hidden_size: int = 96,
    epochs: int = 24,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 1,
    examples: int = 5,
) -> ReportResult:
    if train_dataset.features.shape[0] == 0:
        raise ValueError("Cannot train report head on an empty dataset.")
    if eval_dataset.features.shape[0] == 0:
        raise ValueError("Cannot evaluate report head on an empty dataset.")

    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    report = SelfReportHead(train_dataset.features.shape[-1], hidden_size=hidden_size)
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

    def loss_fn(
        features: mx.array,
        need_labels: mx.array,
        cause_labels: mx.array,
        consequence_labels: mx.array,
    ) -> mx.array:
        need_logits, cause_logits, consequence_logits = report(features)
        return (
            _cross_entropy(need_logits, need_labels)
            + _cross_entropy(cause_logits, cause_labels)
            + _cross_entropy(consequence_logits, consequence_labels)
        )

    loss_and_grad = nn.value_and_grad(report, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch_indices = indices[start : start + max(1, batch_size)]
            mx_indices = mx.array(batch_indices, dtype=mx.int32)
            loss, grads = loss_and_grad(
                train_features[mx_indices],
                train_dataset.need_labels[mx_indices],
                train_dataset.cause_labels[mx_indices],
                train_dataset.consequence_labels[mx_indices],
            )
            optimizer.update(report, grads)
            mx.eval(report.parameters(), optimizer.state, loss)

    need_predictions, cause_predictions, consequence_predictions = _predict(
        report,
        eval_features,
    )
    need_labels = np.asarray(eval_dataset.need_labels)
    cause_labels = np.asarray(eval_dataset.cause_labels)
    consequence_labels = np.asarray(eval_dataset.consequence_labels)
    exact = (
        (need_predictions == need_labels)
        & (cause_predictions == cause_labels)
        & (consequence_predictions == consequence_labels)
    )
    rendered = tuple(
        render_report(
            int(need_predictions[index]),
            int(cause_predictions[index]),
            int(consequence_predictions[index]),
        )
        for index in range(min(examples, len(need_predictions)))
    )
    return ReportResult(
        train_samples=sample_count,
        eval_samples=int(eval_dataset.features.shape[0]),
        need_accuracy=float(np.mean(need_predictions == need_labels)),
        cause_accuracy=float(np.mean(cause_predictions == cause_labels)),
        consequence_accuracy=float(
            np.mean(consequence_predictions == consequence_labels)
        ),
        exact_match_accuracy=float(np.mean(exact)),
        rendered_examples=rendered,
    )


def render_report(need_index: int, cause_index: int, consequence_index: int) -> str:
    return (
        f"need={NEED_LABELS[need_index]}; "
        f"cause={CAUSE_LABELS[cause_index]}; "
        f"consequence={CONSEQUENCE_LABELS[consequence_index]}"
    )


def _append_teacher_report_sample(
    model: RecurrentActorCritic,
    env: HomeostaticSocialGrid,
    obs_vectors: list[np.ndarray],
    *,
    mask_language: bool,
    features: list[np.ndarray],
    need_labels: list[int],
    cause_labels: list[int],
    consequence_labels: list[int],
) -> None:
    ask_index = tuple(Action).index(Action.ASK)
    env_copy = _clone_env(env)
    next_observation, reward, _terminated, _truncated, _info = env_copy.step(Action.ASK)
    if next_observation.teacher_utterance is None:
        return
    current_needs = _needs_array(env.needs)
    observed_next_needs = _needs_array(next_observation.needs)
    features.append(
        _attribution_features(
            model,
            obs_vectors,
            ask_index,
            current_needs=current_needs,
            observed_next_needs=observed_next_needs,
            observed_reward=float(reward),
            teacher_utterance_present=not mask_language,
        )
    )
    need_labels.append(_need_label(current_needs))
    cause_labels.append(TEACHER_CAUSED)
    consequence_labels.append(_consequence_label(observed_next_needs - current_needs))


def _append_self_world_report_samples(
    model: RecurrentActorCritic,
    env: HomeostaticSocialGrid,
    obs_vectors: list[np.ndarray],
    mask: np.ndarray,
    *,
    mask_language: bool,
    features: list[np.ndarray],
    need_labels: list[int],
    cause_labels: list[int],
    consequence_labels: list[int],
) -> None:
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
        delta = observed_next_needs - current_needs
        if np.max(np.abs(delta)) <= 0.04:
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
        need_labels.append(_need_label(current_needs))
        cause_labels.append(SELF_CAUSED)
        consequence_labels.append(_consequence_label(delta))

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
        need_labels.append(_need_label(current_needs))
        cause_labels.append(WORLD_CAUSED)
        consequence_labels.append(_consequence_label(delta))


def _need_label(needs: np.ndarray) -> int:
    if float(np.min(needs)) > 0.65:
        return NEED_LABELS.index("stable")
    lowest = int(np.argmin(needs))
    if lowest == 0:
        return NEED_LABELS.index("need_food")
    if lowest == 1:
        return NEED_LABELS.index("need_water")
    if lowest == 2:
        return NEED_LABELS.index("need_rest")
    return NEED_LABELS.index("unsafe")


def _consequence_label(delta: np.ndarray) -> int:
    if delta[3] < -0.05:
        return CONSEQUENCE_LABELS.index("safety_down")
    if delta[0] > 0.05:
        return CONSEQUENCE_LABELS.index("food_up")
    if delta[1] > 0.05:
        return CONSEQUENCE_LABELS.index("water_up")
    if delta[2] > 0.05:
        return CONSEQUENCE_LABELS.index("energy_up")
    return CONSEQUENCE_LABELS.index("no_major_change")


def _build_report_dataset(
    features: list[np.ndarray],
    need_labels: list[int],
    cause_labels: list[int],
    consequence_labels: list[int],
) -> ReportDataset:
    if not features:
        return ReportDataset(
            features=mx.zeros((0, 1), dtype=mx.float32),
            need_labels=mx.zeros((0,), dtype=mx.int32),
            cause_labels=mx.zeros((0,), dtype=mx.int32),
            consequence_labels=mx.zeros((0,), dtype=mx.int32),
        )
    return ReportDataset(
        features=mx.array(np.stack(features), dtype=mx.float32),
        need_labels=mx.array(need_labels, dtype=mx.int32),
        cause_labels=mx.array(cause_labels, dtype=mx.int32),
        consequence_labels=mx.array(consequence_labels, dtype=mx.int32),
    )


def _cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    one_hot = mx.eye(logits.shape[-1])[labels]
    return -mx.mean(mx.sum(log_probs * one_hot, axis=-1))


def _predict(report: SelfReportHead, features: mx.array) -> tuple[np.ndarray, ...]:
    need_logits, cause_logits, consequence_logits = report(features)
    return (
        np.asarray(mx.argmax(need_logits, axis=-1)),
        np.asarray(mx.argmax(cause_logits, axis=-1)),
        np.asarray(mx.argmax(consequence_logits, axis=-1)),
    )


def main() -> None:
    args = _parse_args()
    model, config = load_checkpoint(args.checkpoint)
    train_dataset = collect_report_dataset(
        model,
        config,
        episodes=args.train_episodes,
        seed=args.seed,
        teacher_mode=args.teacher_mode,
        rollout_policy=args.rollout_policy,
        max_samples=args.max_train_samples,
    )
    print(
        "eval_teacher_mode,train_samples,eval_samples,need_accuracy,cause_accuracy,consequence_accuracy,exact_match,examples"
    )
    for mode in args.eval_teacher_modes:
        eval_dataset = collect_report_dataset(
            model,
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=mode,
            rollout_policy=args.rollout_policy,
            max_samples=args.max_eval_samples,
        )
        result = train_and_evaluate_report_head(
            train_dataset,
            eval_dataset,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            examples=args.examples,
        )
        print(
            ",".join(
                [
                    mode,
                    str(result.train_samples),
                    str(result.eval_samples),
                    f"{result.need_accuracy:.4f}",
                    f"{result.cause_accuracy:.4f}",
                    f"{result.consequence_accuracy:.4f}",
                    f"{result.exact_match_accuracy:.4f}",
                    "|".join(result.rendered_examples),
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--train-episodes", type=int, default=160)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-eval-samples", type=int, default=1200)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=24)
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
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
