from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import _needs_array
from .env import Action, HomeostaticSocialGrid
from .hidden_mediation import (
    NeedCalibrator,
    _collect_need_pairs,
    _predict_needs,
    fit_need_calibrator,
)
from .imitation import load_checkpoint
from .interoception import HISTORY_MODES, _history_control
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .report_head import NEED_LABELS, _need_label
from .teachers import build_teacher, masks_language, normalize_teacher_mode


SEVERITY_LABELS = ("critical", "low", "adequate")
TREND_LABELS = ("improving", "steady", "worsening")
CONFIDENCE_LABELS = ("low", "medium", "high")
CAUSE_LABELS = (
    "body_event",
    "resource_action",
    "hazard",
    "ordinary_action",
    "unknown",
)


@dataclass(frozen=True)
class ConfidenceCalibrator:
    low_threshold: float
    high_threshold: float

    def label(self, margin: float) -> int:
        if margin < self.low_threshold:
            return 0
        if margin < self.high_threshold:
            return 1
        return 2


@dataclass(frozen=True)
class CompositionalReport:
    need: int
    severity: int
    trend: int
    confidence: int
    cause: int


@dataclass(frozen=True)
class CompositionalReportResult:
    model_control: str
    history_mode: str
    samples: int
    need_accuracy: float
    severity_accuracy: float
    trend_accuracy: float
    cause_accuracy: float
    exact_state_match: float
    low_confidence_coverage: float
    medium_confidence_coverage: float
    high_confidence_coverage: float
    low_confidence_accuracy: float
    medium_confidence_accuracy: float
    high_confidence_accuracy: float
    rendered_examples: tuple[str, ...]


def fit_confidence_calibrator(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    need_calibrator: NeedCalibrator,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    max_samples: int = 3000,
) -> ConfidenceCalibrator:
    predicted, _actual = _collect_need_pairs(
        model,
        config,
        episodes=episodes,
        seed=seed,
        teacher_mode=teacher_mode,
        max_samples=max_samples,
    )
    calibrated = np.stack([need_calibrator.apply(row) for row in predicted])
    margins = np.sort(calibrated, axis=1)[:, 1] - np.sort(calibrated, axis=1)[:, 0]
    low, high = np.quantile(margins, [1 / 3, 2 / 3])
    return ConfidenceCalibrator(float(low), float(high))


def evaluate_compositional_reports(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    need_calibrator: NeedCalibrator,
    confidence_calibrator: ConfidenceCalibrator,
    episodes: int,
    seed: int,
    history_mode: str = "full",
    teacher_mode: str = "grounded",
    model_control: str = "trained",
    max_samples: int = 1200,
    examples: int = 3,
) -> CompositionalReportResult:
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
    rng = np.random.default_rng(seed + 1_210_000)
    predictions: list[CompositionalReport] = []
    targets: list[CompositionalReport] = []
    rendered: list[str] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        prior_predicted: np.ndarray | None = None
        prior_actual: np.ndarray | None = None
        terminated = False
        truncated = False

        while not terminated and not truncated and len(targets) < max_samples:
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
            controlled_history = _history_control(history, history_mode, rng)
            predicted = need_calibrator.apply(
                _predict_needs(model, controlled_history)
            )
            actual = _needs_array(observation.needs)

            if prior_predicted is not None and prior_actual is not None:
                prediction = _compose(
                    predicted,
                    prior_predicted,
                    observation.last_event,
                    confidence_calibrator,
                )
                target = _compose(
                    actual,
                    prior_actual,
                    observation.last_event,
                    confidence_calibrator,
                    confidence_source=predicted,
                )
                predictions.append(prediction)
                targets.append(target)
                if len(rendered) < examples:
                    rendered.append(render_compositional_report(prediction))

            prior_predicted = predicted
            prior_actual = actual
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

        if len(targets) >= max_samples:
            break

    return _summarize(
        predictions,
        targets,
        model_control=model_control,
        history_mode=history_mode,
        rendered=rendered,
    )


def render_compositional_report(report: CompositionalReport) -> str:
    return (
        f"need={NEED_LABELS[report.need]}; "
        f"severity={SEVERITY_LABELS[report.severity]}; "
        f"trend={TREND_LABELS[report.trend]}; "
        f"confidence={CONFIDENCE_LABELS[report.confidence]}; "
        f"cause={CAUSE_LABELS[report.cause]}"
    )


def _compose(
    needs: np.ndarray,
    prior_needs: np.ndarray,
    event: str | None,
    confidence_calibrator: ConfidenceCalibrator,
    *,
    confidence_source: np.ndarray | None = None,
) -> CompositionalReport:
    need = _need_label(needs)
    focus = int(np.argmin(needs))
    source = needs if confidence_source is None else confidence_source
    sorted_source = np.sort(source)
    margin = float(sorted_source[1] - sorted_source[0])
    return CompositionalReport(
        need=need,
        severity=_severity(float(np.min(needs))),
        trend=_trend(float(needs[focus] - prior_needs[focus])),
        confidence=confidence_calibrator.label(margin),
        cause=_cause(event),
    )


def _severity(value: float) -> int:
    if value < 0.30:
        return 0
    if value < 0.60:
        return 1
    return 2


def _trend(delta: float) -> int:
    if delta > 0.025:
        return 0
    if delta < -0.025:
        return 2
    return 1


def _cause(event: str | None) -> int:
    if event is None:
        return 4
    if event.startswith("body_"):
        return 0
    if event in {"consumed_food", "consumed_water", "rested_shelter"}:
        return 1
    if event in {"hit_danger", "consumed_danger", "rested_danger"}:
        return 2
    return 3


def _summarize(
    predictions: list[CompositionalReport],
    targets: list[CompositionalReport],
    *,
    model_control: str,
    history_mode: str,
    rendered: list[str],
) -> CompositionalReportResult:
    if not targets:
        raise ValueError("Compositional report evaluation produced no samples.")
    need = np.asarray([value.need for value in predictions])
    target_need = np.asarray([value.need for value in targets])
    severity = np.asarray([value.severity for value in predictions])
    target_severity = np.asarray([value.severity for value in targets])
    trend = np.asarray([value.trend for value in predictions])
    target_trend = np.asarray([value.trend for value in targets])
    cause = np.asarray([value.cause for value in predictions])
    target_cause = np.asarray([value.cause for value in targets])
    confidence = np.asarray([value.confidence for value in predictions])
    correct_need = need == target_need
    exact = correct_need & (severity == target_severity) & (trend == target_trend)
    confidence_accuracies = tuple(
        _conditional_accuracy(correct_need, confidence == index)
        for index in range(len(CONFIDENCE_LABELS))
    )
    confidence_coverage = tuple(
        float(np.mean(confidence == index))
        for index in range(len(CONFIDENCE_LABELS))
    )
    return CompositionalReportResult(
        model_control=model_control,
        history_mode=history_mode,
        samples=len(targets),
        need_accuracy=float(np.mean(correct_need)),
        severity_accuracy=float(np.mean(severity == target_severity)),
        trend_accuracy=float(np.mean(trend == target_trend)),
        cause_accuracy=float(np.mean(cause == target_cause)),
        exact_state_match=float(np.mean(exact)),
        low_confidence_coverage=confidence_coverage[0],
        medium_confidence_coverage=confidence_coverage[1],
        high_confidence_coverage=confidence_coverage[2],
        low_confidence_accuracy=confidence_accuracies[0],
        medium_confidence_accuracy=confidence_accuracies[1],
        high_confidence_accuracy=confidence_accuracies[2],
        rendered_examples=tuple(rendered),
    )


def _conditional_accuracy(correct: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(correct[mask]))


def main() -> None:
    args = _parse_args()
    trained_model, config = load_checkpoint(args.checkpoint)
    models = [("trained", trained_model)]
    if args.random_model_control:
        mx.random.seed(args.seed)
        models.append(
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
        "model_control,history_mode,samples,need_accuracy,severity_accuracy,"
        "trend_accuracy,cause_accuracy,exact_state_match,low_conf_coverage,"
        "medium_conf_coverage,high_conf_coverage,low_conf_accuracy,"
        "medium_conf_accuracy,high_conf_accuracy,examples"
    )
    for model_control, model in models:
        need_calibrator = fit_need_calibrator(
            model,
            config,
            episodes=args.calibration_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            max_samples=args.calibration_samples,
        )
        confidence_calibrator = fit_confidence_calibrator(
            model,
            config,
            need_calibrator,
            episodes=args.calibration_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            max_samples=args.calibration_samples,
        )
        for history_mode in args.history_modes:
            result = evaluate_compositional_reports(
                model,
                config,
                need_calibrator=need_calibrator,
                confidence_calibrator=confidence_calibrator,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                history_mode=history_mode,
                teacher_mode=args.teacher_mode,
                model_control=model_control,
                max_samples=args.max_samples,
                examples=args.examples,
            )
            print(
                ",".join(
                    [
                        result.model_control,
                        result.history_mode,
                        str(result.samples),
                        f"{result.need_accuracy:.4f}",
                        f"{result.severity_accuracy:.4f}",
                        f"{result.trend_accuracy:.4f}",
                        f"{result.cause_accuracy:.4f}",
                        f"{result.exact_state_match:.4f}",
                        f"{result.low_confidence_coverage:.4f}",
                        f"{result.medium_confidence_coverage:.4f}",
                        f"{result.high_confidence_coverage:.4f}",
                        f"{result.low_confidence_accuracy:.4f}",
                        f"{result.medium_confidence_accuracy:.4f}",
                        f"{result.high_confidence_accuracy:.4f}",
                        "|".join(result.rendered_examples),
                    ]
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--calibration-episodes", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=1600)
    parser.add_argument(
        "--history-modes",
        nargs="+",
        choices=HISTORY_MODES,
        default=["full", "latest", "shuffled", "reversed"],
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument("--random-model-control", action="store_true")
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
