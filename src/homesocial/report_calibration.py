from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .attribution import attribution_feature_layout
from .imitation import load_checkpoint
from .report_head import (
    ReportDataset,
    ReportResult,
    collect_report_dataset,
    evaluate_report_head,
    project_report_dataset,
    train_report_head,
)
from .self_battery import RolloutPolicy
from .teachers import TEACHER_MODES


@dataclass(frozen=True)
class CalibrationResult:
    variant: str
    input_size: int
    result: ReportResult


def majority_report_baseline(
    train_dataset: ReportDataset,
    eval_dataset: ReportDataset,
) -> ReportResult:
    need_prediction = _majority(train_dataset.need_labels)
    cause_prediction = _majority(train_dataset.cause_labels)
    consequence_prediction = _majority(train_dataset.consequence_labels)
    need_labels = np.asarray(eval_dataset.need_labels)
    cause_labels = np.asarray(eval_dataset.cause_labels)
    consequence_labels = np.asarray(eval_dataset.consequence_labels)
    exact = (
        (need_labels == need_prediction)
        & (cause_labels == cause_prediction)
        & (consequence_labels == consequence_prediction)
    )
    return ReportResult(
        train_samples=int(train_dataset.features.shape[0]),
        eval_samples=int(eval_dataset.features.shape[0]),
        need_accuracy=float(np.mean(need_labels == need_prediction)),
        cause_accuracy=float(np.mean(cause_labels == cause_prediction)),
        consequence_accuracy=float(
            np.mean(consequence_labels == consequence_prediction)
        ),
        exact_match_accuracy=float(np.mean(exact)),
        rendered_examples=(),
    )


def shuffled_report_dataset(dataset: ReportDataset, *, seed: int) -> ReportDataset:
    sample_count = int(dataset.features.shape[0])
    permutation = np.random.default_rng(seed).permutation(sample_count)
    indices = mx.array(permutation, dtype=mx.int32)
    return ReportDataset(
        features=dataset.features,
        need_labels=dataset.need_labels[indices],
        cause_labels=dataset.cause_labels[indices],
        consequence_labels=dataset.consequence_labels[indices],
    )


def calibrate_report_head(
    train_dataset: ReportDataset,
    eval_dataset: ReportDataset,
    *,
    recurrent_hidden_size: int,
    report_hidden_size: int = 96,
    epochs: int = 24,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 1,
) -> tuple[CalibrationResult, ...]:
    layout = attribution_feature_layout(recurrent_hidden_size)
    if int(train_dataset.features.shape[-1]) != layout.size:
        raise ValueError(
            "Report feature width does not match the recurrent hidden-size layout."
        )

    results: list[CalibrationResult] = [
        CalibrationResult(
            variant="majority",
            input_size=0,
            result=majority_report_baseline(train_dataset, eval_dataset),
        )
    ]

    full_indices = np.arange(layout.size)
    hidden_indices = _slice_indices(layout.hidden)
    engineered_indices = np.arange(layout.hidden.stop, layout.size)
    internal_indices = np.concatenate(
        [
            hidden_indices,
            _slice_indices(layout.predicted_delta),
            _slice_indices(layout.predicted_reward),
        ]
    )
    direct_label_indices = np.concatenate(
        [
            _slice_indices(layout.action),
            _slice_indices(layout.current_needs),
            _slice_indices(layout.observed_delta),
            _slice_indices(layout.teacher_utterance_present),
        ]
    )

    full_trained = train_report_head(
        train_dataset,
        hidden_size=report_hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    results.append(
        CalibrationResult(
            variant="full",
            input_size=layout.size,
            result=evaluate_report_head(full_trained, eval_dataset, examples=0),
        )
    )
    results.append(
        CalibrationResult(
            variant="full_ablate_hidden",
            input_size=layout.size,
            result=evaluate_report_head(
                full_trained,
                eval_dataset,
                examples=0,
                ablate_indices=hidden_indices,
            ),
        )
    )
    results.append(
        CalibrationResult(
            variant="full_ablate_direct_sources",
            input_size=layout.size,
            result=evaluate_report_head(
                full_trained,
                eval_dataset,
                examples=0,
                ablate_indices=direct_label_indices,
            ),
        )
    )

    for variant, indices in (
        ("hidden_only", hidden_indices),
        ("internal_model_only", internal_indices),
        (
            "prediction_only",
            np.concatenate(
                [
                    _slice_indices(layout.predicted_delta),
                    _slice_indices(layout.predicted_reward),
                ]
            ),
        ),
        ("engineered_only", engineered_indices),
        ("direct_label_sources_only", direct_label_indices),
    ):
        projected_train = project_report_dataset(train_dataset, indices)
        projected_eval = project_report_dataset(eval_dataset, indices)
        trained = train_report_head(
            projected_train,
            hidden_size=report_hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        results.append(
            CalibrationResult(
                variant=variant,
                input_size=len(indices),
                result=evaluate_report_head(trained, projected_eval, examples=0),
            )
        )

        if variant == "internal_model_only":
            internal_hidden = np.arange(len(hidden_indices))
            internal_predictions = np.arange(len(hidden_indices), len(indices))
            results.append(
                CalibrationResult(
                    variant="internal_ablate_hidden",
                    input_size=len(indices),
                    result=evaluate_report_head(
                        trained,
                        projected_eval,
                        examples=0,
                        ablate_indices=internal_hidden,
                    ),
                )
            )
            results.append(
                CalibrationResult(
                    variant="internal_ablate_predictions",
                    input_size=len(indices),
                    result=evaluate_report_head(
                        trained,
                        projected_eval,
                        examples=0,
                        ablate_indices=internal_predictions,
                    ),
                )
            )

        if variant == "hidden_only":
            quarter_edges = np.linspace(0, len(indices), 5, dtype=int)
            for quarter in range(4):
                ablated = np.arange(quarter_edges[quarter], quarter_edges[quarter + 1])
                results.append(
                    CalibrationResult(
                        variant=f"hidden_only_ablate_q{quarter + 1}",
                        input_size=len(indices),
                        result=evaluate_report_head(
                            trained,
                            projected_eval,
                            examples=0,
                            ablate_indices=ablated,
                        ),
                    )
                )

    shuffled = shuffled_report_dataset(train_dataset, seed=seed + 77_000)
    shuffled_trained = train_report_head(
        shuffled,
        hidden_size=report_hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    results.append(
        CalibrationResult(
            variant="shuffled_labels_full",
            input_size=len(full_indices),
            result=evaluate_report_head(shuffled_trained, eval_dataset, examples=0),
        )
    )
    shuffled_internal_train = project_report_dataset(shuffled, internal_indices)
    shuffled_internal_eval = project_report_dataset(eval_dataset, internal_indices)
    shuffled_internal_trained = train_report_head(
        shuffled_internal_train,
        hidden_size=report_hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    results.append(
        CalibrationResult(
            variant="shuffled_labels_internal",
            input_size=len(internal_indices),
            result=evaluate_report_head(
                shuffled_internal_trained,
                shuffled_internal_eval,
                examples=0,
            ),
        )
    )
    return tuple(results)


def _slice_indices(value: slice) -> np.ndarray:
    return np.arange(value.start, value.stop)


def _majority(labels: mx.array) -> int:
    values = np.asarray(labels, dtype=np.int32)
    if values.size == 0:
        raise ValueError("Cannot compute a majority baseline on an empty dataset.")
    return int(np.argmax(np.bincount(values)))


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
    eval_dataset = collect_report_dataset(
        model,
        config,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        teacher_mode=args.eval_teacher_mode,
        rollout_policy=args.rollout_policy,
        max_samples=args.max_eval_samples,
    )
    results = calibrate_report_head(
        train_dataset,
        eval_dataset,
        recurrent_hidden_size=config.hidden_size,
        report_hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(
        "variant,input_size,train_samples,eval_samples,need_accuracy,"
        "cause_accuracy,consequence_accuracy,exact_match"
    )
    for calibration in results:
        result = calibration.result
        print(
            ",".join(
                [
                    calibration.variant,
                    str(calibration.input_size),
                    str(result.train_samples),
                    str(result.eval_samples),
                    f"{result.need_accuracy:.4f}",
                    f"{result.cause_accuracy:.4f}",
                    f"{result.consequence_accuracy:.4f}",
                    f"{result.exact_match_accuracy:.4f}",
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1601)
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
        "--eval-teacher-mode",
        choices=TEACHER_MODES,
        default="grounded",
    )
    parser.add_argument(
        "--rollout-policy",
        choices=["teacher", "model"],
        default="teacher",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
