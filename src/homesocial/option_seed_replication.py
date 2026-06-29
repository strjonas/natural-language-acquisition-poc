from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from math import nan

import mlx.core as mx
import numpy as np

from .env import RESOURCE_ECOLOGIES, Action
from .imitation import load_checkpoint
from .observations import observation_vector_size
from .option_counterfactual_language import (
    OPTION_BALANCE_TARGETS,
    STATE_POLICIES,
    collect_option_counterfactual_dataset,
)
from .option_feature_intervention import (
    OPTION_FEATURE_INTERVENTIONS,
    intervene_option_features,
)
from .option_world_model import (
    collect_option_branch_dataset,
    evaluate_option_world_model,
    train_option_world_model,
)
from .recurrent_ac import RecurrentActorCritic
from .self_state_language import (
    SelfStateDataset,
    evaluate_self_state_communication,
    train_self_state_communication,
)


@dataclass(frozen=True)
class OptionSeedReplicationRow:
    seed: int
    model_control: str
    intervention: str
    samples: int
    trend_accuracy: float
    trend_drop_from_original: float
    need_mse: float
    dominant_accuracy: float
    severity_accuracy: float
    low_flag_accuracy: float
    exact_discrete_accuracy: float
    world_final_need_mse: float
    world_trend_accuracy: float
    trained_world_final_need_mse_before: float
    trained_world_final_need_mse_after: float
    trained_world_trend_before: float
    trained_world_trend_after: float


def option_majority_trend_accuracy(
    train_dataset: SelfStateDataset,
    eval_dataset: SelfStateDataset,
) -> float:
    if train_dataset.action_labels is None or eval_dataset.action_labels is None:
        raise ValueError("Option-majority baseline requires option labels.")
    train_options = np.asarray(train_dataset.action_labels)
    train_trends = np.asarray(train_dataset.trend_labels)
    eval_options = np.asarray(eval_dataset.action_labels)
    eval_trends = np.asarray(eval_dataset.trend_labels)
    global_counts = np.bincount(train_trends, minlength=3)
    global_prediction = int(np.argmax(global_counts))
    option_predictions: dict[int, int] = {}
    for option in np.unique(train_options):
        option_trends = train_trends[train_options == option]
        counts = np.bincount(option_trends, minlength=3)
        option_predictions[int(option)] = int(np.argmax(counts))
    predictions = np.array(
        [
            option_predictions.get(int(option), global_prediction)
            for option in eval_options
        ],
        dtype=np.int32,
    )
    return float(np.mean(predictions == eval_trends))


def format_row(row: OptionSeedReplicationRow) -> str:
    return ",".join(
        [
            str(row.seed),
            row.model_control,
            row.intervention,
            str(row.samples),
            _format_float(row.trend_accuracy, 4),
            _format_float(row.trend_drop_from_original, 4),
            _format_float(row.need_mse, 6),
            _format_float(row.dominant_accuracy, 4),
            _format_float(row.severity_accuracy, 4),
            _format_float(row.low_flag_accuracy, 4),
            _format_float(row.exact_discrete_accuracy, 4),
            _format_float(row.world_final_need_mse, 6),
            _format_float(row.world_trend_accuracy, 4),
            _format_float(row.trained_world_final_need_mse_before, 6),
            _format_float(row.trained_world_final_need_mse_after, 6),
            _format_float(row.trained_world_trend_before, 4),
            _format_float(row.trained_world_trend_after, 4),
        ]
    )


def header() -> str:
    return (
        "seed,model_control,intervention,samples,trend_accuracy,"
        "trend_drop_from_original,need_mse,dominant_accuracy,"
        "severity_accuracy,low_flag_accuracy,exact_discrete_accuracy,"
        "world_final_need_mse,world_trend_accuracy,"
        "trained_world_final_need_mse_before,trained_world_final_need_mse_after,"
        "trained_world_trend_before,trained_world_trend_after"
    )


def main() -> None:
    args = _parse_args()
    print(header())
    for seed in args.seeds:
        rows = run_replication_seed(args, seed=seed)
        for row in rows:
            print(format_row(row))


def run_replication_seed(
    args: argparse.Namespace,
    *,
    seed: int,
) -> list[OptionSeedReplicationRow]:
    trained_base, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)

    train_branches = collect_option_branch_dataset(
        config,
        episodes=args.world_train_episodes,
        seed=seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_world_train_samples,
    )
    eval_branches = collect_option_branch_dataset(
        config,
        episodes=args.world_eval_episodes,
        seed=seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_world_eval_samples,
    )
    before_world = evaluate_option_world_model(
        trained_base,
        eval_branches,
        batch_size=args.world_eval_batch_size,
    )
    train_option_world_model(
        trained_base,
        config,
        train_branches,
        checkpoint_path=None,
        epochs=args.world_epochs,
        batch_size=args.world_batch_size,
        learning_rate=args.world_learning_rate,
        seed=seed,
        step_needs_weight=args.step_needs_weight,
        final_needs_weight=args.final_needs_weight,
        observation_prediction_weight=args.observation_prediction_weight,
        reward_prediction_weight=args.reward_prediction_weight,
    )
    after_world = evaluate_option_world_model(
        trained_base,
        eval_branches,
        batch_size=args.world_eval_batch_size,
    )

    rows = _communication_rows(
        trained_base,
        config,
        seed=seed,
        model_control="trained",
        args=args,
        world_final_need_mse=after_world.final_need_mse,
        world_trend_accuracy=after_world.trend_accuracy,
        before_world_final_need_mse=before_world.final_need_mse,
        after_world_final_need_mse=after_world.final_need_mse,
        before_world_trend_accuracy=before_world.trend_accuracy,
        after_world_trend_accuracy=after_world.trend_accuracy,
        include_option_majority=True,
    )

    if args.random_model_control:
        mx.random.seed(seed)
        random_base = RecurrentActorCritic(
            observation_vector_size(
                include_language=config.include_language_channel,
                include_object_kinds=config.include_object_kinds,
                body_dynamics_mode=config.body_dynamics_mode,
            ),
            config.hidden_size,
            len(Action),
        )
        random_world = evaluate_option_world_model(
            random_base,
            eval_branches,
            batch_size=args.world_eval_batch_size,
        )
        rows.extend(
            _communication_rows(
                random_base,
                config,
                seed=seed,
                model_control="random",
                args=args,
                world_final_need_mse=random_world.final_need_mse,
                world_trend_accuracy=random_world.trend_accuracy,
                before_world_final_need_mse=before_world.final_need_mse,
                after_world_final_need_mse=after_world.final_need_mse,
                before_world_trend_accuracy=before_world.trend_accuracy,
                after_world_trend_accuracy=after_world.trend_accuracy,
                include_option_majority=False,
            )
        )
    return rows


def _communication_rows(
    base_model: RecurrentActorCritic,
    config,
    *,
    seed: int,
    model_control: str,
    args: argparse.Namespace,
    world_final_need_mse: float,
    world_trend_accuracy: float,
    before_world_final_need_mse: float,
    after_world_final_need_mse: float,
    before_world_trend_accuracy: float,
    after_world_trend_accuracy: float,
    include_option_majority: bool,
) -> list[OptionSeedReplicationRow]:
    train_dataset = collect_option_counterfactual_dataset(
        base_model,
        config,
        episodes=args.language_train_episodes,
        seed=seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        balance_target=args.balance_target,
        rollout_mode="latent_current",
        state_policy=args.state_policy,
        max_states=args.max_language_train_states,
    )
    eval_dataset = collect_option_counterfactual_dataset(
        base_model,
        config,
        episodes=args.language_eval_episodes,
        seed=seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        balance_target=args.balance_target,
        rollout_mode="latent_current",
        state_policy=args.state_policy,
        max_states=args.max_language_eval_states,
    )
    trained = train_self_state_communication(
        train_dataset,
        population_size=args.population_size,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        epochs=args.language_epochs,
        batch_size=args.language_batch_size,
        learning_rate=args.language_learning_rate,
        agreement_weight=args.agreement_weight,
        balance_weight=args.balance_weight,
        entropy_weight=args.entropy_weight,
        need_weight=args.need_weight,
        low_weight=args.low_weight,
        dominant_weight=args.dominant_weight,
        severity_weight=args.severity_weight,
        trend_weight=args.trend_weight,
        seed=seed,
    )

    rows: list[OptionSeedReplicationRow] = []
    original_trend = nan
    for intervention in args.interventions:
        intervened = intervene_option_features(
            eval_dataset,
            intervention=intervention,
            seed=seed + 20_000,
        )
        result = evaluate_self_state_communication(
            trained,
            intervened,
            model_control=model_control,
            feature_mode="option_counterfactual_latent_current",
            history_mode=intervention,
        )
        if intervention == "original":
            original_trend = result.trend_accuracy
        rows.append(
            OptionSeedReplicationRow(
                seed=seed,
                model_control=model_control,
                intervention=intervention,
                samples=result.samples_per_sender,
                trend_accuracy=result.trend_accuracy,
                trend_drop_from_original=original_trend - result.trend_accuracy,
                need_mse=result.need_mse,
                dominant_accuracy=result.dominant_accuracy,
                severity_accuracy=result.severity_accuracy,
                low_flag_accuracy=result.low_flag_accuracy,
                exact_discrete_accuracy=result.exact_discrete_accuracy,
                world_final_need_mse=world_final_need_mse,
                world_trend_accuracy=world_trend_accuracy,
                trained_world_final_need_mse_before=before_world_final_need_mse,
                trained_world_final_need_mse_after=after_world_final_need_mse,
                trained_world_trend_before=before_world_trend_accuracy,
                trained_world_trend_after=after_world_trend_accuracy,
            )
        )

    if include_option_majority:
        trend_accuracy = option_majority_trend_accuracy(train_dataset, eval_dataset)
        rows.append(
            OptionSeedReplicationRow(
                seed=seed,
                model_control="option_majority",
                intervention="original",
                samples=int(eval_dataset.features.shape[0]),
                trend_accuracy=trend_accuracy,
                trend_drop_from_original=0.0,
                need_mse=nan,
                dominant_accuracy=nan,
                severity_accuracy=nan,
                low_flag_accuracy=nan,
                exact_discrete_accuracy=nan,
                world_final_need_mse=world_final_need_mse,
                world_trend_accuracy=world_trend_accuracy,
                trained_world_final_need_mse_before=before_world_final_need_mse,
                trained_world_final_need_mse_after=after_world_final_need_mse,
                trained_world_trend_before=before_world_trend_accuracy,
                trained_world_trend_after=after_world_trend_accuracy,
            )
        )
    return rows


def _format_float(value: float, decimals: int) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.{decimals}f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[9901, 9902, 9903])
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
        choices=OPTION_BALANCE_TARGETS,
        default="option_trend",
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=OPTION_FEATURE_INTERVENTIONS,
        default=["original", "shuffle_delta", "negate_delta"],
    )
    parser.add_argument("--random-model-control", action="store_true")

    parser.add_argument("--world-train-episodes", type=int, default=1000)
    parser.add_argument("--world-eval-episodes", type=int, default=500)
    parser.add_argument("--max-world-train-samples", type=int, default=14000)
    parser.add_argument("--max-world-eval-samples", type=int, default=7000)
    parser.add_argument("--world-epochs", type=int, default=8)
    parser.add_argument("--world-batch-size", type=int, default=128)
    parser.add_argument("--world-eval-batch-size", type=int, default=128)
    parser.add_argument("--world-learning-rate", type=float, default=4e-4)
    parser.add_argument("--step-needs-weight", type=float, default=2.0)
    parser.add_argument("--final-needs-weight", type=float, default=10.0)
    parser.add_argument("--observation-prediction-weight", type=float, default=0.03)
    parser.add_argument("--reward-prediction-weight", type=float, default=0.4)

    parser.add_argument("--language-train-episodes", type=int, default=1000)
    parser.add_argument("--language-eval-episodes", type=int, default=500)
    parser.add_argument("--max-language-train-states", type=int, default=14000)
    parser.add_argument("--max-language-eval-states", type=int, default=7000)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--language-epochs", type=int, default=120)
    parser.add_argument("--language-batch-size", type=int, default=128)
    parser.add_argument("--language-learning-rate", type=float, default=1e-3)
    parser.add_argument("--agreement-weight", type=float, default=0.05)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--need-weight", type=float, default=2.0)
    parser.add_argument("--low-weight", type=float, default=1.0)
    parser.add_argument("--dominant-weight", type=float, default=1.0)
    parser.add_argument("--severity-weight", type=float, default=0.5)
    parser.add_argument("--trend-weight", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
