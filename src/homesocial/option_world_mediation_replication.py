from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import mlx.core as mx

from .env import RESOURCE_ECOLOGIES, Action
from .imitation import load_checkpoint
from .observations import observation_vector_size
from .option_counterfactual_language import STATE_POLICIES
from .option_feature_intervention import OPTION_FEATURE_INTERVENTIONS
from .option_mediation import (
    OPTION_MEDIATION_BALANCE_TARGETS,
    OPTION_MEDIATION_FEATURE_MODES,
    collect_option_mediation_source,
    evaluate_option_mediator,
    intervene_option_mediation_features,
    majority_option_result,
    option_mediation_dataset_from_source,
    train_option_mediator,
)
from .option_world_model import (
    collect_option_branch_dataset,
    evaluate_option_world_model,
    train_option_world_model,
)
from .recurrent_ac import RecurrentActorCritic


@dataclass(frozen=True)
class OptionWorldMediationReplicationRow:
    seed: int
    model_control: str
    intervention: str
    samples: int
    choice_accuracy: float
    mean_regret: float
    mean_chosen_delta: float
    mean_oracle_delta: float
    world_final_need_mse: float
    world_trend_accuracy: float
    trained_world_final_need_mse_before: float
    trained_world_final_need_mse_after: float
    trained_world_trend_before: float
    trained_world_trend_after: float


def header() -> str:
    return (
        "seed,model_control,intervention,samples,choice_accuracy,mean_regret,"
        "mean_chosen_delta,mean_oracle_delta,world_final_need_mse,"
        "world_trend_accuracy,trained_world_final_need_mse_before,"
        "trained_world_final_need_mse_after,trained_world_trend_before,"
        "trained_world_trend_after"
    )


def format_row(row: OptionWorldMediationReplicationRow) -> str:
    return ",".join(
        [
            str(row.seed),
            row.model_control,
            row.intervention,
            str(row.samples),
            f"{row.choice_accuracy:.4f}",
            f"{row.mean_regret:.6f}",
            f"{row.mean_chosen_delta:.6f}",
            f"{row.mean_oracle_delta:.6f}",
            f"{row.world_final_need_mse:.6f}",
            f"{row.world_trend_accuracy:.4f}",
            f"{row.trained_world_final_need_mse_before:.6f}",
            f"{row.trained_world_final_need_mse_after:.6f}",
            f"{row.trained_world_trend_before:.4f}",
            f"{row.trained_world_trend_after:.4f}",
        ]
    )


def resolved_option_action_noises(args: argparse.Namespace) -> tuple[float, float]:
    shared_noise = float(args.option_action_noise)
    world_noise = getattr(args, "world_option_action_noise", None)
    mediation_noise = getattr(args, "mediation_option_action_noise", None)
    return (
        shared_noise if world_noise is None else float(world_noise),
        shared_noise if mediation_noise is None else float(mediation_noise),
    )


def main() -> None:
    args = _parse_args()
    print(header())
    for seed in args.seeds:
        for row in run_replication_seed(args, seed=seed):
            print(format_row(row))


def run_replication_seed(
    args: argparse.Namespace,
    *,
    seed: int,
) -> list[OptionWorldMediationReplicationRow]:
    trained_base, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)
    world_noise, mediation_noise = resolved_option_action_noises(args)

    train_branches = collect_option_branch_dataset(
        config,
        episodes=args.world_train_episodes,
        seed=seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_world_train_samples,
        option_action_noise=world_noise,
    )
    eval_branches = collect_option_branch_dataset(
        config,
        episodes=args.world_eval_episodes,
        seed=seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_world_eval_samples,
        option_action_noise=world_noise,
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

    train_source = collect_option_mediation_source(
        config,
        episodes=args.mediation_train_episodes,
        seed=seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.mediation_balance_target,
        max_states=args.max_mediation_train_states,
        min_value_gap=args.min_value_gap,
        option_action_noise=mediation_noise,
    )
    eval_source = collect_option_mediation_source(
        config,
        episodes=args.mediation_eval_episodes,
        seed=seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.mediation_balance_target,
        max_states=args.max_mediation_eval_states,
        min_value_gap=args.min_value_gap,
        option_action_noise=mediation_noise,
    )

    rows = _mediation_rows(
        trained_base,
        train_source,
        eval_source,
        seed=seed,
        model_control="trained",
        args=args,
        world_final_need_mse=after_world.final_need_mse,
        world_trend_accuracy=after_world.trend_accuracy,
        before_world_final_need_mse=before_world.final_need_mse,
        after_world_final_need_mse=after_world.final_need_mse,
        before_world_trend_accuracy=before_world.trend_accuracy,
        after_world_trend_accuracy=after_world.trend_accuracy,
        include_target_majority=True,
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
            _mediation_rows(
                random_base,
                train_source,
                eval_source,
                seed=seed,
                model_control="random",
                args=args,
                world_final_need_mse=random_world.final_need_mse,
                world_trend_accuracy=random_world.trend_accuracy,
                before_world_final_need_mse=before_world.final_need_mse,
                after_world_final_need_mse=after_world.final_need_mse,
                before_world_trend_accuracy=before_world.trend_accuracy,
                after_world_trend_accuracy=after_world.trend_accuracy,
                include_target_majority=False,
            )
        )
    return rows


def _mediation_rows(
    base_model: RecurrentActorCritic,
    train_source,
    eval_source,
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
    include_target_majority: bool,
) -> list[OptionWorldMediationReplicationRow]:
    train_dataset = option_mediation_dataset_from_source(
        train_source,
        base_model,
        feature_mode=args.feature_mode,
    )
    eval_dataset = option_mediation_dataset_from_source(
        eval_source,
        base_model,
        feature_mode=args.feature_mode,
    )
    trained = train_option_mediator(
        train_dataset,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        epochs=args.mediation_epochs,
        batch_size=args.mediation_batch_size,
        learning_rate=args.mediation_learning_rate,
        balance_weight=args.mediation_balance_weight,
        entropy_weight=args.mediation_entropy_weight,
        seed=seed,
    )

    rows: list[OptionWorldMediationReplicationRow] = []
    for intervention in args.interventions:
        intervened = intervene_option_mediation_features(
            eval_dataset,
            intervention=intervention,
            seed=seed + 20_000,
        )
        result = evaluate_option_mediator(
            trained,
            intervened,
            model_control=model_control,
            intervention=intervention,
        )
        rows.append(
            _result_row(
                seed,
                result,
                world_final_need_mse=world_final_need_mse,
                world_trend_accuracy=world_trend_accuracy,
                before_world_final_need_mse=before_world_final_need_mse,
                after_world_final_need_mse=after_world_final_need_mse,
                before_world_trend_accuracy=before_world_trend_accuracy,
                after_world_trend_accuracy=after_world_trend_accuracy,
            )
        )

    if include_target_majority:
        majority = majority_option_result(train_dataset, eval_dataset)
        rows.append(
            _result_row(
                seed,
                majority,
                world_final_need_mse=world_final_need_mse,
                world_trend_accuracy=world_trend_accuracy,
                before_world_final_need_mse=before_world_final_need_mse,
                after_world_final_need_mse=after_world_final_need_mse,
                before_world_trend_accuracy=before_world_trend_accuracy,
                after_world_trend_accuracy=after_world_trend_accuracy,
            )
        )
    return rows


def _result_row(
    seed: int,
    result,
    *,
    world_final_need_mse: float,
    world_trend_accuracy: float,
    before_world_final_need_mse: float,
    after_world_final_need_mse: float,
    before_world_trend_accuracy: float,
    after_world_trend_accuracy: float,
) -> OptionWorldMediationReplicationRow:
    return OptionWorldMediationReplicationRow(
        seed=seed,
        model_control=result.model_control,
        intervention=result.intervention,
        samples=result.samples,
        choice_accuracy=result.choice_accuracy,
        mean_regret=result.mean_regret,
        mean_chosen_delta=result.mean_chosen_delta,
        mean_oracle_delta=result.mean_oracle_delta,
        world_final_need_mse=world_final_need_mse,
        world_trend_accuracy=world_trend_accuracy,
        trained_world_final_need_mse_before=before_world_final_need_mse,
        trained_world_final_need_mse_after=after_world_final_need_mse,
        trained_world_trend_before=before_world_trend_accuracy,
        trained_world_trend_after=after_world_trend_accuracy,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[9901, 9902])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--option-action-noise",
        type=float,
        default=0.0,
        help="Probability of replacing a scripted option step with another valid body action.",
    )
    parser.add_argument(
        "--world-option-action-noise",
        type=float,
        default=None,
        help="Override option action noise for option-world branch training/eval.",
    )
    parser.add_argument(
        "--mediation-option-action-noise",
        type=float,
        default=None,
        help="Override option action noise for mediation source collection.",
    )
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
        "--mediation-balance-target",
        choices=OPTION_MEDIATION_BALANCE_TARGETS,
        default="target_option",
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

    parser.add_argument("--mediation-train-episodes", type=int, default=1000)
    parser.add_argument("--mediation-eval-episodes", type=int, default=500)
    parser.add_argument("--max-mediation-train-states", type=int, default=12000)
    parser.add_argument("--max-mediation-eval-states", type=int, default=6000)
    parser.add_argument("--min-value-gap", type=float, default=0.005)
    parser.add_argument(
        "--feature-mode",
        choices=OPTION_MEDIATION_FEATURE_MODES,
        default="latent_current",
    )
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--mediation-epochs", type=int, default=100)
    parser.add_argument("--mediation-batch-size", type=int, default=128)
    parser.add_argument("--mediation-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mediation-balance-weight", type=float, default=0.02)
    parser.add_argument("--mediation-entropy-weight", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
