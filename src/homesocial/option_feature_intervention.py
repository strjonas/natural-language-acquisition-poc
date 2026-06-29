from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

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
from .recurrent_ac import RecurrentActorCritic
from .self_state_language import (
    SelfStateCommunicationResult,
    SelfStateDataset,
    evaluate_self_state_communication,
    train_self_state_communication,
)


OPTION_FEATURE_INTERVENTIONS = (
    "original",
    "zero_current",
    "shuffle_current",
    "zero_future",
    "shuffle_future",
    "zero_delta",
    "shuffle_delta",
    "negate_delta",
)


@dataclass(frozen=True)
class OptionFeatureInterventionResult:
    model_control: str
    intervention: str
    samples: int
    trend_accuracy: float
    need_mse: float
    dominant_accuracy: float
    severity_accuracy: float
    low_flag_accuracy: float
    exact_discrete_accuracy: float


def intervene_option_features(
    dataset: SelfStateDataset,
    *,
    intervention: str,
    seed: int = 1,
) -> SelfStateDataset:
    if intervention not in OPTION_FEATURE_INTERVENTIONS:
        raise ValueError(f"Unknown option feature intervention: {intervention}.")
    features = np.asarray(dataset.features).copy()
    if features.shape[1] != 12:
        raise ValueError("Option feature interventions require latent_current width 12.")
    rng = np.random.default_rng(seed)
    blocks = {
        "current": slice(0, 4),
        "future": slice(4, 8),
        "delta": slice(8, 12),
    }
    if intervention.startswith("zero_"):
        features[:, blocks[intervention.removeprefix("zero_")]] = 0.0
    elif intervention.startswith("shuffle_"):
        block = blocks[intervention.removeprefix("shuffle_")]
        shuffled = features[:, block].copy()
        rng.shuffle(shuffled)
        features[:, block] = shuffled
    elif intervention == "negate_delta":
        features[:, blocks["delta"]] *= -1.0
    return SelfStateDataset(
        features=mx.array(features, dtype=mx.float32),
        needs=dataset.needs,
        low_flags=dataset.low_flags,
        dominant_labels=dataset.dominant_labels,
        severity_labels=dataset.severity_labels,
        trend_labels=dataset.trend_labels,
        action_labels=dataset.action_labels,
    )


def summarize_intervention(
    result: SelfStateCommunicationResult,
    *,
    intervention: str,
) -> OptionFeatureInterventionResult:
    return OptionFeatureInterventionResult(
        model_control=result.model_control,
        intervention=intervention,
        samples=result.samples_per_sender,
        trend_accuracy=result.trend_accuracy,
        need_mse=result.need_mse,
        dominant_accuracy=result.dominant_accuracy,
        severity_accuracy=result.severity_accuracy,
        low_flag_accuracy=result.low_flag_accuracy,
        exact_discrete_accuracy=result.exact_discrete_accuracy,
    )


def _row(result: OptionFeatureInterventionResult) -> str:
    return ",".join(
        [
            result.model_control,
            result.intervention,
            str(result.samples),
            f"{result.trend_accuracy:.4f}",
            f"{result.need_mse:.6f}",
            f"{result.dominant_accuracy:.4f}",
            f"{result.severity_accuracy:.4f}",
            f"{result.low_flag_accuracy:.4f}",
            f"{result.exact_discrete_accuracy:.4f}",
        ]
    )


def main() -> None:
    args = _parse_args()
    trained_base, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)

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
        "model_control,intervention,samples,trend_accuracy,need_mse,"
        "dominant_accuracy,severity_accuracy,low_flag_accuracy,"
        "exact_discrete_accuracy"
    )
    for model_control, base_model in base_models:
        train_dataset = collect_option_counterfactual_dataset(
            base_model,
            config,
            episodes=args.train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            balance_target=args.balance_target,
            rollout_mode="latent_current",
            state_policy=args.state_policy,
            max_states=args.max_train_states,
        )
        trained = train_self_state_communication(
            train_dataset,
            population_size=args.population_size,
            hidden_size=args.hidden_size,
            receiver_size=args.receiver_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            agreement_weight=args.agreement_weight,
            balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight,
            need_weight=args.need_weight,
            low_weight=args.low_weight,
            dominant_weight=args.dominant_weight,
            severity_weight=args.severity_weight,
            trend_weight=args.trend_weight,
            seed=args.seed,
        )
        eval_dataset = collect_option_counterfactual_dataset(
            base_model,
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            balance_target=args.balance_target,
            rollout_mode="latent_current",
            state_policy=args.state_policy,
            max_states=args.max_eval_states,
        )
        for intervention in args.interventions:
            intervened = intervene_option_features(
                eval_dataset,
                intervention=intervention,
                seed=args.seed + 20_000,
            )
            result = evaluate_self_state_communication(
                trained,
                intervened,
                model_control=model_control,
                feature_mode="option_counterfactual_latent_current",
                history_mode=intervention,
            )
            print(_row(summarize_intervention(result, intervention=intervention)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=9951)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--max-train-states", type=int, default=14000)
    parser.add_argument("--max-eval-states", type=int, default=7000)
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
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--agreement-weight", type=float, default=0.05)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--need-weight", type=float, default=2.0)
    parser.add_argument("--low-weight", type=float, default=1.0)
    parser.add_argument("--dominant-weight", type=float, default=1.0)
    parser.add_argument("--severity-weight", type=float, default=0.5)
    parser.add_argument("--trend-weight", type=float, default=2.0)
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
        default=list(OPTION_FEATURE_INTERVENTIONS),
    )
    parser.add_argument("--random-model-control", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
