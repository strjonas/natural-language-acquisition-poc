from __future__ import annotations

import argparse

from .env import RESOURCE_ECOLOGIES
from .option_counterfactual_language import STATE_POLICIES
from .option_feature_intervention import OPTION_FEATURE_INTERVENTIONS
from .option_mediation import (
    OPTION_MEDIATION_BALANCE_TARGETS,
    OPTION_MEDIATION_FEATURE_MODES,
)
from .option_world_mediation_replication import (
    format_row,
    header,
    run_replication_seed,
)


def body_header() -> str:
    return "body_checkpoint," + header()


def iter_checkpoint_seed_pairs(
    checkpoints: list[str],
    seeds: list[int],
) -> list[tuple[str, int]]:
    if len(seeds) == 1:
        return [(checkpoint, seeds[0]) for checkpoint in checkpoints]
    if len(seeds) != len(checkpoints):
        raise ValueError(
            "Provide either one seed for all checkpoints or one seed per checkpoint."
        )
    return list(zip(checkpoints, seeds, strict=True))


def main() -> None:
    args = _parse_args()
    print(body_header())
    for checkpoint, seed in iter_checkpoint_seed_pairs(args.checkpoints, args.seeds):
        run_args = argparse.Namespace(**{**vars(args), "checkpoint": checkpoint})
        for row in run_replication_seed(run_args, seed=seed):
            print(checkpoint + "," + format_row(row))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[9901])
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
    parser.add_argument("--self-model-rank-control", action="store_true")

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
    parser.add_argument("--rank-finetune-epochs", type=int, default=0)
    parser.add_argument("--rank-finetune-batch-size", type=int, default=128)
    parser.add_argument("--rank-finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank-finetune-temperature", type=float, default=0.05)
    parser.add_argument("--rank-finetune-dynamics-weight", type=float, default=0.0)

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
