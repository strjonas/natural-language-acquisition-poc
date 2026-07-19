"""The one benchmark command (gate G1 and its controls).

Trains a fresh organism under each language mode, evaluates on held-out
fixed seeds, and prints a table alongside scripted baselines. All headline
numbers come from here.

    PYTHONPATH=src python3 -m homesocial.organism.harness \
        --train-steps 200000 --language-modes grounded silent shuffled
"""

from __future__ import annotations

import argparse
from pathlib import Path

from homesocial.island.calibrate import run_policy
from homesocial.island.world import IslandConfig
from homesocial.organism.train import (
    OrganismConfig,
    audit_label_to_self_model,
    audit_self_model_actions,
    evaluate_organism,
    train_organism,
)

EVAL_SEED_BASE = 900_000


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, object]] = []

    for baseline in args.baselines:
        stats = run_policy(
            baseline,
            episodes=args.eval_episodes,
            config=IslandConfig(max_steps=args.eval_max_steps),
            base_seed=EVAL_SEED_BASE,
        )
        rows.append({"condition": f"baseline_{baseline}", **stats})

    for language_mode in args.language_modes:
        model_horizon = args.model_horizon or args.planning_horizon
        run_label = language_mode
        if args.bc_warmstart:
            run_label += f"_bc{args.bc_lives}"
        if args.offer_childhood:
            run_label += (
                f"_offer{args.offer_fade_steps}d{args.offer_distance_end}"
            )
        if args.consume_options:
            run_label += "_options"
        if args.inspect_options:
            run_label += "_inspect"
        if args.self_model_planning:
            run_label += (
                f"_plan{args.planning_scale:g}h{args.planning_horizon}"
            )
        if args.replay_updates > 0:
            run_label += f"_replay{args.replay_capacity}x{args.replay_updates}"
        if model_horizon != args.planning_horizon:
            run_label += f"_modelh{model_horizon}"
        config = OrganismConfig(
            language_mode=language_mode,
            total_steps=args.train_steps,
            segment_length=args.segment_length,
            hidden_size=args.hidden_size,
            bc_warmstart_lives=args.bc_lives if args.bc_warmstart else 0,
            bc_epochs=args.bc_epochs,
            caregiver_offer_threshold_start=(
                args.offer_threshold if args.offer_childhood else 0.0
            ),
            caregiver_offer_curriculum_steps=(
                args.offer_fade_steps if args.offer_childhood else 0
            ),
            caregiver_offer_distance_end=(
                args.offer_distance_end if args.offer_childhood else 0
            ),
            consume_options=args.consume_options,
            inspect_options=args.inspect_options,
            self_model_planning_scale=(
                args.planning_scale if args.self_model_planning else 0.0
            ),
            self_model_planning_start_steps=args.planning_start_steps,
            self_model_planning_reward_weight=args.planning_reward_weight,
            self_model_planning_horizon=args.planning_horizon,
            multi_step_model_horizon=model_horizon,
            multi_step_model_weight=args.multi_step_model_weight,
            world_model_replay_capacity=args.replay_capacity,
            world_model_replay_updates=args.replay_updates,
            seed=args.seed,
            max_steps=args.train_max_steps,
            checkpoint=str(
                Path(args.run_dir) / f"organism_{run_label}_seed{args.seed}.npz"
            ),
            stats_csv=str(
                Path(args.run_dir) / f"lives_{run_label}_seed{args.seed}.csv"
            ),
        )
        print(f"=== training organism under {language_mode} ===")
        model, _ = train_organism(config)
        stats = evaluate_organism(
            model,
            language_mode=language_mode,
            episodes=args.eval_episodes,
            base_seed=EVAL_SEED_BASE,
            max_steps=args.eval_max_steps,
            greedy=args.eval_greedy,
            consume_options=args.consume_options,
            inspect_options=args.inspect_options,
            self_model_planning_scale=(
                args.planning_scale if args.self_model_planning else 0.0
            ),
            self_model_planning_reward_weight=args.planning_reward_weight,
            self_model_planning_horizon=args.planning_horizon,
        )
        if args.self_model_audit_decisions > 0:
            audit = audit_self_model_actions(
                model,
                language_mode=language_mode,
                episodes=args.eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_decisions=args.self_model_audit_decisions,
                max_steps=args.eval_max_steps,
                consume_options=args.consume_options,
                inspect_options=args.inspect_options,
                self_model_planning_scale=(
                    args.planning_scale if args.self_model_planning else 0.0
                ),
                reward_weight=args.planning_reward_weight,
                self_model_planning_horizon=args.planning_horizon,
            )
            stats.update({f"self_model_{key}": value for key, value in audit.items()})
        if args.label_self_model_audit_inspections > 0:
            label_audit = audit_label_to_self_model(
                model,
                episodes=args.eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_inspections=args.label_self_model_audit_inspections,
                max_steps=args.eval_max_steps,
                self_model_planning_scale=(
                    args.planning_scale if args.self_model_planning else 0.0
                ),
                self_model_planning_horizon=args.planning_horizon,
            )
            stats.update(
                {f"label_audit_{key}": value for key, value in label_audit.items()}
            )
        rows.append({"condition": f"organism_{run_label}", **stats})
        if args.self_model_interventions and args.self_model_planning:
            for intervention, scale, score_sign in (
                ("planning_removed", 0.0, 1.0),
                ("planning_reversed", args.planning_scale, -1.0),
            ):
                intervention_stats = evaluate_organism(
                    model,
                    language_mode=language_mode,
                    episodes=args.eval_episodes,
                    base_seed=EVAL_SEED_BASE,
                    max_steps=args.eval_max_steps,
                    greedy=args.eval_greedy,
                    consume_options=args.consume_options,
                    inspect_options=args.inspect_options,
                    self_model_planning_scale=scale,
                    self_model_planning_reward_weight=args.planning_reward_weight,
                    self_model_planning_score_sign=score_sign,
                    self_model_planning_horizon=args.planning_horizon,
                )
                rows.append(
                    {
                        "condition": f"organism_{run_label}_{intervention}",
                        **intervention_stats,
                    }
                )
        if args.body_only_self_model_audit and args.self_model_planning:
            body_stats = evaluate_organism(
                model,
                language_mode=language_mode,
                episodes=args.eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_steps=args.eval_max_steps,
                greedy=args.eval_greedy,
                consume_options=args.consume_options,
                inspect_options=args.inspect_options,
                self_model_planning_scale=args.planning_scale,
                self_model_planning_reward_weight=0.0,
                self_model_planning_horizon=args.planning_horizon,
            )
            if args.self_model_audit_decisions > 0:
                body_audit = audit_self_model_actions(
                    model,
                    language_mode=language_mode,
                    episodes=args.eval_episodes,
                    base_seed=EVAL_SEED_BASE,
                    max_decisions=args.self_model_audit_decisions,
                    max_steps=args.eval_max_steps,
                    consume_options=args.consume_options,
                    inspect_options=args.inspect_options,
                    self_model_planning_scale=args.planning_scale,
                    reward_weight=0.0,
                    self_model_planning_horizon=args.planning_horizon,
                )
                body_stats.update(
                    {f"self_model_{key}": value for key, value in body_audit.items()}
                )
            rows.append(
                {
                    "condition": f"organism_{run_label}_body_only",
                    **body_stats,
                }
            )
            if args.self_model_interventions:
                body_reversed = evaluate_organism(
                    model,
                    language_mode=language_mode,
                    episodes=args.eval_episodes,
                    base_seed=EVAL_SEED_BASE,
                    max_steps=args.eval_max_steps,
                    greedy=args.eval_greedy,
                    consume_options=args.consume_options,
                    inspect_options=args.inspect_options,
                    self_model_planning_scale=args.planning_scale,
                    self_model_planning_reward_weight=0.0,
                    self_model_planning_score_sign=-1.0,
                    self_model_planning_horizon=args.planning_horizon,
                )
                rows.append(
                    {
                        "condition": f"organism_{run_label}_body_only_reversed",
                        **body_reversed,
                    }
                )

    _write_and_print(rows, args)


def _write_and_print(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    keys = ["condition"]
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    suffix = f"_bc{args.bc_lives}" if args.bc_warmstart else ""
    if args.offer_childhood:
        suffix += f"_offer{args.offer_fade_steps}d{args.offer_distance_end}"
    if args.consume_options:
        suffix += "_options"
    if args.inspect_options:
        suffix += "_inspect"
    if args.self_model_planning:
        suffix += f"_plan{args.planning_scale:g}h{args.planning_horizon}"
    if args.replay_updates > 0:
        suffix += f"_replay{args.replay_capacity}x{args.replay_updates}"
    model_horizon = args.model_horizon or args.planning_horizon
    if model_horizon != args.planning_horizon:
        suffix += f"_modelh{model_horizon}"
    languages = "-".join(args.language_modes)
    out_path = (
        Path(args.run_dir)
        / f"harness_seed{args.seed}_{languages}{suffix}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    f"{row.get(key):.4f}"
                    if isinstance(row.get(key), float)
                    else str(row.get(key, ""))
                    for key in keys
                )
                + "\n"
            )
    print(f"\nresults ({out_path}):")
    for row in rows:
        formatted = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items()
        )
        print("  " + formatted)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--bc-warmstart",
        action="store_true",
        help="Pretrain on token-masked oracle lives before online learning.",
    )
    parser.add_argument("--bc-lives", type=int, default=100)
    parser.add_argument("--bc-epochs", type=int, default=2)
    parser.add_argument(
        "--offer-childhood",
        action="store_true",
        help="Fade in-loop caregiver food/water offers to zero during training.",
    )
    parser.add_argument("--offer-threshold", type=float, default=0.75)
    parser.add_argument("--offer-fade-steps", type=int, default=100_000)
    parser.add_argument("--offer-distance-end", type=int, default=0)
    parser.add_argument(
        "--consume-options",
        action="store_true",
        help="Add kind-blind visible-slot navigate-and-consume actions.",
    )
    parser.add_argument(
        "--inspect-options",
        action="store_true",
        help="Add kind-blind visible-slot navigate-face-and-ask actions.",
    )
    parser.add_argument(
        "--self-model-planning",
        action="store_true",
        help="Bias action logits with detached predicted future-body value.",
    )
    parser.add_argument("--planning-start-steps", type=int, default=100_000)
    parser.add_argument("--planning-scale", type=float, default=6.0)
    parser.add_argument("--planning-reward-weight", type=float, default=0.5)
    parser.add_argument("--planning-horizon", type=int, choices=[1, 2], default=1)
    parser.add_argument("--model-horizon", type=int, choices=[1, 2], default=None)
    parser.add_argument("--multi-step-model-weight", type=float, default=0.0)
    parser.add_argument("--replay-capacity", type=int, default=0)
    parser.add_argument("--replay-updates", type=int, default=0)
    parser.add_argument(
        "--self-model-audit-decisions",
        type=int,
        default=0,
        help=(
            "For this many held-out states, branch simulator copies to score "
            "all available actions against the model's bodily predictions."
        ),
    )
    parser.add_argument(
        "--label-self-model-audit-inspections",
        type=int,
        default=0,
        help=(
            "Audit this many held-out inspect events under true, silent, and "
            "counterfactual kind labels. Requires consume and inspect options."
        ),
    )
    parser.add_argument(
        "--self-model-interventions",
        action="store_true",
        help="Also evaluate the trained checkpoint with planning removed/reversed.",
    )
    parser.add_argument(
        "--body-only-self-model-audit",
        action="store_true",
        help=(
            "Also evaluate/audit planning with predicted external reward weight "
            "zero, isolating the bodily consequence channel."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-max-steps", type=int, default=1000)
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-greedy",
        action="store_true",
        help="Use argmax actions for a deterministic policy diagnostic.",
    )
    parser.add_argument(
        "--language-modes",
        nargs="+",
        default=["grounded", "silent", "shuffled"],
        choices=["grounded", "silent", "shuffled"],
    )
    parser.add_argument(
        "--baselines", nargs="+", default=["oracle", "random"],
        choices=["oracle", "random"],
    )
    parser.add_argument("--run-dir", default="runs/organism")
    return parser.parse_args()


if __name__ == "__main__":
    main()
