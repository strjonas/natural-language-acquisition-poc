"""Multi-seed replication and anchor controls for the causal self-report result.

Two things the sealed probe57 battery could not do.

First, replication. The causal parameters initialize deterministically, so the
harness's ``mx.random.seed`` never perturbed this stage; only the developmental
world stream, ``seed_base``, can. This module varies it explicitly and reports
spread across seeds.

Second, the birth-anchor control. Belief is initialized from the true body at
birth and thereafter integrated from public observation alone. This measures
how much of the result that one privileged reading is actually carrying, by
degrading or removing it.

Run:

    PYTHONPATH=src python3 -m homesocial.organism.causal_self_replication \\
        --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz \\
        --run-dir runs/organism/probe58_causal_stage_replication
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import heard_need
from homesocial.island.world import SURFACES
from homesocial.organism.causal_self import (
    _apply_causal_transition,
    _public_transition_features,
    audit_structured_causal_self_model,
    causal_social_token,
    evaluate_causal_social_planner,
    train_structured_causal_self_model,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]

# Matches the harness stream for --seed 1 so seed index 0 reproduces probe57.
BASE_SEED_BASE = 6_100_000
# Development consumes seed_base .. seed_base + lives, and a single 80,000-tick
# run lives roughly 2,300 lives. The stride must clear the evaluation bands
# entirely -- the promotion gate uses 6,500,000 (forks at 6,700,000) and the
# planner battery 7,100,000 through 7,500,000 -- or a developmental stream
# would train on the very worlds it is later scored against.
SEED_STRIDE = 10_000_000
EVALUATION_SEED_BANDS = ((6_500_000, 7_600_000),)
DEVELOPMENT_SEED_HEADROOM = 100_000


def causal_replication_seed_base(seed: int) -> int:
    """Developmental world stream for a given ``--seed``, seed 1 being probe57."""

    seed_base = BASE_SEED_BASE + (seed - 1) * SEED_STRIDE
    _check_seed_isolation(seed_base)
    return seed_base


def _check_seed_isolation(seed_base: int) -> None:
    low = seed_base
    high = seed_base + DEVELOPMENT_SEED_HEADROOM
    for band_low, band_high in EVALUATION_SEED_BANDS:
        if low <= band_high and band_low <= high:
            raise ValueError(
                f"Developmental seed_base {seed_base} overlaps the evaluation "
                f"band {band_low}-{band_high}; the replication would train on "
                "its own held-out worlds."
            )

BIRTH_MODES = ("true", "noisy_05", "noisy_10", "noisy_20", "random", "prior_mean")


def _birth_belief(
    mode: str,
    true_birth: np.ndarray,
    config: OrganismConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Degrade the one privileged interoceptive reading the organism receives."""

    if mode == "true":
        return true_birth.copy()
    if mode == "prior_mean":
        return np.full(3, float(np.mean(config.report.birth_levels)))
    if mode == "random":
        return rng.choice(np.asarray(config.report.birth_levels, dtype=np.float64), 3)
    if mode.startswith("noisy_"):
        sigma = int(mode.split("_")[1]) / 100.0
        return np.clip(true_birth + rng.normal(0.0, sigma, 3), 0.0, 1.0)
    raise ValueError(f"Unknown birth belief mode: {mode}.")


def audit_birth_anchor_dependence(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    mode: str,
) -> dict[str, float]:
    """Deploy the learned planner with a degraded birth reading.

    Everything after birth is unchanged: the belief still advances only through
    the learned causal transition applied to public transition features.
    """

    survived = 0
    said = 0
    truthful = 0
    error_sum = 0.0
    error_samples = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        true_birth = np.asarray(packet.vector()[:3], dtype=np.float64)
        belief = _birth_belief(
            mode, true_birth, config, np.random.default_rng(seed + 991)
        )
        motor_hidden = None
        motor_rng = Random(seed + 59_000_003)
        while True:
            token = causal_social_token(
                model,
                belief,
                step_count=packet.step_count,
                help_period=config.report.help_period,
            )
            word = heard_need((token, PAD_ID))
            if packet.step_count >= FIDELITY_WARMUP:
                if word is not None:
                    said += 1
                    truthful += int(word == world.lowest_need())
                error_sum += float(
                    np.abs(belief - np.asarray(packet.needs[:3])).mean()
                )
                error_samples += 1
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear((token, PAD_ID))
            before = packet
            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                action,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            belief = _apply_causal_transition(
                model,
                belief,
                _public_transition_features(config, before, action, packet),
            )
            if terminated or truncated:
                survived += int(not terminated)
                break
    return {
        "birth_mode": mode,
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
        "mean_absolute_need_error": error_sum / max(1, error_samples),
    }


def replicate_causal_stage(
    parent: str,
    *,
    seeds: int,
    lives: int,
    steps: int,
) -> list[dict[str, float]]:
    """Retrain only the causal stage on independent developmental streams."""

    rows: list[dict[str, float]] = []
    for index in range(seeds):
        seed_base = BASE_SEED_BASE + index * SEED_STRIDE
        _check_seed_isolation(seed_base)
        model, config = load_organism_checkpoint(parent)
        if model.has_structured_causal_self_model:
            raise ValueError("The parent checkpoint already has a causal self-model.")
        model.enable_structured_causal_self_model(len(SURFACES))
        config, development = train_structured_causal_self_model(
            model, config, steps=steps, learning_rate=3e-3,
            seed_base=seed_base, log_every_lives=0,
        )
        gate = audit_structured_causal_self_model(
            model, config, lives=lives, seed_base=6_500_000
        )
        grounded = evaluate_causal_social_planner(
            model, config, lives=lives, seed_base=7_100_000
        )
        scrambled = evaluate_causal_social_planner(
            model, config, lives=lives, seed_base=7_100_000,
            listener_mode="scrambled",
        )
        zero = evaluate_causal_social_planner(
            model, config, lives=lives, seed_base=7_100_000,
            belief_intervention="zero",
        )
        row = {
            "seed_index": float(index),
            "seed_base": float(seed_base),
            "balanced_accuracy": gate["balanced_accuracy"],
            "mean_absolute_need_error": gate["mean_absolute_need_error"],
            "listener_food_correct": gate["listener_food_correct"],
            "listener_water_correct": gate["listener_water_correct"],
            "listener_energy_correct": gate["listener_energy_correct"],
            "listener_no_help_accuracy": gate["listener_no_help_accuracy"],
            "fork_following": gate["fork_following"],
            "report_followed_body_rate": gate["report_followed_body_rate"],
            "promotion_gate_passed": gate["gate_passed"],
            "grounded_survival": float(grounded["survival"]),
            "grounded_report_fidelity": float(grounded["report_fidelity"]),
            "scrambled_survival": float(scrambled["survival"]),
            "zero_belief_survival": float(zero["survival"]),
            "base_parameters_unchanged": development["base_parameters_unchanged"],
        }
        rows.append(row)
        print(f"seed {index} (seed_base {seed_base}): " + json.dumps(row))
        if index == 0:
            for mode in BIRTH_MODES:
                anchor = audit_birth_anchor_dependence(
                    model, config, lives=lives, seed_base=7_100_000, mode=mode
                )
                print("  birth anchor: " + json.dumps(anchor))
                rows.append({"seed_index": float(index), **anchor})
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    seed_rows = [row for row in rows if "birth_mode" not in row]
    keys = [key for key in seed_rows[0] if key not in {"seed_index", "seed_base"}]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in seed_rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lives", type=int, default=100)
    parser.add_argument("--steps", type=int, default=80_000)
    args = parser.parse_args()

    rows = replicate_causal_stage(
        args.parent, seeds=args.seeds, lives=args.lives, steps=args.steps
    )
    summary = summarize(rows)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "replication_seeds.json").write_text(json.dumps(rows, indent=2) + "\n")
    (run_dir / "replication_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (run_dir / "replication_seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== replication summary ===")
    for key, stats in summary.items():
        print(
            f"  {key:<28} mean={stats['mean']:.4f} sd={stats['sd']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )


if __name__ == "__main__":
    main()
