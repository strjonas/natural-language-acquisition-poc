"""Online recalibration of the causal self-model after the body's rules change.

Probe57's learned body model is beaten by probe53's hand-coded exact filter, so
nothing in this repository yet shows learning buying anything. This module runs
the experiment that can: change the body's constants after development, and ask
whether a learner that keeps updating overtakes an analytic solution that was
correct when it was written and was never touched again.

See `docs/decisions/2026-07-26-online-adaptation-preregistration.md`. The
adaptation window is interoceptive by declaration; this is recalibration, not
self-supervised adaptation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from random import Random

import mlx.core as mx
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS, heard_need
from homesocial.island.world import SURFACES
from homesocial.organism.causal_self import (
    _apply_causal_transition,
    _public_transition_features,
    causal_social_token,
    train_structured_causal_self_model,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import (
    FIDELITY_WARMUP,
    _ObservableBodyFilter,
    make_report_world,
)
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]

# Locked regime shift. Perceptible only in consequence: nothing announces it.
#
# Calibrated against solvability alone, never against the learner. The report
# ecology sits near its survivable limit by design -- scaling metabolism x1.5
# on its own drops even a perfect body model to 23% survival -- so a shift that
# merely makes the world harder cannot discriminate between belief sources.
# This shift instead scales metabolism and portions together: every constant
# the model holds about its body becomes wrong while the world stays roughly as
# survivable as before. A perfect filter keeps 90% survival here and the
# developed-but-frozen model falls to 53%, so the gap measures stale belief
# rather than an impossible world.
METABOLISM_SCALE = 1.5
MOVE_COST_SCALE = 1.0
SHOCK_SCALE = 1.0
PORTION_SCALE = 1.5

ADAPTATION_TICKS = 40_000
EVAL_SEED_BASE = 7_100_000
ADAPT_SEED_BASE = 82_100_000
DEVELOPED_SEED_BASE = 6_100_000


def shift_report_config(config: OrganismConfig) -> OrganismConfig:
    """Apply the locked post-development change to the body's rules."""

    report = config.report
    return replace(
        config,
        report=replace(
            report,
            food_metabolism=report.food_metabolism * METABOLISM_SCALE,
            water_metabolism=report.water_metabolism * METABOLISM_SCALE,
            energy_metabolism=report.energy_metabolism * METABOLISM_SCALE,
            move_energy_metabolism=report.move_energy_metabolism * MOVE_COST_SCALE,
            shock_size=report.shock_size * SHOCK_SCALE,
            portion_small=report.portion_small * PORTION_SCALE,
            portion_large=report.portion_large * PORTION_SCALE,
        ),
    )


def _learned_constants(model: OrganismModel) -> dict[str, float]:
    drift, move_extra, uptake, shock = model.causal_self_parameters()
    mx.eval(drift, move_extra, uptake, shock)
    drift_np = np.asarray(drift, dtype=np.float64)
    uptake_np = np.asarray(uptake, dtype=np.float64)
    shock_np = np.asarray(shock, dtype=np.float64)
    surface = {name: index for index, name in enumerate(SURFACES)}
    return {
        "food_drift": float(drift_np[0]),
        "water_drift": float(drift_np[1]),
        "energy_drift": float(drift_np[2]),
        "move_extra": float(np.asarray(move_extra)),
        "roots_food_uptake": float(uptake_np[surface["roots"], 0]),
        "spring_water_uptake": float(uptake_np[surface["spring"], 1]),
        "mushroom_energy_uptake": float(uptake_np[surface["mushroom"], 2]),
        "thorn_food_shock": float(shock_np[surface["thorn"], 0]),
        "tree_water_shock": float(shock_np[surface["tree"], 1]),
        "rock_energy_shock": float(shock_np[surface["rock"], 2]),
    }


def _true_constants(config: OrganismConfig) -> dict[str, float]:
    report = config.report
    return {
        "food_drift": -report.food_metabolism,
        "water_drift": -report.water_metabolism,
        "energy_drift": -report.energy_metabolism,
        "move_extra": -report.move_energy_metabolism,
        "roots_food_uptake": report.portion_large,
        "spring_water_uptake": report.portion_large,
        "mushroom_energy_uptake": report.portion_large,
        "thorn_food_shock": -report.shock_size,
        "tree_water_shock": -report.shock_size,
        "rock_energy_shock": -report.shock_size,
    }


def constant_recovery(
    model: OrganismModel, shifted: OrganismConfig
) -> dict[str, float]:
    """How far the learned constants sit from the new truth, per quantity."""

    learned = _learned_constants(model)
    truth = _true_constants(shifted)
    errors = {
        f"error_{name}": abs(learned[name] - truth[name]) for name in truth
    }
    return {
        **{f"learned_{name}": value for name, value in learned.items()},
        **errors,
        "mean_absolute_constant_error": float(np.mean(list(errors.values()))),
    }


def evaluate_belief_source(
    model: OrganismModel,
    world_config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    filter_config: OrganismConfig | None = None,
    planner_model: OrganismModel | None = None,
) -> dict[str, float]:
    """Live the shifted regime, speaking from one belief source.

    ``filter_config is None`` runs the learned causal model. Otherwise the
    analytic probe53 filter drives the belief, using whatever constants
    ``filter_config`` carries -- stale, refit, or true.

    ``planner_model`` selects which model scores tokens, and must be stated
    explicitly for the filter conditions. Belief source and planner are two
    separate things: reusing the adapted model as the planner for an analytic
    baseline lets that baseline inherit the adapted planner's quality, which
    silently couples conditions that are supposed to be independent.
    """

    planner = planner_model if planner_model is not None else model

    survived = 0
    said = 0
    truthful = 0
    error_sum = 0.0
    error_samples = 0
    steps = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(world_config, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        analytic = None
        if filter_config is None:
            belief = np.asarray(packet.vector()[:3], dtype=np.float64)
        else:
            analytic = _ObservableBodyFilter(filter_config)
            analytic.reset(packet)
            belief = analytic.belief
        motor_hidden = None
        motor_rng = Random(seed + 59_000_003)
        while True:
            token = causal_social_token(
                planner,
                belief,
                step_count=packet.step_count,
                help_period=world_config.report.help_period,
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
                planner, packet, motor_hidden, motor_rng
            )
            world.hear((token, PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=world_config.consume_options,
                inspect_options=world_config.inspect_options,
            )
            if analytic is None:
                belief = _apply_causal_transition(
                    model,
                    belief,
                    _public_transition_features(world_config, before, action, packet),
                )
            else:
                analytic.update(before, action, packet)
                belief = analytic.belief
            steps += int(info["duration"])
            if terminated or truncated:
                survived += int(not terminated)
                break
    return {
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
        "mean_absolute_need_error": error_sum / max(1, error_samples),
        "mean_life_steps": steps / lives,
    }


def refit_report_constants(
    model: OrganismModel,
    shifted: OrganismConfig,
    *,
    ticks: int,
    seed_base: int,
) -> OrganismConfig:
    """Least-squares refit of the analytic filter's constants on the same window.

    The correctly specified estimator for a known functional form. Reported as
    an upper bound and context, never as a gate the learner must clear.
    """

    report = shifted.report
    drift_rows: list[tuple[float, float, float]] = []
    drift_targets: list[tuple[float, float, float]] = []
    uptakes: list[tuple[int, float]] = []
    shocks: list[float] = []
    seen = 0
    life = 0
    while seen < ticks:
        seed = seed_base + life
        life += 1
        world = make_report_world(shifted, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        motor_hidden = None
        motor_rng = Random(seed + 71_000_003)
        while seen < ticks:
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            true_need = world.lowest_need()
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[true_need]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=shifted.consume_options,
                inspect_options=shifted.inspect_options,
            )
            duration, moves, uptake, shock = _public_transition_features(
                shifted, before, action, packet
            )
            delta = np.asarray(packet.needs[:3], dtype=np.float64) - np.asarray(
                before.needs[:3], dtype=np.float64
            )
            if not uptake.any() and not shock.any():
                drift_rows.append((duration, moves, 0.0))
                drift_targets.append(tuple(delta))
            elif uptake.any() and not shock.any():
                index = int(np.argmax(uptake))
                gain = float(delta.max())
                if gain > 0.0:
                    uptakes.append((index, gain))
            elif shock.any() and not uptake.any():
                loss = float(delta.min())
                if loss < 0.0:
                    shocks.append(-loss)
            seen += int(info["duration"])
            if terminated or truncated:
                break

    design = np.asarray(
        [[row[0], row[1]] for row in drift_rows], dtype=np.float64
    )
    targets = np.asarray(drift_targets, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    food = max(1e-6, -float(coefficients[0, 0]))
    water = max(1e-6, -float(coefficients[0, 1]))
    energy = max(1e-6, -float(coefficients[0, 2]))
    move_extra = max(energy, -float(coefficients[1, 2]) + energy)
    gains = [gain for _, gain in uptakes]
    large = float(np.percentile(gains, 75)) if gains else report.portion_large
    small = float(np.percentile(gains, 25)) if gains else report.portion_small
    shock_size = float(np.median(shocks)) if shocks else report.shock_size
    return replace(
        shifted,
        report=replace(
            report,
            food_metabolism=food,
            water_metabolism=water,
            energy_metabolism=energy,
            move_energy_metabolism=move_extra,
            portion_large=large,
            portion_small=small,
            shock_size=shock_size,
        ),
    )


def run_online_adaptation(
    parent: str,
    *,
    seeds: int,
    lives: int,
    adaptation_ticks: int = ADAPTATION_TICKS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(seeds):
        adapt_seed = ADAPT_SEED_BASE + index * 10_000_000
        base_model, base_config = load_organism_checkpoint(parent)
        if not base_model.has_structured_causal_self_model:
            raise ValueError("The parent must already carry a causal self-model.")
        shifted = shift_report_config(base_config)

        # frozen: developed parameters, never updated in the new regime
        frozen_model, _ = load_organism_checkpoint(parent)
        frozen = evaluate_belief_source(
            frozen_model, shifted, lives=lives, seed_base=EVAL_SEED_BASE
        )
        frozen_recovery = constant_recovery(frozen_model, shifted)

        # the same frozen model measured before the shift, to prove the shift bites
        unshifted = evaluate_belief_source(
            frozen_model, base_config, lives=lives, seed_base=EVAL_SEED_BASE
        )

        # adapted: continue the developed parameters through the window
        adapted_model, _ = load_organism_checkpoint(parent)
        train_structured_causal_self_model(
            adapted_model, shifted, steps=adaptation_ticks,
            learning_rate=3e-3, seed_base=adapt_seed, log_every_lives=0,
        )
        adapted = evaluate_belief_source(
            adapted_model, shifted, lives=lives, seed_base=EVAL_SEED_BASE
        )
        adapted_recovery = constant_recovery(adapted_model, shifted)

        # reset: same window, but the causal parameters start over
        reset_model, _ = load_organism_checkpoint(parent)
        reset_model.enable_structured_causal_self_model(len(SURFACES))
        reset_model.causal_drift_raw = mx.full((3,), -4.0)
        reset_model.causal_move_extra_raw = mx.array([-4.0])
        reset_model.causal_uptake_raw = mx.full((len(SURFACES), 3), -3.0)
        reset_model.causal_shock_raw = mx.full((len(SURFACES), 3), -3.0)
        reset_model.causal_listener_logits = mx.zeros(
            (reset_model.vocab_size, len(SURFACES) + 1)
        )
        mx.eval(reset_model.parameters())
        train_structured_causal_self_model(
            reset_model, shifted, steps=adaptation_ticks,
            learning_rate=3e-3, seed_base=adapt_seed, log_every_lives=0,
        )
        reset = evaluate_belief_source(
            reset_model, shifted, lives=lives, seed_base=EVAL_SEED_BASE
        )

        # Analytic filters. The planner is stated explicitly: a hand-coded
        # solution deployed once and never updated carries the planner it
        # shipped with, which is the frozen model. Handing these conditions the
        # adapted planner would let them inherit adaptation's benefit and make
        # the comparison meaningless in whichever direction the planner moved.
        stale = evaluate_belief_source(
            frozen_model, shifted, lives=lives, seed_base=EVAL_SEED_BASE,
            filter_config=base_config, planner_model=frozen_model,
        )
        refit_config = refit_report_constants(
            frozen_model, shifted, ticks=adaptation_ticks, seed_base=adapt_seed
        )
        refit = evaluate_belief_source(
            frozen_model, shifted, lives=lives, seed_base=EVAL_SEED_BASE,
            filter_config=refit_config, planner_model=frozen_model,
        )
        oracle = evaluate_belief_source(
            frozen_model, shifted, lives=lives, seed_base=EVAL_SEED_BASE,
            filter_config=shifted, planner_model=frozen_model,
        )
        # Ceiling for the adapted system: perfect belief, adapted planner.
        # Separates "did the belief improve" from "can the planner use it".
        oracle_adapted_planner = evaluate_belief_source(
            adapted_model, shifted, lives=lives, seed_base=EVAL_SEED_BASE,
            filter_config=shifted, planner_model=adapted_model,
        )

        row: dict[str, object] = {
            "seed_index": index,
            "adapt_seed_base": adapt_seed,
            "unshifted_frozen_survival": unshifted["survival"],
            "shift_bites_points": (
                unshifted["survival"] - frozen["survival"]
            ) * 100.0,
            "adapted_survival": adapted["survival"],
            "adapted_fidelity": adapted["report_fidelity"],
            "adapted_body_error": adapted["mean_absolute_need_error"],
            "frozen_survival": frozen["survival"],
            "frozen_fidelity": frozen["report_fidelity"],
            "frozen_body_error": frozen["mean_absolute_need_error"],
            "reset_survival": reset["survival"],
            "reset_body_error": reset["mean_absolute_need_error"],
            "stale_filter_survival": stale["survival"],
            "stale_filter_body_error": stale["mean_absolute_need_error"],
            "refit_filter_survival": refit["survival"],
            "refit_filter_body_error": refit["mean_absolute_need_error"],
            "oracle_filter_survival": oracle["survival"],
            "oracle_filter_body_error": oracle["mean_absolute_need_error"],
            "oracle_adapted_planner_survival": oracle_adapted_planner["survival"],
            "adapted_minus_frozen_points": (
                adapted["survival"] - frozen["survival"]
            ) * 100.0,
            "adapted_minus_stale_points": (
                adapted["survival"] - stale["survival"]
            ) * 100.0,
            "constant_error_before": frozen_recovery[
                "mean_absolute_constant_error"
            ],
            "constant_error_after": adapted_recovery[
                "mean_absolute_constant_error"
            ],
        }
        row["gate_1_beats_frozen"] = float(
            row["adapted_minus_frozen_points"] >= 10.0
            and adapted["mean_absolute_need_error"]
            < frozen["mean_absolute_need_error"]
        )
        row["gate_2_beats_stale_analysis"] = float(
            row["adapted_minus_stale_points"] >= 10.0
        )
        row["gate_3_constants_move_to_truth"] = float(
            row["constant_error_after"] < row["constant_error_before"]
        )
        row["shift_is_informative"] = float(row["shift_bites_points"] >= 15.0)
        rows.append(row)
        print(f"seed {index}: " + json.dumps(row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lives", type=int, default=100)
    parser.add_argument("--adaptation-ticks", type=int, default=ADAPTATION_TICKS)
    args = parser.parse_args()

    rows = run_online_adaptation(
        args.parent,
        seeds=args.seeds,
        lives=args.lives,
        adaptation_ticks=args.adaptation_ticks,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "online_adaptation.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (run_dir / "online_adaptation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== online adaptation summary ===")
    for key in list(rows[0]):
        if key in {"seed_index", "adapt_seed_base"}:
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
        print(f"  {key:<34} mean={values.mean():.4f} sd={sd:.4f}")


if __name__ == "__main__":
    main()
