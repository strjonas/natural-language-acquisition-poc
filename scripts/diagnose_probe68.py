"""What a belief has to be, and how big it has to be, to reach a word.

Probe67 produced a number -- "anything under ~0.12 of body cannot change a
consequential word here" -- and `docs/STATE.md` made it a precondition on an
entire class of future work. A number is not portable. An ecology that has not
been built yet has no measured floor, so the rule cannot be applied to the thing
it most needs to be applied to: deciding what a *larger* world would have to look
like for fine self-knowledge to pay rent in it.

This script replaces the number with two clauses, both read straight off
``need_scores`` and both measured here rather than argued.

**Clause 1 -- the size.** Let ``base`` be the projected body, ``m`` its argmin and
``g = base_(2) - base_(1)`` the gap between the two emptiest axes. Helping any
need other than ``m`` leaves the minimum at ``base_m``; helping ``m`` raises it to
``min(base_m + grant*uptake, base_(2))``. So the decision margin is

    margin  =  E_grant [ min(grant * uptake, g) ]

which *saturates*. Below the axis gap the margin tracks the grant; above it, the
grant buys nothing. Probe67's 0.12 was the saturated branch of this expression,
and the ecology sits about three times past the knee -- which is why the number
looked like a constant of the world rather than a function of two of its knobs.

**Clause 2 -- the shape.** ``need_scores`` is shift-equivariant: adding a constant
to every axis adds it to every score, so it moves no ``argmax`` and changes no
margin. A belief error shared across the needs is therefore invisible to the
word, at any magnitude. **Only the differential part of a belief error can reach a
decision.** That is asserted here by injecting matched errors -- the same
magnitude, once common-mode and once differential -- rather than by deriving it,
because `CLAUDE.md`'s fourth trap is a mechanism that explains everything and is
still not the cause.

The two clauses together are what the record already says in scattered form.
Probe63: "individuality must exist before an individual self-model has anything
to discover" -- individuality is what makes a body error differential. Probe65: an
error about what you are enters multiplied by the horizon -- the horizon is what
lifts a differential above the margin. Probe67: the width is 7--28x too small --
that is clause 1 with the numerator measured.

    PYTHONPATH=src .venv/bin/python scripts/diagnose_probe68.py --lives 8
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.future_request import (
    CONSEQUENTIAL_MARGIN,
    answer_horizon,
    need_scores,
    true_need,
    true_rates,
)
from homesocial.organism.individual_self import _true_body
from homesocial.organism.model import OrganismModel
from homesocial.organism.portion_request import MoveFraction
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)
from homesocial.organism.uncertain_horizon import (
    SURVEY_SEED_BASE,
    _world_kwargs,
    build_arms,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]

# Its own seed band, disjoint from every survey and treatment in the record.
# The highest previously spent is probe67's diagnostic at 1,180,000,000.
DIAGNOSTIC_SEED_BASE = 1_260_000_000

# Multipliers on both portions at once. The ratio small:large is probe64's
# mechanism and is left alone; this sweeps the scale, not the shape.
PORTION_SCALES = (0.125, 0.25, 0.5, 1.0, 2.0)

# Magnitudes for the matched injection. Chosen to straddle the measured margin
# so the control has a live regime and a dead one, and fixed before running.
INJECTION_SIZES = (0.02, 0.05, 0.10, 0.20, 0.40)

# Which self-models to decompose. ``point`` is probe65's ``recursive``; the other
# two are its ceiling and its state-only baseline, so the three of them span the
# arms whose behavioural outcome the record already knows.
DECOMPOSED_ARMS = ("point", "individual", "state_oracle")


def _projection(
    levels: np.ndarray, rates: np.ndarray, arriving: np.ndarray, horizon: int
) -> np.ndarray:
    """The body the caregiver's grant will land on, uncapped.

    The cap is deliberately not applied here. ``need_scores`` applies it, and
    where it binds shift-equivariance is only approximate -- that is a claim
    boundary of clause 2 and is reported as ``capped_share`` rather than hidden.
    """

    return np.asarray(levels, dtype=np.float64) - np.asarray(
        rates, dtype=np.float64
    ) * float(horizon) + np.asarray(arriving, dtype=np.float64)


def _spread(vector: np.ndarray) -> float:
    """Max minus min: how much of a vector can reorder anything.

    This is probe67's own definition of a correction's size, kept identical so
    the two diagnostics are on one instrument.
    """

    return float(np.max(vector) - np.min(vector))


def collect(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    delay: int,
    lives: int,
    seed_base: int,
    overrides: dict | None = None,
) -> list[dict[str, object]]:
    """Record the decision inputs, and the truth, at every scored tick."""

    kwargs = dict(_world_kwargs(delay, 0.03), **(overrides or {}))
    report = replace(organism.report, **kwargs)
    horizon = answer_horizon(delay)
    rows: list[dict[str, object]] = []

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tiers, arms, ledger = build_arms(organism, report, delay=delay)
        for tier in tiers.values():
            tier.reset(packet, world)
        for arm in arms.values():
            arm.errors.reset()
        ledger.reset()
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        hidden = None
        rng = Random(seed + 59_000_003)

        while True:
            fraction = moves.fraction
            now = int(packet.step_count)
            target = true_need(world, report, fraction, delay, ledger, now)

            if packet.step_count >= FIDELITY_WARMUP:
                uptake = arms["point"].tier.believed_uptake()
                arriving = np.asarray(
                    ledger.arriving(now, uptake, delay=delay), dtype=np.float64
                )
                row: dict[str, object] = {
                    "uptake": dict(uptake),
                    "arriving": arriving,
                    "true_levels": np.asarray(_true_body(world), dtype=np.float64),
                    "true_rates": np.asarray(
                        true_rates(world, report, fraction), dtype=np.float64
                    ),
                }
                for name in DECOMPOSED_ARMS:
                    arm = arms[name]
                    row[f"{name}_levels"] = np.asarray(
                        arm.tier.point(), dtype=np.float64
                    )
                    row[f"{name}_rates"] = np.asarray(
                        arm.believed_rates(fraction), dtype=np.float64
                    )
                rows.append(row)

            ledger.record(now, target)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[target]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            if terminated or truncated:
                break
            moves.observe(before, action, packet)
            for tier in tiers.values():
                tier.update(before, action, packet, world, info.get("interoception"))

    return rows


def margins(rows: list[dict[str, object]], report, horizon: int) -> np.ndarray:
    """The decision margin under the *true* body, at a given grant size."""

    out = np.empty(len(rows))
    for index, row in enumerate(rows):
        scores = need_scores(
            believed_levels=row["true_levels"],
            believed_rates=row["true_rates"],
            believed_uptake=row["uptake"],
            arriving=row["arriving"],
            horizon=horizon,
            report=report,
        )
        ordered = np.sort(scores)
        out[index] = float(ordered[-1] - ordered[-2])
    return out


def clause_one(
    rows: list[dict[str, object]], base_report, horizon: int
) -> dict[str, object]:
    """Does ``margin == E_grant[min(grant*uptake, axis gap)]`` hold?"""

    gaps = np.empty(len(rows))
    capped = 0
    for index, row in enumerate(rows):
        raw = _projection(
            row["true_levels"], row["true_rates"], row["arriving"], horizon
        )
        capped += int(bool((raw > 1.0).any()))
        base = np.minimum(1.0, raw)
        ordered = np.sort(base)
        gaps[index] = float(ordered[1] - ordered[0])

    scales: list[dict[str, float]] = []
    for scale in PORTION_SCALES:
        scaled = replace(
            base_report,
            portion_small=base_report.portion_small * scale,
            portion_large=base_report.portion_large * scale,
        )
        actual = margins(rows, scaled, horizon)
        large = float(scaled.large_portion_probability)
        predicted = np.empty(len(rows))
        for index, row in enumerate(rows):
            up = float(np.mean([row["uptake"].get(n, 1.0) for n in REPORT_NEEDS]))
            predicted[index] = (1.0 - large) * min(
                scaled.portion_small * up, gaps[index]
            ) + large * min(scaled.portion_large * up, gaps[index])
        live = actual >= CONSEQUENTIAL_MARGIN
        scales.append(
            {
                "scale": float(scale),
                "expected_grant": float(
                    (1.0 - large) * scaled.portion_small
                    + large * scaled.portion_large
                ),
                "median_margin_consequential": float(np.median(actual[live]))
                if live.any()
                else 0.0,
                "predicted_median_consequential": float(np.median(predicted[live]))
                if live.any()
                else 0.0,
                "formula_mae": float(np.mean(np.abs(predicted - actual))),
            }
        )
    return {
        "median_axis_gap": float(np.median(gaps)),
        "capped_share": float(capped / max(1, len(rows))),
        "scales": scales,
    }


def clause_two(
    rows: list[dict[str, object]], base_report, horizon: int
) -> list[dict[str, float]]:
    """Matched injection: same magnitude, common-mode against differential.

    The common-mode arm adds ``size`` to every axis of the believed body. The
    differential arm adds ``size`` to one axis and nothing to the others, so its
    spread is ``size`` and its common-mode arm's spread is zero. Both perturb the
    belief by the same amount in any norm that does not distinguish them.
    """

    out: list[dict[str, float]] = []
    for size in INJECTION_SIZES:
        common_flips = 0
        differential_flips = 0
        common_margin_shift = []
        differential_margin_shift = []
        ticks = 0
        for index, row in enumerate(rows):
            kwargs = dict(
                believed_rates=row["true_rates"],
                believed_uptake=row["uptake"],
                arriving=row["arriving"],
                horizon=horizon,
                report=base_report,
            )
            clean = need_scores(believed_levels=row["true_levels"], **kwargs)
            base_choice = int(np.argmax(clean))
            ordered = np.sort(clean)
            if float(ordered[-1] - ordered[-2]) < CONSEQUENTIAL_MARGIN:
                continue
            ticks += 1

            shifted = need_scores(
                believed_levels=row["true_levels"] + size, **kwargs
            )
            # Rotate which axis is hit so the differential arm is not always
            # perturbing the same need across the run.
            axis = index % len(REPORT_NEEDS)
            bumped = np.array(row["true_levels"], dtype=np.float64)
            bumped[axis] += size
            differential = need_scores(believed_levels=bumped, **kwargs)

            common_flips += int(int(np.argmax(shifted)) != base_choice)
            differential_flips += int(int(np.argmax(differential)) != base_choice)
            common_margin_shift.append(
                abs(
                    float(np.sort(shifted)[-1] - np.sort(shifted)[-2])
                    - float(ordered[-1] - ordered[-2])
                )
            )
            differential_margin_shift.append(
                abs(
                    float(np.sort(differential)[-1] - np.sort(differential)[-2])
                    - float(ordered[-1] - ordered[-2])
                )
            )
        out.append(
            {
                "size": float(size),
                "consequential_ticks": int(ticks),
                "common_mode_flip_share": float(common_flips / max(1, ticks)),
                "differential_flip_share": float(differential_flips / max(1, ticks)),
                "common_mode_max_margin_shift": float(max(common_margin_shift))
                if common_margin_shift
                else 0.0,
                "differential_max_margin_shift": float(max(differential_margin_shift))
                if differential_margin_shift
                else 0.0,
            }
        )
    return out


def reachability(
    rows: list[dict[str, object]], base_report, horizon: int
) -> list[dict[str, float]]:
    """Per arm: how much of its belief error is differential, against the margin."""

    live_margin = margins(rows, base_report, horizon)
    out: list[dict[str, float]] = []
    for name in DECOMPOSED_ARMS:
        total: list[float] = []
        common: list[float] = []
        differential: list[float] = []
        reaches = 0
        ticks = 0
        for index, row in enumerate(rows):
            if live_margin[index] < CONSEQUENTIAL_MARGIN:
                continue
            ticks += 1
            truth = _projection(
                row["true_levels"], row["true_rates"], row["arriving"], horizon
            )
            believed = _projection(
                row[f"{name}_levels"], row[f"{name}_rates"], row["arriving"], horizon
            )
            error = believed - truth
            mean = float(np.mean(error))
            total.append(float(np.mean(np.abs(error))))
            common.append(abs(mean))
            spread = _spread(error - mean)
            differential.append(spread)
            reaches += int(spread > live_margin[index])
        out.append(
            {
                "arm": name,
                "consequential_ticks": int(ticks),
                "median_abs_error": float(np.median(total)) if total else 0.0,
                "median_common_mode": float(np.median(common)) if common else 0.0,
                "median_differential": float(np.median(differential))
                if differential
                else 0.0,
                "p90_differential": float(np.percentile(differential, 90))
                if differential
                else 0.0,
                "reach_share": float(reaches / max(1, ticks)),
            }
        )
    return out


def reach_by_scale(
    rows: list[dict[str, object]], base_report, horizon: int
) -> list[dict[str, float]]:
    """Reach share against grant size, on one fixed set of trajectories.

    This is the isolation the forward prediction rests on. The lives are the
    ones the ecology actually produced at its own grant size; only the *scoring*
    grant moves, so the axis gaps are held exactly fixed and the margin moves for
    one reason. A treatment run will change the trajectories too -- that is what
    makes the prediction a prediction rather than a restatement.
    """

    out: list[dict[str, float]] = []
    for scale in PORTION_SCALES:
        scaled = replace(
            base_report,
            portion_small=base_report.portion_small * scale,
            portion_large=base_report.portion_large * scale,
        )
        margin = margins(rows, scaled, horizon)
        entry: dict[str, float] = {
            "scale": float(scale),
            "median_margin_consequential": float(
                np.median(margin[margin >= CONSEQUENTIAL_MARGIN])
            )
            if (margin >= CONSEQUENTIAL_MARGIN).any()
            else 0.0,
        }
        for name in DECOMPOSED_ARMS:
            reaches = 0
            ticks = 0
            for index, row in enumerate(rows):
                if margin[index] < CONSEQUENTIAL_MARGIN:
                    continue
                ticks += 1
                truth = _projection(
                    row["true_levels"], row["true_rates"], row["arriving"], horizon
                )
                believed = _projection(
                    row[f"{name}_levels"],
                    row[f"{name}_rates"],
                    row["arriving"],
                    horizon,
                )
                error = believed - truth
                reaches += int(_spread(error - float(np.mean(error))) > margin[index])
            entry[f"{name}_reach"] = float(reaches / max(1, ticks))
            entry["consequential_ticks"] = float(ticks)
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--lives", type=int, default=8)
    parser.add_argument("--delays", default="0,18,24,30")
    parser.add_argument("--seed-base", type=int, default=DIAGNOSTIC_SEED_BASE)
    parser.add_argument(
        "--quantum",
        action="store_true",
        help="measure the compensated sweep: same grant rate, different quantum",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    assert args.seed_base != SURVEY_SEED_BASE, "diagnose off the survey's own lives"
    model, organism = load_organism_checkpoint(Path(args.checkpoint))
    delays = [int(token) for token in args.delays.split(",") if token.strip()]

    if args.quantum:
        from homesocial.organism.decision_granularity import QUANTUM_SWEEP

        horizon = answer_horizon(delays[0])
        print("| quantum | help_period | E[grant] | grant/tick | conseq ticks | "
              "median margin | state_oracle reach | point reach |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        rows_out: list[dict[str, object]] = []
        for scale, period in QUANTUM_SWEEP:
            overrides = {
                "portion_small": organism.report.portion_small * scale,
                "portion_large": organism.report.portion_large * scale,
                "help_period": int(period),
            }
            base_report = replace(
                organism.report, **dict(_world_kwargs(delays[0], 0.03), **overrides)
            )
            rows = collect(
                model,
                organism,
                delay=delays[0],
                lives=args.lives,
                seed_base=args.seed_base,
                overrides=overrides,
            )
            reach = reachability(rows, base_report, horizon)
            margin = margins(rows, base_report, horizon)
            live = margin >= CONSEQUENTIAL_MARGIN
            grant = (
                0.5 * base_report.portion_small + 0.5 * base_report.portion_large
            )
            entry = {
                "quantum": float(scale),
                "help_period": int(period),
                "expected_grant": float(grant),
                "grant_rate": float(grant / period),
                "consequential_ticks": int(live.sum()),
                "median_margin": float(np.median(margin[live])) if live.any() else 0.0,
                "reachability": reach,
            }
            rows_out.append(entry)
            lookup = {r["arm"]: r["reach_share"] for r in reach}
            print(
                f"| {scale:.3f} | {period} | {grant:.4f} | {grant / period:.4f} | "
                f"{int(live.sum())} | {entry['median_margin']:.5f} | "
                f"{lookup['state_oracle']:.4f} | {lookup['point']:.4f} |"
            )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows_out, indent=1, default=float))
        return

    records: list[dict[str, object]] = []
    for delay in delays:
        horizon = answer_horizon(delay)
        base_report = replace(organism.report, **_world_kwargs(delay, 0.03))
        rows = collect(
            model, organism, delay=delay, lives=args.lives, seed_base=args.seed_base
        )
        records.append(
            {
                "help_delay": delay,
                "horizon": horizon,
                "ticks": len(rows),
                "clause_one": clause_one(rows, base_report, horizon),
                "clause_two": clause_two(rows, base_report, horizon),
                "reachability": reachability(rows, base_report, horizon),
                "reach_by_scale": reach_by_scale(rows, base_report, horizon),
            }
        )

    print("## Clause 1 -- the margin is a saturating function of the grant")
    print()
    print("| lag | median axis gap | scale | E[grant] | median margin | "
          "predicted | formula MAE |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for record in records:
        one = record["clause_one"]
        for scale in one["scales"]:
            print(
                f"| {record['help_delay']} | {one['median_axis_gap']:.4f} | "
                f"{scale['scale']} | {scale['expected_grant']:.4f} | "
                f"{scale['median_margin_consequential']:.5f} | "
                f"{scale['predicted_median_consequential']:.5f} | "
                f"{scale['formula_mae']:.6f} |"
            )

    print()
    print("## Clause 2 -- matched injection, common-mode against differential")
    print()
    print("| lag | size | conseq ticks | common-mode flips | differential flips | "
          "common-mode max margin shift |")
    print("|---|---:|---:|---:|---:|---:|")
    for record in records:
        for entry in record["clause_two"]:
            print(
                f"| {record['help_delay']} | {entry['size']:.2f} | "
                f"{entry['consequential_ticks']} | "
                f"{entry['common_mode_flip_share']:.4f} | "
                f"{entry['differential_flip_share']:.4f} | "
                f"{entry['common_mode_max_margin_shift']:.2e} |"
            )

    print()
    print("## Reachability -- what each arm's error actually is")
    print()
    print("| lag | arm | median abs error | median common-mode | "
          "median differential | p90 differential | reach share |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for record in records:
        for entry in record["reachability"]:
            print(
                f"| {record['help_delay']} | {entry['arm']} | "
                f"{entry['median_abs_error']:.5f} | "
                f"{entry['median_common_mode']:.5f} | "
                f"{entry['median_differential']:.5f} | "
                f"{entry['p90_differential']:.5f} | "
                f"{entry['reach_share']:.4f} |"
            )

    print()
    print("## Reach against grant size -- the forward prediction's arithmetic")
    print()
    print("| lag | scale | median margin | conseq ticks | point reach | "
          "individual reach | state_oracle reach |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for record in records:
        for entry in record["reach_by_scale"]:
            print(
                f"| {record['help_delay']} | {entry['scale']} | "
                f"{entry['median_margin_consequential']:.5f} | "
                f"{int(entry['consequential_ticks'])} | "
                f"{entry['point_reach']:.4f} | "
                f"{entry['individual_reach']:.4f} | "
                f"{entry['state_oracle_reach']:.4f} |"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(records, indent=1, default=float))


if __name__ == "__main__":
    main()
