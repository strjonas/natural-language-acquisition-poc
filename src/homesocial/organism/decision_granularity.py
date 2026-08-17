"""Probe68: is the value of a self-model set by the granularity of the decision?

Preregistration: ``docs/decisions/2026-08-17-decision-granularity-preregistration.md``.
Read it before reading this. The gates below are the ones it locked.

Probe67's durable output was a number -- in probe65's ecology a consequential
decision is worth about 0.12 of body, so nothing smaller can change a word that
matters. `docs/STATE.md` made it a precondition on a whole class of future work.
A number cannot be applied to a world that does not exist yet, and that is the
case it most needs to cover: deciding what a *larger* ecology would have to look
like for fine self-knowledge to pay rent in it.

``scripts/diagnose_probe68.py`` replaced the number with a formula, read straight
off ``need_scores`` and measured to an MAE of 0.000000 at two of four lags:

    margin  =  E_grant [ min( grant * uptake, axis gap ) ]

It **saturates**. Below the axis gap the margin tracks the grant; above it the
grant buys nothing. This ecology's expected grant is 0.40 against a median axis
gap of 0.08--0.18, so it sits about three times past its own knee -- which is why
probe67's floor looked like a constant of the world rather than a function of two
of its knobs.

That makes a prediction no amount of retuning could make, and it is the reason
this probe exists rather than another mechanism probe:

    Doubling the caregiver's grant -- twice as much food in the world -- changes
    the decision granularity by exactly nothing, and must therefore change the
    behavioural value of knowing what you are by nothing.

    Quartering it moves the granularity by a factor of two, and must raise that
    value.

One null and one response, from one lever, with the crossover measured before the
treatment ran. And a second null of a *different* kind at lag 0, where the
numerator rather than the denominator is what fails: a rate error of 0.0045 at
horizon 1 is below the smallest margin in the sweep, so no grant size can make
the rate matter there.

Nothing here is trained. The self-models are probe63's, the motor policy is
probe52's frozen parent, the rule is probe60's ``E[min(next body)]`` at probe65's
horizon, and every arm is probe65's, bit-identical. The only thing that moves is
how much the caregiver hands over -- and ``portion_scale`` defaults to 1.0, where
``tests/test_decision_granularity.py`` asserts this module reproduces probe65
exactly rather than approximately.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from homesocial.organism.future_request import (
    INTEROCEPTION,
    OPERATING_DELAY,
    run_closed_loop,
    run_open_loop,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.train import OrganismConfig, load_organism_checkpoint

# The locked sweep. ``1.0`` is probe65's ecology untouched; ``2.0`` is the
# saturation null; ``0.5`` and ``0.25`` are the unsaturated response.
PORTION_SCALES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

# -- the compensated sweep, forced by the ceiling survey ------------------------
#
# The uncompensated sweep above is unusable below scale 1.0: the survey on band
# 1,320,000,000 measured **oracle survival 0.000 at scales 0.5 and 0.25**. Even a
# model that knows everything starves when the grant is halved. That is a fact
# about this ecology worth stating on its own -- probe65's operating point has no
# headroom beneath it, and the grant is pinned from below by viability.
#
# Which is why the granularity looked like a constant. A world whose help must be
# large to keep anything alive has a coarse decision surface *by necessity*, and
# no amount of belief refinement can reach past it.
#
# So the quantum is moved while the *rate* of help is held exactly fixed: smaller
# portions delivered proportionally more often. ``grant / help_period`` is
# identical in every row below, so nothing this sweep finds can be nutrition.
#
#   quantum | help_period | portions        | grant/tick
#   1/3     | 2           | 0.0667, 0.20    | 0.0667
#   1/2     | 3           | 0.10,   0.30    | 0.0667
#   1       | 6           | 0.20,   0.60    | 0.0667  (probe65, untouched)
#   2       | 12          | 0.40,   1.20    | 0.0667
#
# ``help_period`` appears in `docs/STATE.md`'s do-not-reopen list, by name. That
# entry forbids *retuning* it -- moving the caregiver's clock until a result
# appears. Here it is not free: it is pinned by ``portion_scale * 6`` so that the
# nutrition rate cannot move, which is the exact quantity a retuning would change.
# The horizon is untouched as well: ``answer_horizon`` reads the lag and never the
# period, so every row below answers at eighteen ticks like probe65's.
QUANTUM_SWEEP: tuple[tuple[float, int], ...] = (
    (1.0 / 3.0, 2),
    (0.5, 3),
    (1.0, 6),
    (2.0, 12),
)
BASE_HELP_PERIOD = 6

# -- the locked treatment cells ------------------------------------------------
#
# Two levers, each moving one factor alone. That dissociation is the design, and
# neither sweep could have produced it on its own -- the uncompensated one is
# unusable below scale 1.0 and the compensated one is unusable above quantum 1,
# so each covers exactly the half the other cannot reach.
#
#   label            nutrition        granularity       what it tests
#   nutrition_1x     0.0667/tick      0.187             the shared reference
#   nutrition_2x     0.1333/tick      0.187 (identical) resources move, grain does not
#   quantum_half     0.0667/tick      0.101             grain moves, resources do not
#   quantum_third    0.0667/tick      0.078             grain moves further
#   lag0_*           either           either            the numerator null
#
# ``nutrition_1x`` is probe65's ecology with every constant untouched, and is the
# reference every contrast is taken against.
TREATMENT_CELLS: tuple[tuple[str, int, float, int], ...] = (
    ("nutrition_1x", OPERATING_DELAY, 1.0, 6),
    ("nutrition_2x", OPERATING_DELAY, 2.0, 6),
    ("quantum_half", OPERATING_DELAY, 0.5, 3),
    ("quantum_third", OPERATING_DELAY, 1.0 / 3.0, 2),
    ("lag0_1x", 0, 1.0, 6),
    ("lag0_2x", 0, 2.0, 6),
)

# Lag 18 is probe65's operating point and carries P1--P3. Lag 0 carries P4, the
# null whose cause is a numerator rather than a denominator.
TREATMENT_DELAYS: tuple[int, ...] = (OPERATING_DELAY, 0)

# The contrast. ``individual`` knows what it is and not exactly where;
# ``state_oracle`` knows exactly where it is and believes it is typical. The gap
# between them *is* the behavioural value of knowing what you are, and probe65
# measured it at +0.1017 at lag 18 and -0.1220 at lag 0.
CONTRAST = ("individual", "state_oracle")

# The two contrasts the gates are actually taken on, and the reason there are
# two. ``individual - state_oracle`` mixes a rate advantage against a state
# disadvantage, so it cannot say which of them granularity acts on. These can:
#
#   RATE_VALUE    oracle - state_oracle       both hold the true body; they
#                                             differ only in whether the rates
#                                             are this individual's or the
#                                             species'.
#   STATE_VALUE   state_oracle - population   both hold the species rates; they
#                                             differ only in whether the body is
#                                             read or filtered.
#
# One moves one factor each, and STATE_VALUE is the lesion that should kill the
# claim: if a finer decision surface simply makes every decision harder, both
# rise together and there is nothing here about self-knowledge in particular.
RATE_VALUE = ("oracle", "state_oracle")
STATE_VALUE = ("state_oracle", "population")

# Preregistered equivalence band for the two null gates, and the seed count that
# gives it power. Probe65's five seeds gave a paired half-width of ~0.026 on this
# endpoint, which is wider than the band; eight seeds bring it to ~0.018.
EQUIVALENCE_BAND = 0.02
SEEDS = 8

# A band of its own. The highest previously spent is probe67's diagnostic at
# 1,180,000,000 and probe68's own diagnostic at 1,260,000,000; the survey band
# below is disjoint from both and from the treatment.
SURVEY_SEED_BASE = 1_320_000_000
TREATMENT_SEED_BASE = 1_400_000_000
SEED_STRIDE = 2_000_000

# ``_check_seed_isolation``'s reason for existing, from CLAUDE.md's second trap:
# a stride that lets one seed develop on the lives another is scored against.
_BANDS = {
    "survey": SURVEY_SEED_BASE,
    "treatment": TREATMENT_SEED_BASE,
    "diagnostic": 1_260_000_000,
}


def _check_seed_isolation(lives: int, seeds: int = SEEDS) -> None:
    """No seed's lives may overlap another band's, or another seed's."""

    if lives > SEED_STRIDE:
        raise ValueError(
            f"{lives} lives per seed exceeds the {SEED_STRIDE} stride; seeds would overlap."
        )
    spans = []
    for name, base in _BANDS.items():
        spans.append((base, base + seeds * SEED_STRIDE, name))
    spans.sort()
    for (start, end, name), (next_start, _, next_name) in zip(spans, spans[1:]):
        if end > next_start:
            raise ValueError(f"seed bands {name} and {next_name} overlap.")


def _paired_interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and a Student-t 95% interval over per-seed paired differences."""

    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    # Two-sided 95% t quantiles for df = 1..15, then the normal limit. Written
    # out so this module does not depend on scipy.
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131,
    }
    critical = table.get(int(array.size - 1), 1.96)
    half = critical * float(array.std(ddof=1)) / float(np.sqrt(array.size))
    return mean, mean - half, mean + half


def run_scale(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    delay: int,
    portion_scale: float,
    lives: int,
    seeds: int,
    seed_base: int,
    help_period: int | None = None,
    label: str = "",
) -> dict[str, object]:
    """One cell of the sweep: every arm, every seed, at one grant size."""

    per_seed: list[dict[str, object]] = []
    contrasts: list[float] = []
    rate_values: list[float] = []
    state_values: list[float] = []
    for seed in range(seeds):
        record = run_open_loop(
            model,
            organism,
            world_name="metabolic",
            delay=delay,
            lives=lives,
            seed_base=seed_base + seed * SEED_STRIDE,
            interoception=INTEROCEPTION,
            portion_scale=portion_scale,
            help_period=help_period,
        )
        accuracy = record["consequential_accuracy"]
        contrast = float(accuracy[CONTRAST[0]]) - float(accuracy[CONTRAST[1]])
        contrasts.append(contrast)
        rate_values.append(
            float(accuracy[RATE_VALUE[0]]) - float(accuracy[RATE_VALUE[1]])
        )
        state_values.append(
            float(accuracy[STATE_VALUE[0]]) - float(accuracy[STATE_VALUE[1]])
        )
        per_seed.append(
            {
                "seed": seed,
                "contrast": contrast,
                "rate_value": rate_values[-1],
                "state_value": state_values[-1],
                "consequential_accuracy": accuracy,
                "need_accuracy": record["need_accuracy"],
                "regret": record["regret"],
                "consequential_share": record["consequential_share"],
                "scored_ticks": record["scored_ticks"],
                "rate_error": record["rate_error"],
            }
        )
    mean, low, high = _paired_interval(contrasts)
    rate_mean, rate_low, rate_high = _paired_interval(rate_values)
    state_mean, state_low, state_high = _paired_interval(state_values)
    period = int(help_period or organism.report.help_period)
    grant = (
        organism.report.portion_small * portion_scale * 0.5
        + organism.report.portion_large * portion_scale * 0.5
    )
    return {
        "label": label,
        "help_delay": delay,
        "portion_scale": float(portion_scale),
        "help_period": period,
        "expected_grant": float(grant),
        # Held exactly fixed across the compensated rows; doubled across the
        # uncompensated pair. Reading this column is how a reader checks that
        # each lever moved one factor and not two.
        "grant_rate": float(grant / period),
        "lives": lives,
        "seeds": seeds,
        "seed_base": seed_base,
        "contrast_mean": mean,
        "contrast_low": low,
        "contrast_high": high,
        "contrast_half_width": (high - low) / 2.0,
        "contrasts": contrasts,
        "rate_values": rate_values,
        "rate_value_mean": rate_mean,
        "rate_value_low": rate_low,
        "rate_value_high": rate_high,
        "state_values": state_values,
        "state_value_mean": state_mean,
        "state_value_low": state_low,
        "state_value_high": state_high,
        "per_seed": per_seed,
    }


def _difference(
    a: dict[str, object], b: dict[str, object], key: str = "contrasts"
) -> dict[str, float]:
    """Paired per-seed difference between two cells of the sweep."""

    paired = [x - y for x, y in zip(a[key], b[key])]
    mean, low, high = _paired_interval(paired)
    return {
        "mean": mean,
        "low": low,
        "high": high,
        "half_width": (high - low) / 2.0,
        "same_sign": int(sum(1 for value in paired if value * mean > 0)),
        "n": len(paired),
    }


def _equivalence(pair: dict[str, float]) -> str:
    """Three-valued grading for a null gate.

    An interval wider than the band is **UNRESOLVED**, never a pass. Probe65
    made exactly that error in the opposite direction -- it reported a null on
    regret that its own follow-up showed to be a sample-size statement -- and
    `docs/STATE.md` carries the correction. This function is where that
    correction is enforced rather than remembered.
    """

    if pair["half_width"] > EQUIVALENCE_BAND:
        return "UNRESOLVED"
    return "pass" if abs(pair["mean"]) <= EQUIVALENCE_BAND else "fail"


def grade(cells: list[dict[str, object]]) -> dict[str, object]:
    """The locked gates. Nothing here is decided after the numbers."""

    index = {c.get("label"): c for c in cells}
    gates: dict[str, object] = {}

    def pair(a: str, b: str, key: str = "contrasts") -> dict[str, float] | None:
        if a not in index or b not in index:
            return None
        return _difference(index[a], index[b], key)

    # G6 -- the dissociation, and the whole point of the probe. Between these two
    # cells the grant *rate* is identical to the last decimal and only the
    # quantum differs, so the margin halves while nothing about nutrition moves.
    # The law says the value of knowing your own rate rises with that -- a rate
    # error is multiplied by the horizon and was being swallowed by a coarse
    # margin -- and that the value of knowing where you are does not, because a
    # state error enters once and was already small against every margin here.
    #
    # STATE_VALUE is the lesion. If a finer decision surface simply makes every
    # decision harder, both rise together and this claim is empty.
    rate_step = pair("quantum_half", "nutrition_1x", "rate_values")
    state_step = pair("quantum_half", "nutrition_1x", "state_values")
    if rate_step is not None and state_step is not None:
        gates["G6_dissociation"] = {
            "rate_step": rate_step,
            "state_step": state_step,
            "verdict": "pass"
            if (
                rate_step["low"] > 0.0
                and rate_step["same_sign"] >= 6
                and abs(state_step["mean"]) < rate_step["mean"]
            )
            else "fail",
        }

    # G2 -- resources move, granularity does not. Nutrition doubles between these
    # two cells and the measured margin is identical, so the law says nothing
    # happens. A world where "more food makes self-knowledge matter less" fails
    # here, and so does any account in which the grant acts through anything but
    # the grain of the decision.
    g2 = pair("nutrition_2x", "nutrition_1x")
    if g2 is not None:
        g2["verdict"] = _equivalence(g2)
        gates["G2_nutrition_null"] = g2

    # G3 -- granularity moves, resources do not. The grant rate is identical to
    # the last decimal between these two cells; only the quantum differs, and
    # the measured margin halves with it.
    g3 = pair("quantum_half", "nutrition_1x")
    if g3 is not None:
        g3["verdict"] = (
            "pass" if g3["low"] > 0.0 and g3["same_sign"] >= 6 else "fail"
        )
        gates["G3_granularity_response"] = g3

    # G4 -- the second step, predicted to be much smaller than the first. Reach
    # rises 0.441 -> 0.760 from quantum 1 to 1/2 and only 0.760 -> 0.797 from 1/2
    # to 1/3, so a response that is *linear in the quantum* rather than in the
    # reach fails here while passing G3.
    first = pair("quantum_half", "nutrition_1x")
    second = pair("quantum_third", "quantum_half")
    if first is not None and second is not None:
        gates["G4_diminishing_step"] = {
            "first_step": first["mean"],
            "second_step": second["mean"],
            "verdict": "pass"
            if abs(second["mean"]) < abs(first["mean"])
            else "fail",
        }

    # G5 -- the lag-0 null, and its cause is the numerator rather than the
    # denominator: a rate error of 0.0045 at horizon 1 is under every margin in
    # the sweep, so no grant size can lift it to a decision.
    g5 = pair("lag0_2x", "lag0_1x")
    zero = [c for c in cells if c["help_delay"] == 0]
    if g5 is not None and zero:
        all_negative = all(c["contrast_high"] < 0.0 for c in zero)
        verdict = _equivalence(g5)
        if verdict == "pass" and not all_negative:
            verdict = "fail"
        gates["G5_lag0_null"] = {
            **g5,
            "contrast_means": [c["contrast_mean"] for c in zero],
            "all_negative": all_negative,
            "verdict": verdict,
        }
    return gates


def run_ceiling(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    delay: int,
    lives: int,
    seed_base: int,
    scales: tuple[float, ...] = PORTION_SCALES,
) -> list[dict[str, object]]:
    """G6: does any cell of the sweep sit in a collapsed or saturated ecology?

    Probe60 made this binding -- if a perfect model cannot reach the threshold,
    the gate is measuring something else. Here it runs in both directions: an
    ``oracle`` that cannot survive means the grant starved the world, and a
    ``myopic`` that survives as well as the oracle means the grant made the
    decision free.
    """

    out: list[dict[str, object]] = []
    for scale in scales:
        cell: dict[str, object] = {"portion_scale": scale}
        for arm_name in ("oracle", "myopic"):
            record = run_closed_loop(
                model,
                organism,
                arm_name=arm_name,
                world_name="metabolic",
                delay=delay,
                lives=lives,
                seed_base=seed_base,
                portion_scale=scale,
            )
            cell[arm_name] = record
        oracle = float(cell["oracle"]["survival"])
        myopic = float(cell["myopic"]["survival"])
        cell["oracle_survival"] = oracle
        cell["myopic_survival"] = myopic
        # Collapsed if even a perfect model cannot live; free if a model with no
        # lookahead at all lives as well as the perfect one.
        cell["collapsed"] = oracle < 0.30
        cell["free"] = myopic >= oracle
        cell["verdict"] = (
            "collapsed" if cell["collapsed"]
            else "free" if cell["free"]
            else "usable"
        )
        out.append(cell)
    return out


def run_quantum_survey(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    delay: int,
    lives: int,
    seed_base: int,
    sweep: tuple[tuple[float, int], ...] = QUANTUM_SWEEP,
) -> list[dict[str, object]]:
    """Is the compensated sweep usable, and does it move the granularity?

    Three questions in one pass, all of which have to be answered before a gate
    is worth locking:

    1. Does the ecology survive each row? (probe60's binding rule)
    2. Does the quantum actually move the realised decision margin, or does the
       ``arriving`` term absorb it? More requests in flight at a shorter period
       changes the axis gap too, so this cannot be assumed from the formula.
    3. Does the contrast this probe measures exist at each row?
    """

    out: list[dict[str, object]] = []
    for scale, period in sweep:
        row: dict[str, object] = {
            "quantum": float(scale),
            "help_period": int(period),
        }
        panel = run_open_loop(
            model,
            organism,
            world_name="metabolic",
            delay=delay,
            lives=lives,
            seed_base=seed_base,
            interoception=INTEROCEPTION,
            portion_scale=scale,
            help_period=period,
        )
        accuracy = panel["consequential_accuracy"]
        row["grant_rate"] = panel["grant_rate"]
        row["expected_grant"] = panel["expected_grant"]
        row["consequential_share"] = panel["consequential_share"]
        row["scored_ticks"] = panel["scored_ticks"]
        row["consequential_accuracy"] = accuracy
        row["contrast"] = float(accuracy[CONTRAST[0]]) - float(accuracy[CONTRAST[1]])
        for arm_name in ("oracle", "myopic"):
            closed = run_closed_loop(
                model,
                organism,
                arm_name=arm_name,
                world_name="metabolic",
                delay=delay,
                lives=lives,
                seed_base=seed_base,
                portion_scale=scale,
                help_period=period,
            )
            row[f"{arm_name}_survival"] = float(closed["survival"])
        oracle = float(row["oracle_survival"])
        row["verdict"] = (
            "collapsed"
            if oracle < 0.30
            else "free"
            if float(row["myopic_survival"]) >= oracle
            else "usable"
        )
        out.append(row)
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
    parser.add_argument("--lives", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument(
        "--scales", default=",".join(str(s) for s in PORTION_SCALES)
    )
    parser.add_argument(
        "--delays", default=",".join(str(d) for d in TREATMENT_DELAYS)
    )
    parser.add_argument(
        "--treatment",
        action="store_true",
        help="run on the treatment band; otherwise the survey band",
    )
    parser.add_argument(
        "--ceiling",
        action="store_true",
        help="G6 only: is any cell of the sweep collapsed or free?",
    )
    parser.add_argument(
        "--quantum",
        action="store_true",
        help="survey the compensated sweep: same grant rate, different quantum",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seed_base = TREATMENT_SEED_BASE if args.treatment else SURVEY_SEED_BASE
    _check_seed_isolation(args.lives, args.seeds)
    model, organism = load_organism_checkpoint(Path(args.checkpoint))
    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    delays = [int(d) for d in args.delays.split(",") if d.strip()]

    if args.quantum:
        assert not args.treatment, "the quantum survey runs off the treatment band"
        rows = run_quantum_survey(
            model,
            organism,
            delay=OPERATING_DELAY,
            lives=args.lives,
            seed_base=seed_base,
        )
        print("| quantum | help_period | E[grant] | grant/tick | conseq share | "
              "individual - state_oracle | oracle surv | myopic surv | verdict |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            print(
                f"| {row['quantum']:.3f} | {row['help_period']} | "
                f"{row['expected_grant']:.4f} | {row['grant_rate']:.4f} | "
                f"{row['consequential_share']:.4f} | {row['contrast']:+.4f} | "
                f"{row['oracle_survival']:.3f} | {row['myopic_survival']:.3f} | "
                f"{row['verdict']} |"
            )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
            print(f"\nWrote {args.out}")
        return

    if args.ceiling:
        assert not args.treatment, "the ceiling check runs off the treatment band"
        rows = run_ceiling(
            model,
            organism,
            delay=OPERATING_DELAY,
            lives=args.lives,
            seed_base=seed_base,
            scales=tuple(scales),
        )
        print("| scale | oracle survival | myopic survival | verdict |")
        print("|---|---:|---:|---|")
        for row in rows:
            print(
                f"| {row['portion_scale']} | {row['oracle_survival']:.3f} | "
                f"{row['myopic_survival']:.3f} | {row['verdict']} |"
            )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
            print(f"\nWrote {args.out}")
        return

    cells: list[dict[str, object]] = []
    for label, delay, scale, period in TREATMENT_CELLS:
        cells.append(
            run_scale(
                model,
                organism,
                delay=delay,
                portion_scale=scale,
                help_period=period,
                label=label,
                lives=args.lives,
                seeds=args.seeds,
                seed_base=seed_base,
            )
        )
        cell = cells[-1]
        print(
            f"  {label:<15} lag {cell['help_delay']:>2}  grant {cell['expected_grant']:.4f}"
            f"  rate {cell['grant_rate']:.4f}  contrast {cell['contrast_mean']:+.4f}",
            flush=True,
        )

    gates = grade(cells)
    print()
    print("| cell | lag | E[grant] | grant/tick | individual - state_oracle | "
          "95% CI | half-width |")
    print("|---|---:|---:|---:|---:|---|---:|")
    for cell in cells:
        print(
            f"| {cell['label']} | {cell['help_delay']} | "
            f"{cell['expected_grant']:.4f} | {cell['grant_rate']:.4f} | "
            f"{cell['contrast_mean']:+.4f} | "
            f"[{cell['contrast_low']:+.4f}, {cell['contrast_high']:+.4f}] | "
            f"{cell['contrast_half_width']:.4f} |"
        )
    print()
    for name, gate in gates.items():
        print(f"{name}: {gate['verdict']}")

    payload = {
        "treatment": bool(args.treatment),
        "seed_base": seed_base,
        "lives": args.lives,
        "seeds": args.seeds,
        "equivalence_band": EQUIVALENCE_BAND,
        "cells": cells,
        "gates": gates,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=1, default=float))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
