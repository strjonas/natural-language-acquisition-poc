"""Does structure beat scale, or has scale simply never been tried?

Every positive result in this repository came from an explicitly structured,
closed-form mechanism -- probe57's causal template, probe63's RLS, probe66's
exact Beta-Binomial model comparison. Every negative came from a black-box
learned one -- probes 48 through 56. That is a strong pattern, and it is the
pattern on which `docs/VISION_AND_STATUS.md` proposes to bet the opposite way:
its v2 is a recurrent latent with a token decoder, which is the family that has
failed here nine times.

Nobody has ever asked whether the family failed because it is the wrong family
or because it was given 80,000 ticks and 738 gradient updates. Probe54's own
record contains the reason to ask: a disposable ridge readout of its 64-unit
hidden state reached 72.42% balanced accuracy -- *above* the 70% gate its output
head missed at 62.79%. The state held gate-level information the head did not
express. That was written down as a diagnosis and never followed up, because the
line was closed on head variants.

**What this module does not do.** It does not re-run probe54's gate at a larger
budget and report a pass. That gate is failed and stays failed; adjusting a
budget after seeing a result is the thing `CLAUDE.md` forbids most explicitly.
The harness lock at `harness.py:1220` that pins development to 80,000 ticks is
deliberate and is left untouched.

**What it does.** It measures the *shape* of two curves on one matched
instrument -- balanced lowest-need accuracy, `_balanced_accuracy` on
`world.lowest_need()`, the same function and the same truth both audits already
use -- as the developmental budget moves over a wide lever. The endpoint is the
asymptote and the gap between the families, not a threshold. A flat black-box
curve *strengthens* the existing closure rather than reopening it, and that is
the outcome this module is built to be able to report.

Three quantities per cell, all of which the existing audit already returns:

    head    balanced_accuracy                          what it can say
    ridge   diagnostic_hidden_ridge_balanced_accuracy  what it represents
    lesions zero_ / shuffle_balanced_accuracy          that either is causal

`head` against `ridge` separates a representation limit from a readout limit,
which is the distinction that decides what a v2 should scale.

Default off in the sense that matters here: nothing in this module is imported
by the harness, no existing parameter changes, and `OrganismConfig` gains no
knob.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from homesocial.island.world import SURFACES
from homesocial.organism.causal_self import (
    audit_structured_causal_self_model,
    train_structured_causal_self_model,
)
from homesocial.organism.self_belief import (
    audit_explicit_self_belief,
    train_explicit_self_belief,
)
from homesocial.organism.train import load_organism_checkpoint

# -- seed bands ----------------------------------------------------------------
#
# Disjoint from every band in use. The highest previously claimed is probe68's
# treatment at 1,400,000,000; `_check_seed_isolation` below refuses an overlap
# rather than trusting this comment, because `CLAUDE.md` records a 1,000,000
# stride that leaked a developmental stream onto an evaluation base.

SURVEY_DEV_BASE = 1_500_000_000
SURVEY_AUDIT_BASE = 1_700_000_000
TREATMENT_DEV_BASE = 1_900_000_000
TREATMENT_AUDIT_BASE = 2_100_000_000

# Wide enough that the longest budget's world stream cannot reach the next band.
# A life is a few hundred ticks, so a 20,000,000 stride clears a 5,120,000-tick
# development by a factor well over the worst case even if every life were one
# tick long.
SEED_STRIDE = 20_000_000

# The seed the black-box weight initialization is drawn from. The structured
# parameters initialize deterministically, so for that arm only the world stream
# varies -- which is exactly the trap `CLAUDE.md` names, and why `seed_base`
# moves with the seed for both arms rather than `mx.random.seed` alone.
INIT_SEED_BASE = 5_100_003

PARENT = "runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"

# Probe54 and probe55, for orientation only. These were measured at a different
# audit seed base, so they are not a row of any table this module prints; the
# comparison that counts is between cells measured here.
PROBE54_HEAD = 0.6219
PROBE55_HEAD = 0.6279
PROBE55_RIDGE = 0.7242
LOCKED_BUDGET = 80_000


@dataclass(frozen=True)
class Cell:
    """One (family, width, budget) point on the surface."""

    family: str
    width: int
    budget: int

    @property
    def name(self) -> str:
        if self.family == "structured":
            return f"structured@{self.budget}"
        return f"blackbox_w{self.width}@{self.budget}"


def _check_seed_isolation(cells: list[Cell], seeds: int, bands: dict[str, int]) -> None:
    """Refuse a sweep whose development stream can reach an evaluation base.

    The failure this guards is real and is recorded in `CLAUDE.md`: a stride too
    small for the budget puts a later seed's developmental worlds onto the base
    the run is scored against, so the model develops on the lives it is graded
    on and the leak looks like a result.
    """

    longest = max(cell.budget for cell in cells)
    span = seeds * SEED_STRIDE
    if longest >= SEED_STRIDE:
        raise ValueError(
            f"Budget {longest} exceeds the seed stride {SEED_STRIDE}; a "
            "development stream could reach the next seed's worlds."
        )
    ordered = sorted(bands.items(), key=lambda item: item[1])
    for (low_name, low), (high_name, high) in zip(ordered, ordered[1:]):
        if low + span > high:
            raise ValueError(
                f"Band {low_name} at {low} spans {span} seeds and overlaps "
                f"{high_name} at {high}."
            )


def _train_blackbox(cell: Cell, seed: int, dev_base: int, log_every: int):
    model, config = load_organism_checkpoint(PARENT)
    # Varies the GRU initialization, which for this family is a real source of
    # variation -- unlike the structured arm, whose parameters are deterministic.
    mx.random.seed(INIT_SEED_BASE + seed * 7919)
    model.enable_explicit_self_belief(cell.width, relational_urgency=False)
    _, development = train_explicit_self_belief(
        model,
        config,
        steps=cell.budget,
        learning_rate=3e-4,
        seed_base=dev_base,
        log_every_lives=log_every,
    )
    return model, config, development


def _train_structured(cell: Cell, seed: int, dev_base: int, log_every: int):
    model, config = load_organism_checkpoint(PARENT)
    mx.random.seed(INIT_SEED_BASE + seed * 7919)
    model.enable_structured_causal_self_model(len(SURFACES))
    _, development = train_structured_causal_self_model(
        model,
        config,
        steps=cell.budget,
        learning_rate=3e-3,
        seed_base=dev_base,
        log_every_lives=log_every,
    )
    return model, config, development


def run_cell(
    cell: Cell,
    *,
    seed: int,
    lives: int,
    dev_band: int,
    audit_band: int,
    log_every: int = 100_000,
) -> dict[str, object]:
    """Develop one cell from the frozen parent and score it on the audit."""

    dev_base = dev_band + seed * SEED_STRIDE
    audit_base = audit_band + seed * SEED_STRIDE
    started = time.time()
    if cell.family == "blackbox":
        model, config, development = _train_blackbox(cell, seed, dev_base, log_every)
        gate = audit_explicit_self_belief(
            model, config, lives=lives, seed_base=audit_base
        )
        head = float(gate["balanced_accuracy"])
        ridge = float(gate["diagnostic_hidden_ridge_balanced_accuracy"])
        affine = float(gate.get("diagnostic_affine_belief_balanced_accuracy", np.nan))
    else:
        model, config, development = _train_structured(cell, seed, dev_base, log_every)
        gate = audit_structured_causal_self_model(
            model, config, lives=lives, seed_base=audit_base
        )
        head = float(gate["balanced_accuracy"])
        # The structured arm has no hidden state to read out; its belief *is*
        # its output, so representation and readout cannot come apart and the
        # ridge column is not applicable rather than zero.
        ridge = float("nan")
        affine = float("nan")
    return {
        "cell": cell.name,
        "family": cell.family,
        "width": cell.width,
        "budget": cell.budget,
        "seed": seed,
        "lives": lives,
        "dev_seed_base": dev_base,
        "audit_seed_base": audit_base,
        "head_balanced_accuracy": head,
        "ridge_balanced_accuracy": ridge,
        "affine_balanced_accuracy": affine,
        "zero_balanced_accuracy": float(gate["zero_balanced_accuracy"]),
        "shuffle_balanced_accuracy": float(gate["shuffle_balanced_accuracy"]),
        "observation_balanced_accuracy": float(
            gate.get("observation_balanced_accuracy", np.nan)
        ),
        "final_loss": float(development.get("last_quarter_loss", np.nan)),
        "updates": float(development.get("updates", np.nan)),
        "development_ticks": float(development.get("development_ticks", np.nan)),
        "wall_seconds": time.time() - started,
    }


def survey(
    *,
    widths: tuple[int, ...],
    budgets: tuple[int, ...],
    seeds: int,
    lives: int,
    include_structured: bool,
    run_dir: Path,
    dev_band: int = SURVEY_DEV_BASE,
    audit_band: int = SURVEY_AUDIT_BASE,
) -> list[dict[str, object]]:
    cells = [
        Cell("blackbox", width, budget) for width in widths for budget in budgets
    ]
    if include_structured:
        cells += [Cell("structured", 0, budget) for budget in budgets]
    _check_seed_isolation(
        cells,
        seeds,
        {
            "survey_dev": SURVEY_DEV_BASE,
            "survey_audit": SURVEY_AUDIT_BASE,
            "treatment_dev": TREATMENT_DEV_BASE,
            "treatment_audit": TREATMENT_AUDIT_BASE,
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    out = run_dir / "scaling_survey.json"
    for seed in range(seeds):
        for cell in cells:
            row = run_cell(
                cell,
                seed=seed,
                lives=lives,
                dev_band=dev_band,
                audit_band=audit_band,
            )
            rows.append(row)
            print(
                f"[{row['wall_seconds']:6.1f}s] seed {seed} {cell.name:<22} "
                f"head {row['head_balanced_accuracy']:.4f} "
                f"ridge {row['ridge_balanced_accuracy']:.4f} "
                f"zero {row['zero_balanced_accuracy']:.4f}",
                flush=True,
            )
            # Written after every cell so a long run is readable while it runs
            # and survives an interruption.
            out.write_text(json.dumps(rows, indent=2))
    return rows


def summarize(rows: list[dict[str, object]]) -> str:
    """A table per family: budget against head and ridge, averaged over seeds."""

    lines: list[str] = []
    keys = sorted(
        {(str(r["family"]), int(r["width"])) for r in rows},
        key=lambda k: (k[0], k[1]),
    )
    for family, width in keys:
        label = family if family == "structured" else f"{family} width {width}"
        lines.append(f"\n{label}")
        lines.append(f"  {'budget':>10}  {'head':>8}  {'ridge':>8}  {'seeds':>5}")
        budgets = sorted(
            {
                int(r["budget"])
                for r in rows
                if r["family"] == family and int(r["width"]) == width
            }
        )
        for budget in budgets:
            matched = [
                r
                for r in rows
                if r["family"] == family
                and int(r["width"]) == width
                and int(r["budget"]) == budget
            ]
            head = float(np.mean([float(r["head_balanced_accuracy"]) for r in matched]))
            ridge_values = [
                float(r["ridge_balanced_accuracy"])
                for r in matched
                if not np.isnan(float(r["ridge_balanced_accuracy"]))
            ]
            ridge = float(np.mean(ridge_values)) if ridge_values else float("nan")
            lines.append(
                f"  {budget:>10}  {head:>8.4f}  {ridge:>8.4f}  {len(matched):>5}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/organism/probe69_belief_scaling")
    parser.add_argument("--widths", default="64")
    parser.add_argument("--budgets", default="80000,320000,1280000")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--lives", type=int, default=60)
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--treatment-band", action="store_true")
    args = parser.parse_args()

    widths = tuple(int(part) for part in args.widths.split(","))
    budgets = tuple(int(part) for part in args.budgets.split(","))
    rows = survey(
        widths=widths,
        budgets=budgets,
        seeds=args.seeds,
        lives=args.lives,
        include_structured=args.structured,
        run_dir=Path(args.run_dir),
        dev_band=TREATMENT_DEV_BASE if args.treatment_band else SURVEY_DEV_BASE,
        audit_band=TREATMENT_AUDIT_BASE if args.treatment_band else SURVEY_AUDIT_BASE,
    )
    print(summarize(rows))


if __name__ == "__main__":
    main()
