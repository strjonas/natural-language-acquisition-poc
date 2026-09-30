"""Grade probe69 against the gates locked in its preregistration.

Reads the treatment and extrapolation artifacts and prints one line per gate
with the locked threshold beside the measured value, so a failure is visible
rather than reconstructable. Nothing here chooses a threshold; every number in
`GATES` is copied from
`docs/decisions/2026-08-17-belief-scaling-preregistration.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path("runs/organism/probe69_belief_scaling")
BUDGETS = (80_000, 320_000, 1_280_000)
TOP = 5_120_000
PROBE54_PUBLISHED = 0.6219


def load(name: str) -> list[dict]:
    path = RUN / name / "scaling_survey.json"
    return json.loads(path.read_text()) if path.exists() else []


def pick(rows, family, width, budget, seed):
    for row in rows:
        if (
            row["family"] == family
            and int(row["width"]) == width
            and int(row["budget"]) == budget
            and int(row["seed"]) == seed
        ):
            return float(row["head_balanced_accuracy"])
    return None


def interval(values: np.ndarray) -> tuple[float, float]:
    """Normal-approximation 95% interval on the mean of a small sample."""

    if len(values) < 2:
        return (float("nan"), float("nan"))
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    mean = float(np.mean(values))
    return (mean - half, mean + half)


def verdict(passed: bool) -> str:
    return "PASS" if passed else "**FAIL**"


def main() -> None:
    treatment = load("treatment")
    extrapolation = load("extrapolation")
    if not treatment:
        sys.exit("No treatment artifact.")
    seeds = sorted({int(r["seed"]) for r in treatment})

    struct = {b: np.array([pick(treatment, "structured", 0, b, s) for s in seeds]) for b in BUDGETS}
    bb256 = {b: np.array([pick(treatment, "blackbox", 256, b, s) for s in seeds]) for b in BUDGETS}
    bb64 = {b: np.array([pick(treatment, "blackbox", 64, b, s) for s in seeds]) for b in BUDGETS}
    gap = {b: struct[b] - bb256[b] for b in BUDGETS}

    print(f"seeds: {len(seeds)}  cells: {len(treatment)} treatment, {len(extrapolation)} extrapolation\n")
    print(f"{'budget':>9} {'struct':>8} {'bb_w64':>8} {'bb_w256':>8} {'gap_w256':>9}")
    for b in BUDGETS:
        print(
            f"{b:>9} {struct[b].mean():8.4f} {bb64[b].mean():8.4f} "
            f"{bb256[b].mean():8.4f} {gap[b].mean():9.4f}"
        )

    print("\ngate  locked                                              measured        verdict")

    d1 = struct[1_280_000] - struct[80_000]
    print(f"G1    structured 1.28M-80k < +0.05                        {d1.mean():+.4f}         {verdict(d1.mean() < 0.05)}")

    d2 = bb256[1_280_000] - bb256[80_000]
    lo2, hi2 = interval(d2)
    n2 = int((d2 > 0).sum())
    ok2 = d2.mean() > 0.10 and lo2 > 0 and n2 >= 4
    print(f"G2    blackbox256 1.28M-80k > +0.10, CI>0, >=4/5           {d2.mean():+.4f} [{lo2:+.4f},{hi2:+.4f}] {n2}/5  {verdict(ok2)}")

    per_seed3 = gap[1_280_000] < 0.5 * gap[80_000]
    n3 = int(per_seed3.sum())
    ok3 = n3 >= 4
    ratio = gap[1_280_000].mean() / gap[80_000].mean()
    print(f"G3    gap(1.28M) < 0.5*gap(80k), >=4/5 seeds               {ratio:.3f}x  {n3}/5   {verdict(ok3)}")

    worst = 0.0
    for row in treatment + extrapolation:
        worst = max(worst, float(row["zero_balanced_accuracy"]), float(row["shuffle_balanced_accuracy"]))
    print(f"G4    every lesion <= 0.40, all cells all seeds            max {worst:.4f}      {verdict(worst <= 0.40)}")

    ordered = (gap[80_000] > gap[320_000]) & (gap[320_000] > gap[1_280_000])
    n5 = int(ordered.sum())
    print(f"G5    gap 80k > 320k > 1.28M, >=4/5 seeds                  {n5}/5           {verdict(n5 >= 4)}")

    if extrapolation:
        xs = sorted({int(r["seed"]) for r in extrapolation})
        top_bb = np.array([v for s in xs if (v := pick(extrapolation, "blackbox", 256, TOP, s)) is not None])
        top_st = np.array([v for s in xs if (v := pick(extrapolation, "structured", 0, TOP, s)) is not None])
        n = min(len(top_bb), len(top_st))
        if n:
            gtop = top_st[:n] - top_bb[:n]
            in_band = 0.005 <= gtop.mean() <= 0.045
            no_cross = bool((top_bb[:n] < top_st[:n]).all())
            print(
                f"G6    gap(5.12M) in [0.005,0.045] AND no crossing       "
                f"{gtop.mean():+.4f}  cross {n - int((top_bb[:n] < top_st[:n]).sum())}/{n}  "
                f"{verdict(in_band and no_cross)}"
            )
            print(f"      blackbox {top_bb[:n].mean():.4f} vs structured {top_st[:n].mean():.4f} on {n} seed(s)")
        else:
            print("G6    incomplete -- extrapolation arm still running")
    else:
        print("G6    incomplete -- extrapolation arm still running")

    d7 = abs(bb64[80_000].mean() - PROBE54_PUBLISHED)
    print(f"G7    blackbox_w64@80k within +/-0.05 of probe54 0.6219    {bb64[80_000].mean():.4f} (d {d7:.4f}) {verdict(d7 <= 0.05)}")


if __name__ == "__main__":
    main()
