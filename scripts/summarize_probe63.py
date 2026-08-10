#!/usr/bin/env python3
"""Recalculate the public Probe63 headline from committed JSON artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev, stdev


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "organism" / "probe63_individual_self"
TREATMENT = RUN_DIR / "treatment.json"
CLOSED_LOOP = RUN_DIR / "closed_loop.json"
T_CRITICAL_95_DF4 = 2.776


def _load(path: Path):
    return json.loads(path.read_text())


def _summary(values: list[float]) -> tuple[float, float]:
    # Historical result records use NumPy's default ddof=0 across seed blocks.
    return mean(values), pstdev(values)


def _paired_ci(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    half_width = T_CRITICAL_95_DF4 * stdev(values) / math.sqrt(len(values))
    return center, center - half_width, center + half_width


def main() -> None:
    treatment = _load(TREATMENT)
    rows = treatment["rows"]["metabolic"]
    if len(rows) != 5:
        raise ValueError(f"Expected five treatment seed blocks, found {len(rows)}")

    tiers = ("population", "snap", "recursive")
    errors = {tier: [row["body_error"][tier] for row in rows] for tier in tiers}
    accuracy = {
        tier: [row["argmin_accuracy"][tier] for row in rows] for tier in tiers
    }

    error_delta = [s - r for s, r in zip(errors["snap"], errors["recursive"])]
    accuracy_delta = [
        r - s for s, r in zip(accuracy["snap"], accuracy["recursive"])
    ]
    err_center, err_low, err_high = _paired_ci(error_delta)
    acc_center, acc_low, acc_high = _paired_ci(accuracy_delta)

    print("Probe63 public artifact summary (5 seed blocks x 40 lives)")
    print("tier          body error (mean +/- SD*)  report accuracy (mean +/- SD*)")
    for tier in tiers:
        err_mean, err_sd = _summary(errors[tier])
        acc_mean, acc_sd = _summary(accuracy[tier])
        print(
            f"{tier:<13} {err_mean:.4f} +/- {err_sd:.4f}"
            f"            {acc_mean:.4f} +/- {acc_sd:.4f}"
        )
    print("* Across-seed population SD (ddof=0), matching the result record.")

    relative_reduction = err_center / mean(errors["snap"])
    print("\nRLS versus identical-evidence snap control")
    print(f"body-error reduction: {relative_reduction:.1%}")
    print(
        "paired absolute error improvement: "
        f"{err_center:.4f} (95% t CI {err_low:.4f} to {err_high:.4f})"
    )
    print(
        "paired report-accuracy improvement: "
        f"{acc_center * 100:.1f} points "
        f"(95% t CI {acc_low * 100:.1f} to {acc_high * 100:.1f})"
    )

    recursive_gates = treatment["gates"]["recursive"]
    gate_names = [name for name in recursive_gates if name != "all_pass"]
    if not recursive_gates["all_pass"]:
        raise ValueError("Committed artifact does not pass the recursive-arm gates")
    print(f"locked gates: PASS ({len(gate_names)}/{len(gate_names)})")

    closed_loop = {row["tier"]: row for row in _load(CLOSED_LOOP)}
    snap_survival = closed_loop["snap"]["survival"]
    rls_survival = closed_loop["recursive"]["survival"]
    snap_fidelity = closed_loop["snap"]["fidelity"]
    rls_fidelity = closed_loop["recursive"]["fidelity"]
    survival_diffs = [
        r["survival"] - s["survival"]
        for s, r in zip(
            closed_loop["snap"]["per_seed"],
            closed_loop["recursive"]["per_seed"],
        )
    ]
    _, survival_low, survival_high = _paired_ci(survival_diffs)
    print("\nClosed loop (context; not a locked gate)")
    print(f"survival: {snap_survival:.3f} -> {rls_survival:.3f}")
    print(f"paired survival-difference 95% t CI: {survival_low:.3f} to {survival_high:.3f}")
    print(f"report fidelity: {snap_fidelity:.3f} -> {rls_fidelity:.3f}")


if __name__ == "__main__":
    main()
