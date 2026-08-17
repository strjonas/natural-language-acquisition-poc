#!/usr/bin/env python3
"""Recalculate the public Probe64 headline from committed JSON artifacts.

Probe64 asks whether there is a fact about an organism's own body that changes
what it says and that reading its own state perfectly does not supply. The two
contrasts this prints are the whole result:

    recursive - state_oracle    what a learned rate belief buys over knowing
                                exactly where the body is
    individual - state_oracle   how much was there to buy, between two oracles

and the behavioural negative that bounds them: a speaker saying a *random* size
word against one saying the right one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev, stdev

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "organism" / "probe64_portion_request"
TREATMENT = RUN_DIR / "treatment.json"
T_CRITICAL_95_DF4 = 2.776

TIERS = ("population", "snap", "state_oracle", "recursive", "individual", "oracle")


def _summary(values: list[float]) -> tuple[float, float]:
    # Historical result records use NumPy's default ddof=0 across seed blocks.
    return mean(values), pstdev(values)


def _paired_ci(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    half_width = T_CRITICAL_95_DF4 * stdev(values) / math.sqrt(len(values))
    return center, center - half_width, center + half_width


def main() -> None:
    treatment = json.loads(TREATMENT.read_text())
    rows = treatment["rows"]["metabolic"]
    if len(rows) != 5:
        raise ValueError(f"Expected five treatment seed blocks, found {len(rows)}")

    accuracy = {
        tier: [row["size_accuracy"][tier] for row in rows] for tier in TIERS
    }
    discriminating = [row["discriminating_share"] for row in rows]

    print("Probe64 public artifact summary (5 seed blocks x 40 lives)")
    print(f"the species answer differs from the truth on "
          f"{mean(discriminating):.1%} of scored ticks\n")
    print("tier            size accuracy (mean +/- SD*)")
    for tier in TIERS:
        tier_mean, tier_sd = _summary(accuracy[tier])
        print(f"{tier:<15} {tier_mean:.4f} +/- {tier_sd:.4f}")

    print("\npaired per-seed contrasts, 95% CI")
    for label, better, worse in (
        ("a learned rate over a perfect state", "recursive", "state_oracle"),
        ("a true rate over a perfect state", "individual", "state_oracle"),
        ("a self-model over the same evidence", "recursive", "snap"),
    ):
        delta = [b - w for b, w in zip(accuracy[better], accuracy[worse])]
        center, low, high = _paired_ci(delta)
        print(f"  {label:<38} {center:+.4f}  [{low:+.4f}, {high:+.4f}]")

    holdout = treatment["rows"]["holdout"]
    print("\nfirst use of a combination never uttered")
    for mode in ("factored", "tabular"):
        values = [row["first_use_accuracy"][mode] for row in holdout]
        value_mean, value_sd = _summary(values)
        print(f"  {mode:<10} {value_mean:.4f} +/- {value_sd:.4f}")

    print("\nbehaviour under the ration, need pinned at oracle (never gated)")
    behaviour = treatment["rows"]["behaviour"]
    controls = [entry["control"] for entry in behaviour[0]["speakers"]]
    for control in controls:
        values = [
            next(
                entry["survival"]
                for entry in row["speakers"]
                if entry["control"] == control
            )
            for row in behaviour
        ]
        value_mean, value_sd = _summary(values)
        print(f"  {str(control or 'sized'):<13} {value_mean:.4f} +/- {value_sd:.4f}")

    gates = treatment["gates"]
    print("\nlocked gates")
    for name, verdict in gates.items():
        if name == "all_pass":
            continue
        mark = "PASS" if verdict["pass"] else "FAIL"
        print(f"  {mark}  {name}: {verdict['passing']}/{len(verdict['per_seed'])}")
    print(f"  all_pass = {gates['all_pass']}")
    print("\n* population SD across five seed blocks, matching the result records.")


if __name__ == "__main__":
    main()
