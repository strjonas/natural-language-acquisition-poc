#!/usr/bin/env python3
"""Recalculate the public Probe65 headline from committed JSON artifacts.

Probe65 asks what a self-model is *worth*, as a function of how far ahead the
world makes an organism think. Three things this prints are the whole result:

    the crossover        `individual - state_oracle` at every lag. Negative when
                         the caregiver answers at once, positive once the lag is
                         long enough, because state error enters a prediction
                         once and rate error enters it multiplied by the horizon.
    recursive - state_oracle
                         what a *learned* rate belief buys over knowing exactly
                         where the body is, at the operating lag.
    identical present, divergent future
                         whether the word follows the future when the present
                         cannot tell two bodies apart -- and the structural zero
                         of a reporter that reads only the present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev, stdev

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "organism" / "probe65_future_request"
TREATMENT = RUN_DIR / "treatment.json"
T_CRITICAL_95_DF4 = 2.776

ARMS = (
    "population",
    "snap",
    "myopic",
    "state_oracle",
    "recursive",
    "individual",
    "oracle",
)


def _summary(values: list[float]) -> tuple[float, float]:
    # Historical result records use NumPy's default ddof=0 across seed blocks.
    return mean(values), pstdev(values)


def _paired_ci(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    half_width = T_CRITICAL_95_DF4 * stdev(values) / math.sqrt(len(values))
    return center, center - half_width, center + half_width


def main() -> None:
    treatment = json.loads(TREATMENT.read_text())
    rows = treatment["rows"]
    delay = treatment["operating_delay"]
    blocks = rows["treatment"]
    if len(blocks) != 5:
        raise ValueError(f"Expected five treatment seed blocks, found {len(blocks)}")

    def accuracy(condition: str) -> dict[str, list[float]]:
        return {
            arm: [row["need_accuracy"][arm] for row in rows[condition]] for arm in ARMS
        }

    treated = accuracy("treatment")
    prompt = accuracy("prompt")
    moved = [row["future_moved_share"] for row in blocks]

    print(f"Probe65 public artifact summary (5 seed blocks x 40 lives, lag {delay})")
    print(
        "the future disagrees with the present on "
        f"{mean(moved):.1%} of scored ticks\n"
    )
    print(f"arm             lag {delay:<3}          lag 0")
    for arm in ARMS:
        slow_mean, slow_sd = _summary(treated[arm])
        fast_mean, fast_sd = _summary(prompt[arm])
        print(
            f"{arm:<15} {slow_mean:.4f} +/- {slow_sd:.4f}   "
            f"{fast_mean:.4f} +/- {fast_sd:.4f}"
        )

    print(f"\npaired per-seed contrasts at lag {delay}, 95% CI")
    for label, better, worse in (
        ("a learned rate over a perfect state", "recursive", "state_oracle"),
        ("a true rate over a perfect state", "individual", "state_oracle"),
        ("a self-model over the same evidence", "recursive", "snap"),
        ("looking ahead at all", "state_oracle", "myopic"),
    ):
        delta = [b - w for b, w in zip(treated[better], treated[worse])]
        center, low, high = _paired_ci(delta)
        print(f"  {label:<38} {center:+.4f}  [{low:+.4f}, {high:+.4f}]")

    print("\nthe same contrasts with a caregiver that answers at once (the falsifier)")
    for label, better, worse in (
        ("a learned rate over a perfect state", "recursive", "state_oracle"),
        ("a true rate over a perfect state", "individual", "state_oracle"),
    ):
        delta = [b - w for b, w in zip(prompt[better], prompt[worse])]
        center, low, high = _paired_ci(delta)
        print(f"  {label:<38} {center:+.4f}  [{low:+.4f}, {high:+.4f}]")

    print("\nthe crossover: what you are minus where you are, by lag")
    curve = rows["horizon"]
    lags = [entry["help_delay"] for entry in curve[0]["delays"]]
    print("  lag   individual - state_oracle   recursive - state_oracle")
    for index, lag in enumerate(lags):
        gap = {
            key: [
                seed["delays"][index]["need_accuracy"][key]
                - seed["delays"][index]["need_accuracy"]["state_oracle"]
                for seed in curve
            ]
            for key in ("individual", "recursive")
        }
        print(
            f"  {lag:>3}   {mean(gap['individual']):+.4f} +/- "
            f"{pstdev(gap['individual']):.4f}          "
            f"{mean(gap['recursive']):+.4f} +/- {pstdev(gap['recursive']):.4f}"
        )

    print("\nidentical present, divergent future")
    divergence = rows["divergence"]
    pairs = [row["divergence"]["pairs"] for row in divergence]
    print(f"  {mean(pairs):.0f} matched pairs per seed")
    for arm in ARMS:
        diverged = [row["divergence"]["divergence"][arm] for row in divergence]
        correct = [row["divergence"]["correct_divergence"][arm] for row in divergence]
        print(
            f"  {arm:<15} diverges {mean(diverged):.4f} +/- {pstdev(diverged):.4f}   "
            f"correctly {mean(correct):.4f} +/- {pstdev(correct):.4f}"
        )

    print("\nsurvival, each arm speaking for itself")
    behaviour = rows["behaviour"]
    for arm in ARMS:
        values = [
            next(e["survival"] for e in row["arms"] if e["arm"] == arm)
            for row in behaviour
        ]
        value_mean, value_sd = _summary(values)
        print(f"  {arm:<15} {value_mean:.4f} +/- {value_sd:.4f}")

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
