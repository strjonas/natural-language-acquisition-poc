"""Why probe67's uncertainty does or does not reach a word.

The ceiling survey answers whether self-uncertainty buys anything. This answers
the mechanistic question underneath it, which is the one that makes a negative
worth as much as a positive: **how often can the rate posterior reach a decision
at all, and when it cannot, is that because the width is small or because the
decision was never close?**

Probe62's closure is valuable because it carried numbers of exactly this kind --
"the filter saturates at the ceiling on about 9.7% of ticks", "the bias is flat
at +0.051 across eight 50-tick blocks". A closure without them is an opinion.

The rule picks the word with the largest ``E[min(next body)]``. Uncertainty
enters through Jensen: a wider posterior on need ``i`` lowers every score, and
lowers them least for the word that boosts ``i``. So the posterior can only
change what is said when the correction it applies is larger than the gap
between the top two words. This script measures both sides of that inequality on
real lives, at every lag, and reports the distribution rather than the mean --
because a mechanism that acts on 2% of ticks and a mechanism that acts on none
are different findings with the same average.

    PYTHONPATH=src .venv/bin/python scripts/diagnose_probe67.py --lives 8
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
    DELAY_SWEEP,
    answer_horizon,
    true_need,
    true_rates,
)
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

# A seed band of its own: this is a diagnostic and must not be read off the
# lives the survey scores.
DIAGNOSTIC_SEED_BASE = 1_180_000_000


def diagnose(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    delay: int,
    lives: int,
    seed_base: int,
) -> dict[str, object]:
    kwargs = _world_kwargs(delay, 0.03)
    report = replace(organism.report, **kwargs)
    horizon = answer_horizon(delay)

    gaps: list[float] = []
    corrections: list[float] = []
    widths: list[float] = []
    reach: list[bool] = []
    flipped = 0
    ticks = 0
    # The same counts restricted to ticks where the decision is worth at least
    # one tick of metabolism to get right. A mechanism that only ever fires on
    # decisions that do not matter is not a mechanism, and the raw flip rate
    # cannot tell the two apart -- a long horizon compresses the three words
    # together, so flips become both easier and cheaper at the same time.
    stakes_ticks = 0
    stakes_flipped = 0
    # The correction spread restricted to consequential ticks. This is the
    # quantity the closure actually rests on: a spread that is large overall but
    # small *where the decision matters* is a mechanism that cannot reach one,
    # and the unconditional spread cannot tell those apart.
    stakes_corrections: list[float] = []

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
                ticks += 1
                point = arms["point"].scores(now, fraction)
                wide = arms["uncertain"].scores(now, fraction)
                sigma = arms["uncertain"].sigma(fraction)

                ordered = np.sort(point)
                gap = float(ordered[-1] - ordered[-2])
                # How differently the posterior treats the three words: the
                # spread of the per-word Jensen corrections. A correction that is
                # the same for all three cannot reorder anything, which is
                # exactly what the ``flat`` control is built to show.
                shift = point - wide
                correction = float(shift.max() - shift.min())
                gaps.append(gap)
                corrections.append(correction)
                widths.append(float((sigma * horizon).max()))
                reach.append(correction > gap)
                flip = int(int(np.argmax(point)) != int(np.argmax(wide)))
                flipped += flip
                if gap >= CONSEQUENTIAL_MARGIN:
                    stakes_ticks += 1
                    stakes_flipped += flip
                    stakes_corrections.append(correction)

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

    gap = np.asarray(gaps)
    correction = np.asarray(corrections)
    width = np.asarray(widths)
    return {
        "help_delay": delay,
        "horizon": horizon,
        "ticks": ticks,
        "flip_share": flipped / max(1, ticks),
        "consequential_share": stakes_ticks / max(1, ticks),
        "consequential_flip_share": stakes_flipped / max(1, stakes_ticks),
        "p90_correction_consequential": float(
            np.percentile(stakes_corrections, 90)
        )
        if stakes_corrections
        else 0.0,
        "median_gap_consequential": float(
            np.median([g for g in gaps if g >= CONSEQUENTIAL_MARGIN])
        )
        if any(g >= CONSEQUENTIAL_MARGIN for g in gaps)
        else 0.0,
        "reachable_share": float(np.mean(reach)) if reach else 0.0,
        "median_gap": float(np.median(gap)) if len(gap) else 0.0,
        "median_correction": float(np.median(correction)) if len(correction) else 0.0,
        "p90_correction": float(np.percentile(correction, 90)) if len(correction) else 0.0,
        "median_projected_width": float(np.median(width)) if len(width) else 0.0,
        # The share of ticks whose decision is close enough that a correction of
        # this size *could* matter, whatever the correction actually was.
        "near_tie_share": float(np.mean(gap < float(np.median(correction))))
        if len(gap)
        else 0.0,
    }


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
    parser.add_argument(
        "--delays",
        default=",".join(str(d) for d in DELAY_SWEEP),
        help="comma-separated lags to diagnose",
    )
    parser.add_argument("--seed-base", type=int, default=DIAGNOSTIC_SEED_BASE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    assert args.seed_base != SURVEY_SEED_BASE, "diagnose off the survey's own lives"
    model, organism = load_organism_checkpoint(Path(args.checkpoint))

    rows = [
        diagnose(
            model, organism, delay=delay, lives=args.lives, seed_base=args.seed_base
        )
        for delay in tuple(
            int(d) for d in args.delays.split(",") if d.strip()
        )
    ]

    print("| lag | ticks | sigma*H | p90 corr ALL | p90 corr CONSEQ | "
          "median gap CONSEQ | flips ALL | flips CONSEQ |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['help_delay']} | {row['ticks']} | "
            f"{row['median_projected_width']:.5f} | "
            f"{row['p90_correction']:.6f} | "
            f"{row['p90_correction_consequential']:.6f} | "
            f"{row['median_gap_consequential']:.5f} | "
            f"{row['flip_share']:.4f} | "
            f"{row['consequential_flip_share']:.4f} |"
        )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
