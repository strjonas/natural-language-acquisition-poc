"""Caregiver world v2: overfilling makes the body sick, and a held-out word pair.

See docs/decisions/2026-11-21-caregiver-v2-sickness-and-held-out-pair.md
(design and pass rules locked before the first v2 survey).

v1-size with two changes. A delivery that would take a level above 1.0 leaves
it at ``1 - HARM * overflow``, and there is no per-life ration. And the
convention may be ``shared`` by every identity, with a held-out pair
(channel ``HOLD_CHANNEL``, size 0.6) that training bodies rarely need and
held-out bodies do.

    PYTHONPATH=src .venv/bin/python -m homesocial.organism.caregiver_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.organism import active_self
from homesocial.organism import caregiver_v1 as cv
from homesocial.organism.vec_body import BASE_RATES, CHANNELS, LIVES, bodies_from_identities

HARM = 2.0
CONVENTIONS = ("private", "shared")
HOLD_CHANNEL, HOLD_SIZE = 0, 2          # the held-out pair: channel 0 with 0.6
TRAIN_FACTOR = (0.4, 0.7)               # channel 0's rate factor in training bodies (shared convention)
TEST_FACTOR = (1.3, 1.6)                # ... and in held-out bodies
SURVEY_BASE = 6_500_000_000             # development; above speaking_self's band (6.4B)
HOLD_CELLS = ((3, 2), (4, 2))  # iteration 2: cells where the pair cannot be substituted
HOLD_BLOCKS = 3
ARMS = (*cv.arms("size"), "no_held_out_pair")
SHARED_WORDS = np.array([0, 1, 2, 3, 4, -1])  # word w names channel w; word 5 names nothing


class V2Body(cv.V1Body):
    """v1-size with sickness from overfilling and no ration."""

    harm = HARM
    speaker_waste = 0.0  # the spill penalty is the body's own harm, already in the projection

    def __init__(self, bodies: cv.V1Bodies, *, ration: bool = False, **kwargs) -> None:
        self.ration = ration
        if ration:  # v2r: the per-life ration of v1-size, and reference speakers count spilled ration again
            self.speaker_waste = cv.WASTE
        super().__init__(bodies, "size", **kwargs)

    def _on_birth(self, index: np.ndarray) -> None:
        super()._on_birth(index)
        if not self.ration:
            self.left[index] = np.inf

    def _arrivals(self, index, tick, levels, run) -> None:
        for p in range(cv.PENDING):
            landed = np.flatnonzero(run & (self.due_tick[index, p] == tick))
            if len(landed):
                who = index[landed]
                channel = self.due_channel[who, p]
                self.landed.append((who, channel, self.due_amount[who, p].copy(), tick[landed]))
                raw = levels[landed, channel] + self.due_amount[who, p]
                over = np.maximum(raw - 1.0, 0.0)
                self.wasted[who] += over
                levels[landed, channel] = np.where(over > 0, 1.0 - self.harm * over, raw)
                self.due_tick[who, p] = -1


def share(bodies: cv.V1Bodies) -> cv.V1Bodies:
    """Give every identity the shared convention (in place)."""

    bodies.word_channel[:] = SHARED_WORDS
    bodies.channel_word[:] = np.arange(CHANNELS)
    bodies.mod_meaning[:] = np.arange(3)
    bodies.meaning_mod[:] = np.arange(3)
    return bodies


def hold_bodies(identities, ks, socials, factor) -> cv.V1Bodies:
    """Bodies whose channel 0 is active and social, with channel 0's rate
    factor drawn from ``factor``. Identities without an active channel 0 are
    skipped, so the result can be shorter than ``identities``."""

    keep = [i for i, (ident, k) in enumerate(zip(identities, ks))
            if bodies_from_identities([ident], [k]).active[0, HOLD_CHANNEL]]
    idents = [identities[i] for i in keep]
    kk = [ks[i] for i in keep]
    ss = [socials[i] for i in keep]
    b = cv.v1_bodies(idents, kk, [0] * len(idents))
    for i, (ident, social) in enumerate(zip(idents, ss)):
        others = [int(c) for c in np.flatnonzero(b.active[i]) if c != HOLD_CHANNEL]
        extra = Random(active_self.domain_seed(ident, "hold-social")).sample(others, social - 1)
        b.blocked[i, [HOLD_CHANNEL, *extra]] = True
        f = Random(active_self.domain_seed(ident, "hold-rate")).uniform(*factor)
        b.rates[i, HOLD_CHANNEL] = BASE_RATES[HOLD_CHANNEL] * f
    b.ration[:] = cv.RATION_FACTOR * cv.species_need(b.blocked)  # social resources changed after construction
    return share(b)


TRAIN_BIRTH = (0.75, 0.85)


def restrict_hold_rate(bodies: cv.V1Bodies, rng: np.random.Generator) -> cv.V1Bodies:
    """Training-like bodies (iteration 1): channel 0's rate factor is drawn
    from TRAIN_FACTOR, it is born at 0.75 or 0.85, and when it is social it is
    the only social resource. (In place.)"""

    n, live = bodies.n, bodies.active[:, HOLD_CHANNEL]
    f = rng.uniform(*TRAIN_FACTOR, n)
    bodies.rates[:, HOLD_CHANNEL] = np.where(live, BASE_RATES[HOLD_CHANNEL] * f, 0.0)
    born = rng.choice(np.asarray(TRAIN_BIRTH), (n, LIVES))
    bodies.birth[:, :, HOLD_CHANNEL] = np.where(live[:, None], born, bodies.birth[:, :, HOLD_CHANNEL])
    alone = bodies.blocked[:, HOLD_CHANNEL].copy()  # a copy: the next line edits the array it was read from
    bodies.blocked[alone] = False
    bodies.blocked[alone, HOLD_CHANNEL] = True
    bodies.ration[:] = cv.RATION_FACTOR * cv.species_need(bodies.blocked)
    return bodies


def held_out_forbid(n: int) -> np.ndarray:
    forbid = np.zeros((n, CHANNELS, 3), dtype=bool)
    forbid[:, HOLD_CHANNEL, HOLD_SIZE] = True
    return forbid


def run_arm(bodies: cv.V1Bodies, arm: str, ration: bool = False) -> dict:
    env = V2Body(bodies, ration=ration)
    reckoner = None
    if arm in cv.RECKONING_ARMS:
        reckoner = cv.Reckoner(env, np.where(bodies.active, BASE_RATES, 0.0) if arm == "species" else bodies.rates)
    forbid = held_out_forbid(bodies.n) if arm == "no_held_out_pair" else None
    pair = np.zeros(bodies.n)
    hold_requests = np.zeros(bodies.n)
    rows = np.arange(bodies.n)
    while not env.finished.all():
        act, speech = cv.speaker(env, "composed" if arm == "no_held_out_pair" else arm,
                                 None if reckoner is None else reckoner.est, forbid)
        spoken = (speech >= 0) & env.present()
        c = bodies.word_channel[rows, np.maximum(speech, 0) // 3]
        m = bodies.mod_meaning[rows, np.maximum(speech, 0) % 3]
        hold_requests += spoken & (c == HOLD_CHANNEL)
        pair += spoken & (c == HOLD_CHANNEL) & (m == HOLD_SIZE)
        env.step(act, speech)
        if reckoner is not None:
            reckoner.update(env)
    return {"survived": env.survived.copy(), "requests": env.requests.copy(), "wasted": env.wasted.copy(),
            "hold_requests": hold_requests, "held_out_pair": pair}


def survey(*, blocks: int = cv.BLOCKS, identities: int = cv.IDENTITIES, cells_=cv.CELLS, ration: bool = False) -> dict:
    """The v1 criteria in v2 (unrestricted family, private convention)."""

    names = cv.meanings("size")
    arm_list = cv.arms("size")
    out = {"scenario": "caregiver-v2r-size" if ration else "caregiver-v2-size",
           "status": "development ceiling survey; reference speakers",
           "harm": HARM, "blocks": blocks, "identities_per_block": identities, "survey_base": SURVEY_BASE,
           "arms": list(arm_list), "cells": {}}
    for k, social in cells_:
        block_of = np.repeat(np.arange(blocks), identities)
        idents = [SURVEY_BASE + bl * cv.BLOCK_STRIDE + k * cv.K_STRIDE + social * cv.SOCIAL_STRIDE + i * cv.IDENTITY_STRIDE
                  for bl in range(blocks) for i in range(identities)]
        bodies = cv.v1_bodies(idents, [k] * len(idents), [social] * len(idents))
        runs = {arm: run_arm(bodies, arm, ration) for arm in arm_list}
        by_block = {arm: [float(runs[arm]["survived"][block_of == bl].mean()) for bl in range(blocks)]
                    for arm in arm_list}
        singles = [f"single_{m}" for m in names]
        best_single = max(singles, key=lambda a: np.mean(by_block[a]))
        asked = runs["composed"]["requests"].sum(0)
        out["cells"][f"K{k}_social{social}"] = {
            "k": k, "social": social,
            "survival": {arm: float(np.mean(by_block[arm])) for arm in arm_list},
            "survival_by_block": by_block, "best_single": best_single,
            "composed_minus": {arm: active_self._interval([c - o for c, o in zip(by_block["composed"], by_block[arm])])
                               for arm in arm_list if arm != "composed"},
            "reckoned_minus_species": active_self._interval(
                [r - s for r, s in zip(by_block["reckoned"], by_block["species"])]),
            "composed_meaning_share": {m: float(a / max(asked.sum(), 1)) for m, a in zip(names, asked)},
            "wasted_per_life": {arm: float(runs[arm]["wasted"].mean() / LIVES) for arm in arm_list},
        }
    out["criteria"] = cv.criteria(out)
    return out


TRAIN_HOLD_CELLS = ((2, 1), (3, 1), (4, 1))


def hold_survey(*, blocks: int = HOLD_BLOCKS, identities: int = cv.IDENTITIES, ration: bool = False) -> dict:
    """H1-H3 of the decision record: is (channel 0, 0.6) rare in training-like
    bodies, needed in held-out bodies, and worth survival there?"""

    out = {"scenario": f"caregiver-{'v2r' if ration else 'v2'}-size shared, held-out pair",
           "pair": [HOLD_CHANNEL, cv.SIZES[HOLD_SIZE]], "train_factor": TRAIN_FACTOR, "train_birth": TRAIN_BIRTH,
           "test_factor": TEST_FACTOR, "cells": {}}
    rng = np.random.default_rng(20261121)
    for k, social in sorted(set(HOLD_CELLS) | set(TRAIN_HOLD_CELLS)):
        cell = {}
        splits = [s for s, cells in (("train", TRAIN_HOLD_CELLS), ("test", HOLD_CELLS)) if (k, social) in cells]
        for split in splits:
            factor = TRAIN_FACTOR if split == "train" else TEST_FACTOR
            per_block, pair, hold = {"composed": [], "no_held_out_pair": []}, 0.0, 0.0
            for bl in range(blocks):
                base = SURVEY_BASE + 50_000_000 + bl * cv.BLOCK_STRIDE + k * cv.K_STRIDE + social * cv.SOCIAL_STRIDE
                idents = [base + i * cv.IDENTITY_STRIDE for i in range(3 * identities)]
                b = hold_bodies(idents, [k] * len(idents), [social] * len(idents), factor)
                b = b.take(np.arange(min(identities, b.n)))
                if split == "train":
                    restrict_hold_rate(b, rng)
                for arm in per_block:
                    r = run_arm(b, arm, ration)
                    per_block[arm].append(float(r["survived"].mean()))
                    if arm == "composed":
                        pair += r["held_out_pair"].sum()
                        hold += r["hold_requests"].sum()
            cell[split] = {"survival": {a: float(np.mean(v)) for a, v in per_block.items()},
                           "held_out_pair_share": float(pair / max(hold, 1)),
                           "composed_minus_no_pair": active_self._interval(
                               [c - o for c, o in zip(per_block["composed"], per_block["no_held_out_pair"])])}
        out["cells"][f"K{k}_social{social}"] = cell
    train = [c["train"] for c in out["cells"].values() if "train" in c]
    test = [c["test"] for c in out["cells"].values() if "test" in c]
    out["criteria"] = {
        "H1_rare_in_training": all(c["held_out_pair_share"] < 0.02 for c in train),
        "H2_needed_when_held_out": all(c["held_out_pair_share"] >= 0.15 for c in test),
        "H3_worth_survival": all(c["composed_minus_no_pair"]["mean"] >= 0.05
                                 and c["composed_minus_no_pair"]["low"] > 0 for c in test),
    }
    out["criteria"]["pass"] = all(out["criteria"].values())
    return out


def hold_table(result: dict) -> str:
    lines = ["held-out pair (channel 0, 0.6), reference composed speaker, shared convention"]
    for name, c in result["cells"].items():
        for split in [s for s in ("train", "test") if s in c]:
            d = c[split]
            i = d["composed_minus_no_pair"]
            lines.append(f"  {name:<11} {split:<5} pair share {d['held_out_pair_share']:.3f}  survival "
                         f"{d['survival']['composed']:.3f} vs no pair {d['survival']['no_held_out_pair']:.3f}  "
                         f"diff {i['mean']:+.3f} [{i['low']:+.3f},{i['high']:+.3f}]")
    lines.append("criteria: " + ", ".join(f"{k}={v}" for k, v in result["criteria"].items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("all", "survey", "hold"), default="all")
    parser.add_argument("--ration", action="store_true", help="v2r: keep v1-size's per-life ration")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/organism/caregiver_v2"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_v2r" if args.ration else ""
    if args.part in ("all", "survey"):
        result = survey(ration=args.ration)
        (args.out_dir / f"ceiling_survey{tag}.json").write_text(json.dumps(result, indent=1))
        print(cv.table(result), flush=True)
    if args.part in ("all", "hold"):
        result = hold_survey(ration=args.ration)
        (args.out_dir / f"held_out_pair{tag}_iter1.json").write_text(json.dumps(result, indent=1))
        print(hold_table(result), flush=True)


if __name__ == "__main__":
    main()
