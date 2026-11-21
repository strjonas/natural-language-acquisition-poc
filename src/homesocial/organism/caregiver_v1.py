"""Caregiver world v1: three candidate worlds for a second word, and their
ceiling survey. Development instrument, not a learner.

See docs/decisions/2026-11-21-caregiver-v1-ceiling-survey.md (design and the
pass rule, locked before the first survey ran).

``caregiver-v0`` taught a one-slot vocabulary: levels cap at 1.0, nothing
charges for asking too much, and the caregiver could be asked at any decision,
so a speaker could always wait and ask "large". Here the caregiver visits every
``VISIT`` decisions, the organism may act and speak at the same decision, and
the second word either sets an amount drawn from a per-life ration (``size``),
sets when a fixed portion lands, also from the ration (``when``), or names a
second resource to share the visit's basket (``basket``).

    PYTHONPATH=src .venv/bin/python -m homesocial.organism.caregiver_v1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.organism import active_self
from homesocial.organism.vec_body import (
    ACTUATORS, BASE_RATES, CHANNELS, LIFE_TICKS, LIVES, PHYSICS, WINDOW, Bodies, VecBody, bodies_from_identities,
)

VARIANTS = ("size", "when", "basket")
WORDS = 6                       # resource words: five channels and one that names nothing
VISIT = 4                       # the caregiver is present at every fourth decision of a life
SIZES = (0.2, 0.4, 0.6)         # v1-size meanings
DELAYS = (1, 13, 25)            # v1-when meanings (ticks until a 0.6 portion lands)
WHEN_PORTION = 0.6
BASKET = (0.4, 0.2)             # v1-basket: first and second named resource; one resource gets both
RATION_FACTOR = 1.2
SHOCK_DRAIN = PHYSICS.shock_probability * PHYSICS.shock_size / CHANNELS  # expected loss per tick per channel
MARGIN = 0.1
WASTE = 1.5          # reference speakers: a spilled unit of ration costs 1.5 absorbed units
MIN_ABSORBED = 0.1   # reference speakers stay silent when a request would add less than this
HORIZON = VISIT * WINDOW + 1 + WINDOW  # next visit's delivery plus one window
PENDING = 8

CELLS = ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 2))
BLOCKS, IDENTITIES = 5, 100
SURVEY_BASE = 6_300_000_000  # development; above v0 (6.2B) and learned_self (6.1B)
BLOCK_STRIDE, K_STRIDE, SOCIAL_STRIDE, IDENTITY_STRIDE = 2_000_000, 100_000, 10_000, 100


def meanings(variant: str) -> tuple[str, ...]:
    return {"size": tuple(f"{s:.1f}" for s in SIZES), "when": tuple(f"d{d}" for d in DELAYS),
            "basket": ("one", "split")}[variant]


def modifier_words(variant: str) -> int:
    return WORDS if variant == "basket" else len(SIZES)


def utterances(variant: str) -> int:
    """Number of distinct two-word utterances (silence is -1)."""

    return WORDS * modifier_words(variant)


def cells(variant: str):
    return tuple(c for c in CELLS if variant != "basket" or c[1] >= 2)


def arms(variant: str) -> tuple[str, ...]:
    return ("silent", "composed", *(f"single_{m}" for m in meanings(variant)), "reckoned", "species", "scrambled")


RECKONING_ARMS = ("reckoned", "species")


def identity_seed(block: int, k: int, social: int, identity: int) -> int:
    if not (0 <= block < BLOCKS and 1 <= k <= CHANNELS and 0 <= social < 10 and 0 <= identity < IDENTITIES):
        raise ValueError("identity outside the survey band")
    return SURVEY_BASE + block * BLOCK_STRIDE + k * K_STRIDE + social * SOCIAL_STRIDE + identity * IDENTITY_STRIDE


@dataclass
class V1Bodies(Bodies):
    word_channel: np.ndarray  # (N, WORDS) channel each resource word names; -1 names nothing
    channel_word: np.ndarray  # (N, CHANNELS) its inverse
    mod_meaning: np.ndarray   # (N, 3) meaning index of each modifier word (size / when)
    meaning_mod: np.ndarray   # (N, 3) its inverse
    ration: np.ndarray        # (N,) caregiver's ration per life (size / when)


def species_need(blocked: np.ndarray) -> np.ndarray:
    return ((BASE_RATES + SHOCK_DRAIN) * LIFE_TICKS * blocked).sum(1)


def v1_bodies(identities, ks, socials) -> V1Bodies:
    """Probe80 bodies with ``social`` resources only a caregiver can restore
    and a private convention, both from the identity's own seed streams."""

    base = bodies_from_identities(identities, ks)
    n = base.n
    word_channel = -np.ones((n, WORDS), dtype=np.int64)
    channel_word = np.zeros((n, CHANNELS), dtype=np.int64)
    mod_meaning = np.zeros((n, 3), dtype=np.int64)
    for i, (ident, social) in enumerate(zip(identities, socials, strict=True)):
        active = [int(c) for c in np.flatnonzero(base.active[i])]
        if not 0 <= social < len(active):
            raise ValueError("a body keeps at least one resource it can restore itself")
        base.blocked[i, Random(active_self.domain_seed(ident, "social")).sample(active, social)] = True
        convention = Random(active_self.domain_seed(ident, "convention-v1"))
        words = convention.sample(range(WORDS), CHANNELS)
        word_channel[i, words] = np.arange(CHANNELS)
        channel_word[i] = words
        mod_meaning[i] = convention.sample(range(3), 3)
    meaning_mod = np.argsort(mod_meaning, 1)
    return V1Bodies(**{name: getattr(base, name) for name in Bodies.__dataclass_fields__},
                    word_channel=word_channel, channel_word=channel_word, mod_meaning=mod_meaning,
                    meaning_mod=meaning_mod, ration=RATION_FACTOR * species_need(base.blocked))


def decode(variant: str, b: V1Bodies, rows: np.ndarray, w1: np.ndarray, w2: np.ndarray, left: np.ndarray):
    """What an utterance asks for: two deliveries ``(channel, amount, delay)``
    (channel -1 = none), the ration charged, and its meaning index."""

    c1 = b.word_channel[rows, w1]
    named = c1 >= 0
    none = -np.ones_like(c1)
    zero = np.zeros(len(rows))
    if variant == "basket":
        c2 = b.word_channel[rows, w2]
        split = named & (c2 >= 0) & (c2 != c1)
        first = np.where(split, BASKET[0], sum(BASKET)) * named
        return ((np.where(named, c1, -1), first, np.ones_like(c1)),
                (np.where(split, c2, -1), np.where(split, BASKET[1], 0.0), np.ones_like(c1)),
                zero, split.astype(np.int64))
    m = b.mod_meaning[rows, w2]
    want = np.asarray(SIZES)[m] if variant == "size" else np.full(len(rows), WHEN_PORTION)
    delay = np.ones_like(c1) if variant == "size" else np.asarray(DELAYS)[m]
    amount = np.where(named, np.minimum(want, left), 0.0)
    return (np.where(named, c1, -1), amount, delay), (none, zero, np.ones_like(c1)), amount, m


class V1Body(VecBody):
    """``VecBody`` with a visiting caregiver. ``step(actuators, speech)``:
    actuators in -1..9 (-1 none); speech -1 (silent) or ``w1 * M + w2``."""

    def __init__(self, bodies: V1Bodies, variant: str, **kwargs) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        n = bodies.n
        self.variant = variant
        self.m = modifier_words(variant)
        self.due_tick = -np.ones((n, PENDING), dtype=np.int64)
        self.due_channel = np.zeros((n, PENDING), dtype=np.int64)
        self.due_amount = np.zeros((n, PENDING))
        self.left = np.zeros(n)
        self.requests = np.zeros((n, len(meanings(variant))), dtype=np.int64)
        self.wasted = np.zeros(n)
        super().__init__(bodies, **kwargs)

    harm = 0.0           # v1: overflow is lost, nothing more
    speaker_waste = WASTE  # reference speakers' penalty per spilled unit (ration worlds)

    def present(self) -> np.ndarray:
        return ~self.finished & (self.decision % VISIT == 0)

    def _on_birth(self, index: np.ndarray) -> None:
        self.due_tick[index] = -1
        self.left[index] = self.b.ration[index] if self.variant != "basket" else np.inf

    def _arrivals(self, index, tick, levels, run) -> None:
        for p in range(PENDING):
            landed = np.flatnonzero(run & (self.due_tick[index, p] == tick))
            if len(landed):
                who = index[landed]
                channel = self.due_channel[who, p]
                self.landed.append((who, channel, self.due_amount[who, p].copy(), tick[landed]))
                raw = levels[landed, channel] + self.due_amount[who, p]
                self.wasted[who] += np.maximum(raw - 1.0, 0.0)
                levels[landed, channel] = np.minimum(raw, 1.0)
                self.due_tick[who, p] = -1

    def _schedule(self, who: np.ndarray, channel: np.ndarray, amount: np.ndarray, delay: np.ndarray) -> None:
        keep = (channel >= 0) & (amount > 0)
        who, channel, amount, delay = who[keep], channel[keep], amount[keep], delay[keep]
        free = self.due_tick[who] < 0
        if not free.any(1).all():
            raise AssertionError("pending deliveries overflow")
        slot = free.argmax(1)
        self.due_tick[who, slot] = self.tick[who] + delay
        self.due_channel[who, slot] = channel
        self.due_amount[who, slot] = amount

    def step(self, actuators: np.ndarray, speech: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.landed = []
        if speech is not None:
            speech = np.asarray(speech)
            heard = np.flatnonzero(self.present() & (speech >= 0))
            w1, w2 = speech[heard] // self.m, speech[heard] % self.m
            first, second, cost, meaning = decode(self.variant, self.b, heard, w1, w2, self.left[heard])
            asked = first[0] >= 0
            self.left[heard] -= cost
            self._schedule(heard, *first)
            self._schedule(heard, *second)
            np.add.at(self.requests, (heard[asked], meaning[asked]), 1)
        return super().step(actuators)


# ---------------------------------------------------------------- speakers
def _candidates(env: V1Body, arm: str):
    """Candidate requests in meaning space: (first channel, meaning or second
    channel). Rows with fewer social resources repeat silence."""

    b = env.b
    social = [np.flatnonzero(row) for row in b.blocked]
    width = max(len(s) for s in social)
    cand = []
    names = meanings(env.variant)
    allowed = range(len(names)) if not arm.startswith("single_") else [names.index(arm[len("single_"):])]
    for j in range(width):
        for k in range(width if env.variant == "basket" else len(names)):
            if env.variant == "basket":
                if (0 if j == k else 1) not in allowed:
                    continue
            elif k not in allowed:
                continue
            cand.append((j, k))
    first = np.full((env.n, len(cand)), -1, dtype=np.int64)
    second = np.full((env.n, len(cand)), -1, dtype=np.int64)
    for u, (j, k) in enumerate(cand):
        for i, s in enumerate(social):
            if j < len(s) and (env.variant != "basket" or k < len(s)):
                first[i, u] = s[j]
                second[i, u] = s[k] if env.variant == "basket" else k
    return first, second


def _project(env: V1Body, rates: np.ndarray, deliveries, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest projected social level over ``HORIZON`` ticks and overflow, for
    each candidate. ``deliveries``: list of (channel, amount, delay) arrays (N, U)."""

    n, u = deliveries[0][0].shape
    levels = np.repeat(levels[:, None, :], u, 1)
    drain = (rates + SHOCK_DRAIN) * env.b.active
    social = env.b.blocked[:, None, :]
    low = np.full((n, u), np.inf)
    overflow = np.zeros((n, u))
    pend = env.due_tick - env.tick[:, None]
    for tau in range(1, HORIZON + 1):
        for p in range(PENDING):
            hit = (env.due_tick[:, p] >= 0) & (pend[:, p] == tau)
            if hit.any():
                levels[hit, :, env.due_channel[hit, p]] += env.due_amount[hit, p][:, None]
        for channel, amount, delay in deliveries:
            hit = (channel >= 0) & (delay == tau)
            if hit.any():
                r, c = np.nonzero(hit)
                np.add.at(levels, (r, c, channel[r, c]), amount[r, c])
        spill = np.maximum(levels - 1.0, 0.0)
        overflow += spill.sum(-1)
        levels = np.clip(levels - spill * (1.0 + env.harm) - drain[:, None, :], 0.0, 1.0)
        low = np.minimum(low, np.where(social, levels, np.inf).min(-1))
    return low, overflow


def speaker(env: V1Body, arm: str, believed: np.ndarray | None = None,
            forbid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Reference speaker for ``arm``: (actuators, speech). All arms act on own
    resources with Probe80's oracle rule; they differ only in speech.

    ``composed``, ``single_*`` and ``scrambled`` read the true levels.
    ``reckoned`` and ``species`` read ``believed`` levels (a ``Reckoner``),
    tracked with the individual's or the species' rates respectively."""

    if arm not in arms(env.variant):
        raise ValueError(f"unknown arm: {arm}")
    b, n = env.b, env.n
    rows = np.arange(n)
    act = env.oracle_action()
    speech = -np.ones(n, dtype=np.int64)
    if arm == "silent" or not b.blocked.any():
        return act, speech
    here = env.present()
    if not here.any():
        return act, speech
    first, second = _candidates(env, arm if arm.startswith("single_") else "composed")
    n_u = first.shape[1]
    left = np.repeat(env.left[:, None], n_u, 1)
    # in meaning space; build deliveries exactly as ``decode`` would
    valid = first >= 0
    if forbid is not None:  # (N, CHANNELS, 3): pairs this speaker may not say (v1-size meanings)
        valid &= ~forbid[rows[:, None], np.maximum(first, 0), np.maximum(second, 0)]
    if env.variant == "basket":
        split = valid & (second != first)
        d1 = (np.where(valid, first, -1), np.where(split, BASKET[0], sum(BASKET)) * valid, np.ones_like(first))
        d2 = (np.where(split, second, -1), np.where(split, BASKET[1], 0.0), np.ones_like(first))
        cost = np.zeros(first.shape)
        deliveries = [d1, d2]
    else:
        m = np.maximum(second, 0)
        want = np.asarray(SIZES)[m] if env.variant == "size" else np.full(first.shape, WHEN_PORTION)
        delay = np.ones_like(first) if env.variant == "size" else np.asarray(DELAYS)[m]
        cost = np.where(valid, np.minimum(want, left), 0.0)
        deliveries = [(np.where(valid, first, -1), cost, delay)]
    rates = np.where(b.active, BASE_RATES, 0.0) if arm == "species" else b.rates
    levels = believed if arm in RECKONING_ARMS else env.levels
    if levels is None:
        raise ValueError(f"{arm} needs believed levels")
    low, overflow = _project(env, rates, deliveries, levels)
    _, quiet_overflow = _project(env, rates, [(np.full((n, 1), -1), np.zeros((n, 1)), np.ones((n, 1), np.int64))],
                                 levels)
    spilled = overflow - quiet_overflow  # overflow caused by this request
    delivered = sum(amount for _, amount, _ in deliveries)
    absorbed = delivered - spilled
    # 1) keep every social resource above MARGIN until the next visit's delivery;
    # 2) absorb the most; 3) a spilled unit of ration counts against it (WASTE);
    # 4) ties go to the resource that would run out first.
    left_ticks = levels / np.maximum((rates + SHOCK_DRAIN) * b.active, 1e-9)
    urgency = left_ticks[rows[:, None], np.maximum(first, 0)]
    waste = 0.0 if env.variant == "basket" else env.speaker_waste
    score = 10.0 * (low >= MARGIN) + absorbed - waste * spilled - 1e-4 * np.minimum(urgency, 1e3)
    score = np.where(valid, score, -np.inf)
    best = score.argmax(1)
    speak = here & valid[rows, best] & (absorbed[rows, best] >= MIN_ABSORBED)
    c1 = first[rows, best]
    named = (c1 + 1) % CHANNELS if arm == "scrambled" else c1
    w1 = b.channel_word[rows, np.maximum(named, 0)]
    if env.variant == "basket":
        c2 = second[rows, best]
        c2 = np.where(c2 == c1, named, (c2 + 1) % CHANNELS if arm == "scrambled" else c2)
        w2 = b.channel_word[rows, np.maximum(c2, 0)]
    else:
        w2 = b.meaning_mod[rows, np.maximum(second[rows, best], 0)]
    speech[speak] = (w1 * env.m + w2)[speak]
    return act, speech


class Reckoner:
    """Dead reckoning of social levels with believed rates: what a speaker
    that knows its structure but not its levels can track. It starts each life
    from the true levels at the first decision, sees which channel each shock
    hit and which deliveries landed (equal for every reckoning arm), and
    otherwise subtracts its believed rates. Own channels are read as true."""

    def __init__(self, env: V1Body, rates: np.ndarray) -> None:
        self.rates = rates * env.b.active
        self.est = env.levels.copy()
        self._mark(env)

    def _mark(self, env: V1Body) -> None:
        self.tick, self.life, self.done = env.tick.copy(), env.life.copy(), env.finished.copy()

    def update(self, env: V1Body) -> None:
        b = env.b
        same = ~self.done & (env.life == self.life) & ~env.finished
        landed = {}
        for who, channel, amount, tick in env.landed:
            for i, c, a, t in zip(who, channel, amount, tick):
                landed.setdefault((int(i), int(t)), []).append((int(c), float(a)))
        for i in np.flatnonzero(same):
            est, life = self.est[i], int(env.life[i])
            for t in range(int(self.tick[i]) + 1, int(env.tick[i]) + 1):
                for c, a in landed.get((int(i), t), ()):
                    raw = est[c] + a
                    est[c] = 1.0 - env.harm * (raw - 1.0) if raw > 1.0 else raw
                est -= self.rates[i]
                slot = int(b.shock[i, life, t - 1])
                if slot >= 0 and b.active[i, slot]:
                    est[slot] -= PHYSICS.shock_size
                np.clip(est, 0.0, 1.0, out=est)
        reborn = ~same & ~env.finished
        self.est[reborn] = env.levels[reborn]
        own = ~b.blocked
        self.est[own] = env.levels[own]
        self._mark(env)


def run_arm(bodies: V1Bodies, variant: str, arm: str) -> dict:
    env = V1Body(bodies, variant)
    reckoner = None
    if arm in RECKONING_ARMS:
        reckoner = Reckoner(env, np.where(bodies.active, BASE_RATES, 0.0) if arm == "species" else bodies.rates)
    while not env.finished.all():
        env.step(*speaker(env, arm, None if reckoner is None else reckoner.est))
        if reckoner is not None:
            reckoner.update(env)
    return {"survived": env.survived.copy(), "requests": env.requests.copy(), "wasted": env.wasted.copy()}


# ------------------------------------------------------------------ survey
def survey(variant: str, *, blocks: int = BLOCKS, identities: int = IDENTITIES, cells_=None) -> dict:
    """Every arm on the same identities, lives and shocks; contrasts paired
    over blocks (two-sided 95% Student-t)."""

    names = meanings(variant)
    arm_list = arms(variant)
    out = {"scenario": f"caregiver-v1-{variant}", "status": "development ceiling survey; oracle speakers",
           "blocks": blocks, "identities_per_block": identities, "survey_base": SURVEY_BASE,
           "visit": VISIT, "ration_factor": RATION_FACTOR, "margin": MARGIN, "arms": list(arm_list), "cells": {}}
    for k, social in (cells_ or cells(variant)):
        block_of = np.repeat(np.arange(blocks), identities)
        idents = [identity_seed(int(bl), k, social, i) for bl in range(blocks) for i in range(identities)]
        bodies = v1_bodies(idents, [k] * len(idents), [social] * len(idents))
        runs = {arm: run_arm(bodies, variant, arm) for arm in arm_list}
        by_block = {arm: [float(runs[arm]["survived"][block_of == bl].mean()) for bl in range(blocks)]
                    for arm in arm_list}
        singles = [f"single_{m}" for m in names]
        best_single = max(singles, key=lambda a: np.mean(by_block[a]))
        asked = runs["composed"]["requests"].sum(0)
        out["cells"][f"K{k}_social{social}"] = {
            "k": k, "social": social,
            "survival": {arm: float(np.mean(by_block[arm])) for arm in arm_list},
            "survival_by_block": by_block,
            "best_single": best_single,
            "composed_minus": {arm: active_self._interval([c - o for c, o in zip(by_block["composed"], by_block[arm])])
                               for arm in arm_list if arm != "composed"},
            "reckoned_minus_species": active_self._interval(
                [r - s for r, s in zip(by_block["reckoned"], by_block["species"])]),
            "composed_requests_per_life": float(asked.sum() / (len(idents) * LIVES)),
            "composed_meaning_share": {m: float(a / max(asked.sum(), 1)) for m, a in zip(names, asked)},
            "wasted_per_life": {arm: float(runs[arm]["wasted"].mean() / LIVES) for arm in arm_list},
        }
    out["criteria"] = criteria(out)
    return out


def criteria(result: dict) -> dict:
    """The locked pass rule of the decision record."""

    cs = list(result["cells"].values())
    n = len(cs)
    diff = lambda c, a: c["composed_minus"][a]
    reck = lambda c: c["reckoned_minus_species"]
    best = [diff(c, c["best_single"]) for c in cs]
    out = {
        "1_speech": all(diff(c, "silent")["mean"] >= 0.20 for c in cs),
        "2_first_word": all(diff(c, "scrambled")["mean"] >= 0.20 for c in cs),
        "3_second_word": (sum(d["mean"] >= 0.08 and d["low"] > 0 for d in best) >= 2 * n / 3
                          and all(d["mean"] >= 0.03 for d in best)),
        "4_self_knowledge": sum(reck(c)["mean"] >= 0.04 and reck(c)["low"] > 0 for c in cs) >= n / 2,
        "5_combinations": sum(min(c["composed_meaning_share"].values()) >= 0.10 for c in cs) >= 2 * n / 3,
        "6_headroom": all(c["survival"]["composed"] <= 0.97 for c in cs),
    }
    out["pass"] = all(out.values())
    out["worst_second_word_margin"] = float(min(d["mean"] for d in best))
    return out


def table(result: dict) -> str:
    arm_list = result["arms"]
    lines = [f"{result['scenario']} ceiling survey: six-life mean survival of reference speakers",
             f"{'cell':<12}" + "".join(f"{a:>12}" for a in arm_list) + "   meaning shares (composed)"]
    for name, c in result["cells"].items():
        shares = " ".join(f"{m}:{s:.2f}" for m, s in c["composed_meaning_share"].items())
        lines.append(f"{name:<12}" + "".join(f"{c['survival'][a]:>12.3f}" for a in arm_list) + f"   {shares}")
    lines.append("composed minus [95% interval] blocks positive:")
    for name, c in result["cells"].items():
        parts = [f"{a} {d['mean']:+.3f} [{d['low']:+.3f},{d['high']:+.3f}] {d.get('positive', '-')}/{d['n']}"
                 for a, d in c["composed_minus"].items() if a in ("silent", "scrambled", c["best_single"])]
        d = c["reckoned_minus_species"]
        parts.append(f"reckoned-species {d['mean']:+.3f} [{d['low']:+.3f},{d['high']:+.3f}] {d.get('positive', '-')}/{d['n']}")
        lines.append(f"  {name} (best single {c['best_single']}): " + "; ".join(parts))
    lines.append("criteria: " + ", ".join(f"{k}={v}" for k, v in result["criteria"].items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--identities", type=int, default=IDENTITIES)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/organism/caregiver_v1"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        result = survey(variant, identities=args.identities)
        (args.out_dir / f"ceiling_survey_{variant}.json").write_text(json.dumps(result, indent=1))
        print(table(result), flush=True)


if __name__ == "__main__":
    main()
