"""Caregiver world v0: can words about a discovered body pay? A ceiling survey.

Development instrument, not a result and not a learner. It asks the question
this repository requires before any language mechanism is built: in this
ecology, does a *perfect* speaker survive more often when it composes a
request than when it cannot -- and does it need to know its own body to choose
the words? If the oracle gains nothing, no learner can, and the design closes
here at the cost of a few seconds.

Scenario ``caregiver-v0`` (extends Probe80's body family; ``vec_body``):

- Some of the body's resources are *social*: no actuator of its own restores
  them. Only a caregiver can.
- At a decision the organism either uses an actuator or says two words to the
  caregiver: a resource word and a size word.
- The caregiver's convention belongs to the identity and is not given: six
  resource words name the five channels in a random order (one word names
  nothing), and which of the two size words means "large" is random.
- Consequence. A small portion (0.20) arrives on the next tick, like an
  actuator. A large portion (0.60) arrives ``LARGE_DELAY`` ticks later, and
  until it has arrived the caregiver hears nothing else. So the right size
  depends on how soon the resource runs out -- on this body's own rates.

Not built here: what the organism hears (labels, narration), any learner, any
held-out combination. Those wait for the learned-self result.

    PYTHONPATH=src .venv/bin/python -m homesocial.organism.caregiver_world
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
    ACTUATORS, BASE_RATES, CHANNELS, LIVES, PHYSICS, WINDOW, Bodies, VecBody, bodies_from_identities,
)

WORDS = 6
SIZES = 2
ACTIONS = ACTUATORS + WORDS * SIZES
LARGE_DELAY = 12
SAFETY = WINDOW  # a large request is trusted only if it lands a window before the resource runs out
ARMS = ("silent", "small_only", "large_only", "composed", "species_size", "scrambled")
CELLS = ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 2))  # (active resources, of which social)
BLOCKS, IDENTITIES = 5, 100
SURVEY_BASE = 6_200_000_000  # development; disjoint from Probe80 (5.0B-5.2B) and learned_self (6.1B)
BLOCK_STRIDE, K_STRIDE, SOCIAL_STRIDE, IDENTITY_STRIDE = 2_000_000, 100_000, 10_000, 100


def identity_seed(block: int, k: int, social: int, identity: int) -> int:
    if not (0 <= block < BLOCKS and 1 <= k <= CHANNELS and 0 <= social < 10 and 0 <= identity < IDENTITIES):
        raise ValueError("identity outside the survey band")
    return SURVEY_BASE + block * BLOCK_STRIDE + k * K_STRIDE + social * SOCIAL_STRIDE + identity * IDENTITY_STRIDE


@dataclass
class CaregiverBodies(Bodies):
    word_channel: np.ndarray  # (N, WORDS) channel each resource word names; -1 names nothing
    channel_word: np.ndarray  # (N, CHANNELS) its inverse
    large_word: np.ndarray    # (N,) which size word means "large"


def caregiver_bodies(identities, ks, socials) -> CaregiverBodies:
    """Probe80 bodies for these identity seeds, with ``social`` of each body's
    active resources made unreachable by its own actuators, and a private
    caregiver convention. Both are drawn from the identity's own streams."""

    base = bodies_from_identities(identities, ks)
    n = base.n
    word_channel = -np.ones((n, WORDS), dtype=np.int64)
    channel_word = np.zeros((n, CHANNELS), dtype=np.int64)
    large_word = np.zeros(n, dtype=np.int64)
    for i, (ident, social) in enumerate(zip(identities, socials, strict=True)):
        active = [int(c) for c in np.flatnonzero(base.active[i])]
        if not 0 <= social < len(active):
            raise ValueError("a body keeps at least one resource it can restore itself")
        base.blocked[i, Random(active_self.domain_seed(ident, "social")).sample(active, social)] = True
        convention = Random(active_self.domain_seed(ident, "convention"))
        words = convention.sample(range(WORDS), CHANNELS)
        word_channel[i, words] = np.arange(CHANNELS)
        channel_word[i] = words
        large_word[i] = convention.randrange(SIZES)
    return CaregiverBodies(**{name: getattr(base, name) for name in Bodies.__dataclass_fields__},
                           word_channel=word_channel, channel_word=channel_word, large_word=large_word)


class CaregiverBody(VecBody):
    """``VecBody`` with a caregiver. Actions 0..9 are actuators; action
    ``ACTUATORS + word * SIZES + size`` says two words instead."""

    def __init__(self, bodies: CaregiverBodies, *, large_delay: int = LARGE_DELAY, **kwargs) -> None:
        n = bodies.n
        self.large_delay = large_delay
        self.due_tick = -np.ones(n, dtype=np.int64)
        self.due_channel = np.zeros(n, dtype=np.int64)
        self.due_portion = np.zeros(n)
        self.requests = np.zeros((n, SIZES), dtype=np.int64)  # heard requests: [small, large]
        super().__init__(bodies, **kwargs)

    def _on_birth(self, index: np.ndarray) -> None:
        self.due_tick[index] = -1

    def _arrivals(self, index, tick, levels, run) -> None:
        landed = np.flatnonzero(run & (self.due_tick[index] == tick))
        if len(landed):
            who = index[landed]
            channel = self.due_channel[who]
            levels[landed, channel] = np.minimum(levels[landed, channel] + self.due_portion[who], 1.0)
            self.due_tick[who] = -1

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(actions)
        speaking = (actions >= ACTUATORS) & ~self.finished
        code = np.maximum(actions - ACTUATORS, 0)
        word, size = code // SIZES, code % SIZES
        heard = np.flatnonzero(speaking & (self.due_tick < 0))  # a busy caregiver hears nothing
        channel = self.b.word_channel[heard, word[heard]]
        heard, channel = heard[channel >= 0], channel[channel >= 0]  # the empty word asks for nothing
        large = size[heard] == self.b.large_word[heard]
        self.due_channel[heard] = channel
        self.due_portion[heard] = np.where(large, PHYSICS.portion_large, PHYSICS.portion_small)
        self.due_tick[heard] = self.tick[heard] + 1 + np.where(large, self.large_delay, 0)
        np.add.at(self.requests, (heard, large.astype(np.int64)), 1)
        return super().step(np.where(actions >= ACTUATORS, -1, actions))


def speaker_action(env: CaregiverBody, arm: str) -> np.ndarray:
    """Reference speakers. All know the true structure, levels and
    convention; they differ only in what they are able or allowed to say.

    Each acts for the resource whose remedy must be started soonest: an own
    resource by its large actuator, a social one by a request.

    - ``silent``: never speaks.
    - ``small_only`` / ``large_only``: one size word; no composition.
    - ``composed``: small if the resource would run out before a large portion
      could arrive, else large -- judged from this body's own rates.
    - ``species_size``: the same rule, but the *size* is judged from the
      species' rates instead of the individual's. Everything else identical.
    - ``scrambled``: ``composed`` with the resource word of another channel.
    """

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    b, rows = env.b, np.arange(env.n)
    left = np.where(b.active, env.levels / np.maximum(b.rates, 1e-9), np.inf)
    own = b.active & ~b.blocked
    askable = b.blocked & (env.due_tick < 0)[:, None] & (arm != "silent")
    lead_large = 1 + env.large_delay
    if arm == "small_only":
        small = np.ones_like(own)
    elif arm == "large_only":
        small = np.zeros_like(own)
    else:
        judged = left if arm != "species_size" else np.where(b.active, env.levels / np.maximum(BASE_RATES, 1e-9), np.inf)
        small = judged < lead_large + SAFETY
    slack = np.where(own, left - 1, np.where(askable, left - np.where(small, 1, lead_large), np.inf))
    channel = slack.argmin(1)
    named = (channel + 1) % CHANNELS if arm == "scrambled" else channel
    size_word = np.where(small[rows, channel], 1 - b.large_word, b.large_word)
    request = ACTUATORS + b.channel_word[rows, named] * SIZES + size_word
    return np.where(askable[rows, channel], request, b.large_actuator()[rows, channel])


def run_arm(bodies: CaregiverBodies, arm: str, *, large_delay: int = LARGE_DELAY) -> dict:
    env = CaregiverBody(bodies, large_delay=large_delay)
    while not env.finished.all():
        env.step(speaker_action(env, arm))
    return {"survived": env.survived.copy(), "requests": env.requests.copy()}


def survey(*, blocks: int = BLOCKS, identities: int = IDENTITIES, large_delay: int = LARGE_DELAY,
           cells=CELLS) -> dict:
    """Every arm on the same identities, lives and shocks; contrasts are
    paired over blocks (two-sided 95% Student-t)."""

    out = {"scenario": "caregiver-v0", "status": "development ceiling survey; oracle speakers; not a learner",
           "large_delay": large_delay, "blocks": blocks, "identities_per_block": identities,
           "survey_base": SURVEY_BASE, "arms": list(ARMS), "cells": {}}
    for k, social in cells:
        block_of = np.repeat(np.arange(blocks), identities)
        idents = [identity_seed(int(b), k, social, i) for b in range(blocks) for i in range(identities)]
        bodies = caregiver_bodies(idents, [k] * len(idents), [social] * len(idents))
        runs = {arm: run_arm(bodies, arm, large_delay=large_delay) for arm in ARMS}
        by_block = {arm: [float(runs[arm]["survived"][block_of == b].mean()) for b in range(blocks)] for arm in ARMS}
        asked = runs["composed"]["requests"].sum(0)
        out["cells"][f"K{k}_social{social}"] = {
            "k": k, "social": social,
            "survival": {arm: float(np.mean(by_block[arm])) for arm in ARMS},
            "survival_by_block": by_block,
            "composed_minus": {arm: active_self._interval([c - o for c, o in zip(by_block["composed"], by_block[arm])])
                               for arm in ARMS if arm != "composed"},
            "composed_requests_per_life": float(asked.sum() / (len(idents) * LIVES)),
            "composed_share_small": float(asked[0] / max(asked.sum(), 1)),
        }
    return out


def table(result: dict) -> str:
    lines = [f"caregiver-v0 ceiling survey: six-life mean survival of oracle speakers "
             f"(large portions arrive after {result['large_delay']} ticks)",
             f"{'cell':<13}" + "".join(f"{arm:>13}" for arm in ARMS) + f"{'small share':>13}"]
    for name, cell in result["cells"].items():
        lines.append(f"{name:<13}" + "".join(f"{cell['survival'][arm]:>13.3f}" for arm in ARMS)
                     + f"{cell['composed_share_small']:>13.2f}")
    lines.append("composed minus each arm, paired over blocks [95% interval], blocks positive:")
    for name, cell in result["cells"].items():
        parts = [f"{arm} {c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}] {c['positive']}/{c['n']}"
                 for arm, c in cell["composed_minus"].items()]
        lines.append(f"  {name}: " + "; ".join(parts))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--large-delay", type=int, default=LARGE_DELAY)
    parser.add_argument("--identities", type=int, default=IDENTITIES)
    parser.add_argument("--out", type=Path, default=Path("runs/organism/caregiver_world_v0/ceiling_survey.json"))
    args = parser.parse_args()
    result = survey(identities=args.identities, large_delay=args.large_delay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(table(result))


if __name__ == "__main__":
    main()
