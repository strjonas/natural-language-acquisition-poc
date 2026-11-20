"""Probe80's lived body family as NumPy arrays, many identities at once.

``active_self.LivedBody`` is the reference physics: one Python object per life,
one identity at a time. A learner that must live through hundreds of millions
of decisions needs many identities advanced together, so this module restates
the same family over arrays (measured about fifty times the reference's ticks
per second on one core). It adds no physics. For any identity
seed, ``bodies_from_identities`` reproduces ``LivedBody``'s map, rates, birth
levels and shock stream exactly, and ``tests/test_vec_body.py`` checks the two
tick by tick.

Time is advanced one *decision* at a time, mirroring ``active_self.run_arm``:
a life starts with one grant-free window of ``WINDOW`` ticks; after each window
the organism chooses an actuator, which is absorbed on the first tick of the
next window; the 66th choice (tick 396) is followed by the last four ticks.

Structure and constants in ``Bodies`` are private to the physics. A learner is
given only what ``VecBody`` exposes as observations: the two scalar sensations
after every tick of the last window, its own previous choice, and birth events.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from random import Random
from typing import Sequence

import numpy as np

from homesocial.island.expanded_body import EXPANDED_BODY_SCHEMA
from homesocial.organism import active_self
from homesocial.organism.saturation_structure import ACTUATORS, make_physical_world

CHANNELS = len(EXPANDED_BODY_SCHEMA)
PHYSICS = active_self.PHYSICS
LIVES = active_self.LIVES
WINDOW = PHYSICS.help_period
LIFE_TICKS = PHYSICS.life_steps
DECISIONS = (LIFE_TICKS - 1) // WINDOW  # choices per completed life: ticks 6, 12, ..., 396
BASE_RATES = np.array([axis.resting_rate for axis in EXPANDED_BODY_SCHEMA.axes])
BIRTH_LEVELS = np.array(PHYSICS.birth_levels)
SCALE_LOW, SCALE_HIGH = 0.4, 1.6  # LivedBody's literal individual metabolic range

# Probe80's published survey table (docs/decisions/2026-10-05-active-self-discovery-result.md),
# six-life mean survival by K. Different bodies from the same family: a reference
# for development comparisons, not a paired control.
PROBE80_SURVIVAL = {
    "random": (0.436, 0.164, 0.047, 0.017, 0.006),
    "bandit": (0.969, 0.417, 0.039, 0.006, 0.000),
    "posterior": (0.978, 0.864, 0.675, 0.472, 0.272),
    "reset": (0.864, 0.661, 0.469, 0.297, 0.158),
    "oracle": (1.000, 0.906, 0.747, 0.594, 0.572),
}
PROBE80_EXACT_LIFE6 = (1.00, 1.00, 0.93, 0.67, 0.25)


@dataclass
class Bodies:
    """Structure, constants and random streams of N identities."""

    k: np.ndarray        # (N,) number of active resources
    active: np.ndarray   # (N, CHANNELS) bool
    blocked: np.ndarray  # (N, CHANNELS) bool: active but not restorable by any own actuator
    target: np.ndarray   # (N, ACTUATORS) channel each actuator drives
    portion: np.ndarray  # (N, ACTUATORS) portion size
    rates: np.ndarray    # (N, CHANNELS) depletion per tick; zero on inactive channels
    birth: np.ndarray    # (N, LIVES, CHANNELS) birth levels; one on inactive channels
    shock: np.ndarray    # (N, LIVES, LIFE_TICKS) int8: shocked channel, or -1

    @property
    def n(self) -> int:
        return len(self.k)

    def take(self, index: np.ndarray) -> "Bodies":
        return type(self)(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def assign(self, index: np.ndarray, other: "Bodies") -> None:
        for f in fields(self):
            getattr(self, f.name)[index] = getattr(other, f.name)

    def actionable(self) -> np.ndarray:
        """(N, ACTUATORS): the actuator restores an active, unblocked resource."""

        rows = np.arange(self.n)[:, None]
        return self.active[rows, self.target] & ~self.blocked[rows, self.target]

    def large_actuator(self) -> np.ndarray:
        """(N, CHANNELS): the large actuator of each channel."""

        out = np.zeros((self.n, CHANNELS), dtype=np.int64)
        rows, cols = np.nonzero(self.portion == PHYSICS.portion_large)
        out[rows, self.target[rows, cols]] = cols
        return out


def sample_bodies(rng: np.random.Generator, n: int, ks: Sequence[int] = (1, 2, 3, 4, 5)) -> Bodies:
    """Draw N identities from the same distribution as Probe80's generator,
    from a NumPy stream that shares nothing with any identity seed band."""

    k = rng.choice(np.asarray(ks), n)
    rank = rng.random((n, CHANNELS)).argsort(1).argsort(1)
    active = rank < k[:, None]
    catalog = rng.random((n, ACTUATORS)).argsort(1)  # uniform permutation of (channel, size)
    target = catalog // 2
    portion = np.where(catalog % 2 == 1, PHYSICS.portion_large, PHYSICS.portion_small)
    rates = BASE_RATES * rng.uniform(SCALE_LOW, SCALE_HIGH, (n, CHANNELS)) * active
    birth = np.where(active[:, None, :], rng.choice(BIRTH_LEVELS, (n, LIVES, CHANNELS)), 1.0)
    struck = rng.random((n, LIVES, LIFE_TICKS)) < PHYSICS.shock_probability
    shock = np.where(struck, rng.integers(0, CHANNELS, (n, LIVES, LIFE_TICKS)), -1).astype(np.int8)
    return Bodies(k=k.astype(np.int64), active=active, blocked=np.zeros_like(active), target=target,
                  portion=portion, rates=rates, birth=birth, shock=shock)


def bodies_from_identities(identities: Sequence[int], ks: Sequence[int]) -> Bodies:
    """Exactly the bodies ``active_self.LivedBody`` builds for these identity
    seeds: same map, individual rates, birth levels and shock stream."""

    n = len(identities)
    active = np.zeros((n, CHANNELS), dtype=bool)
    target = np.zeros((n, ACTUATORS), dtype=np.int64)
    portion = np.zeros((n, ACTUATORS))
    rates = np.zeros((n, CHANNELS))
    birth = np.ones((n, LIVES, CHANNELS))
    shock = -np.ones((n, LIVES, LIFE_TICKS), dtype=np.int8)
    seed = active_self.domain_seed
    for i, (ident, k) in enumerate(zip(identities, ks, strict=True)):
        world = make_physical_world(seed=seed(ident, "map"), k=int(k))
        active[i, list(world.active_slots)] = True
        for a, assignment in enumerate(world.actuators):
            target[i, a] = assignment.target
            portion[i, a] = PHYSICS.portion_large if assignment.size == "large" else PHYSICS.portion_small
        scale = Random(seed(ident, "scale"))
        for slot, axis in enumerate(EXPANDED_BODY_SCHEMA.axes):
            factor = scale.uniform(SCALE_LOW, SCALE_HIGH)
            if active[i, slot]:
                rates[i, slot] = axis.resting_rate * factor
        for life in range(LIVES):
            draw = Random(seed(ident * 16 + life, "birth"))
            for slot in range(CHANNELS):
                if active[i, slot]:
                    birth[i, life, slot] = draw.choice(PHYSICS.birth_levels)
            stream = Random(seed(ident * 16 + life, "shock"))
            for tick in range(LIFE_TICKS):
                if stream.random() < PHYSICS.shock_probability:
                    shock[i, life, tick] = stream.randrange(CHANNELS)
    return Bodies(k=np.asarray(ks, dtype=np.int64), active=active, blocked=np.zeros_like(active),
                  target=target, portion=portion, rates=rates, birth=birth, shock=shock)


class VecBody:
    """N identities, each living ``LIVES`` lives, advanced one decision at a time.

    After construction and after every ``step`` each unfinished identity is
    waiting to choose an actuator. Observation buffers (valid where
    ``finished`` is False):

    - ``obs_total``, ``obs_min``: (N, WINDOW) sensations after each tick of the
      window just lived;
    - ``prev_total``, ``prev_min``: (N,) sensations just before that window
      (the birth sensation at a life's first decision);
    - ``birth``: this is the first decision of a life;
    - ``prev_action``: the actuator absorbed in that window, -1 at birth;
    - ``life``, ``decision``: position in the identity's six lives.
    """

    def __init__(self, bodies: Bodies, *, alive_reward: float = 0.0, survive_bonus: float = 1.0) -> None:
        self.b = bodies
        self.n = n = bodies.n
        self.alive_reward = alive_reward
        self.survive_bonus = survive_bonus
        self.levels = np.ones((n, CHANNELS))
        self.life = np.zeros(n, dtype=np.int64)
        self.tick = np.zeros(n, dtype=np.int64)
        self.decision = np.zeros(n, dtype=np.int64)
        self.finished = np.zeros(n, dtype=bool)
        self.survived = np.zeros((n, LIVES), dtype=bool)
        self.lived = np.zeros((n, LIVES), dtype=np.int64)
        self.obs_total = np.zeros((n, WINDOW))
        self.obs_min = np.zeros((n, WINDOW))
        self.prev_total = np.zeros(n)
        self.prev_min = np.zeros(n)
        self.birth = np.zeros(n, dtype=bool)
        self.prev_action = -np.ones(n, dtype=np.int64)
        self._start_lives(np.arange(n))

    # ---------------------------------------------------------------- physics
    @staticmethod
    def _total(levels: np.ndarray) -> np.ndarray:
        return CHANNELS * (levels.sum(1) / CHANNELS)

    def _on_birth(self, index: np.ndarray) -> None:
        """Hook for subclasses with state beyond the body (none here)."""

    def _arrivals(self, index: np.ndarray, tick: np.ndarray, levels: np.ndarray, run: np.ndarray) -> None:
        """Hook: restoration arriving at ``tick`` from outside the organism's
        own actuators. Mutates ``levels`` in place. None in this family."""

    def _window(self, index: np.ndarray, actions: np.ndarray | None) -> np.ndarray:
        """Advance ``index`` by one window; return which bodies are still alive."""

        b = self.b
        m = len(index)
        rows = np.arange(m)
        levels = self.levels[index]
        rates, active = b.rates[index], b.active[index]
        life, tick0 = self.life[index], self.tick[index]
        steps = np.minimum(WINDOW, LIFE_TICKS - tick0)
        alive = np.ones(m, dtype=bool)
        ran = np.zeros(m, dtype=np.int64)
        total = np.zeros((m, WINDOW))
        minimum = np.zeros((m, WINDOW))
        if actions is not None:  # -1: no actuator used at this decision
            used = np.maximum(actions, 0)
            tgt = b.target[index, used]
            usable = (actions >= 0) & active[rows, tgt] & ~b.blocked[index, tgt]
            gain = np.where(usable, b.portion[index, used], 0.0)
            levels[rows, tgt] = np.minimum(levels[rows, tgt] + gain, 1.0)
        for j in range(WINDOW):
            run = alive & (j < steps)
            self._arrivals(index, tick0 + j + 1, levels, run)
            after = np.clip(levels - rates, 0.0, 1.0)
            slot = b.shock[index, life, np.minimum(tick0 + j, LIFE_TICKS - 1)].astype(np.int64)
            hit = run & (slot >= 0)
            hit &= active[rows, np.maximum(slot, 0)]
            after[hit, slot[hit]] = np.maximum(after[hit, slot[hit]] - PHYSICS.shock_size, 0.0)
            levels = np.where(run[:, None], after, levels)
            ran += run
            alive &= ~(run & (levels.min(1) <= 0.0))
            total[:, j] = self._total(levels)
            minimum[:, j] = levels.min(1)
        self.levels[index] = levels
        self.tick[index] = tick0 + ran
        self.obs_total[index] = total
        self.obs_min[index] = minimum
        return alive

    def _end_lives(self, index: np.ndarray, survived: np.ndarray | bool) -> None:
        life = self.life[index]
        self.survived[index, life] = survived
        self.lived[index, life] = self.tick[index]
        self.life[index] = life + 1
        self.finished[index] = self.life[index] >= LIVES

    def _start_lives(self, index: np.ndarray) -> None:
        """Begin the current life of each body in ``index`` and live its first,
        grant-free window. A body that dies in it goes straight to its next life."""

        while len(index):
            levels = self.b.birth[index, self.life[index]]
            self.levels[index] = levels
            self.tick[index] = 0
            self.decision[index] = 0
            self.prev_total[index] = self._total(levels)
            self.prev_min[index] = levels.min(1)
            self.birth[index] = True
            self.prev_action[index] = -1
            self._on_birth(index)
            dead = index[~self._window(index, None)]
            self._end_lives(dead, False)
            index = dead[~self.finished[dead]]

    # ------------------------------------------------------------------ API
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every unfinished identity absorbs its chosen actuator and lives one
        window. Returns ``(reward, life_ended)``; where a life ended, the
        observation buffers already hold the next life's first window."""

        index = np.flatnonzero(~self.finished)
        reward = np.zeros(self.n)
        life_ended = np.zeros(self.n, dtype=bool)
        if not len(index):
            return reward, life_ended
        chosen = np.asarray(actions)[index]
        self.prev_total[index] = self.obs_total[index, -1]
        self.prev_min[index] = self.obs_min[index, -1]
        self.birth[index] = False
        self.prev_action[index] = chosen
        alive = self._window(index, chosen)
        self.decision[index] += 1
        completed = alive & (self.tick[index] >= LIFE_TICKS)
        reward[index] = self.alive_reward * alive + self.survive_bonus * completed
        ended = ~alive | completed
        over = index[ended]
        life_ended[over] = True
        self._end_lives(over, completed[ended])
        self._start_lives(over[~self.finished[over]])
        return reward, life_ended

    def replace(self, index: np.ndarray, bodies: Bodies) -> None:
        """Put new identities in the given slots and begin their first lives."""

        self.b.assign(index, bodies)
        self.life[index] = 0
        self.finished[index] = False
        self.survived[index] = False
        self.lived[index] = 0
        self._start_lives(np.asarray(index))

    # ------------------------------------------------- reference policies
    def oracle_action(self) -> np.ndarray:
        """``LivedBody.oracle_choice``: the large actuator of the unblocked
        resource that would run out first, from true levels and rates."""

        b = self.b
        left = np.where(b.active & ~b.blocked, self.levels / np.maximum(b.rates, 1e-9), np.inf)
        return b.large_actuator()[np.arange(self.n), left.argmin(1)]


def reference_survival(bodies: Bodies, policy: str, rng: np.random.Generator) -> np.ndarray:
    """(N, LIVES) survival of a reference policy: ``random`` or ``oracle``."""

    env = VecBody(bodies)
    while not env.finished.all():
        if policy == "random":
            actions = rng.integers(0, ACTUATORS, env.n)
        elif policy == "oracle":
            actions = env.oracle_action()
        else:
            raise ValueError(f"unknown reference policy: {policy}")
        env.step(actions)
    return env.survived
