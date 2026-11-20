"""The vectorized body must be Probe80's body, tick for tick."""

from random import Random

import numpy as np
import pytest

from homesocial.organism import active_self as probe
from homesocial.organism.saturation_structure import ACTUATORS, make_physical_world
from homesocial.organism.vec_body import (
    BIRTH_LEVELS, CHANNELS, DECISIONS, LIFE_TICKS, LIVES, PHYSICS, WINDOW,
    VecBody, bodies_from_identities, reference_survival, sample_bodies,
)

IDENTS = [(probe.identity_seed(probe.DIAGNOSTIC_BASE, 0, k, i), k) for k in (1, 2, 3, 4, 5) for i in range(3)]


def _reference_lives(ident: int, k: int, table: np.ndarray):
    """``active_self.run_arm``'s loop over ``LivedBody`` with a fixed action
    table: per life, the per-tick sensations and whether it was survived."""

    world = make_physical_world(seed=probe.domain_seed(ident, "map"), k=k)
    lives = []
    for life in range(LIVES):
        body = probe.LivedBody(world, identity=ident, life=life)
        senses, pending, grants = [body.sense()], None, 0
        for _ in range(PHYSICS.life_steps):
            absorbed, pending = pending, None
            if not body.tick(absorbed):
                break
            senses.append(body.sense())
            if body.ticks % PHYSICS.help_period == 0:
                pending = int(table[life, grants])
                grants += 1
        lives.append((senses, body.alive, grants, body.ticks))
    return lives


def test_clock_matches_probe80():
    assert (WINDOW, LIFE_TICKS, DECISIONS, LIVES, CHANNELS) == (6, 400, 66, 6, 5)


@pytest.mark.parametrize("policy_seed", [0, 1])
def test_vectorized_lives_equal_lived_bodies_tick_for_tick(policy_seed):
    idents, ks = [i for i, _ in IDENTS], [k for _, k in IDENTS]
    tables = np.array([[[Random(f"{policy_seed}:{ident}:{life}:{d}").randrange(ACTUATORS)
                         for d in range(DECISIONS)] for life in range(LIVES)] for ident in idents])
    reference = [_reference_lives(ident, k, tables[i]) for i, (ident, k) in enumerate(IDENTS)]
    env = VecBody(bodies_from_identities(idents, ks))
    rows = np.arange(env.n)
    seen = [[[] for _ in range(LIVES)] for _ in idents]
    while not env.finished.all():
        for i in np.flatnonzero(~env.finished):
            seen[i][env.life[i]].append((env.prev_total[i], env.prev_min[i], env.obs_total[i].copy(),
                                         env.obs_min[i].copy(), bool(env.birth[i]), int(env.decision[i])))
        actions = tables[rows, np.minimum(env.life, LIVES - 1), np.minimum(env.decision, DECISIONS - 1)]
        env.step(actions)
    checked = 0
    for i in range(len(idents)):
        for life, (senses, alive, grants, ticks) in enumerate(reference[i]):
            assert env.survived[i, life] == alive
            assert env.lived[i, life] == ticks
            assert len(seen[i][life]) == grants
            for d, (prev_total, prev_min, total, minimum, birth, decision) in enumerate(seen[i][life]):
                window = senses[WINDOW * d + 1: WINDOW * (d + 1) + 1]
                assert birth == (d == 0) and decision == d
                np.testing.assert_allclose((prev_total, prev_min), senses[WINDOW * d], rtol=0, atol=1e-12)
                np.testing.assert_allclose(total, [s[0] for s in window], rtol=0, atol=1e-12)
                np.testing.assert_allclose(minimum, [s[1] for s in window], rtol=0, atol=1e-12)
                checked += 1
    assert checked > 500  # the comparison really covered decisions, deaths and rebirths


def test_oracle_action_is_lived_body_oracle_choice():
    for ident, k in IDENTS:
        world = make_physical_world(seed=probe.domain_seed(ident, "map"), k=k)
        body = probe.LivedBody(world, identity=ident, life=0)
        env = VecBody(bodies_from_identities([ident], [k]))
        for _ in range(PHYSICS.help_period):
            body.tick(None)
        if body.alive:
            assert env.oracle_action()[0] == body.oracle_choice()


def test_sampled_bodies_come_from_the_same_family():
    bodies = sample_bodies(np.random.default_rng(3), 4000)
    assert set(np.unique(bodies.k)) == {1, 2, 3, 4, 5}
    assert (bodies.active.sum(1) == bodies.k).all()
    # every channel has exactly one small and one large actuator
    assert (np.sort(bodies.target, 1) == np.repeat(np.arange(CHANNELS), 2)).all()
    assert (np.sort(bodies.portion, 1)[:, [0, -1]] == [PHYSICS.portion_small, PHYSICS.portion_large]).all()
    assert (bodies.actionable().sum(1) == 2 * bodies.k).all()
    assert ((bodies.rates > 0) == bodies.active).all()
    assert np.isin(bodies.birth[bodies.active[:, None, :].repeat(LIVES, 1)], BIRTH_LEVELS).all()
    assert (bodies.birth[~bodies.active[:, None, :].repeat(LIVES, 1)] == 1.0).all()
    assert abs((bodies.shock >= 0).mean() - PHYSICS.shock_probability) < 0.002


def test_sampled_and_seeded_bodies_agree_in_distribution():
    """Random and oracle survival of NumPy-sampled K2 bodies and of
    identity-seeded K2 bodies agree, and sit where Probe80 measured them."""

    rng = np.random.default_rng(11)
    seeded = bodies_from_identities([probe.DIAGNOSTIC_BASE + 7_000_000 + 100 * i for i in range(300)], [2] * 300)
    sampled = sample_bodies(rng, 300, ks=(2,))
    for policy, published in (("random", 0.164), ("oracle", 0.906)):
        a = reference_survival(seeded, policy, rng).mean()
        b = reference_survival(sampled, policy, rng).mean()
        assert abs(a - b) < 0.06 and abs(a - published) < 0.06, (policy, a, b)


def test_replace_starts_a_fresh_identity():
    rng = np.random.default_rng(5)
    env = VecBody(sample_bodies(rng, 8))
    while not env.finished.any():
        env.step(rng.integers(0, ACTUATORS, env.n))
    index = np.flatnonzero(env.finished)
    env.replace(index, sample_bodies(rng, len(index)))
    assert not env.finished.any()
    assert (env.life[index] == 0).all() and env.birth[index].all()
