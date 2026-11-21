"""Mechanics of caregiver-v2 / v2r and the held-out pair."""

import numpy as np
import pytest

from homesocial.organism import caregiver_v1 as cv
from homesocial.organism import caregiver_v2 as c2
from homesocial.organism import speaking_self as ss
from homesocial.organism.vec_body import BASE_RATES, CHANNELS, reference_survival


def _bodies(n=6, k=3, social=1):
    idents = [c2.SURVEY_BASE + k * cv.K_STRIDE + social * cv.SOCIAL_STRIDE + i * cv.IDENTITY_STRIDE for i in range(n)]
    return cv.v1_bodies(idents, [k] * n, [social] * n)


def _quiet(bodies, level, ration=False):
    bodies.shock[:] = -1
    bodies.rates[:] = np.where(bodies.active, 0.001, 0.0)
    env = c2.V2Body(bodies, ration=ration)
    env.levels[:] = np.where(bodies.active, level, 1.0)
    return env


def _ask(b, size):
    rows = np.arange(b.n)
    return b.channel_word[rows, b.blocked.argmax(1)] * 3 + b.meaning_mod[rows, size]


@pytest.mark.parametrize("level,size,expected", [(0.3, 2, 0.9), (0.7, 2, 1.0 - c2.HARM * 0.3), (0.7, 0, 0.9)])
def test_overfilling_makes_the_body_sick(level, size, expected):
    b = _bodies()
    idle = np.full(b.n, -1)
    asked, twin = _quiet(_bodies(), level), _quiet(_bodies(), level)
    asked.step(idle, _ask(b, size))
    twin.step(idle)
    rows, channel = np.arange(b.n), b.blocked.argmax(1)
    np.testing.assert_allclose(asked.levels[rows, channel] - twin.levels[rows, channel], expected - level)


def test_ration_only_in_v2r():
    b = _bodies()
    idle = np.full(b.n, -1)
    free, rationed = _quiet(_bodies(), 0.3), _quiet(_bodies(), 0.3, ration=True)
    assert np.isinf(free.left).all() and np.allclose(rationed.left, b.ration)
    for env in (free, rationed):
        env.step(idle, _ask(b, 2))
    assert np.isinf(free.left).all()
    np.testing.assert_allclose(rationed.left, b.ration - 0.6)
    assert rationed.speaker_waste == cv.WASTE and free.speaker_waste == 0.0


def test_v1_class_is_unchanged_by_the_hooks():
    assert cv.V1Body.harm == 0.0 and cv.V1Body.speaker_waste == cv.WASTE
    b = _bodies(n=30, k=3, social=0)
    np.testing.assert_array_equal(c2.run_arm(b, "silent")["survived"],
                                  reference_survival(b, "oracle", np.random.default_rng(0)))


def test_reckoning_with_own_rates_stays_exact_under_sickness():
    b = _bodies(n=40, k=3, social=2)
    env = c2.V2Body(b, ration=True)
    reckoner = cv.Reckoner(env, b.rates)
    for _ in range(60):
        env.step(*cv.speaker(env, "reckoned", reckoner.est))
        reckoner.update(env)
        live = ~env.finished
        assert np.abs(reckoner.est - env.levels)[live].max() < 1e-9


def test_shared_convention_and_training_restriction():
    b = c2.restrict_hold_rate(c2.share(_bodies(n=200, k=4, social=2)), np.random.default_rng(1))
    assert (b.word_channel == c2.SHARED_WORDS).all() and (b.mod_meaning == np.arange(3)).all()
    live = b.active[:, c2.HOLD_CHANNEL]
    factor = b.rates[live, c2.HOLD_CHANNEL] / BASE_RATES[c2.HOLD_CHANNEL]
    assert factor.min() >= c2.TRAIN_FACTOR[0] and factor.max() <= c2.TRAIN_FACTOR[1]
    assert set(np.unique(b.birth[live][:, :, c2.HOLD_CHANNEL])) <= set(c2.TRAIN_BIRTH)
    alone = b.blocked[:, c2.HOLD_CHANNEL]
    assert alone.any() and (b.blocked[alone].sum(1) == 1).all()
    np.testing.assert_allclose(b.ration, cv.RATION_FACTOR * cv.species_need(b.blocked))


def test_held_out_bodies_and_forbidden_pair():
    idents = [c2.SURVEY_BASE + 60_000_000 + i * 100 for i in range(60)]
    b = c2.hold_bodies(idents, [3] * 60, [2] * 60, c2.TEST_FACTOR)
    assert b.n > 0 and b.blocked[:, c2.HOLD_CHANNEL].all() and (b.blocked.sum(1) == 2).all()
    factor = b.rates[:, c2.HOLD_CHANNEL] / BASE_RATES[c2.HOLD_CHANNEL]
    assert factor.min() >= c2.TEST_FACTOR[0] and factor.max() <= c2.TEST_FACTOR[1]
    np.testing.assert_allclose(b.ration, cv.RATION_FACTOR * cv.species_need(b.blocked))
    forbidden = c2.run_arm(b, "no_held_out_pair", ration=True)
    assert forbidden["held_out_pair"].sum() == 0 and forbidden["hold_requests"].sum() > 0
    assert c2.run_arm(b, "composed", ration=True)["held_out_pair"].sum() > 0


def test_learner_trains_in_v2r_shared_and_mutes_only_the_pair():
    trainer = ss.Trainer(ss.TrainConfig(envs=16, horizon=8, hidden=32, world="v2r", convention="shared"))
    trainer.train_step()
    assert (trainer.env.b.word_channel == c2.SHARED_WORDS).all() and np.isfinite(trainer.env.left).all()
    with pytest.raises(ValueError):
        ss.Trainer(ss.TrainConfig(envs=4, horizon=4, hidden=8, world="v1", convention="shared"))
    sets = ss.eval_sets(6, "shared")
    assert set(sets) == {"dev", "held_out"} and sets["held_out"].blocked[:, c2.HOLD_CHANNEL].all()
    result = ss.evaluate_sets(trainer.model, sets, 0, "v2r")
    assert set(result["held_out"]) == {"none", "scramble_w2", "mute_pair"}
    assert set(result["held_out"]["none"]) == {f"K{k}_social{s}" for k, s in c2.HOLD_CELLS}
    assert CHANNELS == 5
