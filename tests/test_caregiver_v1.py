"""Mechanics of the caregiver-v1 worlds, their reference speakers, and the
speaking learner's plumbing."""

import numpy as np
import pytest

from homesocial.organism import caregiver_v1 as cv
from homesocial.organism import caregiver_world as v0
from homesocial.organism import learned_self as ls
from homesocial.organism import speaking_self as ss
from homesocial.organism.vec_body import ACTUATORS, CHANNELS, LIVES, reference_survival


def _bodies(n=6, k=3, social=1):
    idents = [cv.identity_seed(0, k, social, i) for i in range(n)]
    return cv.v1_bodies(idents, [k] * n, [social] * n)


def _quiet(bodies, variant, level=0.3):
    bodies.shock[:] = -1
    bodies.rates[:] = np.where(bodies.active, 0.001, 0.0)
    env = cv.V1Body(bodies, variant)
    env.levels[:] = np.where(bodies.active, level, 1.0)
    return env


def _say(b, channel, modifier, variant):
    rows = np.arange(b.n)
    w2 = b.meaning_mod[rows, modifier] if variant != "basket" else b.channel_word[rows, modifier]
    return b.channel_word[rows, channel] * cv.modifier_words(variant) + w2


def test_bands_are_isolated():
    seeds = {cv.identity_seed(bl, k, s, i) for bl in range(cv.BLOCKS) for k, s in cv.CELLS for i in range(cv.IDENTITIES)}
    assert len(seeds) == cv.BLOCKS * len(cv.CELLS) * cv.IDENTITIES
    v0_top = v0.identity_seed(v0.BLOCKS - 1, 5, 9, v0.IDENTITIES - 1)
    learner = set(ss.dev_identities(5_000)[0])
    assert max(ls.dev_identities(ls.DEV_MAX_PER_K)[0]) < v0_top < min(seeds) <= max(seeds) < min(learner)
    assert len(learner) == len(ss.dev_identities(5_000)[0])


def test_convention_ration_and_presence():
    b = _bodies(n=40, k=4, social=2)
    rows = np.arange(b.n)[:, None]
    assert (np.sort(b.word_channel, 1) == np.arange(-1, CHANNELS)).all()
    assert (b.word_channel[rows, b.channel_word] == np.arange(CHANNELS)).all()
    assert (np.sort(b.mod_meaning, 1) == np.arange(3)).all()
    assert (b.mod_meaning[rows, b.meaning_mod] == np.arange(3)).all()
    np.testing.assert_allclose(b.ration, cv.RATION_FACTOR * cv.species_need(b.blocked))
    env = _quiet(b, "size", level=0.9)  # nobody dies, so decisions count up within one life
    seen = []
    for _ in range(9):
        seen.append(bool(env.present()[0]))
        env.step(env.oracle_action())
    assert seen == [True, False, False, False, True, False, False, False, True]


@pytest.mark.parametrize("modifier", range(3))
def test_size_word_sets_the_amount_and_draws_the_ration(modifier):
    b = _bodies()
    channel = b.blocked.argmax(1)
    rows = np.arange(b.n)
    idle = np.full(b.n, -1)
    asked, twin = _quiet(_bodies(), "size"), _quiet(_bodies(), "size")
    asked.step(idle, _say(b, channel, modifier, "size"))
    twin.step(idle)
    gain = asked.levels - twin.levels
    np.testing.assert_allclose(gain[rows, channel], cv.SIZES[modifier])
    assert np.abs(gain).sum() == pytest.approx(cv.SIZES[modifier] * b.n)
    np.testing.assert_allclose(asked.left, b.ration - cv.SIZES[modifier])
    assert (asked.requests[:, modifier] == 1).all() and asked.requests.sum() == b.n


def test_absent_caregiver_and_empty_word_deliver_nothing():
    b = _bodies()
    channel = b.blocked.argmax(1)
    idle = np.full(b.n, -1)
    asked, twin = _quiet(_bodies(), "size"), _quiet(_bodies(), "size")
    asked.step(idle)
    twin.step(idle)
    asked.step(idle, _say(b, channel, 2, "size"))  # decision 1: nobody is there
    twin.step(idle)
    np.testing.assert_allclose(asked.levels, twin.levels)
    empty = np.array([int(np.flatnonzero(row < 0)[0]) for row in b.word_channel])
    spoken, silent = _quiet(_bodies(), "size"), _quiet(_bodies(), "size")
    spoken.step(idle, empty * 3)
    silent.step(idle)
    np.testing.assert_allclose(spoken.levels, silent.levels)
    assert spoken.requests.sum() == 0


def test_when_word_delays_the_portion_and_overflow_is_lost():
    b = _bodies()
    channel = b.blocked.argmax(1)
    rows = np.arange(b.n)
    idle = np.full(b.n, -1)
    asked, twin = _quiet(_bodies(), "when"), _quiet(_bodies(), "when")
    asked.step(idle, _say(b, channel, 1, "when"))  # lands 13 ticks later: in the third window
    twin.step(idle)
    for _ in range(2):
        np.testing.assert_allclose(asked.levels, twin.levels)
        asked.step(idle)
        twin.step(idle)
    np.testing.assert_allclose((asked.levels - twin.levels)[rows, channel], cv.WHEN_PORTION)
    full = _quiet(_bodies(), "when", level=0.9)
    full.step(idle, _say(b, channel, 0, "when"))
    np.testing.assert_allclose(full.wasted, 0.9 + cv.WHEN_PORTION - 1.0)


def test_basket_splits_between_two_named_resources():
    b = _bodies(k=3, social=2)
    social = np.array([np.flatnonzero(row) for row in b.blocked])
    rows = np.arange(b.n)
    idle = np.full(b.n, -1)
    asked, twin = _quiet(_bodies(k=3, social=2), "basket"), _quiet(_bodies(k=3, social=2), "basket")
    asked.step(idle, b.channel_word[rows, social[:, 0]] * cv.WORDS + b.channel_word[rows, social[:, 1]])
    twin.step(idle)
    gain = asked.levels - twin.levels
    np.testing.assert_allclose(gain[rows, social[:, 0]], cv.BASKET[0])
    np.testing.assert_allclose(gain[rows, social[:, 1]], cv.BASKET[1])
    one, twin = _quiet(_bodies(k=3, social=2), "basket"), _quiet(_bodies(k=3, social=2), "basket")
    one.step(idle, b.channel_word[rows, social[:, 0]] * (cv.WORDS + 1))
    twin.step(idle)
    np.testing.assert_allclose((one.levels - twin.levels)[rows, social[:, 0]], sum(cv.BASKET))


def test_without_social_resources_the_silent_speaker_is_probe80s_oracle():
    b = _bodies(n=30, k=3, social=0)
    survived = cv.run_arm(b, "size", "silent")["survived"]
    np.testing.assert_array_equal(survived, reference_survival(b, "oracle", np.random.default_rng(0)))


def test_reckoning_with_own_rates_is_exact_and_species_rates_drift():
    b = _bodies(n=40, k=3, social=1)
    for arm, exact in (("reckoned", True), ("species", False)):
        env = cv.V1Body(b, "size")
        reckoner = cv.Reckoner(env, np.where(b.active, cv.BASE_RATES, 0.0) if arm == "species" else b.rates)
        worst = 0.0
        for _ in range(60):
            env.step(*cv.speaker(env, arm, reckoner.est))
            reckoner.update(env)
            live = ~env.finished
            worst = max(worst, float(np.abs(reckoner.est - env.levels)[live].max()))
        assert (worst < 1e-9) == exact


def test_survey_is_deterministic_and_criteria_are_computed():
    kwargs = dict(blocks=2, identities=8, cells_=((3, 1),))
    first = cv.survey("size", **kwargs)
    assert first == cv.survey("size", **kwargs)
    cell = first["cells"]["K3_social1"]
    assert cell["survival"]["composed"] > cell["survival"]["silent"] + 0.3
    assert set(first["criteria"]) >= {"1_speech", "3_second_word", "4_self_knowledge", "pass"}
    with pytest.raises(ValueError):
        cv.speaker(cv.V1Body(_bodies(), "size"), "shouting")


# ---------------------------------------------------------------- learner
def test_training_bodies_are_valid_worlds():
    b = ss.sample_v1_bodies(np.random.default_rng(3), 400)
    rows = np.arange(b.n)[:, None]
    assert not (b.blocked & ~b.active).any()
    assert (b.blocked.sum(1) <= np.minimum(ss.SOCIAL_MAX, b.k - 1)).all()
    assert set(np.unique(b.blocked.sum(1))) == {0, 1, 2}
    assert (np.sort(b.word_channel, 1) == np.arange(-1, CHANNELS)).all()
    assert (b.word_channel[rows, b.channel_word] == np.arange(CHANNELS)).all()
    assert (b.mod_meaning[rows, b.meaning_mod] == np.arange(3)).all()


def test_warm_start_keeps_the_body_policy(tmp_path):
    old = ls.Trainer(ls.TrainConfig(envs=4, horizon=4, hidden=32))
    old.save(tmp_path)
    new = ss.Agent(32)
    ss.warm_start(new, tmp_path)
    import mlx.core as mx
    rng = np.random.default_rng(0)
    x_old = rng.normal(size=(5, 7, ls.D_OBS)).astype(np.float32)
    x_new = np.concatenate([x_old, rng.normal(size=(5, 7, ss.D_OBS - ls.D_OBS)).astype(np.float32)], -1)
    wipe = mx.zeros((5, 7))
    h_old = old.model.unroll(mx.array(x_old), mx.zeros((5, 32)), wipe)
    h_new = new.unroll(mx.array(x_new), mx.zeros((5, 32)), wipe)
    np.testing.assert_allclose(np.array(old.model.pi(h_old)), np.array(new.pi(h_new)), atol=1e-5)


def test_speaking_trainer_runs_saves_resumes_and_evaluates(tmp_path):
    trainer = ss.Trainer(ss.TrainConfig(envs=16, horizon=8, hidden=32))
    record = trainer.train_step()
    assert np.isfinite([record[k] for k in ss.AUX]).all() and 0.0 <= record["speak_rate"] <= 1.0
    trainer.save(tmp_path / "ck", {"next_eval": 0, "evaluations": 0})
    again, state = ss.Trainer.load(tmp_path / "ck")
    assert state["decisions"] == trainer.decisions == 16 * 8
    bodies = cv.v1_bodies(*ss.dev_identities(3, cells=((3, 1), (4, 2))))
    result = ss.evaluate_all(again.model, bodies, 0)
    assert set(result) == set(ss.LESIONS)
    assert set(result["none"]) == {"K3_social1", "K4_social2"}
    env = cv.V1Body(bodies, "size")
    assert env.requests.sum() == 0
    with pytest.raises(ValueError):
        ss.evaluate(again.model, bodies, np.random.default_rng(0), "deafen")


def test_muted_learner_gets_no_deliveries():
    model = ss.Agent(16)
    bodies = cv.v1_bodies(*ss.dev_identities(4, cells=((3, 1),)))
    import mlx.core as mx
    mx.eval(model.parameters())
    model.talk.bias = mx.array([-20.0, 20.0])  # always tries to speak
    out = ss.evaluate(model, bodies, np.random.default_rng(1), "mute")["K3_social1"]
    assert out["speak_rate"] > 0.99 and out["wasted_per_life"] == 0.0
    assert ACTUATORS == 10 and LIVES == 6
