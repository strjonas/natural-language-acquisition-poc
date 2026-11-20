"""Mechanics of the caregiver-v0 ceiling instrument."""

import numpy as np
import pytest

from homesocial.organism import active_self as probe
from homesocial.organism import caregiver_world as cw
from homesocial.organism import learned_self as ls
from homesocial.organism.vec_body import CHANNELS, PHYSICS, reference_survival


def _bodies(n=6, k=3, social=1):
    idents = [cw.identity_seed(0, k, social, i) for i in range(n)]
    return cw.caregiver_bodies(idents, [k] * n, [social] * n)


def _quiet_env(bodies, level=0.5):
    """No shocks, one slow rate and one level everywhere: over the few windows
    these tests run, nothing reaches the floor or the ceiling."""

    bodies.shock[:] = -1
    bodies.rates[:] = np.where(bodies.active, 0.01, 0.0)
    env = cw.CaregiverBody(bodies)
    env.levels[:] = np.where(bodies.active, level, 1.0)
    return env


def _request(bodies, channel, large):
    rows = np.arange(bodies.n)
    size_word = np.where(large, bodies.large_word, 1 - bodies.large_word)
    return cw.ACTUATORS + bodies.channel_word[rows, channel] * cw.SIZES + size_word


def test_survey_band_is_isolated():
    seeds = {cw.identity_seed(b, k, s, i) for b in range(cw.BLOCKS) for k, s in cw.CELLS for i in range(cw.IDENTITIES)}
    assert len(seeds) == cw.BLOCKS * len(cw.CELLS) * cw.IDENTITIES
    learned = set(ls.dev_identities(ls.DEV_MAX_PER_K)[0])
    assert not seeds & learned
    assert min(seeds) > max(learned) > probe.DEVELOPMENT_BASE + 100_000_000
    with pytest.raises(ValueError):
        cw.identity_seed(cw.BLOCKS, 2, 1, 0)


def test_convention_and_social_resources():
    b = _bodies(n=40, k=4, social=2)
    rows = np.arange(b.n)[:, None]
    assert (np.sort(b.word_channel, 1) == np.arange(-1, CHANNELS)).all()  # five channels named, one empty word
    assert (b.word_channel[rows, b.channel_word] == np.arange(CHANNELS)).all()
    assert set(np.unique(b.large_word)) == {0, 1}
    assert (b.blocked & ~b.active).sum() == 0 and (b.blocked.sum(1) == 2).all()
    assert (b.actionable().sum(1) == 2 * (4 - 2)).all()  # its own actuators cannot reach a social resource
    with pytest.raises(ValueError):
        _bodies(n=1, k=2, social=2)


def test_small_portion_arrives_next_tick_and_large_after_the_delay():
    b = _bodies()
    channel = b.blocked.argmax(1)
    rows = np.arange(b.n)
    idle = np.full(b.n, -1)
    for large, windows in ((False, 1), (True, 3)):
        asked, twin = _quiet_env(_bodies()), _quiet_env(_bodies())
        asked.step(_request(b, channel, np.full(b.n, large)))
        twin.step(idle)
        for _ in range(windows - 1):
            np.testing.assert_allclose(asked.levels, twin.levels)  # nothing has arrived yet
            assert (asked.due_tick >= 0).all()
            asked.step(idle)
            twin.step(idle)
        gain = asked.levels - twin.levels
        portion = PHYSICS.portion_large if large else PHYSICS.portion_small
        np.testing.assert_allclose(gain[rows, channel], portion)
        assert np.abs(gain).sum() == pytest.approx(portion * b.n)
        assert (asked.due_tick < 0).all()
        assert (asked.requests[:, int(large)] == 1).all() and asked.requests.sum() == b.n


def test_busy_caregiver_and_empty_word_change_nothing():
    b = _bodies()
    channel = b.blocked.argmax(1)
    idle = np.full(b.n, -1)
    asked, twin = _quiet_env(_bodies()), _quiet_env(_bodies())
    large = _request(b, channel, np.ones(b.n, bool))
    asked.step(large)
    twin.step(large)
    asked.step(_request(b, channel, np.zeros(b.n, bool)))  # asks again while the large portion is under way
    twin.step(idle)
    np.testing.assert_allclose(asked.levels, twin.levels)
    assert asked.requests.sum() == b.n
    empty = np.array([int(np.flatnonzero(row < 0)[0]) for row in b.word_channel])
    silent, spoken = _quiet_env(_bodies()), _quiet_env(_bodies())
    silent.step(idle)
    spoken.step(cw.ACTUATORS + empty * cw.SIZES)
    np.testing.assert_allclose(spoken.levels, silent.levels)
    assert spoken.requests.sum() == 0


def test_without_social_resources_the_silent_speaker_is_probe80s_oracle():
    b = _bodies(n=30, k=3, social=0)
    survived = cw.run_arm(b, "silent")["survived"]
    np.testing.assert_array_equal(survived, reference_survival(b, "oracle", np.random.default_rng(0)))


def test_survey_is_deterministic_and_words_must_be_right():
    kwargs = dict(blocks=2, identities=12, cells=((3, 1),))
    first = cw.survey(**kwargs)
    assert first == cw.survey(**kwargs)
    cell = first["cells"]["K3_social1"]
    assert set(cell["survival"]) == set(cw.ARMS)
    assert cell["survival"]["composed"] > cell["survival"]["silent"] + 0.3
    assert cell["survival"]["composed"] > cell["survival"]["scrambled"] + 0.3
    assert "composed" not in cell["composed_minus"]
    with pytest.raises(ValueError):
        cw.speaker_action(cw.CaregiverBody(_bodies()), "shouting")
