import math

import numpy as np
import pytest

from homesocial.organism.self_posterior import (
    ACTUATORS,
    CHANNELS,
    PosteriorConfig,
    SelfPosterior,
    _P,
    _X,
    log_norm_cdf,
)

SMALL = PosteriorConfig(particles=64)


def test_config_rejects_impossible_values():
    with pytest.raises(ValueError):
        PosteriorConfig(particles=1)
    with pytest.raises(ValueError):
        PosteriorConfig(hazard=1.0)
    with pytest.raises(ValueError):
        PosteriorConfig(shock_attribution="oracle")


def test_log_norm_cdf_matches_erfc():
    for z in (-8.0, -3.0, -1.0, 0.0, 0.5, 2.0, 6.0):
        reference = math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
        assert float(log_norm_cdf(z)) == pytest.approx(reference, rel=1e-6, abs=1e-12)
    assert np.isfinite(log_norm_cdf(-40.0))


def _synthetic_life(steps=60):
    """One active channel at 0.6 with rate 0.01, actuator 3 restores 0.2.
    Returns (total, minimum) pairs exactly as the sensor would produce them."""

    level, rows = 0.6, []
    for t in range(1, steps + 1):
        grant = 3 if t in (7, 31) else None
        if grant is not None:
            level = min(1.0, level + 0.2)
        level -= 0.01
        rows.append((grant, (CHANNELS - 1) + level, level))
    return rows


def _replay(rows, seed=0, relabel=None, config=SMALL):
    posterior = SelfPosterior(config, np.random.default_rng(seed))
    posterior.birth((CHANNELS - 1) + 0.6, 0.6, retain=False)
    for grant, total, minimum in rows:
        if grant is not None and relabel is not None:
            grant = relabel[grant]
        posterior.observe(grant, total, minimum)
    return posterior


def test_learns_portion_rate_and_single_resource_from_two_scalars():
    posterior = _replay(_synthetic_life())
    w = posterior.weights()
    index = np.arange(posterior.n)
    drive = posterior.drives[:, 3]
    assert (drive >= 0).all()
    assert float((w * posterior.active[index, drive]).sum()) > 0.99
    portion = float((w * posterior.mean[:, _P[3]]).sum())
    assert portion == pytest.approx(0.2, abs=0.01)
    report = posterior.structure_report()
    assert 0.0 <= report.confidence <= 1.0
    assert sum(report.count_posterior) == pytest.approx(1.0)
    assert report.resource_count == 1
    assert report.tried == (3,)


def test_minimum_is_respected_after_truncation():
    posterior = _replay(_synthetic_life(30))
    minimum = _synthetic_life(30)[-1][2]
    levels = np.where(posterior.active, posterior.mean[:, _X], np.inf)
    assert (levels.min(axis=1) >= minimum - 0.02).all()


def test_same_history_same_random_stream_is_deterministic():
    a = _replay(_synthetic_life(), seed=5)
    b = _replay(_synthetic_life(), seed=5)
    assert a.evidence == b.evidence
    assert a.structure_report() == b.structure_report()


def test_consistent_actuator_relabeling_transports_the_inference_exactly():
    rows = _synthetic_life()
    rows = [(g if g is None else g, t, m) for g, t, m in rows]
    permutation = list(range(ACTUATORS))
    permutation[3], permutation[8] = 8, 3
    plain = _replay(rows, seed=11).structure_report().partition
    moved = _replay(rows, seed=11, relabel=permutation).structure_report().partition
    assert moved == tuple(sorted(tuple(sorted(permutation[a] for a in g)) for g in plain))


def test_observe_requires_birth_and_known_actuator():
    posterior = SelfPosterior(SMALL, np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        posterior.observe(None, 4.5, 0.5)
    posterior.birth(4.6, 0.6, retain=False)
    with pytest.raises(ValueError):
        posterior.observe(ACTUATORS, 4.6, 0.6)


def test_actions_are_valid_actuators():
    posterior = _replay(_synthetic_life(12))
    for _ in range(20):
        assert 0 <= posterior.sample_action() < ACTUATORS
        assert 0 <= posterior.certainty_action() < ACTUATORS


def test_mixture_shock_attribution_runs_and_differs_only_in_shock_handling():
    rows = _synthetic_life(20)
    shocked = list(rows)
    grant, total, minimum = shocked[12]
    shocked[12] = (grant, total - 0.25, minimum - 0.25)
    shocked[13:] = [(g, t - 0.25, m - 0.25) for g, t, m in shocked[13:]]
    for mode in ("mixture", "sample"):
        posterior = _replay(shocked, config=PosteriorConfig(particles=64, shock_attribution=mode))
        assert np.isfinite(posterior.evidence)
