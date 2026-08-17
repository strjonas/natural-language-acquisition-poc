"""Guards for probe67's uncertainty machinery.

Three kinds of test, in the order that matters.

The first kind protects everything already published: probe67's self-model has
to be probe63's, bit-identically. ``UncertainSelfCalibration`` only reads
quantities ``RecursiveSelfCalibration`` was already computing, so if any point
estimate anywhere moves by a float, probe65's tables silently mean something
else and this catches it.

The second kind protects the arithmetic. ``expected_min`` is the one genuinely
new numerical object here, so it is checked against a closed form it must match,
against the degenerate case it must reduce to, and against itself at eight times
the resolution -- so that no result can turn on the node count.

The third kind protects the claim. Jensen says uncertainty can only lower a
score, and a *uniform* uncertainty must therefore leave the argmax exactly where
it was: if a flat sigma could reorder the words, the survey's ``flat`` control
would not be a control. And the sigma must be zero before the organism has seen
itself, or the mechanism is reading a prior it was never given.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from random import Random

import numpy as np
import pytest

from homesocial.island.report import REPORT_NEEDS, ReportConfig, ReportWorld
from homesocial.island.world import IslandConfig
from homesocial.organism.future_request import need_scores
from homesocial.organism.individual_self import _true_body
from homesocial.organism.train import OrganismConfig
from homesocial.organism.self_calibration import RecursiveSelfCalibration
from homesocial.organism import uncertain_horizon
from homesocial.organism.uncertain_horizon import (
    ARM_NAMES,
    MIN_READINGS_FOR_SIGMA,
    QUADRATURE_NODES,
    UncertainSelfCalibration,
    _normal_upper,
    expected_min,
    uncertain_need_scores,
)


REPORT = ReportConfig(life_steps=240)
_ORGANISM = OrganismConfig()


@contextmanager
def _nodes(count: int):
    """Re-resolve the quadrature everywhere, including inside the scoring rule."""

    previous = uncertain_horizon.QUADRATURE_NODES
    uncertain_horizon.QUADRATURE_NODES = count
    try:
        yield
    finally:
        uncertain_horizon.QUADRATURE_NODES = previous


# -- the arithmetic ------------------------------------------------------------


def test_normal_upper_matches_the_error_function():
    """The A&S approximation has to be far finer than anything reported."""

    import math

    grid = np.linspace(-6.0, 6.0, 401)
    exact = np.asarray([0.5 * math.erfc(z / math.sqrt(2.0)) for z in grid])
    assert np.max(np.abs(_normal_upper(grid) - exact)) < 1e-7


def test_expected_min_of_certain_values_is_the_minimum():
    mu = np.asarray([0.4, 0.7, 0.2])
    assert expected_min(mu, np.zeros(3)) == pytest.approx(0.2, abs=0.0)


def test_expected_min_matches_the_closed_form_for_two_normals():
    """``E[min(X,Y)] = mu_x + mu_y)/2 - E|X-Y|/2`` for independent normals."""

    import math

    mu = np.asarray([0.30, 0.50])
    sigma = np.asarray([0.04, 0.09])
    difference_sd = math.sqrt(float(sigma[0] ** 2 + sigma[1] ** 2))
    delta = float(mu[0] - mu[1])
    z = delta / difference_sd
    # E|D| for D ~ N(delta, s^2).
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * math.erfc(-z / math.sqrt(2.0))
    mean_absolute = delta * (2.0 * cdf - 1.0) + 2.0 * difference_sd * density
    expected = 0.5 * float(mu.sum()) - 0.5 * mean_absolute
    assert expected_min(mu, sigma, cap=10.0) == pytest.approx(expected, abs=1e-6)


def test_expected_min_does_not_depend_on_the_node_count():
    """The quadrature resolution must be accuracy, never a knob.

    Two claims, and the second is the one that matters. The value agrees with
    itself at eight times the resolution to 1e-5 -- a five-hundredth of
    ``CONSEQUENTIAL_MARGIN``, the smallest difference this probe ever calls
    meaningful. And the *decision* agrees exactly, over a thousand draws: the
    rule only ever reads an ``argmax``, so a residual of 1e-6 in a score is only
    a knob if it can reorder one, and it never does.
    """

    rng = np.random.default_rng(7)
    for _ in range(200):
        mu = rng.uniform(-0.6, 1.0, size=3)
        sigma = rng.uniform(0.0, 0.15, size=3)
        coarse = expected_min(mu, sigma, nodes=QUADRATURE_NODES)
        fine = expected_min(mu, sigma, nodes=8 * QUADRATURE_NODES + 1)
        assert coarse == pytest.approx(fine, abs=1e-5)


def test_the_node_count_never_changes_a_word():
    rng = np.random.default_rng(31)
    for _ in range(1000):
        kwargs = _inputs(rng)
        sigma = rng.uniform(0.0, 0.004, size=3)
        coarse = uncertain_need_scores(
            believed_rate_sigma=sigma, horizon=18, **kwargs
        )
        with _nodes(8 * QUADRATURE_NODES + 1):
            fine = uncertain_need_scores(
                believed_rate_sigma=sigma, horizon=18, **kwargs
            )
        assert int(np.argmax(coarse)) == int(np.argmax(fine))


def test_expected_min_respects_the_cap():
    mu = np.asarray([3.0, 4.0, 5.0])
    assert expected_min(mu, np.asarray([0.01, 0.01, 0.01])) == pytest.approx(1.0)


def test_uncertainty_can_only_lower_the_expected_minimum():
    """Jensen, checked rather than asserted, because the whole design rests on it."""

    rng = np.random.default_rng(11)
    for _ in range(200):
        mu = rng.uniform(-0.5, 0.9, size=3)
        sigma = rng.uniform(0.0, 0.2, size=3)
        # 1e-5 is the quadrature residual, not a tolerance on the claim: the
        # certain value is exact and the uncertain one is integrated.
        assert expected_min(mu, sigma) <= expected_min(mu, np.zeros(3)) + 1e-5


# -- reduction to probe65 ------------------------------------------------------


def _inputs(rng):
    return dict(
        believed_levels=rng.uniform(0.05, 0.95, size=3),
        believed_rates=rng.uniform(0.002, 0.02, size=3),
        believed_uptake={need: float(rng.uniform(0.7, 1.3)) for need in REPORT_NEEDS},
        arriving=rng.uniform(0.0, 0.2, size=3),
        report=REPORT,
    )


def test_zero_sigma_is_bit_identical_to_probe65():
    """Not approximately. The default has to be the published rule exactly."""

    rng = np.random.default_rng(3)
    for _ in range(100):
        kwargs = _inputs(rng)
        for horizon in (1, 6, 18, 24):
            reference = need_scores(horizon=horizon, **kwargs)
            actual = uncertain_need_scores(
                believed_rate_sigma=np.zeros(3), horizon=horizon, **kwargs
            )
            assert np.array_equal(actual, reference)


def test_a_flat_sigma_cannot_reorder_the_words():
    """The ``flat`` control is only a control if this holds.

    A uniform width lowers all three scores, but it must not change which one is
    largest -- otherwise ``flat`` beating ``point`` would be evidence for
    something, and it would not be self-knowledge.
    """

    rng = np.random.default_rng(23)
    reordered = 0
    for _ in range(300):
        kwargs = _inputs(rng)
        point = uncertain_need_scores(
            believed_rate_sigma=np.zeros(3), horizon=18, **kwargs
        )
        width = float(rng.uniform(0.001, 0.01))
        flat = uncertain_need_scores(
            believed_rate_sigma=np.full(3, width), horizon=18, **kwargs
        )
        assert np.all(flat <= point + 1e-9)
        reordered += int(int(np.argmax(flat)) != int(np.argmax(point)))
    assert reordered == 0


def test_an_asymmetric_sigma_can_reorder_the_words():
    """And the mechanism is dead if this never happens.

    Two needs projecting to nearly the same place, one of them known well and one
    badly: the concavity of ``min`` has to prefer the one it is sure about.
    """

    scores = uncertain_need_scores(
        believed_levels=np.asarray([0.50, 0.50, 0.95]),
        believed_rates=np.asarray([0.010, 0.010, 0.001]),
        believed_rate_sigma=np.asarray([0.0002, 0.0060, 0.0000]),
        believed_uptake={need: 1.0 for need in REPORT_NEEDS},
        arriving=np.zeros(3),
        horizon=18,
        report=REPORT,
    )
    point = uncertain_need_scores(
        believed_levels=np.asarray([0.50, 0.50, 0.95]),
        believed_rates=np.asarray([0.010, 0.010, 0.001]),
        believed_rate_sigma=np.zeros(3),
        believed_uptake={need: 1.0 for need in REPORT_NEEDS},
        arriving=np.zeros(3),
        horizon=18,
        report=REPORT,
    )
    assert int(np.argmax(point)) != int(np.argmax(scores))
    assert int(np.argmax(scores)) == 1


def test_the_horizon_scales_the_effect_of_uncertainty():
    """The whole story: the same sigma has to matter more further ahead."""

    kwargs = dict(
        believed_levels=np.asarray([0.6, 0.6, 0.9]),
        believed_rates=np.asarray([0.008, 0.008, 0.002]),
        believed_uptake={need: 1.0 for need in REPORT_NEEDS},
        arriving=np.zeros(3),
        report=REPORT,
    )
    sigma = np.asarray([0.0005, 0.0040, 0.0000])
    gaps = []
    for horizon in (1, 18):
        point = uncertain_need_scores(
            believed_rate_sigma=np.zeros(3), horizon=horizon, **kwargs
        )
        wide = uncertain_need_scores(
            believed_rate_sigma=sigma, horizon=horizon, **kwargs
        )
        gaps.append(float(np.abs(point - wide).max()))
    assert gaps[1] > 10.0 * gaps[0]


# -- the self-model is still probe63's -----------------------------------------


def _drive(calibration, *, seed: int, steps: int = 200):
    """One synthetic life of updates, identical for whichever model is passed."""

    world = ReportWorld(IslandConfig(), report=REPORT, seed=seed)
    packet = world.reset(seed)
    calibration.reset(packet, world)
    rng = Random(seed)
    actions = ["move_forward", "consume", "rest", "wait", "turn_left"]
    points = []
    for step in range(steps):
        before = packet
        action = rng.choice(actions)
        packet, _, terminated, truncated, info = world.step(action)
        reading = tuple(_true_body(world)) if step % 7 == 0 else None
        calibration.update(before, 0, packet, world, reading)
        points.append(calibration.point().copy())
        if terminated or truncated:
            break
    return points


def test_the_uncertain_model_is_the_published_one_to_the_last_bit():
    for seed in (3, 17, 41):
        reference = _drive(RecursiveSelfCalibration(_ORGANISM, REPORT), seed=seed)
        actual = _drive(UncertainSelfCalibration(_ORGANISM, REPORT), seed=seed)
        assert len(reference) == len(actual)
        for left, right in zip(reference, actual):
            assert np.array_equal(left, right)


def test_sigma_is_silent_until_the_organism_has_seen_itself():
    model = UncertainSelfCalibration(_ORGANISM, REPORT)
    world = ReportWorld(IslandConfig(), report=REPORT, seed=5)
    packet = world.reset(5)
    model.reset(packet, world)
    assert np.all(model.rate_sigma(0.5) == 0.0)

    rng = Random(5)
    readings = 0
    for step in range(300):
        before = packet
        packet, _, terminated, truncated, info = world.step(
            rng.choice(["move_forward", "consume", "rest", "wait"])
        )
        give = step % 5 == 0
        model.update(before, 0, packet, world, tuple(_true_body(world)) if give else None)
        readings += int(give)
        if readings < MIN_READINGS_FOR_SIGMA:
            assert np.all(model.rate_sigma(0.5) == 0.0)
        if terminated or truncated:
            break
    assert np.any(model.rate_sigma(0.5) > 0.0)
    assert np.all(np.isfinite(model.rate_sigma(0.5)))


def test_every_arm_name_has_a_policy():
    from homesocial.organism.uncertain_horizon import ARM_SPEC

    assert tuple(ARM_SPEC) == ARM_NAMES
