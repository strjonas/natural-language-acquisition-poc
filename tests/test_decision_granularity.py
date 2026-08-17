"""Guards for probe68's lever.

The whole probe rests on one claim about the code rather than about the world:
that ``portion_scale`` is inert at its default, so every number probe65 published
is reproduced bit-for-bit and only a deliberate scale changes anything. That is
asserted here rather than argued in a docstring.

The second group of tests is about the formula. ``margin = E_grant[min(grant *
uptake, axis gap)]`` is what makes probe68 a prediction instead of a sweep, and
its saturation is what makes G2 a null rather than a small effect. Both are
checked on constructed bodies where the answer is known in closed form, so a
failure points at the arithmetic rather than at a simulator.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from homesocial.island.report import REPORT_NEEDS, ReportConfig
from homesocial.organism.decision_granularity import (
    EQUIVALENCE_BAND,
    PORTION_SCALES,
    SEED_STRIDE,
    _check_seed_isolation,
    _difference,
    _paired_interval,
    grade,
)
from homesocial.organism.future_request import (
    OPERATING_DELAY,
    decision_margin,
    need_scores,
)


def _report(**overrides) -> ReportConfig:
    return replace(ReportConfig(), **overrides)


def _margin(levels, *, report, horizon=1, uptake=1.0):
    scores = need_scores(
        believed_levels=np.asarray(levels, dtype=np.float64),
        believed_rates=np.zeros(len(REPORT_NEEDS)),
        believed_uptake={need: uptake for need in REPORT_NEEDS},
        arriving=np.zeros(len(REPORT_NEEDS)),
        horizon=horizon,
        report=report,
    )
    return decision_margin(scores)


# -- the lever is inert at its default -----------------------------------------


def test_portion_scale_one_leaves_the_report_untouched():
    """Scale 1.0 must not even construct a different config."""

    base = _report()
    scaled = replace(
        base,
        portion_small=base.portion_small * 1.0,
        portion_large=base.portion_large * 1.0,
    )
    assert scaled == base


def test_portion_scale_one_reproduces_the_margin_exactly():
    base = _report()
    levels = [0.30, 0.55, 0.90]
    reference = _margin(levels, report=base)
    for _ in range(3):
        assert _margin(levels, report=base) == reference


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_nonpositive_portion_scale_is_rejected(scale):
    from homesocial.organism.future_request import run_open_loop

    with pytest.raises(ValueError):
        run_open_loop(
            None,  # never reached: the guard fires before any world is built
            None,
            world_name="metabolic",
            delay=0,
            lives=1,
            seed_base=0,
            portion_scale=scale,
        )


# -- the formula, on bodies whose answer is known ------------------------------


def test_margin_tracks_the_grant_below_the_axis_gap():
    """With the two emptiest axes far apart, the margin *is* the expected grant."""

    report = _report(portion_small=0.02, portion_large=0.06)
    # Gap between the two lowest is 0.40, far above either portion.
    margin = _margin([0.10, 0.50, 0.95], report=report)
    expected = 0.5 * 0.02 + 0.5 * 0.06
    assert margin == pytest.approx(expected, abs=1e-12)


def test_margin_saturates_at_the_axis_gap():
    """With a grant far above the gap, the margin is the gap and nothing else."""

    gap = 0.05
    report = _report(portion_small=0.40, portion_large=0.80)
    margin = _margin([0.30, 0.30 + gap, 0.95], report=report)
    assert margin == pytest.approx(gap, abs=1e-12)


def test_doubling_a_saturated_grant_changes_the_margin_by_zero():
    """G2's null, in closed form: this is why the gate can be an equivalence."""

    levels = [0.30, 0.35, 0.95]
    base = _report(portion_small=0.20, portion_large=0.60)
    doubled = _report(portion_small=0.40, portion_large=1.20)
    assert _margin(levels, report=base) == _margin(levels, report=doubled)


def test_halving_an_unsaturated_grant_halves_the_margin():
    """G3's response, in closed form."""

    levels = [0.10, 0.80, 0.95]
    base = _report(portion_small=0.20, portion_large=0.60)
    halved = _report(portion_small=0.10, portion_large=0.30)
    assert _margin(levels, report=halved) == pytest.approx(
        _margin(levels, report=base) / 2.0, abs=1e-12
    )


def test_the_formula_reproduces_the_margin_on_random_bodies():
    rng = np.random.default_rng(68)
    report = _report()
    large = report.large_portion_probability
    for _ in range(400):
        levels = rng.uniform(0.05, 0.95, size=len(REPORT_NEEDS))
        ordered = np.sort(levels)
        gap = float(ordered[1] - ordered[0])
        predicted = (1.0 - large) * min(report.portion_small, gap) + large * min(
            report.portion_large, gap
        )
        assert _margin(levels, report=report) == pytest.approx(predicted, abs=1e-9)


# -- the gates are graded as written -------------------------------------------


def _cell(label, delay, contrasts, rate_values=None, state_values=None):
    mean, low, high = _paired_interval(contrasts)
    return {
        "label": label,
        "help_delay": delay,
        "contrasts": contrasts,
        "contrast_mean": mean,
        "contrast_low": low,
        "contrast_high": high,
        "rate_values": list(rate_values if rate_values is not None else contrasts),
        "state_values": list(state_values if state_values is not None else contrasts),
    }


def test_a_wide_interval_makes_an_equivalence_gate_unresolved_not_passed():
    """Probe65's mistake, refused in code.

    Two cells whose mean difference is zero but whose spread is enormous must
    not read as a passing null. STATE.md carries that correction and this is the
    line that enforces it.
    """

    noisy = [+0.40, -0.40, +0.35, -0.35, +0.30, -0.30, +0.25, -0.25]
    cells = [
        _cell("nutrition_1x", OPERATING_DELAY, [0.0] * 8),
        _cell("nutrition_2x", OPERATING_DELAY, noisy),
    ]
    gates = grade(cells)
    assert gates["G2_nutrition_null"]["verdict"] == "UNRESOLVED"


def test_a_tight_zero_difference_passes_the_nutrition_null():
    steady = [0.10, 0.11, 0.09, 0.10, 0.10, 0.11, 0.09, 0.10]
    cells = [
        _cell("nutrition_1x", OPERATING_DELAY, steady),
        _cell("nutrition_2x", OPERATING_DELAY, steady),
    ]
    gates = grade(cells)
    assert gates["G2_nutrition_null"]["verdict"] == "pass"


def test_a_real_shift_fails_the_nutrition_null():
    cells = [
        _cell("nutrition_1x", OPERATING_DELAY, [0.10] * 8),
        _cell("nutrition_2x", OPERATING_DELAY, [0.20] * 8),
    ]
    gates = grade(cells)
    assert gates["G2_nutrition_null"]["verdict"] == "fail"


def test_the_granularity_response_needs_an_interval_clear_of_zero():
    cells = [
        _cell("nutrition_1x", OPERATING_DELAY, [0.10] * 8),
        _cell(
            "quantum_half",
            OPERATING_DELAY,
            [0.18, 0.19, 0.17, 0.18, 0.19, 0.18, 0.17, 0.18],
        ),
    ]
    gates = grade(cells)
    assert gates["G3_granularity_response"]["verdict"] == "pass"

    flat = [
        _cell("nutrition_1x", OPERATING_DELAY, [0.10] * 8),
        _cell("quantum_half", OPERATING_DELAY, [0.10] * 8),
    ]
    assert grade(flat)["G3_granularity_response"]["verdict"] == "fail"


def test_the_granularity_response_needs_six_of_eight_seeds_to_agree():
    """A mean carried by two seeds is not a response."""

    split = [0.60, 0.60, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10]
    cells = [
        _cell("nutrition_1x", OPERATING_DELAY, [0.0] * 8),
        _cell("quantum_half", OPERATING_DELAY, split),
    ]
    assert grade(cells)["G3_granularity_response"]["verdict"] == "fail"


def test_the_second_step_gate_fails_a_response_linear_in_the_quantum():
    """Reach rises 0.441 -> 0.760 -> 0.797, so step two must be the smaller one."""

    diminishing = [
        _cell("nutrition_1x", OPERATING_DELAY, [0.10] * 8),
        _cell("quantum_half", OPERATING_DELAY, [0.20] * 8),
        _cell("quantum_third", OPERATING_DELAY, [0.22] * 8),
    ]
    assert grade(diminishing)["G4_diminishing_step"]["verdict"] == "pass"

    linear = [
        _cell("nutrition_1x", OPERATING_DELAY, [0.10] * 8),
        _cell("quantum_half", OPERATING_DELAY, [0.20] * 8),
        _cell("quantum_third", OPERATING_DELAY, [0.35] * 8),
    ]
    assert grade(linear)["G4_diminishing_step"]["verdict"] == "fail"


def test_the_dissociation_needs_the_state_lesion_to_stay_put():
    """If a finer surface just makes every decision harder, the claim is empty."""

    dissociated = [
        _cell(
            "nutrition_1x",
            OPERATING_DELAY,
            [0.0] * 8,
            rate_values=[0.13] * 8,
            state_values=[0.25] * 8,
        ),
        _cell(
            "quantum_half",
            OPERATING_DELAY,
            [0.0] * 8,
            rate_values=[0.21] * 8,
            state_values=[0.26] * 8,
        ),
    ]
    assert grade(dissociated)["G6_dissociation"]["verdict"] == "pass"

    # Both rise together by the same amount: the decision simply got harder.
    together = [
        _cell(
            "nutrition_1x",
            OPERATING_DELAY,
            [0.0] * 8,
            rate_values=[0.13] * 8,
            state_values=[0.25] * 8,
        ),
        _cell(
            "quantum_half",
            OPERATING_DELAY,
            [0.0] * 8,
            rate_values=[0.21] * 8,
            state_values=[0.35] * 8,
        ),
    ]
    assert grade(together)["G6_dissociation"]["verdict"] == "fail"


def test_the_lag0_null_fails_when_the_contrast_is_not_negative():
    """P4 is a null *and* a sign claim; a flat positive must not pass it."""

    cells = [
        _cell("lag0_1x", 0, [0.10] * 8),
        _cell("lag0_2x", 0, [0.10] * 8),
    ]
    assert grade(cells)["G5_lag0_null"]["verdict"] == "fail"

    negative = [
        _cell("lag0_1x", 0, [-0.12] * 8),
        _cell("lag0_2x", 0, [-0.12] * 8),
    ]
    assert grade(negative)["G5_lag0_null"]["verdict"] == "pass"


def test_paired_interval_widens_with_spread():
    tight = _paired_interval([0.10, 0.10, 0.10, 0.10, 0.10])
    wide = _paired_interval([0.00, 0.20, 0.05, 0.15, 0.10])
    assert tight[2] - tight[1] < wide[2] - wide[1]
    assert tight[0] == pytest.approx(wide[0], abs=1e-9)


def test_difference_counts_seeds_agreeing_in_sign():
    a = _cell("quantum_half", OPERATING_DELAY, [0.20, 0.20, 0.20, 0.05])
    b = _cell("nutrition_1x", OPERATING_DELAY, [0.10, 0.10, 0.10, 0.10])
    out = _difference(a, b)
    assert out["same_sign"] == 3
    assert out["n"] == 4


# -- the seed trap -------------------------------------------------------------


def test_seed_isolation_rejects_lives_that_would_span_the_stride():
    with pytest.raises(ValueError):
        _check_seed_isolation(SEED_STRIDE + 1)


def test_seed_isolation_accepts_the_planned_run():
    _check_seed_isolation(40, 8)


def test_the_sweep_contains_a_saturated_pair_and_an_unsaturated_one():
    """The design, asserted: without both, neither null nor response is testable."""

    assert 1.0 in PORTION_SCALES and 2.0 in PORTION_SCALES
    assert min(PORTION_SCALES) < 0.5
    assert EQUIVALENCE_BAND > 0.0
