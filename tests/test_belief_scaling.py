"""Guards for probe69's comparison.

Probe69 makes one claim about the code rather than about the world, and every
number it reports depends on it: that the two families are scored by the *same*
instrument, so a difference between them is a difference in belief and not in
bookkeeping. `causal_self` imports `_balanced_accuracy` from `self_belief`
rather than defining its own, and that import is asserted here rather than
trusted -- a divergent copy would be exactly `CLAUDE.md`'s documentation-drift
trap, with the drift inside the code instead of a doc.

The second group is about seed isolation. Probe69's budgets run to 5,120,000
ticks, which is 64x anything this repository has developed on before, so the
stride that was comfortable for earlier probes is no longer obviously safe. The
guard has to reject a budget that outruns its own stride, and that is checked on
constructed bands where the answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from homesocial.organism import causal_self, self_belief
from homesocial.organism.belief_scaling import (
    SEED_STRIDE,
    SURVEY_AUDIT_BASE,
    SURVEY_DEV_BASE,
    TREATMENT_AUDIT_BASE,
    TREATMENT_DEV_BASE,
    Cell,
    _check_seed_isolation,
    summarize,
)

BANDS = {
    "survey_dev": SURVEY_DEV_BASE,
    "survey_audit": SURVEY_AUDIT_BASE,
    "treatment_dev": TREATMENT_DEV_BASE,
    "treatment_audit": TREATMENT_AUDIT_BASE,
}


# -- the two arms are scored by one instrument ---------------------------------


def test_both_audits_share_one_balanced_accuracy_function():
    """The matched-instrument claim, asserted on object identity.

    If `causal_self` ever grows its own copy, the structured and black-box
    numbers stop being comparable and every table in probe69 becomes a
    comparison between two different definitions of the same word.
    """

    assert causal_self._balanced_accuracy is self_belief._balanced_accuracy


def test_balanced_accuracy_is_balanced_not_raw():
    """A majority-class predictor must score chance, not the class prior.

    This is what makes the zero-belief lesion land at 0.3333 rather than at
    whatever the commonest lowest-need happens to be, and the lesion is the only
    control standing between probe69 and an audit that scores itself.
    """

    truth = np.array([0, 0, 0, 0, 0, 0, 1, 2])
    always_zero = np.zeros_like(truth)
    assert self_belief._balanced_accuracy(always_zero, truth) == pytest.approx(
        1.0 / 3.0
    )
    perfect = truth.copy()
    assert self_belief._balanced_accuracy(perfect, truth) == pytest.approx(1.0)


# -- seed isolation ------------------------------------------------------------


def test_configured_bands_are_isolated_at_five_seeds():
    """The bands this module actually ships must survive its own guard."""

    cells = [Cell("blackbox", 256, 5_120_000), Cell("structured", 0, 5_120_000)]
    _check_seed_isolation(cells, 5, BANDS)


def test_a_budget_larger_than_the_stride_is_refused():
    """The trap `CLAUDE.md` records: a development stream reaching the next band.

    A budget longer than the stride can walk its world seeds into the following
    seed's territory, so the model develops on lives it is later scored against
    and the leak reads as a replication.
    """

    cells = [Cell("blackbox", 256, SEED_STRIDE + 1)]
    with pytest.raises(ValueError, match="exceeds the seed stride"):
        _check_seed_isolation(cells, 1, BANDS)


def test_overlapping_bands_are_refused():
    """Enough seeds to span the gap between two bases must be rejected."""

    tight = {"dev": 1_000_000_000, "audit": 1_000_000_000 + SEED_STRIDE}
    cells = [Cell("blackbox", 256, 80_000)]
    with pytest.raises(ValueError, match="overlaps"):
        _check_seed_isolation(cells, 5, tight)


def test_isolation_guard_uses_the_longest_budget_not_the_first():
    """A safe first cell must not license an unsafe later one."""

    cells = [Cell("blackbox", 64, 80_000), Cell("blackbox", 256, SEED_STRIDE * 2)]
    with pytest.raises(ValueError, match="exceeds the seed stride"):
        _check_seed_isolation(cells, 1, BANDS)


# -- reporting -----------------------------------------------------------------


def test_structured_ridge_column_is_not_applicable_rather_than_zero():
    """The structured belief *is* its output, so it has no readout to separate.

    Reporting 0.0 there would put a false zero into the head-minus-ridge column
    that probe69 uses to date probe55's readout bottleneck, and would make the
    structured arm look as though its representation were empty.
    """

    rows = [
        {
            "family": "structured",
            "width": 0,
            "budget": 80_000,
            "head_balanced_accuracy": 0.9261,
            "ridge_balanced_accuracy": float("nan"),
        }
    ]
    text = summarize(rows)
    assert "0.9261" in text
    assert "nan" in text
    assert "0.0000" not in text


def test_summarize_averages_over_seeds_within_a_cell():
    rows = [
        {
            "family": "blackbox",
            "width": 256,
            "budget": 80_000,
            "head_balanced_accuracy": head,
            "ridge_balanced_accuracy": ridge,
        }
        for head, ridge in ((0.70, 0.80), (0.80, 0.90))
    ]
    text = summarize(rows)
    assert "0.7500" in text
    assert "0.8500" in text
    assert "    2" in text


def test_cell_names_separate_the_families_and_widths():
    """Width must not appear in a structured name; it has none."""

    assert Cell("blackbox", 64, 80_000).name == "blackbox_w64@80000"
    assert Cell("blackbox", 256, 80_000).name == "blackbox_w256@80000"
    assert Cell("structured", 0, 80_000).name == "structured@80000"
