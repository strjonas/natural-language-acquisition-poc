"""Guards for probe66's learner.

The claim probe66 makes is that the organism *finds out* whether a size word
means the same thing whatever it is asking for, rather than being handed the
answer. These tests hold the mechanism to that: the posterior has to follow the
evidence in both directions, it has to sit exactly on the prior when there is
nothing to compare, and the cell being asked about must not be allowed to
influence the judgement about whether meaning transfers to it.
"""

from __future__ import annotations

import pytest

from homesocial.island.report import REPORT_NEEDS, SIZE_WORDS
from homesocial.organism.discovered_convention import DiscoveredConvention


def _teach(model: DiscoveredConvention, need: str, word: str, large: bool, times: int):
    for _ in range(times):
        model.observe(need, word, large)


def test_needs_that_agree_are_evidence_that_the_word_transfers():
    """Food and water both read "more" as large, so energy probably does too."""

    model = DiscoveredConvention()
    for need in ("food", "water"):
        _teach(model, need, "more", True, 20)
        _teach(model, need, "not", False, 20)
    assert model.probability_factors("more") > 0.5
    assert model.transfers("more")
    assert model.word_for("energy", "large") == "more"
    assert model.word_for("energy", "small") == "not"


def test_needs_that_disagree_are_evidence_that_it_does_not():
    """Food reads "more" as large and water as small, so energy is anyone's guess.

    The right answer here is not a better guess. It is to stop guessing, which is
    what ``word_for`` returning ``None`` means: say the need word and let the
    caregiver draw, exactly as every organism before probe64 did.
    """

    model = DiscoveredConvention()
    _teach(model, "food", "more", True, 20)
    _teach(model, "food", "not", False, 20)
    _teach(model, "water", "more", False, 20)
    _teach(model, "water", "not", True, 20)
    assert model.probability_factors("more") < 0.5
    assert not model.transfers("more")
    assert model.word_for("energy", "large") is None
    assert model.word_for("energy", "small") is None


def test_one_need_alone_can_neither_agree_nor_disagree():
    """With nothing to compare against, the prior stands and nothing is discovered."""

    model = DiscoveredConvention()
    _teach(model, "food", "more", True, 50)
    assert model.probability_factors("more") == 0.5


def test_no_evidence_at_all_leaves_the_prior_untouched():
    model = DiscoveredConvention()
    for word in SIZE_WORDS:
        assert model.probability_factors(word) == 0.5


def test_the_cell_being_asked_about_cannot_influence_the_judgement():
    """A need with no evidence contributes the same factor to both hypotheses.

    This is what makes the posterior a judgement about whether meaning
    *transfers*, rather than a judgement about the cell it is being transferred
    to -- which the organism knows nothing about by construction.
    """

    model = DiscoveredConvention()
    for need in ("food", "water"):
        _teach(model, need, "more", True, 15)
    before = model.probability_factors("more")
    # ``energy`` still has no evidence; asking about it must change nothing.
    model.word_for("energy", "large")
    assert model.probability_factors("more") == before


def test_a_cell_with_its_own_evidence_needs_no_transfer():
    """Where the organism has looked, what it saw settles it.

    Even in a world whose convention does not factor -- food and water disagree,
    so nothing transfers -- a need the organism has actually used the word for is
    answered from that need's own record.
    """

    model = DiscoveredConvention()
    _teach(model, "food", "more", True, 20)
    _teach(model, "water", "more", False, 20)
    assert not model.transfers("more")
    _teach(model, "energy", "more", False, 4)
    _teach(model, "energy", "not", True, 4)
    assert model.word_for("energy", "large") == "not"
    assert model.word_for("energy", "small") == "more"


def test_the_posterior_is_a_probability():
    model = DiscoveredConvention()
    for need in REPORT_NEEDS:
        _teach(model, need, "more", True, 200)
    value = model.probability_factors("more")
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(value)  # finite, not NaN


def test_strong_disagreement_does_not_overflow():
    """Hundreds of grants a life put the log Bayes factor past what exp carries."""

    model = DiscoveredConvention()
    _teach(model, "food", "more", True, 5000)
    _teach(model, "water", "more", False, 5000)
    value = model.probability_factors("more")
    assert 0.0 <= value < 0.5
