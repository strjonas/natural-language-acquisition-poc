"""Guards for Probe71's closed-form first-use composition law."""

from __future__ import annotations

import numpy as np
import pytest

from homesocial.organism.composition_law import (
    SEED_STRIDE,
    TREATMENT_SEED_BASE,
    _check_seed_isolation,
    empirical_first_use_rent,
    estimate_cell_probabilities,
    expected_first_use_rent,
    permute_requests,
    pool_life_summaries,
    resample_requests,
    run_construction_survey,
    score_continuation,
    summarize_composition_rent,
)


def test_one_certain_cell_can_charge_rent_only_once():
    predicted = expected_first_use_rent(
        {("food", "small"): 1.0},
        {("food", "small"): 1.0},
        opportunities=20,
    )
    assert predicted == pytest.approx(1.0 / 20.0)


def test_a_cell_with_no_consequential_difference_pays_no_rent():
    predicted = expected_first_use_rent(
        {"a": 0.5, "b": 0.5},
        {"a": 1.0, "b": 0.0},
        opportunities=10,
    )
    assert predicted == pytest.approx((1.0 - 0.5**10) / 10.0)


def test_empirical_rent_reads_only_the_first_natural_use_of_each_cell():
    combinations = ("a", "a", "b", "b", "c")
    consequential = (False, True, True, False, True)
    # a's later consequential use cannot repair its non-consequential first use.
    assert empirical_first_use_rent(combinations, consequential) == 2.0 / 5.0


def test_cell_probability_estimates_have_no_smoothing_or_fitted_coefficient():
    probabilities, consequential = estimate_cell_probabilities(
        ("a", "a", "a", "b"), (True, False, True, False)
    )
    assert probabilities == {"a": 0.75, "b": 0.25}
    assert consequential == {"a": 2.0 / 3.0, "b": 0.0}


def test_permutation_preserves_every_cell_consequence_pair():
    combinations = ("a", "a", "b", "c", "c")
    consequential = (True, False, True, False, True)
    shuffled_cells, shuffled_consequence = permute_requests(
        combinations, consequential, seed=19
    )
    assert sorted(zip(shuffled_cells, shuffled_consequence)) == sorted(
        zip(combinations, consequential)
    )
    assert (shuffled_cells, shuffled_consequence) != (
        combinations,
        consequential,
    )


def test_iid_resampling_can_omit_a_rare_cell_and_repeat_another():
    combinations = ("common", "common", "common", "rare")
    consequential = (True, True, False, True)
    cells, flags = resample_requests(combinations, consequential, seed=2)
    assert len(cells) == len(combinations)
    assert len(flags) == len(consequential)
    assert list(cells).count("rare") != 1


def test_the_formula_matches_the_mean_over_exchangeable_request_orders():
    combinations = tuple("a" * 6 + "b" * 3 + "c")
    consequential = tuple(cell != "b" for cell in combinations)
    probabilities, q = estimate_cell_probabilities(combinations, consequential)
    predicted = expected_first_use_rent(
        probabilities, q, opportunities=len(combinations)
    )
    observed = []
    for seed in range(4000):
        cells, flags = resample_requests(combinations, consequential, seed=seed)
        observed.append(empirical_first_use_rent(cells, flags))
    assert float(np.mean(observed)) == pytest.approx(predicted, abs=0.002)


def test_summary_keeps_live_order_separate_from_the_permuted_control():
    row = summarize_composition_rent(
        ("a", "a", "b", "b"),
        (False, True, True, False),
        possible_combinations=4,
        permutation_seed=3,
    )
    assert row.opportunities == 4
    assert row.occupied_combinations == 2
    assert row.combinations == 4
    assert row.serial_residual == pytest.approx(
        row.empirical_rate - row.permuted_rate
    )


def test_life_pooling_is_weighted_by_request_opportunities():
    short = summarize_composition_rent(
        ("a",), (True,), possible_combinations=2, permutation_seed=1
    )
    long = summarize_composition_rent(
        ("a",) * 9, (False,) * 9, possible_combinations=2, permutation_seed=2
    )
    pooled = pool_life_summaries((short, long))
    assert pooled["opportunities"] == 10.0
    assert pooled["empirical_rate"] == pytest.approx(0.1)


def test_construction_seed_bands_cannot_overlap_or_exceed_their_stride():
    _check_seed_isolation(lives=100, seeds=5, seed_base=2_500_000_000)
    with pytest.raises(ValueError):
        _check_seed_isolation(
            lives=SEED_STRIDE + 1, seeds=5, seed_base=2_500_000_000
        )
    with pytest.raises(ValueError):
        _check_seed_isolation(
            lives=100, seeds=5, seed_base=TREATMENT_SEED_BASE - SEED_STRIDE
        )


def test_a_smoke_construction_survey_contains_both_bodies_and_scores_clauses():
    record = run_construction_survey(lives=3, seeds=1, seed_base=2_500_000_000)
    assert set(record["conditions"]) == {"K3", "K5"}
    assert record["conditions"]["K3"]["per_seed"][0]["lives"] == 3
    assert not any(
        key.startswith("_")
        for key in record["conditions"]["K3"]["per_seed"][0]
    )
    assert "health" in record["conditions"]["K5"]["request_share"]
    continuation = score_continuation(record)
    assert set(continuation["clauses"]) == {
        "C1_legacy_schema",
        "C2_five_axis_viability",
        "C3_health_is_real",
        "C4_instrument_validity",
        "C5_natural_range",
        "C6_axis_gap_moved",
    }


@pytest.mark.parametrize(
    "probabilities,q",
    [
        ({"a": 0.8}, {"a": 1.0}),
        ({"a": -0.1, "b": 1.1}, {"a": 1.0}),
        ({"a": 1.0}, {"a": 1.1}),
    ],
)
def test_invalid_probability_models_fail_loudly(probabilities, q):
    with pytest.raises(ValueError):
        expected_first_use_rent(probabilities, q, opportunities=5)
