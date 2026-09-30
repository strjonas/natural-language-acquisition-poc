"""Construction guards for Probe71's default-off five-axis body."""

from __future__ import annotations

import pytest

from homesocial.island.expanded_body import (
    EXPANDED_BODY_SCHEMA,
    LEGACY_BODY_SCHEMA,
    BodySchema,
    BodyState,
    ExpandedBody,
    ExpandedBodyConfig,
    AxisSpec,
    portion_for_burden,
    projected_state,
    run_oracle_life,
)
from homesocial.island.report import HELP_SURFACES, NEED_TO_REPORT_WORD, ReportConfig


def test_the_legacy_schema_is_exactly_the_existing_report_body():
    report = ReportConfig()
    assert LEGACY_BODY_SCHEMA.names == ("food", "water", "energy")
    assert LEGACY_BODY_SCHEMA.report_words == tuple(
        NEED_TO_REPORT_WORD[need] for need in LEGACY_BODY_SCHEMA.names
    )
    for axis in LEGACY_BODY_SCHEMA.axes:
        assert axis.resting_rate == getattr(report, f"{axis.name}_metabolism")
        if axis.name == "energy":
            assert axis.moving_rate == report.move_energy_metabolism
        else:
            assert axis.moving_rate == axis.resting_rate
        assert axis.small_surface == HELP_SURFACES[(axis.name, False)]
        assert axis.large_surface == HELP_SURFACES[(axis.name, True)]


def test_the_expanded_schema_has_five_distinguishable_consequence_axes():
    assert EXPANDED_BODY_SCHEMA.names == (
        "food",
        "water",
        "energy",
        "safety",
        "health",
    )
    assert len(EXPANDED_BODY_SCHEMA.report_words) == 5
    assert len(set(EXPANDED_BODY_SCHEMA.report_words)) == 5
    assert len(EXPANDED_BODY_SCHEMA.surfaces) == 15
    assert len(set(EXPANDED_BODY_SCHEMA.surfaces)) == 15


def test_duplicate_words_or_surfaces_cannot_fake_a_larger_message_space():
    first = AxisSpec("a", "one", 0.01, 0.01, "a1", "a2", "a3")
    duplicate_word = AxisSpec("b", "one", 0.01, 0.01, "b1", "b2", "b3")
    with pytest.raises(ValueError, match="Report words"):
        BodySchema((first, duplicate_word))
    duplicate_surface = AxisSpec("b", "two", 0.01, 0.01, "a1", "b2", "b3")
    with pytest.raises(ValueError, match="unique surface"):
        BodySchema((first, duplicate_surface))


def test_adding_axes_does_not_perturb_the_first_three_birth_or_scale_draws():
    legacy = ExpandedBody(LEGACY_BODY_SCHEMA, seed=77)
    expanded = ExpandedBody(EXPANDED_BODY_SCHEMA, seed=77)
    legacy_state = legacy.reset()
    expanded_state = expanded.reset()
    assert expanded_state.values[:3] == legacy_state.values
    assert {
        need: expanded.metabolic_scale[need] for need in LEGACY_BODY_SCHEMA.names
    } == legacy.metabolic_scale


def test_health_is_a_real_viability_axis_and_help_can_restore_it():
    body = ExpandedBody(
        EXPANDED_BODY_SCHEMA,
        config=ExpandedBodyConfig(
            birth_levels=(0.5,), metabolic_spread=0.0, shock_probability=0.0
        ),
        seed=5,
    )
    body.reset()
    body.state = body.state.replace({"health": 0.005})
    transition = body.step()
    assert transition.terminated
    assert body.state.lowest_need() == "health"
    body.grant("health", "large")
    assert body.state.value("health") == pytest.approx(0.6)
    assert body.state.viability() > 0.0


def test_axis_gap_is_the_gap_between_the_two_lowest_values():
    state = BodyState(EXPANDED_BODY_SCHEMA, (0.7, 0.2, 0.9, 0.5, 0.35))
    assert state.lowest_need() == "water"
    assert state.axis_gap() == pytest.approx(0.15)


def test_projection_is_axis_generic_and_clipped():
    state = BodyState(EXPANDED_BODY_SCHEMA, (0.7, 0.2, 0.9, 0.5, 0.35))
    rates = {need: 0.01 for need in EXPANDED_BODY_SCHEMA.names}
    projected = projected_state(state, rates, ticks=30)
    assert projected.value("food") == pytest.approx(0.4)
    assert projected.value("water") == 0.0
    assert projected.value("health") == pytest.approx(0.05)


def test_the_general_portion_rule_moves_with_rate_and_respects_headroom():
    config = ExpandedBodyConfig()
    state = BodyState(EXPANDED_BODY_SCHEMA, (0.5, 0.5, 0.5, 0.5, 0.5))
    slow = portion_for_burden(
        state, "health", rate=0.004, interval=18, config=config
    )
    fast = portion_for_burden(
        state, "health", rate=0.020, interval=18, config=config
    )
    assert slow == "small"
    assert fast == "large"
    nearly_full = state.replace({"health": 0.99})
    assert (
        portion_for_burden(
            nearly_full, "health", rate=0.020, interval=18, config=config
        )
        == "small"
    )


def test_oracle_life_exposes_natural_combinations_and_the_locked_fallback():
    life = run_oracle_life(
        EXPANDED_BODY_SCHEMA,
        seed=91,
        config=ExpandedBodyConfig(
            birth_levels=(0.55,),
            metabolic_spread=0.0,
            shock_probability=0.0,
            life_steps=60,
        ),
    )
    assert life.events
    assert all(event.tick % 6 == 0 for event in life.events)
    assert all(event.need in EXPANDED_BODY_SCHEMA.names for event in life.events)
    assert all(event.size in ("small", "large") for event in life.events)
    assert all(
        event.useful_word_difference == 0.0
        for event in life.events
        if event.size == "large"
    )
    assert sum(life.grant_counts.values()) == len(life.events)
