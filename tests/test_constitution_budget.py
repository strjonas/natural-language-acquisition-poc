"""Guards for the preregistered constitution-budget ceiling instrument."""

from __future__ import annotations

from dataclasses import replace

import pytest

from homesocial.creole.vocab import TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, SIZE_TO_WORD, ReportConfig, ReportWorld
from homesocial.island.world import IslandConfig
from homesocial.organism.constitution_budget import (
    ConstitutionBudgetWorld,
    LifetimeShareAllocator,
    policy_burdens,
    predicted_burdens,
)


def _trace(world: ReportWorld, *, ticks: int = 48) -> list[tuple[object, ...]]:
    tokens = (
        TOKEN_TO_ID[NEED_TO_REPORT_WORD["food"]],
        TOKEN_TO_ID[SIZE_TO_WORD["large"]],
    )
    rows = []
    for _ in range(ticks):
        world.hear(tokens)
        _, _, terminated, truncated, info = world.step("wait")
        rows.append(
            (
                info["granted_need"],
                info["granted_large"],
                info["store_remaining"],
                info["viability"],
            )
        )
        if terminated or truncated:
            break
    return rows


def _world(world_type=ReportWorld, *, store: float, policy: str = "first_come"):
    report = replace(
        ReportConfig(life_steps=80),
        portion_requests=True,
        caregiver_store=store,
        shock_probability=0.0,
    )
    kwargs = {}
    if world_type is ConstitutionBudgetWorld:
        kwargs["allocation_policy"] = policy
    world = world_type(IslandConfig(), report=report, seed=17, **kwargs)
    world.reset(17)
    return world


def test_first_come_hook_is_tick_identical_to_the_existing_caregiver():
    existing = _trace(_world(store=3.0))
    instrument = _trace(
        _world(ConstitutionBudgetWorld, store=3.0, policy="first_come")
    )
    assert instrument == existing


@pytest.mark.parametrize(
    "policy", ("oracle_rates", "species_rates", "permuted_rates")
)
def test_every_allocator_is_inert_when_the_store_is_unlimited(policy):
    first = _trace(
        _world(ConstitutionBudgetWorld, store=0.0, policy="first_come")
    )
    allocated = _trace(
        _world(ConstitutionBudgetWorld, store=0.0, policy=policy)
    )
    assert allocated == first


def test_predicted_burden_moves_with_the_individual_rate_on_the_named_axis():
    report = ReportConfig()
    typical = predicted_burdens(
        report,
        metabolic_scale={need: 1.0 for need in ("food", "water", "energy")},
        uptake_scale={need: 1.0 for need in ("food", "water", "energy")},
    )
    water_fast = predicted_burdens(
        report,
        metabolic_scale={"food": 1.0, "water": 1.6, "energy": 1.0},
        uptake_scale={need: 1.0 for need in ("food", "water", "energy")},
    )
    assert water_fast["water"] > typical["water"]
    assert water_fast["food"] == typical["food"]
    assert water_fast["energy"] == typical["energy"]


def test_permuted_control_preserves_burdens_and_assigns_them_to_wrong_axes():
    report = ReportConfig()
    scale = {"food": 0.5, "water": 1.0, "energy": 1.5}
    uptake = {need: 1.0 for need in scale}
    oracle = policy_burdens(
        "oracle_rates", report, metabolic_scale=scale, uptake_scale=uptake
    )
    permuted = policy_burdens(
        "permuted_rates", report, metabolic_scale=scale, uptake_scale=uptake
    )
    assert sorted(permuted.values()) == sorted(oracle.values())
    assert sum(permuted.values()) == pytest.approx(sum(oracle.values()))
    assert permuted != oracle


def test_fractional_quota_allows_the_crossing_grant_and_refuses_the_next_one():
    allocator = LifetimeShareAllocator("oracle_rates")
    report = replace(ReportConfig(), caregiver_store=1.0, shock_probability=0.0)
    allocator.reset(
        report,
        metabolic_scale={"food": 1.0, "water": 1.0, "energy": 1.0},
        uptake_scale={"food": 1.0, "water": 1.0, "energy": 1.0},
    )
    assert allocator.quotas["food"] < report.portion_large
    assert allocator.permit("food", cost=report.portion_large, capacity=0.4)
    assert not allocator.permit("food", cost=report.portion_small, capacity=0.2)
    assert allocator.spent["food"] == pytest.approx(report.portion_large)
    assert allocator.refused["food"] == 1
    assert allocator.refused_capacity == pytest.approx(0.2)


def test_budget_world_refuses_after_one_need_spends_its_lifetime_share():
    world = _world(
        ConstitutionBudgetWorld, store=1.0, policy="species_rates"
    )
    _trace(world, ticks=60)
    assert world.grant_counts["food"] == 1
    assert world.grant_counts["refused"] > 0
    assert world.allocation.spent["food"] == pytest.approx(
        world.report.portion_large
    )
