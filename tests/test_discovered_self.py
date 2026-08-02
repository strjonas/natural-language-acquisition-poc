"""Guards for probe61: the discovered bodily self-model.

These test the properties the claim rests on -- that the frozen ecology is
untouched, that the ground truth actually moves, that the seed actually varies
the model, and that no need label reaches the learner -- not the numbers, which
live in `docs/decisions/`.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from homesocial.island.report import BODY_NEEDS, REPORT_NEEDS, ReportConfig
from homesocial.organism.discovered_self import (
    DIMENSION_SWEEP,
    DeployedSelf,
    DevelopmentStream,
    DiscoveredSelfConfig,
    DiscoveredSelfModel,
    _check_seed_isolation,
    development_seed_base,
    fit_listener_model,
    recovered_dimension,
)
from homesocial.organism.report_audit import make_report_world
from homesocial.organism.train import OrganismConfig


def _organism() -> OrganismConfig:
    return OrganismConfig()


def test_frozen_report_ecology_is_bit_identical_by_default():
    """A default `ReportConfig` must be the ecology probe57 was scored in."""

    config = ReportConfig()
    assert config.frozen_needs == ()
    assert config.live_needs() == BODY_NEEDS
    assert config.metabolism_of("food") == config.food_metabolism
    assert config.metabolism_of("safety") == config.safety_metabolism


def test_freezing_an_axis_leaves_the_other_axes_bit_identical():
    """Freezing consumes the same random draws, so nothing else may move."""

    organism = _organism()
    seed = 4_242_424
    plain = make_report_world(organism, seed=seed, listener_mode="grounded")
    frozen = make_report_world(
        organism, seed=seed, listener_mode="grounded", frozen_needs=("food",)
    )
    assert plain.grid.needs.water == frozen.grid.needs.water
    assert plain.grid.needs.energy == frozen.grid.needs.energy
    assert frozen.grid.needs.food == 1.0
    assert plain.grid.needs.food != 1.0


def test_a_frozen_axis_never_moves():
    organism = _organism()
    world = make_report_world(
        organism,
        seed=515_151,
        listener_mode="grounded",
        frozen_needs=("food", "safety"),
    )
    for _ in range(120):
        _, _, terminated, truncated, _ = world.step("wait")
        assert world.grid.needs.food == 1.0
        assert world.grid.needs.safety == 1.0
        if terminated or truncated:
            break


def test_the_ground_truth_dimension_actually_moves_across_the_sweep():
    """The sweep is only a test if the true number of variables changes."""

    truths = [
        len(replace(ReportConfig(), frozen_needs=frozen).live_needs())
        for frozen in DIMENSION_SWEEP.values()
    ]
    assert truths == [1, 2, 3, 4]


def test_the_unmodified_ecology_really_has_four_bodily_variables():
    """K4 is the frozen ecology, and `safety` really does deplete in it."""

    config = ReportConfig()
    assert config.safety_metabolism > 0.0
    assert "safety" in config.live_needs()
    assert "safety" not in REPORT_NEEDS


def test_the_seed_actually_varies_the_discovered_model():
    """The repository's oldest trap: a seed that retrains something identical."""

    first = DiscoveredSelfModel(latent_size=8, seed=1)
    second = DiscoveredSelfModel(latent_size=8, seed=2)
    again = DiscoveredSelfModel(latent_size=8, seed=1)
    assert not np.array_equal(
        np.asarray(first.drift_raw), np.asarray(second.drift_raw)
    )
    assert not np.array_equal(
        np.asarray(first.uptake_raw), np.asarray(second.uptake_raw)
    )
    assert np.array_equal(np.asarray(first.drift_raw), np.asarray(again.drift_raw))


def test_developmental_seed_bases_are_isolated_from_every_evaluation_band():
    for world in DIMENSION_SWEEP:
        for seed_index in range(5):
            development_seed_base(seed_index, world)
    with pytest.raises(ValueError):
        _check_seed_isolation(6_200_000)
    with pytest.raises(ValueError):
        _check_seed_isolation(320_100_000)


def test_developmental_seed_bases_never_collide_across_seeds_or_worlds():
    bases = [
        development_seed_base(seed_index, world)
        for world in DIMENSION_SWEEP
        for seed_index in range(5)
    ]
    assert len(set(bases)) == len(bases)
    ordered = sorted(bases)
    gaps = [second - first for first, second in zip(ordered, ordered[1:])]
    assert min(gaps) >= 900_000


def test_the_latent_must_be_overcomplete():
    with pytest.raises(ValueError):
        DiscoveredSelfConfig(latent_size=3)


def test_readouts_are_symmetric_under_permuting_the_latent():
    """Nothing in the sensations can hand over an axis identity or an order."""

    model = DiscoveredSelfModel(latent_size=8, seed=7)
    model.gate_raw = mx.full((8,), 4.0)
    states = mx.array(np.random.default_rng(0).uniform(0.1, 0.9, (3, 5, 8)).astype(np.float32))
    order = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    mean_a, min_a = model.feelings(states)
    permuted = mx.array(np.asarray(states)[:, :, order])
    model.gate_raw = mx.array(np.asarray(model.gate_raw)[order])
    mean_b, min_b = model.feelings(permuted)
    mx.eval(mean_a, min_a, mean_b, min_b)
    assert np.allclose(np.asarray(mean_a), np.asarray(mean_b), atol=1e-5)
    assert np.allclose(np.asarray(min_a), np.asarray(min_b), atol=1e-5)


def test_an_unparticipating_dimension_cannot_reach_either_sensation():
    model = DiscoveredSelfModel(latent_size=8, seed=3)
    model.gate_raw = mx.array(
        np.array([8.0, 8.0, 8.0, -12.0, -12.0, -12.0, -12.0, -12.0], dtype=np.float32)
    )
    base = np.random.default_rng(1).uniform(0.3, 0.9, (2, 4, 8)).astype(np.float32)
    other = base.copy()
    other[:, :, 3:] = 0.0
    mean_a, min_a = model.feelings(mx.array(base))
    mean_b, min_b = model.feelings(mx.array(other))
    mx.eval(mean_a, min_a, mean_b, min_b)
    assert np.allclose(np.asarray(mean_a), np.asarray(mean_b), atol=1e-3)
    assert np.allclose(np.asarray(min_a), np.asarray(min_b), atol=1e-3)


def test_the_planner_never_reads_a_need_label():
    """The map from a discovered dimension to a need is an audit's, not the organism's."""

    rng = np.random.default_rng(11)
    deployed = DeployedSelf(
        drift=-rng.uniform(0.005, 0.02, 8),
        move=-rng.uniform(0.0, 0.01, 8),
        uptake=rng.uniform(0.0, 0.4, (9, 8)),
        shock=-rng.uniform(0.0, 0.2, (9, 8)),
        gate=np.ones(8),
        birth=rng.uniform(0.4, 0.8, 8),
        listener=np.full((60, 10), 0.1),
    )
    token = deployed.token(deployed.birth, step_count=3, help_period=6)
    assert 0 <= token < 60
    order = rng.permutation(8)
    relabelled = DeployedSelf(
        drift=deployed.drift[order],
        move=deployed.move[order],
        uptake=deployed.uptake[:, order],
        shock=deployed.shock[:, order],
        gate=deployed.gate[order],
        birth=deployed.birth[order],
        listener=deployed.listener,
    )
    assert relabelled.token(relabelled.birth, step_count=3, help_period=6) == token


def test_listener_model_is_the_exact_maximum_likelihood_table():
    stream = DevelopmentStream(
        durations=np.zeros((1, 1), dtype=np.float32),
        moves=np.zeros((1, 1), dtype=np.float32),
        uptake=np.zeros((1, 1, 9), dtype=np.float32),
        shock=np.zeros((1, 1, 9), dtype=np.float32),
        mask=np.ones((1, 1), dtype=np.float32),
        mean_feel=np.zeros((1, 1), dtype=np.float32),
        min_feel=np.zeros((1, 1), dtype=np.float32),
        tokens=np.array([5, 5, 5, 5], dtype=np.int32),
        listener_targets=np.array([2, 2, 2, 9], dtype=np.int32),
        true_needs=np.zeros((1, 1, 4), dtype=np.float32),
        true_birth=np.zeros((1, 4), dtype=np.float32),
        ticks=1,
        world="K4",
    )
    logits = np.asarray(fit_listener_model(stream, smoothing=0.0))
    probabilities = np.exp(logits[5] - logits[5].max())
    probabilities /= probabilities.sum()
    assert probabilities[2] == pytest.approx(0.75, abs=1e-5)
    assert probabilities[9] == pytest.approx(0.25, abs=1e-5)


def test_recovered_dimension_counts_only_dimensions_that_carry_the_sensations():
    """A latent whose extra dimensions do nothing must not be credited with them."""

    rng = np.random.default_rng(5)
    lives, steps, latent = 6, 40, 8
    config = DiscoveredSelfConfig(
        latent_size=latent, birth_inference_steps=120, ablation_inference_steps=40
    )
    model = DiscoveredSelfModel(latent_size=latent, seed=0)
    drift = np.full(latent, -6.0, dtype=np.float32)
    drift[:2] = [np.log(np.expm1(0.01)), np.log(np.expm1(0.02))]
    # Two live dimensions; the other six are switched out of both sensations,
    # which is exactly what an unused dimension looks like after fitting.
    model.drift_raw = mx.array(drift)
    model.move_raw = mx.full((latent,), -40.0)
    model.uptake_raw = mx.full((9, latent), -40.0)
    model.shock_raw = mx.full((9, latent), -40.0)
    gate = np.full(latent, -12.0, dtype=np.float32)
    gate[:2] = 8.0
    model.gate_raw = mx.array(gate)
    model.mean_scale_raw = mx.array(np.float32(np.log(np.expm1(0.5))))
    model.mean_offset = mx.array(0.0)
    births = np.zeros((lives, latent), dtype=np.float32)
    births[:, :2] = rng.uniform(0.5, 0.9, (lives, 2))
    truth = np.zeros((lives, steps, 2))
    state = births[:, :2].copy()
    for step in range(steps):
        state = np.clip(state - np.array([0.01, 0.02]), 0.0, 1.0)
        truth[:, step] = state
    stream = DevelopmentStream(
        durations=np.ones((lives, steps), dtype=np.float32),
        moves=np.zeros((lives, steps), dtype=np.float32),
        uptake=np.zeros((lives, steps, 9), dtype=np.float32),
        shock=np.zeros((lives, steps, 9), dtype=np.float32),
        mask=np.ones((lives, steps), dtype=np.float32),
        mean_feel=(0.5 * truth.sum(axis=-1)).astype(np.float32),
        min_feel=truth.min(axis=-1).astype(np.float32),
        tokens=np.zeros((1,), dtype=np.int32),
        listener_targets=np.zeros((1,), dtype=np.int32),
        true_needs=np.zeros((lives, steps, 4), dtype=np.float32),
        true_birth=np.zeros((lives, 4), dtype=np.float32),
        ticks=lives * steps,
        world="K2",
    )
    found = recovered_dimension(model, stream, config)
    assert found["effective_dimension"] == 2.0
