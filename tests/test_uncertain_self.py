"""Guards for the silent-shock observability lever and the ceiling instruments.

The lever's whole validity rests on one claim: silencing a shock changes what
the organism can perceive and nothing about what happens to it. If that fails,
every number in the survey confounds observability with dynamics. It is tested
here over full lives, not over a handful of ticks.
"""

from __future__ import annotations

from dataclasses import replace
from random import Random

import numpy as np
import pytest

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS, ReportConfig
from homesocial.organism.report_audit import _ObservableBodyFilter, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import execute_agent_action, load_organism_checkpoint
from homesocial.organism.uncertain_self import (
    CEILING_SEED_BASE,
    NaiveTier,
    PosteriorTier,
    PosteriorVoteTier,
    ShockStatistics,
    choose_need,
    run_open_loop,
)

PARENT = "runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


@pytest.fixture(scope="module")
def organism_pair():
    try:
        return load_organism_checkpoint(PARENT)
    except (FileNotFoundError, OSError):  # pragma: no cover - artifact absent
        pytest.skip("The probe52 lexical parent checkpoint is not present.")


def _live(organism, model, seed, silence, forced=None):
    """One life. Returns the body trajectory, the actions, and shock counts."""

    world = make_report_world(organism, seed=seed, silent_shock_probability=silence)
    packet = world.reset(seed)
    hidden = None
    rng = Random(seed + 59_000_003)
    bodies: list[list[float]] = []
    actions: list[int] = []
    silent: list[tuple[int, str]] = []
    loud = 0
    index = 0
    while True:
        word = NEED_TO_REPORT_WORD[world.lowest_need()]
        world.hear((TOKEN_TO_ID[word], PAD_ID))
        if forced is None:
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
        else:
            if index >= len(forced):
                break
            action = forced[index]
        actions.append(action)
        index += 1
        tick = packet.step_count
        packet, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            action,
            consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        bodies.append([getattr(world.grid.needs, need) for need in REPORT_NEEDS])
        shock = info.get("shock_need")
        if shock is not None:
            if info.get("shock_silent"):
                silent.append((tick, str(shock)))
            else:
                loud += 1
        if terminated or truncated:
            break
    return np.asarray(bodies), actions, silent, loud


def test_silence_changes_perception_and_never_the_body(organism_pair):
    """The lever must move what can be known and nothing that happens."""

    model, organism = organism_pair
    checked = 0
    for seed in (CEILING_SEED_BASE + 5, CEILING_SEED_BASE + 11):
        loud_bodies, actions, silent_at_zero, loud_count = _live(
            organism, model, seed, 0.0
        )
        quiet_bodies, _, silent_at_one, loud_at_one = _live(
            organism, model, seed, 1.0, forced=actions
        )
        assert silent_at_zero == []
        assert loud_at_one == 0
        # Every shock that was audible is now silent, and the same ones.
        assert len(silent_at_one) == loud_count > 0
        span = min(len(loud_bodies), len(quiet_bodies))
        assert span > 100, "This guard is worthless on a life that ends early."
        assert np.array_equal(loud_bodies[:span], quiet_bodies[:span])
        checked += 1
    assert checked == 2


def test_silence_off_leaves_the_frozen_ecology_untouched(organism_pair):
    """The default ecology must be bit-identical to itself with the lever at 0."""

    model, organism = organism_pair
    assert ReportConfig().silent_shock_probability == 0.0
    seed = CEILING_SEED_BASE + 5
    default, actions, silent, _ = _live(organism, model, seed, 0.0)
    explicit_world = make_report_world(organism, seed=seed)
    assert explicit_world.report.silent_shock_probability == 0.0
    assert silent == []
    replayed, _, _, _ = _live(organism, model, seed, 0.0, forced=actions)
    assert np.array_equal(default, replayed)


def test_silent_sets_are_nested_across_the_sweep(organism_pair):
    """A shock silent at a low rate must still be silent at a higher one.

    The sweep is only a comparison of nested observability over one fixed shock
    history if the silence draws do not re-randomize per rate.
    """

    model, organism = organism_pair
    seed = CEILING_SEED_BASE + 11
    _, actions, _, _ = _live(organism, model, seed, 0.0)
    sets = []
    for silence in (0.25, 0.5, 1.0):
        _, _, silent, _ = _live(organism, model, seed, silence, forced=actions)
        sets.append({entry for entry in silent})
    assert sets[0] <= sets[1] <= sets[2]
    assert len(sets[2]) > len(sets[0]) > 0


def test_the_visible_history_filter_is_exact_when_nothing_is_silent(organism_pair):
    """Probe53's filter is exact only because every event is perceptible.

    This is the fact the whole survey rests on, and the reason the lever exists.
    """

    model, organism = organism_pair
    row = run_open_loop(
        model,
        organism,
        silence=0.0,
        lives=3,
        seed_base=CEILING_SEED_BASE,
        particles=16,
    )
    # Exact up to float accumulation over four hundred ticks, not approximately
    # right: the residual here is ~2e-10 against a null body error of 0.15.
    assert row["body_error"]["naive"] == pytest.approx(0.0, abs=1e-8)
    assert row["body_error"]["mean_corrected"] == pytest.approx(0.0, abs=1e-8)
    assert row["body_error"]["posterior"] == pytest.approx(0.0, abs=1e-8)
    # With nothing hidden the cloud must be a point, so every tier must name the
    # same need as the oracle does.
    for tier in ("naive", "mean_corrected", "posterior", "posterior_vote"):
        assert row["named_need_accuracy"][tier] == pytest.approx(
            row["named_need_accuracy"]["oracle"], abs=1e-12
        )


def test_the_audit_filter_carries_a_cloud_identically(organism_pair):
    """Generalizing the filter to a particle cloud must not change its dynamics.

    A one-particle cloud has to reproduce the plain vector filter exactly, or the
    posterior is measuring different dynamics from the point tiers.
    """

    model, organism = organism_pair
    seed = CEILING_SEED_BASE + 5
    report = replace(organism.report, silent_shock_probability=0.0)
    statistics = ShockStatistics(report.shock_probability, report.shock_size, 0.0)
    world = make_report_world(organism, seed=seed)
    packet = world.reset(seed)
    point = NaiveTier(organism, report)
    point.reset(packet, world)
    cloud = PosteriorTier(organism, report, statistics, particles=1, seed=seed)
    cloud.reset(packet, world)
    hidden = None
    rng = Random(seed + 59_000_003)
    steps = 0
    while steps < 120:
        # Speak, or no help arrives and the life ends before the guard is worth
        # anything.
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[world.lowest_need()]], PAD_ID))
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        before = packet
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            action,
            consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        if terminated or truncated:
            break
        point.update(before, action, packet, world)
        cloud.update(before, action, packet, world)
        assert np.allclose(point.point(), cloud.point())
        steps += 1
    assert steps > 50


def test_the_cloud_covers_the_truth_near_its_nominal_rate(organism_pair):
    """A posterior that does not contain the body is not a posterior."""

    model, organism = organism_pair
    row = run_open_loop(
        model,
        organism,
        silence=1.0,
        lives=4,
        seed_base=CEILING_SEED_BASE,
        particles=48,
    )
    # Nominal is 0.90 for a 5th-to-95th percentile band, measured per need.
    assert 0.80 <= row["credible_interval_coverage"] <= 0.99
    assert row["mean_spread"] > 0.02
    # And the spread has to track the error it is a claim about.
    assert row["posterior_spread_error_correlation"] > 0.15
    assert (
        row["error_in_least_certain_quartile"]
        > 1.5 * row["error_in_most_certain_quartile"]
    )


def test_expected_loss_matches_the_cloud_it_summarizes():
    """The corrected point estimate must be the posterior mean, not a guess."""

    statistics = ShockStatistics(probability=0.025, size=0.25, silent_probability=1.0)
    rng = np.random.default_rng(0)
    duration, particles = 7, 200_000
    interior = duration - 1
    events = rng.binomial(
        interior, statistics.interior_event_rate(), particles
    ) + rng.binomial(1, statistics.terminal_event_rate(marker_seen=False), particles)
    taken = np.zeros((particles, 3), dtype=np.int64)
    total = int(events.sum())
    owner = np.repeat(np.arange(particles), events)
    target = rng.integers(0, 3, total)
    np.add.at(taken, (owner, target), 1)
    sampled = (taken * statistics.size).mean()
    assert sampled == pytest.approx(
        statistics.expected_loss(duration=duration, marker_seen=False), rel=0.05
    )


def test_a_seen_marker_leaves_nothing_hidden_on_that_tick():
    statistics = ShockStatistics(probability=0.025, size=0.25, silent_probability=1.0)
    assert statistics.terminal_event_rate(marker_seen=True) == 0.0
    assert statistics.expected_loss(duration=1, marker_seen=True) == 0.0
    # With no silence a tick with no marker is fully explained by no shock.
    quiet = ShockStatistics(probability=0.025, size=0.25, silent_probability=0.0)
    assert quiet.terminal_event_rate(marker_seen=False) == 0.0


def test_the_choice_ignores_the_order_of_the_particles(organism_pair):
    """Nothing in the decision may depend on how the cloud happens to be
    arranged; a permutation carries no information about the body."""

    model, organism = organism_pair
    report = replace(organism.report, silent_shock_probability=1.0)
    statistics = ShockStatistics(report.shock_probability, report.shock_size, 1.0)
    drift = np.array(
        [report.food_metabolism, report.water_metabolism, report.energy_metabolism]
    )
    tier = PosteriorVoteTier(organism, report, statistics, particles=32, seed=3)
    rng = np.random.default_rng(1)
    cloud = rng.uniform(0.05, 0.95, size=(32, 3))
    tier._filter._belief = cloud.copy()
    first = choose_need(tier, step_count=3, report=report, drift=drift)
    tier._filter._belief = cloud[rng.permutation(32)].copy()
    second = choose_need(tier, step_count=3, report=report, drift=drift)
    assert first == second
