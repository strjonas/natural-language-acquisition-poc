"""Guards for probe63's two ecology levers.

Both levers make a claim the whole survey rests on, and each is tested over
full lives rather than a handful of ticks.

``metabolic_spread`` claims to change *what this body is* -- and only that. At
zero it must leave the frozen ecology bit-identical; above zero it must actually
vary the body across lives, by exactly the drawn factor, or the survey would be
measuring a difference that is not there. That second half is this repository's
"seed that does nothing" trap: a lever whose knob turns and whose world does not
move produces a beautifully replicated null.

``interoception_probability`` claims the opposite -- to change *what can be
known* and nothing that happens. It is the probe62 silence guard run the other
way round.
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import (
    HELP_SURFACES,
    BODY_NEEDS,
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportConfig,
)
from homesocial.organism.report_audit import make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import execute_agent_action, load_organism_checkpoint

PARENT = "runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
SEED_BASE = 990_000_000


@pytest.fixture(scope="module")
def organism_pair():
    try:
        return load_organism_checkpoint(PARENT)
    except (FileNotFoundError, OSError):  # pragma: no cover - artifact absent
        pytest.skip("The probe52 lexical parent checkpoint is not present.")


def _live(organism, model, seed, *, forced=None, **overrides):
    """One life. Returns bodies, actions, and every interoceptive reading."""

    world = make_report_world(organism, seed=seed, **overrides)
    packet = world.reset(seed)
    hidden = None
    rng = Random(seed + 59_000_003)
    bodies: list[list[float]] = []
    actions: list[int] = []
    readings: list[tuple[int, tuple[float, ...]]] = []
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
        packet, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            action,
            consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        bodies.append([getattr(world.grid.needs, need) for need in REPORT_NEEDS])
        reading = info.get("interoception")
        if reading is not None:
            readings.append((packet.step_count, tuple(reading)))
        if terminated or truncated:
            break
    return np.asarray(bodies), actions, readings, world


# -- metabolic_spread: this life's own body -----------------------------------


def test_default_ecology_is_the_species_body():
    """Both levers are default-inert, and the default body is the species one."""

    report = ReportConfig()
    assert report.metabolic_spread == 0.0
    assert report.interoception_probability == 0.0


def test_zero_spread_leaves_the_frozen_ecology_bit_identical(organism_pair):
    model, organism = organism_pair
    for seed in (SEED_BASE + 3, SEED_BASE + 17):
        base_bodies, actions, base_readings, base_world = _live(organism, model, seed)
        held_bodies, _, _, held_world = _live(
            organism, model, seed, forced=actions, metabolic_spread=0.0
        )
        assert base_readings == []
        assert base_world.metabolic_scale == {need: 1.0 for need in BODY_NEEDS}
        assert np.array_equal(base_bodies, held_bodies)
        assert len(base_bodies) > 100


def test_spread_actually_varies_the_body_across_lives(organism_pair):
    """The "seed that does nothing" trap, applied to the body itself."""

    model, organism = organism_pair
    scales = []
    for life in range(12):
        _, _, _, world = _live(
            organism, model, SEED_BASE + 100 + life, metabolic_spread=0.5
        )
        scales.append(world.metabolic_scale)

    for need in BODY_NEEDS:
        drawn = np.array([scale[need] for scale in scales])
        assert drawn.min() < 0.85, f"{need} never drew a slow body."
        assert drawn.max() > 1.15, f"{need} never drew a fast body."
        assert np.ptp(drawn) > 0.4, f"{need} barely varies across lives."


def test_spread_scales_depletion_by_exactly_the_drawn_factor(organism_pair):
    """The body must burn fuel at this individual's rate, not the species rate.

    Measured on WAIT ticks with no help and no shock, where the only thing that
    moved the body is metabolism, so the comparison is exact rather than
    statistical.
    """

    _, organism = organism_pair
    seed = SEED_BASE + 41
    world = make_report_world(
        organism,
        seed=seed,
        metabolic_spread=0.6,
        shock_probability=0.0,
        help_period=10_000,
    )
    world.reset(seed)
    scale = world.metabolic_scale
    report = world.report

    before = world.grid.needs
    for _ in range(20):
        world.step("wait")
    after = world.grid.needs

    for need in ("food", "water", "energy"):
        observed = (getattr(before, need) - getattr(after, need)) / 20.0
        expected = getattr(report, f"{need}_metabolism") * scale[need]
        assert observed == pytest.approx(expected, abs=1e-12), need
        # And it is genuinely not the species rate this life is being scored on.
        assert abs(observed - getattr(report, f"{need}_metabolism")) > 1e-6


def test_individual_report_carries_this_lifes_rates(organism_pair):
    """The ceiling tier's constants must be this body's, energy costs included."""

    _, organism = organism_pair
    seed = SEED_BASE + 77
    world = make_report_world(organism, seed=seed, metabolic_spread=0.5)
    world.reset(seed)
    scale = world.metabolic_scale
    mine = world.individual_report()
    species = world.report

    assert mine.food_metabolism == pytest.approx(
        species.food_metabolism * scale["food"]
    )
    assert mine.water_metabolism == pytest.approx(
        species.water_metabolism * scale["water"]
    )
    # Resting and moving energy scale together: one fact about the self, not two.
    assert mine.energy_metabolism == pytest.approx(
        species.energy_metabolism * scale["energy"]
    )
    assert mine.move_energy_metabolism == pytest.approx(
        species.move_energy_metabolism * scale["energy"]
    )
    assert mine.metabolic_spread == species.metabolic_spread


# -- uptake_spread: the other way a body can be individual --------------------


def test_uptake_spread_is_default_inert_and_independent_of_metabolism(organism_pair):
    """Varying absorption must leave the burn rates of the same seed untouched.

    The two levers have to be separable or the localization test is meaningless:
    a world where only absorption is individual must have exactly the species
    metabolism, seed for seed.
    """

    _, organism = organism_pair
    assert ReportConfig().uptake_spread == 0.0
    for seed in (SEED_BASE + 31, SEED_BASE + 53):
        metabolic_only = make_report_world(
            organism, seed=seed, metabolic_spread=0.5, uptake_spread=0.0
        )
        metabolic_only.reset(seed)
        both = make_report_world(
            organism, seed=seed, metabolic_spread=0.5, uptake_spread=0.5
        )
        both.reset(seed)
        assert metabolic_only.metabolic_scale == both.metabolic_scale
        assert metabolic_only.uptake_scale == {need: 1.0 for need in REPORT_NEEDS}
        assert both.uptake_scale != metabolic_only.uptake_scale


def test_uptake_spread_scales_the_granted_portion_and_not_its_surface(organism_pair):
    """Absorption is private; the surface the caregiver produced is not.

    The organism must still see which portion class it was granted, because that
    is a public fact about the caregiver's act, while how much good it does is a
    fact about this body that no surface reveals.
    """

    _, organism = organism_pair
    seed = SEED_BASE + 67
    world = make_report_world(
        organism,
        seed=seed,
        uptake_spread=0.6,
        metabolic_spread=0.0,
        shock_probability=0.0,
        help_period=2,
    )
    world.reset(seed)
    scale = world.uptake_scale
    report = world.report

    amounts: set[float] = set()
    surfaces: set[str] = set()
    for _ in range(60):
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD["food"]], PAD_ID))
        world.step("wait")
        for obj in world.grid.objects:
            if obj.food_delta > 0.0:
                surfaces.add(obj.name)
                amounts.add(round(obj.food_delta, 12))

    assert amounts, "No food help was ever granted."
    assert scale["food"] != 1.0
    # Every delivered amount is a species portion scaled by this body's uptake.
    expected = {
        round(report.portion_small * scale["food"], 12),
        round(report.portion_large * scale["food"], 12),
    }
    assert amounts <= expected
    # None of them is the species amount, so absorption really is individual.
    species = {report.portion_small, report.portion_large}
    assert all(
        min(abs(amount - value) for value in species) > 1e-9 for amount in amounts
    )
    # And the surface the caregiver produced is the ordinary public one, which
    # is what keeps the portion *class* perceptible while the uptake is not.
    assert surfaces <= {HELP_SURFACES[("food", False)], HELP_SURFACES[("food", True)]}


# -- interoception: what can be known -----------------------------------------


def test_interoception_never_changes_the_body(organism_pair):
    """The probe62 silence guard, run the other way round.

    Silence withheld a reading the organism used to get. This grants one it
    never had. Both must leave the body bit-identical or every tier comparison
    confounds knowledge with dynamics.
    """

    model, organism = organism_pair
    checked = 0
    for seed in (SEED_BASE + 5, SEED_BASE + 23):
        blind_bodies, actions, blind_readings, _ = _live(
            organism, model, seed, metabolic_spread=0.5
        )
        seeing_bodies, _, seeing_readings, _ = _live(
            organism,
            model,
            seed,
            forced=actions,
            metabolic_spread=0.5,
            interoception_probability=1.0,
        )
        assert blind_readings == []
        span = min(len(blind_bodies), len(seeing_bodies))
        assert span > 100, "This guard is worthless on a life that ends early."
        assert np.array_equal(blind_bodies[:span], seeing_bodies[:span])
        # And the channel really did open.
        assert len(seeing_readings) >= span
        checked += 1
    assert checked == 2


def test_a_reading_is_the_true_body_at_that_tick(organism_pair):
    """A reading must be truth, not a noisy or stale echo of it."""

    model, organism = organism_pair
    seed = SEED_BASE + 9
    bodies, _, readings, _ = _live(
        organism,
        model,
        seed,
        metabolic_spread=0.5,
        interoception_probability=1.0,
    )
    assert len(readings) > 100
    for index, (_, reading) in enumerate(readings):
        assert np.array_equal(np.asarray(reading), bodies[index])


def test_reading_rate_tracks_the_lever(organism_pair):
    """Realized rate is below nominal, and the survey must report it, not assume it.

    Only the terminal tick of a lived action carries its reading forward, so an
    option that spans several primitive ticks discards the interior ones. That
    is the correct semantics -- a filter's belief after an update refers to the
    terminal tick -- but it means the rate a self-model actually experiences is
    lower than the knob, and by an amount that depends on the policy.
    """

    model, organism = organism_pair
    seed = SEED_BASE + 61
    rates = {}
    for nominal in (0.0, 0.05, 0.25):
        bodies, _, readings, _ = _live(
            organism,
            model,
            seed,
            metabolic_spread=0.5,
            interoception_probability=nominal,
        )
        rates[nominal] = len(readings) / len(bodies)

    assert rates[0.0] == 0.0
    assert 0.0 < rates[0.05] <= 0.05
    assert rates[0.05] < rates[0.25] <= 0.25
