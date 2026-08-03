"""Guards for probe63's self-calibration mechanism.

The mechanism rests on one claim that is easy to state and easy to get wrong:
that ``base + (exp(m) - 1) * J`` is what this organism's body filter would have
predicted if its own constants really were scaled by ``exp(m)``. If that is only
approximately true, every attribution built on it is measuring the approximation
error. It is checked here against a filter actually rebuilt with the scaled
constants, over full lives, to twelve decimal places.

The rest guard the things a self-model must not do: learn without evidence,
learn from another organism's evidence, or discover something about itself in a
world where there is nothing to discover.
"""

from __future__ import annotations

from dataclasses import replace
from random import Random

import numpy as np
import pytest

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.individual_self import (
    CALIBRATION_PARAMETERS,
    SelfJacobian,
)
from homesocial.organism.report_audit import _ObservableBodyFilter, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.self_calibration import (
    MAX_LOG_CORRECTION,
    SEED_BASE,
    OracleCalibration,
    RecursiveSelfCalibration,
    SelfCalibration,
    run_world,
)
from homesocial.organism.train import execute_agent_action, load_organism_checkpoint

PARENT = "runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


@pytest.fixture(scope="module")
def organism_pair():
    try:
        return load_organism_checkpoint(PARENT)
    except (FileNotFoundError, OSError):  # pragma: no cover - artifact absent
        pytest.skip("The probe52 lexical parent checkpoint is not present.")


def _drive(organism, model, seed, callback, *, ticks=120, **overrides):
    """Run one life, handing each transition to ``callback``."""

    world = make_report_world(organism, seed=seed, **overrides)
    packet = world.reset(seed)
    hidden = None
    rng = Random(seed + 59_000_003)
    callback("reset", packet, world, None, None)
    for _ in range(ticks):
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        # Speak the truly lowest need, so these guards run over long lives
        # rather than over an organism that dies of thirst asking for food.
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[world.lowest_need()]], PAD_ID))
        before = packet
        packet, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            action,
            consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        if terminated or truncated:
            break
        callback("step", packet, world, (before, action), info)
    return world


def test_the_self_jacobian_is_the_real_derivative(organism_pair):
    """The whole attribution rests on this, so it is checked against the truth.

    For each of its own constants, the organism's claimed sensitivity must equal
    what a filter genuinely rebuilt with that constant scaled would predict.
    The predicted body is linear in every one of these constants, so the
    correction is exact rather than first order and the agreement must be
    numerical, not approximate.
    """

    model, organism = organism_pair
    scale = 1.35

    for parameter in CALIBRATION_PARAMETERS:
        compared = 0
        for seed in (SEED_BASE + 3, SEED_BASE + 29):
            jacobian = SelfJacobian(organism, organism.report)
            scaled_report = replace(
                organism.report,
                **{parameter: getattr(organism.report, parameter) * scale},
            )
            rebuilt = _ObservableBodyFilter(replace(organism, report=scaled_report))
            index = CALIBRATION_PARAMETERS.index(parameter)
            # Clipping is a genuine nonlinearity and the one place the two
            # legitimately differ. It is not enough to skip a need sitting on a
            # bound *now*: one clip anywhere in its past has already destroyed
            # the linear relation, so a need is excluded from then on.
            ever_clipped = np.zeros(len(REPORT_NEEDS), dtype=bool)

            def callback(kind, packet, world, transition, info):
                nonlocal compared
                if kind == "reset":
                    jacobian.reset(packet)
                    rebuilt.reset(packet)
                    return
                before, action = transition
                jacobian.update(before, action, packet)
                rebuilt.update(before, action, packet)
                actual = rebuilt.belief
                claimed = jacobian.belief + (scale - 1.0) * jacobian.matrix()[:, index]
                for series in (actual, claimed, jacobian.belief):
                    ever_clipped[
                        (series <= 1e-9) | (series >= 1.0 - 1e-9)
                    ] = True
                free = ~ever_clipped
                if free.any():
                    np.testing.assert_allclose(
                        claimed[free], actual[free], atol=1e-12, rtol=0.0
                    )
                    compared += int(free.sum())

            _drive(organism, model, seed, callback, ticks=200)
        assert compared > 40, f"{parameter} was never compared off the bounds."


def test_a_new_organism_assumes_it_is_typical(organism_pair):
    """Before any evidence the self-model is exactly the species model."""

    model, organism = organism_pair
    seed = SEED_BASE + 5
    learner = SelfCalibration(organism, organism.report)
    plain = _ObservableBodyFilter(organism)
    checked = 0

    def callback(kind, packet, world, transition, info):
        nonlocal checked
        if kind == "reset":
            learner.reset(packet, world)
            plain.reset(packet)
            assert np.array_equal(
                learner.correction(), np.zeros((len(REPORT_NEEDS), 7))
            )
            return
        before, action = transition
        learner.update(before, action, packet, world, None)
        plain.update(before, action, packet)
        np.testing.assert_allclose(learner.point(), plain.belief, atol=1e-12)
        checked += 1

    _drive(organism, model, seed, callback, metabolic_spread=0.6)
    assert checked > 50
    # No readings arrived, so nothing about itself was learned.
    assert np.array_equal(learner.correction(), np.zeros((len(REPORT_NEEDS), 7)))


def test_it_learns_its_own_burn_rate(organism_pair):
    """With readings, the recovered rate must move toward this body's truth."""

    model, organism = organism_pair
    for arm in (SelfCalibration, RecursiveSelfCalibration):
        errors = []
        priors = []
        for life in range(4):
            seed = SEED_BASE + 900 + life
            learner = arm(organism, organism.report)

            def callback(kind, packet, world, transition, info):
                if kind == "reset":
                    learner.reset(packet, world)
                    return
                before, action = transition
                learner.update(
                    before, action, packet, world, info.get("interoception")
                )

            world = _drive(
                organism,
                model,
                seed,
                callback,
                ticks=380,
                metabolic_spread=0.6,
                interoception_probability=0.10,
            )
            truth = world.metabolic_scale
            recovered = learner.recovered_scale()
            errors.append(
                np.mean([abs(recovered[n] - truth[n]) for n in REPORT_NEEDS])
            )
            priors.append(np.mean([abs(1.0 - truth[n]) for n in REPORT_NEEDS]))
        assert np.mean(errors) < np.mean(priors), arm.__name__


def test_another_organisms_readings_teach_nothing_true(organism_pair):
    """The preregistered G5 control, at the level of one mechanism.

    A self-model that improves on another body's readings was never reading the
    evidence; it was exploiting the act of being corrected.
    """

    model, organism = organism_pair
    honest = run_world(
        model, organism, world_name="metabolic", lives=4, seed_base=SEED_BASE + 400
    )
    shuffled = run_world(
        model,
        organism,
        world_name="metabolic",
        lives=4,
        seed_base=SEED_BASE + 400,
        shuffle_readings=True,
    )
    for arm in ("learned", "recursive"):
        assert honest["body_error"][arm] < honest["body_error"]["population"]
        assert shuffled["body_error"][arm] >= honest["body_error"][arm]


def test_a_typical_body_discovers_nothing(organism_pair):
    """In a world where nothing is individual there is nothing to find."""

    model, organism = organism_pair
    row = run_world(
        model, organism, world_name="null", lives=4, seed_base=SEED_BASE + 600
    )
    for arm in ("learned", "recursive"):
        assert row["attribution"][arm]["total_correction"] < 1e-3
        assert abs(
            row["body_error"][arm] - row["body_error"]["population"]
        ) <= 0.005


def test_the_sanity_clamp_never_binds_on_a_real_body(organism_pair):
    """The clamp exists for the shuffled control and must be inert otherwise.

    If it ever bound on an organism reading its own body it would be a hidden
    prior on how atypical that body is allowed to be, and the reported
    corrections would be measuring the clamp.
    """

    model, organism = organism_pair
    for world_name in ("metabolic", "absorption", "both", "null"):
        row = run_world(
            model, organism, world_name=world_name, lives=4, seed_base=SEED_BASE
        )
        for arm in ("learned", "recursive"):
            total = float(row["attribution"][arm]["total_correction"])
            # 21 entries, each clamped at MAX_LOG_CORRECTION.
            budget = 21 * MAX_LOG_CORRECTION
            assert total < 0.25 * budget, (world_name, arm, total)


def test_the_oracle_tier_knows_itself_without_being_told_twice(organism_pair):
    """The context tier must carry this life's truth and never take a reading."""

    model, organism = organism_pair
    seed = SEED_BASE + 7
    oracle = OracleCalibration(organism, organism.report)
    seen = []

    def callback(kind, packet, world, transition, info):
        if kind == "reset":
            oracle.reset(packet, world)
            seen.append(world)
            return
        before, action = transition
        oracle.update(before, action, packet, world, info.get("interoception"))

    world = _drive(
        organism,
        model,
        seed,
        callback,
        metabolic_spread=0.6,
        interoception_probability=1.0,
    )
    truth = world.metabolic_scale
    correction = oracle.correction()
    for index, need in enumerate(REPORT_NEEDS):
        assert correction[index, index] == pytest.approx(np.log(truth[need]))
    # It was handed readings on every tick and used none of them.
    assert oracle.readings == 0
