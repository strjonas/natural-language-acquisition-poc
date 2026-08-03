"""Is there anything about *this* body worth knowing? A ceiling survey.

Probes 59, 60, 61 and 62 all died against the same wall, and probe62 finally put
a number on it: the body is bounded in [0,1], the filter saturates on ~9.7% of
ticks, and every saturation erases the accumulated error. A self-model whose
error is capped at 0.05 by homeostasis cannot be load-bearing however accurate
it becomes.

This module asks a question one level below that one, which this repository has
never asked:

    Is there anything in this organism's self-model that is about *itself*?

Today, no. Every organism in this ecology burns fuel at exactly the species
rate. ``food_metabolism`` is 0.008 for all of them, in every life, forever. So
the "self-model" is a model of *bodies in general* -- a physics that a designer
knows in closed form, which is precisely why probe53's hand-written filter is
exact and still beats everything learned. There is nothing individual to learn,
so learning cannot buy anything, and the standing obstacle in ``docs/STATE.md``
-- *nothing yet shows the learned model doing what the analytic filter cannot*
-- is not a gap in the mechanisms tried. It is a property of the ecology.

Probe63 changes that with two coupled levers, both default-inert:

    metabolic_spread            this life's body burns fuel at its own rate,
                                drawn per axis and fixed for the life. Species
                                constants are now systematically wrong about
                                this organism, in a need-specific direction that
                                persists -- phase B1's "systematically biased
                                body the current model cannot represent".
    interoception_probability   the only channel through which that could be
                                found out. The body is otherwise visible exactly
                                once, at birth, and then never again for 400
                                ticks.

Neither is meaningful alone. Without the first there is nothing about the self
to know; without the second the fact is unknowable in principle rather than
merely unknown.

Five tiers share one history, one policy and one motor path, and differ in
exactly one thing -- what they believe their own body is and does:

    oracle       reads the true body. The ceiling.
    population   probe53's filter on species constants. *Today's repository.*
    individual   the same filter on this life's true constants. The ceiling on
                 what knowing your own body is worth, with no evidence at all.
    snap         species constants, corrected to truth whenever a reading
                 arrives. Evidence without a model of the self.
    corrigible   estimates its own rates from those same readings, and carries
                 the estimate between them.

The contrasts are the point:

    individual - population   headroom. Is individuality worth anything here?
    snap - population         what raw evidence buys. Phase B1 warns that its
                              first gate is "satisfiable by clipping"; this is
                              that baseline, measured rather than assumed.
    corrigible - snap         what *modelling yourself* buys over merely being
                              corrected. The only contrast that is about a self
                              model rather than about an observation.
    oracle - individual       what is left for anything better.

Nothing here is learned. These are instruments for measuring the ecology,
exactly as in probe62, and this file is a survey rather than a result.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.env import Action
from homesocial.island.report import (
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportWorld,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import (
    FIDELITY_WARMUP,
    _ObservableBodyFilter,
    make_report_world,
)
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    ACTIONS,
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
TIERS = ("oracle", "population", "individual", "snap", "corrigible")
SPREAD_SWEEP = (0.0, 0.15, 0.30, 0.45, 0.60)
INTEROCEPTION_SWEEP = (0.01, 0.03, 0.10)
# Disjoint from every developmental and evaluation band already in use,
# including probe62's 880,000,000.
INDIVIDUAL_SEED_BASE = 930_000_000
# A belief within this of a bound has had its history erased by clipping, so a
# residual measured there says nothing about the rate that produced it.
BOUND_EPSILON = 1e-6


def _true_body(world: ReportWorld) -> np.ndarray:
    return np.asarray(
        [getattr(world.grid.needs, need) for need in REPORT_NEEDS], dtype=np.float64
    )


# -- how much metabolism one lived action spends ------------------------------


def metabolic_ticks(
    filter_: _ObservableBodyFilter, before, action_index: int, after
) -> tuple[int, int, int]:
    """``(duration, moving_energy_ticks, resting_energy_ticks)`` for one action.

    Mirrors the arithmetic inside ``_ObservableBodyFilter.update`` exactly:
    food and water spend one tick of metabolism per primitive tick, while energy
    spends the moving rate on ticks that moved and the resting rate otherwise.
    Everything it reads is already learner-visible -- the option's routed
    distance and the organism's own selected action -- so counting this requires
    no privileged information.
    """

    duration = after.step_count - before.step_count
    _, _, routed_moves = filter_._terminal_surface(before, action_index)
    primitive_move = (
        action_index < len(ACTIONS)
        and ACTIONS[action_index] == Action.MOVE_FORWARD
    )
    pre_ticks = duration - 1
    moves = min(pre_ticks, routed_moves) + int(primitive_move)
    return duration, moves, duration - moves


# -- the five bodies ----------------------------------------------------------


class _Tier:
    """Common interface. ``point()`` is what the report and the audit read."""

    def reset(self, packet, world: ReportWorld) -> None:
        raise NotImplementedError

    def update(
        self, before, action_index: int, after, world: ReportWorld, reading
    ) -> None:
        raise NotImplementedError

    def point(self) -> np.ndarray:
        raise NotImplementedError

    def scale(self) -> dict[str, float] | None:
        """This tier's belief about its own rates, if it has one."""

        return None


class OracleTier(_Tier):
    """Reads the true body. No filtering, no error, no self-model."""

    def __init__(self) -> None:
        self._body = np.zeros(3)

    def reset(self, packet, world: ReportWorld) -> None:
        self._body = _true_body(world)

    def update(self, before, action_index, after, world, reading) -> None:
        self._body = _true_body(world)

    def point(self) -> np.ndarray:
        return self._body.copy()


class FilterTier(_Tier):
    """Probe53's exact filter over some set of constants, optionally corrected.

    ``population`` and ``individual`` differ only in which report they are
    handed; ``snap`` is ``population`` plus the readings. That keeps the whole
    ladder one class, so no contrast can be an artefact of two implementations
    of the same filter.
    """

    def __init__(
        self, organism: OrganismConfig, report, *, use_individual: bool, snap: bool
    ) -> None:
        self._organism = organism
        self._species = report
        self._use_individual = use_individual
        self._snap = snap
        self._filter = _ObservableBodyFilter(replace(organism, report=report))

    def reset(self, packet, world: ReportWorld) -> None:
        report = world.individual_report() if self._use_individual else self._species
        self._filter = _ObservableBodyFilter(
            replace(self._organism, report=report)
        )
        self._filter.reset(packet)

    def update(self, before, action_index, after, world, reading) -> None:
        self._filter.update(before, action_index, after)
        if self._snap and reading is not None:
            belief = self._filter._belief
            assert belief is not None
            belief[...] = np.asarray(reading, dtype=np.float64)

    def point(self) -> np.ndarray:
        return self._filter.belief


class CorrigibleTier(_Tier):
    """Estimates its own metabolic rates from its own readings, and keeps them.

    The estimator is exact arithmetic, not a fit. Over the interval between two
    readings the species-rate filter's error decomposes cleanly. Writing ``B``
    for the metabolic loss the species rate predicts over that interval and
    ``A`` for the loss actually incurred, uptake and visible shocks are common
    to both and cancel, leaving

        species_belief - reading  ==  A - B

    so ``A = B + residual`` with everything on the right observable. The rate
    multiplier is then the ratio of totals, shrunk toward the species prior:

        scale = (sum A + lambda) / (sum B + lambda)

    with ``lambda`` fixed a priori at one help period of species metabolism --
    "one help window of evidence to move off the prior". It is not tuned against
    any outcome in this survey.

    Two things corrupt the residual and are handled rather than ignored. A
    belief or a reading sitting on a bound has had its history erased by
    clipping, so those intervals are skipped per need. And a shock that landed
    inside a temporally abstract option is invisible, so it is charged to
    metabolism; that biases every estimate upward, identically for any learner
    facing this ecology, and it is reported rather than corrected away.
    """

    def __init__(self, organism: OrganismConfig, report) -> None:
        self._organism = organism
        self._species = report
        self._prior = {
            need: getattr(report, f"{need}_metabolism") * report.help_period
            for need in REPORT_NEEDS
        }
        self._belief_filter = _ObservableBodyFilter(
            replace(organism, report=report)
        )
        self._reference = _ObservableBodyFilter(replace(organism, report=report))
        self._loss = {need: 0.0 for need in REPORT_NEEDS}
        self._predicted = {need: 0.0 for need in REPORT_NEEDS}
        self._pending = np.zeros(3)
        self._scale = {need: 1.0 for need in REPORT_NEEDS}
        self.skipped = 0
        self.used = 0

    # -- the estimate --------------------------------------------------------

    def _scaled_report(self):
        return replace(
            self._species,
            food_metabolism=self._species.food_metabolism * self._scale["food"],
            water_metabolism=self._species.water_metabolism * self._scale["water"],
            energy_metabolism=self._species.energy_metabolism * self._scale["energy"],
            move_energy_metabolism=(
                self._species.move_energy_metabolism * self._scale["energy"]
            ),
        )

    def _refresh_scale(self) -> None:
        for index, need in enumerate(REPORT_NEEDS):
            prior = self._prior[need]
            self._scale[need] = (self._loss[need] + prior) / (
                self._predicted[need] + prior
            )
        self._belief_filter.report = self._scaled_report()

    def reset(self, packet, world: ReportWorld) -> None:
        self._belief_filter = _ObservableBodyFilter(
            replace(self._organism, report=self._species)
        )
        self._reference = _ObservableBodyFilter(
            replace(self._organism, report=self._species)
        )
        self._belief_filter.reset(packet)
        self._reference.reset(packet)
        self._loss = {need: 0.0 for need in REPORT_NEEDS}
        self._predicted = {need: 0.0 for need in REPORT_NEEDS}
        self._pending = np.zeros(3)
        self._scale = {need: 1.0 for need in REPORT_NEEDS}
        self._refresh_scale()

    def update(self, before, action_index, after, world, reading) -> None:
        duration, moves, rests = metabolic_ticks(
            self._reference, before, action_index, after
        )
        species = self._species
        self._pending += np.asarray(
            [
                species.food_metabolism * duration,
                species.water_metabolism * duration,
                species.move_energy_metabolism * moves
                + species.energy_metabolism * rests,
            ]
        )

        self._belief_filter.update(before, action_index, after)
        self._reference.update(before, action_index, after)
        if reading is None:
            return

        truth = np.asarray(reading, dtype=np.float64)
        reference = self._reference.belief
        for index, need in enumerate(REPORT_NEEDS):
            predicted = float(self._pending[index])
            at_bound = (
                reference[index] <= BOUND_EPSILON
                or reference[index] >= 1.0 - BOUND_EPSILON
                or truth[index] <= BOUND_EPSILON
                or truth[index] >= 1.0 - BOUND_EPSILON
            )
            if at_bound or predicted <= 0.0:
                self.skipped += 1
                continue
            residual = float(reference[index] - truth[index])
            self._loss[need] += predicted + residual
            self._predicted[need] += predicted
            self.used += 1
        self._refresh_scale()

        # Both filters restart from the reading: the estimator's algebra assumes
        # the interval begins at a known body, and the belief has no reason to
        # carry an error it has just been shown.
        for filter_ in (self._belief_filter, self._reference):
            belief = filter_._belief
            assert belief is not None
            belief[...] = truth
        self._pending = np.zeros(3)

    def point(self) -> np.ndarray:
        return self._belief_filter.belief

    def scale(self) -> dict[str, float]:
        return dict(self._scale)


def make_tier(name: str, organism: OrganismConfig, report) -> _Tier:
    if name == "oracle":
        return OracleTier()
    if name == "population":
        return FilterTier(organism, report, use_individual=False, snap=False)
    if name == "individual":
        return FilterTier(organism, report, use_individual=True, snap=False)
    if name == "snap":
        return FilterTier(organism, report, use_individual=False, snap=True)
    if name == "corrigible":
        return CorrigibleTier(organism, report)
    raise ValueError(f"Unknown tier: {name}.")


# -- which of its own constants could be wrong --------------------------------

# The organism's own causal parameters: every constant in its model of its body
# that could differ between it and its species. Ordered so that the first four
# are how fast it spends and the last three are what the world does to it.
CALIBRATION_PARAMETERS = (
    "food_metabolism",
    "water_metabolism",
    "energy_metabolism",
    "move_energy_metabolism",
    "portion_small",
    "portion_large",
    "shock_size",
)
METABOLIC_PARAMETERS = CALIBRATION_PARAMETERS[:4]
UPTAKE_PARAMETERS = CALIBRATION_PARAMETERS[4:6]
JACOBIAN_DELTA = 1e-3


class SelfJacobian:
    """How this organism's predicted body moves when its own constants move.

    One filter per parameter, each carrying the identical history with exactly
    one of its own constants perturbed. The column-wise difference from the
    unperturbed filter is the derivative of its predicted body with respect to
    that parameter.

    This is the reflexive quantity. It is not about where the body is; it is
    about how the organism's *model* of its body would respond if a particular
    belief about itself were wrong -- which is the question it has to answer to
    find out what about itself is miscalibrated. Everything it needs is its own
    model and its own history, so nothing privileged enters.
    """

    def __init__(
        self,
        organism: OrganismConfig,
        report,
        *,
        delta: float = JACOBIAN_DELTA,
    ) -> None:
        self._delta = delta
        self._base = _ObservableBodyFilter(replace(organism, report=report))
        self._perturbed = {
            name: _ObservableBodyFilter(
                replace(
                    organism,
                    report=replace(
                        report, **{name: getattr(report, name) * (1.0 + delta)}
                    ),
                )
            )
            for name in CALIBRATION_PARAMETERS
        }

    def reset(self, packet) -> None:
        self._base.reset(packet)
        for filter_ in self._perturbed.values():
            filter_.reset(packet)

    def update(self, before, action_index: int, after) -> None:
        self._base.update(before, action_index, after)
        for filter_ in self._perturbed.values():
            filter_.update(before, action_index, after)

    @property
    def belief(self) -> np.ndarray:
        return self._base.belief

    def matrix(self) -> np.ndarray:
        """``(3, P)``: d(predicted body) / d(log parameter), at the current history."""

        base = self._base.belief
        return np.stack(
            [
                (self._perturbed[name].belief - base) / self._delta
                for name in CALIBRATION_PARAMETERS
            ],
            axis=1,
        )

    def snap(self, truth: np.ndarray) -> None:
        """Restart every filter from a known body, so the next interval is clean."""

        for filter_ in (self._base, *self._perturbed.values()):
            belief = filter_._belief
            assert belief is not None
            belief[...] = truth


def localization_ceiling(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    metabolic_spread: float,
    uptake_spread: float,
    interoception: float,
    lives: int,
    seed_base: int,
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Can *anything* tell apart a body that burns fast from one that absorbs badly?

    Pools every (Jacobian, residual) pair the organism could observe across all
    lives and solves one least-squares problem for the parameter corrections
    that best explain them. This is the ceiling on localization, not a learner:
    if the pooled fit cannot separate the two families of parameter, then no
    online mechanism can either, and a localization gate would be measuring
    collinearity rather than self-knowledge.

    Pooling across lives deliberately overstates what a single organism can do.
    Every life has a *different* body, so a pooled fit recovers only what the
    spread has in common -- which is nothing. It is run per life and averaged.
    """

    rows: list[np.ndarray] = []
    recovered: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(
            organism,
            seed=seed,
            metabolic_spread=metabolic_spread,
            uptake_spread=uptake_spread,
            interoception_probability=interoception,
        )
        packet = world.reset(seed)
        jacobian = SelfJacobian(organism, world.report)
        jacobian.reset(packet)
        motor_hidden = None
        motor_rng = Random(seed + 59_000_003)
        design: list[list[np.ndarray]] = [[] for _ in REPORT_NEEDS]
        target: list[list[float]] = [[] for _ in REPORT_NEEDS]

        while True:
            need = REPORT_NEEDS[int(np.argmin(jacobian.belief))]
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID))
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
            jacobian.update(before, action, packet)
            reading = info.get("interoception")
            if reading is None:
                continue
            truth = np.asarray(reading, dtype=np.float64)
            residual = truth - jacobian.belief
            matrix = jacobian.matrix()
            # A need sitting on a bound has had its history erased by clipping,
            # so its row explains nothing about any parameter.
            for index in range(len(REPORT_NEEDS)):
                if BOUND_EPSILON < truth[index] < 1.0 - BOUND_EPSILON:
                    design[index].append(matrix[index])
                    target[index].append(float(residual[index]))
            jacobian.snap(truth)

        if min(len(design[index]) for index in range(len(REPORT_NEEDS))) < 8:
            continue
        # One fit per need, not one pooled fit. A shared coefficient would force
        # "how well I absorb food" and "how well I absorb water" to be the same
        # fact about this body; they are not, and a self-model that cannot say
        # so has nowhere to put a per-need difference except into a per-need
        # parameter it does have -- its metabolism. That is mis-specification
        # presenting as misattribution, and it is a property of the model rather
        # than of the ecology.
        solution = np.zeros((len(REPORT_NEEDS), len(CALIBRATION_PARAMETERS)))
        for index in range(len(REPORT_NEEDS)):
            matrix = np.asarray(design[index])
            vector = np.asarray(target[index])
            gram = matrix.T @ matrix + ridge * np.eye(len(CALIBRATION_PARAMETERS))
            solution[index] = np.linalg.solve(gram, matrix.T @ vector)
        recovered.append(solution)
        metabolic = world.metabolic_scale
        uptake = world.uptake_scale
        truths.append(
            np.asarray(
                [
                    np.log(metabolic["food"]),
                    np.log(metabolic["water"]),
                    np.log(metabolic["energy"]),
                    np.log(metabolic["energy"]),
                    np.log(np.mean([uptake[n] for n in REPORT_NEEDS])),
                    np.log(np.mean([uptake[n] for n in REPORT_NEEDS])),
                    0.0,
                ]
            )
        )
        rows.append(np.asarray([sum(len(block) for block in design)]))

    if not recovered:
        raise RuntimeError("No life produced enough readings to identify anything.")

    # (lives, needs, parameters) -> mean |estimate| per parameter, pooled over
    # the needs each parameter can act on.
    estimates = np.asarray(recovered)
    magnitude = np.abs(estimates).mean(axis=0).sum(axis=0)
    metabolic_mass = float(magnitude[:4].sum())
    uptake_mass = float(magnitude[4:6].sum())
    shock_mass = float(magnitude[6])
    total = metabolic_mass + uptake_mass + shock_mass
    # In a world where nothing is individual the correct answer is no correction
    # at all, and a share of nothing is not a number. Reporting 0.41/0.47 there
    # would look like a smeared attribution when what actually happened is that
    # the fit returned exactly zero.
    identified = total > 1e-9
    return {
        "metabolic_spread": metabolic_spread,
        "uptake_spread": uptake_spread,
        "interoception_probability": interoception,
        "lives_identified": len(recovered),
        "rows_per_life": float(np.mean([row[0] for row in rows])),
        "parameters": list(CALIBRATION_PARAMETERS),
        "mean_abs_estimate": magnitude.tolist(),
        "total_correction": total,
        "metabolic_share": metabolic_mass / total if identified else None,
        "uptake_share": uptake_mass / total if identified else None,
        "shock_share": shock_mass / total if identified else None,
        "mean_abs_truth": np.abs(np.asarray(truths)).mean(axis=0).tolist(),
    }


# -- the survey ---------------------------------------------------------------


def run_open_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    spread: float,
    interoception: float,
    lives: int,
    seed_base: int,
    driver: str = "population",
) -> dict[str, object]:
    """Every tier scored on one shared history.

    One tier drives and all of them watch the same transitions, so a difference
    between tiers is a difference between estimators and not between three
    trajectories that diverged. This is probe62's open-loop design, and it
    carries the endpoint that no policy can flatten: how far each tier's belief
    is from the body it is a belief about.
    """

    report = replace(
        organism.report,
        metabolic_spread=spread,
        interoception_probability=interoception,
    )
    error = {tier: 0.0 for tier in TIERS}
    argmin_hits = {tier: 0 for tier in TIERS}
    ticks = 0
    readings = 0
    steps = 0
    scale_error = 0.0
    scale_lives = 0
    skipped = 0
    used = 0

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(
            organism,
            seed=seed,
            metabolic_spread=spread,
            interoception_probability=interoception,
        )
        packet = world.reset(seed)
        tiers = {name: make_tier(name, organism, report) for name in TIERS}
        for tier in tiers.values():
            tier.reset(packet, world)
        motor_hidden = None
        motor_rng = Random(seed + 59_000_003)

        while True:
            truth = _true_body(world)
            if packet.step_count >= FIDELITY_WARMUP:
                ticks += 1
                for name, tier in tiers.items():
                    believed = tier.point()
                    error[name] += float(np.abs(believed - truth).mean())
                    argmin_hits[name] += int(
                        int(np.argmin(believed)) == int(np.argmin(truth))
                    )
            need = REPORT_NEEDS[int(np.argmin(tiers[driver].point()))]
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            steps += int(info["duration"])
            reading = info.get("interoception")
            readings += int(reading is not None)
            if terminated or truncated:
                break
            for tier in tiers.values():
                tier.update(before, action, packet, world, reading)

        corrigible = tiers["corrigible"]
        estimated = corrigible.scale()
        assert estimated is not None
        actual = world.metabolic_scale
        scale_error += float(
            np.mean([abs(estimated[n] - actual[n]) for n in REPORT_NEEDS])
        )
        scale_lives += 1
        skipped += corrigible.skipped
        used += corrigible.used

    denominator = max(1, ticks)
    return {
        "metabolic_spread": spread,
        "interoception_probability": interoception,
        "lives": lives,
        "driver": driver,
        "scored_ticks": ticks,
        "readings_per_life": readings / lives,
        "mean_life_steps": steps / lives,
        "body_error": {tier: error[tier] / denominator for tier in TIERS},
        "argmin_accuracy": {
            tier: argmin_hits[tier] / denominator for tier in TIERS
        },
        "rate_error": scale_error / max(1, scale_lives),
        "prior_rate_error": _prior_rate_error(spread),
        "estimator_intervals_used": used / lives,
        "estimator_intervals_skipped": skipped / lives,
    }


def _prior_rate_error(spread: float) -> float:
    """Mean |scale - 1| for the species prior: what believing you are typical costs.

    Uniform on [1-s, 1+s], so the expectation is s/2 exactly. It is the number
    the corrigible tier's own rate error has to beat to have learned anything
    about itself at all.
    """

    return spread / 2.0


def run_closed_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    tier_name: str,
    spread: float,
    interoception: float,
    lives: int,
    seed_base: int,
) -> dict[str, object]:
    """One tier speaking for itself, so it lives its own life. Survival only.

    Reported and never gated. Probes 59 and 60 established that this ecology's
    help loop flattens belief-side differences far larger than anything here is
    likely to produce, so survival is carried as context for the belief numbers
    rather than as evidence about them.
    """

    report = replace(
        organism.report,
        metabolic_spread=spread,
        interoception_probability=interoception,
    )
    survived = 0
    truthful = 0
    said = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(
            organism,
            seed=seed,
            metabolic_spread=spread,
            interoception_probability=interoception,
        )
        packet = world.reset(seed)
        tier = make_tier(tier_name, organism, report)
        tier.reset(packet, world)
        motor_hidden = None
        motor_rng = Random(seed + 59_000_003)
        while True:
            need = REPORT_NEEDS[int(np.argmin(tier.point()))]
            if packet.step_count >= FIDELITY_WARMUP:
                said += 1
                truthful += int(need == world.lowest_need())
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            if terminated or truncated:
                survived += int(not terminated)
                break
            tier.update(before, action, packet, world, info.get("interoception"))
    return {
        "tier": tier_name,
        "metabolic_spread": spread,
        "interoception_probability": interoception,
        "lives": lives,
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
    }


def survey(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    closed_loop_lives: int,
) -> dict[str, object]:
    """The headroom sweep, then the evidence sweep, then closed-loop context."""

    headroom = [
        run_open_loop(
            model,
            organism,
            spread=spread,
            interoception=0.0,
            lives=lives,
            seed_base=seed_base,
        )
        for spread in SPREAD_SWEEP
    ]

    strongest = max(SPREAD_SWEEP)
    evidence = [
        run_open_loop(
            model,
            organism,
            spread=strongest,
            interoception=rate,
            lives=lives,
            seed_base=seed_base,
        )
        for rate in INTEROCEPTION_SWEEP
    ]

    closed = []
    if closed_loop_lives:
        for tier_name in TIERS:
            closed.append(
                run_closed_loop(
                    model,
                    organism,
                    tier_name=tier_name,
                    spread=strongest,
                    interoception=INTEROCEPTION_SWEEP[-1],
                    lives=closed_loop_lives,
                    seed_base=seed_base + 500_000,
                )
            )

    return {
        "headroom": headroom,
        "evidence": evidence,
        "closed_loop": closed,
        "tiers": list(TIERS),
        "spread_sweep": list(SPREAD_SWEEP),
        "interoception_sweep": list(INTEROCEPTION_SWEEP),
        "seed_base": seed_base,
    }


def _format(rows: list[dict[str, object]], key: str, axis: str) -> str:
    header = f"| {axis} | " + " | ".join(TIERS) + " |"
    rule = "|---:|" + "---:|" * len(TIERS)
    lines = [header, rule]
    for row in rows:
        values = row[key]
        assert isinstance(values, dict)
        cells = " | ".join(f"{values[tier]:.4f}" for tier in TIERS)
        lines.append(f"| {row[axis]:.2f} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--lives", type=int, default=40)
    parser.add_argument("--closed-loop-lives", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=INDIVIDUAL_SEED_BASE)
    parser.add_argument(
        "--out", default="runs/organism/probe63_individual_self/ceiling_survey.json"
    )
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(args.checkpoint)
    result = survey(
        model,
        organism,
        lives=args.lives,
        seed_base=args.seed_base,
        closed_loop_lives=args.closed_loop_lives,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("\n== body error, no interoception ==")
    print(_format(result["headroom"], "body_error", "metabolic_spread"))
    print("\n== named-need accuracy, no interoception ==")
    print(_format(result["headroom"], "argmin_accuracy", "metabolic_spread"))
    print(f"\n== body error at spread {max(SPREAD_SWEEP)} ==")
    print(_format(result["evidence"], "body_error", "interoception_probability"))
    print(f"\n== named-need accuracy at spread {max(SPREAD_SWEEP)} ==")
    print(_format(result["evidence"], "argmin_accuracy", "interoception_probability"))
    print("\n== corrigible tier's model of its own rates ==")
    print("| q | readings/life | rate error | species prior error |")
    print("|---:|---:|---:|---:|")
    for row in result["evidence"]:
        print(
            f"| {row['interoception_probability']:.2f} "
            f"| {row['readings_per_life']:.1f} "
            f"| {row['rate_error']:.4f} "
            f"| {row['prior_rate_error']:.4f} |"
        )
    for row in result["closed_loop"]:
        print(
            f"closed loop {row['tier']:>11}: survival {row['survival']:.3f} "
            f"fidelity {row['report_fidelity']:.3f}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
