"""Is there anything for self-uncertainty to buy? A ceiling survey, not a result.

Probes 59 and 60 both ran into the same wall. A mechanism was built, it worked
on the belief side, and the behavioural gate came back flat -- not because the
mechanism failed but because the ecology had no headroom for it. Probe60's
verdict was explicit: *check the oracle ceiling before locking a gate; if a
perfect model cannot reach the threshold, the gate measures something else.*

This module is that check, run before anything is preregistered. It asks one
question:

    In this ecology, is there any gap between an organism that knows where its
    body is, one that knows only its average hidden drift, and one that knows
    how uncertain it is?

If the three collapse onto each other, no amount of learned uncertainty can pay
rent here and the ecology has to change first. If they separate, the gap is the
budget any later mechanism has to earn.

The lever is `ReportConfig.silent_shock_probability`. A silent shock changes the
body exactly as a loud one does, is drawn from the same stream, and differs only
in withholding its perceptible marker. So sweeping it moves what the organism
can *know* about itself while leaving what *happens* to it untouched. That
matters for the comparison this repository has not been able to make: probe53's
hand-written filter is exact precisely because every body-changing event is
perceptible. Take that away and being told your own dynamics stops being
equivalent to knowing them.

Four tiers share one policy, one motor path, one listener, and one set of lives.
They differ only in what stands in for the body:

    oracle          the true body, read directly. The ceiling.
    naive           probe53's exact visible-history filter. Sees loud shocks,
                    silently misses the rest, so it drifts optimistic.
    mean_corrected  the same filter minus the *expected* unobserved loss. A
                    point estimate that knows a statistical fact about itself.
    posterior       a particle cloud over the same history, conditioned on the
                    one thing the organism does observe about its hidden state
                    -- that it is still alive -- and acting on the expected
                    minimum rather than the minimum of the expectation.

`mean_corrected - naive` is what knowing your own average hidden drift is worth.
`posterior - mean_corrected` is what representing your uncertainty is worth on
top of that. `oracle - posterior` is what is left for anything better.

Nothing here is learned. These are instruments for measuring the ecology.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import (
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    SHOCK_SURFACE_NEEDS,
    ReportWorld,
)
from homesocial.island.world import SURFACES
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import (
    FIDELITY_WARMUP,
    _ObservableBodyFilter,
    make_report_world,
)
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
TIERS = ("oracle", "naive", "mean_corrected", "posterior", "posterior_vote")
SILENCE_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)
# Disjoint from every developmental and evaluation band already in use.
CEILING_SEED_BASE = 880_000_000


# -- what an organism could know about the shocks it cannot see ---------------


@dataclass(frozen=True)
class ShockStatistics:
    """The rate of body change that leaves no trace in the visible history.

    Two things hide a shock. Silence withholds its marker outright. Temporal
    abstraction hides it too: a shock landing inside a multi-tick option is
    already invisible in this ecology whatever the silence rate, which is why
    the naive filter is 99.96% rather than exact. Both are folded in here.
    """

    probability: float
    size: float
    silent_probability: float
    needs: int = len(REPORT_NEEDS)

    def per_need(self) -> float:
        return self.probability / self.needs

    def interior_event_rate(self) -> float:
        """P(an unobserved shock *event* on a non-terminal tick).

        Nothing inside an option is ever perceptible, so silence is irrelevant.
        An event lands on exactly one need, which is the whole point: the world
        never shocks two needs at once, so a posterior that draws each need
        independently invents variance the body does not have -- and ranking
        which need is lowest is precisely the question that variance corrupts.
        """

        return self.probability

    def terminal_event_rate(self, *, marker_seen: bool) -> float:
        """P(an unobserved shock event on the terminal tick).

        A marker settles the tick outright. With no marker the tick was either
        quiet or silently shocked, and the posterior over the two is exact.
        """

        if marker_seen:
            return 0.0
        quiet = 1.0 - self.probability
        silent = self.probability * self.silent_probability
        denominator = quiet + silent
        if denominator <= 0.0:
            return 0.0
        return silent / denominator

    def interior_rate(self) -> float:
        """P(a *given need* took an unobserved shock on a non-terminal tick)."""

        return self.interior_event_rate() / self.needs

    def terminal_rate(self, *, marker_seen: bool) -> float:
        """P(a *given need* took an unobserved shock on the terminal tick)."""

        return self.terminal_event_rate(marker_seen=marker_seen) / self.needs

    def expected_loss(self, *, duration: int, marker_seen: bool) -> float:
        """Expected unobserved loss for one need across one lived transition.

        Exactly the mean of the particle cloud's injection, so the corrected
        point estimate is the posterior mean and nothing else. Interior ticks
        are hidden by abstraction whatever the silence rate; the terminal tick
        is hidden only when no marker arrived.
        """

        interior = max(0, duration - 1)
        return self.size * (
            interior * self.interior_rate()
            + self.terminal_rate(marker_seen=marker_seen)
        )


def marker_need(packet) -> str | None:
    """The shock need a marker in this packet names, if any."""

    for _, _, surface_index in packet.visible:
        need = SHOCK_SURFACE_NEEDS.get(SURFACES[surface_index])
        if need is not None:
            return need
    return None


# -- the four bodies ----------------------------------------------------------


class _Tier:
    """Common interface. `body()` is what the policy and the audit both read."""

    def reset(self, packet, world: ReportWorld) -> None:
        raise NotImplementedError

    def update(self, before, action_index: int, after, world: ReportWorld) -> None:
        raise NotImplementedError

    def point(self) -> np.ndarray:
        raise NotImplementedError

    def cloud(self) -> np.ndarray:
        """Body samples, shape (particles, 3). A point tier returns one row."""

        return self.point()[None, :]


class OracleTier(_Tier):
    """Reads the true body. No filtering, no error, no uncertainty."""

    def __init__(self, statistics: ShockStatistics) -> None:
        self._statistics = statistics
        self._body = np.zeros(3)

    def reset(self, packet, world: ReportWorld) -> None:
        self._body = _true_body(world)

    def update(self, before, action_index: int, after, world: ReportWorld) -> None:
        self._body = _true_body(world)

    def point(self) -> np.ndarray:
        return self._body.copy()


class NaiveTier(_Tier):
    """Probe53's exact visible-history filter, unchanged."""

    def __init__(self, organism: OrganismConfig, report) -> None:
        self._filter = _ObservableBodyFilter(replace(organism, report=report))

    def reset(self, packet, world: ReportWorld) -> None:
        self._filter.reset(packet)

    def update(self, before, action_index: int, after, world: ReportWorld) -> None:
        self._filter.update(before, action_index, after)

    def point(self) -> np.ndarray:
        return self._filter.belief


class MeanCorrectedTier(NaiveTier):
    """The same filter, minus the loss it has learned to expect but cannot see."""

    def __init__(
        self, organism: OrganismConfig, report, statistics: ShockStatistics
    ) -> None:
        super().__init__(organism, report)
        self._statistics = statistics

    def update(self, before, action_index: int, after, world: ReportWorld) -> None:
        super().update(before, action_index, after, world)
        belief = self._filter._belief
        assert belief is not None
        belief -= self._statistics.expected_loss(
            duration=after.step_count - before.step_count,
            marker_seen=marker_need(after) is not None,
        )
        np.clip(belief, 0.0, 1.0, out=belief)


class PosteriorTier(_Tier):
    """A particle cloud over the same visible history.

    Every particle runs the identical exact dynamics; they differ only in the
    unobserved shocks each one supposes it took. Two observations prune them:
    a marker, which settles its tick outright, and being alive, which rules out
    every particle whose supposed history would already have killed the
    organism. That second one is the only evidence about its hidden state that
    the organism gets for free, and it is real evidence.
    """

    def __init__(
        self,
        organism: OrganismConfig,
        report,
        statistics: ShockStatistics,
        *,
        particles: int,
        seed: int,
    ) -> None:
        self._filter = _ObservableBodyFilter(replace(organism, report=report))
        self._statistics = statistics
        self._particles = particles
        self._rng = np.random.default_rng(seed)
        self._collapsed = 0

    def reset(self, packet, world: ReportWorld) -> None:
        self._filter.reset(packet)
        born = self._filter.belief
        self._filter._belief = np.repeat(born[None, :], self._particles, axis=0)

    def update(self, before, action_index: int, after, world: ReportWorld) -> None:
        self._filter.update(before, action_index, after)
        cloud = self._filter._belief
        assert cloud is not None

        duration = after.step_count - before.step_count
        interior = max(0, duration - 1)
        seen = marker_need(after) is not None
        needs = len(REPORT_NEEDS)
        # Shock *events*, then one need per event -- the world's own generative
        # order. Drawing per need independently would let a particle suppose
        # three simultaneous shocks and would erase the anti-correlation that
        # tells the organism which need is lowest.
        events = self._rng.binomial(
            interior, self._statistics.interior_event_rate(), self._particles
        ) + self._rng.binomial(
            1,
            self._statistics.terminal_event_rate(marker_seen=seen),
            self._particles,
        )
        taken = np.zeros((self._particles, needs), dtype=np.int64)
        total = int(events.sum())
        if total:
            owner = np.repeat(np.arange(self._particles), events)
            target = self._rng.integers(0, needs, total)
            np.add.at(taken, (owner, target), 1)
        proposed = cloud - taken * self._statistics.size

        # Conditioning on being alive. A particle that would already be dead is
        # not a possible present, whatever its prior weight.
        alive = np.all(proposed > 0.0, axis=-1)
        if alive.all():
            # No particle is refuted, so every weight is equal and resampling
            # would be pure loss: drawing with replacement from a uniform cloud
            # duplicates particles and collapses its diversity a little on every
            # one of four hundred ticks. Only a refutation may reshape it.
            cloud[...] = np.clip(proposed, 0.0, 1.0)
            return
        if not alive.any():
            # Every hypothesis is refuted. The cloud has lost the body; keep the
            # least-refuted particles rather than inventing a new prior.
            self._collapsed += 1
            order = np.argsort(np.min(proposed, axis=-1))[::-1]
            keep = order[: max(1, self._particles // 8)]
            resampled = proposed[self._rng.choice(keep, size=self._particles)]
        else:
            living = np.flatnonzero(alive)
            resampled = proposed[self._rng.choice(living, size=self._particles)]
        cloud[...] = np.clip(resampled, 0.0, 1.0)

    def point(self) -> np.ndarray:
        cloud = self._filter._belief
        assert cloud is not None
        return np.asarray(cloud.mean(axis=0))

    def cloud(self) -> np.ndarray:
        cloud = self._filter._belief
        assert cloud is not None
        return np.asarray(cloud)

    @property
    def collapses(self) -> int:
        return self._collapsed


class PosteriorVoteTier(PosteriorTier):
    """The same cloud, reporting the need most likely to be the lowest one.

    Naming `argmin` of the posterior *mean* is not the Bayes rule for the
    question the organism is actually asked. Under a broad posterior the need
    most often lowest across particles need not be the need lowest on average,
    and the difference is exactly what representing uncertainty is supposed to
    buy. If this tier does not beat the point tiers, uncertainty is not paying
    for itself here whatever its calibration looks like.
    """

    def lowest_probabilities(self, *, ticks_to_help: int, drift: np.ndarray) -> np.ndarray:
        projected = self.cloud() - ticks_to_help * drift[None, :]
        winners = np.argmin(projected, axis=-1)
        counts = np.bincount(winners, minlength=len(REPORT_NEEDS))
        return counts / max(1, counts.sum())


def _true_body(world: ReportWorld) -> np.ndarray:
    return np.array(
        [getattr(world.grid.needs, need) for need in REPORT_NEEDS], dtype=np.float64
    )


# -- one policy, shared by every tier -----------------------------------------


def choose_need(
    tier: _Tier, *, step_count: int, report, drift: np.ndarray
) -> str:
    """Probe60's promoted objective: the expected minimum, not the minimum
    expected.

    Enumerates the three need words against the true listener rule -- the named
    need is granted, small or large with equal probability -- and takes the
    expectation over the portion draw and, for a tier that has one, over the
    body cloud. Identical arithmetic for every tier. Only `tier.cloud()` differs.
    """

    # Forward projection is metabolism only and is the same arithmetic for
    # every tier. The tiers are allowed to differ in exactly one thing -- where
    # they believe the body is right now -- so that a gap cannot be credited to
    # a better forecast of what the world is about to do.
    ticks_to_help = report.help_period - (step_count % report.help_period)
    if isinstance(tier, PosteriorVoteTier):
        probabilities = tier.lowest_probabilities(
            ticks_to_help=ticks_to_help, drift=drift
        )
        return REPORT_NEEDS[int(np.argmax(probabilities))]
    projected = tier.cloud() - ticks_to_help * drift[None, :]
    portions = (report.portion_small, report.portion_large)
    best_index, best_score = 0, -np.inf
    for index in range(len(REPORT_NEEDS)):
        score = 0.0
        for portion in portions:
            branch = projected.copy()
            branch[:, index] += portion
            score += 0.5 * float(np.min(branch, axis=-1).mean())
        if score > best_score:
            best_index, best_score = index, score
    return REPORT_NEEDS[best_index]


# -- the survey ---------------------------------------------------------------


def _make_tier(
    name: str,
    organism: OrganismConfig,
    report,
    statistics: ShockStatistics,
    *,
    particles: int,
    seed: int,
) -> _Tier:
    if name == "oracle":
        return OracleTier(statistics)
    if name == "naive":
        return NaiveTier(organism, report)
    if name == "mean_corrected":
        return MeanCorrectedTier(organism, report, statistics)
    if name == "posterior":
        return PosteriorTier(
            organism, report, statistics, particles=particles, seed=seed
        )
    if name == "posterior_vote":
        return PosteriorVoteTier(
            organism, report, statistics, particles=particles, seed=seed
        )
    raise ValueError(f"Unknown tier: {name}.")


def run_condition(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    tier_name: str,
    silence: float,
    lives: int,
    seed_base: int,
    particles: int,
) -> dict[str, object]:
    """One tier, one silence rate, `lives` held-out lives."""

    report = replace(organism.report, silent_shock_probability=silence)
    statistics = ShockStatistics(
        probability=report.shock_probability,
        size=report.shock_size,
        silent_probability=silence,
    )
    drift = np.array(
        [
            report.food_metabolism,
            report.water_metabolism,
            report.energy_metabolism,
        ]
    )

    survived = 0
    said = 0
    truthful = 0
    error_sum = 0.0
    error_ticks = 0
    argmin_hits = 0
    silent_shocks = 0
    collapses = 0
    steps_sum = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(
            organism, seed=seed, silent_shock_probability=silence
        )
        packet = world.reset(seed)
        tier = _make_tier(
            tier_name,
            organism,
            report,
            statistics,
            particles=particles,
            seed=seed + 17,
        )
        tier.reset(packet, world)
        motor_hidden = None
        motor_rng = Random(seed + 59_000_003)
        while True:
            need = choose_need(
                tier, step_count=packet.step_count, report=report, drift=drift
            )
            utterance = (TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID)
            truth = _true_body(world)
            if packet.step_count >= FIDELITY_WARMUP:
                said += 1
                truthful += int(need == world.lowest_need())
                believed = tier.point()
                error_sum += float(np.abs(believed - truth).mean())
                argmin_hits += int(int(np.argmin(believed)) == int(np.argmin(truth)))
                error_ticks += 1
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear(utterance)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            silent_shocks += int(bool(info.get("shock_silent")))
            steps_sum += int(info["duration"])
            if terminated or truncated:
                survived += int(not terminated)
                break
            tier.update(before, action, packet, world)
        if isinstance(tier, PosteriorTier):
            collapses += tier.collapses
    return {
        "tier": tier_name,
        "silent_shock_probability": silence,
        "lives": lives,
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
        "body_error": error_sum / max(1, error_ticks),
        "argmin_accuracy": argmin_hits / max(1, error_ticks),
        "mean_life_steps": steps_sum / lives,
        "silent_shocks_per_life": silent_shocks / lives,
        "posterior_collapses_per_life": collapses / lives,
    }


def run_open_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    silence: float,
    lives: int,
    seed_base: int,
    particles: int,
    driver: str = "naive",
) -> dict[str, object]:
    """Every tier scored on one shared history, so policy divergence cannot
    contaminate the estimate.

    In the closed loop each tier lives a different life: it says something
    different, so it is granted something different, so its body goes somewhere
    different. That makes closed-loop body error a statement about three
    trajectories rather than about three estimators. Here one tier drives and
    every tier watches the same transitions, which is the clean comparison of
    estimation quality.

    It also carries the endpoint that matters most for reflection and that no
    policy can flatten: whether the organism's own spread predicts its own
    error. That is a claim about the model of the model, and it is measurable
    with the planner switched off entirely.
    """

    report = replace(organism.report, silent_shock_probability=silence)
    statistics = ShockStatistics(
        probability=report.shock_probability,
        size=report.shock_size,
        silent_probability=silence,
    )
    drift = np.array(
        [
            report.food_metabolism,
            report.water_metabolism,
            report.energy_metabolism,
        ]
    )
    names = TIERS
    error_sum = {name: 0.0 for name in names}
    argmin_hits = {name: 0 for name in names}
    named_hits = {name: 0 for name in names}
    ticks = 0
    spreads: list[float] = []
    posterior_errors: list[float] = []
    covered = 0

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(
            organism, seed=seed, silent_shock_probability=silence
        )
        packet = world.reset(seed)
        tiers = {
            name: _make_tier(
                name, organism, report, statistics, particles=particles, seed=seed + 17
            )
            for name in names
        }
        for tier in tiers.values():
            tier.reset(packet, world)
        motor_hidden = None
        motor_rng = Random(seed + 59_000_003)
        while True:
            need = choose_need(
                tiers[driver],
                step_count=packet.step_count,
                report=report,
                drift=drift,
            )
            utterance = (TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID)
            truth = _true_body(world)
            if packet.step_count >= FIDELITY_WARMUP:
                ticks += 1
                lowest = world.lowest_need()
                for name, tier in tiers.items():
                    believed = tier.point()
                    error_sum[name] += float(np.abs(believed - truth).mean())
                    argmin_hits[name] += int(
                        int(np.argmin(believed)) == int(np.argmin(truth))
                    )
                    # What this tier would say, scored on the shared history.
                    named_hits[name] += int(
                        choose_need(
                            tier,
                            step_count=packet.step_count,
                            report=report,
                            drift=drift,
                        )
                        == lowest
                    )
                cloud = tiers["posterior"].cloud()
                spreads.append(float(cloud.std(axis=0).mean()))
                posterior_errors.append(
                    float(np.abs(cloud.mean(axis=0) - truth).mean())
                )
                lower = np.quantile(cloud, 0.05, axis=0)
                upper = np.quantile(cloud, 0.95, axis=0)
                # Per need, not jointly: a joint band over three needs would
                # read as under-coverage at exactly the nominal rate.
                covered += float(
                    np.mean((truth >= lower) & (truth <= upper))
                )
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear(utterance)
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
            for tier in tiers.values():
                tier.update(before, action, packet, world)

    scale = max(1, ticks)
    spread_array = np.asarray(spreads)
    error_array = np.asarray(posterior_errors)
    if spread_array.size > 2 and spread_array.std() > 0 and error_array.std() > 0:
        calibration = float(np.corrcoef(spread_array, error_array)[0, 1])
        order = np.argsort(spread_array)
        low = order[: order.size // 4]
        high = order[-(order.size // 4) :]
        low_error = float(error_array[low].mean())
        high_error = float(error_array[high].mean())
    else:
        calibration, low_error, high_error = 0.0, 0.0, 0.0
    return {
        "silent_shock_probability": silence,
        "driver": driver,
        "lives": lives,
        "scored_ticks": ticks,
        "body_error": {name: error_sum[name] / scale for name in names},
        "argmin_accuracy": {name: argmin_hits[name] / scale for name in names},
        "named_need_accuracy": {name: named_hits[name] / scale for name in names},
        "posterior_spread_error_correlation": calibration,
        "error_in_least_certain_quartile": high_error,
        "error_in_most_certain_quartile": low_error,
        "mean_spread": float(spread_array.mean()) if spread_array.size else 0.0,
        "credible_interval_coverage": covered / scale,
    }


def run_inspection_value(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    silence: float,
    lives: int,
    seed_base: int,
    particles: int,
    threshold: float,
) -> dict[str, object]:
    """Is the organism's own uncertainty worth acting on? A matched-budget test.

    The open-loop survey shows that a calibrated posterior cannot name the
    lowest need any better than a biased point filter: the oracle's advantage is
    information the silent shocks destroyed, and no estimator recovers it. What
    the posterior does have that no point filter has is knowing *when* it is
    wrong. That can only pay through an action that a certain organism would not
    take -- asking to be looked at rather than fed.

    This measures the information value of that before any ecology is built to
    carry it. On one fixed history the cloud may be reset to the truth on a
    limited number of ticks. Choosing those ticks by the organism's own spread
    is compared against choosing the *same number* of ticks at random. The
    rate-matched control is the whole experiment: an organism that inspects more
    often will look better for no reason, so only the timing may differ.
    """

    report = replace(organism.report, silent_shock_probability=silence)
    statistics = ShockStatistics(
        probability=report.shock_probability,
        size=report.shock_size,
        silent_probability=silence,
    )
    drift = np.array(
        [
            report.food_metabolism,
            report.water_metabolism,
            report.energy_metabolism,
        ]
    )
    policies = ("none", "uncertainty", "random", "oracle_error")
    hits = {name: 0 for name in policies}
    inspections = {name: 0 for name in policies}
    ticks = 0

    for life in range(lives):
        seed = seed_base + life
        # Replay the same fixed history for every policy, so the only thing that
        # differs is which ticks were spent looking.
        history = _record_history(
            model, organism, report, drift, seed=seed, statistics=statistics,
            particles=particles
        )
        if not history["transitions"]:
            continue
        budget = None
        chosen_by_error = None
        for policy in policies:
            result = _replay_with_inspections(
                organism,
                report,
                statistics,
                history,
                drift,
                particles=particles,
                seed=seed + 17,
                policy=policy,
                threshold=threshold,
                budget=budget,
                error_ranked=chosen_by_error,
            )
            if policy == "uncertainty":
                budget = result["inspections"]
                chosen_by_error = history["error_order"]
            hits[policy] += result["hits"]
            inspections[policy] += result["inspections"]
        ticks += history["scored"]

    scale = max(1, ticks)
    return {
        "silent_shock_probability": silence,
        "threshold": threshold,
        "lives": lives,
        "scored_ticks": ticks,
        "named_need_accuracy": {name: hits[name] / scale for name in policies},
        "inspections_per_life": {
            name: inspections[name] / max(1, lives) for name in policies
        },
        "uncertainty_over_random_points": 100.0
        * (hits["uncertainty"] - hits["random"]) / scale,
        "uncertainty_over_none_points": 100.0
        * (hits["uncertainty"] - hits["none"]) / scale,
        "oracle_error_over_random_points": 100.0
        * (hits["oracle_error"] - hits["random"]) / scale,
    }


def _record_history(
    model: OrganismModel,
    organism: OrganismConfig,
    report,
    drift: np.ndarray,
    *,
    seed: int,
    statistics: ShockStatistics,
    particles: int,
) -> dict[str, object]:
    """One life driven by the naive filter, stored for replay."""

    world = make_report_world(
        organism, seed=seed, silent_shock_probability=report.silent_shock_probability
    )
    packet = world.reset(seed)
    driver = NaiveTier(organism, report)
    driver.reset(packet, world)
    probe = PosteriorTier(
        organism, report, statistics, particles=particles, seed=seed + 17
    )
    probe.reset(packet, world)
    motor_hidden = None
    motor_rng = Random(seed + 59_000_003)
    transitions: list[tuple] = []
    truths: list[np.ndarray] = []
    lowest: list[str] = []
    scored_flags: list[bool] = []
    errors: list[float] = []
    while True:
        need = choose_need(
            driver, step_count=packet.step_count, report=report, drift=drift
        )
        utterance = (TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID)
        truth = _true_body(world)
        scored = packet.step_count >= FIDELITY_WARMUP
        truths.append(truth)
        lowest.append(world.lowest_need())
        scored_flags.append(scored)
        errors.append(float(np.abs(probe.point() - truth).mean()))
        action, motor_hidden = _sample_motor_action(
            model, packet, motor_hidden, motor_rng
        )
        world.hear(utterance)
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
        transitions.append((before, action, packet))
        driver.update(before, action, packet, world)
        probe.update(before, action, packet, world)
    scored_indices = [i for i, flag in enumerate(scored_flags) if flag]
    order = sorted(scored_indices, key=lambda i: errors[i], reverse=True)
    return {
        "birth": truths[0],
        "transitions": transitions,
        "truths": truths,
        "lowest": lowest,
        "scored_flags": scored_flags,
        "scored": len(scored_indices),
        "error_order": order,
        "steps": [t[0].step_count for t in transitions],
    }


def _replay_with_inspections(
    organism: OrganismConfig,
    report,
    statistics: ShockStatistics,
    history: dict[str, object],
    drift: np.ndarray,
    *,
    particles: int,
    seed: int,
    policy: str,
    threshold: float,
    budget: int | None,
    error_ranked: list[int] | None,
) -> dict[str, int]:
    tier = PosteriorVoteTier(
        organism, report, statistics, particles=particles, seed=seed
    )
    truths: list[np.ndarray] = history["truths"]  # type: ignore[assignment]
    lowest: list[str] = history["lowest"]  # type: ignore[assignment]
    flags: list[bool] = history["scored_flags"]  # type: ignore[assignment]
    transitions = history["transitions"]  # type: ignore[assignment]
    rng = Random(seed + 991)

    cloud = np.repeat(truths[0][None, :], particles, axis=0)
    tier._filter._belief = cloud.copy()

    scored_indices = [i for i, flag in enumerate(flags) if flag]
    if policy == "random" and budget:
        picks = set(rng.sample(scored_indices, min(budget, len(scored_indices))))
    elif policy == "oracle_error" and budget and error_ranked is not None:
        picks = set(error_ranked[:budget])
    else:
        picks = set()

    hits = 0
    used = 0
    for index in range(len(truths)):
        if policy == "uncertainty":
            spread = float(tier.cloud().std(axis=0).mean())
            if flags[index] and spread > threshold:
                tier._filter._belief = np.repeat(
                    truths[index][None, :], particles, axis=0
                )
                used += 1
        elif index in picks:
            tier._filter._belief = np.repeat(
                truths[index][None, :], particles, axis=0
            )
            used += 1
        if flags[index]:
            step = transitions[index][0].step_count if index < len(transitions) else 0
            named = choose_need(
                tier, step_count=step, report=report, drift=drift
            )
            hits += int(named == lowest[index])
        if index < len(transitions):
            before, action, after = transitions[index]
            tier.update(before, action, after, None)  # type: ignore[arg-type]
    return {"hits": hits, "inspections": used}


def run_survey(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    particles: int,
    silences: tuple[float, ...] = SILENCE_SWEEP,
    tiers: tuple[str, ...] = TIERS,
    seed_base: int = CEILING_SEED_BASE,
    log: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for silence in silences:
        for tier_name in tiers:
            row = run_condition(
                model,
                organism,
                tier_name=tier_name,
                silence=silence,
                lives=lives,
                seed_base=seed_base,
                particles=particles,
            )
            rows.append(row)
            if log:
                print(
                    f"  q={silence:<5} {tier_name:<15}"
                    f" survival {row['survival']:.3f}"
                    f" fidelity {row['report_fidelity']:.3f}"
                    f" body err {row['body_error']:.4f}"
                    f" argmin {row['argmin_accuracy']:.3f}",
                    flush=True,
                )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """The three gaps the survey exists to measure, per silence rate."""

    by_silence: dict[float, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_silence.setdefault(float(row["silent_shock_probability"]), {})[
            str(row["tier"])
        ] = row
    gaps = []
    for silence in sorted(by_silence):
        tier_rows = by_silence[silence]
        if not all(name in tier_rows for name in TIERS):
            continue

        def survival(name: str) -> float:
            return float(tier_rows[name]["survival"])

        def error(name: str) -> float:
            return float(tier_rows[name]["body_error"])

        gaps.append(
            {
                "silent_shock_probability": silence,
                "survival": {name: survival(name) for name in TIERS},
                "body_error": {name: error(name) for name in TIERS},
                "rate_knowledge_points": 100.0
                * (survival("mean_corrected") - survival("naive")),
                "uncertainty_points": 100.0
                * (survival("posterior") - survival("mean_corrected")),
                "headroom_points": 100.0
                * (survival("oracle") - survival("posterior")),
                "oracle_over_naive_points": 100.0
                * (survival("oracle") - survival("naive")),
            }
        )
    return {"per_silence": gaps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lives", type=int, default=100)
    parser.add_argument("--particles", type=int, default=48)
    parser.add_argument("--silences", default=",".join(str(q) for q in SILENCE_SWEEP))
    parser.add_argument("--tiers", default=",".join(TIERS))
    parser.add_argument("--thresholds", default="0.06,0.10,0.14")
    parser.add_argument("--skip-closed-loop", action="store_true")
    args = parser.parse_args()

    parent, organism = load_organism_checkpoint(args.parent)
    silences = tuple(float(part) for part in args.silences.split(",") if part.strip())
    tiers = tuple(part.strip() for part in args.tiers.split(",") if part.strip())
    thresholds = tuple(
        float(part) for part in args.thresholds.split(",") if part.strip()
    )

    print("open loop: every tier on one shared history", flush=True)
    open_loop = []
    for silence in silences:
        row = run_open_loop(
            parent,
            organism,
            silence=silence,
            lives=args.lives,
            seed_base=CEILING_SEED_BASE,
            particles=args.particles,
        )
        open_loop.append(row)
        named = row["named_need_accuracy"]
        print(
            f"  q={silence:<5} named "
            + " ".join(f"{name} {named[name]:.3f}" for name in TIERS)
            + f" | calib r={row['posterior_spread_error_correlation']:+.3f}"
            f" coverage {row['credible_interval_coverage']:.3f}",
            flush=True,
        )

    print("inspection value: matched budget", flush=True)
    inspection = []
    for silence in silences:
        if silence <= 0.0:
            continue
        for threshold in thresholds:
            row = run_inspection_value(
                parent,
                organism,
                silence=silence,
                lives=args.lives,
                seed_base=CEILING_SEED_BASE,
                particles=args.particles,
                threshold=threshold,
            )
            inspection.append(row)
            print(
                f"  q={silence:<5} tau={threshold:<5}"
                f" budget {row['inspections_per_life']['uncertainty']:.1f}/life"
                f" | uncertainty {row['named_need_accuracy']['uncertainty']:.3f}"
                f" random {row['named_need_accuracy']['random']:.3f}"
                f" oracle-error {row['named_need_accuracy']['oracle_error']:.3f}"
                f" none {row['named_need_accuracy']['none']:.3f}"
                f" | over random {row['uncertainty_over_random_points']:+.1f}"
                f" oracle-error over random"
                f" {row['oracle_error_over_random_points']:+.1f}",
                flush=True,
            )

    rows: list[dict[str, object]] = []
    if not args.skip_closed_loop:
        print("closed loop: each tier lives its own life", flush=True)
        rows = run_survey(
            parent,
            organism,
            lives=args.lives,
            particles=args.particles,
            silences=silences,
            tiers=tiers,
        )
    payload = {
        "config": {
            "lives": args.lives,
            "particles": args.particles,
            "silences": silences,
            "tiers": tiers,
            "thresholds": thresholds,
            "parent": args.parent,
            "seed_base": CEILING_SEED_BASE,
        },
        "open_loop": open_loop,
        "inspection": inspection,
        "rows": rows,
        "gaps": summarize(rows) if rows else {},
    }
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ceiling_survey.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
