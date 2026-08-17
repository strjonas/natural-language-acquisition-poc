"""Probe65: a request that has to be about a body the organism is not in yet.

Probe64 ended with a measured obstacle rather than a vague one. A self-model
95.3% correct about its own burn rate changed what the organism said, by a
margin perfect knowledge of its own *state* could not reach -- and bought no
survival, because *the consequence horizon was shorter than the correction
interval*. A need re-served every 18 ticks recovers before a mis-sized portion
can matter. Homeostasis absorbed exactly the quantity the self-model improved.
Probes 59, 60 and 62 died against the same wall from other directions.

`docs/STATE.md` names the fix and forbids the cheap route to it: make one wrong
request unrecoverable, and do not reach it by retuning ``help_period``. So this
probe changes neither the caregiver's supply nor its clock. It changes when the
caregiver *acts*:

    help_delay      the grant made now answers the last request heard at or
                    before ``help_delay`` ticks ago.

That single change converts every request from a statement about the present
into a prediction. The word said now is answered when the body is somewhere
else, and where it will be is

    level - rate * delay

which contains a term nothing in this repository could previously make
load-bearing: *this individual's rate*. And the two things an organism can know
about itself enter that expression completely differently.

    Knowing where you are is worth a constant. Knowing what you are is worth
    something that grows with the horizon.

State error enters once. Rate error enters multiplied by the delay. So there
must be a horizon at which an organism that reads its own body perfectly and
believes it is a typical member of its species is beaten by one that is unsure
where it is and knows what it is -- and the crossover is predictable in advance
from the ratio of the two errors, not fitted afterwards. That is the claim, and
the delay sweep is built to falsify it: at delay 0 the ordering must be the
other way round, or the mechanism is measuring something else.

Seven arms share one history, one policy, one motor path and one line of
arithmetic. They differ only in what they believe about themselves, and in one
case only in whether they think ahead at all:

    myopic          the true body, and the belief that help comes at once --
                    which every organism before this one correctly held.
    state_oracle    the same true body, looking ahead with species rates.
                    Perfect knowledge of where it is; none of what it is.
    recursive       probe63's RLS self-calibration, looking ahead with what it
                    worked out about itself.
    individual      its own true rates, and no readings at all.

``myopic`` and ``state_oracle`` are the *same object* asked with two horizons,
so nothing except the lookahead can separate them. It is not a straw arm: with
no lag it is the joint best of the seven.

The second half is the control that `md/archive/DIRECTION_2026-07-26.md` phase
C2 names as the one that kills templated narration: **reports must diverge when
the future diverges while the present is identical**. That is testable here
exactly rather than approximately. Collect the ticks at which two organisms'
true bodies occupy the same cell of a fine grid and their true futures differ.
A reporter whose inputs are the present body and the species constants is a
deterministic function of quantities that agree on such a pair -- so it *cannot*
diverge, and its divergence rate is near zero for a structural reason and not a
statistical one. A self-model can, because the fact that separates the pair is a
fact about the two bodies rather than about their states.

Nothing here is trained by reward. The self-model is probe63's, unchanged and
untouched; the motor policy is probe52's frozen parent; the rule is probe60's
repaired ``E[min(next body)]`` evaluated at the horizon the lag imposes, one
expression shared bit-identically by every arm. What differs between arms is only
what each of them believes it is, and how far ahead it is asked to look.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.individual_self import _true_body
from homesocial.organism.model import OrganismModel
from homesocial.organism.portion_request import (
    MoveFraction,
    SelfModelTier,
    species_rate,
)
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]

# Each arm is (which self-model, what it believes the caregiver's lag to be).
# ``None`` means it believes the truth. ``myopic`` believes zero, which is what
# this repository's organism believed before probe65 existed; it shares
# ``state_oracle``'s self-model exactly, so one integer is the whole difference
# between them.
ARMS: dict[str, tuple[str, int | None]] = {
    "population": ("population", None),
    "snap": ("snap", None),
    "myopic": ("state_oracle", 0),
    "state_oracle": ("state_oracle", None),
    "recursive": ("recursive", None),
    "individual": ("individual", None),
    "oracle": ("oracle", None),
}
ARM_NAMES = tuple(ARMS)
# The self-models that have to be instantiated. ``myopic`` reuses one.
TIER_NAMES = tuple(dict.fromkeys(tier for tier, _ in ARMS.values()))

# Disjoint from every band already in use. The highest previously spent is
# probe64's treatment at 970,000,000 with a 2,000,000 stride over five seeds,
# which reaches 978,000,040. Survey, construction and treatment are disjoint
# from one another for the reason probe64 made binding: a threshold read off a
# life a gate then scores is a threshold fitted to its own answer.
SURVEY_SEED_BASE = 1_000_000_000
CONSTRUCTION_SEED_BASE = 1_010_000_000
SEED_BASE = 1_020_000_000
SEED_STRIDE = 2_000_000
SEEDS = 5

INTEROCEPTION = 0.03
TREATMENT_WORLD = "metabolic"
WORLDS = {
    "metabolic": {"metabolic_spread": 0.60, "uptake_spread": 0.00},
    "both": {"metabolic_spread": 0.60, "uptake_spread": 0.60},
    "null": {"metabolic_spread": 0.00, "uptake_spread": 0.00},
}

# The horizon sweep. Zero is the ecology of every earlier probe and is carried
# as the falsifier: if the ordering at zero is not the reverse of the ordering
# at the operating point, the delay is not what produced the effect.
DELAY_SWEEP = (0, 3, 6, 9, 12, 18, 24)

# Two true bodies count as the same present when they fall in the same cell of
# a grid this fine, per axis. Chosen against the quantities it has to separate
# rather than against any outcome: ``snap``'s body error is 0.042 and one tick
# of the fastest metabolism is 0.035, so a cell of 0.01 is inside both. The
# fact that has to survive the match is a rate difference times the horizon,
# which at the operating point is several times larger.
PRESENT_CELL = 0.01

# A decision worth less than this to get right is a near-tie. Fixed a priori at
# roughly one tick of a typical need's metabolism -- food burns 0.008 and water
# 0.012 -- and never swept. Probe64's negative is the reason it exists: a
# belief-side gain concentrated on near-ties is a gain the world cannot feel,
# and the only way to know is to score the consequential ticks separately.
CONSEQUENTIAL_MARGIN = 0.01


# -- the rule every arm speaks by ----------------------------------------------


def answer_horizon(delay: int) -> int:
    """Ticks of body between saying a word and the grant that answers it.

    The caregiver acts at boundary ``T`` on what it heard at ``T - delay``, so
    a word that is answered at all is answered exactly ``delay`` ticks after it
    is said, and the body has moved that far by the time the help lands. There
    is nothing to average over: an utterance that is not the one at ``T - delay``
    is not acted on, so the horizon *conditional on mattering* is the lag
    itself. This is a property of the caregiver's public clock, exactly as
    ``service_interval`` is in probe64, and never a parameter of any organism.

    At zero lag it is one tick -- the word said just before a boundary is acted
    on at that boundary, one tick of metabolism later. That is the horizon this
    repository's organism has always had, which is what makes ``myopic`` that
    organism rather than a straw one: it was never blind to the future, it just
    only ever saw one tick of it.
    """

    return max(1, int(delay))


def choose_need(
    *,
    believed_levels: np.ndarray,
    believed_rates: np.ndarray,
    believed_uptake: dict[str, float],
    arriving: np.ndarray,
    horizon: int,
    report,
) -> str:
    """The word whose grant leaves the worst-off axis of this body highest.

    This is probe60's repaired objective, ``E[min(next body)]``, evaluated at
    the horizon the caregiver's lag actually imposes. Probe59 established why
    the obvious alternative is not usable: an ``argmin`` over believed levels
    caps survival near 0.50 whatever the belief is, so a behavioural gate scored
    on it measures the planner and not the self-model. Probe60 repaired it to
    0.904 and recorded that it may stay as a controlled instrument, which is
    what it is here -- one expression, shared bit-identically by every arm, with
    only the beliefs fed into it differing.

    Four quantities go in and three are beliefs about the self. ``levels`` is
    where this body is, which a reading can settle. ``rates`` is what this body
    does, which no reading can: two organisms in identical states burn at
    different speeds and are heading to different places. ``uptake`` is how much
    good a portion does this body. ``arriving`` is what it has already asked for
    and not yet been given.

    The projection is capped above at one, because a body holds no more than
    that and help aimed at a need already full is wasted. It is deliberately
    *not* clipped below: under a long lag a fast-burning axis projects past
    zero, and clipping would tie every doomed axis at zero and throw away the
    ordering that says which one is furthest gone.
    """

    return REPORT_NEEDS[
        int(
            np.argmax(
                need_scores(
                    believed_levels=believed_levels,
                    believed_rates=believed_rates,
                    believed_uptake=believed_uptake,
                    arriving=arriving,
                    horizon=horizon,
                    report=report,
                )
            )
        )
    ]


def need_scores(
    *,
    believed_levels: np.ndarray,
    believed_rates: np.ndarray,
    believed_uptake: dict[str, float],
    arriving: np.ndarray,
    horizon: int,
    report,
) -> np.ndarray:
    """``E[min(next body)]`` for each of the three words. See ``choose_need``."""

    base = np.minimum(
        1.0,
        np.asarray(believed_levels, dtype=np.float64)
        - np.asarray(believed_rates, dtype=np.float64) * float(horizon)
        + np.asarray(arriving, dtype=np.float64),
    )
    large = float(report.large_portion_probability)
    branches = (
        (float(report.portion_small), 1.0 - large),
        (float(report.portion_large), large),
    )
    scores = np.zeros(len(REPORT_NEEDS))
    for index, need in enumerate(REPORT_NEEDS):
        for portion, probability in branches:
            if probability <= 0.0:
                continue
            branch = base.copy()
            branch[index] = min(
                1.0, branch[index] + portion * float(believed_uptake.get(need, 1.0))
            )
            scores[index] += probability * float(branch.min())
    return scores


def decision_margin(scores: np.ndarray) -> float:
    """How much the best word is worth over the next best, in body units.

    A decision whose margin is a hundredth of the body is one where being wrong
    costs about a tick of metabolism, and a word that flips on such a tick tells
    you very little. Reporting accuracy separately above and below this line is
    what stops a belief-side result being carried by near-ties -- which probe64's
    negative makes the first thing to check, because a change the world cannot
    feel is exactly what it found.
    """

    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    return float(ordered[-1] - ordered[-2])


class RequestLedger:
    """What this organism has already asked for and has not yet been given.

    A caregiver that answers slowly puts every speaker in the position of a
    controller with dead time: ask for water, and water is still the emptiest
    thing you can see for three more grants, so you ask three more times and
    starve of food instead. The survey confirms this is not a subtlety -- with
    no ledger a *perfect* speaker dies at delay 12, identically to every other
    arm, and the endpoint measures the oscillation rather than the self-model.

    Everything the ledger reads is the organism's own: the words it said, the
    ticks it said them on, and the caregiver's public clock. A grant made at
    boundary ``T`` answers the last request made at or before ``T - delay``, so
    what is still on its way is recoverable exactly from its own memory. It is
    the one component here that is about the organism's own *actions* rather
    than its body, and it is shared bit-identically by every arm.
    """

    def __init__(self, *, help_period: int, delay: int, expected_portion: float) -> None:
        self._period = int(help_period)
        self._delay = int(delay)
        self._portion = float(expected_portion)
        self._said: list[tuple[int, str]] = []

    def reset(self) -> None:
        self._said = []

    def record(self, tick: int, need: str) -> None:
        self._said.append((int(tick), need))
        # Nothing older than one delay plus one help period can ever answer a
        # future grant, so the memory a speaker needs is bounded by the
        # caregiver's own lag.
        horizon = self._delay + self._period
        if len(self._said) > 4 * max(1, horizon):
            cutoff = int(tick) - horizon
            self._said = [entry for entry in self._said if entry[0] >= cutoff]

    def _request_at(self, deadline: int) -> str | None:
        chosen = None
        for said_tick, need in self._said:
            if said_tick > deadline:
                break
            chosen = need
        return chosen

    def counts(self, now: int, *, delay: int | None = None) -> tuple[int, ...]:
        """How many grants for each need land before the current word is answered.

        Belief-independent: it is a fact about what was said and about the
        caregiver's clock, and every arm in a shared history reads the same one.
        The divergence audit matches on it so that two lives being compared
        differ in what their bodies *are* and in nothing else.
        """

        lag = self._delay if delay is None else int(delay)
        tally = {need: 0 for need in REPORT_NEEDS}
        if lag > 0:
            first = (int(now) // self._period + 1) * self._period
            # Boundaries strictly inside ``(now, now + lag)``. The boundary at
            # ``now + lag`` is excluded because it is the one that answers the
            # word being chosen right now: counting it would let the organism
            # believe help is already on its way for the very need it is about
            # to ask about, and then ask for something else. That mistake is not
            # hypothetical -- with it in place a *perfect* speaker survives worse
            # than one that ignored the lag entirely.
            for boundary in range(first, int(now) + lag, self._period):
                need = self._request_at(boundary - lag)
                if need is not None:
                    tally[need] += 1
        return tuple(tally[need] for need in REPORT_NEEDS)

    def arriving(
        self, now: int, uptake: dict[str, float], *, delay: int | None = None
    ) -> np.ndarray:
        """Expected help landing before the word said now is answered, per need.

        The request being decided right now is deliberately excluded: it is the
        decision, not part of the situation the decision is made in. ``delay``
        overrides the caregiver's true lag so that an arm which does not believe
        in the lag reads its own history consistently with that mistake.
        """

        return np.asarray(
            [
                count * self._portion * float(uptake.get(need, 1.0))
                for count, need in zip(self.counts(now, delay=delay), REPORT_NEEDS)
            ],
            dtype=np.float64,
        )


def expected_portion(report) -> float:
    """What a granted portion is worth to a typical body, before absorption.

    The caregiver draws its own size when nothing asks for one, so this is its
    public mixture and not a fact about any organism. ``portion_requests`` is
    off throughout this probe: the size channel is probe64's and holding it
    fixed is what keeps this result about the need channel.
    """

    probability = float(report.large_portion_probability)
    return probability * float(report.portion_large) + (1.0 - probability) * float(
        report.portion_small
    )


def believed_rates(tier: SelfModelTier, move_fraction: float) -> np.ndarray:
    return np.asarray(
        [tier.believed_rate(need, move_fraction) for need in REPORT_NEEDS],
        dtype=np.float64,
    )


def true_rates(world, report, move_fraction: float) -> np.ndarray:
    scale = world.metabolic_scale
    return np.asarray(
        [
            species_rate(report, need, move_fraction) * scale[need]
            for need in REPORT_NEEDS
        ],
        dtype=np.float64,
    )


def true_need(
    world,
    report,
    move_fraction: float,
    believed_delay: int,
    ledger: RequestLedger,
    now: int,
) -> str:
    """What the rule returns when every one of its inputs is the truth.

    This is the target every arm is scored against, so ``oracle`` scores 1.000
    by construction and is carried as the anchor that says so. The ledger is the
    same object every arm reads -- what is on its way is a fact about the shared
    history, not about any belief -- and only the absorption it is valued at is
    this body's true one.
    """

    return choose_need(
        believed_levels=_true_body(world),
        believed_rates=true_rates(world, report, move_fraction),
        believed_uptake=world.uptake_scale,
        arriving=ledger.arriving(now, world.uptake_scale, delay=believed_delay),
        horizon=answer_horizon(believed_delay),
        report=report,
    )


class Arm:
    """One self-model asked at one horizon, and nothing else.

    ``believed_delay`` is what this arm thinks the caregiver's lag is. Every arm
    but ``myopic`` believes the truth; ``myopic`` believes zero, which makes it
    the organism this repository already had. Its self-model is the same object
    ``state_oracle`` uses, so the two are separated by one integer and nothing
    else.
    """

    def __init__(
        self,
        name: str,
        tier: SelfModelTier,
        believed_delay: int,
        ledger: RequestLedger,
        report,
    ) -> None:
        self.name = name
        self.tier = tier
        self.believed_delay = believed_delay
        self.ledger = ledger
        self._report = report

    def horizon(self) -> int:
        return answer_horizon(self.believed_delay)

    def named_need(self, now: int, move_fraction: float) -> str:
        uptake = self.tier.believed_uptake()
        return choose_need(
            believed_levels=self.tier.point(),
            believed_rates=believed_rates(self.tier, move_fraction),
            believed_uptake=uptake,
            # An organism that has not noticed the lag has no reason to think
            # anything is in flight either, so ``myopic`` reads its own ledger
            # at the lag it believes in. One mistaken belief, followed through.
            arriving=self.ledger.arriving(
                now, uptake, delay=self.believed_delay
            ),
            horizon=self.horizon(),
            report=self._report,
        )


def build_arms(
    organism: OrganismConfig, report, *, delay: int
) -> tuple[dict[str, SelfModelTier], dict[str, Arm], RequestLedger]:
    """Every arm over one set of self-models and one shared record of what was said.

    The ledger is shared because in an open-loop run only the driver speaks, so
    there is exactly one history of requests and every arm is asked what it
    would say given that history. In a closed-loop run the arm under test is the
    only speaker, so the shared ledger is its own.
    """

    tiers = {name: SelfModelTier(name, organism, report) for name in TIER_NAMES}
    ledger = RequestLedger(
        help_period=report.help_period,
        delay=delay,
        expected_portion=expected_portion(report),
    )
    arms = {
        name: Arm(
            name,
            tiers[tier],
            delay if believed is None else believed,
            ledger,
            report,
        )
        for name, (tier, believed) in ARMS.items()
    }
    return tiers, arms, ledger


# -- one condition -------------------------------------------------------------


def _world_kwargs(
    world_name: str, *, delay: int, interoception: float, store: float = 0.0
) -> dict:
    return dict(
        WORLDS[world_name],
        interoception_probability=interoception,
        help_delay=delay,
        caregiver_store=store,
    )


def _donor_bodies(organism: OrganismConfig, model, *, seed: int, kwargs):
    """One other life's body trajectory, for the shuffled-readings control."""

    world = make_report_world(organism, seed=seed, **kwargs)
    packet = world.reset(seed)
    hidden = None
    rng = Random(seed + 59_000_003)
    bodies = []
    while True:
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD["food"]], PAD_ID))
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            action,
            consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        bodies.append(_true_body(world))
        if terminated or truncated:
            break
    return bodies


def run_open_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    world_name: str,
    delay: int,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
    store: float = 0.0,
    shuffle_readings: bool = False,
    driver: str = "oracle",
    collect_pairs: bool = False,
    portion_scale: float = 1.0,
    help_period: int | None = None,
) -> dict[str, object]:
    """Every arm scored on one shared history.

    One arm drives the utterance and all of them watch the same transitions, so
    a difference between arms is a difference between self-models and never a
    difference between trajectories that diverged.

    Probes 63 and 64 drove with ``population`` on the principle that the weakest
    arm should never be scored on a history it steered. That principle breaks
    here for a reason the survey found rather than assumed: under a delay the
    species filter dies at around tick 155, so at delay 18 it leaves five scored
    ticks per life and every arm's accuracy is an artefact of how quickly the
    driver starved. ``oracle`` drives instead, which produces a full 400-tick
    life for every arm to be measured on. It cannot advantage itself -- it is
    scored against its own rule applied to the truth and is 1.000 whatever
    history it is given -- and it is the same instrument probe64 used when it
    pinned the need channel to keep a behavioural number about the size channel.
    """

    kwargs = _world_kwargs(
        world_name, delay=delay, interoception=interoception, store=store
    )
    # Probe68's lever, default off. The grant size is the *denominator* of the
    # decision -- how much a word is worth over the next-best word is
    # ``E_grant[min(grant * uptake, axis gap)]`` -- so scaling it moves the
    # granularity of the whole decision surface without touching any belief.
    # The branch is guarded rather than written as an unconditional multiply so
    # that at scale 1.0 this function's every value is the one probe65 produced;
    # ``test_probe68_scale_one_is_probe65`` asserts that rather than trusting it.
    if portion_scale != 1.0:
        if portion_scale <= 0.0:
            raise ValueError("portion_scale must be positive.")
        kwargs = dict(
            kwargs,
            portion_small=organism.report.portion_small * portion_scale,
            portion_large=organism.report.portion_large * portion_scale,
        )
    if help_period is not None:
        if int(help_period) <= 0:
            raise ValueError("help_period must be positive.")
        kwargs = dict(kwargs, help_period=int(help_period))
    report = replace(organism.report, **kwargs)

    hits = {arm: 0 for arm in ARM_NAMES}
    stakes_hits = {arm: 0 for arm in ARM_NAMES}
    regret = {arm: 0.0 for arm in ARM_NAMES}
    body_error = {arm: 0.0 for arm in ARM_NAMES}
    ticks = 0
    stakes_ticks = 0
    discriminating = 0
    future_moved = 0
    readings = 0
    scale_error = 0.0
    lives_scored = 0
    records: list[dict[str, object]] = []

    for life in range(lives):
        seed = seed_base + life
        donor = (
            _donor_bodies(organism, model, seed=seed + 777_777, kwargs=kwargs)
            if shuffle_readings
            else None
        )
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tiers, arms, ledger = build_arms(organism, report, delay=delay)
        for tier in tiers.values():
            tier.reset(packet, world)
        ledger.reset()
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        hidden = None
        rng = Random(seed + 59_000_003)
        reading_index = 0

        while True:
            fraction = moves.fraction
            now = int(packet.step_count)
            spoken = arms[driver].named_need(now, fraction)
            if packet.step_count >= FIDELITY_WARMUP:
                ticks += 1
                target = true_need(world, report, fraction, delay, ledger, now)
                # What the truth itself would say if it had not noticed the lag.
                # The gap between the two is how much of this task is actually
                # about the future rather than about reading a dial.
                present = true_need(world, report, fraction, 0, ledger, now)
                # How often the question is worth asking at all, from two
                # directions. The first is the share of ticks where believing
                # you are typical gives a different answer from the truth; the
                # second is the share where the future disagrees with the
                # present, which is what makes this a prediction rather than a
                # reading. No arm can beat ``population`` by more than the
                # first, and none can beat ``myopic`` by more than the second.
                discriminating += int(
                    arms["population"].named_need(now, fraction) != target
                )
                future_moved += int(present != target)
                truth = _true_body(world)
                # How much this decision is worth getting right, under the
                # truth. A tick whose best and second-best word are worth almost
                # the same is a tick where being wrong costs almost nothing, and
                # accuracy carried by such ticks would be exactly the kind of
                # improvement probe64 found the world could not feel.
                true_scores = need_scores(
                    believed_levels=truth,
                    believed_rates=true_rates(world, report, fraction),
                    believed_uptake=world.uptake_scale,
                    arriving=ledger.arriving(now, world.uptake_scale),
                    horizon=answer_horizon(delay),
                    report=report,
                )
                best_score = float(true_scores.max())
                consequential = decision_margin(true_scores) >= CONSEQUENTIAL_MARGIN
                stakes_ticks += int(consequential)
                named = {}
                for name, arm in arms.items():
                    chosen = arm.named_need(now, fraction)
                    named[name] = chosen
                    hits[name] += int(chosen == target)
                    stakes_hits[name] += int(consequential and chosen == target)
                    # What this word actually costs, in body units, against the
                    # best word available on this tick. Accuracy counts errors;
                    # regret weighs them. Probe64's negative is a warning that
                    # the two can come apart, and only the second is in the
                    # currency survival is denominated in.
                    regret[name] += best_score - float(
                        true_scores[REPORT_NEEDS.index(chosen)]
                    )
                    body_error[name] += float(
                        np.abs(arm.tier.point() - truth).mean()
                    )
                if collect_pairs:
                    records.append(
                        {
                            # Everything a reporter of the present could read:
                            # where the body is, and what is already on its way.
                            # Matching on both is what leaves the two bodies
                            # differing in what they *are* and in nothing else.
                            "cell": tuple(
                                int(np.floor(value / PRESENT_CELL)) for value in truth
                            ),
                            "inflight": ledger.counts(now),
                            "life": life,
                            "target": target,
                            "present": present,
                            "named": named,
                        }
                    )

            ledger.record(now, spoken)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[spoken]], PAD_ID))
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
            reading = info.get("interoception")
            if reading is not None:
                readings += 1
                if donor is not None:
                    # Matched in count and timing; only the body is another
                    # organism's.
                    reading = tuple(donor[reading_index % len(donor)])
                reading_index += 1
            moves.observe(before, action, packet)
            for tier in tiers.values():
                tier.update(before, action, packet, world, reading)

        recovered = tiers["recursive"].believed_scale()
        actual = world.metabolic_scale
        scale_error += float(
            np.mean([abs(recovered[n] - actual[n]) for n in REPORT_NEEDS])
        )
        lives_scored += 1

    denominator = max(1, ticks)
    result: dict[str, object] = {
        "world": world_name,
        "help_delay": delay,
        "portion_scale": float(portion_scale),
        "expected_grant": float(expected_portion(report)),
        "help_period": int(report.help_period),
        # Grant per tick. Probe68's compensated sweep holds this exactly fixed
        # while the quantum moves, so nothing it finds can be nutrition.
        "grant_rate": float(expected_portion(report)) / float(report.help_period),
        "caregiver_store": store,
        "shuffled_readings": shuffle_readings,
        "interoception_probability": interoception,
        "lives": lives,
        "seed_base": seed_base,
        "scored_ticks": ticks,
        "readings_per_life": readings / max(1, lives),
        "discriminating_share": discriminating / denominator,
        "future_moved_share": future_moved / denominator,
        "consequential_share": stakes_ticks / denominator,
        "need_accuracy": {arm: hits[arm] / denominator for arm in ARM_NAMES},
        # The same accuracy over only the ticks where getting it right is worth
        # at least one tick of metabolism. If the ordering survives here, the
        # gain is not made of near-ties.
        "consequential_accuracy": {
            arm: stakes_hits[arm] / max(1, stakes_ticks) for arm in ARM_NAMES
        },
        # Mean cost of the word actually said, against the best word available
        # on that tick, in body units. This is the endpoint in the currency the
        # world is denominated in, and the one that can be compared with
        # survival without a change of units.
        "regret": {arm: regret[arm] / denominator for arm in ARM_NAMES},
        "body_error": {arm: body_error[arm] / denominator for arm in ARM_NAMES},
        "rate_error": scale_error / max(1, lives_scored),
        "prior_rate_error": WORLDS[world_name]["metabolic_spread"] / 2.0,
    }
    if collect_pairs:
        result["divergence"] = score_divergence(records)
    return result


# -- identical present, divergent future ---------------------------------------


def score_divergence(records: list[dict[str, object]]) -> dict[str, object]:
    """Does the report follow the future when the present cannot tell them apart?

    Every pair below comes from two *different lives* whose true bodies fall in
    the same cell of a grid of side ``PRESENT_CELL``, whose present-lowest need
    is the same, and whose future-lowest need is not. On such a pair the present
    is identical in every respect the present can be read, and the only thing
    that differs is what the two bodies are.

    ``myopic`` is the arm this is scored against, and its zero is structural
    rather than statistical: with no lookahead its word is a function of the
    present body and the species constants alone, both of which agree across a
    matched pair, so it *returns the same word twice* and could not do otherwise.
    That zero is the control -- not a weak baseline that might have done better
    with more evidence, but a reporter that cannot answer this question at all.

    ``state_oracle`` is the same function plus what is already on its way, which
    is why the pairs are bucketed on the in-flight vector as well as on the body:
    without that, two lives with different outstanding requests would separate it
    for a reason that has nothing to do with what either body is.
    """

    buckets: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        buckets[(record["cell"], record["inflight"])].append(record)  # type: ignore[index,arg-type]

    pairs = 0
    diverged = {arm: 0 for arm in ARM_NAMES}
    correct = {arm: 0 for arm in ARM_NAMES}
    for group in buckets.values():
        if len(group) < 2:
            continue
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if left["life"] == right["life"]:
                    # Two ticks of one life share a body *and* a set of rates,
                    # so nothing about the self separates them.
                    continue
                if left["present"] != right["present"]:
                    continue
                if left["target"] == right["target"]:
                    continue
                pairs += 1
                for arm in ARM_NAMES:
                    said_left = left["named"][arm]  # type: ignore[index]
                    said_right = right["named"][arm]  # type: ignore[index]
                    diverged[arm] += int(said_left != said_right)
                    correct[arm] += int(
                        said_left == left["target"] and said_right == right["target"]
                    )
    denominator = max(1, pairs)
    return {
        "cell": PRESENT_CELL,
        "records": len(records),
        "pairs": pairs,
        "divergence": {arm: diverged[arm] / denominator for arm in ARM_NAMES},
        "correct_divergence": {arm: correct[arm] / denominator for arm in ARM_NAMES},
    }


# -- living the life its own words bought --------------------------------------


def run_closed_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    arm_name: str,
    world_name: str,
    delay: int,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
    store: float = 0.0,
    portion_scale: float = 1.0,
    help_period: int | None = None,
) -> dict[str, object]:
    """One arm speaking for itself, so a wrong word is a wrong word it lives with.

    This is where the delay bites. The need channel is the one probes 59 and 60
    showed to dominate survival here, and it is now the channel the self-model
    has to reach -- the exact inverse of probe64, which put the self-model on a
    channel survival could not hear. Nothing else about the caregiver moves: the
    same clock, the same supply, the same portions.
    """

    kwargs = _world_kwargs(
        world_name, delay=delay, interoception=interoception, store=store
    )
    # Probe68's lever, default off; see ``run_open_loop`` for why the branch is
    # guarded rather than an unconditional multiply.
    if portion_scale != 1.0:
        if portion_scale <= 0.0:
            raise ValueError("portion_scale must be positive.")
        kwargs = dict(
            kwargs,
            portion_small=organism.report.portion_small * portion_scale,
            portion_large=organism.report.portion_large * portion_scale,
        )
    if help_period is not None:
        if int(help_period) <= 0:
            raise ValueError("help_period must be positive.")
        kwargs = dict(kwargs, help_period=int(help_period))
    report = replace(organism.report, **kwargs)

    survived = 0
    truthful = 0
    future_truthful = 0
    said = 0
    steps = 0
    grants = 0
    silent = 0
    deaths: dict[str, int] = defaultdict(int)

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tiers, arms, ledger = build_arms(organism, report, delay=delay)
        for tier in tiers.values():
            tier.reset(packet, world)
        ledger.reset()
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        arm = arms[arm_name]
        hidden = None
        rng = Random(seed + 59_000_003)

        while True:
            fraction = moves.fraction
            now = int(packet.step_count)
            spoken = arm.named_need(now, fraction)
            if packet.step_count >= FIDELITY_WARMUP:
                said += 1
                truthful += int(spoken == world.lowest_need())
                future_truthful += int(
                    spoken == true_need(world, report, fraction, delay, ledger, now)
                )
            ledger.record(now, spoken)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[spoken]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            steps += int(info["duration"])
            if terminated or truncated:
                survived += int(not terminated)
                if terminated:
                    deaths[str(info.get("death_need"))] += 1
                break
            moves.observe(before, action, packet)
            # Only the arm under test is advanced: it is the only self-model
            # that had any effect on this life.
            arm.tier.update(before, action, packet, world, info.get("interoception"))

        counts = world.grant_counts
        grants += int(counts["total"]) - int(counts["none"])
        silent += int(counts["none"])

    return {
        "arm": arm_name,
        "world": world_name,
        "help_delay": delay,
        "caregiver_store": store,
        "lives": lives,
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
        "future_fidelity": future_truthful / max(1, said),
        "mean_life_steps": steps / lives,
        "grants": grants / lives,
        "silent_grants": silent / lives,
        "deaths_by_need": dict(deaths),
    }


# -- the survey ----------------------------------------------------------------


OPERATING_DELAY = 18

# Probe64's basket, reused here as a *second* survey axis rather than a new
# knob. A lag makes a request be about the future; a finite store makes a
# request that goes to the wrong need cost supply that never comes back. Probe64
# established the band and its operating point was 30.0; zero is the unlimited
# caregiver every probe before it faced.
STORE_SWEEP = (0.0, 30.0, 24.0, 18.0)


def survey(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    closed_loop_lives: int,
) -> dict[str, object]:
    """Is there a horizon at which what you are outweighs where you are?

    Three things have to be true at once at the operating point and none is
    guaranteed. The future has to *disagree with the present* often enough that
    looking ahead is a different task from reading a dial. A learned rate has to
    beat perfect state knowledge on the belief-side endpoint, or the mechanism
    closes. And a perfect speaker has to still be able to *survive* -- probe64's
    result warns that the failure mode of an unrecoverable world is a floor, in
    which every arm dies and the endpoint measures nothing. This finds the band
    before any gate is locked, at the operating point the gates will be scored
    at, on a seed band disjoint from the treatment.
    """

    horizon = []
    for delay in DELAY_SWEEP:
        row = run_open_loop(
            model,
            organism,
            world_name=TREATMENT_WORLD,
            delay=delay,
            lives=lives,
            seed_base=seed_base,
        )
        horizon.append(row)
        for key, label in (
            ("need_accuracy", "open  "),
            ("consequential_accuracy", "conseq"),
            ("regret", "regret"),
        ):
            accuracy = row[key]
            print(
                f"{label} delay {delay:>2}: "
                + " ".join(f"{arm[:4]} {accuracy[arm]:.4f}" for arm in ARM_NAMES)
                + f" | moved {row['future_moved_share']:.3f}"
                f" | stakes {row['consequential_share']:.3f}"
                f" | ticks {row['scored_ticks']}",
                flush=True,
            )
    null = run_open_loop(
        model,
        organism,
        world_name="null",
        delay=OPERATING_DELAY,
        lives=lives,
        seed_base=seed_base,
    )
    divergence = run_open_loop(
        model,
        organism,
        world_name=TREATMENT_WORLD,
        delay=OPERATING_DELAY,
        lives=lives,
        seed_base=seed_base,
        collect_pairs=True,
    )

    ration = []
    if closed_loop_lives:
        for delay in DELAY_SWEEP:
            row = [
                run_closed_loop(
                    model,
                    organism,
                    arm_name=arm,
                    world_name=TREATMENT_WORLD,
                    delay=delay,
                    lives=closed_loop_lives,
                    seed_base=seed_base + 500_000,
                )
                for arm in ARM_NAMES
            ]
            ration.extend(row)
            print(
                f"close delay {delay:>2}: "
                + " ".join(
                    f"{entry['arm'][:4]} {entry['survival']:.3f}" for entry in row
                )
                + " | steps "
                + " ".join(f"{entry['mean_life_steps']:.0f}" for entry in row),
                flush=True,
            )

    # Does a finite basket make the allocation matter? A lag makes a request be
    # about the future; a store makes a request aimed at the wrong need cost
    # supply that never comes back. Both are existing default-off levers and
    # neither is retuned here.
    store = []
    if closed_loop_lives:
        for amount in STORE_SWEEP:
            row = [
                run_closed_loop(
                    model,
                    organism,
                    arm_name=arm,
                    world_name=TREATMENT_WORLD,
                    delay=OPERATING_DELAY,
                    store=amount,
                    lives=closed_loop_lives,
                    seed_base=seed_base + 500_000,
                )
                for arm in ARM_NAMES
            ]
            store.extend(row)
            print(
                f"store {amount:>5.1f}: "
                + " ".join(
                    f"{entry['arm'][:4]} {entry['survival']:.3f}" for entry in row
                ),
                flush=True,
            )

    return {
        "horizon": horizon,
        "null": null,
        "divergence": divergence,
        "closed_loop": ration,
        "store": store,
        "arms": list(ARM_NAMES),
        "delay_sweep": list(DELAY_SWEEP),
        "store_sweep": list(STORE_SWEEP),
        "operating_delay": OPERATING_DELAY,
        "seed_base": seed_base,
    }


# -- the locked gates ----------------------------------------------------------


def _verdict(flags: list[bool], needed: int = 4) -> dict[str, object]:
    return {
        "per_seed": flags,
        "passing": sum(flags),
        "required": needed,
        "pass": sum(flags) >= needed,
    }


def score_gates(rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Exactly the gates in the preregistration, scored per seed.

    Nothing here reads a threshold from the data. Every number below is written
    in `docs/decisions/2026-08-16-future-request-preregistration.md` and was
    fixed against a ceiling survey run on a disjoint seed band, at this
    operating point, before any treatment seed existed.
    """

    def accuracy(condition: str, arm: str) -> list[float]:
        return [float(row["need_accuracy"][arm]) for row in rows[condition]]  # type: ignore[index]

    def survival(arm: str) -> list[float]:
        return [
            float(next(e for e in row["arms"] if e["arm"] == arm)["survival"])  # type: ignore[index]
            for row in rows["behaviour"]
        ]

    g1 = [
        oracle - myopic >= 0.20
        for oracle, myopic in zip(
            accuracy("treatment", "oracle"), accuracy("treatment", "myopic")
        )
    ]
    g2 = [
        learned - state >= 0.04
        for learned, state in zip(
            accuracy("treatment", "recursive"), accuracy("treatment", "state_oracle")
        )
    ]
    g3 = [
        individual - state >= 0.04
        for individual, state in zip(
            accuracy("treatment", "individual"), accuracy("treatment", "state_oracle")
        )
    ]
    # The falsifier. Whatever the self-model buys at the operating horizon, it
    # must *lose* at zero horizon -- otherwise the advantage was never about the
    # future and this probe is measuring a better estimator by another route.
    g4 = [
        state - learned >= 0.02 and state - individual >= 0.02
        for state, learned, individual in zip(
            accuracy("prompt", "state_oracle"),
            accuracy("prompt", "recursive"),
            accuracy("prompt", "individual"),
        )
    ]
    g5 = [
        abs(learned - species) <= 0.005
        for learned, species in zip(
            accuracy("null", "recursive"), accuracy("null", "population")
        )
    ]
    g6 = [
        learned - species <= 0.01
        for learned, species in zip(
            accuracy("shuffled", "recursive"), accuracy("shuffled", "population")
        )
    ]
    g7 = [
        float(row["divergence"]["correct_divergence"]["recursive"]) >= 0.60  # type: ignore[index]
        and float(row["divergence"]["divergence"]["myopic"]) <= 0.05  # type: ignore[index]
        for row in rows["divergence"]
    ]
    g8 = [
        state - myopic >= 0.15
        for state, myopic in zip(survival("state_oracle"), survival("myopic"))
    ]

    def gap(seed_row: dict[str, object], lag: int) -> float:
        entry = next(
            item for item in seed_row["delays"] if int(item["help_delay"]) == lag  # type: ignore[index]
        )
        return float(entry["need_accuracy"]["individual"]) - float(
            entry["need_accuracy"]["state_oracle"]
        )

    g9 = [
        gap(seed_row, 24) > gap(seed_row, 12) > gap(seed_row, 0)
        for seed_row in rows["horizon"]
    ]

    gates = {
        "G1_the_task_is_about_the_future": _verdict(g1),
        "G2_self_model_over_perfect_state": _verdict(g2),
        "G3_rate_headroom_exists": _verdict(g3),
        "G4_the_horizon_is_what_bought_it": _verdict(g4),
        "G5_no_false_discovery": _verdict(g5),
        "G6_shuffled_readings": _verdict(g6),
        "G7_identical_present_divergent_future": _verdict(g7),
        "G8_looking_ahead_is_load_bearing": _verdict(g8),
        "G9_the_advantage_grows_with_the_horizon": _verdict(g9),
    }
    gates["all_pass"] = all(gate["pass"] for gate in gates.values())  # type: ignore[index]
    return gates


def treatment(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    seeds: int,
    closed_loop_lives: int,
    delay: int = OPERATING_DELAY,
    store: float = 0.0,
    divergence_lives: int = 120,
) -> dict[str, object]:
    """Five seeds of everything the preregistration scores."""

    conditions = (
        "treatment",
        "prompt",
        "null",
        "both",
        "shuffled",
        "divergence",
        "behaviour",
        "horizon",
    )
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in conditions}

    for index in range(seeds):
        seed_base = SEED_BASE + SEED_STRIDE * index

        # The curve is run once and read twice. ``treatment`` is its entry at
        # the operating lag and ``prompt`` is its entry at zero -- the ecology
        # of every probe before this one, and where the ordering has to reverse.
        curve = []
        for each in DELAY_SWEEP:
            row = run_open_loop(
                model,
                organism,
                world_name=TREATMENT_WORLD,
                delay=each,
                lives=lives,
                seed_base=seed_base,
            )
            curve.append(row)
            accuracy = row["need_accuracy"]
            print(
                f"seed {index} lag {each:>2}: "
                + " ".join(f"{arm[:4]} {accuracy[arm]:.3f}" for arm in ARM_NAMES)
                + f" | moved {row['future_moved_share']:.3f}",
                flush=True,
            )
        rows["horizon"].append({"seed": index, "delays": curve})
        rows["treatment"].append(
            next(row for row in curve if int(row["help_delay"]) == delay)
        )
        rows["prompt"].append(
            next(row for row in curve if int(row["help_delay"]) == 0)
        )

        for name, world_name, shuffled in (
            ("null", "null", False),
            ("both", "both", False),
            ("shuffled", TREATMENT_WORLD, True),
        ):
            row = run_open_loop(
                model,
                organism,
                world_name=world_name,
                delay=delay,
                lives=lives,
                seed_base=seed_base,
                shuffle_readings=shuffled,
            )
            rows[name].append(row)
            accuracy = row["need_accuracy"]
            print(
                f"seed {index} {name:>10}: "
                + " ".join(f"{arm[:4]} {accuracy[arm]:.3f}" for arm in ARM_NAMES),
                flush=True,
            )

        # Matched pairs are rare -- 47 per 40 lives in the survey -- so the
        # divergence condition is run on three times the lives. It scores no
        # other endpoint, so the extra lives cannot affect anything else.
        divergence = run_open_loop(
            model,
            organism,
            world_name=TREATMENT_WORLD,
            delay=delay,
            lives=divergence_lives,
            seed_base=seed_base,
            collect_pairs=True,
        )
        rows["divergence"].append(divergence)
        scored = divergence["divergence"]
        print(
            f"seed {index} {'divergence':>10}: {scored['pairs']} pairs | "
            f"state {scored['divergence']['state_oracle']:.3f} "
            f"rls {scored['divergence']['recursive']:.3f} "
            f"(correct {scored['correct_divergence']['recursive']:.3f})",
            flush=True,
        )

        arms = [
            run_closed_loop(
                model,
                organism,
                arm_name=arm,
                world_name=TREATMENT_WORLD,
                delay=delay,
                store=store,
                lives=closed_loop_lives,
                seed_base=seed_base,
            )
            for arm in ARM_NAMES
        ]
        rows["behaviour"].append({"seed": index, "arms": arms})
        print(
            f"seed {index} {'behaviour':>10}: "
            + " ".join(f"{e['arm'][:4]} {e['survival']:.3f}" for e in arms),
            flush=True,
        )

    return {
        "rows": rows,
        "gates": score_gates(rows),
        "operating_delay": delay,
        "operating_store": store,
    }


def _format(rows: list[dict[str, object]], key: str, axis: str) -> str:
    lines = [
        "| " + axis + " | " + " | ".join(ARM_NAMES) + " |",
        "|---:|" + "---:|" * len(ARM_NAMES),
    ]
    for row in rows:
        values = row[key]
        assert isinstance(values, dict)
        lines.append(
            f"| {row[axis]} | "
            + " | ".join(f"{values[arm]:.4f}" for arm in ARM_NAMES)
            + " |"
        )
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
    parser.add_argument("--closed-loop-lives", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=SURVEY_SEED_BASE)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--delay", type=int, default=OPERATING_DELAY)
    parser.add_argument("--store", type=float, default=0.0)
    parser.add_argument("--divergence-lives", type=int, default=120)
    parser.add_argument(
        "--treatment",
        action="store_true",
        help="Run the preregistered five-seed treatment instead of the survey.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(args.checkpoint)

    if args.treatment:
        result = treatment(
            model,
            organism,
            lives=args.lives,
            seeds=args.seeds,
            closed_loop_lives=args.closed_loop_lives,
            delay=args.delay,
            store=args.store,
            divergence_lives=args.divergence_lives,
        )
        out = Path(args.out or "runs/organism/probe65_future_request/treatment.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        gates = result["gates"]
        print("\n== locked gates ==")
        for name, verdict in gates.items():
            if name == "all_pass":
                continue
            mark = "PASS" if verdict["pass"] else "FAIL"
            print(
                f"{mark}  {name}: {verdict['passing']}/"
                f"{len(verdict['per_seed'])} {verdict['per_seed']}"
            )
        print(f"all_pass = {gates['all_pass']}")
        print(f"\nWrote {out}")
        return

    result = survey(
        model,
        organism,
        lives=args.lives,
        seed_base=args.seed_base,
        closed_loop_lives=args.closed_loop_lives,
    )

    out = Path(args.out or "runs/organism/probe65_future_request/ceiling_survey.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))

    print("== named-need accuracy against the true future ==")
    print(_format(result["horizon"], "need_accuracy", "help_delay"))
    print("\n== how often the future disagrees with the present ==")
    print("| delay | future moved | discriminating |")
    print("|---:|---:|---:|")
    for row in result["horizon"]:
        print(
            f"| {row['help_delay']} | {row['future_moved_share']:.4f} "
            f"| {row['discriminating_share']:.4f} |"
        )
    print("\n== the crossover: what you are minus where you are ==")
    print("| delay | individual - state_oracle | recursive - state_oracle |")
    print("|---:|---:|---:|")
    for row in result["horizon"]:
        accuracy = row["need_accuracy"]
        print(
            f"| {row['help_delay']} "
            f"| {accuracy['individual'] - accuracy['state_oracle']:+.4f} "
            f"| {accuracy['recursive'] - accuracy['state_oracle']:+.4f} |"
        )
    divergence = result["divergence"]["divergence"]
    print(
        f"\n== identical present, divergent future: {divergence['pairs']} pairs "
        f"from {divergence['records']} ticks =="
    )
    print("| arm | diverges | diverges correctly |")
    print("|---|---:|---:|")
    for arm in ARM_NAMES:
        print(
            f"| {arm} | {divergence['divergence'][arm]:.4f} "
            f"| {divergence['correct_divergence'][arm]:.4f} |"
        )
    if result["closed_loop"]:
        print("\n== survival, each arm speaking for itself ==")
        print("| delay | " + " | ".join(ARM_NAMES) + " |")
        print("|---:|" + "---:|" * len(ARM_NAMES))
        by_delay: dict[int, dict[str, float]] = defaultdict(dict)
        for row in result["closed_loop"]:
            by_delay[int(row["help_delay"])][str(row["arm"])] = float(row["survival"])
        for delay in DELAY_SWEEP:
            print(
                f"| {delay} | "
                + " | ".join(f"{by_delay[delay].get(arm, 0.0):.3f}" for arm in ARM_NAMES)
                + " |"
            )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
