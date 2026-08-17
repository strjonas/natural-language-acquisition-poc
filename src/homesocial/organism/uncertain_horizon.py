"""Probe67, ceiling survey: is knowing how unsure you are worth anything at a horizon?

Probe62 closed self-uncertainty in this repository and did it properly -- it
measured the ceiling instead of assuming one, and wrote down the two facts that
closed it:

1. "``posterior_vote`` is the Bayes-optimal decision rule given the observable
   history, so no estimator using the same information can beat it. It **ties
   the biased point filter at every silence rate**."
2. "The bound on the state ... does the self-model's job for free. An
   uncertainty that is nearly stationary is an uncertainty barely worth timing.
   For self-uncertainty to pay rent the ecology must make it **bursty**."

That closure was scoped, in its own words, to "this ecology, under this bounded
homeostatic body". Probe65 built a different one, and it dissolves both facts for
a reason that is structural rather than lucky:

    Probe62's uncertainty was about **where the body is**. The quantity probe65
    made load-bearing is **what the body is**.

Take the two reasons in turn.

**The bound does not reach the rate.** The homeostatic clip confines the body to
[0,1], which is what equilibrated probe62's state error at a flat +0.051 and
erased its burstiness. It has no such grip on the rate estimate: probe63's RLS
*skips* a reading whose truth is clipped (`self_calibration.py`, the
``BOUND_EPSILON`` guard), because a saturated residual says nothing about the
constants that produced it. So saturation removes evidence about the rate rather
than injecting it. The one mechanism that flattened probe62's uncertainty makes
this one larger.

**And the tie was a tie at horizon one.** State error enters the projection
``level - rate * H`` once; rate error enters multiplied by ``H``. So does rate
*uncertainty*. Probe62 measured at ``H = 1``, where a rate posterior is worth a
1x correction to a quantity probe64 then measured at 0.0045 per tick -- far below
anything an endpoint could see. At probe65's operating lag it is worth 18x that.

There is also, finally, a burst. Uncertainty about what you are is maximal at
birth and decays as readings arrive at 0.03 per tick: probe65's ``recursive``
ends a life at rate error 0.092 against ``individual``'s handed 0.000, and
`docs/STATE.md` item 2 identifies that gap as exactly the part of each life spent
still finding out what it is. That is a non-stationary uncertainty with a large
early episode, which is what probe62 said the state could not produce.

## The mechanism, which has no free parameters

Probe60's repaired objective is ``E[min(next body)]``, where the expectation is
over the caregiver's portion draw. Probe65 evaluates it at the horizon the lag
imposes. This survey changes exactly one thing: the expectation is also taken
over the organism's **posterior on its own rate**.

    point rule        E_portion [ min_i ( level_i - rhat_i * H + arriving_i ) ]
    uncertain rule    E_rate E_portion [ min_i ( level_i - r_i * H + arriving_i ) ]

``min`` is concave, so by Jensen the second is never larger than the first --
uncertainty makes every word look worse. That alone is not self-knowledge, and
this survey is built so that it cannot be mistaken for it: a *uniform* pessimism
shifts all three scores together and changes no argmax. Only a rate posterior
that is **wider for the need the organism knows less about** reorders the words.
So the endpoint is the differential, and ``flat`` and ``shuffled`` are the arms
that say whether the differential is what is doing the work.

Nothing is fitted. The covariance is the one probe63's RLS has carried since
2026-08-03 and no probe has ever read (``RecursiveSelfCalibration._covariance``),
scaled by the innovation variance estimated from its own residuals -- the textbook
RLS noise estimate, an estimate rather than a knob. When every sigma is zero the
rule is bit-identical to probe65's, which is a test rather than a claim.

## What this survey is for

It is a feasibility survey, not a treatment. **No gate is preregistered here and
none is claimed.** Probe60 made it binding to check the oracle ceiling before
locking a gate, and probe62 is the precedent for what to do with the answer: if a
*perfectly calibrated* uncertainty buys nothing at the operating horizon, this
closes and gets written up as a negative, exactly as probe62 did.

Three questions, in the order that can kill the mechanism soonest:

    Q1  Is the covariance calibrated at all -- does the organism's own sigma
        track its own rate error? Belief-side, planner switched off. If not,
        nothing downstream matters.
    Q2  Is there headroom? ``oracle_now`` is handed the magnitude of its own
        rate error one tick ago, which no posterior could achieve, and is
        therefore a ceiling above any conceivable learned width. If *it* does not
        beat ``point``, the mechanism is closed for the same reason probe62's was,
        and no cleverer estimator could have rescued it.
    Q3  Does the horizon produce it? Lag 0 is carried throughout as the
        falsifier: an effect that is the same size with no horizon is not about
        the horizon, and the story is wrong. See ``CLOSED_LOOP_DELAYS`` for which
        lags this spends lives on and for the measurement that decided it.

## Which endpoint can see this, and which one cannot

Stated in advance because the smoke run made it obvious and it changes how the
tables must be read. Probe65's endpoints are *named-need accuracy* and *regret*,
both scored against ``true_need`` -- the rule applied to the true body and the
true rates, with no uncertainty anywhere in it. That target is **risk-neutral**.

The mechanism here is **risk-averse**: it hedges toward the need it knows least
about, because that is the need whose realised value has the widest spread. So
whenever hedging is the right move, this rule *disagrees with the risk-neutral
oracle on purpose*, and accuracy and regret both score that as an error. They
cannot distinguish a good hedge from a bad word.

Survival can, because death is absorbing and therefore prices the tail. A
risk-neutral rule and a risk-averse one differ in exactly one place -- what they
pay to avoid a bad draw -- and only an endpoint that is nonlinear in the outcome
charges for it. Probe65 found that survival tracks regret rather than accuracy;
the step past that is that a risk-averse rule can be *better* than a risk-neutral
one on survival while being worse on both of the endpoints probe65 used.

So the closed loop is the primary panel here, with the minimum body each life
reaches carried alongside it as the quantity hedging is supposed to raise.
Accuracy and regret are reported in full and are expected to be flat or slightly
negative; if they are *strongly* negative while survival is flat, that is a
mechanism that costs something and buys nothing, and this closes.

Arms, sharing one history, one policy, one motor path, one line of arithmetic:

    point           probe65's ``recursive``, unchanged. The baseline.
    uncertain       the same self-model, reading its own covariance.
    flat            every need given the *mean* of its own sigmas. Matched
                    pessimism with the differential removed.
    shuffled        the same sigmas, rolled onto the wrong needs. Same
                    magnitudes, same mean, destroyed assignment.
    oracle_now      sigma set to the magnitude of its rate error one tick ago --
                    recorded after that tick's word, so it never reads the
                    decision it is making. Unachievable even so, and deliberately
                    so: it is the ceiling Q2 turns on.
    oracle_rms      sigma set to the running RMS of that error, which is roughly
                    what a well-calibrated posterior would report. The achievable
                    ceiling.
    individual      true rates, no uncertainty. Probe65's ceiling.
    state_oracle    true body, species rates. Carried so this table can be read
                    against probe65's.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.future_request import (
    CONSEQUENTIAL_MARGIN,
    DELAY_SWEEP,
    INTEROCEPTION,
    OPERATING_DELAY,
    RequestLedger,
    answer_horizon,
    expected_portion,
    need_scores,
    true_need,
    true_rates,
)
from homesocial.organism.individual_self import _true_body
from homesocial.organism.model import OrganismModel
from homesocial.organism.portion_request import (
    MoveFraction,
    SelfModelTier,
    species_rate,
)
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.self_calibration import (
    BOUND_EPSILON,
    CALIBRATION_PARAMETERS,
    METABOLIC_PARAMETERS,
    RecursiveSelfCalibration,
)
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]

_METABOLIC_INDEX = [CALIBRATION_PARAMETERS.index(p) for p in METABOLIC_PARAMETERS]

# Disjoint from every band already spent. The highest previously reached is
# probe66's treatment at 1,060,000,000 with a 2,000,000 stride over five seeds
# and its lives on top. Probe64 made it binding that a survey never shares a
# band with the treatment it informs: a threshold read off a life a gate then
# scores is a threshold fitted to its own answer.
SURVEY_SEED_BASE = 1_100_000_000
SEED_STRIDE = 2_000_000
SEEDS = 5

TREATMENT_WORLD = "metabolic"

# Where this survey spends its compute, and why it is not the whole of probe65's
# ``DELAY_SWEEP``.
#
# ``scripts/diagnose_probe67.py`` measures the two quantities the mechanism turns
# on, on a seed band of its own: the spread of the per-word Jensen correction,
# and the gap between the top two words that the correction has to exceed before
# anything is said differently. Ten lives a lag, 0.03 interoception:
#
#     lag   scored ticks   sigma*H   median gap   p90 correction   words changed
#       0           3516   0.00093       0.0782         0.000000          0.0000
#       3           3523   0.00358       0.0787         0.000046          0.0000
#       6           3141   0.00364       0.1285         0.000071          0.0000
#       9           2950   0.00793       0.1029         0.000709          0.0000
#      12           2324   0.00697       0.1628         0.000587          0.0000
#      18           2204   0.02022       0.1258         0.004363          0.0440
#      24           1472   0.03147       0.0395         0.008716          0.2418
#      30           1101   0.01616       0.0294         0.001859          0.1244
#      36            730   0.05034       0.0057         0.014942          0.3233
#
# Three things, and the second was not predicted.
#
# **The correction scales with the horizon**, as the mechanism requires: from
# nothing to 0.0149. That is the story working.
#
# **The gap it must clear shrinks with the horizon too** -- 0.078 to 0.006. That
# was not anticipated and it is the larger effect. A long projection sends every
# need downward together, so the three words converge: at lag 36 the median
# decision is worth 0.006 of body, well under ``CONSEQUENTIAL_MARGIN``. So the
# mechanism fires more at long horizons partly because it is stronger and partly
# because the decisions have stopped mattering, and those two must not be
# conflated. It is why the consequential split is reported separately.
#
# **Below lag 18 it is structurally inert**: it changes no word at all, on any
# tick, on any of ten lives. Measuring survival there measures probe65 with extra
# arithmetic. That is a fact about the arithmetic rather than about whether
# uncertainty helps, which is what makes it legitimate to stop spending lives on
# it rather than a choice made after seeing an outcome.
#
# And past 24 the instrument fails for an unrelated reason: scored ticks fall
# from 3,516 to 730 as lives die younger, and the survivors are selected for
# having guessed their own rate well. Lag 36 is 73 scored ticks a life against
# lag 0's 350.
#
# What is kept: 0 as the falsifier, because an effect the same size with no
# horizon is not about the horizon; 18 because it is probe65's operating point
# and the honest place to report that this mechanism barely fires there; 24
# because it is where it fires while lives are still long enough to score; and 30
# to carry one lag past that, because `docs/STATE.md` item 1 independently names
# "lag 24 or beyond" as the next operating point -- written before this survey
# existed and for an unrelated reason.
CLOSED_LOOP_DELAYS = (0, 18, 24, 30)

ARM_NAMES = (
    "point",
    "uncertain",
    "flat",
    "shuffled",
    "oracle_now",
    "oracle_rms",
    "individual",
    "state_oracle",
)

# Quadrature resolution for ``expected_min``, in nodes per need per sigma-range.
# A numerical-accuracy choice and not a model parameter, and guarded as one: the
# test suite re-resolves the entire scoring rule at eight times this and checks
# both that the value moves by less than 1e-5 -- a thousandth of
# ``CONSEQUENTIAL_MARGIN``, the smallest difference this probe calls meaningful --
# and that the ``argmax`` the rule actually reads never moves at all.
#
# Measured convergence against a 8,193-node reference over sigmas up to 0.15,
# far wider than anything ``MAX_SIGMA_FRACTION`` permits: 6.7e-5 at 129 nodes,
# 1.7e-5 at 257, 4.8e-6 at 513, 1.1e-6 at 1,025. Clean second order, as the
# trapezoid rule requires. 513 costs 95 microseconds a call, which is affordable
# at this probe's scale and buys an error four hundred times below the effect it
# is trying to see.
QUADRATURE_NODES = 513
QUADRATURE_TAILS = 8.0

# Below this a sigma is treated as exactly zero, so an arm carrying no
# uncertainty reduces to probe65's rule bit-identically rather than nearly.
SIGMA_FLOOR = 1e-12

# A rate posterior wider than this is not information, it is an unpriced
# reciprocal in a covariance that has not seen enough data. Fixed at the
# species spread itself: ``metabolic_spread`` 0.60 means the true rate lies
# within +/-30% of the species rate, so a claimed sigma above 0.30 of the
# species rate is claiming less knowledge than an organism has at birth.
MAX_SIGMA_FRACTION = 0.30

# The organism has to have seen its own body at least twice before its residuals
# say anything about the size of its own errors. Before that its sigma is the
# prior width, which is a species-level fact of exactly the kind it is already
# given (it is handed the species rates themselves).
MIN_READINGS_FOR_SIGMA = 2


# -- expectation of a minimum over independent normals --------------------------


def _normal_upper(z: np.ndarray) -> np.ndarray:
    """``P(Z > z)`` for a standard normal, vectorized, no SciPy in this venv.

    Zelen & Severo (A&S 26.2.17), whose absolute error is below 7.5e-8 -- two
    orders of magnitude finer than the quadrature it feeds and four below any
    quantity this probe reports.
    """

    absolute = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * absolute)
    density = np.exp(-0.5 * absolute * absolute) / math.sqrt(2.0 * math.pi)
    poly = t * (
        0.319381530
        + t
        * (
            -0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
        )
    )
    upper = density * poly
    return np.where(z >= 0.0, upper, 1.0 - upper)


def expected_min(
    mu: np.ndarray, sigma: np.ndarray, *, cap: float = 1.0, nodes: int | None = None
) -> float:
    """``E[min_i min(cap, X_i)]`` for independent ``X_i ~ N(mu_i, sigma_i^2)``.

    Via the survivor identity ``E[Y] = L + int_L^U P(Y > t) dt``, which needs no
    sampling and is therefore deterministic -- a Monte Carlo estimate here would
    put a seed inside the decision rule and make an arm's word depend on its own
    draw rather than on its belief.

    ``P(min > t)`` is the product of the per-need upper tails because the RLS
    covariance is kept separately per need, so the three rate posteriors are
    independent by construction rather than by assumption.

    The cap enters exactly rather than approximately: capping each axis at one
    and then taking the minimum is the same as capping the minimum, so the
    integral simply stops at ``cap``.

    Two details keep this honest as a *fixed* rule rather than a tuned one.
    A need whose sigma is zero contributes a step to the integrand, and no
    fixed-resolution grid integrates a step accurately -- so those needs are not
    integrated at all. They can only lower the minimum to their own value, which
    is exactly what the cap already does, so they are folded into the ceiling and
    the quadrature runs over the uncertain needs alone. And the grid is laid out
    in units of each need's *own* sigma rather than uniformly across the range,
    so a probe that mixes a width of 1e-5 with one of 1e-2 resolves both. Without
    that the node count would silently become a parameter, which the test suite
    checks it is not.
    """

    # Read at call time rather than bound at definition time, so that the test
    # suite can re-resolve the whole rule and check that no word depends on it.
    nodes = QUADRATURE_NODES if nodes is None else int(nodes)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    uncertain = sigma > SIGMA_FLOOR
    ceiling = float(cap)
    if not np.all(uncertain):
        ceiling = min(ceiling, float(mu[~uncertain].min()))
    if not np.any(uncertain):
        return ceiling

    centre = mu[uncertain]
    width = sigma[uncertain]
    lower = float((centre - QUADRATURE_TAILS * width).min())
    if ceiling <= lower:
        return ceiling
    upper = min(ceiling, float((centre + QUADRATURE_TAILS * width).max()))

    steps = np.linspace(-QUADRATURE_TAILS, QUADRATURE_TAILS, nodes)
    grid = np.unique(
        np.clip(
            np.concatenate(
                [
                    (centre[:, None] + steps[None, :] * width[:, None]).ravel(),
                    np.asarray([lower, upper]),
                ]
            ),
            lower,
            upper,
        )
    )
    scaled = (grid[:, None] - centre[None, :]) / width[None, :]
    survivor = np.prod(_normal_upper(scaled), axis=1)
    return lower + float(np.trapezoid(survivor, grid))


def uncertain_need_scores(
    *,
    believed_levels: np.ndarray,
    believed_rates: np.ndarray,
    believed_rate_sigma: np.ndarray,
    believed_uptake: dict[str, float],
    arriving: np.ndarray,
    horizon: int,
    report,
) -> np.ndarray:
    """Probe65's ``need_scores`` with the expectation taken over the rate too.

    Every term is probe65's. The only new one is ``believed_rate_sigma``, which
    enters where the rate does and is therefore multiplied by the same horizon --
    which is the whole reason this can matter at lag 18 and cannot at lag 0.

    With every sigma zero this returns probe65's array bit-identically, and
    ``tests/test_uncertain_horizon.py`` asserts that against ``need_scores``
    rather than trusting the reading.
    """

    sigma = np.asarray(believed_rate_sigma, dtype=np.float64)
    if not np.any(sigma > SIGMA_FLOOR):
        return need_scores(
            believed_levels=believed_levels,
            believed_rates=believed_rates,
            believed_uptake=believed_uptake,
            arriving=arriving,
            horizon=horizon,
            report=report,
        )

    horizon = float(horizon)
    centre = (
        np.asarray(believed_levels, dtype=np.float64)
        - np.asarray(believed_rates, dtype=np.float64) * horizon
        + np.asarray(arriving, dtype=np.float64)
    )
    spread = sigma * horizon

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
            shifted = centre.copy()
            shifted[index] += portion * float(believed_uptake.get(need, 1.0))
            scores[index] += probability * expected_min(shifted, spread)
    return scores


# -- the uncertainty the model has been carrying all along ----------------------


class UncertainSelfCalibration(RecursiveSelfCalibration):
    """Probe63's RLS, additionally reporting the width of its own rate estimate.

    It adds no parameter and changes no update. ``_a`` and ``_covariance`` evolve
    exactly as they have since 2026-08-03; this subclass only accumulates the
    innovations that were already being computed and thrown away, so an arm built
    on it is bit-identical to ``recursive`` in everything except that it can
    answer a question ``recursive`` could not be asked.

    The rate is linear in ``a`` restricted to the metabolic parameters -- the
    believed burn over an interval is ``sum_k (1 + a_k) J_k`` -- so the believed
    metabolic scale is ``1 + w . a`` with ``w`` the normalized metabolic
    sensitivity row, and its variance is the exact quadratic form ``w' Cov(a) w``.
    No delta-method approximation is involved.

    ``Cov(a) = s^2 P``, where ``P`` is the RLS covariance and ``s^2`` the
    innovation variance. ``s^2`` is estimated online from the arm's own residuals
    by the standard normalization ``e^2 / (1 + phi' P phi)``, which is the
    textbook RLS noise estimate. It is an estimate, not a threshold, and it is
    computed from quantities the organism already has.
    """

    def __init__(self, organism: OrganismConfig, report) -> None:
        super().__init__(organism, report)
        self._noise_sum = np.zeros(len(REPORT_NEEDS))
        self._noise_count = np.zeros(len(REPORT_NEEDS))

    def reset(self, packet, world) -> None:
        super().reset(packet, world)
        self._noise_sum = np.zeros(len(REPORT_NEEDS))
        self._noise_count = np.zeros(len(REPORT_NEEDS))

    def update(self, before, action_index: int, after, world, reading) -> None:
        if reading is not None:
            truth = np.asarray(reading, dtype=np.float64)
            # Read before ``super()`` moves anything, so these are the true
            # innovations of the update that is about to happen.
            base = self._jacobian.belief
            matrix = self._jacobian.matrix()
            for index in range(len(REPORT_NEEDS)):
                if not BOUND_EPSILON < truth[index] < 1.0 - BOUND_EPSILON:
                    continue
                regressor = matrix[index]
                covariance = self._covariance[index]
                projected = covariance @ regressor
                normalizer = 1.0 + float(regressor @ projected)
                if normalizer <= 0.0:
                    continue
                innovation = float(truth[index] - base[index]) - float(
                    regressor @ self._a[index]
                )
                self._noise_sum[index] += innovation * innovation / normalizer
                self._noise_count[index] += 1.0
        super().update(before, action_index, after, world, reading)

    def rate_sigma(self, move_fraction: float) -> np.ndarray:
        """Standard deviation of this organism's belief about its own burn rate.

        Zero until it has seen itself twice, because before that the residuals
        carry no statement about how large its own errors are. The species spread
        would be the honest prior there and is deliberately *not* substituted:
        handing the organism the width of its own species' variation is a second
        gift of information, and the point of this survey is to find out what the
        covariance it already owns is worth.
        """

        matrix = self._jacobian.matrix()
        sigma = np.zeros(len(REPORT_NEEDS))
        for index, need in enumerate(REPORT_NEEDS):
            if self._noise_count[index] < MIN_READINGS_FOR_SIGMA:
                continue
            row = matrix[index, _METABOLIC_INDEX]
            total = float(row.sum())
            if abs(total) < 1e-12:
                continue
            weights = np.zeros(len(CALIBRATION_PARAMETERS))
            weights[_METABOLIC_INDEX] = row / total
            covariance = self._covariance[index]
            variance = float(weights @ covariance @ weights)
            if variance <= 0.0:
                continue
            noise = float(self._noise_sum[index] / self._noise_count[index])
            scale_sigma = math.sqrt(max(0.0, noise * variance))
            rate = species_rate(self._report, need, move_fraction)
            sigma[index] = min(
                rate * scale_sigma, rate * MAX_SIGMA_FRACTION
            )
        return sigma


class UncertainTier(SelfModelTier):
    """``SelfModelTier("recursive")`` with the covariance-reporting body swapped in.

    Constructed through the parent so that every other field -- the recovered
    scale, the recovered uptake, the clipping bounds -- is probe64's object
    exactly, and only ``_body`` differs.
    """

    def __init__(self, organism: OrganismConfig, report) -> None:
        super().__init__("recursive", organism, report)
        self._body = UncertainSelfCalibration(organism, report)

    def rate_sigma(self, move_fraction: float) -> np.ndarray:
        return self._body.rate_sigma(move_fraction)


# -- how each arm turns that width into a word ---------------------------------


class RateErrorTracker:
    """One self-model's rate error measured from the truth, in two forms.

    Both are handed for exactly the reason ``oracle`` is handed the true body:
    they are ceilings, and a ceiling has to be measured before a gate is locked.
    Both know the *size* of the error and never its sign -- an arm that knew the
    sign would simply correct it, and that arm already exists as ``individual``.

    ``current`` is the magnitude of the error as of the previous tick -- it is
    updated after the word is chosen, so no arm ever reads the error of the
    decision it is making. No posterior can achieve it even so: knowing how wrong
    you were one tick ago is strictly more than being calibrated, because the
    rate barely moves between ticks. It is carried because a survey's job is to find the most generous
    defensible ceiling -- if uncertainty is worth nothing even to an arm handed
    its own instantaneous error, the mechanism is closed and no learned width
    could have rescued it.

    ``rms`` is the running root-mean-square over the life so far, which *is*
    roughly what a well-calibrated posterior would report. It is the achievable
    ceiling, and it is systematically too wide late in life -- the early ticks
    when the organism knew nothing stay in the average forever. The gap between
    the two is therefore itself informative: it says how much of any prize needs
    a width that tracks the error rather than merely averaging it.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum = np.zeros(len(REPORT_NEEDS))
        self._count = 0
        self._current = np.zeros(len(REPORT_NEEDS))

    def observe(self, believed: np.ndarray, actual: np.ndarray) -> None:
        error = np.asarray(believed, dtype=np.float64) - np.asarray(
            actual, dtype=np.float64
        )
        self._current = np.abs(error)
        self._sum += error * error
        self._count += 1

    def current(self) -> np.ndarray:
        if self._count < MIN_READINGS_FOR_SIGMA:
            return np.zeros(len(REPORT_NEEDS))
        return self._current.copy()

    def rms(self) -> np.ndarray:
        if self._count < MIN_READINGS_FOR_SIGMA:
            return np.zeros(len(REPORT_NEEDS))
        return np.sqrt(self._sum / self._count)


class Arm:
    """One self-model, one uncertainty policy, one horizon.

    ``point`` and ``uncertain`` share a self-model object outright, so the only
    thing that can separate them is the sigma passed to the rule. ``flat`` and
    ``shuffled`` read the *same* sigma vector and rearrange it: ``flat`` replaces
    every entry with their mean, which keeps the average pessimism and removes
    all of the differential; ``shuffled`` rolls the vector by one, which keeps the
    magnitudes, the mean and the spread and destroys only which need each belongs
    to. If either scores what ``uncertain`` scores, the effect is not
    self-knowledge and this survey has closed the mechanism.
    """

    def __init__(
        self,
        name: str,
        tier: SelfModelTier,
        ledger: RequestLedger,
        report,
        *,
        delay: int,
        sigma_policy: str,
    ) -> None:
        self.name = name
        self.tier = tier
        self.ledger = ledger
        self.delay = int(delay)
        self.sigma_policy = sigma_policy
        self._report = report
        self.errors = RateErrorTracker()

    def horizon(self) -> int:
        return answer_horizon(self.delay)

    def believed_rates(self, move_fraction: float) -> np.ndarray:
        return np.asarray(
            [self.tier.believed_rate(need, move_fraction) for need in REPORT_NEEDS],
            dtype=np.float64,
        )

    def sigma(self, move_fraction: float) -> np.ndarray:
        if self.sigma_policy == "none":
            return np.zeros(len(REPORT_NEEDS))
        if self.sigma_policy == "oracle_now":
            return self.errors.current()
        if self.sigma_policy == "oracle_rms":
            return self.errors.rms()
        own = self.tier.rate_sigma(move_fraction)
        if self.sigma_policy == "own":
            return own
        if self.sigma_policy == "flat":
            return np.full(len(REPORT_NEEDS), float(own.mean()))
        if self.sigma_policy == "shuffled":
            return np.roll(own, 1)
        raise ValueError(f"Unknown sigma policy: {self.sigma_policy}.")

    def scores(self, now: int, move_fraction: float) -> np.ndarray:
        uptake = self.tier.believed_uptake()
        return uncertain_need_scores(
            believed_levels=self.tier.point(),
            believed_rates=self.believed_rates(move_fraction),
            believed_rate_sigma=self.sigma(move_fraction),
            believed_uptake=uptake,
            arriving=self.ledger.arriving(now, uptake, delay=self.delay),
            horizon=self.horizon(),
            report=self._report,
        )

    def named_need(self, now: int, move_fraction: float) -> str:
        return REPORT_NEEDS[int(np.argmax(self.scores(now, move_fraction)))]


# ``uncertain``, ``flat``, ``shuffled`` and both oracles all sit on the *same*
# self-model, so every difference between them is a difference in what they do
# with its width and never in what they believe the rate to be.
ARM_SPEC: dict[str, tuple[str, str]] = {
    "point": ("uncertain", "none"),
    "uncertain": ("uncertain", "own"),
    "flat": ("uncertain", "flat"),
    "shuffled": ("uncertain", "shuffled"),
    "oracle_now": ("uncertain", "oracle_now"),
    "oracle_rms": ("uncertain", "oracle_rms"),
    "individual": ("individual", "none"),
    "state_oracle": ("state_oracle", "none"),
}
TIER_NAMES = tuple(dict.fromkeys(tier for tier, _ in ARM_SPEC.values()))


def build_arms(organism: OrganismConfig, report, *, delay: int):
    tiers: dict[str, SelfModelTier] = {}
    for name in TIER_NAMES:
        tiers[name] = (
            UncertainTier(organism, report)
            if name == "uncertain"
            else SelfModelTier(name, organism, report)
        )
    ledger = RequestLedger(
        help_period=report.help_period,
        delay=delay,
        expected_portion=expected_portion(report),
    )
    arms = {
        name: Arm(
            name,
            tiers[tier],
            ledger,
            report,
            delay=delay,
            sigma_policy=policy,
        )
        for name, (tier, policy) in ARM_SPEC.items()
    }
    return tiers, arms, ledger


def _world_kwargs(delay: int, interoception: float) -> dict:
    return dict(
        metabolic_spread=0.60,
        uptake_spread=0.00,
        interoception_probability=interoception,
        help_delay=int(delay),
        caregiver_store=0.0,
    )


# -- Q1 and Q2: one shared history, every arm scored on it ----------------------


def run_open_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    delay: int,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
) -> dict[str, object]:
    """Every arm asked what it would say, on lives none of them steered.

    Probe65's driver, for probe65's measured reason: under a delay the species
    filter dies around tick 155, so a weak driver leaves five scored ticks per
    life and every number becomes an artefact of how fast the driver starved.
    The truth drives instead. It cannot advantage any arm -- it is not one of
    them, it is the rule applied to the world's own state -- and it produces a
    full-length life for all seven to be measured on.
    """

    kwargs = _world_kwargs(delay, interoception)
    report = replace(organism.report, **kwargs)

    hits = {arm: 0 for arm in ARM_NAMES}
    stakes_hits = {arm: 0 for arm in ARM_NAMES}
    regret = {arm: 0.0 for arm in ARM_NAMES}
    ticks = 0
    stakes_ticks = 0

    # Q1's raw material: the organism's own claimed width against its own actual
    # error, paired tick by tick, with the planner playing no part.
    sigmas: list[float] = []
    absolute_errors: list[float] = []
    covered = 0
    sigma_ticks = 0
    live_sigma_ticks = 0

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tiers, arms, ledger = build_arms(organism, report, delay=delay)
        for tier in tiers.values():
            tier.reset(packet, world)
        for arm in arms.values():
            arm.errors.reset()
        ledger.reset()
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        hidden = None
        rng = Random(seed + 59_000_003)

        while True:
            fraction = moves.fraction
            now = int(packet.step_count)
            target = true_need(world, report, fraction, delay, ledger, now)
            truth = true_rates(world, report, fraction)

            if packet.step_count >= FIDELITY_WARMUP:
                ticks += 1
                true_scores = need_scores(
                    believed_levels=_true_body(world),
                    believed_rates=truth,
                    believed_uptake=world.uptake_scale,
                    arriving=ledger.arriving(now, world.uptake_scale, delay=delay),
                    horizon=answer_horizon(delay),
                    report=report,
                )
                best = float(true_scores.max())
                ordered = np.sort(true_scores)
                margin = float(ordered[-1] - ordered[-2])
                consequential = margin >= CONSEQUENTIAL_MARGIN
                stakes_ticks += int(consequential)
                for name, arm in arms.items():
                    spoken = arm.named_need(now, fraction)
                    correct = int(spoken == target)
                    hits[name] += correct
                    stakes_hits[name] += correct if consequential else 0
                    regret[name] += best - float(
                        true_scores[REPORT_NEEDS.index(spoken)]
                    )

                believed = arms["uncertain"].believed_rates(fraction)
                claimed = arms["uncertain"].sigma(fraction)
                error = np.abs(believed - truth)
                for index in range(len(REPORT_NEEDS)):
                    sigma_ticks += 1
                    if claimed[index] > SIGMA_FLOOR:
                        live_sigma_ticks += 1
                        sigmas.append(float(claimed[index]))
                        absolute_errors.append(float(error[index]))
                        covered += int(error[index] <= 1.96 * claimed[index])

            # After the word, so neither oracle ever sees the tick it is
            # answering.
            for arm in arms.values():
                arm.errors.observe(arm.believed_rates(fraction), truth)

            ledger.record(now, target)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[target]], PAD_ID))
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
            moves.observe(before, action, packet)
            for tier in tiers.values():
                tier.update(before, action, packet, world, info.get("interoception"))

    denominator = max(1, ticks)
    return {
        "help_delay": delay,
        "horizon": answer_horizon(delay),
        "lives": lives,
        "seed_base": seed_base,
        "scored_ticks": ticks,
        "consequential_share": stakes_ticks / denominator,
        "need_accuracy": {a: hits[a] / denominator for a in ARM_NAMES},
        "consequential_accuracy": {
            a: stakes_hits[a] / max(1, stakes_ticks) for a in ARM_NAMES
        },
        "regret": {a: regret[a] / denominator for a in ARM_NAMES},
        "calibration": score_calibration(
            sigmas,
            absolute_errors,
            covered=covered,
            live=live_sigma_ticks,
            total=sigma_ticks,
        ),
    }


def score_calibration(
    sigmas: list[float],
    errors: list[float],
    *,
    covered: int,
    live: int,
    total: int,
) -> dict[str, object]:
    """Probe62's instruments, pointed at the rate instead of the state.

    Probe62 asked three things of its particle cloud and this asks the same
    three of the RLS covariance, so the two are directly comparable: does the
    claimed width cover the truth at its nominal rate, does it correlate with
    the error it is describing, and is the error in the least-certain quartile
    actually larger than in the most-certain one. That last ratio is the one
    that matters here, because a width that does not vary cannot reorder a word.
    """

    if not sigmas:
        return {"live_share": 0.0, "samples": 0}

    sigma = np.asarray(sigmas, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    order = np.argsort(sigma)
    quartile = max(1, len(sigma) // 4)
    least = error[order[-quartile:]].mean()
    most = error[order[:quartile]].mean()
    correlation = (
        float(np.corrcoef(sigma, error)[0, 1])
        if sigma.std() > 0 and error.std() > 0
        else 0.0
    )
    return {
        "samples": int(len(sigma)),
        "live_share": live / max(1, total),
        "mean_sigma": float(sigma.mean()),
        "mean_absolute_error": float(error.mean()),
        # A perfectly calibrated normal has mean |error| = sigma * sqrt(2/pi).
        "calibration_ratio": float(
            sigma.mean() * math.sqrt(2.0 / math.pi) / max(1e-12, error.mean())
        ),
        "coverage_95": covered / max(1, len(sigma)),
        "spread_error_correlation": correlation,
        "least_certain_quartile_error": float(least),
        "most_certain_quartile_error": float(most),
        # Floored at one ten-thousandth of the mean error, so a quartile that
        # happens to contain a near-zero error cannot report a ratio in the
        # thousands. Probe62 reported this as 2.4x; a number here is only
        # comparable with that one if it cannot be produced by a single tick.
        "quartile_ratio": float(
            least / max(float(error.mean()) * 1e-4, most, 1e-12)
        ),
    }


# -- Q3: each arm speaking for itself ------------------------------------------


def run_closed_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    arm_name: str,
    delay: int,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
) -> dict[str, object]:
    """One arm alone in the world, so a wrong word is a wrong word it lives with.

    Carried in the survey rather than left to the treatment because probe64's
    negative is the standing warning: a belief-side gain the world cannot feel is
    exactly what this repository has produced before. If survival here is flat at
    the operating lag for every arm including ``oracle_now``, that is worth
    knowing before a gate is written rather than after.
    """

    kwargs = _world_kwargs(delay, interoception)
    report = replace(organism.report, **kwargs)

    survived = 0
    steps = 0
    future_truthful = 0
    said = 0
    floors: list[float] = []
    margins: list[float] = []
    deaths: dict[str, int] = defaultdict(int)

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tiers, arms, ledger = build_arms(organism, report, delay=delay)
        for tier in tiers.values():
            tier.reset(packet, world)
        for arm in arms.values():
            arm.errors.reset()
        ledger.reset()
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        arm = arms[arm_name]
        hidden = None
        rng = Random(seed + 59_000_003)
        floor = 1.0

        while True:
            fraction = moves.fraction
            now = int(packet.step_count)
            # The quantity a hedge is supposed to raise: how close to zero the
            # worst axis of the true body ever came. A risk-averse rule that
            # works buys its survival here, by never letting any need get as low
            # as a risk-neutral one lets it get.
            floor = min(floor, float(np.min(_true_body(world))))
            spoken = arm.named_need(now, fraction)
            if packet.step_count >= FIDELITY_WARMUP:
                said += 1
                future_truthful += int(
                    spoken == true_need(world, report, fraction, delay, ledger, now)
                )
            arm.errors.observe(
                arm.believed_rates(fraction), true_rates(world, report, fraction)
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
            # Only the arm under test steered this life, so only its self-model
            # has any claim on the transitions.
            arm.tier.update(before, action, packet, world, info.get("interoception"))

        floors.append(floor)

    return {
        "arm": arm_name,
        "help_delay": delay,
        "lives": lives,
        "survival": survived / lives,
        "future_fidelity": future_truthful / max(1, said),
        "mean_life_steps": steps / lives,
        # Mean over lives of the lowest the worst axis ever reached. Zero is
        # death, so this is the margin the organism kept against it, and it is
        # the endpoint a risk-averse rule should move even where survival is
        # too coarse to resolve a change.
        "mean_floor": float(np.mean(floors)) if floors else 0.0,
        "floor_per_life": floors,
        "deaths_by_need": dict(deaths),
    }


# -- the survey ----------------------------------------------------------------


def _paired(rows: list[dict[str, object]], key: str, left: str, right: str):
    """Per-seed differences and their 95% interval, matched on seed and lag.

    Paired because the arms share a history: an unpaired interval over five
    seeds would be dominated by the seed-to-seed variation in the lives
    themselves, which is exactly the variance the shared history removes.
    """

    deltas = [
        float(row[key][left]) - float(row[key][right])  # type: ignore[index]
        for row in rows
    ]
    array = np.asarray(deltas, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "low": mean, "high": mean, "per_seed": deltas}
    error = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return {
        "mean": mean,
        "low": mean - error,
        "high": mean + error,
        "per_seed": deltas,
        "sign_consistent": bool(np.all(array > 0) or np.all(array < 0)),
    }


def survey(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    closed_loop_lives: int,
    seed_base: int,
    seeds: int,
    delays: tuple[int, ...] = CLOSED_LOOP_DELAYS,
    interoception: float = INTEROCEPTION,
) -> dict[str, object]:
    open_loop: list[dict[str, object]] = []
    for seed in range(seeds):
        base = seed_base + seed * SEED_STRIDE
        for delay in delays:
            row = run_open_loop(
                model,
                organism,
                delay=delay,
                lives=lives,
                seed_base=base,
                interoception=interoception,
            )
            row["seed"] = seed
            open_loop.append(row)
            print(
                f"  lag {delay:>2} seed {seed}  "
                + "  ".join(
                    f"{a}={row['need_accuracy'][a]:.4f}"  # type: ignore[index]
                    for a in ("point", "uncertain", "oracle_now")
                ),
                flush=True,
            )

    contrasts: dict[str, object] = {}
    for delay in delays:
        rows = [r for r in open_loop if r["help_delay"] == delay]
        contrasts[str(delay)] = {
            "accuracy": {
                "oracle_now_over_point": _paired(
                    rows, "need_accuracy", "oracle_now", "point"
                ),
                "oracle_rms_over_point": _paired(
                    rows, "need_accuracy", "oracle_rms", "point"
                ),
                "uncertain_over_point": _paired(
                    rows, "need_accuracy", "uncertain", "point"
                ),
                "uncertain_over_flat": _paired(
                    rows, "need_accuracy", "uncertain", "flat"
                ),
                "uncertain_over_shuffled": _paired(
                    rows, "need_accuracy", "uncertain", "shuffled"
                ),
            },
            "regret": {
                "point_over_oracle_now": _paired(
                    rows, "regret", "point", "oracle_now"
                ),
                "point_over_uncertain": _paired(rows, "regret", "point", "uncertain"),
                "shuffled_over_uncertain": _paired(
                    rows, "regret", "shuffled", "uncertain"
                ),
            },
        }

    closed_loop: list[dict[str, object]] = []
    if closed_loop_lives > 0:
        for seed in range(seeds):
            base = seed_base + seed * SEED_STRIDE
            for delay in delays:
                for arm_name in ARM_NAMES:
                    row = run_closed_loop(
                        model,
                        organism,
                        arm_name=arm_name,
                        delay=delay,
                        lives=closed_loop_lives,
                        seed_base=base,
                        interoception=interoception,
                    )
                    row["seed"] = seed
                    closed_loop.append(row)
                print(
                    f"  closed lag {delay:>2} seed {seed}  "
                    + "  ".join(
                        f"{r['arm']}={r['survival']:.3f}"
                        for r in closed_loop
                        if r["seed"] == seed and r["help_delay"] == delay
                    ),
                    flush=True,
                )

    return {
        "probe": 67,
        "phase": "ceiling_survey",
        "world": TREATMENT_WORLD,
        "interoception_probability": interoception,
        "seed_base": seed_base,
        "seeds": seeds,
        "lives": lives,
        "closed_loop_lives": closed_loop_lives,
        "delay_sweep": list(delays),
        "arms": list(ARM_NAMES),
        "open_loop": open_loop,
        "contrasts": contrasts,
        "closed_loop": closed_loop,
    }


def _mean(rows: list[dict[str, object]], key: str, arm: str) -> float:
    return float(np.mean([float(r[key][arm]) for r in rows]))  # type: ignore[index]


def report_survey(result: dict[str, object]) -> str:
    rows = result["open_loop"]  # type: ignore[assignment]
    lines: list[str] = []

    lines.append("== Q1: is the covariance calibrated as a rate uncertainty ==")
    lines.append(
        "| lag | live share | mean sigma | mean |err| | ratio | cov95 | corr | q4/q1 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for delay in result["delay_sweep"]:  # type: ignore[index]
        cells = [r["calibration"] for r in rows if r["help_delay"] == delay]  # type: ignore[index]
        cells = [c for c in cells if c.get("samples")]
        if not cells:
            lines.append(f"| {delay} | 0.0000 | - | - | - | - | - | - |")
            continue

        def avg(key: str) -> float:
            return float(np.mean([float(c[key]) for c in cells]))

        lines.append(
            f"| {delay} | {avg('live_share'):.4f} | {avg('mean_sigma'):.5f} | "
            f"{avg('mean_absolute_error'):.5f} | {avg('calibration_ratio'):.3f} | "
            f"{avg('coverage_95'):.4f} | {avg('spread_error_correlation'):+.3f} | "
            f"{avg('quartile_ratio'):.3f} |"
        )

    lines.append("")
    lines.append("== Q2/Q3: named-need accuracy against the true future ==")
    lines.append("| lag | " + " | ".join(ARM_NAMES) + " |")
    lines.append("|---:|" + "---:|" * len(ARM_NAMES))
    for delay in result["delay_sweep"]:  # type: ignore[index]
        cells = [r for r in rows if r["help_delay"] == delay]  # type: ignore[index]
        lines.append(
            f"| {delay} | "
            + " | ".join(f"{_mean(cells, 'need_accuracy', a):.4f}" for a in ARM_NAMES)
            + " |"
        )

    lines.append("")
    lines.append("== regret, in body units ==")
    lines.append("| lag | " + " | ".join(ARM_NAMES) + " |")
    lines.append("|---:|" + "---:|" * len(ARM_NAMES))
    for delay in result["delay_sweep"]:  # type: ignore[index]
        cells = [r for r in rows if r["help_delay"] == delay]  # type: ignore[index]
        lines.append(
            f"| {delay} | "
            + " | ".join(f"{_mean(cells, 'regret', a):.4f}" for a in ARM_NAMES)
            + " |"
        )

    lines.append("")
    lines.append("== the contrasts that decide whether to preregister ==")
    lines.append(
        "| lag | oracle_now - point | uncertain - point | uncertain - shuffled |"
    )
    lines.append("|---:|---:|---:|---:|")
    for delay in result["delay_sweep"]:  # type: ignore[index]
        block = result["contrasts"][str(delay)]["accuracy"]  # type: ignore[index]

        def cell(key: str) -> str:
            entry = block[key]
            return f"{entry['mean']:+.4f} [{entry['low']:+.4f}, {entry['high']:+.4f}]"

        lines.append(
            f"| {delay} | {cell('oracle_now_over_point')} | "
            f"{cell('uncertain_over_point')} | {cell('uncertain_over_shuffled')} |"
        )

    closed = result["closed_loop"]  # type: ignore[assignment]
    if closed:
        for label, key in (
            ("survival, each arm speaking for itself -- the primary panel", "survival"),
            ("mean floor: how close the worst axis ever came to zero", "mean_floor"),
            ("mean life steps", "mean_life_steps"),
        ):
            lines.append("")
            lines.append(f"== {label} ==")
            lines.append("| lag | " + " | ".join(ARM_NAMES) + " |")
            lines.append("|---:|" + "---:|" * len(ARM_NAMES))
            for delay in sorted({int(r["help_delay"]) for r in closed}):
                lines.append(
                    f"| {delay} | "
                    + " | ".join(
                        "{:.4f}".format(
                            float(
                                np.mean(
                                    [
                                        r[key]
                                        for r in closed
                                        if r["arm"] == a and r["help_delay"] == delay
                                    ]
                                )
                            )
                        )
                        for a in ARM_NAMES
                    )
                    + " |"
                )

        lines.append("")
        lines.append("== closed-loop contrasts, paired per seed ==")
        lines.append("| lag | endpoint | uncertain - point | oracle_now - point |")
        lines.append("|---:|---|---:|---:|")
        for delay in sorted({int(r["help_delay"]) for r in closed}):
            for key in ("survival", "mean_floor"):

                def gap(left: str, right: str) -> str:
                    deltas = []
                    for seed in sorted({int(r["seed"]) for r in closed}):
                        picked = {
                            r["arm"]: float(r[key])
                            for r in closed
                            if r["seed"] == seed and r["help_delay"] == delay
                        }
                        deltas.append(picked[left] - picked[right])
                    array = np.asarray(deltas, dtype=np.float64)
                    mean = float(array.mean())
                    if len(array) < 2:
                        return f"{mean:+.4f}"
                    error = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
                    return f"{mean:+.4f} [{mean - error:+.4f}, {mean + error:+.4f}]"

                lines.append(
                    f"| {delay} | {key} | {gap('uncertain', 'point')} | "
                    f"{gap('oracle_now', 'point')} |"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        # Probe65's parent, unchanged. Every arm here shares one motor policy and
        # one lexicon, so a different parent would make this table incomparable
        # with the one it is built to be read against.
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--lives", type=int, default=40)
    parser.add_argument("--closed-loop-lives", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=SURVEY_SEED_BASE)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument(
        "--delays",
        default=",".join(str(d) for d in CLOSED_LOOP_DELAYS),
        help="comma-separated lags; see CLOSED_LOOP_DELAYS for why these",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(Path(args.checkpoint))
    result = survey(
        model,
        organism,
        lives=args.lives,
        closed_loop_lives=args.closed_loop_lives,
        seed_base=args.seed_base,
        seeds=args.seeds,
        delays=tuple(int(d) for d in args.delays.split(",") if d.strip()),
    )
    print(report_survey(result))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
