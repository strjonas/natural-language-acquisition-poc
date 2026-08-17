# Self-uncertainty at a horizon: a ceiling survey

Date: 2026-08-16. Status: **feasibility survey, not a treatment result.** No gate
was preregistered and none is claimed. Its purpose is to decide whether an
uncertainty mechanism is worth preregistering in probe65's ecology, after
probe62 decided it was not worth preregistering in its own.

Artifacts: `runs/organism/probe67_uncertain_horizon/ceiling_survey.json`,
`runs/organism/probe67_uncertain_horizon/diagnostic.json`.
Module: `src/homesocial/organism/uncertain_horizon.py`.
Diagnostic: `scripts/diagnose_probe67.py`. Guards:
`tests/test_uncertain_horizon.py`.

## Why this was reopened at all

`docs/STATE.md` lists probe62 as **closed**, and closed lines are not reopened
here without contrary evidence. The contrary evidence is probe62's own text.

Probe62 closed self-uncertainty for two measured reasons and scoped both, in its
own words, to "this ecology, under this bounded homeostatic body":

1. "``posterior_vote`` is the Bayes-optimal decision rule given the observable
   history ... It **ties the biased point filter at every silence rate**."
2. "The bound on the state ... does the self-model's job for free ... For
   self-uncertainty to pay rent the ecology must make it **bursty**."

Probe65 built a different ecology, and the difference is exactly the one those
two sentences turn on:

> Probe62's uncertainty was about **where the body is**. The quantity probe65
> made load-bearing is **what the body is**.

**The bound does not reach the rate.** The homeostatic clip confines the body to
[0,1], which is what equilibrated probe62's state error at a flat +0.051 and
erased its burstiness. It has no such grip on the rate estimate: probe63's RLS
*skips* a reading whose truth is clipped, because a saturated residual says
nothing about the constants that produced it. Saturation removes evidence about
the rate rather than injecting it, so the one mechanism that flattened probe62's
uncertainty makes this one larger.

**And the tie was a tie at horizon one.** State error enters `level - rate * H`
once; rate error, and rate *uncertainty*, enter multiplied by `H`. Probe62
measured at `H = 1`. Probe65's operating point is `H = 18`.

**And there is finally a burst.** Uncertainty about what you are is maximal at
birth and decays as readings arrive at 0.03 a tick. Probe65's `recursive` ends a
life at rate error 0.092 against `individual`'s handed 0.000, and `STATE.md`
item 2 identifies that gap as the part of each life spent still finding out what
it is.

This is a different mechanism answering a phase the ladder never ran, not a
retuning of probe62's. `md/archive/DIRECTION_2026-07-26.md` phase C1 has never
had a treatment; probe62 is the survey that declined to write one.

## The mechanism, which has no free parameters

Probe60's repaired objective is `E[min(next body)]`, the expectation taken over
the caregiver's portion draw. Probe65 evaluates it at the horizon the lag
imposes. This survey changes exactly one thing:

    point rule        E_portion [ min_i ( level_i - rhat_i * H + arriving_i ) ]
    uncertain rule    E_rate E_portion [ min_i ( level_i - r_i * H + arriving_i ) ]

Nothing is fitted. The rate posterior is the covariance probe63's RLS has
carried since 2026-08-03 and **no probe has ever read**, scaled by the innovation
variance estimated from its own residuals — the textbook RLS noise estimate, an
estimate rather than a knob. The rate is exactly linear in the RLS state
restricted to the metabolic parameters, so its variance is an exact quadratic
form and no delta-method approximation is involved.

`min` is concave, so by Jensen uncertainty can only lower a score. That alone is
not self-knowledge, and the survey is built so it cannot be mistaken for it: a
*uniform* pessimism lowers all three scores together and changes no `argmax`
(asserted in the test suite, not assumed). Only a posterior that is **wider for
the need the organism knows less about** can reorder the words.

With every sigma zero the rule is **bit-identical** to probe65's, asserted
against `need_scores` itself rather than checked by reading.

## Which endpoint can see this, and which one cannot

Recorded before the numbers, because it determines how every table below is
read.

Probe65's endpoints — named-need accuracy and regret — are both scored against
`true_need`, the rule applied to the true body and the true rates with no
uncertainty in it. That target is **risk-neutral**. This mechanism is
**risk-averse**: it hedges toward the need whose future is least predictable. So
whenever hedging is right, this rule disagrees with the risk-neutral oracle *on
purpose*, and accuracy and regret both score that as an error.

Survival can see it, because death is absorbing and therefore prices the tail.
So the closed loop is the primary panel, with the mean floor — how close the
worst axis ever came to zero — carried alongside as the quantity a hedge is
supposed to raise.

**The closing condition was fixed in advance**, in the module docstring, before
any treatment number existed: *accuracy and regret are expected to be flat or
slightly negative; if they are strongly negative while survival is flat, that is
a mechanism that costs something and buys nothing, and this closes.*

## The arms

Eight, sharing one history, one policy, one motor path and one line of
arithmetic. Four of them sit on the **same self-model object**, so every
difference between those four is a difference in what they do with its width and
never in what they believe the rate to be.

| arm | width it uses |
|---|---|
| `point` | none. Probe65's `recursive`, unchanged. |
| `uncertain` | its own RLS covariance. |
| `flat` | the mean of its own widths, on every need. Matched pessimism, differential removed. |
| `shuffled` | its own widths, rolled onto the wrong needs. Same magnitudes, same mean, destroyed assignment. |
| `oracle_now` | the magnitude of its rate error one tick ago. Unachievable, and deliberately so — the ceiling. |
| `oracle_rms` | the running RMS of that error. Roughly what a calibrated posterior would report. |
| `individual` | none, and handed its true rates. Probe65's ceiling. |
| `state_oracle` | none, true body, species rates. Carried so this reads against probe65's table. |

Seed band 1,100,000,000 with a 2,000,000 stride, disjoint from probe66's
treatment at 1,060,000,000 and from everything before it. The diagnostic runs on
1,180,000,000, disjoint again, because a number read off the lives a survey
scores is a number fitted to its own answer.

## The diagnostic: when can the posterior reach a word at all

The rule reads an `argmax`, so the posterior can only change what is said when
the spread of its per-word correction exceeds the gap between the top two words.
Both sides of that inequality are measurable, on their own seed band, ten lives a
lag.

**Provenance, stated exactly, because the lag set was chosen on this.** A first
version of this table, over four lives and the full `DELAY_SWEEP`, was run and
read **before any treatment number from this probe had been looked at** — the
first survey launch was killed without its output ever being read, precisely
because that table showed lags 0 through 12 to be structurally inert. The ten-life
version below, and the consequential split in the next section, were run
afterwards and overlap in time with the survey. Both agree with the four-life
version on every qualitative claim made here; the four-life numbers were
0.0000 flips at every lag through 12, 0.0120 at 18 and 0.2093 at 24.

Nothing here is a fitted threshold. `CONSEQUENTIAL_MARGIN` is probe65's, fixed a
priori at roughly one tick of a typical need's metabolism and never swept, and
the quantities compared against it are properties of the arithmetic rather than
of any outcome.

| lag | scored ticks | sigma*H | median gap | p90 correction | words changed |
|---:|---:|---:|---:|---:|---:|
| 0 | 3516 | 0.00093 | 0.0782 | 0.000000 | 0.0000 |
| 3 | 3523 | 0.00358 | 0.0787 | 0.000046 | 0.0000 |
| 6 | 3141 | 0.00364 | 0.1285 | 0.000071 | 0.0000 |
| 9 | 2950 | 0.00793 | 0.1029 | 0.000709 | 0.0000 |
| 12 | 2324 | 0.00697 | 0.1628 | 0.000587 | 0.0000 |
| 18 | 2204 | 0.02022 | 0.1258 | 0.004363 | **0.0440** |
| 24 | 1472 | 0.03147 | 0.0395 | 0.008716 | **0.2418** |
| 30 | 1101 | 0.01616 | 0.0294 | 0.001859 | **0.1244** |
| 36 | 730 | 0.05034 | 0.0057 | 0.014942 | **0.3233** |

Three findings, and the second was not predicted.

**The correction scales with the horizon.** From nothing to 0.0149. That is the
mechanism working as designed.

**The gap it must clear shrinks with the horizon too** — 0.078 to 0.006 — and
this is the larger effect. A long projection sends every need downward together,
so the three words converge, and by lag 36 the median decision is worth 0.006 of
body, well below `CONSEQUENTIAL_MARGIN`. **The mechanism therefore fires more at
long horizons partly because it is stronger and partly because the decisions
have stopped mattering**, and those must not be conflated. It is why the
consequential split is reported separately.

**Below lag 18 it is structurally inert.** It changes no word at all, on any
tick, on any of ten lives. Measuring survival there measures probe65 with extra
arithmetic — a fact about the arithmetic and not about whether uncertainty
helps, which is what makes it legitimate to stop spending lives there rather
than a choice made after seeing an outcome.

Past 24 the instrument fails for an unrelated reason: scored ticks fall from
3,516 to 730 as lives die younger, and the survivors are selected for having
guessed their own rate well. Lag 36 is 73 scored ticks a life against lag 0's
350.

The survey therefore spends its lives on lags **0, 18, 24, 30**: 0 as the
falsifier, 18 because it is probe65's operating point and the honest place to
report that the mechanism barely fires there, 24 because it is where it fires
while lives are still long enough to score, and 30 to carry one lag past that —
`STATE.md` item 1 having independently named "lag 24 or beyond" before this
survey existed and for an unrelated reason.

## The measurement that closes it

The flip rate above counts every tick. `CONSEQUENTIAL_MARGIN` — probe65's, fixed
a priori at roughly one tick of a typical need's metabolism and never swept —
splits those ticks into decisions worth getting right and near-ties. Ten lives a
lag, same seed band:

| lag | flips, all ticks | share consequential | **flips on consequential ticks** |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.9374 | **0.0000** |
| 18 | 0.0440 | 0.8153 | **0.0000** |
| 24 | 0.2418 | 0.5829 | **0.0012** |
| 30 | 0.1244 | 0.5495 | **0.0000** |

**The rate posterior never changes a word that matters.** Every one of the 24% of
flips at lag 24 falls on a near-tie: on the 58% of ticks where the word is worth
at least one tick of metabolism, the flip rate is zero to three decimal places at
every lag. Backing that out, the flip rate *on near-ties* at lag 24 is about 0.58
— so the correction is reliably the size of a tie and reliably not the size of a
decision.

That is probe62's shape exactly, and it is worth stating in probe62's own terms:
the failure is not that a learned width would be too weak. **A perfect one buys
nothing**, because it is not the right size to reach the question.

And the size is not fixable inside this ecology. Reaching a consequential
decision needs a projected width comparable to the median consequential gap,
about 0.1 of body. The widest this mechanism ever produces is 0.031 at lag 24.
Closing that needs roughly three times the horizon, and three times the horizon
is where the instrument itself fails: at lag 36 the scored ticks fall to 73 a
life against lag 0's 350, because almost everything is dead.

> Within probe65's ecology the organism's posterior over its own rate is
> **structurally too narrow to reach a decision that matters**, at every horizon
> the ecology survives.

## Q2 and Q3: the survey proper, five seeds, thirty lives a cell

Named-need accuracy against the true future, every arm scored on one shared
history that none of them steered. Paired per-seed 95% intervals.

| lag | `point` | `uncertain` | `oracle_now` | `uncertain` - `point` |
|---:|---:|---:|---:|---|
| 0 | 0.9367 | 0.9368 | 0.9368 | **+0.0000 [-0.0000, +0.0001]** |
| 18 | 0.9358 | 0.8684 | 0.8838 | -0.0674 [-0.0859, -0.0489] |
| 24 | 0.9128 | 0.8163 | 0.8371 | -0.0965 [-0.1251, -0.0680] |
| 30 | 0.8880 | 0.7816 | 0.8239 | -0.1064 [-0.1396, -0.0731] |

**Q3 passes as a falsifier and that is the only thing here that passes.** At lag
0 the contrast is +0.0000 with an interval that excludes anything to four
decimals, on all five seeds — with no horizon the mechanism is inert, so
everything below is produced by the horizon rather than by the arithmetic.

**Q2 fails, and it fails at the ceiling.** `oracle_now` is handed the magnitude of
its own rate error and is worse than the point rule at every horizon —
0.8838 against 0.9358 at lag 18. No posterior can do better than being told the
size of its own error, so no learned width could have rescued this. That is
probe62's finding transposed exactly: the failure is not that a learned
mechanism would be too weak.

The loss also grows monotonically with the horizon, which is the mechanism
working and costing: the further ahead it looks, the more near-ties it flips away
from the risk-neutral answer, and the diagnostic below says none of those flips
touch a decision worth making.

## What is *not* being closed here

The direction of the hedge — toward the need whose future is least predictable —
was not a design choice, and this matters for how far the closure reaches.

`E_rate E_portion [ min(...) ]` is the Bayes-optimal action under probe60's own
objective given the organism's own posterior. Integrating the posterior into the
objective is the unique correct treatment of it; there is no free parameter and
no alternative rule that is more optimal under the same objective and the same
belief. So this survey is not reporting that one heuristic among several failed.

It is reporting that **probe60's objective is insensitive to the width of the
rate belief at every horizon this ecology survives**. That is a statement about
the objective and the ecology, not about the estimator or about the way it was
used, which is what makes a negative here worth as much as it would be if a
learned width had simply been too weak.

## Why, mechanically

Two reasons, and the second is the one that generalizes.

**The decisions this ecology poses are an order of magnitude larger than the
belief's width can move them.** Recorded as measured rather than as reasoned,
because two earlier explanations of this closure were written down here, each
accounted for every number then in hand, and each was wrong — which is
`CLAUDE.md`'s fourth trap arriving on schedule. What settled it was conditioning
the measurement instead of arguing about it.

What can move an `argmax` is the *spread* of the per-word corrections; a
correction applied equally to all three moves nothing, which is what the `flat`
control exists to show. Measured both unconditionally and restricted to the ticks
where the decision is worth at least one tick of metabolism, twelve lives a lag:

| lag | sigma*H | p90 spread, all | p90 spread, consequential | median gap, consequential | flips there |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.0212 | 0.005636 | 0.005523 | **0.15817** | 0.0000 |
| 24 | 0.0315 | 0.008716 | 0.011421 | **0.16624** | 0.0012 |
| 30 | 0.0162 | 0.001859 | 0.001839 | **0.12789** | 0.0000 |
| 36 | 0.0503 | 0.014942 | 0.017012 | **0.12325** | 0.0000 |

The correction on consequential ticks is **not** smaller than elsewhere — at lag
24 and 36 it is larger. So the closure is not that the mechanism is suppressed
where decisions matter. It is that **the gap distribution is bimodal**: real
decisions cluster at 0.123 to 0.166 of body, near-ties sit at essentially zero,
and there is little in between. `CONSEQUENTIAL_MARGIN` at 0.0100 separates the
two populations rather than sitting inside either.

The largest correction this mechanism ever produces is 0.017. Against a median
real decision of 0.123 that is **seven to twenty-eight times too small**, at every
horizon, including the ones where the ecology has already collapsed:

| lag | scored ticks | p90 spread | flips, all | flips, consequential |
|---:|---:|---:|---:|---:|
| 36 | 730 | 0.014942 | 0.3233 | **0.0000** |
| 42 | 59 | 0.001148 | 0.0000 | **0.0000** |
| 48 | 59 | 0.001876 | 0.0000 | **0.0000** |

At 42 and 48 the question stops being answerable at all: 59 scored ticks over
twelve lives, about five a life, and `sigma * H` reads 0.00000 because the organism
dies before it has seen itself twice.

> The rate posterior is not the wrong shape and it is not badly calibrated. It is
> **roughly an order of magnitude too narrow** — seven to twenty-eight fold — to
> move a decision this ecology treats as worth making, and the horizon does not
> close the gap because it widens the near-ties at the same rate.

**Risk aversion pays for irreversibility, and this ecology has none of the right
kind.** Probe65 made *systematic* rate error deadly — `myopic` survives 0.0900
against `state_oracle`'s 0.3150 — because a bias compounds over a whole life. It
did not make any *single* wrong word unrecoverable: needs are re-served on the
caregiver's clock and the body is bounded, so symmetric noise in the word channel
averages out. A width is a tool for managing variance, and variance is exactly
what homeostasis already absorbs. This is `CLAUDE.md`'s third trap — the
environment doing the model's job — reappearing on a channel probe62 did not
test, and it is the same bound, doing it again.

The obvious lever for irreversibility is already closed: `STATE.md` records that
stacking probe64's `caregiver_store` onto probe65's lag "was swept and does
nothing: at 30 it never binds, at 24 it lowers every arm equally, at 18 nothing
survives."

## Q1: the posterior is real, and better calibrated than probe62's was

Belief-side, planner playing no part. The same three instruments probe62 pointed
at its particle cloud, pointed here at the RLS covariance so the two are directly
comparable. Five seeds, thirty lives a cell.

| lag | live share | mean sigma | mean abs error | sigma/error | coverage 95 | corr(sigma, error) | least/most certain quartile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8404 | 0.00082 | 0.00072 | 0.997 | 0.8228 | +0.353 | **5.53** |
| 18 | 0.8075 | 0.00083 | 0.00068 | 1.182 | 0.8233 | +0.214 | **2.98** |
| 24 | 0.8099 | 0.00072 | 0.00055 | 1.369 | 0.8517 | +0.257 | **6.48** |
| 30 | 0.8110 | 0.00071 | 0.00037 | 1.907 | 0.9075 | +0.288 | **3.88** |

**Q1 passes.** The width is live on about 81% of ticks, its ratio to the realised
error is near one and drifts only mildly wide at long lags, its correlation with
that error is positive at every lag, and the error in its least-certain quartile
is **three to six and a half times** the error in its most-certain. Probe62
reported **2.4x** on that last instrument for the *state* posterior and called it
"second-order self-knowledge ... the organism does know when it does not know".

So this organism knows when it does not know **what it is**, on the same
instrument and by a wider margin than it knew when it did not know **where it
is**. Coverage runs under nominal — 0.82 to 0.91 against 0.95 — which is the
expected signature of an RLS covariance under a forgetting factor and is
recorded rather than corrected, since no gate rests on it.

This matters for how the negative below is read. The mechanism does not fail
because the belief is bad.

## The primary panel: each arm speaking for itself

8 arms x 4 lags x 5 seeds x 80 lives = 12,800 lives. 400 lives an arm a lag.

| lag | `point` | `uncertain` | `flat` | `shuffled` | `oracle_now` | `oracle_rms` | `individual` | `state_oracle` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6400 | 0.6400 | 0.6400 | 0.6400 | 0.6400 | 0.6400 | 0.6275 | 0.6550 |
| 18 | 0.3250 | 0.3250 | 0.3250 | 0.3225 | 0.3300 | 0.3350 | 0.3225 | 0.3175 |
| 24 | 0.2100 | 0.2100 | 0.2100 | 0.2125 | 0.2100 | 0.2125 | **0.2375** | 0.1950 |
| 30 | 0.1175 | 0.1200 | 0.1175 | 0.1175 | 0.1150 | 0.1125 | **0.1350** | 0.1200 |

Mean floor — how close the worst axis of the true body ever came to zero, the
quantity a hedge is supposed to raise:

| lag | `point` | `uncertain` | `oracle_now` | `individual` |
|---:|---:|---:|---:|---:|
| 0 | 0.2421 | 0.2421 | 0.2422 | 0.2310 |
| 18 | 0.0879 | 0.0883 | 0.0875 | 0.0911 |
| 24 | 0.0605 | 0.0606 | 0.0596 | 0.0661 |
| 30 | 0.0445 | 0.0448 | 0.0437 | 0.0480 |

Paired per-seed survival contrasts:

| lag | `uncertain` - `point` | `shuffled` - `point` | `oracle_now` - `point` |
|---:|---|---|---|
| 0 | **+0.0000 [+0.0000, +0.0000]** | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| 18 | **+0.0000 [+0.0000, +0.0000]** | -0.0025 [-0.0074, +0.0024] | +0.0050 [-0.0048, +0.0148] |
| 24 | **+0.0000 [+0.0000, +0.0000]** | +0.0025 [-0.0024, +0.0074] | +0.0000 [+0.0000, +0.0000] |
| 30 | +0.0025 [-0.0024, +0.0074] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0074, +0.0024] |

### And the words really did change

Reported because "identical survival" would be worth nothing if the two arms had
simply said the same things. They did not. Closed-loop fidelity against the true
future, each arm steering its own lives:

| lag | `point` | `uncertain` | `flat` | `shuffled` | `oracle_now` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9266 | 0.9266 | 0.9266 | 0.9266 | 0.9264 |
| 18 | 0.9329 | **0.8683** | 0.8778 | 0.8849 | 0.8857 |
| 24 | 0.9172 | **0.8133** | 0.8244 | 0.8442 | 0.8435 |
| 30 | 0.9097 | **0.7986** | 0.8076 | 0.8298 | 0.8440 |

> The organism using its own calibrated self-uncertainty says materially
> different things — **eleven points worse** against the true future at lag 30,
> while steering its own life — and survives **identically**, with the same
> margin against death, on 400 lives an arm.

That is the preregistered closing condition, met exactly and stated in its own
words: *strongly negative on accuracy while survival is flat is a mechanism that
costs something and buys nothing.* It is a stronger statement than a null. A null
would say the mechanism did nothing. This says it did a great deal and none of it
mattered.

The panel also reproduces probe65 as a positive control: `individual`, handed its
true rates, leads at lags 24 and 30 (0.2375 and 0.1350 against `point`'s 0.2100
and 0.1175) and carries the best floor at every horizon. **Knowing what you are
still pays here. Knowing how unsure you are about it does not.**

## The number future probes should check against first

The most reusable thing this survey produced is not its verdict. It is the
measured shape of the decision this ecology actually poses.

> In probe65's ecology the median **consequential** decision — the ticks where
> the word is worth at least one tick of metabolism — is worth **0.123 to 0.166
> of body**, at every horizon from 18 to 36. Near-ties sit at essentially zero.
> The distribution is bimodal and there is very little in between.

That is a ceiling on an entire class of mechanism, not just this one:

> **Any belief-side mechanism whose effect on the projected body is smaller than
> roughly 0.12 cannot change a consequential word here, however well calibrated,
> however principled, and at any horizon.**

Probe60 made it binding to check the oracle ceiling before locking a gate.
This adds a cheaper check that comes before even that one: compute the size of
the effect the mechanism can produce on the decision variable, and compare it
with 0.12, before building anything. `scripts/diagnose_probe67.py` is that check
and it costs a few minutes.

Read backwards, it also explains why probe65 succeeded where this failed. A rate
*error* of 0.0045 a tick times a horizon of 18 is 0.081 of body, which is the
same order as 0.12 and does reach a real decision. The *width* of the belief
about that rate is several times smaller again, and does not. The horizon channel
probe65 opened is wide enough to carry a point estimate and not wide enough to
carry a posterior.

## What this survey does not show

- It does not show that self-uncertainty is worthless in general, only that in
  this ecology, on the need channel, through this objective, it cannot reach a
  decision that matters.
- It does not test a *learned* uncertainty head. It tests the ceiling above one,
  and finds the ceiling below the baseline.
- It does not test **reporting** uncertainty, only **acting** on it. Property 5
  of `md/archive/DIRECTION_2026-07-26.md` asks the organism to represent and say
  facts about its own model; nothing here gives a listener anything to do with
  such a statement, and phase C1's original design — a caregiver that must triage,
  or that could be asked to *look* rather than to feed — remains untested. Probe62
  closed the budget-triage version in its own ecology; it did not close this one.
- It does not test uncertainty as a driver of **information gathering**. Probe62
  measured that for state uncertainty and found timing worth about a tenth of
  inspecting at all. Rate uncertainty is non-stationary in a way state
  uncertainty was not — maximal at birth, decaying as readings arrive — so
  inspection timing driven by it is a genuinely different question. It needs the
  motor policy to become responsive, which is frozen here, so it is not cheap.
- The closed-loop panel below is powered to resolve differences of roughly 0.05
  in survival at 400 lives an arm, and nothing smaller. It cannot distinguish a
  true null from a small effect, and is not claimed to.

## What this closes and what it opens

**Closed.** Do not preregister a mechanism that *acts* on the width of the rate
belief through probe60's objective, in this ecology, at any horizon. Three
independent reasons, and the first is the strong one:

1. **The ceiling loses to the baseline.** `oracle_now` is handed the magnitude of
   its own error and is worse than the point rule at every horizon. No posterior
   beats being told the size of its own error, so no learned width could have
   rescued this. This is probe62's reason, on a different quantity.
2. **The effect never reaches a decision worth making** — 0.0000 to 0.0012 of
   consequential words changed, at every lag from 0 to 48 — because the width is
   seven to twenty-eight fold too small for a decision this ecology treats as
   worth making.
3. **Measured behaviourally at 400 lives an arm**, an eleven-point change in what
   is said produces zero change in survival and none in the floor.

The closure is about the objective and the ecology and not about the estimator,
because the rule is the Bayes-optimal action under probe60's objective given this
posterior, and because Q1 shows the posterior is well calibrated.

**Not closed, and now sharper.**

- **Reporting** uncertainty is untested. Q1 establishes that the organism *has*
  the second-order self-knowledge property 5 asks about — better calibrated than
  probe62's state posterior — and this survey only shows that one particular use
  of it is worthless. Property 5 asks the organism to *say* such facts, and no
  listener here can do anything with such a statement. Phase C1's original design,
  a caregiver that must triage or that can be asked to **look** rather than to
  feed, is the untested route and it is a listener-side change rather than a
  belief-side one — so the 0.12 floor does not obviously apply to it.
- **Information gathering.** Probe62 measured inspection timing for state
  uncertainty and found it worth about a tenth of inspecting at all, in an ecology
  where the width was nearly stationary. Q1 shows this width is not: the
  least-certain quartile carries up to 6.5x the error of the most-certain. That is
  the burstiness probe62 said was missing, and it is the precondition its own
  closing argument named. This needs the frozen motor policy to become responsive,
  so it is not cheap and it wants a ceiling survey first.

**The reusable output** is neither of those. It is the 0.12 floor, and the check
that produces it in minutes.

## What this survey does not show

- It does not show that self-uncertainty is worthless in general, only that in
  this ecology, on the need channel, through this objective, it cannot reach a
  decision that matters.
- It does not test a *learned* uncertainty head. It measures the ceiling above one
  and finds that ceiling below the baseline.
- It does not test reporting uncertainty, or uncertainty as a reason to gather
  information. See above.
- It does not establish that the RLS covariance is the *best* available width,
  only that it is well calibrated and that a strictly better one — `oracle_now` —
  buys nothing either.
- The closed-loop panel resolves survival differences of roughly 0.05 at 400 lives
  an arm and nothing smaller. The observed contrasts are 0.0000 to 0.0050, so the
  panel cannot distinguish a true zero from a very small effect and does not claim
  to. What it does establish is that the effect is far smaller than the
  eleven-point change in behaviour that produced it.
- Coverage of the 95% band runs at 0.82 to 0.91 rather than 0.95. The width is
  therefore slightly optimistic, in the direction that would *understate* the
  mechanism's effect. Since the finding is negative, this cuts against the
  conclusion rather than for it, and is recorded for that reason.
