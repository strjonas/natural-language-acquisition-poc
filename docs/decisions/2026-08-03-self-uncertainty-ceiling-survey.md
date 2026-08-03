# Self-uncertainty: a ceiling survey, and why it closes the mechanism

Date: 2026-08-03. Status: **feasibility survey, not a treatment result.** No gate
was preregistered and none is claimed. Its purpose is to decide whether an
uncertainty-communication mechanism is worth preregistering at all. It decides
that it is not, in this ecology, for a reason that is measured rather than
assumed.

Artifacts: `runs/organism/probe62_uncertain_self/ceiling_survey.json`.
Module: `src/homesocial/organism/uncertain_self.py`. Guards:
`tests/test_uncertain_self.py`.

## Why this was run before anything was built

Probe59 and probe60 hit the same wall from opposite sides. A mechanism worked on
the belief side and the behavioural gate came back flat -- not because the
mechanism failed but because the ecology had no room for it. Probe60's binding
instruction was to *check the oracle ceiling before locking a gate; if a perfect
model cannot reach the threshold, the gate measures something else.*

Reflection -- an organism reporting not its state but the reliability of its own
model of its state -- was the next phase on the ladder. This survey asks the
precondition question first: **is self-uncertainty worth anything here?**

## The lever

`ReportConfig.silent_shock_probability`, default `0.0`, default-inert.

A silent shock changes the body exactly as a loud one does and is drawn from the
same stream; the only difference is that its perceptible marker is withheld. The
silence draw comes from its own generator, so no other stream moves, and the
draws are nested: the silent set at 0.25 is a subset of the silent set at 0.50.

This is the lever that matters for this repository's oldest open problem. The
probe53 filter is exact -- and, per `docs/STATE.md`, still better than anything
learned -- *only because every body-changing event is perceptible*. Withhold
some, and being told your own dynamics stops being equivalent to knowing them.

The lever's validity is guarded, not asserted. Over full 400-tick lives with a
forced identical action sequence, the body trajectory at `q=1.0` is **bit-identical**
to the body trajectory at `q=0.0` while every shock flips from loud to silent
(`test_silence_changes_perception_and_never_the_body`, max difference 0.0e+00
over 400 ticks, 7 and 13 shocks flipped on the two seeds). Silence moves what can
be known and nothing that happens.

## The four bodies

One policy, one motor path, one listener, one set of lives. The tiers differ in
exactly one thing -- where they believe the body is now -- so no gap can be
credited to a better forecast of what the world is about to do. Forward
projection is metabolism only, identically for all of them.

| tier | body |
|---|---|
| `oracle` | the true body, read directly. The ceiling. |
| `naive` | probe53's exact visible-history filter, unchanged. Misses what it cannot see, so it drifts optimistic. |
| `mean_corrected` | the same filter minus the *expected* unobserved loss. A point estimate that knows a statistical fact about itself. |
| `posterior` | a particle cloud over the same history, conditioned on being alive; reports `argmin` of its mean. |
| `posterior_vote` | the same cloud, reporting the need **most likely to be lowest** -- the Bayes rule for the question the organism is actually asked. |

Every tier is scored on **one shared history**. In a closed loop each tier says
something different, is granted something different, and ends up somewhere
different, which makes closed-loop body error a statement about three
trajectories rather than three estimators.

## What the survey found

Named-need accuracy: does the organism name the need that is truly lowest.
40 lives per cell, 48 particles, `seed_base` 880,000,000.

| q | oracle | naive | mean_corrected | posterior | **posterior_vote** | spread~error r | coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.945 | 0.945 | 0.945 | 0.945 | **0.945** | -0.116 | 0.942 |
| 0.25 | 0.952 | 0.868 | 0.870 | 0.852 | **0.868** | +0.356 | 0.926 |
| 0.50 | 0.957 | 0.803 | 0.800 | 0.778 | **0.804** | +0.432 | 0.931 |
| 0.75 | 0.961 | 0.736 | 0.749 | 0.721 | **0.742** | +0.434 | 0.927 |
| 1.00 | 0.959 | 0.693 | 0.705 | 0.683 | **0.692** | +0.415 | 0.923 |

Four things, in order of how much they constrain the next phase.

**1. The lever opens a large gap, and it is real.** The oracle is flat at ~0.95
because it reads the body directly and silence cannot touch it. Every observer
degrades monotonically, to a **26.6-point** gap at `q=1.0`. This is the first
ecology in this repository with substantial headroom on the report endpoint;
probes 59 and 60 had almost none.

**2. None of that gap is recoverable.** `posterior_vote` is the Bayes-optimal
decision rule given the observable history, so no estimator using the same
information can beat it. It **ties the biased point filter at every silence
rate** -- 0.868/0.868, 0.804/0.803, 0.742/0.736, 0.692/0.693. The oracle's
advantage is not superior inference. It is information the silent shocks
destroyed. A better self-model cannot get it back.

**3. The posterior is nonetheless genuinely calibrated.** Coverage of the 5th-to-95th
percentile band is 0.92--0.94 against a nominal 0.90; the correlation between the
cloud's spread and its own error is +0.36 to +0.43; and the error in the
least-certain quartile of ticks is about **2.4x** the error in the most-certain
quartile (0.100 against 0.042 at `q=1.0`). The organism does know when it does
not know. That is second-order self-knowledge and it is measurable with the
planner switched off entirely, which is exactly the kind of endpoint probes 59
and 60 said to reach for.

**4. That calibration is actionable, but barely.** If knowing *where* the body is
cannot be improved, calibrated uncertainty can still pay through an action a
certain organism would not take: asking to be looked at rather than fed. The
matched-budget test allows the cloud to be reset to the truth on a limited number
of ticks of one fixed history, and compares choosing those ticks by the
organism's own spread against choosing the **same number** of ticks at random.

Inspecting helps a great deal. Choosing *when* to inspect helps about a tenth as
much.

- Inspecting at all, over not inspecting: **+0.0 to +22.1 points**, rising with
  budget.
- Uncertainty-timed over rate-matched random: **+0.0 to +2.7 points**. It is
  consistently around +2.5 once `q >= 0.75`, and indistinguishable from zero at
  `q = 0.25`.
- Timing by the *true* error instead: **-6.2 to +3.3 points** against random,
  with no consistent sign.

That last row matters for how the second one is read. Greedy targeting of the
largest current error is a **worse** policy than random at large budgets, so it is
not an upper bound on what timing could achieve, and the best timing rule is not
established here. What is established is the size of the prize: timing competes
for roughly a tenth of what the decision to inspect is worth.

And the budget it needs is not affordable. The +2.7 at `q = 1.0` costs **34.3
inspections per life** -- against about 66 help windows in a 400-tick life, so
more than half of all help spent looking instead of feeding. This test grants
inspections for free. At the affordable end of the sweep, 2.3 inspections per
life, the gain is +2.7 at `q = 1.0` and +0.0 at `q <= 0.75`.

## Why timing buys so little here

Measured directly rather than inferred. At `q=1.0` the naive filter's signed bias
is **+0.051** and it is flat across the whole life -- 0.043, 0.056, 0.062, 0.061,
0.043, 0.044, 0.041, 0.057 across eight 50-tick blocks -- against roughly **0.68**
of cumulative hidden loss over that life. The bias does not accumulate. It
equilibrates.

The mechanism is the bound on the state. The true body is confined to [0,1], and
the filter saturates at the ceiling on about **9.7%** of ticks. Every saturation
destroys the accumulated offset: the optimistic belief hits 1.0 while the truth
is below it, and the error is silently erased. The bound is an unmodelled
evidence channel -- *I know I am not above full* -- and it does the self-model's
job for free.

This is `CLAUDE.md`'s third trap, "the environment doing the model's job", now
with a number attached. It also explains probes 59 and 60 after the fact: a
self-model whose error is capped at 0.05 by homeostasis cannot be load-bearing,
however accurate it becomes.

An uncertainty that is nearly stationary is an uncertainty barely worth timing.
For self-uncertainty to pay rent the ecology must make it **bursty** -- episodes
where the organism is genuinely far more lost than usual -- and a bounded,
frequently-saturated state variable cannot produce that.

## What this closes and what it opens

**Closed.** Do not preregister an uncertainty-communication mechanism in this
ecology. Two independent reasons, and the first is the strong one:

1. Uncertainty cannot improve the report at all. The Bayes-optimal rule ties the
   point filter at every silence rate, so the failure is not that a learned
   mechanism would be too weak -- a perfect one buys nothing.
2. Uncertainty *can* improve inspection timing, by about +2.5 points, but only
   at budgets that would consume more than half of all help, and this survey
   grants inspections for free. Building a listener response, a token, and a rent
   structure to compete for that is not warranted, particularly since probes
   59--60 showed this ecology's policy flattening belief-side gains far larger
   than 2.5 points.

The second reason is a judgement about what is worth building, not a measured
impossibility, and it is recorded as such.

**Not closed, and now sharper.** The silent-shock lever itself is worth keeping.
It is the first thing in this repository that makes probe53's hand-written filter
strictly wrong rather than merely redundant, which is the standing obstacle
recorded in `docs/STATE.md` under "what this result is missing", item 2. A model
fitted to how the body *actually behaves* -- which is what probe61 does -- absorbs
an unobservable drift that a filter hand-coded to the visible events cannot
represent. That comparison is now runnable and has not been run.

## What this survey does not show

- It does not show that self-uncertainty is worthless in general, only in this
  ecology, under this bounded homeostatic body, at these constants.
- It does not test a *learned* uncertainty head. It tests the ceiling above one.
  For the report endpoint that ceiling is exactly the point filter's performance;
  for inspection timing the ceiling is not established, only shown to be worth at
  most a few points at affordable budgets.
- It does not establish the best inspection-timing rule. The true-error oracle
  turned out to be a poor policy rather than a bound, so a better rule than
  posterior spread may exist. The measured size of the prize is what closes the
  mechanism, not the optimality of the rule tested.
- The inspection test resets belief to truth at zero cost in the world. A real
  checkup would cost a help window, which can only make the timing gap smaller,
  never larger.
- Closed-loop survival was not the endpoint and is not reported as one; per
  probes 59--60 it would have been flattened by the help loop regardless.
- Coverage slightly exceeds nominal (0.92--0.94 against 0.90), so the cloud is
  marginally conservative. That biases *for* the uncertainty tiers, not against
  them, and they still tie.
