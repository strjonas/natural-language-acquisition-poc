# Online adaptation under changed body dynamics preregistration

Date: 2026-07-26
Status: locked after `v1-causal-bodily-self-report` and before implementation

## Why this experiment

Probe57 learned a causal model of its own body that probe53's hand-coded exact
filter still beats (99.96% and zero error against 91.8%). So far the learned
model is a lossy approximation of a closed form we already possessed, and no
result in this repository shows learning buying anything.

This experiment is the first that can. Change the body's dynamics after
development. The exact filter carries constants that are now wrong and cannot
repair them. A learner can. If the learner does not beat the stale filter here,
the structured causal line has no advantage over analysis and Phase A closes.

## Regime shift

After development, and before any evaluation, multiply the body's constants:

- food/water/energy metabolism x1.5;
- portion sizes x1.5 (small 0.30, large 0.90);
- move energy cost and shock magnitude unchanged.

These are perceptible in consequence but never announced. No signal marks the
change. The organism is not told a regime exists, and nothing in the observation
distinguishes "before" from "after" except the outcomes themselves.

### How this shift was chosen, and why not a harsher one

The first draft of this preregistration specified metabolism x2.0 with portions
x0.6. Piloting that shift showed **every** condition at 0% survival including
the oracle filter given the true new constants: the shifted world was simply
unsurvivable, so survival could not distinguish a good body model from a bad
one. A sweep confirmed the report ecology sits near its survivable limit by
design -- metabolism x1.5 alone drops the oracle filter to 23%.

The shift was therefore re-chosen to be **difficulty-neutral**: metabolism and
portions scale together, so every constant the model holds becomes wrong while
the world stays about as survivable as before. This isolates the variable of
interest, which is stale belief rather than an impossible world.

The selection criterion used **only** the oracle filter and the frozen model --
"is the world still solvable with a perfect model, and does the stale model
actually suffer" -- and required oracle >= 80% with frozen <= 70%. It never
consulted the adapted condition, so it cannot have tuned toward gate 2. The
first configuration meeting the criterion was locked and the search stopped
there. Pilot measurements at 40 lives: oracle 0.90, frozen 0.53, stale filter
0.70.

That the stale *analytic* filter (0.70) degrades more gracefully than the stale
*learned* model (0.53) is itself worth noting: the learned constants were
approximations to begin with, so they had less margin to lose.

## What adaptation is allowed to learn from

The adult report regime masks interoception, so in the shifted regime there is
no body signal to regress against and "continue the developmental loss" is not
literally possible. This experiment therefore grants a bounded **interoceptive
adaptation window**: 40,000 ticks in the shifted regime during which the body
is readable, exactly as in development, after which evaluation runs fully
masked. Every learning condition gets the same window.

This is declared as a limitation, not hidden. It makes A1 a test of *continual
recalibration*, which is what the terminal claim requires ("keeps that model
accurate online as its body changes"). It is not a test of self-supervised
adaptation. Two harder variants are deferred rather than smuggled in: adapting
from death events alone, which reveal that some need reached zero, and adapting
from intermittent interoception, which is Phase B1.

## Conditions

All share one probe57-developed parent and one evaluation seed stream.

| Condition | Causal parameters during the shifted regime |
|---|---|
| `adapted` | continue online through the adaptation window, Adam 3e-3 |
| `frozen` | fixed at their developed values |
| `reset` | reinitialized and relearned from scratch in the same window |
| `stale_exact_filter` | probe53's analytic filter, pre-shift constants, never updated |
| `refit_exact_filter` | probe53's analytic filter, constants re-estimated from the same window |
| `oracle_exact_filter` | probe53's analytic filter given the true new constants |

`refit_exact_filter` is reported as an **upper bound and context, not a gate**.
It is a correctly specified estimator for a known functional form and should be
expected to win; being beaten by it is not a failure. Stating this in advance
prevents the comparison being quietly re-framed after the fact.

Only the causal parameters may move in `adapted` and `reset`. Every motor,
lexical, recurrent, world-model, and legacy-mouth parameter stays bit-identical,
checked as in probe57. The listener model may continue to learn from visible
help outcomes; the listener itself is unchanged by the shift.

## Locked gates

On >= 5 seeds, >= 100 evaluation lives per seed, evaluated in the shifted regime
after an online adaptation budget of 40,000 ticks:

1. **Adaptation beats frozen.** `adapted` survival exceeds `frozen` by >= 10
   points, and `adapted` mean absolute body error is lower.
2. **Adaptation beats deployed analysis.** `adapted` survival exceeds
   `stale_exact_filter` by >= 10 points. *This is the gate the experiment
   exists for*: a hand-coded solution shipped once with the old constants is
   exactly what this repository has been beaten by until now, and recovering
   from a regime it cannot follow is the first thing learning buys.
3. **Parameters move toward truth.** Learned metabolism, move cost, uptake, and
   shock magnitudes are each closer to the new true constants after adaptation
   than before, by sign and by absolute error.
4. **No lexical damage.** The post-adaptation lexical comprehension gate still
   passes: >= 95% intact resource choice, <= 10% cyclic control.
5. **Base parameters bit-identical.**

Gate 2 is the discriminating one. Gates 1, 3, and 4 can pass while 2 fails, and
that combination is a failure of the experiment, not a partial success.

## The null this must survive

The v1 audit established that the belief self-corrects through homeostatic
clipping: error falls over life and mid-life corruption at sd 0.25 costs no
survival. So `frozen` may not collapse on its own.

**Therefore `frozen` is measured, never assumed.** If `frozen` survival in the
shifted regime does not drop at least 15 points below its unshifted 90.6%
baseline, the regime shift was too weak to be informative. In that case
strengthen the shift and rerun, and record the weak shift as a failed attempt.
Reporting a null `frozen` as evidence for adaptation is the specific error this
paragraph exists to prevent.

## Failure closes

If gate 2 fails at this shift magnitude, the structured causal line is not
shown to beat analysis, and Phase A2 (structure discovery) becomes the next
move rather than any further tuning of this mechanism. Do not adjust the
learning rate, the budget, the shift magnitude in the passing direction, or the
thresholds to rescue it.

## Not in scope

No vocabulary growth, no external corpus, no larger compute, no new
`OrganismConfig` knobs. The mechanism goes in its own module beside
`causal_self.py`.
