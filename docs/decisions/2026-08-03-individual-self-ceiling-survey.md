# Individuality: a ceiling survey, and why it reopens the oldest question here

Date: 2026-08-03. Status: **feasibility survey, not a treatment result.** No gate
was preregistered and none is claimed. Its purpose is to decide whether an
individual body is worth modelling at all before anything is built on the idea.
It decides that it is, for reasons that are measured rather than assumed.

Artifacts: `runs/organism/probe63_individual_self/ceiling_survey.json`.
Module: `src/homesocial/organism/individual_self.py`. Guards:
`tests/test_individual_self.py` (10).

## Why this was run

Probe62 closed the uncertainty mechanism and, in doing so, produced the sharper
finding: a self-model whose error is capped at 0.05 by homeostasis cannot be
load-bearing however accurate it becomes. Four probes -- 59, 60, 61, 62 -- had by
then failed against the same wall.

Reading that alongside the standing obstacle in `docs/STATE.md` --

> The **hand-coded probe53 filter is still better** (99.96%, zero error). ...
> Nothing yet shows the learned model doing what the analytic filter cannot.

-- suggests a diagnosis one level below any of the mechanisms tried. Every
organism in this ecology burns fuel at exactly the species rate.
`food_metabolism` is 0.008 for all of them, in every life, forever. So what this
repository has been calling a self-model is a model of **bodies in general**: a
physics whose constants the designer knows in closed form. That is exactly why
probe53's filter is exact, and it means no learner could ever have beaten it.
**There has been nothing individual to learn.**

The organism is also blind to itself. `report.py` masks the body on every tick
after birth, so it gets one interoceptive reading and then flies for 400 ticks on
dead reckoning. Even if there were a fact about itself, it would be unknowable in
principle rather than merely unknown.

## The levers

Three, all default-inert, all guarded rather than asserted.

| lever | what it makes individual |
|---|---|
| `metabolic_spread` | how fast this body spends each need |
| `uptake_spread` | how much good the same granted help does it |
| `interoception_probability` | whether it can ever find out |

Each life draws its own multipliers from `Uniform(1-s, 1+s)`. Every draw comes
from a dedicated generator, so at the defaults every other stream in the ecology
is bit-identical; that is checked over full lives, not a handful of ticks.
`metabolic_spread` is checked to scale depletion by *exactly* the drawn factor
and to genuinely vary across lives -- this repository's "seed that does nothing"
trap applied to the body itself. `interoception_probability` is checked to leave
the body trajectory bit-identical while opening the channel, which is probe62's
silence guard run the other way round: silence withheld a reading the organism
used to get, this grants one it never had, and neither may move what happens.

Readings are delivered in `info` and never in the packet, so the motor path and
everything the policy sees are untouched at any rate.

## What the survey found

Five tiers, one shared history, one policy, 40 lives per cell, seed base
930,000,000. The tiers differ in exactly one thing -- what they believe their own
body is and does.

### Headroom, with no interoception at all

| `metabolic_spread` | oracle | population | individual |
|---:|---:|---:|---:|
| 0.00 | 0.0000 | 0.0000 | 0.0000 |
| 0.15 | 0.0000 | 0.0214 | 0.0000 |
| 0.30 | 0.0000 | 0.0432 | 0.0000 |
| 0.45 | 0.0000 | 0.0618 | 0.0000 |
| 0.60 | 0.0000 | 0.0833 | 0.0000 |

Named-need accuracy over the same cells: `population` falls 0.9997 -> 0.8829 ->
0.7801 -> 0.7135 -> **0.6596**, while `individual` holds at **1.0000**.

Three things follow.

1. **The gap is large.** 34.0 points at spread 0.60.
2. **The gap is entirely individuality.** `individual` is probe53's filter with
   this life's constants substituted and nothing else changed, and it is
   *exact*. So at spread 0.60 the whole of the species filter's error is not
   knowing whose body it is in.
3. **Unlike probe62's, the gap is recoverable.** Probe62's 26.6 points were
   information the silent shocks had destroyed; the Bayes-optimal rule tied the
   biased point filter at every silence rate. Here a filter that knows the
   constants reaches the oracle exactly. The information exists. It has to be
   found out.

### What evidence buys, and what a self-model buys on top

At spread 0.60, sweeping the reading rate:

| rate | population | snap | corrigible | readings/life |
|---:|---:|---:|---:|---:|
| 0.01 | 0.0833 | 0.0691 | **0.0482** | 3.2 |
| 0.03 | 0.0833 | 0.0478 | **0.0254** | 9.9 |
| 0.10 | 0.0833 | 0.0225 | **0.0059** | 33.5 |

Named-need accuracy: `snap` 0.724 / 0.817 / 0.921 against `corrigible` **0.823 /
0.908 / 0.981**.

`snap` corrects its belief to truth whenever a reading arrives and carries
nothing between readings. Phase B1 warns that its first gate is "satisfiable by
clipping"; `snap` is that warning made into a control, and the warning is
justified -- snapping alone recovers a large part of the gap. But carrying a
model of one's own rates between readings buys **a further 6.0 to 9.9 points**,
and cuts body error by a further 30% to 74%. The advantage is largest when
readings are sparse, which is the direction that makes mechanistic sense: the
rate model is what has to carry the organism between the moments it can see
itself.

The corrigible instrument also recovers the rates themselves: mean |scale
- truth| of 0.206 / 0.151 / 0.071 against a species prior error of 0.300.

### Survival is finally sensitive

Closed loop at spread 0.60, rate 0.10, 40 lives, each tier living its own life:

| tier | survival | fidelity |
|---|---:|---:|
| population | 0.525 | 0.645 |
| snap | 0.675 | 0.872 |
| corrigible | 0.675 | 0.956 |
| oracle / individual | 0.700 | 1.000 |

Probes 59 and 60 moved survival by +1.4 and +0.0 points against locked +5-point
gates, and probe60 concluded the three-way help decision was simply insensitive
to numerical belief error. It is not insensitive to *this* error: +15.0 points
against a ceiling of +17.5.

It is still a blunt instrument, and the survey says so. `snap` and `corrigible`
tie on survival at 0.675 while differing by 8.4 points of fidelity, so survival
separates the species filter from everything else and does **not** separate
evidence from a self-model. The belief-side endpoints are what carry that
contrast, exactly as probes 59--60 said to expect.

Note also that the oracle survives only 0.700 here, against 0.92 in the v1
ecology. Individuality makes some bodies unsurvivable at this spread. No number
in this survey is comparable to probe57's.

## Can it find out *what* about itself is different?

The most interesting result, and the one that changed the design.

The instrument is the organism's own parameter sensitivities: one copy of its
body filter per constant it could be wrong about, each with exactly that constant
perturbed, giving `J[need, parameter] = d(predicted body)/d(log parameter)`. That
is not a fact about where the body is; it is a fact about how its own *model*
would respond if a particular belief about itself were wrong. Pooled least
squares over those sensitivities and the residuals at each reading is the ceiling
on localization -- not a learner, a measurement of whether the question is
answerable at all.

At rate 0.10, 8 lives per world:

| world | metabolic share | uptake share | total correction |
|---|---:|---:|---:|
| metabolism only | **0.909** | 0.073 | 1.01 |
| absorption only | 0.255 | **0.622** | 1.84 |
| both | 0.363 | 0.556 | 2.42 |
| neither | — | — | **0.00** |

It discriminates. In a world where nothing is individual the fit returns exactly
zero and the shares are undefined rather than smeared -- there is no false
discovery to report.

**The finding that changed the design.** The first version of this instrument
gave the organism *one* absorption parameter for its whole body, matching
`ReportConfig`, which carries a single `portion_small`. In the absorption world
it then reported metabolic share **0.703** -- confidently attributing to
metabolism a difference that was entirely absorption. Nothing was wrong with the
ecology or with the fit. The self-model simply could not express "I absorb food
badly but water fine", and a per-need truth with nowhere to go went into the
per-need parameter it did have.

So: **a self-model can only localize a fact about itself that its own parameter
set can express, and when it cannot, it does not fail loudly -- it produces a
confident wrong answer.** Giving the organism per-need parameters fixed it
(0.703 -> 0.255 misattribution) without touching the world.

## What this survey does not show

- It is a survey. Nothing here is a preregistered result, and the `corrigible`
  and least-squares instruments are hand-written analytic estimators, not
  learned. They measure what is available, not what a mechanism achieves.
- Localization was measured at reading rate 0.10 with a batch fit pooled over a
  whole life. An online learner at rate 0.03 has strictly less to work with, and
  the treatment preregistration should not have taken this number as its ceiling
  without saying so.
- Interoception is granted free and unprompted. The organism never chooses to
  look at itself, so nothing here bears on the value of self-inspection -- which
  is the endpoint probe62 measured and found nearly worthless in *its* ecology.
- The ecology is changed, so no number here is comparable to probe57's or to any
  v1 result. The null world exists to show the change is inert when the levers
  are off, and it is.
- Nothing here is about structure discovery. The organism is still told it has
  three needs and what kinds of thing act on them. Probe61 is closed and this
  does not reopen it.

## What it opens

The comparison this repository has never been able to make is now available and
is no longer marginal. The hand-written probe53 filter is not merely worse in
this ecology -- it is **structurally incapable**, because the constants it would
need are not knowable at design time. They are facts about an individual that
does not exist until it is born.

That is preregistered as probe63 in
`docs/decisions/2026-08-03-individual-self-calibration-preregistration.md`.
