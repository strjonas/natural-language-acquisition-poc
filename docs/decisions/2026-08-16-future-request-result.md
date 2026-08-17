# Result: probe65, what a self-model is worth grows with the horizon

Date: 2026-08-16
Preregistration: `2026-08-16-future-request-preregistration.md`
Ceiling survey: `2026-08-16-future-request-ceiling-survey.md`
Module: `src/homesocial/organism/future_request.py`
Artifact: `runs/organism/probe65_future_request/treatment.json`
Summary: `.venv/bin/python scripts/summarize_probe65.py`

## Verdict

**All nine locked gates pass, 5 of 5 on every one.** 5 seeds x 40 lives,
thresholds fixed before the run against a survey on a disjoint seed band.

| gate | what it asks | result |
|---|---|---|
| G1 the task became a prediction | `oracle - myopic >= 0.20` | **pass 5/5** |
| G2 self-model over perfect state | `recursive - state_oracle >= 0.04` | **pass 5/5** |
| G3 rate headroom exists | `individual - state_oracle >= 0.04` | **pass 5/5** |
| G4 **the falsifier** | at lag 0 the ordering *reverses*, `>= 0.02` both ways | **pass 5/5** |
| G5 no false discovery | null world, `\|recursive - population\| <= 0.005` | **pass 5/5** |
| G6 shuffled readings | `recursive - population <= 0.01` | **pass 5/5** |
| G7 identical present, divergent future | `recursive >= 0.60`, `myopic <= 0.05` | **pass 5/5** |
| G8 looking ahead is load-bearing | `state_oracle - myopic >= 0.15` survival | **pass 5/5** |
| G9 the advantage grows with the horizon | monotone at lags 0, 12, 24 | **pass 5/5** |

## The mechanism, in one paragraph

The caregiver keeps its clock, its supply and its portions exactly as they were.
The only change is one default-off integer, `help_delay`: the grant made at
boundary `T` answers the last request heard at or before `T - help_delay`. That
turns every word from a statement about the present into a prediction, because
the body when the help lands is `level - rate * horizon`. And the two things an
organism can know about itself enter that expression completely differently -- an
error about **where you are** enters once, an error about **what you are** enters
multiplied by the horizon. Everything below follows from that asymmetry.

## The headline

Lag 18, `metabolic_spread` 0.60, reading rate 0.03. Every arm on one shared
history, one rule, one motor path; only the beliefs differ. The right-hand column
is the same world with the caregiver answering at once -- the ecology of every
probe before this one.

| arm | what it knows | lag 18 | lag 0 |
|---|---|---:|---:|
| `population` | species rates, filtered state | 0.5676 +/- 0.0281 | 0.5724 +/- 0.0238 |
| `snap` | species rates, corrected at every reading | 0.6781 +/- 0.0144 | 0.7315 +/- 0.0125 |
| `myopic` | **true body**, and no idea the caregiver is slow | 0.5779 +/- 0.0068 | **0.9810 +/- 0.0016** |
| `state_oracle` | **the true body every tick**, species rates | 0.8312 +/- 0.0146 | **0.9810 +/- 0.0016** |
| `recursive` | probe63's RLS self-calibration | **0.9329 +/- 0.0152** | 0.9211 +/- 0.0072 |
| `individual` | **its own true rates**, and no readings at all | **0.9359 +/- 0.0147** | 0.8590 +/- 0.0078 |
| `oracle` | true body and true rates | 1.0000 | 1.0000 |

Paired per-seed contrasts, 95% CI over five seeds:

| contrast | at lag 18 | at lag 0 |
|---|---|---|
| a learned rate over a **perfect state** | **+0.1017 [+0.0758, +0.1275]** | **-0.0599 [-0.0695, -0.0504]** |
| a true rate over a perfect state | **+0.1047 [+0.0712, +0.1381]** | **-0.1220 [-0.1315, -0.1125]** |
| a self-model over the same evidence (`snap`) | +0.2547 [+0.2315, +0.2779] | |
| looking ahead at all (`state_oracle - myopic`) | +0.2533 [+0.2249, +0.2818] | |

Every interval excludes zero, and **the two columns have opposite signs**. That
reversal is the whole claim, stated so it can be attacked:

> Whether it is better to know exactly where your body is or exactly what your
> body is, is not a fact about self-models. It is a fact about **how far ahead
> the world makes you think**. With a caregiver that answers at once, reading
> your own state perfectly is worth 12.2 points more than knowing your own
> rates. With one that answers eighteen ticks later, it is worth 10.5 points
> less.

## The crossover, predicted before it was measured

The preregistration fixed the number in advance, from probe64's already-published
state error and the frozen species constants, with nothing fitted:

```text
crossover horizon = 0.0418 / 0.0045 = 9.3 ticks
```

Measured, `individual - state_oracle` by lag, five seeds:

| lag | 0 | 3 | 6 | 9 | 12 | 18 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `individual - state_oracle` | -0.1220 | -0.0797 | -0.0369 | **-0.0014** | **+0.0109** | +0.1047 | +0.2281 |
| `recursive - state_oracle` | -0.0599 | -0.0245 | +0.0156 | +0.0215 | +0.0191 | +0.1017 | +0.2191 |
| sd of the first row | 0.0069 | 0.0061 | 0.0116 | 0.0101 | 0.0111 | 0.0241 | 0.0426 |

**The sign changes between lag 9 and lag 12, and at lag 9 the difference is
-0.0014.** The predicted crossover was 9.3. This is the first quantitative a
priori prediction this repository has made about a number it had not yet
measured, and it is correct to within a tick.

The learned arm crosses earlier, between lag 3 and lag 6, because probe63's RLS
calibrator carries readings as well as rates: it is never as blind to its state
as `individual` is.

## Identical present, divergent future

`md/archive/DIRECTION_2026-07-26.md` phase C2 names one control as the thing that
kills templated narration: *reports must diverge when the future diverges while
the present is identical*. It is run here exactly. A pair is two ticks from
**different lives** whose true bodies fall in the same cell of a 0.01 grid, whose
outstanding requests match, whose present-lowest need agrees, and whose true
futures differ. 598 such pairs per seed, out of 120 lives.

| arm | diverges | diverges **correctly** |
|---|---:|---:|
| `population` | 0.4921 +/- 0.0222 | **0.0006 +/- 0.0011** |
| `snap` | 0.3572 +/- 0.0296 | 0.0009 +/- 0.0017 |
| **`myopic`** | **0.0117 +/- 0.0080** | **0.0000 +/- 0.0000** |
| `state_oracle` | 0.0350 +/- 0.0062 | 0.0064 +/- 0.0044 |
| `recursive` | 0.7808 +/- 0.0500 | **0.7422 +/- 0.0472** |
| `individual` | 0.9209 +/- 0.0121 | **0.8665 +/- 0.0282** |
| `oracle` | 1.0000 | 1.0000 |

Two failures, and they fail in opposite directions, which is what makes the
positive mean something.

`myopic` **cannot** diverge. Its word is a function of the present body and the
species constants alone, and both agree across a matched pair, so it returns the
same word twice -- 0.0000 correct on every seed with zero variance, the same
structural zero probe64's unfactored listener had at exactly 0.5000. `population`
diverges on *half* of all pairs and is right on **0.0006** of them: it has the
freedom to say different things and no information about which, so its divergence
is noise. Only an arm that knows what the two bodies *are* diverges and is right.

That is a report whose referent is a bodily state the organism is not yet in,
established by intervention rather than by fluency. **Phase C2's named control
passes.**

## Survival: the horizon is load-bearing, the rate is not yet

Each arm speaking for itself from birth, 40 lives x 5 seeds.

| arm | survival | mean life steps | regret | consequential accuracy |
|---|---:|---:|---:|---:|
| `population` | 0.2250 +/- 0.0387 | 170.4 | 0.0361 | 0.6514 |
| `snap` | 0.2850 +/- 0.0300 | 188.7 | 0.0205 | 0.7630 |
| **`myopic`** | **0.0900 +/- 0.0561** | **89.1** | 0.0403 | 0.6432 |
| `state_oracle` | 0.3150 +/- 0.0464 | 204.1 | 0.0060 | 0.8849 |
| `recursive` | 0.3450 +/- 0.0557 | 205.9 | **0.0038** | **0.9493** |
| `individual` | 0.3500 +/- 0.0354 | 213.0 | 0.0049 | 0.9367 |
| `oracle` | 0.3550 +/- 0.0430 | 213.7 | 0.0000 | 1.0000 |

**G8 passes on a wide margin and it is the first behavioural gate in this
repository that a belief has carried.** `myopic` survives 0.0900 against
`state_oracle`'s 0.3150 -- a 3.5x ratio, and 89 mean life steps against 204 --
from one integer, with the identical self-model object underneath. Probes 59, 60,
62 and 64 all reported that a belief-side improvement bought no survival here.
It buys survival now, and what it bought it with is **temporal extent**.

`myopic` also carries **body error 0.0000**, because it reads the true body every
tick, and it is the worst arm in the table. Perfect knowledge of where you are is
worse than useless when you are answering the wrong question.

**The rate does not separate on survival, as predicted.** `recursive` beats
`state_oracle` by +0.0300 and `individual` by +0.0350, both inside the +/- 0.05
band the preregistration committed to in advance, and both with sign changes
across seeds (`recursive`: -0.025, 0.000, +0.050, +0.075, +0.050). The
preregistered null held.

## The one prediction that did not hold, and it is the interesting one

The preregistration also predicted, from the survey, that **regret would be
essentially identical** between `state_oracle` and the self-model arms at the
operating lag -- 0.0057 against 0.0064 -- and that this was why survival would be
flat. **Five seeds contradict that.** At lag 18 the ordering is:

| lag | 0 | 3 | 6 | 9 | 12 | 18 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `state_oracle` regret | 0.0001 | 0.0006 | 0.0014 | 0.0031 | 0.0037 | 0.0060 | 0.0082 |
| `recursive` regret | 0.0038 | 0.0037 | 0.0031 | 0.0050 | 0.0053 | **0.0038** | **0.0039** |
| `individual` regret | 0.0093 | 0.0086 | 0.0092 | 0.0079 | 0.0075 | **0.0049** | **0.0050** |
| `myopic` regret | 0.0001 | 0.0275 | 0.0538 | 0.0719 | 0.0736 | 0.0403 | 0.0352 |

`state_oracle`'s regret rises **82-fold** across the sweep. `recursive`'s is flat
at 0.003-0.005 and `individual`'s falls. The regret crossover is complete by lag
18 rather than pending at it: the self-model arms are already 1.2x to 1.6x
cheaper to be wrong with than perfect state knowledge.

So the honest reading changes. The survival difference is **not** absent because
the errors cost the same; it is +0.030 and +0.035 in the direction the regret
ordering predicts, and it is unresolvable because 200 lives per arm cannot
separate 0.03 from 0. That is a sample-size statement, not a null, and it names
the next experiment with a number: separating a 0.03 survival difference at 95%
needs roughly **700 lives per arm**, not 200.

`myopic`'s regret is the other half of the same story -- 0.0403 at lag 18, an
order of magnitude above either self-model arm -- and it is the arm whose survival
*does* move. **Survival tracks regret.** The three tables agree once regret is
the quantity being read.

## The controls

**Null world** (`metabolic_spread` 0), at the operating lag: `population` 1.000,
`recursive` 0.999. There is nothing individual to find and nothing is found. The
same row carries a check nothing else does -- `myopic` scores **0.674** in a world
where no organism differs from any other, which says the horizon effect is about
*time* and not about individuality at all.

**Shuffled readings** (another organism's body, matched in count and timing):
`recursive` falls to **0.488** against `population`'s 0.568, and `state_oracle`
is untouched at 0.831. The self-model is not slowed by bad evidence about itself,
it is destroyed by it -- which is what says the readings did the work.

**Rate recovery**: `recursive` ends each life with rate error **0.092** against a
species prior error of 0.300.

**The `both` world** (metabolic and absorption spread together, reported not
gated): `population` 0.492, `snap` 0.587, `myopic` 0.541, `state_oracle` 0.718,
`recursive` **0.786**, `individual` **0.869**. The ordering holds: the learned
rate recovers 6.8 points over perfect state knowledge and the true rate 15.1.

**Near-ties are not the story.** 79.6% of scored ticks at lag 18 have a decision
margin of at least one tick of metabolism, and restricted to those the ordering is
unchanged and wider: `recursive` 0.9493 against `state_oracle` 0.8849.

## What this does not show

- **That the rule is learned or optimal.** It is probe60's `E[min(next body)]`,
  designer-supplied, one expression, identical across arms. What varies is only
  the belief fed into it. Regret is measured against that same rule, so it bounds
  how wrong a word is *given the rule*, not in general.
- **That the request ledger is discovered.** The organism is told the caregiver's
  clock and its lag, exactly as probe64 was told the service interval. What it is
  never told is anything about its own body. Without the ledger the world is not
  controllable at all -- the survey found a *perfect* speaker surviving 0.225
  against 0.475 for one that ignored the lag -- so it is a precondition of the
  experiment rather than a result of it.
- **That the report is grammatically future-tense.** The word said is the same
  word; what changes is what it refers to. The claim is referential and rests on
  the divergence control, not on any surface form. The vocabulary is three need
  words and the protocol is unchanged in size.
- **That the self-model is load-bearing for survival.** It is not, at this sample
  size, and the section above says so with the number required to settle it.
- **Anything about ecologies other than this one.** That a crossover exists is the
  general claim; that it is near 9 ticks is a fact about this body's rates and
  this spread.

## Where the five properties now stand

From `md/archive/DIRECTION_2026-07-26.md` section 2.

1. **Discovered** -- *partial*, unchanged.
2. **Corrigible** -- **yes**, unchanged. Reconfirmed by G6, where shuffled
   evidence destroys the model rather than slowing it.
3. **Load-bearing across uses** -- *partial*, **advanced**. One self-model object
   now drives probe64's size word and probe65's need word, and for the first time
   a belief reaches **survival**: G8, 0.3150 against 0.0900. The belief that
   reached it is the horizon rather than the rate, so this is not yet the full
   property, and the rate's remaining gap is now quantified rather than vague.
4. **Productive under novel demand** -- *partial*, unchanged from probe64.
5. **Reflexive** -- *partial*, unchanged. The organism now also represents what it
   has itself asked for and not yet received, which is a fact about its own past
   actions rather than about its body, but it still reports none of it.

The terminal claim in section 2 has five clauses. Probe65 is the first evidence
for the fourth -- *can describe states it has not yet been in* -- and the
divergence control is what makes it evidence rather than fluency.

## What this hands to the next rung

1. **The behavioural gate on the rate is now a sample-size problem with a
   number.** Lag 24 or beyond, ~700 lives per arm, fresh seed band. The regret
   ordering predicts the sign in advance, which makes it a real prediction rather
   than a fishing expedition.
2. **Cross-life self-knowledge is promoted.** `individual` beats `recursive` at
   every lag from 12 up, and the gap is exactly the part of each life `recursive`
   spends still finding out what it is -- it ends with rate error 0.092 against a
   handed 0.000. A prior about *itself* carried across lives would close that gap
   and nothing else in the ladder addresses it.
3. **Self-uncertainty now has a use it never had.** At a long horizon a bad rate
   estimate is expensive in a way probe62's ecology could not make it. Probe62's
   closure was scoped to its own ecology; this is not that ecology.
