# Ceiling survey: probe65, what a self-model is worth as a function of horizon

Date: 2026-08-16
Status: SURVEY. Instruments, not a result. No gate is scored here; this is what
the gates in `2026-08-16-future-request-preregistration.md` were locked against.
It is a faithful record of what was known before the treatment and is not
rewritten afterwards. One of its readings **was overturned**: section 10 finds
the regret of `state_oracle` and `individual` essentially tied at the operating
lag, and five treatment seeds show the self-model arms already 1.2x to 1.6x
cheaper. See `2026-08-16-future-request-result.md`, "The one prediction that did
not hold".
Module: `src/homesocial/organism/future_request.py`
Artifact: `runs/organism/probe65_future_request/ceiling_survey.json`
Seed band: `1,000,000,000` -- disjoint from probe64's treatment band, which
reaches 978,000,040, and from probe65's own treatment band at `1,020,000,000`.

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.future_request --lives 40 --closed-loop-lives 40
```

## 1. Why this survey exists

Probe64 ended with a number on the wall four probes had died against: a
self-model 95.3% correct about its own burn rate bought no survival, because
*the consequence horizon was shorter than the correction interval*. Homeostasis
absorbed exactly the quantity the self-model improved.

`docs/STATE.md` names the next rung and forbids the cheap route to it -- make one
wrong request unrecoverable, and do not reach it by retuning `help_period`. It
also names the failure mode to check for first:

> survey whether a perfect speaker can survive at all there, because the failure
> mode is a floor.

This survey does that. The floor is not reached -- a perfect speaker still
survives 0.375 at the operating lag -- and survival turns out to respond very
strongly to *one* belief and not at all to another. Both halves are reported in
full because the preregistration is written against them.

## 2. The lever

`help_delay`, default 0, on `ReportConfig`. The caregiver grants on exactly the
clock it always did and hands over exactly the portions it always did; the only
change is that the grant made at boundary `T` answers the last request heard at
or before `T - help_delay`. `tests/test_future_request.py` checks that a life at
lag 0 is tick-for-tick identical to one that never mentions the lever, and that
a delayed request is answered once, in order, and never twice.

A word that is answered at all is answered exactly `help_delay` ticks after it is
said, so the horizon is the lag itself -- read off the world in
`test_the_horizon_matches_what_the_caregiver_actually_does` rather than asserted.

## 3. The arithmetic this survey was built to test

The predicted body when the help lands is `level - rate * horizon`. The two
things an organism can know about itself enter that expression completely
differently:

- an error in **level** enters once;
- an error in **rate** enters multiplied by the **horizon**.

So an organism that reads its own body perfectly and believes it is a typical
member of its species should be beaten, at a long enough horizon, by one that is
unsure where it is and knows what it is. The crossover is computable in advance
from quantities this repository has already measured:

```text
individual's state error          0.0418   (probe64, five seeds)
mean |rate - species rate|        0.0045   per tick
     food  0.008 * 0.30 = 0.0024
     water 0.012 * 0.30 = 0.0036
     energy ~0.0255 * 0.30 = 0.0077
crossover horizon = 0.0418 / 0.0045 =  9.3 ticks
```

Nothing in that derivation is fitted. `0.30` is `metabolic_spread / 2`, the mean
of `|Uniform(0.4, 1.6) - 1|`.

One honest correction, recorded here because it was measured in this survey and
before any treatment seed existed. Probe64's 0.0418 was `individual`'s state
error in *probe64's* ecology, which carried a rationed caregiver. In probe65's
ecology the same arm's measured body error is **0.0272**, which puts the derived
crossover at `0.0272 / 0.0045 = 6.0` ticks rather than 9.3. Both derivations are
recorded; the observed sign change falls between them, at a lag between 6 and 9.
The preregistration locks the version computed from the already-published number,
so that the prediction cannot be said to have used this survey's own data.

## 4. Seven arms

`metabolic_spread` 0.60, `uptake_spread` 0.00, `interoception_probability` 0.03,
40 lives per cell. Every arm is scored on one shared history driven by `oracle`,
so no difference between arms is a difference of trajectory. One rule --
probe60's repaired `E[min(next body)]` at the horizon the lag imposes -- shared
bit-identically; only the beliefs fed into it differ.

| arm | body belief | rate belief | believed lag |
|---|---|---|---|
| `population` | species filter | species | true |
| `snap` | species filter, corrected at every reading | species | true |
| `myopic` | **true body** | species | **zero** |
| `state_oracle` | **true body** | species | true |
| `recursive` | probe63 RLS self-calibration | **learned** | true |
| `individual` | species filter, **no readings** | **true** | true |
| `oracle` | true body | true | true |

`myopic` and `state_oracle` are the same self-model asked with two horizons, so
one integer is the entire difference between them. `myopic` is not a straw arm.
It believes the caregiver answers at once, which is what every organism before
probe65 correctly believed, and at lag 0 it is the **joint best arm in the
table** at 0.983. What the lag does to it is what the lag does to the stance this
repository has always taken.

## 5. Named-need accuracy against the true future

| lag | `population` | `snap` | `myopic` | `state_oracle` | `recursive` | `individual` | `oracle` | future moved | scored ticks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.598 | 0.745 | **0.983** | **0.983** | 0.925 | 0.878 | 1.000 | 0.000 | 13377 |
| 3 | 0.631 | 0.742 | 0.651 | 0.952 | 0.920 | 0.895 | 1.000 | 0.345 | 13377 |
| 6 | 0.714 | 0.821 | 0.548 | 0.947 | 0.950 | 0.910 | 1.000 | 0.452 | 9458 |
| 9 | 0.671 | 0.782 | 0.330 | 0.907 | 0.903 | 0.912 | 1.000 | 0.672 | 10993 |
| 12 | 0.693 | 0.801 | 0.461 | 0.925 | 0.942 | 0.937 | 1.000 | 0.540 | 8737 |
| 18 | 0.562 | 0.667 | 0.586 | 0.828 | **0.908** | **0.924** | 1.000 | 0.411 | 6323 |
| 24 | 0.466 | 0.559 | 0.489 | 0.673 | **0.894** | **0.921** | 1.000 | 0.507 | 3715 |

"future moved" is the share of scored ticks where the truth's own answer changes
when it is told the caregiver is slow -- how much of this task is a prediction
rather than a reading.

### The crossover

| lag | `individual` - `state_oracle` | `recursive` - `state_oracle` |
|---:|---:|---:|
| 0 | **-0.105** | **-0.058** |
| 3 | -0.057 | -0.032 |
| 6 | -0.037 | +0.003 |
| 9 | +0.005 | -0.004 |
| 12 | +0.012 | +0.017 |
| 18 | **+0.096** | **+0.080** |
| 24 | **+0.248** | **+0.221** |

The sign changes between lag 6 and lag 9, and **section 3's arithmetic predicted
that band before the sweep was run** -- 6.0 ticks from this ecology's own state
error, 9.3 from probe64's published one, with nothing fitted in either. The
observation falls between the two.

At lag 0 the ordering is the other way round and decisively so: perfect state
knowledge is worth 10.5 points more than perfect self-knowledge. That reversal is
what makes the effect at long lag attributable to the horizon rather than to one
estimator simply being better, and it is locked as a gate.

Two further readings of the same table:

- **`individual` never sees its own body and still beats `snap`, which sees it
  constantly**, from lag 6 onwards -- 0.924 against 0.667 at lag 18.
- **`myopic` is worse than `population` at lags 9 through 24.** Perfect knowledge
  of where you are is worse than useless when you are answering the wrong
  question; the species filter that at least knows the caregiver is slow beats
  the oracle that does not.

## 6. The floor, which is real

`metabolic_spread` 0.60, each arm speaking for itself from birth, 40 lives, the
unlimited caregiver of every earlier probe. Survival, then mean life steps.

| lag | `population` | `snap` | `myopic` | `state_oracle` | `recursive` | `individual` | `oracle` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.525 | 0.600 | 0.650 | 0.650 | 0.625 | 0.600 | **0.700** |
| 3 | 0.475 | 0.525 | 0.650 | 0.650 | 0.575 | 0.575 | 0.600 |
| 6 | 0.350 | 0.500 | 0.475 | 0.475 | 0.500 | 0.475 | 0.475 |
| 9 | 0.525 | 0.475 | **0.375** | 0.600 | 0.550 | 0.500 | 0.600 |
| 12 | 0.300 | 0.375 | **0.300** | 0.475 | 0.400 | 0.425 | 0.475 |
| 18 | 0.250 | 0.250 | **0.075** | 0.350 | 0.300 | 0.350 | 0.375 |
| 24 | 0.200 | 0.200 | **0.025** | 0.225 | 0.250 | 0.250 | 0.275 |

| lag | `population` | `snap` | `myopic` | `state_oracle` | `recursive` | `individual` | `oracle` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 307 | 334 | 344 | 344 | 335 | 339 | 361 |
| 9 | 293 | 286 | **244** | 318 | 307 | 298 | 318 |
| 18 | 192 | 188 | **88** | 216 | 210 | 210 | 217 |
| 24 | 144 | 144 | **59** | 156 | 154 | 170 | 174 |

Three things this says, and only the first is good news for the probe.

**The lag is behaviourally enormous, and survival can feel it.** `myopic` --
the organism this repository already had -- falls from 0.650 at lag 0 to 0.075
at lag 18 and 0.025 at lag 24, while `state_oracle`, the identical self-model
told how slow the caregiver is, holds 0.350 and 0.225. That is a 4.7x survival
ratio from one integer. Survival in this ecology is *not* saturated: it responds
to a belief, strongly, which is the first time in this repository that has been
true.

**The oracle ceiling falls with the lag but never reaches the floor.** 0.700 at
lag 0, 0.375 at 18, 0.275 at 24. The world is harder but it is not a floor, so a
behavioural endpoint measured here is measuring something.

**The rate does not reach survival.** At lag 18, `individual` -- born knowing its
own rates -- survives 0.350 against `state_oracle`'s 0.350, exactly. At lag 24 it
is 0.250 against 0.225, inside the noise of 40 lives. The 9.6-point advantage
`individual` holds on *naming the right need* does not become a survival
advantage. This is probe64's wall, met again from the need channel, and it is
what section 9 refuses to engineer around.


## 7. Divergence: identical present, divergent future

From the same run at lag 18: 47 matched pairs out of 6,323 scored ticks. A pair
is two ticks from **different lives** whose true bodies fall in the same cell of
a 0.01 grid, whose present-lowest need agrees, and whose true futures differ.

| arm | diverges | diverges correctly |
|---|---:|---:|
| `population` | 0.5745 | 0.3191 |
| `snap` | 0.5745 | 0.3830 |
| **`myopic`** | **0.0000** | **0.0000** |
| `state_oracle` | 0.5106 | 0.4894 |
| `recursive` | 0.8936 | 0.8723 |
| `individual` | 0.8936 | **0.8936** |
| `oracle` | 1.0000 | 1.0000 |

`myopic` scores exactly zero on every pair, and the zero is structural rather
than statistical: its inputs are the present body and the species constants,
both of which agree across a matched pair, so it *returns the same word twice*
and could not do otherwise. That is the control
`md/archive/DIRECTION_2026-07-26.md` phase C2 asks for -- **reports must diverge
when the future diverges while the present is identical** -- with the failing
arm failing for a reason that can be read off its inputs.

`state_oracle` diverges 0.51 rather than 0.00 because it also reads what is
already in flight, and two lives can have different requests outstanding. The
treatment therefore matches on the in-flight vector as well, which removes that
channel and is expected to push `state_oracle` toward `myopic`'s zero. 47 pairs
per seed is thin; the treatment runs the divergence condition on 120 lives.


### What the arms actually believe

Body error and recovered rate error at each lag, so the crossover arithmetic can
be checked rather than taken on trust.

| lag | `population` body err | `snap` | `individual` | `recursive` | `recursive` rate err |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0934 | 0.0493 | 0.0245 | 0.0144 | 0.050 |
| 6 | 0.0978 | 0.0490 | 0.0245 | 0.0127 | 0.047 |
| 12 | 0.1042 | 0.0537 | 0.0273 | 0.0173 | 0.072 |
| 18 | 0.1072 | 0.0525 | 0.0272 | 0.0182 | 0.091 |
| 24 | 0.1162 | 0.0533 | 0.0280 | 0.0162 | 0.120 |

The species prior's rate error is 0.300 at this spread, so `recursive` ends each
life knowing its own burn rate three to six times better than "I am typical".
`individual` carries no readings and a body error of 0.027; `recursive` carries
readings and 0.018; `population` carries 0.107. Every one of those is a fact
about the *state*, and none of them is what separates the arms at long lag.

## 8. The rationed caregiver, swept and not adopted

Probe64's `caregiver_store` was swept as a second axis, on the reasoning that a
lag makes a request be about the future while a finite basket makes a request
aimed at the wrong need cost supply that never comes back. 40 lives, survival.

| lag | store | `population` | `snap` | `myopic` | `state_oracle` | `recursive` | `individual` | `oracle` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0 | 0.300 | 0.375 | 0.300 | 0.475 | 0.400 | 0.425 | 0.475 |
| 12 | 30 | 0.300 | 0.375 | 0.300 | 0.475 | 0.400 | 0.425 | 0.475 |
| 12 | 24 | 0.200 | 0.250 | 0.175 | 0.300 | 0.300 | 0.275 | 0.325 |
| 12 | 18 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 18 | 0 | 0.250 | 0.250 | 0.075 | 0.350 | 0.300 | 0.350 | 0.375 |
| 18 | 30 | 0.250 | 0.250 | 0.075 | 0.350 | 0.300 | 0.350 | 0.375 |
| 18 | 24 | 0.150 | 0.175 | 0.050 | 0.225 | 0.225 | 0.250 | 0.250 |
| 18 | 18 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The basket does nothing this probe needs, and the sweep says why in three rows.
At store 30 every number is **bit-identical to the unlimited caregiver**: a
delayed organism dies long before it can spend thirty portions, so probe64's
operating basket never binds here at all. At 24 the basket binds and lowers every
arm by about the same amount -- the ordering is untouched and no contrast opens.
At 18 nothing survives.

That is the expected shape on reflection: a store penalises how *much* is asked
for, which is probe64's size channel, and a grant aimed at the wrong need costs
exactly as much basket as one aimed at the right need. It is reported because it
was measured, and it is **not adopted**: probe65 is scored with one lever.

## 9. What this survey fixes before any gate is locked

**The lag: 18 ticks.** Chosen on a stated principle rather than on the effect
size. One grant arrives per `help_period` and three needs compete for it, so a
need is re-served about every `help_period * 3 = 18` ticks -- the same constant
probe64 derived for the size decision. A caregiver whose lag equals that interval
is exactly one whose mis-aimed request cannot be repaired before that need's next
opportunity, which is `docs/STATE.md`'s "make one wrong request unrecoverable"
stated as an equation. Lag 24 shows larger effects on every belief-side endpoint
and is *not* chosen, because its lives are half as long, its scored ticks 41%
fewer, and picking the lag with the biggest number after seeing all seven would
be selection.

**The store: zero.** Probe64's basket was swept as a second axis and is reported
in section 8. Probe65 introduces one lever and is scored with one lever; adding
a second to reach for a behavioural result is the move this repository forbids.

**The falsifier is the lag-0 column.** Whatever the self-model buys at lag 18, it
must *lose* at lag 0, where perfect state knowledge is worth 10.5 points more
than perfect self-knowledge. A gate is locked on that reversal, because without
it the effect at long lag could be one estimator simply being better.

**What is deliberately not gated.** The rate-versus-state contrast on *survival*.
The survey measured it and it is flat -- 0.350 against 0.350 at the operating
lag. Gating a quantity already shown to be flat would be theatre. The
preregistration records the prediction that it will stay flat, so that a
contradiction is reported as a surprise rather than as a success.

## 10. Regret: the endpoint that reconciles the two halves

Accuracy counts wrong words. **Regret** weighs them: the true value of the best
word on that tick minus the true value of the word actually said, in body units,
under the same shared rule. It is the belief-side endpoint denominated in the
currency survival is denominated in, and it is why the two halves of this survey
do not contradict each other.

| lag | `population` | `snap` | `myopic` | `state_oracle` | `recursive` | `individual` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0316 | 0.0153 | 0.0001 | **0.0001** | 0.0039 | 0.0079 |
| 3 | 0.0283 | 0.0157 | 0.0289 | **0.0005** | 0.0041 | 0.0064 |
| 6 | 0.0299 | 0.0129 | 0.0538 | **0.0010** | 0.0032 | 0.0088 |
| 9 | 0.0340 | 0.0170 | 0.0731 | **0.0029** | 0.0082 | 0.0064 |
| 12 | 0.0396 | 0.0203 | 0.0778 | **0.0029** | 0.0043 | 0.0066 |
| 18 | 0.0379 | 0.0230 | 0.0425 | 0.0057 | 0.0065 | **0.0064** |
| 24 | 0.0253 | 0.0156 | 0.0302 | 0.0041 | 0.0069 | **0.0028** |

**`state_oracle`'s regret rises 57-fold across the sweep, from 0.0001 to 0.0057.
`individual`'s does not move at all: 0.0079, 0.0064, 0.0088, 0.0064, 0.0066,
0.0064, 0.0028.** That is section 3's arithmetic in the currency of the world.
`state_oracle` is the arm that does not know *what* it is, and its cost grows
with the horizon; `individual` is the arm that does not know *where* it is, and
its cost is a constant the horizon never touches. The two cross between lag 18
and lag 24.

It also explains the survival table without special pleading. At lag 18
`state_oracle` and `individual` have all but identical regret -- 0.0057 against
0.0064 -- so it would be surprising if they survived differently, and they do not.
`myopic`'s regret is 0.0425, six times either, and it dies. **Survival tracks
regret, not accuracy**, and at lag 18 the accuracy gap of 9.6 points sits on
top of a regret gap of essentially nothing.

The share of ticks whose decision is worth at least one tick of metabolism
(`stakes`) is 0.92, 0.93, 0.96, 0.96, 0.97, 0.81, 0.61 across the sweep, so the
accuracy numbers are not made of near-ties either. Accuracy and regret come
apart here for a different reason: an arm that reads its state exactly makes
*cheap* errors, because when it is wrong the two candidates were genuinely close.
Probe63 recorded the same shape of finding -- prediction and attribution come
apart -- and this is its behavioural twin.

