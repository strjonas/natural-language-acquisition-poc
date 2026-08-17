# Preregistration: probe65, what a self-model is worth grows with the horizon

Date: 2026-08-16
Status: PREREGISTRATION. Written before the treatment is run. Gates below are
locked and are not adjusted after seeing results. A failed gate closes its
mechanism.
Module: `src/homesocial/organism/future_request.py`
Survey this rests on: `2026-08-16-future-request-ceiling-survey.md`

## 1. The claim under test

Four probes -- 59, 60, 62 and 64 -- have now found the same wall from four
directions. Probe64 put a number on it: a self-model 95.3% correct about its own
burn rate changed what the organism said and bought no survival, because *the
consequence horizon was shorter than the correction interval*.

Probe65 proposes that this was never a fact about self-models. It was a fact
about **horizons**, and it is derivable rather than empirical:

> The predicted body when help arrives is `level - rate * horizon`. An error in
> *where you are* enters that expression once. An error in *what you are* enters
> it multiplied by the horizon. So the value of knowing your own state is a
> constant and the value of knowing your own nature grows without bound, and
> there is a crossover horizon at which an organism that reads its own body
> perfectly while believing it is typical is beaten by one that is unsure where
> it is and knows what it is.

Stated so it can be attacked, and with the number fixed in advance:

```text
individual's state error       0.0418   (probe64, five seeds, already published)
mean |rate - species rate|     0.0045   per tick, from the frozen constants
crossover horizon              0.0418 / 0.0045 = 9.3 ticks
```

Nothing in that is fitted. If the crossover is not in that neighbourhood, or if
the ordering does not *reverse* at zero horizon, the account is wrong.

## 2. The lever

One, default off: `help_delay` on `ReportConfig`. The caregiver keeps its clock,
its supply and its portions exactly; the grant at boundary `T` answers the last
request heard at or before `T - help_delay`.

`help_period` is **not** retuned. `docs/STATE.md` forbids it by name and the
temptation is recorded here so that it cannot be taken later. No second lever is
added: probe64's `caregiver_store` was swept as a survey axis and is **not**
adopted, because reaching for a behavioural result by stacking levers is the move
this repository forbids.

## 3. Mechanism

Nothing is trained. The self-model is probe63's RLS calibrator, untouched; the
motor policy is probe52's frozen parent; the rule is probe60's repaired
`E[min(next body)]` at the horizon the lag imposes, one expression shared
bit-identically by every arm:

```text
base   = min(1, level - rate * horizon + arriving)
score(word) = E_portion[ min(base with that word's portion added) ]
             -> the word with the highest score
```

`arriving` is the organism's own outstanding requests, recovered from what it
said, when it said it, and the caregiver's public clock. It is required rather
than optional: the survey found that without it a *perfect* speaker survives
0.225 against 0.475 for one that ignores the lag, because a controller with dead
time and no compensation oscillates.

Seven arms, differing only in what they believe:

| arm | body belief | rate belief | believed lag |
|---|---|---|---|
| `population` | species filter | species | true |
| `snap` | species filter, corrected at every reading | species | true |
| `myopic` | **true body** | species | **zero** |
| `state_oracle` | **true body** | species | true |
| `recursive` | probe63 RLS self-calibration | **learned** | true |
| `individual` | species filter, **no readings** | **true** | true |
| `oracle` | true body | true | true |

`myopic` and `state_oracle` share one self-model object; one integer is the whole
difference. `myopic` believes the caregiver answers at once -- which every
organism before probe65 correctly believed -- and at lag 0 it is the joint best
arm in the survey at 0.983, so it is not a straw control.

## 4. Operating point, fixed by the survey

- treatment world `metabolic`: `metabolic_spread` 0.60, `uptake_spread` 0.00
- `interoception_probability` 0.03
- **`help_delay` 18**, `caregiver_store` 0.0, `portion_requests` off
- 5 seeds x 40 lives; the divergence condition on 120 lives
- seed band `1,020,000,000`, stride `2,000,000`, disjoint from the survey band
- checkpoint `runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz`

**Why 18 and not 24.** One grant arrives per `help_period` and three needs
compete for it, so a need is re-served about every `help_period * 3 = 18` ticks
-- the constant probe64 derived for the size decision. A caregiver whose lag
equals that interval is exactly one whose mis-aimed request cannot be repaired
before that need's next opportunity. Lag 24 shows larger effects on every
belief-side endpoint and is deliberately **not** chosen: its lives are half as
long and picking the largest of seven numbers after seeing all seven is
selection.

The `both` world (`uptake_spread` 0.60 as well) is run and **reported, not
gated**, on probe64's precedent.

## 5. Locked gates

Each is scored per seed and passes at **4 of 5**.

**G1 -- the task became a prediction.**
`oracle - myopic >= 0.20` on named-need accuracy at the operating lag. Survey
margin 0.414. If this fails, the lag did not change the task.

**G2 -- the decisive one. A learned self-model beats perfect state knowledge.**
`recursive - state_oracle >= 0.04`. Survey margin 0.079. If this fails, probe65's
central claim is false and the result document says so.

**G3 -- the headroom is real, oracle against oracle.**
`individual - state_oracle >= 0.04`. Survey margin 0.096. G2 without G3 would be
a learner beating a control by accident; G3 without G2 would be headroom nothing
reaches.

**G4 -- the falsifier. The horizon is what bought it.**
At lag 0, in the same world on the same seeds: `state_oracle - recursive >= 0.02`
**and** `state_oracle - individual >= 0.02`. Survey margins 0.058 and 0.105.
This is the gate that distinguishes probe65's account from "the self-model is
simply a better estimator". The advantage must not merely shrink at zero
horizon; it must **reverse**.

**G5 -- no false discovery.**
Null world (`metabolic_spread` 0) at the operating lag:
`|recursive - population| <= 0.005`.

**G6 -- the readings are what did it.**
Shuffled readings (another organism's body, matched in count and timing):
`recursive - population <= 0.01`. The self-model must be destroyed, not slowed.

**G7 -- identical present, divergent future.**
On pairs of ticks from *different lives* whose true bodies fall in the same 0.01
cell, whose in-flight requests match, whose present-lowest need agrees, and whose
true futures differ: `recursive` correct divergence `>= 0.60` **and** `myopic`
divergence `<= 0.05`. Survey values 0.872 and 0.000.

This is `md/archive/DIRECTION_2026-07-26.md` phase C2's named control -- *reports
must diverge when the future diverges while the present is identical* -- and the
failing arm fails structurally: `myopic`'s inputs are the present body and the
species constants, both matched, so it returns the same word twice and cannot do
otherwise.

**G8 -- looking ahead is load-bearing for survival.**
`state_oracle - myopic >= 0.15` on survival at the operating lag, each arm
speaking for itself. Survey margin 0.275 (0.350 against 0.075). This is the first
behavioural gate in this repository that a belief has been able to carry.

**G9 -- the advantage grows with the horizon.**
Per seed, `individual - state_oracle` at lag 24 `>` at lag 12 `>` at lag 0.
Survey values +0.248, +0.012, -0.105. This tests the mechanism -- error times
horizon -- at three well-separated points rather than at one.

## 6. What is deliberately not gated, and the prediction recorded in advance

**The rate does not gate survival.** The survey measured it and it is flat: at
the operating lag `individual` survives 0.350 against `state_oracle`'s 0.350,
exactly, while holding a 9.6-point advantage on naming the right need. Gating a
quantity already shown to be flat would be theatre.

The prediction recorded here, before the run: **`recursive` and `individual` will
not separate from `state_oracle` on survival, within +/- 0.05.** If the treatment
contradicts this, it is a surprise and is reported as one.

The survey also states why, and that explanation is itself a prediction:
**survival tracks regret, not accuracy.** Mean regret -- the true value of the
best word minus the true value of the word said -- is 0.0057 for `state_oracle`
and 0.0064 for `individual` at the operating lag, essentially identical, while
`myopic`'s is 0.0425. The prediction is that the treatment reproduces this
ordering, and that regret rises with the horizon for `state_oracle` while staying
flat for `individual`.

## 7. Failure conditions

- **G2 fails**: the central claim is false. Do not rescue it by changing the
  operating lag, the rule, the reading rate, the spread, the treatment world, or
  by adding `caregiver_store`. Each is named here so that none can be adjusted
  afterwards.
- **G3 fails**: there is no rate headroom at this horizon and the probe is closed
  regardless of G2.
- **G4 fails**: the effect was never about the horizon, and the whole account in
  section 1 is wrong even if G2 passes.
- **G6 fails**: whatever `recursive` gained was not the readings.
- **G7 fails**: the report does not track the future and phase C2 stays untouched.
- **G8 fails**: survival cannot hear the horizon either, and this ecology's
  behavioural endpoint is closed to belief-side work entirely.

`help_period`, `portion_small`, `portion_large`, `life_steps`, `shock_size` and
every other frozen ecology constant stay frozen.

## 8. Claim boundaries, written before the result

Whatever happens, probe65 will **not** show:

- **that the rule is learned or optimal.** It is designer-supplied, one
  expression, identical across arms. What varies is only the belief fed into it.
  The regret endpoint is measured against that same rule, so it bounds how wrong
  a word is *given the rule*, not in general.
- **that the request ledger is discovered.** The organism is told the caregiver's
  clock and its lag, exactly as probe64 was told the service interval. What it is
  not told is anything about its own body.
- **that the self-model is load-bearing for survival.** Section 6 predicts in
  advance that it is not, and says so.
- **that the report is grammatically future-tense.** The word said is the same
  word; what changes is what it refers to. The claim is referential and is
  established by the divergence control, not by any surface form.
- **anything about ecologies other than this one.** In particular the crossover
  horizon 9.3 is a number about *this* body's rates and *this* spread. The claim
  that a crossover exists is general; the claim that it is at 9 is not.
