# Probe72: does a finer decision surface carry individual-rate knowledge to survival? — preregistration

Date: 2026-08-22. This document fixes the endpoint, cells, controls, gates,
sample size, seed band and failure rule before the runner is implemented and
before any treatment life is generated.

Parent result: `docs/decisions/2026-08-17-decision-granularity-result.md`.
Existing closed-loop implementation:
`homesocial.organism.future_request.run_closed_loop`.

## 1. Why this is the next experiment

Probe68 showed in open loop that halving the help quantum while holding help per
tick fixed raises the value of knowing this individual's metabolic rates by
0.0855 [0.0753, 0.0956], while the matched value of state knowledge moves only
0.0101 and spans zero. Its result explicitly stops at what is said. Probe65 is
still the only experiment in which a belief has reached survival, and its rate
contrast was +0.030 at 200 lives per arm — the predicted sign, but too little
data to distinguish from zero.

This experiment supplies the missing physical endpoint without adding a model,
training a parent, changing a rate, or fitting another planner. It asks whether
Probe68's one licensed ecology intervention makes the clean individual-rate
contrast matter to whether the body lives.

## 2. Fixed construction

Everything is Probe68's lag-18 metabolic ecology and Probe65's frozen parent.
Only the already-implemented, default-off `portion_scale` and `help_period`
arguments move.

| ecology | portion scale | help period | relative grant/tick |
|---|---:|---:|---:|
| `baseline` | 1.0 | 6 | 1/6 |
| `fine` | 0.5 | 3 | 1/6 |

The caregiver answers after 18 ticks in both rows. Portion size halves and the
clock doubles, so expected help per tick is identical exactly. No store is
added. The body, shocks, interoception, motor policy, frozen parent, metabolic
spread, maximum life and report rule do not move.

There is no learned-component budget comparison in this probe: all arms use the
same frozen parent once per motor decision and none trains. The computation
budget is therefore matched by construction. Probe69's saturation warning does
not license an estimator-ceiling claim here, and none will be made.

## 3. Arms and the two one-factor contrasts

Three existing arms run in each ecology:

| arm | body state used for the request | metabolic rates used |
|---|---|---|
| `oracle` | true current body | this individual's true rates |
| `state_oracle` | true current body | species rates |
| `population` | filtered body | species rates |

They define the same clean contrasts Probe68 used:

    RATE_VALUE(q)  = survival(oracle, q) - survival(state_oracle, q)
    STATE_VALUE(q) = survival(state_oracle, q) - survival(population, q)

`RATE_VALUE` changes rates with state held perfect. `STATE_VALUE` changes state
knowledge with rates held at the species value. The primary quantity is the
paired difference in differences:

    RATE_STEP = RATE_VALUE(fine) - RATE_VALUE(baseline)

and the specificity lesion is:

    STATE_STEP = STATE_VALUE(fine) - STATE_VALUE(baseline)

The `oracle` arm is deliberately a ceiling, not a learned self-model. A positive
result would show that true individual-rate knowledge has physical rent in this
ecology and that the decision grain controls that rent. It would not show that
the recursive learner captures all of it.

## 4. Sample, seed isolation and intervals

Five seed blocks x 140 consecutive lives per cell = **700 lives per arm**, the
sample size Probe65 fixed before Probe68 existed for separating a survival step
near 0.03. All six cells see the same life seeds, making every contrast paired.

- runtime/power-pilot band: `2,700,000,000`;
- treatment band: `2,800,000,000`;
- seed-block stride: `2,000,000`;
- maximum treatment seed used: `2,808,000,139`.

Both bands are disjoint from Probe71's latest band at 2.6 billion and from each
other. The treatment is processed `seed -> cell -> chunk`; chunks are an
operational checkpoint only and do not change the five statistical units.

Means and two-sided 95% Student-t intervals are computed over the five paired
seed-block values, not over 700 lives as if they were 700 independent model
replications. A same-sign requirement of 4/5 blocks accompanies each positive
gate.

The pre-treatment timing/power pilot contained five lives per cell and is far
too small to set a gate. For transparency, at seed 2,700,200,000 it returned
baseline survival 0.4/0.4/0.6 and fine survival 1.0/1.0/0.4 for
`oracle`/`state_oracle`/`population`; it contains no positive rate contrast.
The direction below comes from Probe68's locked open-loop treatment, and the
sample size comes from Probe65's prior power statement, not from this pilot.

## 5. Locked gates

All gates must pass.

| gate | requirement |
|---|---|
| **G1: grain carries rate value** | `RATE_STEP > 0`, its 95% interval excludes zero, and at least 4/5 seed blocks have the mean's sign. |
| **G2: state specificity** | `abs(STATE_STEP) < RATE_STEP`. A finer world that raises state value just as much does not isolate individual-rate knowledge. |
| **G3: physical rate value exists** | `RATE_VALUE(fine) > 0`, its 95% interval excludes zero, and at least 4/5 seed blocks agree. |
| **G4: both ecologies are viable** | Mean `oracle` survival is at least 0.30 in both rows. |
| **G5: nutrition is fixed** | The computed expected grant per tick is equal between rows to within `1e-12`. |

G1 is the probe. G2 is its lesion. G3 prevents a positive interaction made only
from a worse negative baseline. G4 is Probe60's ceiling rule. G5 asserts the
construction that gives the causal contrast its meaning.

## 6. Failure and continuation rules

If G1 or G2 fails, the claim that decision granularity controls the survival
value of individual-rate knowledge is closed in this fixed ecology. An accuracy
effect is not softened into a physical effect.

If G3 fails while G1 passes, the interaction exists but individual-rate
knowledge still has no demonstrated positive survival value. If G4 or G5 fails,
the construction is invalid and no epistemic claim is made.

No threshold, ecology constant, lag, arm or contrast is changed after seeing the
treatment. The recursive and individual arms may be a later, separately locked
experiment only if this clean ceiling says rate knowledge can reach survival.

## 7. Claim boundary fixed in advance

Passing would show a causal interaction between help quantum and the survival
value of true individual metabolic rates in one three-need ecology. It would not
show learning, language generation, structure discovery, uncertainty reporting,
cross-life memory, a larger body, or a general law across ecologies. The mouth
still selects one of three designer-provided need words and the frozen motor
parent still runs at its original compute budget.
