# Outcome-aware causal self-planner result

Date: 2026-07-30
Decision: **overall gate fails. G2 planner repair passes; G3 continual-model
load-bearing gate fails.**
Preregistration: `2026-07-30-outcome-aware-self-planner-preregistration.md`
Artifact: `runs/organism/probe60_outcome_aware_self_planner/`

## Outcome against the locked gates

| Gate | Requirement | Result | |
|---|---|---:|---|
| F0 | truthful shifted-ecology survival >= 0.90 | **0.920** | pass |
| G1 | reverse the tied-deficit Jensen error | exact reversal | pass |
| G2 | survival >=0.80 mean, >=0.75 on 5/5; >=15 points over legacy; fidelity >=0.90 | **0.904**, 5/5; **+40.4 points**; 0.911 | pass |
| G3 | >=5 points over frozen and stale beliefs; body error <=0.03; constants improve 5/5 | **+1.4 / +0.0 points**; 0.0158; 5/5 | **fail** |
| G4 | scrambled listener and zero belief each cost >=30 points | both **-90.4 points** | pass |
| G5 | oracle ceiling, base/lexical persistence | 0.920; all checks 5/5 | pass |

All promotion gates were conjunctive. The experiment therefore fails overall.
No threshold, seed, body shift, objective, or evaluation life was changed after
the treatment results were observed.

## What the planner repair did

The treatment changed no parameter and added no semantic rule. It retained the
full learned distribution over each token's listener consequences until after
applying the nonlinear homeostatic utility:

```
legacy:        min(E[next body | token])
outcome-aware: E[min(next body) | token]
```

The defect-specific intervention passed exactly. With a clear single deficit,
both planners selected the deterministic targeted token. With two close
deficits, the legacy planner selected a word whose listener effect was spread
uniformly over three outcomes; the outcome-aware planner selected the word
whose realized consequence repaired the lowest axis.

Across the five locked adaptation seeds:

| Condition | Survival mean +/- sd | min--max | Report fidelity |
|---|---:|---:|---:|
| adapted belief + outcome-aware planner | **0.904 +/- 0.009** | 0.89--0.91 | 0.911 |
| adapted belief + legacy planner | 0.500 +/- 0.095 | 0.43--0.66 | 0.962 |
| oracle belief + outcome-aware planner | 0.920 | 0.92--0.92 | 0.935 |
| scrambled real listener | 0.000 | 0.00--0.00 | 0.949 emitted |
| zero planner belief | 0.000 | 0.00--0.00 | 0.000 |

The 40.4-point survival gain is not a fluent-report artifact. Scrambling the
real listener while leaving the internal consequence model untouched reduced
survival to zero. Clamping only the planner's belief to zero also reduced
survival to zero. An arbitrary non-need vocabulary token wins the unit test
when its learned listener consequence is changed to the useful one; the
treatment selector contains no need-word table.

Probe59's fidelity paradox also disappears. The legacy planner spoke a need
word on only 67.5% of primitive ticks and looked 96.2% truthful conditional on
speaking one. The outcome-aware planner spoke one on 93.3% of ticks, survived,
and retained 91.1% fidelity. Selective silence was not rewarded.

## Why the overall claim fails

The factorial conditions isolate the negative result:

| Belief source | Planning model | Survival | Body error | Fidelity |
|---|---|---:|---:|---:|
| adapted causal | adapted causal | **0.904** | **0.0158** | 0.911 |
| adapted causal | frozen causal | **0.904** | 0.0158 | 0.919 |
| stale analytic | adapted causal | **0.904** | 0.0669 | 0.841 |
| frozen causal | frozen causal | 0.890 | 0.0665 | 0.855 |
| oracle analytic | adapted causal | 0.920 | ~0 | 0.935 |

The adapted model remains genuinely better as a body observer: body error is
4.2 times lower than the frozen/stale conditions, and constant error falls
from 0.0978 to 0.0175 with every seed moving toward the new truth. But the
outcome-aware decision rule makes the coarse help choice robust to the stale
models' numerical error. A stale analytic belief survives exactly as often as
the adapted belief, despite being less faithful to the current lowest-need
label. The ecology has enough supply and the action is only a three-way help
allocation, so many numerical self-estimates induce the same viable decisions.

This falsifies the preregistered Phase A1b claim that online improvement of the
current three-axis self-model becomes behaviourally load-bearing under this
one-step planner. The new objective repairs the policy; it does not show that
continual recalibration controls the repaired policy.

## Integrity and persistence

- Adaptation used 40,000 primitive ticks independently at seed bases
  82,100,000 through 122,100,000.
- Mean body error was 0.0158 +/- 0.0046; mean constant error after adaptation
  was 0.0175 +/- 0.0081.
- Every pre-existing non-causal tensor remained bit-identical on 5/5 seeds.
- The lexical gate passed 5/5: intact comprehension 1.000, cyclic control
  0.0089, paired action change 0.9911.
- Oracle belief was never below the learned condition; it survived 0.920 on
  every seed.
- Row-level JSON and CSV contain every condition and seed. The headline gate
  arithmetic was independently re-read from those artifacts after the run.

## Decision and next work

The legacy max-min-of-mean defect is closed as a planner bug. The
outcome-aware planner is a useful, causally validated instrument for later
experiments, but it is not promoted as evidence that continual self-model
learning matters to behaviour. The exact one-step mechanism is closed as a
route to G3; it will not be rescued with soft-min temperatures, hand-weighted
deficits, special tie rules, vocabulary restrictions, or ecology retuning.

Phase A2 structure discovery remains next. Its endpoints are belief-side and
cannot be flattened by a robust three-way policy. The result also sharpens a
binding design requirement for B1 and later phases: a corrigibility experiment
must include an intervention whose correct action changes when the belief is
corrected, and must demonstrate that change before survival is used as a gate.

This result does not show latent structure discovery, evidence correction,
uncertainty, future-self reflection, consciousness, sentience, or unrestricted
language.
