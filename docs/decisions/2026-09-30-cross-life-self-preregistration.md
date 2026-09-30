# Probe74: persistent identity and cross-episode rate memory — preregistration

Date: 2026-09-30. Written before implementing the new mechanism or generating
survey episodes. This is a construction and mechanism survey; a later
confirmatory treatment needs its own disjoint seed band and preregistration.

Parents: Probe73's rate-only survival lesion and its remaining true-minus-learned
value of +0.0243, Probe68's decision-grain diagnostic, and the exact next work in
`docs/STATE.md`. The question is whether episodic rediscovery can be avoided by
retaining evidence about an individual whose body really persists.

## Fixed construction

Use the frozen Probe52 parent and Probe73's unchanged metabolic ecology: delay
24, caregiver period 6, original portions, unlimited store, metabolic spread
0.60, uptake spread 0, interoception probability 0.03, 400-tick maximum episode.
An identity has one draw of **all four** metabolic multipliers, including safety.
Those constants persist across resets. Birth state, world layout, shocks,
interoception timing, help randomness and motor randomness get new episode
seeds. Neither the identity label nor the true constants enter the learner.

One calibration episode per identity uses the ordinary reset-rate request rule.
Its recursive sufficient statistics become the prior for five subsequent test
episodes. Every test cell uses a fresh episodic recursive tier for **current
state and uptake**, plus a separate recursive calibrator for rate memory. Both
are updated on the same public transitions and readings. The memory calibrator
resets its Jacobian and birth state each episode while preserving its amplitude
estimate, covariance, metabolic readout accumulators and evidence counters.
No state, request ledger, motor hidden state or movement fraction crosses an
episode boundary. The existing `recovered_scale()` rate readout is retained.

## Cells and controls

| cell | rates supplied to the request rule | initial memory |
|---|---|---|
| `reset_rate` | current episode's recursive estimate | own calibration |
| `persistent_rate` | retained recursive estimate | own calibration |
| `true_rate` | actual individual rates | own calibration |
| `swapped_rate` | retained recursive estimate | next identity's calibration |

All cells compute both estimators. The reset cell withholds the retained rate
from the planner but still updates it; the true-rate cell is an oracle ceiling.
The swapped cell cyclically assigns identity i the calibration statistics of
identity (i+1) modulo the block size, then updates them with its own new evidence.
This changes the identity of the prior evidence, not its nominal episode budget.
It may recover; it is not assumed to collapse. Each identity must have at least
one other identity in its block, and no memory is pooled between identities.

On persistent-cell histories, score reset, persistent and swapped-prior rates
against truth while holding the exact state, uptake, ledger and time fixed.
For this same-history swapped counterfactual, keep a third, donor-initialized
calibrator; compute and update it in **every** test cell to match compute.

## Budgets, sample and seed isolation

Five independent paired seed blocks, 28 identities per block, one calibration
episode and five test episodes per identity: **700 test episodes per arm**, plus
140 shared calibration episodes, 2,940 episodes total. Episodes of an identity
are correlated; intervals use the five paired block values, never 700 purported
independent replications. Report per-episode survival, start-of-episode rate
MAE, same-history regret, words changed, and actual evidence readings and ticks.

The per-episode maximum evidence budget is 400 ticks with stochastic readings
at probability 0.03. The retained estimator can use one calibration plus up to
five test episodes (2,400 ticks); the reset readout can use only the current
episode (400). Equal estimator compute does not mean equal evidence budget.
No estimator-family superiority or saturation claim is made.

- Diagnostic/timing band: 3,100,000,000 (ordinary Probe68 diagnostic).
- Identity draws: 3,200,000,000 + block * 2,000,000 + identity index.
- Episode streams: 3,300,000,000 + block * 2,000,000 + identity index * 16
  + episode index, where calibration is 0 and tests are 1 through 5.
- No new development, no checkpoint training, no reuse of Probe73's 3.0B band.

Run `scripts/diagnose_probe68.py` at delay 24 before the survey. Because within-
episode dynamics are unchanged, that instrument measures the relevant ecology;
also report the survey's actual rate-only word changes and matched regret.
Checkpoint calibration episodes and complete five-episode identity/cell units
atomically. Resume refuses changed configuration or parent checkpoint contents.
Complete one entire paired seed block before starting the next.

## Locked continuation gates

Positive contrasts need a positive mean, a two-sided 95% Student-t interval
excluding zero over the five blocks, and at least four positive blocks. All six
clauses are binding; no threshold is moved after a survey episode.

| gate | requirement |
|---|---|
| **C1: persistent-identity physical ceiling** | `survival(true_rate) - survival(reset_rate)` is positive by the rule above. |
| **C2: retained evidence reaches survival** | `survival(persistent_rate) - survival(reset_rate)` is positive by that rule. |
| **C3: matched regret bridge** | On persistent histories, reset-word regret minus persistent-word regret is positive by that rule. |
| **C4: less rediscovery at birth** | On persistent histories, start-of-test-episode reset-rate MAE minus persistent-rate MAE is positive by that rule. Score before the first transition or new reading, with movement fraction reset. |
| **C5: memory belongs to the individual** | On persistent histories, start-of-episode donor-memory rate MAE minus own-memory MAE is positive by that rule. Also report the independent swapped-cell survival contrast without gating it. |
| **C6: viable construction** | Mean true-rate test survival is at least 0.15. |

C1 is checked before interpreting C2: the old independently redrawn-body ceiling
is not automatically a persistent-identity ceiling. C3 separates a rate-only
decision effect from a changed state filter. C4 is the belief-side endpoint.
C5 tests identity specificity without requiring the wrong prior never to recover.
The calibration episode is excluded from every survival gate.

## Failure rule and claim boundary

If C1 fails, this fixed persistent-identity construction does not establish
physical headroom at this survey resolution. If C2 fails with C1 passing,
this exact retained-RLS mechanism does not establish a survival benefit. Failed
C3, C4 or C5 prevents the corresponding bridge, memory or identity claim even
if survival happens to rise. If C6 fails, no epistemic claim is made. Record
every failure; do not add calibration episodes, change clocks, move thresholds,
or increase the sample to rescue this survey.

Passing all clauses licenses an independent replication, not a general claim
about lifelong agency. This is episodic reset of a simulated persistent body,
including reset after death, not literal biological survival across death.
Structure, request rule and words remain supplied; the motor policy still does
not read the self-model. No language generation, discovered axes, uncertainty
reporting, consciousness or sentience claim is licensed.
