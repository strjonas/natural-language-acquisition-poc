# Bound-event transition-input identifiability result

Date: 2026-07-25
Decision: **identifiable but attenuated**. The exact lived transition input
perfectly separates food and water bound restorations; the learned delta
amplitude remains severely compressed.

## Integrity

The checkpoint, collection cap, exact feature blocks, offline probes,
thresholds, and interpretation were committed in
`2026-07-25-bound-event-input-identifiability-preregistration.md` before the
audit was implemented or run.

A pre-data amendment corrected an invalid proposed direct-column ablation:
the recurrent core has already read the lexical bank, so removing only direct
binding columns is descriptive rather than causal. That amendment was
committed before implementation and before any input was collected.

The audit collected the full 1,000 events in 16,235 local ticks and updated no
organism parameter. Raw examples are retained as JSON.

Artifacts: `runs/organism/probe43_bound_event_input_identifiability/`.
All 88 organism tests pass.

## Locked gates

| Measurement | Result | Requirement | Decision |
|---|---:|---:|---|
| Full held-out resource accuracy | **100.00%** | >= 90% | pass |
| Lexical-only held-out accuracy | **100.00%** | >= 90% | pass |
| Full-input nearest-neighbor agreement | **100.00%** | >= 80% | pass |
| Food predicted/target magnitude | **40.47%** | >= 75% | attenuated |
| Water predicted/target magnitude | **28.20%** | >= 75% | **attenuated** |

The no-direct-binding descriptive probe is 96.50% overall (100% food, 93.75%
water), consistent with the known second lexical path through recurrent reads.

## Actual lived values

| Resource | Events | Target delta mean ± SD | Predicted delta mean |
|---|---:|---:|---:|
| Food | 441 | 0.3195 ± 0.0522 | 0.1293 |
| Water | 559 | 0.3118 ± 0.0441 | 0.0879 |

Water is not underrepresented in this fresh fixed-policy audit; it is the
majority class. Nor is its identity ambiguous. The same lexical value that
classifies it perfectly drives only 28% of its physical magnitude.

## What is now ruled out

The remaining amplitude failure is not:

- event scarcity;
- valid-binding scarcity;
- replay dilution;
- generic reset contamination alone;
- lexical collision on the scoped/bound model;
- downstream semantic-rank collapse;
- food/water count imbalance alone;
- gradient cancellation; or
- transition-input target aliasing.

Within the measured local mechanisms, the remaining diagnosis is objective
equilibrium: the bound event signal is useful, aligned, and identifiable, but
its declared scale leaves the shared transition output attenuated.

Probe42 measured the water-present lexical event/base norm ratio at 0.4336 for
weight 0.01. Because this auxiliary gradient is linear in its scalar weight at
a fixed checkpoint, matching its lexical norm to the base gives one analytic
value:

`0.01 / 0.4336 = 0.02306`, rounded before training to **0.0231**.

This licenses one preregistered value, not a grid.

No external compute or generated data is justified.

## Claim boundary

This shows that a grounded lexical memory perfectly identifies the direction
of a lived bodily consequence while its learned magnitude is compressed. It is
not reflection, self-report, generated language, consciousness, or subjective
experience.
