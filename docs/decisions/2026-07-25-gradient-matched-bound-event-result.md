# Gradient-matched bound-event calibration result

Date: 2026-07-25
Decision: **gate 1 passes decisively; gate 2 fails**. The analytic scale solves
endpoint calibration, but the online stream learns water substantially faster
than food in the final window. The treatment is not promoted.

## Integrity

The derivation, single value 0.0231, three fresh runs, gates, and stop rule were
committed in
`2026-07-25-gradient-matched-bound-event-preregistration.md` before any of the
three runs began. No neighboring weight was trained.

Artifacts: `runs/organism/probe44_gradient_matched_bound_event/`.

## Gate 1: pass

Final replay valid-bound restoration MAE:

| Resource | Zero | Fresh 0.01 reference | 0.0231 treatment | Improvement vs reference | Requirement |
|---|---:|---:|---:|---:|---|
| Food | 0.3182 | 0.2532 | **0.1227** | **51.56%** | < 0.15 and >= 20% |
| Water | 0.2580 | 0.1520 | **0.0508** | **66.57%** | < 0.15 and >= 20% |

Both clauses pass for both resources by large margins. The analytic norm match
is not merely directional: it moves water from predicting a small fraction of
the event to a final replay error close to the 0.05 calibration boundary.

The treatment's sampled event/base lexical gradient ratio is 0.6435 rather
than exactly one. The derivation used fixed-checkpoint water-present segments;
the training distribution and parameters move. It nevertheless lands in a
useful, nonsaturating regime without a grid.

## Gate 2: fail

Online pre-update valid-bound MAE:

| Resource | First 10k | Last 10k | Improvement | Required improvement |
|---|---:|---:|---:|---:|
| Food | 0.3487 | **0.1326** | **0.2161** | >= 0.15 |
| Water | 0.3472 | **0.0569** | **0.2903** | >= 0.15 |

Both learning-progress clauses pass. The locked last-window balance clause does
not:

`abs(0.1326 - 0.0569) = 0.0757`, required <= 0.03.

Gate 2 is conjunctive and fails. Per the stop rule, real terminal calibration,
forecast, memory, and feasibility were not run.

## Why the imbalance remains

The bound loss balances active needs *within one segment*. It cannot balance a
need absent from that segment. The online stream changed across training:

| Window | Bound food entries | Bound water entries |
|---|---:|---:|
| 0-10k | 117 | 137 |
| 20-30k | 127 | 157 |
| 30-40k | 151 | 207 |
| 40-50k | 180 | 286 |
| 50-60k | **190** | **300** |

Water receives 58% more bound calibration events in the final window. The
result is the mirror image of probe39's early food advantage: local per-segment
normalization cannot control cumulative credit under an on-policy,
nonstationary stream.

## What is solved and what is not

Solved locally:

- the amplitude is learnable with the existing representation and model;
- the derived scale is sufficient for both resources on final replay;
- no larger model, generated data, or external compute is needed;
- the failure is no longer event selection, lexical geometry, gradient
  direction, target identifiability, or gross objective scale.

Not solved:

- balanced continual acquisition across a changing sequence of bound bodily
  events.

The next intervention must balance credit **across updates**, not within one
segment. A learner-side running count or stratified bound-event replay may use
only consume action, selected-binding validity, and own-body target dimension.
It must keep weight 0.0231 fixed and compare against a fresh unbalanced 0.0231
control. Do not run another weight.

No external compute or generated data is justified.

## Claim boundary

The agent can now fit the magnitude of grounded bodily consequences on replay,
but the online learning rate remains need-imbalanced. This is not reflection,
self-report, generated language, consciousness, or subjective experience.
