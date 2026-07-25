# Need-balanced replay result

Date: 2026-07-25
Decision: **the sampler works exactly, but the learning gates fail**. Segment-
level need balancing is rejected and no replay-size, update-count, or weight
sweep is licensed.

## Integrity

The mechanism, learner-visible eligibility rule, fresh pair, accounting
invariant, conditional gates, and stop rule were committed in
`2026-07-25-need-balanced-replay-preregistration.md` before implementation or
training. The implementation was then sealed in commit `8241bb0` after a
6,000-tick smoke run and 270 tests plus 13 subtests.

Artifacts: `runs/organism/probe45_need_balanced_replay/`.

## Default-path control

The fresh uniform control reproduces probe44 exactly, despite passing through
the new default-off implementation:

| Metric | Probe44 | Fresh control |
|---|---:|---:|
| Final replay food MAE | 0.122666 | 0.122666 |
| Final replay water MAE | 0.050826 | 0.050826 |
| Last-window food MAE | 0.132611 | 0.132611 |
| Last-window water MAE | 0.056932 | 0.056932 |

The sealed uniform RNG path therefore survived the code change, not merely an
approximate distributional test.

## Gate 1: pass

The balanced treatment performed 1,021 replay updates for 1,021 collected
segments/lives. Once both pools existed:

- requested needs were food 511 and water 510, difference one;
- 1,021 / 1,021 requests found an eligible segment, or **100.00%** versus the
  locked >= 95%; and
- uniform fallback count was zero.

Selection used only consume action, selected lexical-row validity, and lived
own-body delta. No event name, hidden resource kind, correctness, or
counterfactual outcome entered the sampler.

## Gate 2: fail

Final replay valid-bound restoration MAE:

| Resource | Uniform control | Balanced treatment | Treatment gate |
|---|---:|---:|---:|
| Food | 0.122666 | **0.141372** | <= 0.10 |
| Water | 0.050826 | **0.037006** | <= 0.10 |

Neither treatment error is more than 0.02 worse than control: food is worse by
0.018705 and water improves. But food misses the absolute gate, so the
conjunctive endpoint gate fails.

## Gate 3: fail

Online pre-update valid-bound MAE:

| Resource | First 10k | Last 10k | Improvement | Last gate |
|---|---:|---:|---:|---:|
| Food | 0.353858 | **0.162708** | 0.191150 | <= 0.10 |
| Water | 0.351709 | **0.038837** | 0.312872 | <= 0.10 |

Both progress clauses pass, but food misses 0.10 and the final absolute gap is
**0.123871** versus <= 0.03. Per the locked stop rule, terminal calibration,
forecast, memory, and feasibility were not run.

## What the failure localizes

Every selected segment contained its requested need. Across all 1,021 replay
updates, selected segments were eligible for food 737 times and water 750
times. Because every update hit at least one requested pool, inclusion-
exclusion gives:

- food-only: 271 updates;
- water-only: 284 updates;
- both: 466 updates.

Thus active-need event-objective exposure differed by only 1.76%. The original
"a need absent from a segment gets no credit" diagnosis was real, and this
sampler removed it, but it was not the whole causal mechanism.

The closed-loop stream moved in the opposite direction. Bound water/food event
count ratio rose from 1.326 in the control to 1.670 in treatment overall, and
from 1.579 to 1.728 in the last window. Segment-balanced auxiliary updates can
change the shared recurrent representation and hence the policy that generates
the next stream. Binary replay eligibility does not control that endogenous
behavioral feedback.

The immediate unanswered question is whether equal food/water event gradients
are mutually helpful, orthogonal, or conflicting inside the shared transition
and lexical pathway. The prior probe42 measured calibration versus base-loss
alignment; it did **not** measure food versus water gradient geometry. That
distinction must be tested before any gradient projection, need-specific head,
or further replay mechanism is considered.

## Claim boundary

This is a clean negative result about continual, embodied calibration. It is
not evidence of reflection, self-report, consciousness, or subjective
experience. No larger compute, external data, or generated language is
justified.
