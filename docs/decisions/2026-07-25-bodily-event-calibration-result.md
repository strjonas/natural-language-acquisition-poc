# Bodily event-calibration result

Date: 2026-07-25
Decision: the scale-balanced event term is **rejected**. The fresh control
reproduces terminal compression, but the treatment neither expands the
counterfactual terminal margin nor preserves food-label semantics. Forecast and
persistent memory remain intact.

## Integrity

The per-entry event term, physical scale, single weight, fresh zero control,
evaluation order, gates, and stop rules were committed in
`2026-07-25-bodily-event-calibration-preregistration.md` before the loss changed
or either checkpoint was trained.

One unevaluated pilot control was discarded because its saved dormant planning
horizon was 1 rather than probe33's 2. The exact control and treatment both
record planning and model horizons 2 and otherwise reproduce probe33's
configuration.

Artifacts: `runs/organism/probe36_bodily_event_calibration/`.

## Manipulation

The treatment adds one loss term over lived per-need targets whose absolute
delta exceeds the sealed 0.175 event boundary:

```text
0.01 * mean(event squared error) / 0.4^2
```

Zero is the default and is unit-tested as numerically identical to the previous
loss. No hidden kind, counterfactual, or language target enters learning.

## Control reproduction

The fresh zero-weight control passes all three reproduction requirements:

| Gate | Requirement | Control |
|---|---:|---:|
| C1 food margin ratio | < 0.25 | **0.0422** |
| C1 water margin ratio | < 0.25 | **0.0685** |
| C2 demanded choice, food / water | >= 90% | **99.28% / 100.00%** |
| C3 post-return need ordering | >= 90% | **100.00%** |
| C3 rounds 3-8 reuse | each >= 90% | **100.00% each** |

The causal comparison is licensed. Terminal compression is not peculiar to the
sealed probe33 checkpoint.

## Mechanism failure

| Quantity | Fresh control | Event treatment | Requirement |
|---|---:|---:|---:|
| Food margin ratio | 4.22% | **3.90%** | 75-125% |
| Water margin ratio | 6.85% | **6.10%** | 75-125% |
| Food terminal-score MAE | 0.1804 | **0.1563** | < 0.05 |
| Water terminal-score MAE | 0.1368 | **0.1359** | < 0.05 |
| Food demanded choice | 99.28% | **76.26%** | >= 90% |
| Water demanded choice | 100.00% | 100.00% | >= 90% |

E1, E2, and E3 all fail. The treatment slightly lowers absolute score error but
does so by moving all scores, not by restoring the demanded-minus-wrong
contrast. It also damages the food-label ranking.

The result is not a near miss. A scale-normalized loss on lived event entries
does not calibrate the binding-conditioned hypothetical terminal state.

## Non-regression

Both conditions retain:

- 100.00% post-return urgent-index survival;
- 100.00% cross-round reuse in every round 3-8;
- acute controls between 30.00% and 38.33%, below 45%; and
- two valid rows grounded, zero in both controls.

The treatment fails specifically at event value and food semantics, not at
metabolic forecast or persistent storage.

## Feasibility characterization

These rows cannot rescue the failed mechanism and are reported because the
preregistration fixed the full evaluation order.

| Gate | Requirement | Control | Treatment |
|---|---:|---:|---:|
| 1 Mean intact advantage | > 0 | -0.0063 | -0.0032 |
| 2 Positive contexts | >= 75% | 46.33% | 46.33% |
| 3 Write advantage drop | >= 0.05 | 0.0182 | **0.0778** |
| 4 Collapsed-label drop | >= 0.05 | 0.0182 | **0.0778** |
| 5 Branch contingency | >= 60% | 94.89% | 79.22% |
| 6 Matching label selects target | >= 60% | 99.44% | 88.78% |
| 7 Danger label avoids target | >= 90% | 94.56% | 90.44% |

The event term makes the two causal magnitude gates pass, but not by calibrating
the terminal event margin: it pushes the no-write advantage from -0.0245 to
-0.0811 while intact remains slightly negative. The demanded-resource sign
split remains exact. This is why mechanism-first ordering matters.

## What is learned from the negative

The training term supervises **lived** event targets. The failed audit queries a
terminal action after an explicit hypothetical lexical write and three public
settling observations. Two explanations remain distinguishable:

1. the treatment does not calibrate even real, in-distribution labeled
   consumption events; or
2. it calibrates lived events, but the explicit counterfactual write/settling
   state does not carry their learned magnitude even though it carries enough
   information to rank objects.

The next experiment is therefore read-only: compare terminal event magnitude on
the real acquired-label path with the explicit-write path for both fresh
checkpoints. Do not change the weight, loss, budget, branch, or thresholds
until that localization is measured.

No compute or data request is justified.

## Claim boundary

This is a controlled negative result about transferring learned bodily-event
magnitude into a lexical counterfactual. It is not reflection, self-report,
generated language, or evidence of consciousness.
