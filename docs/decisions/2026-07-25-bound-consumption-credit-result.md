# Bound-consumption credit result

Date: 2026-07-25
Decision: **rejected at gate 1**. Bound, per-need credit improves both resources
but does not meet the locked water fit requirement. Count imbalance contributes
but is not sufficient.

## Integrity

The objective, paired fresh runs, gates, and stop rule were committed in
`2026-07-25-bound-consumption-credit-preregistration.md` before implementation
or training.

The fresh unbalanced control exactly reproduced probe39's stream and endpoint,
confirming the new objective's zero path is inert on this run.

Artifacts: `runs/organism/probe41_bound_consumption_credit/`.
All 87 organism tests pass.

## Gate 1

Final replay valid-bound restoration MAE:

| Resource | Fresh unbalanced control | Treatment | Relative improvement | Requirement | Decision |
|---|---:|---:|---:|---:|---|
| Food | 0.2303 | **0.1935** | **15.99%** | < 0.20 and >= 15% | pass |
| Water | 0.2534 | **0.2271** | **10.40%** | < 0.20 and >= 15% | **fail** |

Gate 1 is conjunctive and fails. Per the locked stop rule, balanced-progress,
real terminal, forecast, memory, and feasibility gates were not run.

## What the manipulation established

The valid-bound online target counts moved from 1,256 food / 952 water in the
control to 1,190 / 1,039 in the treatment. The exposure gap therefore fell
from 31.9% to 14.5%, but water remained 0.0336 worse on final replay.

The bound objective is not optimizer-starved. Its median event/base gradient
ratio is 28.10% globally and **36.91% on the lexical path**, compared with
24.28% and 24.53% for the unbalanced control.

Food improves enough to satisfy both locked clauses. Water moves in the same
direction, but less. This falsifies two simple accounts:

- the generic event term failed only because exogenous resets dominated
  (causal, but incomplete; probe39);
- the residual scoped failure was only unequal food/water event counts
  (contributory, but incomplete; probe41).

## Exact next question

The audits have measured gradient norm but not gradient direction. A 36.9%
lexical event gradient can still be canceled if it opposes the ordinary bodily
objective on the same bound segments.

The next audit should collect fresh local lives from the fixed probe41 control
and treatment checkpoints, update nothing, and compute cosine similarity
between:

- the sealed change-boosted bodily-loss gradient; and
- the checkpoint's event-calibration gradient,

both globally and on the lexical path. Food-bound, water-bound, and
both-present segments must be separated. This distinguishes cancellation from
insufficient effective scale without another loss variant.

Do not tune a weight, add replay, or change architecture before that
measurement. No external compute or generated data is justified.

## Claim boundary

This result localizes residual learned-body underfit after grounded,
need-balanced credit. It is not reflection, self-report, generated language,
consciousness, or subjective experience.
