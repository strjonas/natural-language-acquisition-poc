# Consumption-scoped bodily-event calibration result

Date: 2026-07-25
Decision: **rejected at gate 1**. Causal selection materially improves grounded
restoration fit, but water misses the locked 25% improvement threshold.

## Integrity

The selector, exact 60,000-tick run, gates, and stop rule were committed in
`2026-07-25-consumption-scoped-event-calibration-preregistration.md` before the
new objective was implemented or trained.

Artifacts:
`runs/organism/probe39_consumption_scoped_event_calibration/`.

All 84 organism tests pass. The generic and consumption-scoped objectives are
separately selectable and default off.

## Gate 1

Final replay MAE on valid-bound resource restorations:

| Resource | Generic event | Required | Consumption-scoped | Relative improvement | Decision |
|---|---:|---:|---:|---:|---|
| Food | 0.3397 | < 0.2548 and <= 0.3125 | **0.2303** | **32.21%** | pass |
| Water | 0.3253 | < 0.2440 and <= 0.2556 | **0.2534** | **22.08%** | **fail** |

The water result is directionally better than the generic treatment and 0.85%
better than the fresh zero-event control, but it does not meet the declared
25% mechanism gate. Gate 1 is conjunctive, so the treatment is rejected.

Per the locked stop rule, the real terminal calibration, forecast, memory, and
feasibility gates were not run. A directional result is not silently promoted
by weakening its threshold after reading it.

## What changed

The selector did what the diagnosis predicted:

- food replay MAE fell by 0.1094;
- water replay MAE fell by 0.0718;
- the already-weighted event gradient rose from 16.50% to 24.28% of the base
  bodily gradient globally, and from 16.44% to 24.53% on the lexical path;
- the forced-reset target remained in the ordinary world-model losses but no
  longer received the extra calibration term.

This is evidence that exogenous reset contamination was causal, but not
sufficient evidence that the proposed objective solves calibration.

## Remaining asymmetry

Windowed bound-restoration MAE:

| Window | Food | Water | Food lexical feature norm | Water lexical feature norm |
|---|---:|---:|---:|---:|
| 0-10k | 0.3566 | 0.3550 | 0.516 | 0.517 |
| 20-30k | 0.3318 | 0.3534 | 0.442 | 0.428 |
| 30-40k | 0.3095 | 0.3356 | 1.133 | 0.918 |
| 40-50k | 0.2723 | 0.2961 | 1.204 | 0.780 |
| 50-60k | **0.2314** | **0.2733** | **1.266** | **0.815** |

The two resources begin together. After 30k, food's learned lexical transition
feature grows while water's remains substantially smaller, and the fit
diverges with it. Food also receives more experience: 3,211 restoration
entries and 1,256 bound entries, against 2,868 and 952 for water.

Those observations admit at least two explanations that must not be conflated:

1. water's acquired lexical value is representationally closer to another
   semantic value or has lower downstream transition gain; or
2. the on-policy stream supplies fewer effective bound water targets, so this
   is an experience asymmetry rather than a collision.

The next measurement is read-only binding/consequence geometry on the three
fixed checkpoints (zero, generic, scoped). It must distinguish lexical
embedding collision from downstream transition collapse before any sampling or
representation change.

Do not tune the 0.01 weight. No larger compute or generated data is justified.

## Claim boundary

Causally selecting agent-generated consumption endpoints improves the learned
body model but does not yet satisfy the locked calibration gate. This is not
reflection, self-report, generated language, consciousness, or subjective
experience.
