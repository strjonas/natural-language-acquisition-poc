# Bound-event gradient-alignment result

Date: 2026-07-25
Decision: **aligned, not canceled**. Water underfit is not explained by
opposition between the bound calibration gradient and the ordinary bodily
objective.

## Integrity

The checkpoints, fresh collection budget, measurements, thresholds, and stop
rule were committed in
`2026-07-25-bound-event-gradient-alignment-preregistration.md` before the audit
was implemented or run.

Each checkpoint supplied 160 fresh valid-bound segments before 13,000 local
ticks. No optimizer update, replay insertion, or parameter mutation occurred.
Raw per-segment cosines are retained as JSON.

Artifacts: `runs/organism/probe42_bound_event_gradient_alignment/`.
All 88 organism tests pass.

## Locked result

### Bound-balanced treatment

| Segments | Scope | Median cosine | Negative | Event/base norm | Classification |
|---|---|---:|---:|---:|---|
| All 160 | global | **+0.611** | 2.50% | 0.336 | aligned |
| All 160 | lexical | **+0.457** | 5.00% | 0.455 | aligned |
| Water present, 92 | global | **+0.483** | 4.35% | 0.325 | aligned |
| Water present, 92 | lexical | **+0.407** | 7.61% | **0.434** | aligned |
| Food present, 76 | lexical | +0.610 | 1.32% | 0.460 | aligned |

Every relevant median exceeds the preregistered +0.25 aligned threshold, and
negative fractions are far below 50%. Water's event gradient is not only
aligned; on the lexical path it is 43.4% of the ordinary bodily gradient norm.

### Fresh unbalanced control

The control is also aligned:

| Segments | Global cosine | Lexical cosine | Lexical event/base norm |
|---|---:|---:|---:|
| All | +0.761 | +0.639 | 0.248 |
| Water present | +0.763 | +0.601 | 0.216 |
| Food present | +0.766 | +0.684 | 0.259 |

The bound selector increases effective lexical signal and slightly lowers
alignment, especially for water, but never approaches conflict under the
locked rule.

## What is ruled out

- Gradient cancellation is not the residual blocker.
- A gradient-projection or surgery method is not licensed.
- Gross lexical gradient starvation is not the blocker.
- The earlier raw-norm-only diagnosis is now complete: the signal is both
  substantial and directionally useful.

Per the locked stop rule, the next audit is input identifiability. If the exact
transition features on lived bound consumes already distinguish food versus
water outcomes, the remaining amplitude shortfall is an objective-equilibrium
problem. Only then could a one-shot analytically gradient-matched scale test be
preregistered; a grid remains prohibited.

No external compute or generated data is justified.

## Claim boundary

This measures optimization interaction inside a grounded learned body model.
It is not reflection, self-report, generated language, consciousness, or
subjective experience.
