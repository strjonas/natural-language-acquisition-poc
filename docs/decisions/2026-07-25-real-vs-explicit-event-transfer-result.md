# Real-versus-explicit bodily-event transfer result

Date: 2026-07-25
Decision: **lived-event failure**. The event-loss treatment is already severely
compressed on the real, in-distribution acquired-label path; explicit
counterfactual transfer is not the primary defect.

## Integrity

The three paths, matched public state, fixed checkpoints and seeds,
measurements, interpretation thresholds, and stop rules were committed in
`2026-07-25-real-vs-explicit-event-transfer-preregistration.md` before the audit
was implemented or run.

Checkpoints: the exact fresh zero control and 0.01 event treatment from
probe36. Artifacts: `runs/organism/probe37_real_explicit_event_transfer/`.

No parameter was updated. All 79 organism tests pass.

## Treatment result

Predicted demanded-minus-best-wrong margin divided by the exact 0.4000 realized
margin:

| Path | Food | Water | Overall choice |
|---|---:|---:|---:|
| Real immediate label + return | **4.18%** | **16.32%** | 100.00% |
| Real label, matched three reads | **4.08%** | **8.83%** | 97.33% |
| Explicit write, matched three reads | **3.89%** | **6.11%** | 88.33% |

Both real paths are below the locked 25% lived-event-failure threshold for both
resources. The classification is unambiguous.

The real immediate path is maximally in-distribution: the organism processes
the label packet produced by its own real inspect action and then the real
six-tick return packet. Even there the event term leaves food at a 0.0167
predicted margin and water at 0.0653, against 0.4000 realized.

## Explicit transfer is a secondary loss

After matching both paths to three post-write reads:

| Demand | Real settled margin | Explicit settled | Difference | Real larger |
|---|---:|---:|---:|---:|
| Food | 0.0163 | 0.0155 | +0.0008 | 47.48% |
| Water | 0.0353 | 0.0244 | +0.0109 | 88.82% |
| Overall | 0.0265 | 0.0203 | +0.0062 | 69.67% |

The explicit write does lose some water magnitude, but it fails the
preregistered counterfactual-transfer criterion: the real path is nowhere near
calibrated, the paired rate is below 75% overall, and food has essentially no
transfer gap. Repairing explicit transfer could not close a fourteenfold lived
event deficit.

## Fresh control

The zero-weight control reproduces compressed explicit margins with at least
90% choice for both resources, as required. Its real immediate path is also
compressed:

| Path | Food ratio | Water ratio |
|---|---:|---:|
| Real immediate | 4.87% | 26.06% |
| Real settled | 4.50% | 9.75% |
| Explicit settled | 4.09% | 6.97% |

The event treatment does not improve any of those six ratios.

## What is ruled out

- Return-origin aliasing is already refuted by probe34.
- Terminal compression is not created mainly by the explicit-write branch.
- More post-write settling does not recover magnitude; it reduces it.
- A per-entry, scale-normalized loss on lived events does not make those same
  real event states accurate.
- Forecast and persistent storage remain intact, so this is not global model
  collapse.

## Exact next question

The next diagnostic must inspect what the event loss actually receives:

- count event entries and unique label/body/kind contexts in fresh training;
- separate resource-restoration and poison/health events;
- measure how often a resource event occurs with a valid binding for the
  consumed surface and how often its lexical value has reached the recurrent
  core;
- measure event-term gradient mass before and after masking; and
- compare event prediction error on the training/replay distribution with the
  held-out real-immediate path.

This distinguishes target scarcity, conditioning failure, replay imbalance,
and optimization failure. It must be instrumentation first. The rejected 0.01
weight is not increased or swept.

No compute or data request is justified.

## Claim boundary

This is a controlled localization of bodily event-value failure along real and
hypothetical lexical state paths. It is not reflection, self-report, generated
language, or consciousness.
