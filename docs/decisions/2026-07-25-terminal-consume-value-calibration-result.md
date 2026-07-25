# Terminal consumption-value calibration result

Date: 2026-07-25
Decision: terminal bodily value is **severely compressed**. The organism ranks
the demanded resource correctly in 99.67% of contexts but predicts only 6.91%
of its realized advantage over the best wrong object.

## Integrity

The audit construction, fixed checkpoint and seeds, measurements, compression
criterion, and stop rule were committed in
`2026-07-25-terminal-consume-value-calibration-preregistration.md` before the
diagnostic was implemented or run.

Checkpoint, unchanged: probe33 60,000-tick seed 1. Artifact:
`runs/organism/probe35_terminal_consume_calibration/`.

No parameter was updated. All 76 organism tests pass.

## Result

At the exact post-return state used by the protocol branch, with a matching
word written to the demanded-resource surface:

| Quantity | Predicted | Realized | Predicted / realized |
|---|---:|---:|---:|
| Demanded-resource score | 0.4979 | 0.7799 | — |
| Other-resource score | 0.4674 | 0.3799 | — |
| Poison score | 0.4676 | 0.3799 | — |
| Demanded minus best wrong | **0.0276** | **0.4000** | **6.91%** |
| Demanded minus other resource | 0.0305 | 0.4000 | 7.63% |
| Demanded minus poison | 0.0303 | 0.4000 | 7.57% |
| Chooses demanded resource | **99.67%** | 100.00% | — |

Mean absolute error over the three terminal scores is 0.1525. The predicted
demanded-minus-best-wrong margin is smaller than realization in **300/300**
paired contexts.

## Resource split

| Demand | Contexts | Predicted margin | Realized margin | Ratio | Predicted choice |
|---|---:|---:|---:|---:|---:|
| Food | 139 | 0.0191 | 0.4000 | **4.78%** | 99.28% |
| Water | 161 | 0.0350 | 0.4000 | **8.74%** | 100.00% |

Terminal-score MAE is 0.1771 for food and 0.1312 for water.

The preregistered compression criterion passes all four clauses:

- realized margins are positive for both resources;
- predicted demanded-resource choice exceeds 90% for both;
- both predicted margins are far below 75% of realization; and
- prediction is smaller in 100% of contexts, above the 75% threshold.

## What this explains

The prior feasibility battery looked contradictory only because ranking and
magnitude were conflated:

- Gate 6 reads ranking and reaches 99.78%.
- Gates 2-4 read instrumental magnitude. A word contributes only +0.0466
  because the model predicts almost no bodily difference between the correct
  and wrong terminal actions.
- The demanded-resource sign split is then dominated by a larger
  write-independent offset: +0.0563 for food and -0.0974 for water under the
  no-write control.

The lexical mapping is not weak. It changes the selected object almost
perfectly. The learned self-consequence attached to that selection is weak by a
factor of roughly fourteen.

## Causal localization

Within a context, the predicted return body is common to all three terminal
actions. It cancels from the demanded-minus-wrong margin. The compression
therefore cannot be repaired by another return forecast or branch origin. It is
in the binding-conditioned terminal consumption delta itself.

Learning already sees the lived post-consumption body, but the current loss
averages an event on one need with three drift entries and then normalizes
across a stream dominated by non-events. The scale-balanced drift correction
fixed the opposite regime. The evidence now licenses the symmetric question:
can a per-entry, scale-normalized term make rare bodily events accurate in
magnitude without losing forecast or lexical persistence?

That intervention requires its own preregistration and a freshly trained zero-
weight control because training is not bit-reproducible.

No branch, utility, reuse count, gate, or compute budget is changed. No compute
or data request is justified.

## Claim boundary

This establishes a calibrated distinction between knowing which embodied
action is right and accurately predicting how much that action will change the
body. It is not reflection, self-report, generated language, or evidence of
consciousness.
