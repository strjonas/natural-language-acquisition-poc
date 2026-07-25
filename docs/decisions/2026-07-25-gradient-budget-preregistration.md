# Gradient budget preregistration

Date: 2026-07-25
Status: locked before the clip is changed and before any output of the changed
clip is read.

## The diagnosis this tests

Two experiments have now shown the same tension, and two candidate explanations
for it have been refuted by their own controls.

| Attempt | Forecast | Lexical memory |
|---|---:|---:|
| Sealed baseline | 18.00% | 100.00% |
| Drift term, shared head, hidden 64 | 100.00% | 63.33% |
| Drift term, **split head**, hidden 64 | 100.00% | 57.83% |
| Split head, **no drift term** (control) | 50.33% | **100.00%** |
| Drift term, split head, hidden 128 (diagnostic) | 100.00% | 50.28% |
| Drift term, split head, hidden 256 (diagnostic) | 100.00% | 54.39% |

- **Not the output head.** Giving metabolism its own range-limited path did not
  help, and the architecture-only control keeps the memory at 100.00%, so the
  extra path is harmless on its own.
- **Not capacity.** Four times the hidden units leaves the memory at 50-54%.
  This refutes the premise of the previous stop rule, which pointed at a
  capacity increase; that request is **not** justified and is withdrawn.

A direct measurement of the optimizer identifies the actual coupling. Raw
gradient norm over a matched 3,000-tick run, before clipping, against the
repository's `max_grad_norm = 1.0`:

| Condition | Median norm | p90 | Updates clipped |
|---|---:|---:|---:|
| Baseline | 1.271 | 4.541 | 59.4% |
| Drift term | 7.500 | 20.208 | **100.0%** |
| Drift term + split head | 8.235 | 20.598 | **100.0%** |

The drift term inflates the raw gradient roughly sixfold, so **every** update
saturates the clip and every other objective is rescaled down with it. Global
norm clipping is a shared budget. The lexical mapping is learned from about two
events per life, and shrinking its effective step by that factor over a fixed
30,000-tick budget is a sufficient account of the degradation.

This explains all three observations at once, including why neither head
separation nor capacity helped: clipping is global, so it is neither
representational nor per-head.

## Mechanism

`max_grad_norm` is raised from **1.0 to 10.0**, above the p90 of the observed
raw norms, so the trust region stops binding on essentially every update. That
is the single manipulated variable. The loss, its weight (0.01), its threshold
(0.175), its scale (0.02), the split head, the architecture, hidden size 64,
the optimizer, the learning rate, the budget and the planner are all held.

Raising the clip is the direct test of the coupling rather than a way around
it. Reducing `bodily_drift_loss_weight` until the norms match would test the
same hypothesis, but the weight grid is closed by a previous locked stop rule
and is not reopened.

## Runs

1. Split head, `drift_weight = 0.01`, `max_grad_norm = 10.0` — experimental.
2. Unsplit, `drift_weight = 0.0`, `max_grad_norm = 10.0` — control that the
   raised clip alone does not damage the memory. This matters because the
   baseline was itself clipped on 59.4% of updates, so the clip change is not
   inert for the sealed configuration.

## Locked gates, judged conjunctively on run 1

Unchanged from the previous preregistration.

- **P1 forecast.** `urgent_index_survives_predicted_post_return` >= **90%**.
- **P2 memory.** Cross-round reuse in each of rounds 3-8 >= **90%**.
- **P3 controls.** Acute silence and acute write suppression each <= **45%**.
- **P4 endpoint.** Protocol-branch feasibility gate 6 >= **60%**.
- **P5 battery.** Feasibility gates 1, 2, 3, 4, 5 and 7 pass at their original
  thresholds.

Run 2 must show cross-round reuse >= 90%; if the raised clip alone breaks the
memory, the manipulation is confounded and run 1 proves nothing either way.

## Stop rule

If P1 and P2 still cannot hold together with the trust region removed, then the
tension is not the output path, not capacity, and not the gradient budget, and
all three of the cheap explanations are exhausted. In that case stop
manipulating the optimizer and record that a fixed 30,000-tick online budget
cannot serve both objectives, with the measured gradient-norm evidence as the
argument for a longer budget rather than a bigger model. Do not widen any grid,
do not add a planner, do not weaken any gate.
