# Split drift/event bodily head preregistration

Date: 2026-07-25
Status: locked before the architecture is changed and before any output of the
changed architecture is read.

## What the previous experiment settled, and what it left

`2026-07-25-metabolic-drift-supervision-result.md` established two facts that
together make this experiment worth running and make its outcome interpretable
either way.

1. **The forecast repair works and is sufficient for the endpoint.** Urgent-need
   survival through the planner's ten-tick chain went from 18.00% to 100.00%,
   and gate 6 — failed by three successive planners at 19.11%, 18.00% and
   18.00% — came in at **87.11%**.
2. **The same intervention destroys the lexical memory**, monotonically, from
   100.00% cross-round reuse to 63.33%, 46.11% and 9.89% as the weight rises.
   Three `drift_weight = 0` retrains all sit at exactly 100.00%, so this is an
   attribution, not retraining variance.

The two objectives compete. They are emitted through one output,
`need_deltas = 0.5 * tanh(next_needs(h))`, which must simultaneously represent
a metabolic drift of 0.014 and a consumption jump of 0.5. Those demand
incompatible operating regimes of a saturating nonlinearity, and the dense
objective wins because it is present in every transition.

This is not a retune of the rejected term. The loss, its weight, its threshold
and its scale are all held at the values the previous locked rule selected. The
single manipulated variable is the **output path**.

## Mechanism

The bodily-delta head gains a second, deliberately range-limited path:

```python
need_deltas = (
    DRIFT_REGIME_THRESHOLD * mx.tanh(self.drift_needs(h))   # new
    + 0.5 * mx.tanh(self.next_needs(h))                      # unchanged
)
```

- The drift path is bounded by **0.175**, the same measured gap in the delta
  histogram that defines the drift regime in the loss. It can express any
  metabolic drift the world produces and *cannot* express a consumption jump,
  so the dense objective has somewhere to live that is structurally incapable
  of overwriting the sparse one.
- The event path keeps its existing form and range exactly, so the consumption
  predictions the lexical result depends on are architecturally untouched.
- Both paths read the same latent `h`. Competition in the trunk is not claimed
  to be removed — only the output-level interference, which is where a
  saturating nonlinearity does the damage.

`split_drift_head = False` is the default and constructs no extra layer, so
every sealed checkpoint loads and reproduces unchanged.

Everything else is held at the sealed probe24 seed-1 values: hidden size 64,
optimizer, learning rate, entropy, reward shaping, replay, the 30,000-tick
budget, the planner, the utility rule, the settling count, the reuse count, the
planning scale, and `bodily_drift_loss_weight = 0.01`.

## Runs

Exactly three, all at the identical configuration:

1. `split_drift_head` on, `drift_weight = 0.01` — the experimental condition.
2. `split_drift_head` on, `drift_weight = 0.0` — isolates the architecture from
   the supervision, so a change cannot be attributed to the wrong one.
3. The two existing `drift_weight = 0.0`, unsplit retrains serve as the
   baseline; they are already run and both sit at 100.00% reuse.

## Locked gates, judged conjunctively

All five must hold **on the same checkpoint**. No checkpoint in this project's
history has held P2 and P4 at once; that conjunction is the point.

- **P1 forecast.** `urgent_index_survives_predicted_post_return` >= **90%**.
  Unsplit baseline 18.00%; drift term alone 100.00%.
- **P2 memory.** Cross-round label reuse in each of rounds 3-8 >= **90%**.
  Unsplit baseline 100.00%; drift term alone 60.00-68.00%.
- **P3 controls.** Acute silence and acute write suppression each <= **45%**.
- **P4 endpoint.** Protocol-branch feasibility gate 6 >= **60%**.
  Unsplit baseline 18.00%; drift term alone 87.11%.
- **P5 battery.** Feasibility gates 1, 2, 3, 4, 5 and 7 all pass at their
  original preregistered thresholds, unchanged.

### On M1 and M2

The previous preregistration's M1 (resource drift bias < 0.01) and M2 (worst
per-need absolute error < 0.05) are **not** entry conditions here. They were
written as sufficient conditions for M3 derived from the measured 0.2000
urgency margin, and the completed experiment showed them to be neither
necessary nor sufficient: M3 reached 100.00% with M1 at 0.0567 and M2 at
0.0800, because the residual error is concentrated in the return step and is
proportional across needs, so the ordering the utility reads survives.

They remain **reported diagnostics** in every run. This demotion is decided now,
before any output of the new architecture is seen, and is justified by evidence
already in the record rather than by convenience.

## If everything passes

Then and only then the behavioral pair: one fresh matched write-enabled /
write-disabled training pair at the same configuration with the planner
activated after 15,000 ticks, judged by the nine unchanged promotion gates of
`2026-07-25-persistent-childhood-learning-preregistration.md`. Seed 2 and
freshly trained silent and shuffled controls follow that, and nothing is
claimed about generality until they run.

## Stop rule

If P1 and P2 cannot hold on the same checkpoint, the tension is architectural
at this capacity and no further output-path variant is to be tried. Do not
widen the weight grid, do not move the 0.175 bound, do not add a planner, and
do not weaken any gate. The conclusion in that case is that a 64-unit shared
core cannot carry both a dense metabolic model and a sparse lexical memory, and
the next step is the capacity and budget increase that this project has so far
never had grounds to request — which would be the first justified compute
request in its history, and must be argued from this measured tension rather
than from ambition.
