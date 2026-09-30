# Probe75: disjoint-band persistent-self replication — preregistration

Date: 2026-09-30. Written before implementation of the replication runner or
generation of its identities/episodes. Parent: Probe74, which passed six
construction gates but explicitly licensed only an independent replication.

## Question and fixed intervention

Does retaining evidence about the same stationary individual body improve
survival on a new population and episode stream? Reuse Probe74's exact
`run_episode`, recursive estimator, four rate-only cells, cyclic donor control,
one reset-rate calibration episode and five test episodes. The parent weights,
state reset, uptake reset, ledger, movement estimate, ecology, duration,
interoception probability, help clock and word rule remain identical.

Five paired blocks × 28 identities × five scored episodes: 700 episodes per
cell, 140 calibration episodes excluded from endpoints. Blocks are the interval
units. All cells compute all three estimators; relevant evidence history, not
per-tick compute, differs. Report actual ticks/readings as well as nominal
400-tick reset and 2,400-tick retained budgets.

## Seed bands and reproducibility

Diagnostic/timing identities start at 3,400,000,000 and episodes at
3,500,000,000. Confirmation identities start at **3,600,000,000** and episodes
at **3,700,000,000**, using Probe74's unchanged 2,000,000 block stride and
16 episode stride. These bands are disjoint from Probe74's 3.2B/3.3B bands,
Probe73's 3.0B lives, and the 3.1B diagnostic. Derived motor offsets must fit
unsigned 32-bit seeds. The new runner must reject overlapping identity/episode
bands and pin their values in resumable configuration. Probe74 defaults remain
unchanged.

Time a two-identity, one-test-episode diagnostic before the full run. This is a
runtime/inertness check, not a gate or pilot for selecting a seed band. Quote
the full-run duration using that measured cost, accounting for five test
episodes and process overhead. Run one entire paired block before the next.
Save complete calibration and identity/cell units atomically.

## Locked confirmation criteria

Reuse the **exact Probe74 `grade` function**, with all six clauses unchanged:

1. C1: true-rate minus reset-rate survival positive.
2. C2: retained-rate minus reset-rate survival positive.
3. C3: reset-word minus retained-word regret positive on retained histories.
4. C4: reset minus retained birth-rate MAE positive on retained histories.
5. C5: donor minus own birth-rate MAE positive on retained histories.
6. C6: mean true-rate survival at least 0.15.

For C1–C5 positive means a positive mean, two-sided 95% Student-t interval
excluding zero over exactly five blocks, and at least four positive blocks.
Also report retained minus swapped survival, per-block and per-episode results,
and true minus retained survival without gating them. No survey/confirmation
pooling to rescue a failure. No sample increase, ecology change, calibration
extension or threshold amendment after the confirmation starts.

## Failure rule and claim boundary

Any failed clause means this exact confirmation fails. State the failed claim
plainly and reconsider the route using the measured belief and decision
endpoints. Even a full pass establishes replication across new bodies and
episode streams under **one shared frozen lexical/motor parent**. It does not
replicate parent development, discover axes, recognize identity, establish
saturation, model bodily change, drive the motor policy from the self-model,
report uncertainty or acquire generative language. Persistent bodies are
simulated episodic resets, including after death. The full repository vision
remains unfinished whatever this replication finds.
