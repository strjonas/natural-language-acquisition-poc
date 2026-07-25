# Online budget preregistration

Date: 2026-07-25
Status: locked before the confirmatory battery is run. Declares honestly what
has already been seen.

## What has already been seen, and how

Three cheap explanations for the forecast/memory tension were tested and
refuted by their own controls: the output path (split head, with an
architecture-only control at 100.00% memory), capacity (hidden 128 and 256,
memory 50-54%), and the gradient budget (`max_grad_norm` 10.0, with a control
at 98.89% memory confirming the raised clip is itself harmless).

A labelled diagnostic then varied the one thing every previous run held fixed —
the online experience budget — and found the tension dissolves:

| Budget | Forecast survival | Cross-round reuse | Acute silence |
|---:|---:|---:|---:|
| 30,000 ticks | 100.00% | 62.56% | 31.22% |
| **60,000 ticks** | **100.00%** | **100.00%** | 30.94% |
| 120,000 ticks | 100.00% | 100.00% | 30.72% |

**These numbers were seen before this document was written.** P1 (forecast),
P2 (memory) and P3 (controls) are therefore *not* blind endpoints here and are
reported as diagnostic replications, not as preregistered passes. Saying so is
the point of this section.

What has **not** been run or seen at any budget above 30,000 ticks:

- the protocol-branch feasibility battery, including gate 6;
- the matched write-disabled training control.

Those are the blind endpoints locked below.

## Mechanism

The single manipulated variable is `--train-steps` (and the matching
`--semantic-choice-childhood-steps`), raised from 30,000 to **60,000**. The
smaller of the two working budgets is taken deliberately: 120,000 was not
better on any measured quantity, and the weaker manipulation is the more
conservative claim.

Held fixed at the values earlier locked rules selected: `drift_weight = 0.01`,
`drift_threshold = 0.175`, `drift_scale = 0.02`, the split drift head,
`max_grad_norm = 10.0`, hidden size 64, seed 1, the optimizer, the learning
rate, the planner, the utility rule, the settling count, the reuse count and
the planning scale.

The training is rerun into a committed run directory rather than reusing the
diagnostic checkpoint, so provenance is clean and the diagnostic is
independently replicated by the rerun.

## Locked blind endpoints

On the write-enabled 60,000-tick checkpoint:

- **P4.** Protocol-branch feasibility gate 6 — a branch whose label names the
  demanded resource selects the labeled object — at least **60%**. The sealed
  baseline is 18.00% and three planners failed this gate.
- **P5.** Feasibility gates 1, 2, 3, 4, 5 and 7 all pass at their original
  preregistered thresholds: positive mean intact inspect advantage; at least
  75% positive-advantage contexts; write-suppression and collapsed-label
  advantage drops of at least 0.05 each; at least 60% label-contingent terminal
  choices; and at least 90% danger-label avoidance.

Reported alongside, as replications rather than blind gates: P1 >= 90%,
P2 >= 90% in each of rounds 3-8, P3 <= 45% for both acute controls.

## Interpretation rules, fixed now

- If P4 and P5 both pass, this is the first checkpoint in the project's history
  to hold the forecast, the lexical memory and the whole feasibility battery at
  once, and the immediate next step is the matched write-disabled training pair
  and the nine promotion gates of
  `2026-07-25-persistent-childhood-learning-preregistration.md`. It is **not**
  a claim about seeds, transfer, or generated speech, none of which have run.
- If P4 passes but P5 fails, the forecast repair is confirmed and the branch
  battery is the remaining obstacle; report which gates fail and stop.
- If P4 fails at a budget where P1 and P2 both hold, then the forecast was
  necessary but not sufficient for the planner, which contradicts the oracle
  bound of `2026-07-25-metabolic-drift-supervision-preregistration.md`, and
  that contradiction becomes the next object of study rather than another
  patch.

## Stop rule

One budget, one seed, one rerun. Do not sweep the budget further, do not
reopen the weight grid, the 0.175 bound, the scale, the clip or the
architecture, and do not add a planner. Seed 2, freshly trained silent and
shuffled controls, open-island transfer and generated speech remain blocked
until the write-disabled pair passes its own gates.

A compute request remains unjustified: the working budget is 60,000 ticks,
which runs locally in under a minute.
