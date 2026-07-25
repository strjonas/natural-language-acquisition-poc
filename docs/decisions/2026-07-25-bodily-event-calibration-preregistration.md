# Bodily event-calibration preregistration

Date: 2026-07-25
Status: locked before changing the loss or training either condition.

## Result selecting the intervention

On the unchanged 60,000-tick checkpoint, a matching demanded-resource word
makes the organism choose its labeled target in 99.67% of 300 terminal
calibration contexts. The simulator-scored demanded-minus-best-wrong bodily
margin is 0.4000. The self-model predicts 0.0276, or **6.91%** of the realized
margin, and underestimates it in all 300 contexts.

The split is 4.78% for food and 8.74% for water. This is the remaining
feasibility failure's first direct magnitude localization: the persistent word
changes ranking almost perfectly, but the binding-conditioned consumption
consequence is compressed by roughly fourteenfold.

## Mechanism

`bodily_delta_prediction_loss` currently:

1. averages four per-need errors into one transition error;
2. boosts the transition when any need changes sharply; and
3. normalizes across a stream dominated by metabolic non-events.

A resource consumption changes one target need while the other three entries
are ordinary drift. The event error is therefore diluted once within the
transition and again across transitions. The scale-balanced drift term fixed
the complementary failure by selecting individual drift entries and
normalizing them by their physical scale.

Add the symmetric, independently selectable event term:

```text
event_mask = abs(target_delta) > 0.175
event_mse = sum(mask * squared_error) / sum(mask)
loss += event_weight * event_mse / 0.4^2
```

The 0.175 regime boundary is the existing sealed separation between slow drift
and bodily events. The scale 0.4 is not tuned: it is the exact
demanded-minus-wrong terminal margin measured by the simulator in all 300
calibration contexts. The single treatment weight is **0.01**, matching the
dimensionless scale-balanced drift coefficient already selected by the locked
forecast experiment.

Zero must be numerically identical to the current loss and remains the default.

No hidden kind, language target, simulator counterfactual, or direct language
reward enters training. The mask and target come only from the lived
post-action body already used by the world-model loss.

## Fixed training comparison

Train two fresh seed-1 checkpoints:

- **control:** `event_weight = 0.0`;
- **treatment:** `event_weight = 0.01`.

Both use the exact probe33 configuration: 60,000 primitive ticks of three-way,
eight-round semantic childhood; drift weight 0.01; split drift head; global
gradient norm 10; hidden size 64; two-decision world-model loss; replay
256 x 1; episodic binding width 16; and every other optimizer, environment, and
loss setting unchanged.

The fresh zero control is mandatory because training is not bit-reproducible at
a fixed seed. The treatment is read against that control, not only against the
sealed probe33 checkpoint.

## Evaluation order

For each checkpoint, in this order:

1. terminal consumption-value calibration, 300 contexts;
2. metabolic drift forecast, 300 contexts;
3. cross-round reuse with grounded, acute silence, and acute write suppression,
   300 lives;
4. protocol-branch feasibility, 300 contexts, reuse count seven.

Do not read a later endpoint to reinterpret an earlier gate.

## Locked gates

### Control reproduction

- **C1 compression:** demanded-minus-best-wrong predicted/realized ratio is
  below 0.25 for both food and water.
- **C2 semantics:** predicted demanded-resource choice is at least 90% for both
  resources.
- **C3 retained headline:** post-return urgent-index survival and each of
  rounds 3-8 cross-round reuse are at least 90%.

If C1-C3 fail, training variance did not reproduce the starting state. Report
both checkpoints but make no causal attribution.

### Treatment mechanism

- **E1 magnitude:** demanded-minus-best-wrong predicted/realized ratio is in
  [0.75, 1.25] for both food and water.
- **E2 score accuracy:** terminal-score MAE is below 0.05 for both resources.
- **E3 semantics:** predicted demanded-resource choice is at least 90% for both.

### Non-regression

- **F1 forecast:** post-return urgent-index survival is at least 90%.
- **F2 memory:** every cross-round reuse row for rounds 3-8 is at least 90%.
- **F3 acute controls:** silence and write suppression are each at most 45%.

### Existing feasibility battery, unchanged

All seven gates must pass: positive mean intact advantage; at least 75%
positive contexts; write and collapsed-label advantage drops at least 0.05;
branch contingency at least 60%; matching-label target selection at least 60%;
and danger-label avoidance at least 90%.

## Interpretation

- If the control reproduces and E1-E3, F1-F3, and all seven feasibility gates
  pass, the rare event-magnitude correction is accepted and the already-locked
  matched write-disabled behavioral training pair is next.
- If E1-E3 pass but forecast, memory, or feasibility regresses, the correction
  is rejected as an objective conflict; record the exact dissociation and do
  not tune the weight.
- If magnitude remains below 0.75 with the control reproduced, the per-entry
  event loss is insufficient. Record the negative; do not sweep weights.
- If the control does not reproduce, stop at the variance diagnosis.

## Stop rule

One weight, one fresh treatment, one fresh zero control, one seed. No weight
grid, larger budget, architecture change, branch change, reuse-count change,
gate change, seed 2, transfer, or generated speech.

Both local training runs should complete in roughly a minute. No compute or
data-generation request is justified.

## Claim boundary

Passing would show that a word-grounded terminal choice becomes accurately
valuable only when the organism learns the magnitude of its own rare bodily
events, while preserving metabolic forecast and persistent memory. It would not
establish reflection, self-report, generated language, or consciousness.
