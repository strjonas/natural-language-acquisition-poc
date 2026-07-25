# Binding/consequence geometry preregistration

Date: 2026-07-25
Status: locked before checkpoint geometry is measured.

## Question

Probe39's consumption-scoped loss improves bound food replay error by 32.21%
but water by only 22.08%. After 30k ticks, food's selected lexical transition
feature grows to norm 1.266 while water remains at 0.815. This audit asks
whether the remaining asymmetry originates in the learned lexical value or
downstream in its mapping to bodily consequences.

No environment is stepped and no parameter is updated.

## Fixed checkpoints

1. Probe38 fresh zero-event control.
2. Probe38 generic-event treatment.
3. Probe39 consumption-scoped treatment.

## Measurements

For the exact controlled childhood packets `("this", "food")`,
`("this", "water")`, and `("this", "danger")`:

- extract the 16-dimensional value written by
  `tanh(binding_value(token_encoder(packet)))`;
- report each norm, every pairwise Euclidean distance and cosine similarity,
  the minimum-to-maximum pairwise-distance ratio, and singular values after
  centering the three values;
- report the transition layer's binding-column Frobenius norm and singular
  values, using the exact columns supplied to the transition MLP.

Then isolate the full nonlinear downstream map at 300 fixed, learner-visible
three-object choice observations. For every visible slot, hold the recurrent
core, body, geometry, selected surface, and consume action fixed; intervene
only on that surface's external memory value with each of the three controlled
packets. Measure immediately and after the sealed three padding-only settling
observations:

- the 3-label by 4-need predicted-delta matrix;
- for food, the food-label food-delta minus the mean of the water/danger-label
  food-deltas;
- for water, the analogous water-label water-delta contrast;
- each contrast divided by the corresponding pairwise lexical distance;
- the centered consequence matrix singular values; and
- the fraction of contexts in which the intended label has the largest
  predicted delta on its named need.

This is representation geometry only. It does not compare against simulator
outcomes, choose an action, run the terminal calibration gate, or update a
model.

## Locked interpretation

- **Lexical collision:** minimum/maximum pairwise lexical distance < 0.25 or
  any cross-kind cosine similarity > 0.95.
- **Downstream collapse:** lexical collision is absent, but the centered
  consequence matrix's second singular value is < 10% of its first or either
  named-need intended-label rate is < 75%.
- **Water-specific gain loss:** collision and downstream collapse are absent,
  but the settled water contrast per lexical-distance is < two thirds of the
  food value on the scoped checkpoint.
- **No geometric defect:** none of the above. In that case the measured
  1,256-versus-952 bound-target exposure asymmetry becomes the next causal
  candidate; do not alter the representation.

The same classifications are reported for all three checkpoints. A defect
must be specific to or worsened by the scoped treatment to explain probe39;
a shared defect can explain absolute compression but not the new
food/water divergence.

## Stop rule

Do not change architecture, sampling, loss weight, or budget in this audit.
Whichever stage first collapses decides the next intervention. If geometry is
healthy, the next experiment may balance bound food/water consumption exposure
using only public body context and action metadata, with an independently
trained unbalanced control.

No external compute or generated data is justified.

## Claim boundary

This audit measures whether acquired symbols remain distinct and causally
separable inside a learned bodily transition model. It is not reflection,
self-report, generated language, consciousness, or subjective experience.
