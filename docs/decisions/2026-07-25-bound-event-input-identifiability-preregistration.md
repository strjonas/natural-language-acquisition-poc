# Bound-event transition-input identifiability preregistration

Date: 2026-07-25
Status: locked before lived transition inputs are collected.

## Pre-data amendment

Locked before implementation and before any lived transition input was
collected: the originally written lexical-identification gate also required
full-probe accuracy to exceed the no-direct-binding probe by 10 points. That is
not a valid lexical ablation in this architecture. The recurrent core has
already read the external binding before the transition input is formed, so
removing only the direct binding columns leaves a second lexical path intact.
The no-direct-binding probe remains reported descriptively, but only
lexical-only accuracy decides lexical identifiability. A future causal ablation
would have to remove both the bank and all of its prior recurrent reads.

## Question

The scoped checkpoint has healthy controlled lexical/consequence geometry
(probe40), and its water event gradient is aligned and substantial (probe42).
The remaining representational possibility is that actual lived bound-consume
inputs alias food and water at the transition MLP.

This audit collects fresh fixed-policy experience from the probe41
bound-balanced checkpoint and updates nothing.

## Data and exact representation

Collect up to 20,000 primitive ticks or 1,000 valid-bound food/water
restorations. For each selected transition, retain exactly the vector supplied
to the first transition layer:

- recurrent core state;
- public action one-hot;
- selected learner-visible object features;
- selected lexical binding value and validity.

The target class is which own-body dimension has the above-threshold positive
delta (food or water). This is read from the same lived body target already
used by the loss, never hidden object kind or simulator event metadata.

## Fixed offline probes

Use a deterministic chronological split: every fifth example is held out.
Standardize from the training partition only and fit ridge-regularized
least-squares two-class probes (`lambda = 1e-3`) on:

1. the full transition input;
2. the lexical binding features alone;
3. the transition input with lexical binding features removed.

Also report leave-one-out nearest-neighbor class agreement in standardized full
input, class counts, within-class target-delta mean/standard deviation, and the
checkpoint's own predicted named-need delta and attenuation
(`predicted / target`).

These probes are analysis only and are not installed in the organism.

## Locked interpretation

- **Input collision:** full held-out accuracy < 90% or nearest-neighbor
  agreement < 80%.
- **Lexical value not identifying:** lexical-only accuracy < 90%. The
  full-minus-no-direct-binding difference is descriptive only for the
  read-before-write recurrent architecture explained above.
- **Identifiable but attenuated:** both identification gates pass, while the
  checkpoint's mean named-need prediction is < 75% of the target for either
  resource.
- **Input and amplitude adequate:** identification passes and both attenuation
  ratios are >= 75%.

If inputs are identifiable but attenuated, no representation or sampling
change is licensed. A subsequent treatment may set one event weight
analytically so its median water-present lexical gradient norm matches the base
norm (using probe42's ratio), with one fresh zero-path control and no grid.

## Stop rule

Input collision sends the next intervention to the earliest failing feature
block. Lexical non-identifiability sends it to binding acquisition. Identifiable
attenuation sends it to the declared one-shot scale test. Do not weaken these
thresholds or train the offline probe into the agent.

No larger compute or generated data is justified.

## Claim boundary

This tests whether a grounded memory makes lived bodily outcomes identifiable
at a learned transition input. It is not reflection, self-report, generated
language, consciousness, or subjective experience.
