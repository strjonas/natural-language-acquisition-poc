# Bound-event gradient-alignment preregistration

Date: 2026-07-25
Status: locked before final-checkpoint gradient directions are measured.

## Question

Probe41's bound event term has 36.91% of the base bodily gradient norm on the
lexical path yet leaves water underfit. Norm cannot distinguish a useful
auxiliary signal from one canceled by the ordinary objective. This audit
measures their direction on fresh, fixed-policy embodied segments.

## Fixed checkpoints

1. Probe41 fresh unbalanced consumption-scoped control.
2. Probe41 bound-balanced treatment.

For each checkpoint, reconstruct its exact training environment and collect
fresh seed-locked semantic-choice lives until 160 segments containing at least
one valid-bound resource restoration have been measured or 20,000 primitive
ticks have elapsed. No optimizer update, replay insertion, or parameter
mutation occurs.

## Measurements

On each selected segment, compute from identical model state and targets:

- gradient of the sealed change-boosted bodily objective, including its
  horizon-two term;
- gradient of that checkpoint's already-weighted event objective, including
  its horizon-two term;
- cosine similarity and norm ratio globally;
- cosine similarity and norm ratio on
  `token_embedding`/`binding_value`/`binding_read`;
- the same lexical cosine grouped into food-bound only, water-bound only, and
  segments containing both.

The simulator event name is used only after collection to group a segment; it
does not enter either objective. Raw cosines and group counts are retained.

## Locked interpretation

For global and lexical gradients separately:

- **Direct conflict:** median cosine < 0 or more than 50% of samples are
  negative.
- **Weak alignment:** median cosine is in [0, 0.25).
- **Aligned:** median cosine >= 0.25.

The water-specific classification uses water-containing segments. If overall
gradients are aligned but water-only lexical gradients are directly
conflicting, water cancellation is the blocker. If water gradients are
aligned and the event/base lexical norm ratio remains >= 0.25, neither
direction nor gross scale explains the residual; inspect target
identifiability at the transition input rather than tuning.

## Stop rule

This audit cannot license a weight sweep. Direct conflict licenses a
preregistered projection or objective-separation test with a nonconflicting
control. Aligned, adequately sized gradients rule that out and require a
read-only input-identifiability audit. Weak alignment is reported as
indeterminate; collect more local segments before any intervention.

No larger compute or generated data is justified.

## Claim boundary

This measures optimization interaction inside a grounded bodily model. It is
not reflection, self-report, generated language, consciousness, or subjective
experience.
