# Self-model-guided option planning preregistration

Date: 2026-07-19
Goal-level metrics: unsupported adult survival and causal use of predicted self.

## Mechanism

For every currently available primitive or perceived-object option, the shared
action-consequence head predicts the organism's next four bodily needs and
environment reward. A detached score (minimum predicted need plus 0.5 times
predicted reward) biases policy logits. Detachment is binding: policy gradients
cannot improve an action by making its predicted consequence falsely
optimistic; the world-model losses must fit experienced outcomes.

Planning begins only when caregiver support reaches zero, after the first 100k
world ticks have trained the consequence head. Invalid object slots are masked
in both acting and policy loss. This corrects a probe-7 bug where absent slots
silently acted as `wait` and could receive unconstrained optimistic predictions.

## Evidence before the full probe

On development seeds, applying the corrected availability mask to the saved
probe-7 checkpoint made scale-6 planning improve lifespan from 61.9 to 71.7 and
successful resource uses from 5.45 to 9.0, though it remained water-skewed. A
fresh 64-hidden/20k smoke with planning activated at 10k achieved 10% unsupported
survival at a 400-step horizon, 14.9 successful resources (7.1 food, 7.8 water),
versus zero survival in every earlier organism smoke.

## Probe 8

- Grounded, seed 1, hidden 256, 200k actual environment ticks.
- Proximal-to-distal caregiver childhood: threshold 0.75 and distance 0 -> 2,
  both ending at 100k.
- Kind-blind visible-slot consume options with duration-aware GAE.
- Planning scale 6, reward weight 0.5, activated at 100k.
- Held-out unsupported adult evaluation: 20 lives x 1000 ticks, fixed seeds.

Primary pass: nonzero 1000-step adult survival and improvement over corrected
option-only control. Required follow-up before any language claim: grounded,
silent, shuffled, plus self-model score interventions (zero/shuffle/reverse)
on matched seeds. A result that survives without predicted-needs information is
motor abstraction, not self-model-guided control.

## Result

The full probe missed the primary gate: 0/20 held-out adult lives survived
1,000 ticks. Mean lifespan was 91.85 ticks with 12.55 successful resource uses
(5.20 food, 7.35 water). Removing the planning scores reduced mean lifespan to
78.65 but left survival at zero. Reversing the scores did not degrade behavior:
mean lifespan was 94.0 with 14.0 successful resource uses. Consequently the
planning head is not accepted as causally useful.

A diagnostic counterfactual audit found -0.055 within-state correlation between
predicted and actual action scores over 2,001 outcomes. The next probe targets
this action-ranking failure; see `2026-07-19-residual-self-model-preregistration.md`.
