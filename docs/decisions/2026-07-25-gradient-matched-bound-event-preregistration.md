# Gradient-matched bound-event calibration preregistration

Date: 2026-07-25
Status: locked before weight 0.0231 is trained.

## Derivation

Probe43 finds perfect full and lexical-only identifiability on 1,000 lived
bound events, with predicted/target magnitude only 40.47% food and 28.20%
water. Probe42 finds aligned water-present lexical gradients and an
event/base norm ratio of 0.4336 at bound weight 0.01.

The event gradient is linear in its scalar weight at a fixed checkpoint.
Matching its median water-present lexical norm to the ordinary bodily gradient
therefore gives:

`0.01 / 0.4336 = 0.02306`.

The single locked treatment value is **0.0231**. It is derived, not selected
from outcomes. No neighboring value or grid will be run.

## Locked runs

Three fresh seed-1 exact probe41 runs:

1. **Zero control:** bound-event weight 0.
2. **Reference:** bound-event weight 0.01.
3. **Gradient-matched treatment:** bound-event weight 0.0231.

Every other setting is identical: 60,000 ticks, hidden 64, 64-decision
segments, three-object 40-tick childhood for all ticks, six-tick returns, eight
rounds, low need 0.55, consume/inspect options, binding 16, horizon-two model
loss at weight 1, replay 256 with one update, drift 0.01, split drift head, and
clip 10. Training-path instrumentation remains enabled.

The zero control must retain at least 90% forecast ordering and 90% reuse in
rounds 3-8 if later behavioral gates are reached. The reference is the
paired retraining-noise comparator for the scale effect.

## Locked gates

1. **Training calibration.** Treatment final-replay valid-bound food and water
   MAE must each be < 0.15 and at least 20% below the fresh 0.01 reference.
2. **Balanced online learning.** From first to last complete 10,000-tick
   window, both food and water bound MAE must improve by >= 0.15 and their last
   window gap must be <= 0.03.
3. **Real terminal calibration.** If gates 1-2 pass, on 300 fixed contexts the
   predicted demanded-minus-best-wrong margin must be >= 25% of the exact
   realized 0.4000 margin for food and water separately, with demanded-choice
   accuracy >= 90% for each.
4. **Forecast and memory preservation.** If gate 3 passes, ten-tick
   post-return need ordering must be >= 90%; reuse in every round 3-8 >= 90%;
   acute silence and write suppression each <= 45%.
5. **Feasibility.** Only if gates 1-4 pass, run and report the unchanged
   seven-gate protocol battery.

## Stop rule

- If gate 1 fails, the analytic scale match is rejected and no weight sweep is
  allowed.
- If gate 1 passes but gate 2 fails, endpoint fit is not balanced continual
  learning and is rejected.
- If training fits but terminal transfer fails, objective scale is no longer
  the blocker.
- Any forecast or memory regression rejects promotion.

Seed expansion, write-disabled training pairs, generated language, larger
models, and external compute remain blocked until all five gates pass.

## Claim boundary

This tests an analytically scaled, grounded bodily consequence objective. It is
not reflection, self-report, generated language, consciousness, or subjective
experience.
