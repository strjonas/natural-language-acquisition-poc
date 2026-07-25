# Cross-need gradient geometry result

Date: 2026-07-25
Decision: **integrity gate fails; no geometry result**.

## Integrity result

The audit and its 128 paired-segment / 20,000-tick requirement were committed
before execution. On the probe45 uniform-control checkpoint, the fixed policy
produced only **20** segments containing both learner-visible valid-bound food
and water restorations in 20,044 primitive ticks.

This fails the first locked integrity gate by a wide margin. The treatment
checkpoint was therefore not run, the 20 control samples are not interpreted,
and no gradient projection, parameter separation, or sensitivity mechanism is
licensed.

Artifacts: `runs/organism/probe46_cross_need_gradient_geometry/control/`.

## What failed

The objective and gradient implementation passed its finite two-segment smoke
test and the repository passed 271 tests plus 13 subtests. The failure is the
sampling unit: requiring both resource outcomes inside one fresh fixed-policy
segment is too rare at mature-policy event rates. Extending the collection
budget after seeing this result would violate the locked design.

The correct replacement is a population task-gradient comparison: collect
food-present and water-present lived segments independently, compute the
resource-specific bound objective on each, average gradients within resource,
and compare the two population gradients. Multi-task gradient methods compare
task minibatches in exactly this way; same-segment co-occurrence is not a
requirement. The replacement must be separately preregistered and must not
reuse these 20 samples as evidence.

## Claim boundary

This is a failed diagnostic design, not evidence for or against cross-need
gradient conflict. It changes no training claim and licenses no compute or
data request.
