# Delayed-choice information-rent upper-bound preregistration

Date: 2026-07-25  
Status: locked after the negative learned-planner feasibility result and before
running the simulator upper bound

## Result selecting this diagnostic

The preregistered 300-context observation-branching audit produced:

- candidate-prior normalized entropy: 0.9069;
- intact branch contingency: 81.22%;
- matching-label target selection: 64.11%;
- danger-label target avoidance: 100.00%;
- intact best-inspect advantage: -0.1264;
- positive-advantage contexts: 0/300;
- no-write branch contingency: 28.44%;
- no-write best-inspect advantage: -0.1255; and
- collapsed-label best-inspect advantage: -0.1167.

Thus the learned lexical memory causes contingent counterfactual choices, but
the inspect action is never predicted to pay its physical cost. The nearly
identical no-write value suggests the dominant failure may be the delayed task
substrate rather than the observation backup.

## Goal-level metric

This diagnostic tests whether truthful information can improve embodied
viability at all under the current delayed-choice mechanics. It is an
audit-only simulator upper bound. No simulator state or counterfactual enters
learning.

## Fixed evaluation

Run 10,000 deterministic fresh three-object contexts beginning at seed
`1_800_000`, with:

- low need 0.55;
- choice horizon 40;
- inspect and consume options of four primitive ticks;
- six-tick fixed padding return after each inspection;
- one food, one water, and one poison;
- grounded minimal labels; and
- final utility equal to the minimum of the four exact bodily needs.

All policies identify objects only by learner-visible surface identity and
read labels only from the public token packet. Hidden kind is used after a
rollout solely to score correctness.

Compare four fixed policies on matched seed sets:

1. **Blind immediate:** consume the first visible surface without inspection.
2. **One inspection:** inspect the first visible surface and return. If its
   public label matches the currently low bodily need, consume it; otherwise
   consume the first remaining visible surface. The remaining pair is
   exchangeable, so no hidden-kind tie-break is permitted.
3. **Two inspections with elimination:** inspect the first surface. If it
   matches the low need, consume it. Otherwise inspect the first remaining
   surface. If that label matches, consume it; otherwise consume the last
   surface by elimination from the known one-food/one-water/one-danger task
   structure.
4. **Clairvoyant immediate ceiling:** consume the correct resource immediately.
   This policy may read hidden kind and is reported only as an unattainable
   ceiling, never as an agent comparison.

Report mean final minimum need, mean primitive ticks, correct/wrong-resource/
poison rates, and paired mean utility differences against blind immediate.

## Decision rule

- If one inspection beats blind by at least 0.02, the learned planner remains
  the primary bottleneck.
- Else if two inspections beat blind by at least 0.02, the current one-label
  backup is too shallow; implement a sequential observation tree.
- Else the delayed substrate does not pay semantic rent even to the exact
  two-inspection Bayes policy. Propagate the negative result upward: redesign
  information cost/urgency before any further learned-planner training.

The clairvoyant ceiling must exceed blind; otherwise the basic bodily payoff or
audit implementation is invalid.

No learned-model parameter, task constant, policy tie-break, or gate may be
changed after these results are read. No fresh training is permitted from this
diagnostic alone.
