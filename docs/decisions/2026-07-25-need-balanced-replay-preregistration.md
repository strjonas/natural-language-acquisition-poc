# Need-balanced replay preregistration

Date: 2026-07-25
Status: locked before replay selection is changed or retrained.

## Diagnosis

Probe44 solves final-replay event magnitude at the single analytic bound-event
weight 0.0231, but fails balanced online learning. In the last 10,000 ticks it
receives 190 valid-bound food restorations and 300 water restorations; online
MAE is 0.1326 food versus 0.0569 water. The bound loss balances active body
dimensions within a segment, but cannot supply a dimension absent from that
segment.

This is the setting studied by class-balanced replay methods for temporally
correlated, imbalanced online streams. Chrysakis and Moens introduce
class-balancing reservoir sampling specifically for that setting
(https://proceedings.mlr.press/v119/chrysakis20a.html). GRASP's broad rehearsal
comparison likewise finds uniform class-balanced sampling a strong baseline
(https://proceedings.mlr.press/v274/harun25a.html). The present test uses the
smallest deterministic sampling analogue and introduces no class-balance
hyperparameter.

## Manipulation

Keep the existing 256-segment reservoir and exactly one world-model replay
update per online segment. Add a default-off `need_balanced_replay` selector:

1. When a segment enters the reservoir, derive the set of resource dimensions
   for which it contains a valid-bound consume event. Eligibility uses only:
   public consume action, selected external-memory validity, and a positive
   lived own-body delta above 0.175 in food or water.
2. Replay requests alternate deterministically food, water, food, water.
3. Select uniformly from current reservoir segments eligible for the requested
   need. If none exists, use the unchanged uniform reservoir sampler.
4. A segment containing both needs is eligible for both requests.

The selected segment receives the exact existing world-model replay loss. No
update is added, no loss weight changes, and no event is duplicated in the
reservoir. Hidden kind, simulator event name, current demand, correctness, and
counterfactual outcome are unavailable to selection.

The sealed uniform reservoir remains the default and its random-number path is
unchanged when balancing is off.

## Locked paired runs

Two fresh seed-1 exact probe44 treatment configurations:

1. **Uniform control:** existing replay, bound-event weight 0.0231.
2. **Balanced treatment:** need-balanced replay, bound-event weight 0.0231.

Both train for 60,000 ticks with hidden 64, 64-decision segments,
three-object 40-tick childhood throughout, six-tick returns, eight rounds, low
need 0.55, consume/inspect options, binding 16, horizon-two loss at weight 1,
replay capacity 256 with one update, drift 0.01, split drift head, and clip 10.
Training-path instrumentation remains enabled.

## Accounting invariant

After both food- and water-eligible reservoir segments exist, requested replay
needs must differ in count by at most one. Actual selected-need counts and
uniform fallbacks are reported. The total replay update count must equal the
uniform control's rule: one per collected segment after the reservoir first
becomes nonempty.

## Locked gates

1. **Replay mechanism.** The accounting invariant holds; at least 95% of
   balanced requests find an eligible segment after both need pools first
   become nonempty.
2. **Endpoint calibration.** Treatment final-replay valid-bound food and water
   MAE are each <= 0.10 and neither is more than 0.02 worse than the fresh
   uniform control.
3. **Balanced online learning.** In the last complete 10,000-tick window,
   treatment food and water valid-bound MAE are each <= 0.10, their absolute
   gap is <= 0.03, and each improves by >= 0.15 from the first window.
4. **Real terminal calibration.** If gates 1-3 pass, on 300 fixed contexts the
   demanded-minus-best-wrong predicted margin is >= 25% of the exact realized
   0.4000 margin for food and water separately, with demanded-choice accuracy
   >= 90% for each.
5. **Forecast and memory.** If gate 4 passes, ten-tick post-return need ordering
   is >= 90%; reuse in every round 3-8 is >= 90%; acute silence and acute write
   suppression are each <= 45%.
6. **Feasibility.** Only if gates 1-5 pass, run and report the unchanged
   seven-gate protocol battery.

## Stop rule

- If gate 1 fails, reject the implementation as not testing balanced replay.
- If gate 1 passes but gate 2 or 3 fails, balanced replay is insufficient and
  no replay-capacity, update-count, or weight sweep is allowed.
- If local learning gates pass but terminal transfer fails, replay balance is
  no longer the blocker.
- Any forecast or memory regression rejects promotion.

No larger model, external compute, generated data, seed expansion, or
write-disabled training pair is licensed by this local paired test.

## Claim boundary

This tests stable continual credit across a nonstationary embodied stream. It
is not reflection, self-report, generated language, consciousness, or
subjective experience.
