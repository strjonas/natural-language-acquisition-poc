# Consumption-scoped bodily-event calibration preregistration

Date: 2026-07-25
Status: locked before the new selector is implemented or trained.

## Diagnosis

Probe38 excludes target scarcity, semantic-conditioning scarcity, replay
dilution, and gradient starvation. The generic 0.01 event term receives 18,688
large-delta entries, of which 10,499 (56.18%) are forced between-round body
resets. Bound food/water restorations are only 2,380 entries (12.74%). Across
training, reset MAE improves 0.1064 while bound food and water improve only
0.0143 and 0.0321.

The event term is active but its magnitude-only mask selects a majority of
exogenous protocol resets. This test changes its causal scope, not its weight.

## Manipulation

Add a separate `bodily_consumption_event_loss_weight`, default zero. At the
locked value 0.01:

- the one-step extra calibration term selects above-threshold entries only
  when the learner-visible action is primitive `CONSUME` or a consume-object
  option;
- the horizon-two extra calibration term selects above-threshold cumulative
  entries only when the rollout's final action is one of those consume
  actions.

The final-action rule includes the lived return-then-consume chain used by the
planner and excludes consume-then-forced-reset endpoints. The original bodily
loss, drift correction, generic world-model loss, reward/token losses, policy
loss, replay sampling, and environment are unchanged. Hidden kind, event name,
choice correctness, and simulator counterfactuals are unavailable to the
selector.

The generic `bodily_event_loss_weight` is zero in the new treatment. Its sealed
0.01 probe38 treatment is the contemporaneous comparator. Both weights remain
independently selectable; all zero paths preserve prior behavior.

## Locked run

One fresh seed-1 treatment uses the exact probe38 60,000-tick configuration:
hidden 64, 64-decision segments, three-object 40-tick semantic-choice childhood
for all ticks, six-tick returns, eight rounds, low need 0.55, consume and
inspect options, binding size 16, horizon-two model loss at weight 1, replay
256 with one update, drift weight 0.01, split drift head, clip 10. The only
learning difference from probe38 treatment is generic event 0.01 versus
consumption-scoped event 0.01.

Training-path instrumentation remains enabled.

## Locked gates

Judged in order:

1. **Training fit.** Final replay MAE on valid-bound food and water
   restorations must each fall at least 25% relative to the probe38 generic
   treatment (below 0.2548 food and 0.2440 water), and neither may exceed its
   fresh zero control (0.3125 food, 0.2556 water).
2. **Real terminal calibration.** On 300 fixed probe35 contexts, predicted
   demanded-minus-best-wrong margin must be at least 25% of the exact realized
   0.4000 margin for food and water separately, with demanded-choice accuracy
   at least 90% for each.
3. **Forecast preservation.** Ten-tick predicted post-return need ordering must
   remain at least 90%.
4. **Memory preservation.** Cross-round reuse in every round 3-8 must remain at
   least 90%, while acute silence and write suppression each remain at most
   45%.
5. **Feasibility.** If gates 1-4 pass, run the unchanged seven-gate
   protocol-branch battery and report every gate.

The probe38 generic checkpoint is evaluated through the same terminal,
forecast, and memory audit commands wherever a paired endpoint is needed.

## Interpretation and stop rule

- If gate 1 fails, causal event selection is not sufficient. Reject the loss
  change and inspect representational collisions within bound consume contexts;
  do not tune its weight.
- If gate 1 passes but gate 2 fails, the training distribution is fit and real
  acquired-label transfer remains the blocker.
- If gate 2 passes but gate 3 or 4 fails, calibration trades away an already
  established capability and is rejected.
- Only if gates 1-4 all pass may the feasibility battery decide promotion.

No seed expansion, write-disabled training pair, generated data, larger model,
or external compute is licensed by this single mechanism test.

## Claim boundary

This tests whether a causally scoped embodied event objective improves the
agent's learned consequence model. It is not reflection, self-report,
generated language, consciousness, or evidence of subjective experience.
