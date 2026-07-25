# Bound-consumption credit balancing preregistration

Date: 2026-07-25
Status: locked before the bound selector is implemented or trained.

## Diagnosis

Probe40 rules out lexical collision, downstream rank collapse, and
water-specific downstream gain loss on the consumption-scoped checkpoint.
Probe39 nevertheless contains 1,256 valid-bound food restorations against 952
valid-bound water restorations, and its global event average gives the larger
group proportionally more credit.

This test asks whether calibration must be assigned specifically and equally
to bodily consequences for which the agent already has a selected-surface
lexical binding.

## Manipulation

Add a separate `bodily_bound_consumption_event_loss_weight`, default zero. At
the locked value 0.01 it selects an entry only when all are true:

1. the learner-visible action is primitive consume or a consume-object option;
2. the selected visible surface's external-memory validity bit is already set;
3. the own-body target magnitude exceeds the sealed 0.175 event threshold.

For horizon two, consumption must be the final action and binding validity is
read from that final lived action state. The term computes an MSE separately
for each of the four body dimensions that has at least one selected entry,
then averages those active per-need MSEs before applying the unchanged 0.01
weight and 0.4 scale. Thus food, water, and safety each receive equal credit
when present rather than credit proportional to their counts.

Action ID, visual surface, memory validity, and body delta are all available to
the learner. Hidden object kind, simulator event name, correctness, choice
demand, and counterfactual outcomes are not used.

All sealed ordinary bodily, drift, multi-step, reward, token, actor, and replay
losses remain. The prior generic and consumption-scoped objectives remain
separately selectable and default off.

## Locked paired runs

Both are fresh seed-1 60,000-tick exact probe39 configurations with
training-path instrumentation:

1. **Unbalanced control:** consumption-scoped event weight 0.01.
2. **Bound-balanced treatment:** bound-consumption event weight 0.01, with the
   unbalanced scoped weight zero.

Everything else is identical: hidden 64, 64-decision segments, three-object
40-tick childhood for all ticks, six-tick returns, eight rounds, low need 0.55,
consume/inspect options, binding 16, horizon-two loss at weight 1, replay 256
with one update, drift 0.01, split drift head, and clip 10.

## Locked gates

1. **Training fit.** On the final replay reservoir, valid-bound food and water
   restoration MAE must each be below 0.20 and at least 15% lower than its
   fresh unbalanced control.
2. **Balanced progress.** From first to last complete 10,000-tick window, both
   food and water valid-bound MAE must improve by at least 0.10, and the
   absolute food-water MAE gap in the last window must be <= 0.03.
3. **Real terminal calibration.** If gates 1-2 pass, on 300 fixed probe35
   contexts the predicted demanded-minus-best-wrong margin must be at least
   25% of the exact realized 0.4000 margin for food and water separately, with
   demanded-choice accuracy >= 90% for each.
4. **Forecast and memory.** If gate 3 passes, ten-tick post-return need ordering
   must remain >= 90%; reuse in every round 3-8 must remain >= 90%; acute
   silence and write suppression must each remain <= 45%.
5. **Feasibility.** Only if gates 1-4 pass, run and report the unchanged
   seven-gate protocol battery.

## Stop rule

- If gate 1 fails, reject bound-balanced credit; the remaining underfit is not
  explained by count imbalance, and the weight is not tuned.
- If gate 1 passes but gate 2 fails, improved aggregate fit is not balanced
  continual learning and is rejected.
- If training gates pass but terminal transfer fails, local training fit is no
  longer the blocker.
- Any forecast or memory regression rejects promotion.

No seed expansion, generated data, larger model, or external compute is
licensed by this local mechanism test.

## Claim boundary

This tests credit allocation from causally grounded lexical memory to learned
body consequences. It is not reflection, self-report, generated language,
consciousness, or subjective experience.
