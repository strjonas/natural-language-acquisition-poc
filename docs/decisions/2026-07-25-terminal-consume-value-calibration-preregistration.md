# Terminal consumption-value calibration preregistration

Date: 2026-07-25
Status: locked before implementing or running the diagnostic.

## Result selecting this diagnostic

The preregistered protocol return-origin treatment was refuted: chaining the
inspect latent worsened return prediction in every one of 300 contexts. A
read-only grouped decomposition then found that demanded resource determines
inspect-advantage sign exactly:

- all 139 food-demand contexts are positive under intact, write-suppressed, and
  collapsed branches;
- all 161 water-demand contexts are negative under all three;
- a lexical write adds +0.0466 mean advantage but changes no sign.

The organism nevertheless selects a labeled demanded-resource target 99.78% of
the time. Correct semantic ranking and insufficient instrumental magnitude can
coexist if the terminal consequence model compresses the value difference
between consuming the correct and wrong objects.

## Question

At the exact state where the protocol branch makes its terminal choice, how
large does the organism predict the bodily benefit of the correct consumption
to be, and how large is that benefit in the simulator?

This is a calibration audit, not a new planner. Hidden object kind is used only
to choose matched audit rows and score predictions; it never enters a model
input or update.

## Fixed audit

Use the unchanged probe33 checkpoint and the same 300 seeds beginning at
`1_700_000`. For each context:

1. identify the demanded-resource slot for scoring;
2. compute the existing four-tick inspect prediction for that slot;
3. write the canonical matching word to its learner-visible surface;
4. use the sealed aliased return query, predicted return body, and three public
   settling observations exactly as the current protocol branch does;
5. obtain the organism's terminal consume score for all three visible slots
   under urgent-deficit utility;
6. in an audit-only simulator copy, really inspect the demanded-resource
   object, execute the fixed six-tick return, and branch the three learner-
   visible consume options from that same returned state; and
7. score each realized terminal body on the same currently urgent need.

For both predicted and realized scores report, overall and separately for food-
and water-demand contexts:

- demanded-resource score;
- other-resource score;
- poison score;
- demanded-minus-best-wrong margin;
- demanded-minus-other-resource margin;
- demanded-minus-poison margin;
- demanded-resource choice accuracy;
- mean absolute error over the three terminal scores.

Also report the predicted/realized ratio for each margin and the paired fraction
of contexts in which the predicted demanded-minus-best-wrong margin is smaller
than the realized margin.

## Locked interpretation

Call the terminal value **compressed** if all of the following hold:

- realized demanded-minus-best-wrong margin is positive for both resource
  groups;
- predicted demanded-resource choice accuracy remains at least 90%;
- predicted demanded-minus-best-wrong margin is below 75% of realized in
  either resource group; and
- prediction is smaller than realization in more than 75% of paired contexts.

If compressed, the next manipulated variable belongs to bodily event-magnitude
calibration. Preregister it separately; do not alter drift supervision, the
branch, reuse count, or a feasibility threshold.

Call the terminal value **calibrated** only if, for both resource groups, the
predicted/realized demanded-minus-best-wrong ratio lies in [0.75, 1.25] and
terminal-score MAE is below 0.05. If calibrated, the resource sign split must be
attributed upstream to the economics represented by the persistent
continuation, and that component becomes the next read-only audit.

Any mixed outcome is reported as mixed and localized by resource and target
kind. Do not implement a correction from a mixed result.

## Stop rule

One checkpoint, one fixed seed set, one diagnostic. No retraining, scale sweep,
loss change, threshold change, seed 2, transfer, or generated speech. No compute
or data request can follow from this read-only local audit.

## Claim boundary

This diagnostic tests whether an embodied world model accurately values the
consequences of its own terminal action. It is not reflection, self-report,
generated language, or evidence of consciousness.
