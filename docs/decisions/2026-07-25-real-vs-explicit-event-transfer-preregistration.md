# Real-versus-explicit bodily-event transfer preregistration

Date: 2026-07-25
Status: locked before implementing or running the audit.

## Result selecting this diagnostic

A scale-balanced loss on lived bodily event entries failed to expand the
protocol branch's demanded-minus-wrong terminal margin:

| Demand | Fresh zero control | Event treatment | Realized |
|---|---:|---:|---:|
| Food | 0.0169 | 0.0156 | 0.4000 |
| Water | 0.0274 | 0.0244 | 0.4000 |

The treatment preserves forecast and 100% cross-round reuse but lowers food
terminal choice to 76.26%. It is rejected.

The failed audit enters the terminal state through an explicit hypothetical
surface-keyed lexical write. Training sees only real label observations and
real subsequent consumption. It remains unknown whether the new loss failed to
calibrate even those in-distribution states or whether calibrated lived event
magnitude fails to transfer through the explicit write/settling construction.

## Fixed paired audit

Use the exact fresh zero control and event treatment from probe36. Use the same
300 contexts beginning at seed `1_700_000`. In each context, inspect the
demanded-resource object and really execute the six-tick return. The simulator
supplies the exact returned public body/scene and scores terminal actions; it
never supplies a model input unavailable to the embodied organism.

From the same initial recurrent state compare:

1. **Real immediate.** Process the real post-inspect label observation with its
   grounded tokens, then process the real padding-only return observation.
   Score all terminal consume options immediately.
2. **Real settled.** From the real immediate state, process two additional
   copies of the same padding-only public return observation. Together with the
   real return, the external row has had three post-write observation steps.
3. **Explicit settled.** Write the canonical matching word directly to the
   demanded object's learner-visible surface without touching the core, then
   process three copies of the exact same padding-only public return
   observation.

Paths 2 and 3 therefore have the same exact public terminal vector, same
canonical word meaning, same surface key, same number of post-write reads, same
terminal action model, and same utility. They differ only in whether the
lexical acquisition travelled through the real label observation or the
explicit counterfactual write.

For each path, overall and split by food/water demand, report:

- demanded-resource, other-resource, and poison terminal scores;
- demanded-minus-best-wrong margin;
- predicted/realized margin ratio;
- demanded-resource choice accuracy; and
- terminal-score MAE against the three exact simulator consume branches.

Also report real-settled minus explicit-settled margin for every resource and
the paired fraction on which the real-settled margin is larger.

## Locked interpretation

The fresh zero control must reproduce compressed explicit margins below 0.25 of
realization and at least 90% demanded-resource choice for both resources.

For the event treatment:

- **Lived-event failure:** both real-immediate and real-settled margin ratios
  are below 0.25 for either resource. The loss does not calibrate its own
  in-distribution labeled event state; explicit transfer is not the primary
  defect.
- **Counterfactual-transfer failure:** real-settled ratios are at least 0.75
  with at least 90% choice for both resources, while explicit-settled is below
  0.25 for either, and real-settled exceeds explicit-settled in more than 75%
  of paired contexts.
- **Successful transfer:** both real-settled and explicit-settled ratios are in
  [0.75, 1.25], MAE is below 0.05, and choice is at least 90% for both
  resources.
- Anything else is **mixed** and is reported by path and resource. It does not
  license a correction.

## Stop rule

Read-only evaluation of two existing checkpoints. No retraining, loss or weight
change, scale sweep, branch change, settling-count change, threshold change,
seed 2, transfer task, or generated speech.

If lived-event failure is observed, the next diagnostic must inspect training
target coverage/conditioning rather than strengthen the same loss. If
counterfactual-transfer failure is observed, the next intervention belongs to
learned state transfer and requires its own preregistration.

No compute or data request is justified.

## Claim boundary

This audit tests whether learned bodily consequence magnitude transfers from
real lexical experience into a causally matched hypothetical memory state. It
is not reflection, self-report, generated language, or consciousness.
