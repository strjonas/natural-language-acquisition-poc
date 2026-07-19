# Residual bodily self-model preregistration

Date: 2026-07-19
Goal-level metrics: accurate action-conditioned bodily prediction and causal use.

## Failure that motivates this probe

Probe 8 did not pass adult survival. On the same 20 held-out lives, normal
planning produced 91.85 mean steps and 12.55 successful resource uses; removing
planning produced 78.65 and 12.10. Reversing the self-model scores did not hurt:
it produced 94.0 steps and 14.0 resource uses. The planner is therefore not yet
load-bearing.

An audit that branched simulator *copies for diagnosis only* compared 2,001
candidate outcomes at 200 states. Absolute next-needs MAE was 0.1366, but the
within-state predicted/actual score correlation was -0.0550, predicted-best
action accuracy was 26.5%, mean regret was 0.0127, and the predicted-best
action's actual advantage over predicted-worst was 0.00019. The existing head
learned average next state rather than action effects.

## Intervention

The self-model will predict the change in each bodily need, not its absolute
next value. The sensed current needs are an explicit identity path:

`predicted next needs = clip(current needs + predicted need delta, 0, 1)`

Training targets remain exclusively the organism's experienced transition.
Transitions with large bodily changes receive higher prediction weight so rare
food, water, rest, poison, and collision consequences are not erased by the
many near-identity transitions. No counterfactual simulator outcome, hidden
object kind, language reward, or oracle action enters training.

## Smoke gate

Use the same offer childhood, kind-blind consume options, and held-out adult
world. Before another full 200k run, require on at least 200 audited held-out
states:

- within-state action-score correlation above 0.20;
- predicted-best over predicted-worst actual advantage above 0.005;
- lower counterfactual next-needs MAE than 0.1366; and
- normal planning better than reversed planning on lifespan or balanced
  resource use.

Failure closes residualization/weighting as a sufficient fix. It does not count
as evidence for a self-model merely because factual prediction loss falls.

## Result

The 40k-tick smoke failed the ranking gate. Counterfactual needs MAE improved to
0.0395 and chosen-action needs MAE was 0.0336, but within-state score correlation
was -0.0756. Predicted-best accuracy was 17%, regret was 0.0334, and
predicted-best actions had -0.0051 actual advantage over predicted-worst. Adult
survival was 1/20 at 400 ticks. Residualization repaired absolute prediction but
is not sufficient for action-conditioned self-knowledge.
