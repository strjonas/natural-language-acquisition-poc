# Protocol return-origin preregistration

Date: 2026-07-25
Status: locked before changing or rerunning the protocol branch.

## Observation

The sealed 60,000-tick checkpoint passes the previously blocked semantic gate:
a branch whose label names the demanded resource selects the labeled object
99.78% of the time. Its lexical memory and ten-tick need ordering are both at
100.00%. Four feasibility gates nevertheless fail:

- positive inspect advantage in 46.33% of contexts, against 75%;
- write-suppression and collapsed-label advantage drops of 0.0466, against
  0.05;
- danger-label avoidance at 85.33%, against 90%.

The sign anomaly is exact: intact, write-suppressed and collapsed branches all
have the same 46.33% positive-context rate even though the intact mean advantage
is +0.0205 and both controls are -0.0262.

Code inspection identifies one severed edge in the learned counterfactual.
`observation_branching_inspect_values` computes the action-conditioned
`inspect_state` and decodes the four-tick inspection from it. For the next
transition, however, the protocol branch discards that latent and applies
`WAIT` to `states`, the pre-inspection choice state. It then interprets the
answer as the six-tick forced return. The world model is therefore queried for
the wrong conditional transition.

This is not an absence of multi-step supervision. The sealed checkpoint was
trained with `multi_step_model_horizon = 2` and
`multi_step_model_weight = 1.0`; its open-loop loss explicitly chains the
latent produced by the first lived decision into the second.

## Hypothesis

The aliased return origin, rather than lexical semantics or the repaired
metabolic forecast, determines the context-level sign of inspect advantage.
Chaining the already-computed `inspect_state` into the return prediction will
make the branch match the model's trained two-decision causal graph and will
remove the sign anomaly.

This follows the semi-Markov/options requirement that a temporally extended
action be modeled from the state at which that action begins, and the
multi-time-model requirement that abstract-action planning compose the
appropriate action-conditioned transition:

- Precup and Sutton, *Multi-time Models for Temporally Abstract Planning*:
  https://papers.nips.cc/paper/1362-multi-time-models-for-temporally-abstract-planning.pdf
- Khetarpal et al., *Temporally Abstract Partial Models*:
  https://papers.nips.cc/paper/2021/file/0f3d014eead934bbdbacb62a01dc4831-Paper.pdf

## Single manipulation

Add a selectable return-origin mode to the existing observation branch.

- **Aliased control:** preserve the current code exactly. Write the
  hypothetical lexical row into the pre-inspection recurrent state and query
  the return transition from that state.
- **Chained-inspect treatment:** keep that recurrent/memory state exactly as in
  the control for all subsequent public observation settling, but make a
  second copy of the same lexical write in `inspect_state` and use only that
  action-conditioned latent to predict the return bodily delta.

This separation is deliberate. The transition latent is licensed as an input
to the next world-model transition by the two-step loss. It is not substituted
for the organism's observation-driven recurrent state, so the public memory
protocol, three settling observations, lexical write, branch probabilities,
terminal utility, reuse count, and terminal consume model are held fixed.

No parameter is trained. The checkpoint, 300 seeds, thresholds, label
candidates, utility, reuse count seven, and all prior configuration remain
sealed.

## Measurements

Before reading the endpoint battery, add a read-only return-origin audit on the
same 300 contexts. For aliased and chained origins, report:

- mean predicted and realized return delta for each need;
- per-need bias and mean absolute error;
- worst per-need return absolute error;
- fraction of contexts in which the predicted post-return urgent need still
  names the demanded resource;
- paired fraction on which chaining reduces absolute bodily-delta error.

Also expose the existing feasibility battery under each selectable origin.
The old default must reproduce the sealed aliased numbers before the treatment
is interpreted.

## Locked gates

### Instrument and mechanism

- **R1 reproduction:** the aliased mode reproduces every sealed feasibility
  metric to printed precision, including 46.33%, +0.0205, -0.0262, 0.0466,
  85.11%, 99.78%, and 85.33%.
- **R2 causal-chain accuracy:** the chained origin has a strictly smaller worst
  per-need return absolute error than the aliased origin and improves absolute
  bodily-delta error on more than 50% of paired contexts.
- **R3 ordering:** demanded-resource urgent-index survival remains at least
  90%. This prevents an apparent value improvement bought by breaking the
  repaired need ordering.

### Existing feasibility battery, unchanged

On the chained treatment:

1. mean intact inspect advantage is greater than zero;
2. at least 75% of contexts have positive inspect advantage;
3. write suppression reduces mean inspect advantage by at least 0.05;
4. collapsed labels reduce mean inspect advantage by at least 0.05;
5. at least 60% of terminal choices are label-contingent;
6. a matching resource label selects the labeled target at least 60%;
7. a danger label avoids the labeled target at least 90%.

The treatment must pass all seven. Existing thresholds are not changed.

## Interpretation and stop rules

- If R1-R3 and all seven feasibility gates pass, the branch construction is
  repaired. Run the existing test suite, record the result, and proceed to the
  already-preregistered matched write-disabled training pair. Do not retrain
  this checkpoint.
- If R2 fails, the proposed origin was not the defect. Record the negative and
  use the paired per-context decomposition to identify the next factor; do not
  tune latent mixtures or duration scales.
- If R2 passes but any feasibility gate fails, record the dissociation and
  stop. Do not add another planner or alter a threshold.

Seed 2, transfer, generated speech, and larger compute remain blocked. This
experiment is read-only on one local checkpoint and cannot justify a compute or
data request.

## Claim boundary

Passing would establish only that a learned, action-conditioned bodily
self-prediction must be composed from the causally correct latent state for a
persistent word to acquire reliable instrumental value. It would not establish
reflection, self-report, consciousness, or generated language.
