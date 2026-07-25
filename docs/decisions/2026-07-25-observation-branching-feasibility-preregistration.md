# Observation-branching planner feasibility preregistration

Date: 2026-07-25  
Status: locked before implementing the audit and before reading any resulting
planner score from the sealed checkpoint

## Goal-level metric

This experiment targets **behavioral language acquisition through the bodily
self-model**. It asks whether the current learned model contains enough
predictive structure for an inspect action to have positive instrumental value
without an inspect reward, curiosity reward, language reward, hidden-kind
target, or simulator branch in learning.

The experiment does not claim reflection, generated self-report, or
consciousness. It tests the missing bridge selected by the prior failure:
planning over possible observations that change the agent's own memory and
therefore its subsequent embodied choice.

## Fixed checkpoint and untouched training artifacts

The read-only target is:

`runs/organism/probe18_dual_code_delayed/organism_grounded_choice30000x3h40r6n0.55_options_inspect_bind16_plan6h2_replay256x1_seed1.npz`

SHA-256:
`bb73bf3c00a262e34873ad4644ba74aa85b441d8fde069e86bc15c483b141ace`

This is the write-enabled model from the sealed delayed three-object result.
The checkpoint, its metadata, its training CSV, and the prior harness CSV must
not be modified.

## Mechanism fixed before evaluation

At a canonical pre-choice state:

1. Score every immediate consume option with the existing learned bodily
   transition: minimum predicted absolute need after consumption. External
   reward weight is zero.
2. For each visible inspect option, apply the learned option transition and
   decode its predicted label-observation vector, bodily delta, and
   caregiver-token logits.
3. Enumerate exactly three learner-possible utterances:
   `this food`, `this water`, and `this danger`, each EOS-terminated and padded
   through the repository's ordinary tokenizer.
4. Compute a normalized categorical branch probability from the learned
   caregiver-token logits by summing the per-position log probabilities of
   each complete candidate sequence and softmaxing across the three candidates.
5. In each branch, feed the hypothetical utterance through the existing token
   encoder. Write its value to the selected learner-visible surface key using
   the ordinary learned binding projection and validity bit. The planner
   receives the selected surface one-hot from the current observation; it
   never receives environment kind, the correct action, reward, or audit label.
6. Apply the learned fixed-return transition, including its predicted bodily
   cost. Construct the post-return observation from the exact public protocol:
   the original center geometry and visible surfaces, NORTH heading, WAIT as
   last action, padding tokens, and the learned predicted absolute needs. This
   is known option kinematics, not a hidden-kind lookup.
7. From the post-return state, score only terminal consume options with the
   existing learned bodily transition. The branch value is the best terminal
   minimum predicted absolute need. The inspect value is the
   probability-weighted branch value. Thus inspect, fixed return, and consume
   metabolism are all charged, and no information bonus is added.

The implementation may vectorize these operations but may not change their
semantics after results are read.

The deployed planner will replace only delayed semantic-choice inspect scores.
Immediate consume scores remain one-step terminal bodily scores. Other actions
and the ordinary open-island planner remain unchanged.

## Read-only audit

Use 300 fresh deterministic held-out three-object contexts beginning at seed
`1_700_000`, with choice horizon 40, low need 0.55, and fixed return duration
6. No optimizer update or environment action is permitted. The harness must
load the sealed checkpoint and compute all conditions on identical states.

Report:

- candidate-prior normalized entropy;
- fraction of inspect options whose three branches select at least two
  different terminal consume slots;
- fraction where the target is selected under its body-matching label;
- fraction where the target is avoided under `danger`;
- mean best-inspect value minus best-immediate-consume value;
- fraction of contexts with positive best-inspect advantage;
- the same quantities with episodic writes disabled on the identical model;
- the same quantities with all candidate utterances collapsed to padding; and
- direct differences between the intact and each ablation.

True hidden kind may be read only after planner outputs exist, to stratify
accuracy. It may not enter a planner function or candidate probability.

## Feasibility gates

The mechanism is feasible only if all of the following hold:

1. Candidate-prior normalized entropy is at least 0.80. This ensures the
   observation model does not collapse to one token branch.
2. At least 80% of intact inspect options produce at least two distinct
   terminal choices across the three possible labels.
3. In at least 80% of body-matching label branches the inspected target is the
   selected terminal consume; in at least 80% of `danger` branches it is
   avoided.
4. Mean intact best-inspect advantage over immediate consumption is positive,
   and at least 60% of contexts have positive advantage.
5. Disabling episodic writes reduces mean inspect advantage by at least 0.02
   absolute bodily-value units and reduces branch contingency by at least
   30 percentage points.
6. Collapsing all label branches to padding makes mean inspect advantage
   nonpositive.

Failure of gates 1-3 diagnoses an inadequate observation or
label-to-consequence model. Failure only of gate 4 diagnoses excessive physical
cost or poor prior calibration. Failure of gates 5-6 diagnoses a shortcut in
the proposed planner. No fresh training pair is allowed unless all gates pass.

The independently shuffled language control is deliberately not an acute
feasibility gate. A grounded checkpoint has no explicit learned reliability
belief that could infer a newly randomized channel before observing it.
Claiming that an acute shuffle must erase ex-ante information value would
confound semantics with channel reliability. Independently trained shuffled
and silent models remain mandatory promotion controls after a fresh grounded
training pass.

## If and only if the feasibility gates pass

Preregister a fresh seed-1 write-enabled/write-disabled training pair with the
observation-branching planner active from the same 15,000-tick point as the
sealed baseline planner. Preserve the 30,000 primitive-tick budget, architecture,
initialization, task draws, replay, loss weights, and evaluation seeds. Do not
tune planner scale or gates on the sealed behavioral endpoint.

No seed 2, language-control training, open-island transfer, larger compute, or
data generation follows from feasibility alone.

## Methodological basis

The backup is the smallest exact observation branch for this controlled POMDP:
belief-space planning values sensing only through downstream expected utility.
It follows the decision-theoretic structure used by POMDP and Bayes-adaptive
planning, while retaining the repository's learned recurrent world model and
object-local memory. It intentionally does not add a generic information-gain
reward such as VIME; that would make inspection valuable even when the
information cannot improve embodied action.

Primary references:

- Silver and Veness (2010), *Monte-Carlo Planning in Large POMDPs*.
- Ross et al. (2011), *A Bayesian Approach for Learning and Planning in
  Partially Observable Markov Decision Processes*.
- Guez, Silver, and Dayan (2012), *Efficient Bayes-Adaptive Reinforcement
  Learning using Sample-Based Search*.
- Hafner et al. (2023), *Mastering Diverse Domains through World Models*.
