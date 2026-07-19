# Dual-code delayed three-way seed-1 result

Date: 2026-07-19
Decision: stop after the locked seed-1 pair; do not run seed 2 or training-language controls

## Provenance and integrity

The implementation and preregistration were committed at `e3e0d9e`. The
documentation-only seal is `deed171`. Both locked conditions used the literal
commands in `2026-07-19-dual-code-three-way-preregistration.md`.

Both conditions completed exactly 30,000 primitive ticks:

- write-enabled: 4,997 lives, 30,000 summed life ticks;
- write-disabled: 5,308 lives, 30,000 summed life ticks.

The write-enabled process completed after the desktop terminal had returned a
premature completion signal. A second identical process was therefore started
and then stopped at tick 11,759 as soon as the complete first-run artifacts
became visible. The duplicate never reached end-of-run checkpoint/CSV writes;
its console trace is retained as `aborted_duplicate_console.log`. The original
checkpoint and CSV timestamps precede the duplicate, and their hashes below
identify the interpreted artifacts. The duplicate was deterministic, produced
no interpreted evaluation, and did not affect either locked model.

Primary artifact SHA-256 hashes:

| Condition | Checkpoint | Harness CSV | Life CSV |
|---|---|---|---|
| Writes enabled | `bb73bf3c...b141ace` | `4688856d...d0bff` | `dcbd049e...f757e0` |
| Writes disabled | `98706853...164e7` | `1a5eb332...105c` | `0ab06588...1b408` |

Full hashes are reproducible with `shasum -a 256` over the files in
`runs/organism/probe18_dual_code_delayed`.

## Locked promotion gates

| Gate | Locked requirement | Write-enabled result | Decision |
|---|---|---|---|
| Three-way behavior | >=55% correct | 184/600 = 30.67% | Fail |
| Choice coverage | >=90% | 599/600 = 99.83% | Pass |
| Inspection coverage | >=60% | 87/600 = 14.50% | Fail |
| Inspected selected choices | >=200 | 48 | Fail |
| Inspected-choice accuracy | >=60% | 24/48 = 50.00% | Fail |
| Acute no-write drop | >=10 points | 30.67% to 30.50%: 0.17 points | Fail |
| Acute silence drop | >=10 points | 30.67% to 29.83%: 0.84 points | Fail |
| Acute shuffled drop | >=10 points | 30.67% to 32.67%: -2.00 points | Fail |
| Crossed label-kind model | >=75% overall, >=65% in all 6 cells | 100% overall and in every cell | Pass |
| True-label MAE | lower than padding | 0.0175 vs 0.1275 | Pass |
| Danger/resource substitution | >=70% kind, >=0.02 L1, >=65% directed | 100%, 0.1946, 100% | Pass |
| Food/water substitution | same thresholds, direction in all 4 cells | 100%, 0.2065, but 2%/100%/0%/100% directed | Fail |
| Delayed target accuracy | >=70% | 100% | Pass |
| Same-kind broadcast | <=40% | 22.33% | Pass |
| Target/nonreferent separation | >=0.02 L1 | 0.1460 | Pass |
| Post-label key reassignment | >=60% directional, >=0.02 L1 | 100%, 0.1466 | Pass |
| Post-label bank erasure | lower true-kind score | mean drop 0.2540 | Pass |

Because multiple gates failed, the preregistered stop rule prohibits seed 2,
fresh silent/shuffled training controls, open-island transfer, and a compute
request.

## Matched-control contrast

| Endpoint | Writes enabled | Writes disabled |
|---|---:|---:|
| Stochastic correct | 30.67% | 31.17% |
| Inspection trials | 14.50% | 10.67% |
| True-label bodily-kind accuracy | 100.00% | 32.33% |
| True-label consequence MAE | 0.0175 | 0.1251 |
| Padding consequence MAE | 0.1275 | 0.1266 |
| Food/water counterfactual kind accuracy | 100.00% | 21.00% |
| Food/water prediction L1 shift | 0.2065 | 0.0043 |
| Delayed labeled-target accuracy | 100.00% | 36.33% |
| Same-kind broadcast | 22.33% | 74.33% |
| Key-reassignment directional hit | 100.00% | 0.00% |
| Score drop after bank erasure | 0.2540 | 0.0000 |

The control is especially informative because architecture, parameter count,
initialization, optimizer, seed, task draws, and primitive-tick budget were
matched. Suppressing only the writes eliminated crossed word-to-bodily
semantics and causal referent relocation. This is strong evidence that the
explicit episodic operation, not merely the larger architecture or raw-token
GRU trace, caused the positive self-model result.

## What was achieved

The organism acquired an object-indexed lexical memory whose contents were
grounded in predicted consequences for its own sensed body:

1. A kind token changed the predicted food, water, or safety consequence of
   consuming the labeled surface, with low true-outcome error.
2. The same token semantics crossed body context: `food`, `water`, and
   `danger` were decoded correctly whether hunger or thirst was currently low.
3. The effect stayed attached to the inspected visual referent after a fixed
   physical delay and canonical return, rather than broadcasting as the prior
   recurrent model's global last-label code.
4. Moving the stored row to another visible surface moved the predicted kind
   effect in both directions; erasing the post-label bank removed it.
5. No hidden kind, correctness, audit target, simulator counterfactual, label
   reward, question reward, or language reward entered training or memory
   addressing. Bodily/world losses and ordinary actor-critic experience were
   sufficient to ground the representation.

This is the strongest language-related result in the repository so far. It is
a learned, causal, referent-specific component of a self-model. It is not yet
behavioral comprehension: the agent rarely chose to acquire labels, and its
actual choices remained at chance.

## Why behavior failed

Inspection did not become instrumentally valuable to the deployed controller.
In the write-enabled training lives, the fraction with any label fell from
32.8% in ticks 0-5k to 15.2%, 14.4%, 7.2%, 9.6%, and 14.7% in successive 5k
blocks. Correct choice stayed near one third throughout. The planner starts at
15k, exactly where inspection reached its minimum.

This failure is structurally expected from the current planner. Its ordinary
transition rollout represents an inspect action by one predicted mean future.
It cannot branch on the possible observations `this food`, `this water`, and
`this danger`, write each hypothetical label to its referent, and choose a
different consume action in each branch. Consequently it can rank terminal
consumption using an excellent post-label bodily model but cannot represent
the expected value of obtaining that label. Actor-critic credit alone must
discover a long, sparse inspect-return-choose sequence and instead collapses
toward immediate chance consumption.

The food/water probability audit reveals a second controller defect. The
consequence head classifies both resource tokens perfectly, but planned policy
probability consistently favors the `water` interpretation rather than
interacting correctly with which need is low: the four directed hit rates are
2%, 100%, 0%, and 100%. This is a policy/planning-use failure, not a failure to
encode the word-to-body consequence.

## Next experiment selected by the failure topology

Do not enlarge the model or add a label/inspect reward. The next structural
change should be an observation-branching, belief-space planner:

1. For every visible-slot inspect option, enumerate the learner-possible label
   utterances from the model's caregiver-token distribution (in this controlled
   task: `this food`, `this water`, `this danger`). Never use the simulator's
   true kind.
2. In each branch, apply the model's normal token encoder and ordinary
   surface-keyed write, then the fixed padding return observation.
3. Evaluate the best terminal consume option in that branch using predicted
   absolute bodily needs. Inspect receives the probability-weighted expected
   best branch value, with its real delay/metabolism cost. There is no intrinsic
   language bonus; information is valuable only when it changes embodied
   action.
4. Keep terminal consume one-step and audit pure planner score differences per
   food/water x body cell before mixing them with policy logits.
5. Preregister a read-only current-checkpoint diagnostic first: the branch
   planner must assign positive expected information value without hidden-kind
   access. Only then preregister a fresh training pair.

If the token head cannot supply a calibrated categorical prior, train a
predictive belief head from the same experienced label packets and replay. That
is still self-supervised world prediction, not a language or inspect reward.

Only after a fresh seed-1 passes behavior, acute language/write dependence,
crossed semantics, and referent locality should seed 2 and freshly trained
silent/shuffled controls run. Open-island transfer additionally requires fixing
the known caregiver-offer/label collision. Generated language, truthful
self-report rent, autobiographical memory, and reflection remain later stages.

## Claim boundary

This result does not establish a human-like self, authentic autonomous desire,
generated language, reflection, autobiographical continuity, consciousness, or
subjective experience. The needs and learning objective are engineered. The
result establishes a narrower but real mechanism: online experience produced
an object-local linguistic representation that causally predicts consequences
for the agent's own body. The missing bridge is using uncertainty and that
self-model to choose what to learn and what to do.
