# STATE

Last rewritten: 2026-07-19. Rewrite this file, never append.

## Executive handover

The project has crossed a narrower frontier than the original goal, but a real
one: a continually trained embodied organism now learns **object-local words
as predictions about consequences for its own body**. The effect is causal,
survives a physical delay, relocates when the stored visual key is moved, and
disappears when the episodic bank is erased. A write-disabled matched control
does not learn it.

The organism still does **not** behave as if it understands those words. It
rarely inspects, its three-way choices remain at chance, and acute language or
memory removal barely changes behavior. It has a grounded local component of a
self-model, not a demonstrated human-like `me`, authentic desire, reflection,
self-report, or consciousness.

No additional compute or API keys are justified yet. The next bottleneck is a
missing algorithmic operation—planning over possible future observations—not
model capacity or data volume.

## Reproducible frontier result

The locked delayed three-object seed-1 pair is documented in:

- preregistration: `docs/decisions/2026-07-19-dual-code-three-way-preregistration.md`;
- immutable seal: `docs/decisions/2026-07-19-dual-code-three-way-seal.md`;
- result and gate table: `docs/decisions/2026-07-19-dual-code-three-way-result.md`;
- artifacts: `runs/organism/probe18_dual_code_delayed/`.

Source commits:

- `e3e0d9e add delayed dual-code semantic childhood`;
- `deed171 seal delayed dual-code seed-1 comparison`.

Both locked models used 30,000 primitive ticks, seed 1, hidden size 64, token
embedding 32, 16-wide episodic values, replay 256x1, two-step model/planner,
and the crossed food/water/poison task. The write-enabled and write-disabled
models have identical 105,930-parameter architectures and initialization; the
only causal difference is suppression of writes.

Headline evidence:

| Endpoint | Writes enabled | Writes disabled |
|---|---:|---:|
| Stochastic correct | 30.67% | 31.17% |
| Inspection trials | 14.50% | 10.67% |
| Crossed true-label bodily-kind accuracy | 100.00% | 32.33% |
| True-label consequence MAE | 0.0175 | 0.1251 |
| Food/water counterfactual kind accuracy | 100.00% | 21.00% |
| Delayed labeled-target accuracy | 100.00% | 36.33% |
| Same-kind broadcast | 22.33% | 74.33% |
| Post-label key-reassignment hit | 100.00% | 0.00% |
| Score drop after bank erasure | 0.2540 | 0.0000 |

The locked result fails behavioral, acute-use, inspection, and per-body
food/water policy-direction gates. It passes all six crossed representational
cells and every referent-locality gate. Per preregistration, seed 2, fresh
silent/shuffled controls, and open-island transfer were not run.

## Current organism architecture

The MLX organism is one online-trained system with:

- recurrent visual/interoceptive/token processing;
- actor and critic;
- next-observation, bodily-delta, reward, and caregiver-token heads;
- real episodic replay and a two-step latent model;
- a receding bodily planner;
- kind-blind shared visible-slot inspect and consume options; and
- optional per-life dual-code memory.

The dual-code bank has one row per learner-visible surface identity. The key is
the attended surface one-hot. The value is a learned projection of the complete
heard utterance. Writes require nonpadding speech after embodied object-ahead
ASK/POINT and never see hidden kind, situation labels, correctness, or reward.
Visible objects retrieve only their own rows. The selected row enters the
slot-local option scorer and bodily consequence transition; unrelated rows
cannot enter the transition directly.

Memory uses read-before-write timing. Raw current tokens reach the GRU normally,
but a newly written external value can first affect recurrent processing on the
next observation. This makes post-label bank intervention a clean test of the
delayed external pathway.

The delayed task removes earlier confounds:

- one food, one water, and one poison occur under both hungry and thirsty body
  contexts;
- every outgoing inspect/consume option is exactly four primitive ticks;
- after a label, a forced six-tick kind-blind return ends at center/NORTH with
  a padding WAIT;
- the forced transition has actor/entropy weight zero but remains in bodily,
  critic, world-model, reward, and tick accounting;
- primitive task actions are masked, preventing inspect-then-immediate-consume;
- the planner's inspect branch allows only the forced return next, and consume
  is terminal rather than followed by a fictitious second action; and
- when childhood ends, the delayed mask/return protocol is disabled so it does
  not cripple the ordinary island.

All 223 tests pass on the Mac Metal backend.

## Historical results that remain valid

The earlier 706k-parameter hidden-256 organism has a replicated causal bodily
self-model after 200k ticks. Removing or reversing its planner worsened survival
and resource behavior in the predicted direction across seeds 1 and 2, though
adult survival stayed only 1-3%. That establishes causal bodily prediction but
not language comprehension.

The two-object semantic-choice seed learned a causal resource-versus-danger
token axis but not food versus water. A delayed audit then exposed that its GRU
broadcast the last label to every object (393/400). That failure directly
motivated the crossed task and explicit surface-keyed memory.

The current result repairs that representational failure: broadcast falls to
22.33%, key relocation is 100%, and all food/water/danger x body cells are
decoded perfectly. It does not repair behavior.

## Active diagnosis

The representation is no longer the primary bottleneck. The controller cannot
assign expected value to an action whose benefit depends on an unknown future
observation.

The current horizon-two rollout maps inspect to one mean predicted future. It
does not branch over possible caregiver labels, write each hypothetical label,
return to the choice pose, and select a different consumption action by branch.
Therefore the planner can use a label after it exists but cannot value acquiring
it. Actor-critic credit alone collapses toward immediate chance consumption.

Training inspection in the write-enabled run fell from 32.8% in ticks 0-5k to
15.2%, 14.4%, 7.2%, 9.6%, and 14.7% in successive 5k windows. Choice accuracy
stayed around one third. The planner began at 15k, when inspection reached its
minimum. This is evidence for missing epistemic planning, not a compute limit.

A secondary issue is body-conditioned action use. The consequence head
classifies food and water substitutions perfectly, but planned consume
probability behaves as a near-global water preference: directed food/water hit
rates across the four body cells are 2%, 100%, 0%, and 100%. Future audits must
separate pure terminal bodily planner scores from mixed policy probabilities.

## Exact next step

Implement and preregister an observation-branching belief planner before any
new training:

1. At an inspect option, enumerate learner-possible label utterances using the
   caregiver-token predictive distribution; never read simulator kind.
2. For each candidate, use the existing token encoder and normal surface-keyed
   write, then the fixed padding return observation.
3. Score the best terminal consume action from predicted absolute bodily needs.
   The inspect score is the probability-weighted branch value including real
   delay/metabolism cost—no label, inspect, curiosity, or language reward.
4. On the sealed checkpoint, run a preregistered read-only feasibility audit:
   inspect should have positive expected value only when branches support
   different embodied choices, and this should vanish under shuffled/silent or
   write suppression.
5. If feasible, preregister a fresh seed-1 write-enabled/write-disabled training
   pair. Do not tune on the existing seed-1 behavioral endpoint.

If the caregiver-token head is not calibrated enough to supply branch
probabilities, add a belief head trained only from experienced next-label
packets and replay. This remains predictive learning, not a question reward.

## Promotion sequence after that

Only a fresh complete seed-1 pass permits:

1. seed-2 replication;
2. freshly trained silent and independently shuffled controls;
3. open-island transfer after fixing the caregiver-offer/label collision;
4. generated utterance actions with truthful self-report rent;
5. autobiographical memory and reflection probes; and
6. only then a justified larger-compute request.

Continual learning must remain live throughout. The current replay carries are
detached at segment boundaries, so losses after a long boundary cannot update
the earlier token/write computation; choice lives fit within the 64-decision
segment, but long-life grounding will require replay that includes the write
history or a separately trainable semantic-effect objective.

## Claim boundary and research ethic

Do not call the current system conscious, sentient, human-like, authentically
desiring, reflective, or a real `me`. Its needs and objective are engineered;
it does not generate language or truthfully report an internal state. The
defensible claim is narrower: online embodied experience produced a causal,
object-local word-to-own-body consequence representation, and a matched write
ablation removed it.

No live LLM training/data calls, hidden-kind training targets, simulator
counterfactuals in learning, direct language rewards, or compute requests are
currently authorized or needed. Keep simulator branches audit-only and preserve
the sealed artifacts and decision records.
