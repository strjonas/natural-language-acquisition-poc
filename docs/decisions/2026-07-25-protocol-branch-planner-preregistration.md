# Protocol-branch planner preregistration

Date: 2026-07-25
Status: locked before implementation and before any output of the mechanism is
read

## Why this is a defect fix and not another variant

`2026-07-25-homeostatic-terminal-utility-result.md` isolated the failure of the
previous branch to a single component: the planner fabricated the post-inspect
observation with a decoder whose mean L1 reconstruction error is 13.55 and
which was never trained for reconstruction. Replacing the tokens, the write
mechanism, or the return observation each left the real path at 100%; only the
fabricated observation collapsed it.

This preregistration removes that component. It fabricates no sensory scene.
Every observation it applies is one the public protocol already fixes, and the
same construction is already validated at 100% in row 3 of that result. No
utility rule is searched: the corrected homeostatic utility is carried over
unchanged and remains the one locked in
`2026-07-25-homeostatic-terminal-utility-preregistration.md`.

Nothing about capacity, optimizer, reward, entropy, training budget, label
traffic, inspection subsidy, or hidden targets changes.

## Mechanism

For each visible inspect option, and for each of the three learner-possible
label packets `this food`, `this water`, `this danger`:

1. take the branch probability from the learned caregiver-token head exactly as
   before, normalized across the three candidates;
2. apply the learned surface-keyed write for that candidate to the organism's
   own current state, using the attended surface one-hot from its own
   observation. No observation is fabricated and the recurrent core is not fed
   an invented scene;
3. apply the three public padding-only observations that separate a label from
   its next-round use: the fixed return, the forced round transition, and the
   next choice pose. Each is built by the existing
   `_semantic_choice_return_vector` construction, at center, heading NORTH,
   last action WAIT, padding tokens, with the same visible surfaces, and with
   the bodily needs of the context being queried. The count is three because
   the protocol has three such observations, not because three performed best;
4. score the best terminal consume option under the corrected utility;
5. query the two recurring learner-observable bodily contexts, hungry and
   thirsty, in the same way, and average them; and
6. weight that reuse value by the number of remaining rounds, as before.

An immediate consume option receives the identical future-round treatment under
the organism's *current* memory, so only the expected improvement caused by a
hypothetical word can offset inspection's real metabolic and delay cost. A
surface whose memory row is already valid falls back to the ordinary costly
rollout. No label, inspection, curiosity, novelty, correctness, hidden kind, or
simulator counterfactual is rewarded anywhere.

## Locked read-only test, part one: feasibility

Same checkpoint
`runs/organism/probe24_persistent_childhood_corrected/organism_grounded_choice30000x3h40r6n0.55q8_options_inspect_bind16_replay256x1_seed1.npz`,
SHA-256 `02bb27563d6af08e1255e0d8d48e4a46e048d4ae48be54b2fd0620827cac728e`,
same 300 fixed held-out contexts from seed `1_700_000`, reuse count 7,
comparing intact hypothetical writes, acute write suppression, and collapsed
all-padding branches.

The seven gates are carried over verbatim and are not weakened:

1. intact mean best-inspect advantage over immediate consumption is positive;
2. at least 75% of contexts have positive intact inspect advantage;
3. write suppression reduces mean inspect advantage by at least 0.05;
4. collapsed labels reduce mean inspect advantage by at least 0.05;
5. at least 60% of inspect options support label-contingent terminal choices;
6. a matching-resource branch selects the inspected target at least 60%; and
7. a danger branch avoids the inspected target at least 90%.

## Locked read-only test, part two: behavior

Only if part one passes in full, activate the planner read-only on the same
checkpoint over 300 fixed eight-round held-out lives at the repository's
existing locked planning scale 6.0, reporting grounded, acute silent, acute
independently shuffled, acute write-suppressed, and planner-removed behavior on
identical seeds.

Behavioral promotion requires:

1. at least 60% aggregate grounded correctness;
2. at least 65% correctness in each of rounds 3-8;
3. grounded exceeds planner-removed correctness by at least 15 points;
4. acute write suppression reduces correctness by at least 15 points; and
5. both acute silence and acute shuffle reduce correctness by at least 15
   points.

## Locked confirmatory audit: cross-round reuse

Independently of the planner, the persistence result of the previous document
is promoted from diagnosis to a gated endpoint, on 300 fixed lives from seed
`1_700_000`, with labels acquired only in rounds 1-2 by a scripted read-only
driver:

1. rounds 3-8 each select the needed object at least 90% of the time;
2. the same driver with acute write suppression falls to at most 45%; and
3. the same driver with acute silence falls to at most 45%.

This audit reads the organism's own terminal scores, never the simulator's
kinds, when choosing.

## If everything passes

One fresh matched write-enabled / write-disabled training pair at the identical
30,000-tick seed-1 configuration of
`2026-07-25-persistent-childhood-learning-preregistration.md`, planner
activated after 15,000 ticks, judged by that document's nine promotion gates
unchanged.

Seed 2, freshly trained silent and shuffled controls, open-island transfer,
generated speech, and any compute or data request remain blocked until that
fresh pair passes.

## Stop rule

If part one fails, stop. Do not implement a fourth planner on this task, do not
adjust the settling count, the reuse count, or the planning scale, and do not
re-derive a new utility. Report the confirmatory cross-round audit on its own
merits, since it does not depend on the planner, and take the substrate
question upward: whether inspection can be motivated at all without a
reconstructive world model, or whether the organism must instead learn the
value of knowing from experienced cross-round reuse under a longer training
budget.
