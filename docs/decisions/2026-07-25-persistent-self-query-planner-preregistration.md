# Persistent self-query planner preregistration

Date: 2026-07-25  
Status: locked before implementation or evaluation

## Why this is the one permitted mechanism follow-up

The corrected seed-1 persistent-childhood pair isolates the remaining failure.
The write-enabled organism reaches 38.17% held-out correctness, versus 32.42%
for the independently trained write-disabled control and 32.62% under acute
write suppression. Its counterfactual word-to-bodily-kind accuracy and delayed
target localization are both 100%. It therefore learns and causally uses a
persistent lexical self-model, but does not inspect enough to exploit it.

The earlier observation-branching planner correctly found that a label is not
worth its acquisition cost in a one-shot round. The exact environment audit
then showed that the same mapping becomes valuable when reused across four or
more recurring bodily demands. The missing operation is consequently a backup
of a hypothetical lexical write across recurring future self-states.

This is the single mechanism-level follow-up allowed by the persistent
childhood preregistration. No capacity, optimizer, reward, entropy, label
traffic, training budget, or hidden target is changed.

## Mechanism

Add a persistent self-query value-of-information backup to the existing
observation-branching planner.

For every currently visible inspect option:

1. predict the public distribution over the three possible caregiver label
   packets with the learned token head;
2. for each packet, apply the organism's normal learned token encoder and
   surface-keyed lexical write;
3. complete the public padding-only return;
4. score the best terminal consume consequence for the current sensed body;
5. query the same hypothetical persistent memory under the two recurring
   learner-observable bodily contexts, hungry and thirsty, and average the
   best learned bodily consequence in those contexts; and
6. add seven future reuse queries, matching the public eight-round task
   horizon.

An immediate consume action receives the same seven future self-context
queries under the *current* memory. Thus only the expected improvement caused
by a hypothetical observation can offset inspection's real metabolism and
delay cost. No label, inspection, curiosity, novelty, correctness, hidden
kind, or simulator counterfactual is rewarded.

The two query bodies are public task observations: one need is 0.55, the other
three needs are 0.75. Object surfaces and positions are copied from the
learner's current vector. The mechanism never reads the simulator mapping.

An already valid surface binding is treated as known under the stationary
caregiver protocol, so reinspecting it falls back to the ordinary costly
inspect rollout. This gate reads only the organism's own explicit memory-valid
bit.

## Locked read-only test

Use the corrected write-enabled seed-1 checkpoint:

```text
runs/organism/probe24_persistent_childhood_corrected/
organism_grounded_choice30000x3h40r6n0.55q8_options_inspect_bind16_replay256x1_seed1.npz
```

First run 300 fixed held-out initial contexts with reuse count 7. Compare:

- intact hypothetical lexical writes;
- acute write suppression; and
- collapsed all-padding label branches.

Feasibility passes only if all hold:

1. intact mean best-inspect advantage over immediate consumption is positive;
2. at least 75% of contexts have positive intact inspect advantage;
3. write suppression reduces mean inspect advantage by at least 0.05;
4. collapsed labels reduce mean inspect advantage by at least 0.05;
5. at least 60% of inspect options support label-contingent terminal choices;
6. a matching-resource branch selects the inspected target at least 60%; and
7. a danger branch avoids the inspected target at least 90%.

If feasibility passes, activate the planner read-only on the same checkpoint
for 300 fixed eight-round held-out lives. Scale is the repository's existing
locked value 6.0. Report grounded, silent, independently shuffled, acute
write-suppressed, and planner-removed behavior on identical seeds.

Behavioral promotion requires:

1. at least 60% aggregate grounded correctness;
2. at least 65% correctness in each of rounds 3-8;
3. grounded exceeds planner-removed correctness by at least 15 points;
4. acute write suppression reduces correctness by at least 15 points; and
5. both acute silence and acute shuffle reduce correctness by at least
   15 points.

If the read-only mechanism fails, do not train it or tune reuse count/scale on
seed 1. If it passes, one fresh matched write-enabled/write-disabled training
pair may be run with the same 30,000-tick configuration and the planner
activated only after 15,000 ticks. Seed 2 and larger compute remain forbidden
until the read-only and fresh seed-1 behavioral gates pass.

## Claim boundary

A pass would establish learned, causal value-of-information planning over
counterfactual future bodily states using persistent lexical memory. It would
not establish consciousness, phenomenology, unrestricted reflection, or
truthful generated self-report.
