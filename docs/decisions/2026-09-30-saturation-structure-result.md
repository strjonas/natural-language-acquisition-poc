# Probe76: scalar interactions identify reachable restoration groups

Date: 2026-09-30. Preregistration:
`docs/decisions/2026-09-30-saturation-structure-preregistration.md`. Module:
`src/homesocial/organism/saturation_structure.py`. Artifact:
`runs/organism/probe76_saturation_structure/survey.json`.

**All six locked continuation gates pass.** From scalar branch sensations and
opaque actuator IDs alone, interaction-based inference recovers the exact
reachable-resource count and partition in **25/25 worlds**: five independent
blocks at each K1–K5. Mean recovered counts are **1, 2, 3, 4, 5**. Held-out
overlap prediction is correct on **750/750 actuator/representative pairs**.

This establishes an identification route for *reachable restoration groups*
within an authored bounded-resource physical family. It does not establish
online state-variable discovery by a living organism. The experiment permits
matched forks and removes shocks; those privileges remain the next obstacle.

## Locked gates and controls

| gate | observation | verdict |
|---|---|---|
| G1 count | exact K on 5/5 blocks for every K; means 1 < 2 < 3 < 4 < 5 | pass |
| G2 partition | exact active actuator set and target-equivalence partition, 25/25 | pass |
| G3 held-out prediction | every world has accuracy and both recalls 1.000; 150 true positives, 600 true negatives, zero errors | pass |
| G4 ID permutation | counts invariant and partitions transport exactly, 25/25 | pass |
| G5 destroyed sensory pairing | exact active-set/partition conjunction fails 25/25; every shuffled inference collapses to one component | pass |
| G6 physical remapping | stale grouping mispredicts the moved alias, 20/20 K2–K5 worlds; fresh inference recovers changed partition, 20/20 | pass |

The shuffled control's count of one coincides with K1. Its active-set/partition
is nevertheless wrong on all five K1 blocks; this is why the precommitted
control tests the conjunction rather than count alone. Relabeling the actions
preserves the mechanism, while reassigning a small restorative action to a
different physical axis invalidates it. The remapped model also predicts
held-out overlap perfectly: **700/700 pairs**, 140 positive and 560 negative.
No gate, tolerance, sample or context budget was changed after a survey context.

An independent standard-library reconstruction, importing no inference code,
verifies every grouping, held-out confusion matrix, remapping, control and gate.
It also reconstructs all 532,400 branch outputs from tagged physical draws and
closed-form thirteen-tick dynamics: maximum mean/minimum discrepancy is
**8.88e-16**, minimum final viability **0.0182053**. Thus the scalar interaction
result is checked against the simulator arithmetic as well as its own summaries.

## What the learner receives and computes

All worlds expose the same ten opaque actuator IDs and scalar mean/minimum
sensations. Active subsets and physical ID assignments vary across blocks.
Inactive slots remain full and inert. The scalar mean always divides by five,
even at K1: its scaling cannot supply K. Inference receives no axis names,
dimensions, portion sizes, individual rates, state vectors or target mapping.
The fixed physical family retains the original six birth levels, uniform
metabolic multipliers [0.4,1.6], resting rates and 0.20/0.60 portions.

A context has 121 unique thirteen-tick branches: one no-help branch, ten
first-only branches, ten second-only branches and one hundred paired branches.
Requests occur at ticks 6 and 12 and are absorbed at ticks 7 and 13. Sensations
are read after full physical ticks. No branch dies; the runner asserts this.

For each action pair the learner computes

`C[a,b] = mean(AB) - mean(A0) - mean(0B) + mean(00)`.

Different independent resources give zero interaction. Restorations of the same
resource can compete at the upper bound, giving a negative interaction. Active
IDs are identified by single-action effects above the locked numerical tolerance
1e-8. Negative interactions connect IDs into components across forty development
contexts. No two-alias group size or predetermined count is assumed. The tests
also check unequal synthetic group sizes, so dividing the active actuator count
by two cannot substitute for inference.

Representatives are selected from development by single-action effect. Forty
disjoint contexts then test whether group membership predicts causal overlap.
Held-out observations do not enter grouping or representative selection.
This predicts whether interventions can occlude, not a future state vector.

## Unreachable variables remain unresolved

Twenty additional K1–K4 challenges activate one extra metabolic slot while
blocking both of its actuators. In **20/20**, inference recovers the reachable
partition and returns K groups, while actual bodily count is K+1. Every record
explicitly marks **total bodily dimension unresolved**.

This is a limitation observed in the experiment, not a successful discovery of
the extra variable. Multiple unactuated variables can share an aggregate drift,
and a variable that never becomes the minimum can remain observationally hidden.
Do not equate an action-equivalence partition with arbitrary total state count.

## Evidence budget, independence and verification

Five blocks × K1–K5 × forty development and forty held-out contexts. K2–K5
add forty remapping-development and forty remapping-test contexts; K1–K4 add
forty unreachable-variable contexts. Permutation and sensory-pairing controls
reuse existing transcripts rather than spending fictional physical ticks.

Total actual budget: **4,400 contexts, 532,400 unique physical branches and
6,921,200 ticks**. These support 440,000 four-branch contrasts with reused
controls. The blocks, not their contexts or branch references, are learner
replication units. Full recorded execution time is **53.46 seconds** locally;
the two-context diagnostic used a separate band and was excluded from gates.

Main base 3,800,000,000, block stride 2,000,000, world stride 100,000;
development +1,000, held-out +10,000, permutation +20,000, shuffled +30,000,
remapping-development +40,000, remapping-held-out +41,000, unreachable +50,000.
Diagnostic base 3,850,000,000. Tagged SHA-256 local generators separate birth,
metabolism, mapping and control domains. The pre-run salt-collision correction
is recorded in the preregistration and occurred before any diagnostic or survey
data. The guard validates **4,444 physical context seeds and 14,450 derived
domain seeds**, including the diagnostic; matched forks deliberately reuse one
context's physical draws.

The JSON preserves all scalar evidence separately from audit-only physical
identities. Progress saves complete block/world units atomically and pins
configuration, adapter, substrate and preregistration. A completed real survey
resumed without regenerating a world. Source hashes include:

- Adapter: `f6a6487cab511e64e214a726a70220a44ef8f31c3c920476f5099df37a8d3d80`.
- Substrate: `34c141661a7591a0737271453c557847dd5777cdd254788ab0ea7a1f5acda214`.
- Raw artifact: `0de4f5b6f7d460889092a668c553ac87e886751ae4cb0800877e46752c7efab9`.

The raw artifact is about 21.5 MB and remains ignored; these headline numbers
are the durable record. Eleven mechanism guards pass. Full regression:
**579 tests and 13 subtests passed**, in 69.35 seconds while the survey ran.
Existing simulator/global model configuration is unchanged; this module is
an explicitly invoked instrument.

```bash
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.saturation_structure \
  --diagnostic --out /tmp/probe76-diagnostic.json
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.saturation_structure \
  --out /tmp/probe76-survey.json
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Use a fresh output path after source changes. Changed partial runs are refused.

## Claim boundary and direction

The useful difference from Probe61 is causal grouping through saturation,
without fitting and pruning an overcomplete state vector. Its closed mechanism
is not reopened, and Probe71's failed composition/retraining gates remain failed.
The new positive shows which hidden restorative causes are identifiable when
the agent can obtain matched scalar interventions.

It does not produce a state estimator, infer arbitrary dynamics, recover
unreachable dimensions, learn self-recognition, adapt a changing body, guide a
motor controller from a self-model, report uncertainty, generate language or
demonstrate survival value. Exact independent bounded-resource dynamics are
still supplied by the environment. There is no parent developmental replication,
consciousness or sentience claim.

Next: test a sequential exploration mechanism on ordinary bodily episodes with
no matched forks and restored shocks, keeping scalar-only information and a
moved-ground-truth sweep. First establish whether the accessible experience
identifies the grouping at an explicit budget. A noisy or unpaired interaction
test is a new mechanism; it requires preregistration and controls rather than
copying this instrument's perfect scores into an online-organism claim.
