# Homeostatic terminal-utility result

Date: 2026-07-25
Decision: the utility correction is validated and kept; part one of its
preregistration nevertheless **fails** on gate 6, so part two is not run and the
branching planner is not promoted. The diagnosis produces the strongest
substrate result the project has: an acquired lexical mapping drives perfect
embodied choice for at least six later rounds of the same life.

## Integrity

The corrected utility and all seven feasibility gates were locked in
`2026-07-25-homeostatic-terminal-utility-preregistration.md` before the
mechanism was implemented or any corrected planner output was read.

Checkpoint (unchanged, write-enabled, seed 1):
`runs/organism/probe24_persistent_childhood_corrected/organism_grounded_choice30000x3h40r6n0.55q8_options_inspect_bind16_replay256x1_seed1.npz`,
SHA-256 `02bb27563d6af08e1255e0d8d48e4a46e048d4ae48be54b2fd0620827cac728e`.

Feasibility CSV:
`runs/organism/probe26_urgent_deficit_feasibility/harness_seed1_grounded.csv`.

No parameter was updated in any measurement below.

## Part one: feasibility

300 fixed held-out contexts from seed `1_700_000`, reuse count 7.

| Gate | Requirement | Minimum rule | Corrected rule | Decision |
|---|---:|---:|---:|---|
| 1 Mean intact inspect advantage | > 0 | +0.2345 | +0.2584 | Pass |
| 2 Positive-advantage contexts | >= 75% | 99.67% | 100.00% | Pass |
| 3 Write-suppression advantage drop | >= 0.05 | 0.0431 | 0.2137 | Pass |
| 4 Collapsed-label advantage drop | >= 0.05 | 0.0478 | 0.2261 | Pass |
| 5 Label-contingent terminal choices | >= 60% | 46.78% | 89.22% | Pass |
| 6 Matching label selects the target | >= 60% | 2.11% | 19.11% | **Fail** |
| 7 Danger label avoids the target | >= 90% | 99.67% | 99.67% | Pass |

Six of seven gates pass, and the three that previously failed on causal
attribution now pass by four to five times their thresholds. Gate 6 remains
below chance. Per the preregistration, part two is not run.

## Diagnosis, all read-only

### The corrected utility is right; the imagined branch is wrong

Real inspections, real heard labels and the real six-tick return were executed
in the real environment on 200 fixed contexts, and the terminal consume score
was read at the true choice pose.

| Case, true label heard | Selects the inspected object |
|---|---:|
| The object really is the demanded resource | **100.00%** |
| The object is the other resource | 0.00% |
| The object is poison | 0.00% |

That is perfect label-conditioned embodied choice after a delay, from a
checkpoint whose deployed behavior is 38.17%. The same states scored with the
deployed minimum rule give 31.50%, i.e. chance. The corrected utility is
therefore not a fitting knob; it is the difference between reading the
organism's self-model and not reading it.

### One factor at a time from the real path to the branch

Each row replaces exactly one component of the real path with its imagined
counterpart, on the matching-label case, 150 contexts.

| Path | Selects the demanded object |
|---|---:|
| 0 real everything | 1.0000 |
| 1 canonical `this <kind>` tokens instead of the sampled bank utterance | 1.0000 |
| 2 explicit surface-keyed write instead of the pose-derived write | 1.0000 |
| 3 constructed public return observation instead of the real one | 1.0000 |
| 4 **decoder-reconstructed post-inspect observation** | **0.3600** |
| 4a real needs with reconstructed visual/pose part | 0.3667 |
| 4b reconstructed needs with real visual/pose part | 1.0000 |
| 5 all of the above together, i.e. the deployed branch | 0.2067 |

Tokens, the explicit write and the constructed return are all harmless. The
single responsible component is the freely decoded post-inspect observation,
and within it the visual/pose part, not the predicted needs. Its mean L1
reconstruction error against the real observation is 13.55. This is expected
rather than surprising: the world model is trained with a next-observation
weight of 0.1 and a bodily-change loss boost of 20, so it was never optimized
to reconstruct a scene well enough to be re-encoded.

Substituting the learner's own current geometry instead does not help
(0.3867), and neither does writing the memory row without any observation step
(0.1867).

### The acquired mapping is persistent and sufficient on its own

A scripted read-only driver acquired one label in round 1 and one in round 2 of
an eight-round life, then never inspected again. At each later round's choice
pose the terminal consume score was read under the corrected utility.

| Round | Picks the needed object | Valid memory rows |
|---:|---:|---:|
| 3 | 100.00% | 2.00 |
| 4 | 100.00% | 2.00 |
| 5 | 100.00% | 2.00 |
| 6 | 100.00% | 2.00 |
| 7 | 100.00% | 2.00 |
| 8 | 100.00% | 2.00 |

100 lives per round. Two words, acquired once by ordinary embodied experience,
determine the correct consumption choice in every subsequent recurring bodily
demand of that life, including when the demand alternates between the two
words' referents. This is the persistent, causally grounded, object-local
semantic self-model the pivot set out to build, now shown to be behaviorally
sufficient rather than only decodable.

### Why an injected memory row alone is not enough

An externally injected memory row needs several observation steps before it
influences the recurrent core:

| Public settling observations after the injected write | Selects the demanded object |
|---:|---:|
| 0 | 18.67% |
| 2 | 63.33% |
| 3 | 66.67% |
| 5 | 81.33% |

The real task supplies exactly such steps between hearing a label and reusing
it in a later round, which is why real acquisition reaches 100% while a
zero-step hypothetical write reaches chance.

## Interpretation

Three separate things were conflated by the previous failures and are now
distinguished:

1. **The utility.** The minimum-need rule cannot express a resource choice
   after a delay. Corrected, and validated at 100% against 31.5%.
2. **The mapping.** Real acquisition produces a persistent surface-keyed
   lexical memory that alone drives perfect choice for at least six later
   rounds. This is a positive substrate result.
3. **The counterfactual.** Valuing an inspection requires imagining an
   observation the organism has never been trained to reconstruct. This is
   what fails, and it fails for a reason internal to the world model's loss
   weighting, not to the semantics.

The negative result therefore propagates up as required: the obstacle is not
the utility and not the memory, but the attempt to fabricate a sensory scene
inside the planner. The next experiment must value inspection without
reconstructing any observation, using only the public post-label protocol,
whose observations are already known to be safe (row 3) and which supplies the
settling steps the architecture needs.

## Decision

Keep the corrected utility, which is now selectable and off by default so every
sealed artifact stays reproducible. Do not run part two, do not promote the
current branch, and do not search further over utility rules. The successor
mechanism is preregistered separately in
`2026-07-25-protocol-branch-planner-preregistration.md`.

No compute, API, or data-generation request is justified by this result.
