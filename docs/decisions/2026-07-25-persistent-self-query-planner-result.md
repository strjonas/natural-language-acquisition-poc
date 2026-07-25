# Persistent self-query planner result

Date: 2026-07-25
Decision: the preregistered mechanism fails its feasibility gates. Do not train
it. The read-only diagnosis locates the cause in the planner's terminal utility,
not in the organism's learned representation.

## Integrity

The mechanism, the checkpoint, the fixed context set, and all seven feasibility
gates were locked in
`2026-07-25-persistent-self-query-planner-preregistration.md` before the audit
was implemented in the harness or run.

Target checkpoint (write-enabled, corrected persistent childhood, seed 1):

`runs/organism/probe24_persistent_childhood_corrected/organism_grounded_choice30000x3h40r6n0.55q8_options_inspect_bind16_replay256x1_seed1.npz`

SHA-256: `02bb27563d6af08e1255e0d8d48e4a46e048d4ae48be54b2fd0620827cac728e`

Result CSV:
`runs/organism/probe25_persistent_self_query/harness_seed1_grounded.csv`

No parameter was updated and no environment action was taken.

## Feasibility outcome

300 fixed held-out contexts from seed `1_700_000`, reuse count 7.

| Gate | Requirement | Result | Decision |
|---|---:|---:|---|
| 1 Mean intact inspect advantage | > 0 | +0.2345 | Pass |
| 2 Positive-advantage contexts | >= 75% | 99.67% | Pass |
| 3 Write-suppression advantage drop | >= 0.05 | 0.0431 | Fail |
| 4 Collapsed-label advantage drop | >= 0.05 | 0.0478 | Fail |
| 5 Label-contingent terminal choices | >= 60% | 46.78% | Fail |
| 6 Matching label selects the target | >= 60% | 2.11% | Fail |
| 7 Danger label avoids the target | >= 90% | 99.67% | Pass |

Gate 6 is not merely below threshold; it is far below the 33% chance rate. With
any label written, the planner's imagined terminal choice moves *away* from the
labeled object. Gate 7 passes for the same reason rather than for a semantic
one. Candidate-prior entropy was healthy at 0.9206.

## Read-only diagnosis

Two further read-only measurements were taken on the same checkpoint. Both
executed real inspections and the real forced return in the real environment;
neither trained anything.

### The imagined chain is not the defect

For 100 contexts x 3 objects, each object was really inspected, its true label
really heard, and the six-tick return really played out. At the resulting true
choice-pose observation the learned terminal consume score was read:

| Path | Picks the needed object |
|---|---:|
| Real inspection, real label, real return | 31.67% |
| Imagined planner branch from the same state | 35.33% |

The imagined branch reproduces the real state's behavior. In both, the labeled
object scores *below* the two unlabeled ones for every kind (real: food 0.4662
vs 0.4855; water 0.4777 vs 0.4960; poison 0.4616 vs 0.4923).

### The defect is the terminal utility

At the identical true post-label states, the same learned consequence
predictions were rescored under different utilities. 200 contexts, 600 labeled
returns. Exactly one object carries a label, so a perfect single-label reasoner
reaches 1/3 + 2/3 x 1/2 = **66.67%**.

| Terminal utility | Picks the needed object |
|---|---:|
| Minimum predicted need (deployed) | 31.50% |
| Homeostatic drive, n=8 | 34.33% |
| Learned critic value | 36.67% |
| Homeostatic drive, n=4 | 39.67% |
| Homeostatic drive, n=2 | 48.83% |
| Mean predicted need | 49.17% |
| Deficit-weighted predicted gain | 50.17% |
| Predicted reward head | 50.33% |
| **Predicted level of the currently lowest need** | **68.83%** |

The last rule is at the single-label information ceiling. It is fully
learner-observable: the currently lowest need is read from the organism's own
interoception, and in 100.00% of these contexts it is exactly the demanded
resource need. No hidden kind, correctness signal, or simulator fact is used.

### Why the deployed utility is degenerate here

The deployed score is the predicted minimum over the four needs after acting,
i.e. the `n -> infinity` limit of a homeostatic drive. At the choice pose the
mean observed needs are `[0.55, 0.51, 0.555, 0.83]`: the inspect-plus-return
detour costs ten primitive ticks, and energy has decayed to roughly the level of
the demanded resource need. After that delay the post-consumption minimum is
set by energy, which no consumption choice affects. The score is therefore
almost constant across the three objects.

Worse, it is biased against knowledge. For an unlabeled object the model
predicts a smeared average delta spread over several needs, which raises the
minimum more than a correct, concentrated single-need increase does. That is
the exact mechanism producing the 2.11% on gate 6.

The upper-bound audits of `2026-07-25-persistent-mapping-rent-result.md` remain
valid: they measured final minimum need over a whole round, where an
immediate four-tick consumption leaves the other needs at 0.67 and the minimum
still discriminates. The degeneracy appears only after a delay long enough for a
non-consumable need to become the minimum.

## Interpretation

The organism's learned self-model is not the bottleneck. At the true post-label
state it already supports optimal embodied choice given one label. The planner
that reads that model asks it a question whose answer cannot depend on the
label.

This also explains the earlier one-shot feasibility failure recorded in
`2026-07-25-observation-branching-and-rent-result.md`, whose gate 3 sat at
64.11% under the same utility, and it is consistent with the persistent
childhood's flat inspection rate: nothing in the value path could reward
knowing.

## Decision

Do not train the self-query planner as specified, and do not tune its reuse
count or scale. Correct the terminal utility, which is a defect shared by every
planner in the repository, and re-run the identical locked gates without
weakening them. That correction is preregistered separately in
`2026-07-25-homeostatic-terminal-utility-preregistration.md`.

No compute, API, or data-generation request is justified by this result.
