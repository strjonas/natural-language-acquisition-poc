# Protocol-branch planner result and cross-round reuse result

Date: 2026-07-25
Decision: the protocol branch **fails** gate 6 and is not promoted; the stop
rule is honoured and no fourth planner is implemented on this task. The
confirmatory cross-round reuse audit, which does not depend on any planner,
**passes all three of its gates at ceiling** and is the headline result.

## Integrity

The mechanism, the settling count, the seven feasibility gates and the three
confirmatory gates were locked in
`2026-07-25-protocol-branch-planner-preregistration.md` before implementation.

Checkpoint (unchanged, write-enabled, seed 1):
`runs/organism/probe24_persistent_childhood_corrected/organism_grounded_choice30000x3h40r6n0.55q8_options_inspect_bind16_replay256x1_seed1.npz`,
SHA-256 `02bb27563d6af08e1255e0d8d48e4a46e048d4ae48be54b2fd0620827cac728e`.

Artifacts:

- `runs/organism/probe27_protocol_branch_feasibility/harness_seed1_grounded_bind16_plan6h2.csv`
- `runs/organism/probe28_cross_round_reuse/harness_seed1_grounded_bind16.csv`

No parameter was updated in either run. All 240 local tests pass.

## Part one: feasibility

300 fixed held-out contexts from seed `1_700_000`, reuse count 7.

| Gate | Requirement | Reconstructive branch | Protocol branch | Decision |
|---|---:|---:|---:|---|
| 1 Mean intact inspect advantage | > 0 | +0.2584 | +0.1735 | Pass |
| 2 Positive-advantage contexts | >= 75% | 100.00% | 99.00% | Pass |
| 3 Write-suppression advantage drop | >= 0.05 | 0.2137 | 0.2371 | Pass |
| 4 Collapsed-label advantage drop | >= 0.05 | 0.2261 | 0.2371 | Pass |
| 5 Label-contingent terminal choices | >= 60% | 89.22% | 99.89% | Pass |
| 6 Matching label selects the target | >= 60% | 19.11% | **18.00%** | **Fail** |
| 7 Danger label avoids the target | >= 90% | 99.67% | 100.00% | Pass |

Removing the fabricated observation worked as intended on everything it was
meant to fix: branch contingency is now essentially perfect and collapses to
exactly zero without a write, and inspection's value is entirely write-caused
(without the write the mean advantage is negative, -0.0635, and only 9% of
contexts are positive). Gate 6 did not move.

## Why gate 6 fails, measured

The cause is neither the utility, nor the memory, nor the fabricated scene. It
is the world model's own bodily forecast over the ten-tick detour. The
corrected utility scores the need the organism observes as most urgent, so it
is only as good as the organism's prediction of which need that will be:

| Body state the urgent need is read from | Names the demanded resource |
|---|---:|
| The real current observation | 100.00% |
| The model's predicted post-inspect body | 60.67% |
| The model's predicted post-return body | 18.44% |

Gate 6 came in at 18.00%. The branch chooses correctly almost exactly when its
predicted body still ranks its own needs correctly. The organism's semantics
are intact; its ten-tick metabolic forecast is not. This is consistent with the
training weights, where next-observation prediction carries weight 0.1 and the
bodily-change loss is boosted 20-fold on consumption events, so slow metabolic
drift over a long detour is the least-supervised quantity in the model.

Per the stop rule, no further planner is implemented, and the settling count,
reuse count, planning scale and utility are left untouched.

## Confirmatory result: an acquired word steers choice for the rest of the life

300 fixed lives, eight rounds each, 1,800 measured rounds per condition. A
scripted read-only driver acquires one food label and one water label in rounds
1-2 and never inspects again; every later round is decided by reading the
organism's own terminal consume scores.

| Round | Grounded | Acute silence | Acute write suppression |
|---:|---:|---:|---:|
| 3 | 100.00% | 36.67% | 37.00% |
| 4 | 100.00% | 30.67% | 31.00% |
| 5 | 100.00% | 36.67% | 37.00% |
| 6 | 100.00% | 31.33% | 31.00% |
| 7 | 100.00% | 37.33% | 37.33% |
| 8 | 100.00% | 31.33% | 31.33% |
| **Aggregate** | **100.00%** | **34.00%** | **34.11%** |

Valid memory rows: 2.00 grounded, 0.00 in both controls. All 300 lives
completed all eight rounds in every condition.

| Gate | Requirement | Result | Decision |
|---|---|---:|---|
| Rounds 3-8 each | >= 90% | 100.00% each | Pass |
| Acute write suppression | <= 45% | 34.11% | Pass |
| Acute silence | <= 45% | 34.00% | Pass |

Two words, heard once each during ordinary embodied experience, determine the
correct consumption choice in every one of six subsequent recurring bodily
demands, with the demand alternating between the two words' referents. Removing
the language leaves chance. Removing only the writes, with the language still
present, leaves chance. The effect is carried by the persistent object-local
lexical bank, not by the utterance being audible at decision time.

## What this establishes and what it does not

Established: a continually trained embodied organism has a persistent,
object-local, causally load-bearing semantic memory that converts heard words
into correct actions on its own body, across a life, long after the episode in
which the words were acquired. Under the corrected homeostatic utility this is
behavioral sufficiency at ceiling, not merely a decodable representation.

Not established: the organism does not yet *choose* to acquire those words. Its
deployed inspection rate is 14%, and the missing operation is still the value of
information. Three planners have now failed to supply it, and the diagnosis has
converged on a single cause that is not a planner at all: the bodily forecast
over the acquisition detour degrades until the organism can no longer tell which
of its own needs will be most urgent when it returns.

Nothing here is reflection, generated self-report, or consciousness.

## Next question, one level up

The next experiment must address the forecast, not the planner. The two
candidates are (a) supervising the slow metabolic drift the current loss
neglects, so that a ten-tick forecast preserves need ordering, and (b) letting
the organism learn the value of knowing directly from experienced cross-round
reuse under a longer budget, since the environment now genuinely pays for it.
Option (a) is a loss-weighting change to the world model and is cheap to test
locally; it should be preregistered first, with the same gate 6 as its endpoint.

No compute, API, or data-generation request is justified by this result.
