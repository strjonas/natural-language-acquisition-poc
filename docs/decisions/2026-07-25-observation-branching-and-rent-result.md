# Observation branching and delayed information-rent result

Date: 2026-07-25  
Decision: do not train the observation-branching planner on the one-shot
delayed substrate

## Integrity

The observation-branching mechanism and its gates were locked in
`2026-07-25-observation-branching-feasibility-preregistration.md` before the
sealed checkpoint was evaluated.

The first sandboxed command terminated before model loading because Metal was
unavailable. The authorized Metal execution was the first interpreted run and
completed all 300 fixed contexts. Its CSV is:

`runs/organism/probe19_observation_branching_feasibility/harness_seed1_grounded-silent-shuffled.csv`

The exact-environment upper bound was then separately locked in
`2026-07-25-delayed-choice-information-upper-bound-preregistration.md`. The
authorized execution completed 10,000 fixed contexts. Its CSV is:

`runs/organism/probe20_information_upper_bound/harness_seed1_grounded-silent-shuffled.csv`

Neither run updated model parameters or acted in a learning environment.

## Learned-planner feasibility

| Gate/endpoint | Requirement | Result | Decision |
|---|---:|---:|---|
| Candidate entropy | >= 0.80 | 0.9069 | Pass |
| Contingent branches | >= 80% | 81.22% | Pass |
| Matching label selects target | >= 80% | 64.11% | Fail |
| Danger label avoids target | >= 80% | 100.00% | Pass |
| Mean inspect advantage | > 0 | -0.1264 | Fail |
| Positive-advantage contexts | >= 60% | 0.00% | Fail |
| No-write advantage drop | >= 0.02 | -0.0009 | Fail |
| No-write contingency drop | >= 30 points | 52.78 points | Pass |
| Collapsed-label inspect advantage | <= 0 | -0.1167 | Pass |

The model's observation prior covers all labels, and the learned object-local
write makes downstream choices branch on those labels. This is causal:
disabling only the write reduces contingency from 81.22% to 28.44%.
Collapsing all utterances to padding reduces it to zero.

But the branches are not sufficiently body-directed, and inspection is
strictly dominated in all 300 contexts. The representation changes imagined
action, but not in a way that earns back physical cost.

## Exact-environment upper bound

| Policy | Final min need | Ticks | Correct | Poison | Gain vs blind |
|---|---:|---:|---:|---:|---:|
| Blind immediate | 0.5583 | 4.00 | 33.55% | 32.70% | 0.0000 |
| One inspection | 0.4436 | 14.00 | 66.20% | 17.20% | -0.1147 |
| Two inspections/elimination | 0.3454 | 20.65 | 100.00% | 0.00% | -0.2129 |
| Clairvoyant immediate | 0.6700 | 4.00 | 100.00% | 0.00% | +0.1117 |

The audit is valid because the clairvoyant ceiling exceeds blind by 0.1117 and
achieves 100% correct choices. Nevertheless, truthful inspection loses more to
metabolism than it gains from avoiding uncertainty. Even an exact policy that
always obtains the correct object after at most two labels is much worse than
blind immediate consumption under the current final-body utility.

## Interpretation

The one-shot childhood structurally cannot teach instrumental language
acquisition. It charges ten primitive ticks to acquire a label, then erases the
entire surface-kind mapping at the end of the single consumption. An
object-local persistent memory has no opportunity to amortize acquisition.

This explains both the sealed training behavior and the learned-planner
feasibility failure without invoking capacity:

- actor-critic correctly drifts away from inspection;
- a mean-observation planner cannot value information;
- an observation-branching planner can represent contingent choices but
  correctly assigns negative net value; and
- adding inspect, curiosity, or language reward would contradict the exact
  environment utility.

## Decision and next experiment

Do not train this planner on the one-shot task and do not tune planning scale,
model size, or label reward.

Move the negative result up to the substrate. The next controlled childhood
must preserve one hidden surface-kind mapping across multiple bodily choice
rounds in one continuing life. A label should be paid for once and reused when
hunger and thirst recur. Before implementing that environment, run an
exact-dynamics persistent-mapping upper bound at fixed round counts to verify
the information cost is actually amortized.

The observation-branching code remains a tested mechanism but is not promoted
to training. No compute or data request is justified.
