# Probe77: randomized sequential occlusion fails at a lived budget

Date: 2026-09-30. Preregistration:
`2026-09-30-sequential-structure-preregistration.md`. Module:
`src/homesocial/organism/sequential_structure.py`.

## Decision

**All five continuation gates fail at both budgets.** Close randomized
predecessor-effect contrasts at one and six ordinary episodes. No online
state-estimator treatment or integrated discovery claim is licensed by this
method. This does not reopen Probe61 or invalidate Probe76's forked result.

Five blocks x 20 identities x K1–K5 = **500 bodies**. The instrument now pays
for sequential experience: shock probability 0.025, unknown opaque actions,
ordinary six-tick help, and death truncation. The six-episode condition retains
the same identity and pools evidence across fresh births; it is not a single
uninterrupted life. No simulator forks or true state enter inference. This
still uses the nonspatial expanded-body substrate and automatic absorption of
offered grants; it does not test navigation or the frozen motor policy.

| reachable resources | exact partition, one episode | exact partition, six episodes | actual ticks, one episode | usable grant effects, one episode | actual ticks, six episodes |
|---|---:|---:|---:|---:|---:|
| 1 | 4% | 36% | 222.65 | 36.48 | 1401.05 |
| 2 | 0% | 6% | 131.28 | 21.26 | 834.30 |
| 3 | 0% | 1% | 82.18 | 13.08 | 506.09 |
| 4 | 0% | 0% | 66.92 | 10.52 | 382.98 |
| 5 | 0% | 0% | 52.26 | 8.09 | 297.07 |

Nominal budgets are 400 and 2,400 ticks. The agent cannot spend most of that
budget because it dies. At K5, even six episodes yield only **45.92** usable
grant effects on average. Only **1.11%** of directed pairs of actionable IDs
have the estimator's minimum supporting observations there. A ten-action
random policy wastes grants on inactive IDs at smaller K and does not reliably
replenish the urgent resource at larger K.

### Post-result budget check (not a preregistered gate)

Death is not the only bottleneck. Even an uninterrupted 400-tick life has only
66 absorbed grants. For 66 i.i.d. draws from ten IDs, an exact finite-state
calculation gives only **0.0260919** probability that a specified ordered pair
of distinct IDs appears at least three times. That is a necessary, not
sufficient, condition for this estimator to support that directed contrast.
The calculation tracks whether the preceding ID was `a` and the number of
`a,b` transitions, capped at three; each next draw is `a` with probability .1,
`b` with probability .1, and another ID with probability .8. It assumes perfect
survival and ignores additional active-ID and variance requirements.

This post-result check strengthens the methodological limitation: merely
keeping the same random policy alive would not make its one-episode contrast
matrix well supported. A next proposal needs more efficient intervention
selection or statistical sharing, not just longer survival. This calculation
does not change any gate or upgrade the result into an impossibility theorem.

## What the estimator does with that experience

Mean inferred group counts at the six-episode budget are **1.57, 3.20, 4.97,
6.16, 7.30**, against K1–K5. Counts respond to the active-action set but largely
fail to merge aliases: detecting that an action does something is easier than
identifying what else affects the same hidden resource. The one-episode
counts **0.98, 1.11, 0.99, 0.89, 0.76** do not track K at all.

Held-out effect prediction improves over a per-action mean by only **0.612%**
on average at six episodes (block changes **+1.721%, −0.238%, +0.111%, +0.764%,
+0.701%**), well below the locked 10% continuation threshold. One-episode
improvements average approximately **0.0012%**, effectively absent.

Shuffled effect pairing recovers zero exact partitions at every K. This is
consistent with the intended control, but the treatment's gain is below the
locked 20 percentage points except at K1 in the six-episode condition.
Consistent ID permutation transports the inferred groups exactly for all
1,000 identity/budget combinations.

Fresh six-episode evidence after physical remapping recovers **6%, 0%, 0%, 0%**
of partitions at K2–K5. A stale wrong answer is not evidence of adaptation:
the new evidence must recover the changed truth, and it does not.
Unreachable challenges keep total bodily dimension explicitly unresolved.

## Verification and costs

The separate five-world diagnostic used 4,234 physical ticks and 0.039 seconds
of recorded execution. The locked survey, including held-out, remapping and
unreachable episodes, used **765,200 actual physical ticks**, finishing in
**23.93 seconds** including atomic checkpoint writes. Nine mechanism guards
pass, covering inference with unequal group sizes, death, delayed absorption,
past-only drift estimation, held-out exclusion, seed separation, permutation,
source-pinned resume and locked grading. No diagnostic record enters a gate.
Full repository regression after both new instruments: **602 tests and 13
subtests pass**, in 114.23 seconds while the changing-body survey runs.

Artifacts: `runs/organism/probe77_sequential_structure/{diagnostic,survey}.json`.
The full survey stores every grant-effect transcript and manifest; large raw
records remain ignored. A compact public summary retains block and K metrics.
The standard-library `scripts/summarize_probe77.py` reconstructs all aggregate
gates from the per-body scores and agrees with the runner. It verifies
aggregation, not the grouping algorithm or physical simulator independently.

## What follows

The immediate obstacle for this candidate is collecting informative action
pairs while remaining alive, followed by weak inference from the available
unpaired data. This is a measured limitation of **uniform random exploration
plus predecessor contrasts**, not an information-theoretic impossibility
result. The scalar mean is available each tick; another learner may use more
of its history, directed exploration, or additional naturally available
observations. Those would be distinct mechanisms with distinct budgets and
controls, not repairs by adding episodes or relaxing these gates.

Proceed with the independently motivated changing-body memory ceiling. Do not
claim that successful stationary rate memory depends on this discovery method.

```bash
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.sequential_structure \
  --out /tmp/probe77-survey.json
```
