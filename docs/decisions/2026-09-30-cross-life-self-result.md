# Probe74: retained evidence about the same body reaches survival

Date: 2026-09-30. Preregistration:
`docs/decisions/2026-09-30-cross-life-self-preregistration.md`. Runner:
`src/homesocial/organism/cross_life_self.py`. Guards:
`tests/test_cross_life_self.py`. Artifacts:
`runs/organism/probe74_cross_life_self/{granularity_diagnostic,survey}.json`.

**All six locked continuation gates pass.** Holding an individual's physical
constants fixed across episodes and retaining its recursive rate-learning
statistics improves test survival from **0.2271 to 0.2571**: **+0.0300
[+0.0141, +0.0459]**, positive on **5/5 paired blocks**. That is 21 additional
survivors in 700 test episodes. This is a controlled local survey that licenses
independent replication, not a confirmatory treatment or a saturation result.

| gate | measured contrast or endpoint | verdict |
|---|---|---|
| C1: persistent-identity physical ceiling | true − reset survival **+0.0300 [+0.0054, +0.0546], 5/5** | pass |
| C2: retained evidence reaches survival | retained − reset survival **+0.0300 [+0.0141, +0.0459], 5/5** | pass |
| C3: matched regret bridge | reset-word − retained-word regret **+0.001846 [+0.000222, +0.003471], 5/5** | pass |
| C4: less rediscovery at birth | reset − retained birth-rate MAE **+0.003725 [+0.003064, +0.004386], 5/5** | pass |
| C5: memory belongs to the individual | donor − own birth-rate MAE **+0.002788 [+0.001839, +0.003736], 5/5** | pass |
| C6: viable construction | true-rate survival **0.2571**, floor 0.15 | pass |

## 1. Persistent identity has physical headroom, and learned memory uses it

Five blocks x 28 identities x five test episodes give 700 episodes per arm.
Each identity first supplies one shared calibration episode, excluded from all
survival gates. The four metabolic multipliers, including safety, stay fixed
across all six episodes of that identity. Birth state, layout, shocks, help,
reading times and motor randomness are independently seeded each episode.

| request-planning rates | survivors / 700 | survival | mean episode steps |
|---|---:|---:|---:|
| episode-reset recursive rates | 159 | 0.2271 | 150.2 |
| **retained recursive rates** | **180** | **0.2571** | **165.1** |
| true individual rates | 180 | 0.2571 | 167.0 |
| retained rates initialized from another body | 158 | 0.2257 | 146.8 |

| block | reset | retained | true | swapped | retained − reset |
|---|---:|---:|---:|---:|---:|
| 1 | 0.3357 | 0.3786 | 0.4000 | 0.3571 | +0.0429 |
| 2 | 0.1571 | 0.1714 | 0.1786 | 0.1429 | +0.0143 |
| 3 | 0.1929 | 0.2214 | 0.2071 | 0.1786 | +0.0286 |
| 4 | 0.1929 | 0.2143 | 0.2214 | 0.1857 | +0.0214 |
| 5 | 0.2571 | 0.3000 | 0.2786 | 0.2643 | +0.0429 |

The true-rate contrast establishes the physical precondition in the new
persistent-identity construction rather than borrowing Probe73's ceiling from
independently redrawn bodies. The retained-rate contrast is positive in all five
blocks and extends mean episode length by 14.9 ticks. Water deaths fall from
153 to 134 and food deaths from 92 to 84; safety deaths rise from 68 to 76.
The claim is the preregistered net of 21 survivors, not improvement on every
death category.

The true-rate and retained arms happen to have the same aggregate survivor
count. **This does not establish equivalence or exhausted headroom.** Their
paired true-minus-retained contrast is 0.0000 [−0.0217, +0.0217], with three
positive and two negative blocks. The fixed word rule is myopic, and perfect
rates do not guarantee optimal survival. There was no evidence-budget sweep.

## 2. State resets; evidence about the rate does not

Every cell computes an ordinary episodic recursive state/uptake tier, an
own-memory recursive calibrator, and a donor-memory recursive calibrator on
every transition. Planning always reads the **episodic tier's state and uptake**.
Only the rate vector is switched. The own-memory and donor-memory calibrators
retain RLS amplitudes, covariance, metabolic readout accumulators and evidence
counters. They reset the Jacobian and its birth-state anchor every episode.
Nuisance uptake coefficients remain in the regression statistics to identify
rates, but their retained uptake belief never reaches the planner.

No current state, request, movement-fraction estimate or motor hidden state
crosses the boundary. Identity constants are set inside the simulator, not
supplied to the learner. The true-rate cell and audit scoring alone read them.
Tests establish a fresh Jacobian and new birth state with retained statistics,
independent episode streams with identical four-axis constants, no truth access
at learner reset, and interruption-safe resume. A real diagnostic episode also
reproduced the existing reset estimator exactly with fresh rate memory.

Matched machinery does not mean numerically identical downstream closed-loop
states: different words change later trajectories. The counterfactual panel
below holds the exact current state, uptake, ledger and trajectory fixed.

## 3. The same-history rate and regret panel

On retained-cell histories, all counterfactual words use the identical episodic
state, uptake belief, movement fraction, public ledger and horizon. Only rates
differ. Start-of-episode MAE is scored before any new transition or reading.

| endpoint on retained histories | reset-rate readout | retained own-memory readout | donor-memory readout |
|---|---:|---:|---:|
| birth-rate MAE | 0.004734 | **0.001010** | 0.003797 |
| rate MAE per scored tick | 0.000841 | **0.000396** | 0.002158 |
| word regret per scored tick | 0.005500 | **0.003654** | 0.008804 |

Own memory removes **78.7%** of birth-rate error and **52.9%** of scored-tick
rate error relative to episode reset. Its rate-only replacement changes **5.33%**
of scored need words and removes **33.6%** of matched regret. The locked regret
contrast clears zero on all five blocks, although its lower interval bound is
only +0.000222.

The pre-survey eight-life Probe68 diagnostic at lag 24 measured a median
consequential margin of **0.13836**, formula MAE 0.000615, recursive point-error
reach 0.4701 and species-rate state-oracle reach 0.8022. It licensed the unchanged
word surface; the survey's actual word changes and regret measure the new
rate-only effect directly. No portion or help-clock tuning was introduced.

## 4. Memory is about this individual

The swapped cell begins with the next identity's calibration evidence and is
allowed to correct it with its own new readings. It is not frozen and is not
required to collapse. Its survival is 0.2257, against 0.2571 for own memory and
0.2271 for episode reset.

The ungated retained-minus-swapped survival contrast is **+0.0314
[+0.0213, +0.0415], 5/5**. The locked same-history identity contrast also passes:
donor memory has +0.002788 more birth-rate MAE than own memory. Thus the benefit
does not come merely from handing the requester an arbitrary previous body's
calibration. These controls test specificity within one stationary metabolic
population; they do not test recognition of an identity without externally
maintaining the assignment of memories to bodies.

## 5. Evidence and compute budgets

The nominal reset evidence budget is at most one 400-tick episode. The retained
budget is one calibration plus up to five test episodes, at most 2,400 ticks.
Reading probability remains 0.03. All cells compute the same three estimators
and four rate-word counterfactuals; the advantage deliberately comes from
**more relevant past evidence**, not more per-tick inference compute.

The 140 calibration episodes total 20,956 physical ticks and 609 readings:
149.7 ticks and 4.35 readings per identity on average. On retained histories,
test episodes average 165.1 physical ticks and 4.81 processed readings. By the
end of test episode five, own memory carries a mean **969.1 nonterminal evidence
ticks and 28.39 readings**, including calibration. The terminal transition is
not fitted, matching the inherited Probe73 runner. The wrong-identity control
matches the nominal one-episode prior budget; donor and recipient calibration
episodes can have different actual lengths and reading counts.

Episodes within an identity are dependent. All intervals use the **five paired
seed-block values**, not 700 independent identities or model replications.
The new draw has 140 distinct bodies; absolute survival is not a paired
comparison with Probe73's 700 independently drawn bodies.

The survey took about eleven minutes locally. Its atomic JSON records all 140
calibration episodes and 560 completed identity/cell units. A completed real
artifact was resumed successfully without rerunning an episode. The artifact
pins both parent weights and configuration by SHA-256:
`ddd7c5a9e1cdb4274a63a98c078065e55702b6116457c1cbc659a2b8c32135d4`.

## 6. Claim boundary and next work

This advances persistent, load-bearing individual self-knowledge in a narrow
simulated setting: evidence retained about the **same** body improves its rate
estimate at the next birth, changes speech and regret, and improves survival.
All six precommitted continuation clauses pass, licensing a disjoint-band
replication. It does not license calling the mechanism saturated because the
oracle aggregate happens to tie.

The body and word rule are supplied, the mouth still chooses among three need
words, and the frozen motor policy still does not read the self-model. The
memory is externally assigned to the correct identity; identity recognition is
not learned. Metabolism is stationary. These are resets of a simulated body,
including resets after death, not literal biological persistence through death.
No discovery of state variables, generative language, reflexive reporting,
consciousness or sentience is established. All five blocks share one frozen
lexical/motor parent. Do not promote this survey to a general lifelong-agent
claim.

Next: independently replicate the locked construction and rate-only lesions
on a disjoint identity/episode band before extending memory to changing bodies
or motor action. Keep the oracle comparison, donor-memory control, per-episode
and cumulative evidence budgets, and five-block intervals.

## 7. Reproduction and verification

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_probe68.py \
  --lives 8 --delays 24 --seed-base 3100000020 \
  --out runs/organism/probe74_cross_life_self/granularity_diagnostic.json
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.cross_life_self
PYTHONPATH=src .venv/bin/python -m pytest -q
```

New module only; no `OrganismConfig` or simulator schema knobs. Default behavior
is unchanged. Full verification: **525 tests and 13 subtests passed**.
