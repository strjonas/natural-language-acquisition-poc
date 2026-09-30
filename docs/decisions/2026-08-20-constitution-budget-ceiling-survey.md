# Constitution report: oracle-budget ceiling survey

Date: 2026-08-20. Decision: **the constitution mechanism is not licensed.** A
caregiver that partitions its lifetime basket with oracle individual rates loses
to the existing first-come caregiver at every finite store, on every seed.

Preregistered before implementation in
`docs/decisions/2026-08-20-constitution-budget-ceiling-preregistration.md`.
Module: `src/homesocial/organism/constitution_budget.py`. Artifact:
`runs/organism/probe70_constitution_budget/ceiling_survey.json`. Five seeds x 40
lives per arm and cell, band 2,300,000,000, 5,600 lives total. No treatment was
run and no report word was built.

## 1. What was tested

The organism already asks for a need and portion at every grant, but cannot see
the caregiver's private `caregiver_store`. The proposed use for a constitution
report was to let the caregiver reserve that finite basket across the whole life
according to this body's per-need metabolic burden.

Every arm used the same strongest existing requester: true current body, true
individual rates, known size-word convention, the same frozen motor parent and
the same paired life seeds. Only the caregiver differed:

| arm | allocation |
|---|---|
| `first_come` | existing caregiver; serve requests until the global store is empty |
| `oracle_rates` | fixed lifetime accounts proportional to true individual burden |
| `species_rates` | the same accounts from species-average rates |
| `permuted_rates` | the oracle account sizes assigned to the wrong need labels |

The permuted arm preserves the exact multiset and total of oracle burdens. It is
the control that distinguishes knowing *which need* burns fast from merely
knowing how much total demand exists.

## 2. Verdict: the ceiling points the wrong way

| store | first-come | oracle rates | species rates | permuted rates | oracle − first (95% CI) |
|---:|---:|---:|---:|---:|---:|
| unlimited | 0.715 | 0.715 | 0.715 | 0.715 | **0.000 exact** |
| 34 | 0.695 | 0.575 | 0.385 | 0.235 | **−0.120 [−0.187, −0.053]** |
| 30 | 0.655 | 0.465 | 0.300 | 0.185 | **−0.190 [−0.254, −0.126]** |
| 28 | 0.630 | 0.425 | 0.250 | 0.160 | **−0.205 [−0.269, −0.141]** |
| 26 | 0.525 | 0.410 | 0.235 | 0.145 | **−0.115 [−0.167, −0.063]** |
| 24 | 0.460 | 0.335 | 0.205 | 0.120 | **−0.125 [−0.183, −0.067]** |
| 22 | 0.375 | 0.270 | 0.155 | 0.110 | **−0.105 [−0.165, −0.045]** |

All **30 of 30** paired finite-store seed contrasts are negative. First-come is
neither at floor nor ceiling anywhere in the sweep, every interval excludes
zero in the harmful direction, and no adjacent-store clause can pass because
there is no positive cell to be adjacent to.

The preregistered continuation rule therefore returns `licensed = false`. This
is not a near miss on its +0.10 threshold: the largest magnitude is **−0.205**.

## 3. The controls say the rates are informative and still not enough

Two results have to be kept together.

1. **The mechanism is genuinely reading individual content.** `oracle_rates`
   beats `species_rates` at all six finite stores by 0.115–0.190 survival, and
   beats the label-permuted control by 0.160–0.340. Assigning energy's large
   account to the wrong axis kills primarily through energy; at store 30 its
   energy death rate is 0.660.
2. **Correct individual content still loses to live requests.** The same oracle
   allocator trails first-come by 0.105–0.205. Knowing the constitution improves
   a bad fixed-account policy; it does not make that policy good.

The unlimited-store control is exact on every recorded endpoint and every seed:
survival 0.715, mean life 380.055 ticks, spend 22.238, zero refusals. The new
world hook is inert when there is no allocation problem.

## 4. Why it loses: constitution is not realized demand

Store 30 is the clearest operating cell.

| | first-come | oracle rates | difference |
|---|---:|---:|---:|
| survival | 0.655 | 0.465 | **−0.190** |
| mean life ticks | 377.90 | 367.96 | −9.94 |
| refusals / life | 0.44 | 2.26 | **+1.82** |
| store spent / life | 21.796 | 20.223 | **−1.573** |
| useful uptake / store | 0.722 | 0.732 | +0.010 |
| water deaths | 0.015 | **0.210** | +0.195 |
| food deaths | 0.010 | 0.055 | +0.045 |
| energy deaths | 0.065 | 0.040 | −0.025 |

The allocator becomes slightly more basket-efficient while losing nineteen
survival points. It refuses useful help and leaves more of the global basket
unspent because mass reserved to one account cannot be borrowed by another.
Its mean refused capacity is 1.126 body units per life, **0.498 per refusal**,
with individual refusals reaching 0.60.

Probe68's lag-0 diagnostic was rerun on its disjoint band: median consequential
margin **0.08485**, formula MAE 0.000610. The mean capacity withheld per refusal
is 5.9 times that margin. This negative is therefore not another belief change
whose possible consequence sits below a coarse decision surface; the survival
endpoint confirms that the ecology hears the allocator clearly and hears it
negatively.

The fixed accounts use rates, expected shocks and a public 0.5 movement mixture.
The request stream contains more: realized action, shock timing, current state,
uptake, clipping and the consequences of earlier grants. First-come continually
reallocates toward that live signal. The constitution is a prior over demand,
not demand itself, and freezing it into a lifetime partition discards the better
evidence already arriving on every grant.

This also repeats probe64's warning in a sharper form: **resource efficiency and
survival can move in opposite directions.** A caregiver that wastes less per
unit basket can still kill more organisms by refusing at the wrong time.

## 5. Consequence for the route

Do not build a word meaning “I burn water fast” whose only consequence is this
fixed lifetime-share allocator. The oracle ceiling fails before a learner or
lexicon exists, exactly as the route required it to.

This closes:

- fixed per-need lifetime quotas derived from reported metabolic rates;
- species, oracle or label-permuted variants of the same rule;
- moving the store within 22–34 to rescue that rule;
- treating basket efficiency as a substitute endpoint for survival.

It does **not** prove that no constitution report can ever pay. A caregiver that
dynamically rebalances reserves from the live request stream would be a new
mechanism, but it must first name what it can do that first-come does not already
do. Without that structural distinction it is another coarser copy of the
working per-request channel, the objection that killed the route's first naive
form.

The next justified construction is therefore the more-axes ecology already
listed behind this one in `docs/STATE.md`: it moves probe68's untouched axis-gap
term, supplies probe61's true `K = 5` world, gives probe66 more than an
edge-of-identifiability three-need problem, and grows the message space needed
to measure the composition law.

## 6. Claim boundary

- This is a ceiling survey of one allocation mechanism, not a treatment result.
- No organism says, learns or composes a constitution report.
- No self-model property advances; the failed ceiling prevents the proposed
  reflexive/productive mechanism from being built.
- The result is about fixed lifetime accounts, one frozen parent and a
  three-reportable-need ecology. It is not an optimality proof over all possible
  caregiver policies.
- Both compared policies are analytic and constant-time; there is no training
  budget or unsaturated losing arm. Simulation ticks and motor compute are
  matched, satisfying probe69's budget requirement.
