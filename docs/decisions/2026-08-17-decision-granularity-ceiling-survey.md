# Decision granularity: a ceiling survey

Date: 2026-08-17. Status: **feasibility survey.** No gate is locked here and none
is claimed; the gates are in
`docs/decisions/2026-08-17-decision-granularity-preregistration.md`. This document
records what the survey established about the **ecology**, which stands whatever
the treatment returns.

Artifacts: `runs/organism/probe68_granularity/diagnostic.json`,
`ceiling_survey.json`, `quantum_survey.json`, `quantum_margin.json`.
Diagnostic: `scripts/diagnose_probe68.py`. Bands: diagnostic 1,260,000,000,
survey 1,320,000,000 — both disjoint from the treatment's 1,400,000,000.

## 1. Probe67's floor is not a constant. It is a saturating formula.

`docs/STATE.md` carries probe67's number as a precondition on a class of future
work: a consequential decision here is worth 0.123–0.166 of body, so a smaller
belief-side effect cannot change a word that matters. **A number cannot be
applied to an ecology that does not exist yet**, and that is the case it most
needs to cover.

Read straight off `need_scores`: with `base` the projected body, `m` its argmin
and `g = base_(2) − base_(1)` the gap between the two emptiest axes, helping
anything but `m` leaves the minimum at `base_m`, and helping `m` raises it to
`min(base_m + grant·uptake, base_(2))`. So

    margin  =  E_grant [ min( grant · uptake, g ) ]

Measured against the realised margin on real lives:

| lag | formula MAE |
|---|---|
| 0 | 0.00021 – 0.00102 |
| 18 | 0.00028 – 0.00050 |
| 24 | **0.000000** |
| 30 | **0.000000** |

Both branches are checked in closed form in `tests/test_decision_granularity.py`.

**Probe67's 0.12 was the saturated branch.** Expected grant 0.40 against a median
axis gap of 0.078–0.182 — about three times past the knee. Which is why the floor
looked like a constant of the world:

| lag | scale 0.125 | 0.25 | 0.5 | **1.0** | **2.0** |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 0.0701 | 0.0898 | **0.09005** | **0.09005** |
| 18 | 0.0500 | 0.0971 | 0.1230 | **0.14837** | **0.14837** |
| 24 | 0.0500 | 0.0790 | 0.1040 | **0.10800** | **0.10800** |

Doubling the grant changes the granularity by **exactly zero** at three of four
lags. The ecology cannot be made coarser; it is already as coarse as it gets.

## 2. And it cannot be made finer either, because viability pins the grant

The obvious lever is to lower the grant. The survey says no, on 30 lives:

| scale | oracle survival | myopic survival | verdict |
|---|---:|---:|---|
| 0.25 | **0.000** | 0.000 | collapsed |
| 0.5 | **0.000** | 0.000 | collapsed |
| 1.0 | 0.267 | 0.000 | usable |
| 2.0 | 0.333 | 0.033 | usable |

**Even a model that knows everything starves when the grant is halved.** This is
the finding of the survey and it was not anticipated:

> Probe65's operating point has **no headroom beneath it**. The grant is pinned
> from below by viability, and *that* is why the decision surface is coarse. A
> world whose help must be large to keep anything alive has a coarse decision
> surface **by necessity** — and no refinement of belief can reach past it.

Read against probes 59, 60, 62, 64 and 67 — five separate mechanisms that
improved a belief and bought no behaviour — this is a structural account of the
whole run of negatives rather than five unrelated ones. It also says something
about a *larger* ecology that this repository has not been able to say before:
**the grain of the decision surface is not a free parameter of a design. It is
downstream of how much help a body needs to stay alive.**

## 3. The lever that does exist: same help, smaller quanta

Nutrition and granularity can be separated by moving the **quantum** while
holding the **rate** of help fixed — smaller portions, proportionally more often,
with `grant / help_period` identical in every row.

| quantum | help_period | grant/tick | median margin | `state_oracle` reach | oracle surv |
|---|---:|---:|---:|---:|---:|
| 1/3 | 2 | 0.0667 | 0.0669 | 0.7972 | 0.400 |
| 1/2 | 3 | 0.0667 | 0.0878 | 0.7600 | 0.400 |
| 1 | 6 | 0.0667 | 0.1749 | 0.4407 | 0.320 |
| 2 | 12 | 0.0667 | 0.2067 | 0.3279 | **0.160 — collapsed** |

Halving the quantum halves the margin at identical nutrition. The two levers are
**complementary**: the grant sweep is unusable below scale 1.0 and the quantum
sweep above quantum 1, so each covers exactly the half the other cannot, and
together they move nutrition and granularity one at a time.

`answer_horizon` reads the lag and never the period, so every row still answers
at eighteen ticks.

## 4. The homeostatic bound is part of the decision surface, for the third time

`need_scores` is shift-equivariant on paper — add a constant to every axis and
every score moves together — which would mean only the **differential** part of a
belief error can ever reach a word. Tested by matched injection rather than
asserted, and **false at the operating point**:

| lag | injection | common-mode flips | differential flips | common-mode max margin shift |
|---|---:|---:|---:|---:|
| 0 | 0.02 | **0.0000** | 0.0251 | **1.11e-16** |
| 18 | 0.02 | **0.0299** | 0.0082 | 2.00e-02 |

The cause is measured: the cap binds on **82.6%** of lag-18 ticks against **0.0%**
at lag 0, because the request ledger's `arriving` term pushes the projected body
past 1.0. Where the cap does not bind, shift-equivariance is **exact**. Where it
binds, the cap converts common-mode error into differential error — and
common-mode injection then flips *more* words than differential injection does.

Probe62 found the bound caps self-model error. Probe63 and 67 found RLS discards
clipped readings. This is the third time it has turned out to be an active part
of a result rather than a backdrop. **Probe65's operating point sits almost
entirely inside the clipped regime**, which no probe had recorded.

The clause is closed. Probes 59 and 60 remain **unretrodicted** by this account.

## 5. Reach retrodicts probe65's sign reversal from arithmetic alone

    reach share = P( differential belief error > margin | consequential tick )

`state_oracle` knows where it is and not what it is, so its reach *is* the value
of knowing your own rate:

| lag | `state_oracle` reach | probe65's measured `individual − state_oracle` |
|---|---:|---|
| 0 | **0.0009** | **−0.1220** |
| 18 | **0.3274** | **+0.1017** |

and reach orders all three arms correctly at lag 18 — 0.0493 (`individual`)
< 0.0799 (`recursive`) << 0.3274 (`state_oracle`) against measured accuracies
0.9359 > 0.9329 >> 0.8312.

Reach is an **upper bound** on error, not a point predictor: reaching a decision
is not flipping it, and it over-predicts `state_oracle`'s error by 0.16. The
claim it supports is about ordering and response, never level.

## 6. What the survey suggests and cannot settle

Band 1,320,000,000, 3 seeds × 14 lives, nutrition identical in all three rows:

| quantum | margin | `oracle − state_oracle` | `state_oracle − population` |
|---|---:|---:|---:|
| 1/3 | 0.0669 | **+0.2897** | +0.2674 |
| 1/2 | 0.0878 | **+0.2124** | +0.2891 |
| 1 | 0.1749 | **+0.1299** | +0.2458 |

The value of knowing **what you are** more than doubles as the grain halves. The
value of knowing **where you are** does not move. That is the dissociation the
treatment gates, on a disjoint band — a threshold read off a survey's own lives
is not a threshold.

**A warning carried forward.** A one-seed reading of the original contrast
(`individual − state_oracle`) came back −0.0127 where +0.0637 was expected, and at
three seeds it read +0.0600 against +0.0396 — the other way. Single-seed
measurements on this endpoint are worth nothing, and the first design was nearly
abandoned on one.

## 7. Claim boundary

This survey establishes a **formula** and two facts about **this ecology**. It
establishes no causal claim: that the granularity *controls* the value of a
self-model is what the treatment tests, and until it returns, section 6 is a
suggestion measured on 3 seeds.

It says nothing about survival — every number above except the viability column is
open-loop. It does not move the axis-gap term of the formula, which needs a body
with more axes than this simulator has. And it says nothing about discovery,
uncertainty reporting, language productivity, consciousness, or sentience.
