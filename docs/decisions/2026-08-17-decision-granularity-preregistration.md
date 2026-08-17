# Probe68: the granularity of an ecology's decisions — preregistration

Date: 2026-08-17. Status: **preregistration.** The gates below are the ones in
`grade()` in `src/homesocial/organism/decision_granularity.py`, written and
covered by `tests/test_decision_granularity.py` **before the treatment ran**.
Diagnostic: `scripts/diagnose_probe68.py`. Artifacts:
`runs/organism/probe68_granularity/`.

## 0. Provenance, because this design was overturned twice before it was locked

The first two designs in this document were killed by their own surveys, on a
seed band disjoint from the treatment, before any gate existed. Both are kept
below rather than deleted, because a preregistration that hides its false starts
is not one. Section 3 and section 5 are those two corrections.

**Why this is a swept variable and not a forbidden retuning.** `docs/STATE.md`
lists "retuning `help_period` or any frozen ecology constant" under do-not-reopen,
by name, and this probe moves both `help_period` and the grant. The prohibition
is against changing the ecology **until a gate passes**. Here:

- the ecology parameter is the **independent variable**, swept over a locked grid;
- the compensated sweep **pins** `grant / help_period` so the nutrition rate
  cannot move — the exact quantity a retuning would change;
- one of the locked predictions is a **null**, and a retuning cannot predict that
  doubling a resource changes nothing;
- the horizon is untouched: `answer_horizon` reads the lag and never the period,
  so every row answers at eighteen ticks exactly as probe65's did.

This is probe65's own design. Probe65 swept `help_delay` — an ecology parameter —
with the crossover fixed at 9.3 ticks from published numbers before running.

## 1. What probe67 measured, and the one thing it could not

Probe67's durable output is a number: a consequential decision here is worth
**0.123 to 0.166 of body**, so nothing smaller can change a word that matters.
A number cannot be applied to a world that does not exist yet, which is the case
it most needs to cover — deciding what a **larger** ecology must look like for
fine self-knowledge to pay rent in it.

## 2. The formula, with no free parameters

Read off `need_scores`. With `base` the projected body, `m` its argmin and
`g = base_(2) − base_(1)` the gap between the two emptiest axes: helping anything
but `m` leaves the minimum at `base_m`, and helping `m` raises it to
`min(base_m + grant·uptake, base_(2))`. So

    margin  =  E_grant [ min( grant · uptake, g ) ]

**Measured against the realised margin**, band 1,260,000,000:

| lag | formula MAE |
|---|---|
| 0 | 0.00021 – 0.00102 |
| 18 | 0.00028 – 0.00050 |
| 24 | **0.000000** |
| 30 | **0.000000** |

It **saturates**. Probe67's 0.12 was the saturated branch: expected grant 0.40
against a median axis gap of 0.078–0.182, about three times past the knee.
`tests/test_decision_granularity.py` checks both branches in closed form.

## 3. First correction: a clause that was derived, tested, and refuted

`need_scores` is shift-equivariant on paper, so only the **differential** part of
a belief error should ever reach a word. That would have retrodicted probes 59
and 60. It was tested by matched injection rather than asserted, and **it is
false at the operating point**:

| lag | injection | common-mode flips | differential flips | common-mode max margin shift |
|---|---:|---:|---:|---:|
| 0 | 0.02 | **0.0000** | 0.0251 | **1.11e-16** |
| 18 | 0.02 | **0.0299** | 0.0082 | 2.00e-02 |

The cause is measured: the homeostatic cap binds on **82.6%** of lag-18 ticks
against **0.0%** at lag 0, because the ledger's `arriving` term pushes the
projected body past 1.0. Where the cap does not bind, shift-equivariance is
**exact**. Where it binds, the cap converts common-mode error into differential
error. This is the third time the homeostatic bound has turned out to be an
active part of a result rather than a backdrop — probe62, probe63/67, and now.

**The clause is closed and carried into no gate.** Probes 59 and 60 therefore
remain **unretrodicted**, and section 9 says so.

## 4. What the law claims

    reach share  =  P( differential belief error > margin | consequential tick )

**Reach, not belief accuracy, orders the behavioural value of a self-model.** It
retrodicts probe65's sign reversal from arithmetic alone — `state_oracle` is the
arm that knows where it is and not what it is, so its reach *is* the value of
knowing your own rate:

| lag | `state_oracle` reach | probe65 measured `individual − state_oracle` |
|---|---:|---|
| 0 | **0.0009** | **−0.1220** |
| 18 | **0.3274** | **+0.1017** |

Reach is an **upper bound** on error, not a point predictor — it over-predicts
`state_oracle`'s error by 0.16. Every gate below is on ordering and response,
never on level.

## 5. Second correction: the ceiling survey killed the first lever

The first design swept the grant alone: scales 0.25, 0.5, 1.0, 2.0. The survey
on band 1,320,000,000, 30 lives:

| scale | oracle survival | myopic survival | verdict |
|---|---:|---:|---|
| 0.25 | **0.000** | 0.000 | collapsed |
| 0.5 | **0.000** | 0.000 | collapsed |
| 1.0 | 0.267 | 0.000 | usable |
| 2.0 | 0.333 | 0.033 | usable |

**Even a model that knows everything starves when the grant is halved.** The
positive half of the prediction was untestable with that lever, and this is a
fact about the ecology worth stating on its own:

> Probe65's operating point has **no headroom beneath it**. The grant is pinned
> from below by viability — which is *why* the decision surface is coarse. A
> world whose help must be large to keep anything alive has a coarse decision
> surface by necessity, and no refinement of belief can reach past it.

So the quantum is moved with the **rate of help held exactly fixed**: smaller
portions, proportionally more often. `grant / help_period` is identical in every
compensated row.

| quantum | help_period | portions | grant/tick | oracle surv | verdict |
|---|---:|---|---:|---:|---|
| 1/3 | 2 | 0.0667, 0.20 | 0.0667 | 0.400 | usable |
| 1/2 | 3 | 0.10, 0.30 | 0.0667 | 0.400 | usable |
| 1 | 6 | 0.20, 0.60 | 0.0667 | 0.320 | usable |
| 2 | 12 | 0.40, 1.20 | 0.0667 | **0.160** | **collapsed** |

The two levers are **complementary**: the uncompensated sweep is unusable below
scale 1.0 and the compensated one above quantum 1, so each covers exactly the
half the other cannot. Together they move nutrition and granularity **one at a
time**, which neither could do alone.

Measured granularity under compensation, band 1,260,000,000, 28 lives:

| quantum | median margin | `state_oracle` reach | `point` reach |
|---|---:|---:|---:|
| 1/3 | 0.0669 | 0.7972 | 0.2820 |
| 1/2 | 0.0878 | 0.7600 | 0.2774 |
| 1 | 0.1749 | 0.4407 | 0.1909 |

## 6. Third correction: the contrast had to change, and it brought its own lesion

The original endpoint was probe65's `individual − state_oracle`. That contrast
**mixes two errors that move in opposite directions**: `individual` has true rates
and a filtered state, so a finer surface exposes its state error at the same time
as it exposes `state_oracle`'s rate error. A one-seed survey read −0.0127 where
+0.0637 was predicted; at three seeds it read +0.0600 against +0.0396, the other
way. The contrast is too noisy and too confounded to gate.

Two contrasts replace it, each moving one factor:

    RATE_VALUE   = oracle − state_oracle        both hold the true body; they
                                                differ only in whose rates.
    STATE_VALUE  = state_oracle − population    both hold species rates; they
                                                differ only in read vs filtered body.

Survey, band 1,320,000,000, 3 seeds × 14 lives:

| quantum | margin | **RATE_VALUE** | **STATE_VALUE** |
|---|---:|---:|---:|
| 1/3 | 0.0669 | **+0.2897** | +0.2674 |
| 1/2 | 0.0878 | **+0.2124** | +0.2891 |
| 1 | 0.1749 | **+0.1299** | +0.2458 |

**STATE_VALUE is the lesion this claim needs.** If a finer decision surface simply
makes every decision harder, both rise together and there is nothing here about
self-knowledge in particular. The survey says rate value more than doubles while
state value does not move — and the treatment is what decides it, on a disjoint
band, because a threshold read off a survey's own lives is not a threshold.

## 7. The locked cells

`nutrition_1x` is probe65's ecology with every constant untouched, and is the
reference every contrast is taken against.

| cell | lag | grant | grant/tick | what moves |
|---|---:|---:|---:|---|
| `nutrition_1x` | 18 | 0.4000 | 0.0667 | — reference |
| `nutrition_2x` | 18 | 0.8000 | **0.1333** | resources double, grain does not |
| `quantum_half` | 18 | 0.2000 | 0.0667 | grain halves, resources do not |
| `quantum_third` | 18 | 0.1333 | 0.0667 | grain halves again |
| `lag0_1x` | 0 | 0.4000 | 0.0667 | the numerator null |
| `lag0_2x` | 0 | 0.8000 | 0.1333 | the numerator null |

**8 seeds × 30 lives**, treatment band 1,400,000,000, stride 2,000,000, disjoint
from the diagnostic (1,260,000,000) and survey (1,320,000,000) bands and asserted
so by `_check_seed_isolation`. Eight seeds rather than five because probe65's
five gave a paired half-width of ~0.026, wider than the equivalence band; the
interval is dominated by between-seed variance, so seeds buy more than lives.

## 8. Gates

| gate | requirement |
|---|---|
| **G6** | **The dissociation.** `RATE_VALUE(quantum_half) − RATE_VALUE(nutrition_1x) > 0`, CI excluding zero, ≥ 6/8 seeds same sign, **and** `\|STATE_VALUE step\| < RATE_VALUE step`. |
| **G2** | **The nutrition null.** `contrast(nutrition_2x) − contrast(nutrition_1x)` inside ±0.02 **with CI half-width below 0.02**. |
| **G3** | `contrast(quantum_half) − contrast(nutrition_1x) > 0`, CI excluding zero, ≥ 6/8 seeds. |
| **G4** | **The diminishing step.** The 1/2 → 1/3 step is strictly smaller in magnitude than the 1 → 1/2 step, matching reach 0.441 → 0.760 → 0.797. A response *linear in the quantum* passes G3 and fails here. |
| **G5** | **The lag-0 null.** `contrast(lag0_2x) − contrast(lag0_1x)` inside ±0.02 with half-width below 0.02, **and** the contrast negative at both scales. |

**G2 and G5 are equivalence gates and are graded three-valued: pass, fail, or
UNRESOLVED.** An interval wider than the band is UNRESOLVED, never a pass.
Probe65 made exactly that error in the opposite direction — a reported null on
regret that its own follow-up showed to be a sample-size statement — and
`_equivalence()` is where that correction is enforced rather than remembered.
`test_a_wide_interval_makes_an_equivalence_gate_unresolved_not_passed` guards it.

**G6 is the probe.** G2–G5 are its supports.

## 9. Failure condition

**The law is closed if G6 fails** — if halving the grain at identical nutrition
does not raise the value of knowing your own rate, or raises the value of knowing
where you are just as much. No reformulation of the margin rescues that.

**G2 failing while G6 passes** means behaviour tracks resources as well as grain,
and the result must be reported as "granularity matters, and it is not the only
thing that does", not as a clean dissociation.

Gates are not adjusted after seeing results. G6's failure closes the forward
claim while leaving the **formula** (section 2) standing as a description; the
result doc must then say the granularity is computable and **not** shown to be
causal, and must not report that as a partial success.

## 10. What this cannot show, stated in advance

- It does **not** retrodict probes 59 and 60. Section 3's clause was the proposed
  account and it is refuted; those two remain unexplained.
- It does **not** move the second term of the formula. The axis gap is set by how
  many bodily axes compete; moving it needs a body this simulator does not have.
  That is the lever connecting this to "more needs", and it is out of scope.
- The endpoint is **open-loop accuracy on a shared history**, not survival. The
  closed loop appears only in the viability check. A granularity effect on what
  is *said* is not yet an effect on how long anything *lives*.
- `nutrition_2x` does not hold the realised decision surface fixed as cleanly as
  the arithmetic does: a larger grant changes the body distribution and so the
  axis gap, and its consequential share falls from 0.797 to 0.247. G2 is a weaker
  null than G6 is a positive, and is reported as such.
- One ecology, one frozen parent, one motor policy, three needs. A granularity law
  measured in a world with three axes is not shown to hold in a world with thirty.
- Nothing about discovery, uncertainty reporting, structure, language
  productivity, consciousness, or sentience.
