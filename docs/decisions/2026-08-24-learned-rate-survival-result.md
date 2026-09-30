# Probe73: does the learned individual rate improve survival? — result

Date: 2026-08-24. Preregistration:
`docs/decisions/2026-08-24-learned-rate-survival-preregistration.md`.
Module: `src/homesocial/organism/learned_rate_survival.py`. Guards:
`tests/test_learned_rate_survival.py`. Artifacts:
`runs/organism/probe73_learned_rate_survival/{granularity_diagnostic,treatment}.json`.

Five paired seed blocks x 140 lives per cell = 700 lives per arm, treatment
band 3,000,000,000, lag 24. Every cell runs the same recursive calibrator and
frozen parent. Only the rate vector exposed to the fixed request rule changes:
species, learned, or true.

**Verdict: all five locked gates pass.** The recursive self-model's learned rate
improves survival by **+0.0143 [+0.0003, +0.0283]** over a species-rate lesion
with the same filtered state, uptake belief, readings, compute and planner. Four
of five paired blocks are positive and the fifth is an exact tie. The lower
bound clears zero narrowly, so this is a small positive result, not a large one.

| gate | requirement | result | |
|---|---|---|---|
| **G1** learned rate reaches survival | positive, interval clear, >= 4/5 | **+0.0143 [+0.0003, +0.0283], 4/5 positive, one tie** | pass |
| **G2** matched physical ceiling | positive, interval clear, >= 4/5 | **+0.0386 [+0.0213, +0.0559], 5/5** | pass |
| **G3** matched regret predicts sign | positive, interval clear, >= 4/5 | **+0.00616 [+0.00421, +0.00811], 5/5** | pass |
| **G4** rate was learned | species MAE - learned MAE positive | **+0.00290 [+0.00262, +0.00317], 5/5** | pass |
| **G5** viable construction | true-rate survival >= 0.15 | **0.2129** | pass |

## 1. Survival: the learned rate is physically load-bearing

| request-planning rates | survival | per seed | mean life steps |
|---|---:|---|---:|
| species | 0.1743 | .1429, .1429, .2143, .1429, .2286 | 126.1 |
| **learned** | **0.1886** | .1571, .1429, .2214, .1714, .2500 | **131.3** |
| true | 0.2129 | .1857, .1714, .2357, .2000, .2714 | 145.5 |

The primary paired values are +0.0143, 0.0000, +0.0071, +0.0286 and +0.0214.
They add ten survivors over 700 lives: 132 for learned rates against 122 for
species rates. Mean life length moves by +5.2 ticks in the same direction.

The matched true-rate ceiling adds 27 survivors over the species lesion and is
positive on all five blocks: **+0.0386 [+0.0213, +0.0559]**. The learned rate
captures about **37%** of that survival value. The remaining true-minus-learned
gap is +0.0243 [+0.0164, +0.0322], reported but not gated. It is the physical
rent available to faster or cross-life self-knowledge, not evidence that such a
mechanism already exists.

The death distribution is consistent with the direction rather than dominated
by one flipped life. The learned-rate arm has 241 energy and 185 water deaths,
against 252 and 195 for the species lesion. Food and safety deaths rise by 6 and
5, leaving the preregistered net of ten additional survivors. The true-rate arm
reduces energy and water deaths further.

## 2. The lesion changes rates and nothing else

All cells instantiate `SelfModelTier("recursive")`. Each request reads that
tier's current filtered body, current uptake belief and the same public request
ledger. The species cell still computes and updates the recursive rate estimate
on every transition; only the request rule is denied access to it. It therefore
matches estimator compute as well as state.

"Matched state" here means matched state-estimation machinery and a rate-only
planner intervention. Once two closed-loop cells say different words their
later bodily trajectories can and should diverge; holding those downstream
states numerically identical would remove the survival pathway being tested.
The regret panel below supplies the stricter same-trajectory counterfactual,
where both rate vectors are scored on the exact same current state and ledger.

On the `learned_rate` histories, replacing only the rate vector changes the
selected need word on **27.5%** of scored ticks. This is not an inert numeric
improvement hidden below Probe68's decision grain.

The pre-treatment granularity diagnostic predicted that a rate correction could
reach the word surface at lag 24: species-rate projection error reached 75.6% of
consequential decisions, and the recursive point error still reached 25.5%.
That precondition holds without changing the portions or caregiver period.

## 3. Belief and regret endpoints agree with survival

On the same `learned_rate` histories, the learned-rate counterfactual and the
species-rate counterfactual share the exact current state, uptake, ledger and
trajectory. Scored against the true `need_scores`:

| endpoint | species word | learned word | value of learning |
|---|---:|---:|---:|
| regret per scored tick | 0.01097 | **0.00481** | **+0.00616 [+0.00421, +0.00811]** |
| rate MAE per tick | 0.00355 | **0.00065** | **+0.00290 [+0.00262, +0.00317]** |

Every block agrees on both signs. The recursive estimator removes **81.7%** of
the species rate error in per-tick units, and exposing that estimate to the
request rule removes **56.1%** of the species-word regret. Probe65's statement
that survival tracks regret therefore survives the one-factor test that result
was missing.

The three currencies line up in their preregistered order:

    learned rate -> lower rate error -> lower word regret -> higher survival

This is stronger than comparing `recursive` with `state_oracle`, because that
older contrast changed state error and rate error together.

## 4. What advances, and what does not

This advances **load-bearing across uses**, but it does not finish that
property. The same online self-calibrator now supplies a bodily constant to a
future prediction, changes the need word selected from that prediction, and
changes whether the organism survives. Damaging one readout of the model —
replacing its learned rates with species rates — damages regret and survival
together.

The property remains partial because the frozen motor policy does not read this
self-model, bodily variables and planner structure remain designer-supplied,
and the mouth is still `argmax` over three need words. Discovery, productive
language and reflexive reporting do not advance.

This also resolves Probe65's +0.030 survival hint more cleanly than merely adding
lives to the old mixed contrast. The point estimate is smaller, +0.0143, because
the species lesion retains the recursive arm's improved state instead of
replacing it with a different state estimator.

## 5. Claim boundary

- One fixed lag-24 metabolic ecology, one three-need body, one recursive
  calibrator and one frozen parent. The primary interval's lower bound is
  +0.0003; replication in another band or ecology would be valuable before
  treating the magnitude as stable.
- This shows value from using a learned individual rate. It does not show that
  the future-request rule, request ledger, body axes or lexical inventory were
  learned.
- It does not show cross-life identity or memory. The calibrator resets each
  life and leaves 63% of the matched true-rate ceiling unused.
- It does not reverse Probe72. The finer help clock still fails as a survival
  amplifier; Probe73 uses the unchanged baseline clock at a longer,
  independently selected horizon.
- No estimator family is compared and nothing trains in this probe. The compute
  budget is matched cell by cell, so Probe69's sample-efficiency result is not
  promoted into a ceiling claim.
- Nothing here is evidence about consciousness or sentience.

## 6. Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_probe68.py \
  --lives 8 --delays 24 --seed-base 2900000000 \
  --out runs/organism/probe73_learned_rate_survival/granularity_diagnostic.json
PYTHONPATH=src .venv/bin/python -m homesocial.organism.learned_rate_survival
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The treatment is resumable from its JSON artifact and refuses a configuration
change on resume. Final verification: **519 tests and 13 subtests passed**.
