# STATE

Last rewritten: 2026-08-17. Rewrite this file, never append.

## Where the next agent should start

One result landed on 2026-08-17, and it changes how a probe should be *chosen*
rather than adding another mechanism to the pile.

1. `docs/decisions/2026-08-17-decision-granularity-result.md` (probe68, **4 of 5
   locked gates pass; G5 fails and is not rewritten**).
2. `docs/decisions/2026-08-17-decision-granularity-ceiling-survey.md` -- the
   survey, which contains two findings about the ecology that stand independently
   of the treatment and are arguably worth more than it.

**Read the survey first.** It replaces the number `STATE.md` has been carrying
since probe67 with a formula, and then finds that the formula's own lever is
pinned.

### The bias probe68 was built to correct

Since probe59 **every probe has held the ecology fixed and varied the mechanism.**
Five found the same negative -- 59, 60, 62, 64, 67: the belief improves and the
behaviour does not hear it. The two that succeeded, **63 and 65, succeeded by
changing the ecology.** The repository had been optimizing the numerator of a
ratio whose denominator it controls and had measured once, as a constant.

## Probe68: the grain of the decision surface, and what it decides

### The floor is a formula, not a constant

Read straight off `need_scores`, with `g` the gap between the two emptiest
projected axes:

    margin  =  E_grant [ min( grant * uptake, g ) ]

Reproduced against the realised margin with MAE **0.000000** at lags 24 and 30 and
<= 0.001 elsewhere. Probe67's 0.123--0.166 is its **saturated branch**: expected
grant 0.40 against a median axis gap of 0.078--0.182, about three times past the
knee. Doubling the grant changes the granularity by **exactly zero** at three of
four lags.

**The binding check is no longer "is my effect bigger than 0.12".** It is *what is
the margin of the ecology I am proposing, and is my effect bigger than that.*
`scripts/diagnose_probe68.py` computes it in minutes for any config.

### And the grain is pinned from below by viability

| grant scale | oracle survival | verdict |
|---|---:|---|
| 0.25 | **0.000** | collapsed |
| 0.5 | **0.000** | collapsed |
| 1.0 | 0.267 | usable |
| 2.0 | 0.333 | usable |

**Even a model that knows everything starves when the grant is halved.**

> Probe65's operating point has no headroom beneath it. The grain of a decision
> surface is **downstream of how much help a body needs to stay alive** -- so a
> world whose help must be large to keep anything alive is coarse *by necessity*,
> and no refinement of belief can reach past it.

That is one structural account of five separate negatives. It is the most portable
thing in this result, because it is a statement about **ecology design** rather
than about this organism.

### The lever that does exist, and the treatment

Move the **quantum** and hold the **rate** of help fixed: smaller portions,
proportionally more often, `grant / help_period` identical to the last decimal.
`answer_horizon` reads the lag and never the period, so the horizon stays 18.

8 seeds x 30 lives, band 1,400,000,000:

| quantum | help_period | grant/tick | margin | **rate value** | **state value** |
|---|---:|---:|---:|---|---|
| 1 | 6 | 0.0667 | 0.1749 | **+0.1272** [+0.1038, +0.1506] | +0.2271 [+0.1934, +0.2607] |
| 1/2 | 3 | 0.0667 | 0.0878 | **+0.2127** [+0.1846, +0.2409] | +0.2371 [+0.2152, +0.2590] |
| 1/3 | 2 | 0.0667 | 0.0669 | **+0.2775** [+0.2404, +0.3145] | +0.2293 [+0.2091, +0.2495] |

    rate value  = oracle - state_oracle       differ only in whose rates
    state value = state_oracle - population   differ only in read vs filtered body

**G6 passes.** Paired per seed, `quantum_half` against `nutrition_1x`: rate step
**+0.0855 [+0.0753, +0.0956]** on **8/8** seeds against a state step of **+0.0101
[-0.0088, +0.0289]** on 5/8, interval containing zero. **8.5x, with the lesion
holding.**

> Halving the grain of the help, with the amount of help held exactly fixed, more
> than doubles the value of knowing what kind of body you are, and leaves the value
> of knowing where that body is where it was.

State error enters `level - rate * H` once; rate error enters multiplied by `H`. At
`H = 18` the rate error is a large quantity and the coarse margin was **swallowing**
it. Probe65 showed the horizon puts it into the decision variable; probe68 shows
the grain decides whether the decision variable can express it.

**G4 passes and is the sharper half.** Reach was measured *before* the treatment at
0.4407 / 0.7600 / 0.7972 -- steep then flat -- and the response follows it: first
step +0.0335, second **+0.0166**. A value linear in the quantum passes G3 and fails
G4.

### Four hundredfold, from two ecology parameters

| ecology | rate value |
|---|---|
| lag 0, grant 0.40 | **+0.0007** [+0.0005, +0.0010] |
| lag 0, grant 0.80 | +0.0009 [+0.0003, +0.0015] |
| lag 18, quantum 1/3 | **+0.2775** [+0.2404, +0.3145] |

At lag 0 `state_oracle` scores **0.9995**: a body read perfectly and believed
typical is essentially never wrong, so a self-model has nothing to add. Same
self-model object, same arithmetic, same frozen parent.

### What must be read honestly

- **G2 passes and is not a null.** +0.0186 [+0.0109, +0.0264] -- inside the locked
  +/-0.02 band, but the interval excludes zero on 8/8 seeds. "+0.019, smaller than
  the band" and not "nothing happened". And the cell is confounded as
  preregistered: doubling the grant collapses state value +0.2271 -> +0.0857 and the
  consequential share 0.753 -> 0.229, because it moves the axis gap even while the
  grant term stays saturated. **G2 is a weak null; G6 is a strong positive. Do not
  read them as equal evidence.**
- **G5 fails, +0.1072 against a +/-0.02 band, and is not rewritten.** It was the one
  gate left on the *mixed* contrast `individual - state_oracle`, which at lag 0
  contains no rate signal and is therefore pure state value -- a quantity the grant
  moves a great deal. The prediction G5 was written for passed to four decimals
  (+0.0007). A locked gate failed and the thing it aimed at is true; that measures
  the gate's construction, not the world.
- **Two designs were killed by their own surveys before any gate existed**, both
  recorded in the preregistration rather than deleted: an uncompensated grant sweep
  (collapsed), and the mixed contrast (confounded, and a single-seed reading of it
  pointed the wrong way).
- **A derived clause was refuted by measurement.** `need_scores` is
  shift-equivariant on paper, so only *differential* belief error should reach a
  word -- which would have retrodicted probes 59 and 60. Matched injection says no:
  the homeostatic cap binds on **82.6%** of lag-18 ticks against 0.0% at lag 0 and
  converts common-mode error into differential. Where the cap does not bind,
  equivariance is exact (1.11e-16). **Probes 59 and 60 remain unretrodicted**, and
  probe65's operating point sits almost entirely inside the clipped regime -- which
  no probe had recorded. Third time the homeostatic bound has turned out to be an
  active part of a result rather than a backdrop.

## Where the five properties now stand

From `md/archive/DIRECTION_2026-07-26.md` section 2.

1. **Discovered** -- *partial*, **unchanged since 2026-08-03**. Still the
   longest-stalled leg, and DIRECTION calls it "the single largest step toward 'not
   parroted'". See "Exact next work" item 1: probe68 makes the construction it
   needs also serve two other stalled lines.
2. **Corrigible** -- **yes**, unchanged.
3. **Load-bearing across uses** -- *partial*, and probe68 says *how much* it is
   load-bearing is an ecology parameter rather than a property of the model.
4. **Productive under novel demand** -- *partial*, unchanged since probe66. What
   keeps it short of yes is the hypothesis space.
5. **Reflexive** -- *partial*, unchanged since probe67. Probe68 does **not** advance
   it, and slightly reframes probe67's blocker: acting on the *width* failed at a
   margin of 0.12--0.17, and probe68 can now make the margin 0.067. That is a live
   question and item 5 below states its cost.

## Evidence ladder and closed lines

| Probe | Mechanism | Result |
|---|---|---|
| 48--56 | neural mouths, COMA critics, recurrent self-belief, ranked belief | all **fail**; recurrent line closed |
| 57 | structured learned causal self + social planner | **all local gates pass** |
| 58 | causal-stage replication, 5 seeds | **5/5 pass**, sd <= 0.009 |
| 59 | online adaptation after a body-rule change | recalibration 5/5; survival gates **fail** |
| 60 | outcome-aware realized-consequence planner | planner repair passes; load-bearing gate **fails** |
| 61 | discovered bodily structure from two sensations | **4 of 5 fail**; closed |
| 62 | self-uncertainty ceiling survey | posterior **ties** the point filter; closed in its ecology |
| 63 | individual body + online self-calibration | **7/7 pass**; standing obstacle closed |
| 64 | portion requests under a rationed caregiver | **7/7 pass**; **survival hears none of it** |
| 65 | a caregiver that answers eighteen ticks late | **9/9 pass**; ordering *reverses* with the horizon; **a belief reaches survival** |
| 66 | a caregiver whose size words do not mean one thing | **6/6 pass**; factorization **found rather than given** |
| 67 | acting on the width of the rate belief | **survey, no**; ceiling arm loses to baseline; a 0.12 floor |
| 68 | the grain of the decision surface | **4/5 pass, G5 fails**; the 0.12 is a **formula**, the grain is **pinned by viability**, and finer grain doubles the value of a rate self-model at fixed nutrition |

Do not reopen without contrary evidence:

- report entropy, head width, vocabulary size, replay, sparse-return loss weights,
  horizons as a tuning knob, counterfactual token-credit variants;
- more guided lexical exposure after comprehension reaches ceiling;
- black-box recurrent self-belief width/loss/head variants;
- a larger legacy neural mouth on the same state;
- compute scale as a substitute for causal structure;
- probe61's structure discovery **mechanism** (a sparsity-penalized latent);
- stacking probe64's `caregiver_store` onto probe65's lag;
- acting on the rate posterior through probe60's objective, at any horizon;
- **lowering the grant without compensating the period** -- probe68 measured oracle
  survival 0.000 at half the grant; the ecology has no headroom beneath it;
- **the mixed `individual - state_oracle` contrast as a gate.** Use `rate value` and
  `state value`. The mixed one confounds a rate advantage with a state
  disadvantage and cost probe68 one gate and two designs.

`help_period` is no longer categorically frozen: probe68 moves it **pinned** to the
portion so that `grant / help_period` cannot change. Moving it *freely* remains
closed.

## Exact next work

Every item carries the granularity check, which is now `diagnose_probe68.py`
rather than a comparison against 0.12.

1. **Build the body with more axes, and get three stalled lines for one
   construction.** This is the axis-gap term of probe68's formula -- the one lever
   never moved, and the one the formula says is *binding* here (the grant term is
   saturated). It is also, independently:
   - the world **probe61's own claim boundary asks for**: "a world with true `K = 5`
     would discriminate the two readings and this ecology does not contain one".
     Aiming probe66's mechanism (exact Bayesian model comparison, no free
     parameters, moved ground truth) at bodily structure is a **new mechanism for a
     failed phase, not a rescue of a failed one** -- state that in the
     preregistration, because it is the first thing a reviewer will challenge;
   - what **probe66 itself asks for**: "with three needs the design is at the edge of
     identifiability, so this wants more needs before more model";
   - the only route to property 1, the longest-stalled leg.

   **Cost, honestly:** `REPORT_NEEDS` and `BODY_NEEDS` are in `island/report.py` and
   the body is in `env.py`, and the frozen probe52 parent checkpoint was trained on
   the current axes. Adding axes means retraining that parent, which no probe since
   57 has had to do. This is the most expensive item on this list and the only one
   that unblocks three things at once.

2. **Re-run probe67's width mechanism at quantum 1/3.** Probe67 closed acting on the
   rate posterior and scoped the closure to a margin of 0.12--0.17. Its largest
   correction was 0.017 against a median decision of 0.123 -- "seven to
   twenty-eight times too small". Probe68 can make the median decision **0.0669**,
   which moves the ratio from ~0.14 to ~0.25. **That is still under 1**, so the
   honest expectation is that it stays closed; the value is that the closure would
   then be stated against a *swept* margin rather than a single one. Cheap
   (`uncertain_horizon.py` needs only the two new kwargs). Run the granularity check
   first and do not run it if the ratio has not crossed.

3. **Give probe68 a survival endpoint.** Every probe68 number is open-loop. Probe65
   is still the only place a belief has reached survival, and probe68's claim is
   about what is *said*. The closed-loop runner exists (`run_closed_loop`, already
   takes both new kwargs); what it needs is N -- probe65's own diagnosis puts
   ~700 lives per arm on resolving a 0.03 survival difference.

4. **Push past the regret crossover and gate survival on the rate** (was item 2).
   Lag 24 or beyond, fresh seed band, ~700 lives per arm. Now cheaper to justify:
   probe68 raises the rate's decision-side effect by 8.5x its own state-side
   control, so the effect being chased is larger than probe65's was.

5. **Cross-life self-knowledge** (was item 3). `individual` beats `recursive` at
   every lag from 12 up, and the gap is the part of each life spent still finding
   out what it is. A prior about itself carried across lives closes it. Measure
   parameter and belief recovery, not survival alone.

6. **Score probe66's declining against survival.** `discovered_convention.py` has no
   closed-loop runner and no survival endpoint; the loop has to be written, as
   probes 64, 65, 67 and 68 each did.

7. **Uncertainty as something *said*, or as a reason to *look*.** Unchanged from
   probe67's list. Both need the frozen motor policy or the caregiver to gain a
   response.

8. **Run probe61's missing controls** if F0 or G2 are ever to be cited.

9. **Replicate the full probe52 childhood-to-adult pipeline.** All causal-stage
   replications still share one lexical/motor parent -- and item 1 forces a new
   parent anyway, so these two should be planned together.

Do not request an external corpus or generated data for these steps.

## On `docs/VISION_AND_STATUS.md`

That document (2026-08-10) recommends freezing this line and building a clean
neural v2. Probes 64--68 were all run after it.

Its diagnosis is right about one thing this repository should stop arguing with:
**the mouth is `argmax` over three scores and that does not become language by
growing.** Where it is wrong is the *order*. The v2 environment cannot be specified
without knowing what grain it needs, and probe68 is the first measurement of that.
The sequence is: grain law (done) -> more bodily axes (item 1) -> how many words can
pay rent as a function of grain -> then a generative learner at the grain the law
says supports it. Reversing it builds a bigger version of a solved toy, which is
what DIRECTION section 4 already forbids.

Revisit when property 1 moves.

## Executive handover (probes 57--60, unchanged)

> A parameter-persistent embodied organism learns a public word lexicon, a
> persistent causal belief over its own hidden food/water/energy state, and a
> full-vocabulary model of how its words change caregiver help. It composes those
> models to communicate its inferred need with 94.99% fidelity and 92% survival on
> held-out lives, while matched causal controls fail.

Five-seed replication: balanced accuracy 0.9181 +/- 0.0066, grounded survival
0.9060 +/- 0.0182, fidelity 0.9471 +/- 0.0034, belief fork 1.0000, scrambled
listener 0.0420, zero belief 0.0000, full promotion gate 5/5.

Binding on all later phases: state a **belief-side endpoint** alongside any
behavioural one; **check the oracle ceiling before locking a gate**; **run the
survey on a seed band disjoint from the treatment**; **state the endpoint in the
currency the world is denominated in**; **measure the margin of the ecology you are
proposing and compare your effect with it** (probe68 supersedes probe67's constant
here); and **do not gate on a contrast that mixes two belief errors** (probe68).

## Standing findings

- **Prediction and attribution come apart** (probe63).
- **A self-model can only localize a fact its own parameter set can express**
  (probe63).
- **Accuracy and regret come apart** (probe65).
- **A belief's point and its width live on different scales relative to the
  decision**, and only the first reaches one (probe67).
- **A belief error reaches a word only if it exceeds the grain, and the grain is
  pinned from below by viability** (probe68).
- **The homeostatic bound is an active part of the decision surface, not a
  backdrop** -- probe62, probe63/67, probe68.

## Artifacts and records

- `runs/organism/probe68_granularity/` -- `treatment.json`, `ceiling_survey.json`,
  `quantum_survey.json`, `quantum_margin.json`, `diagnostic.json`
- `runs/organism/probe67_uncertain_horizon/`, `probe65_future_request/`,
  `probe66_discovered_convention/`, `probe64_portion_request/`,
  `probe63_individual_self/`, `probe62_uncertain_self/`,
  `probe61_discovered_self/full/`, `probe57_structured_causal_self/`,
  `probe58_causal_stage_replication/`, `probe59_online_adaptation/`,
  `probe60_outcome_aware_self_planner/`
- `docs/decisions/2026-08-17-decision-granularity-{preregistration,ceiling-survey,result}.md`
- `docs/decisions/2026-08-16-uncertain-horizon-ceiling-survey.md`
- `docs/decisions/2026-08-16-{future-request,discovered-convention}-result.md`
- `docs/decisions/2026-08-14-portion-request-result.md`
- `docs/decisions/2026-08-03-individual-self-calibration-result.md`
- `docs/decisions/2026-08-02-discovered-self-structure-result.md`
- `docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`
- `docs/decisions/2026-07-26-causal-self-report-independent-verification.md`

Latest verification: **464 tests passed**. All new behavior is default off.
`tests/test_decision_granularity.py` checks that `portion_scale` is inert at 1.0,
that the margin formula holds in closed form on both branches, that doubling a
saturated grant changes the margin by exactly zero while halving an unsaturated one
halves it, that an equivalence gate reads UNRESOLVED rather than pass when its
interval is wider than the band, that the dissociation gate fails when the state
lesion moves with the rate, and that the seed bands cannot overlap. Simulator
event/kind metadata remains audit-only and never enters causal belief, listener
learning, utterance planning, policy, or replay.

## Research basis

- Tang et al. (ICML 2023), self-predictive representation learning:
  https://proceedings.mlr.press/v202/tang23d.html
- Jaques et al. (ICML 2019), counterfactual social influence:
  https://proceedings.mlr.press/v97/jaques19a.html
- Foerster et al. (AAAI 2018), counterfactual multi-agent policy gradients:
  https://ojs.aaai.org/index.php/AAAI/article/view/11794
- Mesnard et al. (ICML 2021), counterfactual credit assignment:
  https://proceedings.mlr.press/v139/mesnard21a.html
- Lambrechts et al. (ICML 2025), asymmetric actor-critic under partial
  observability: https://proceedings.mlr.press/v267/lambrechts25a.html

These papers motivate mechanisms and controls. Only repository experiments are
evidence about this organism.
