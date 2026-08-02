# STATE

Last rewritten: 2026-08-02. Rewrite this file, never append.

## Where the next agent should start

**Phase A2 (probe61, structure discovery) is preregistered, implemented, tested
and hyperparameter-selected. The five-seed treatment run has not been run yet.**
Everything needed to run it is in place; the immediate next action is at the
bottom of this section.

The mechanism, in one sentence: the organism is given an overcomplete
eight-dimensional latent and only two scalar sensations per transition -- the
**mean** and the **minimum** of its own bodily variables, which are exactly the
world's own reward signal and its death signal -- and has to work out for itself
how many bodily variables it has, what each one's depletion rate is, and which
of the world's resources restores which. It is told neither the number three,
nor an axis order, nor an axis name, nor a per-life birth reading. This is a
strict *reduction* of privilege against probe57, which was handed the true
per-axis body vector at every developmental transition.

The ground truth is varied **in the world**, by freezing bodily axes, so the
recovered dimension is checked against a number that moves. `K4` is the
unmodified frozen report ecology and its true bodily dimension is **four**,
because `safety` really does deplete at 0.002/tick and enters both viability
signals -- it is precisely the variable probe57's hand-written three-axis
template cannot represent.

| condition | frozen | live variables | true `K` |
|---|---|---|---:|
| K1 | food, water, safety | energy | 1 |
| K2 | food, safety | water, energy | 2 |
| K3 | safety | food, water, energy | 3 |
| K4 | none -- the frozen ecology | food, water, energy, **safety** | 4 |

### What is already established

Preregistration: `docs/decisions/2026-08-02-discovered-self-structure-preregistration.md`,
with three amendments all recorded **before any treatment run** and all
concerning fitting or measurement scale, never a gate, a threshold, or a
control.

Single-fit pilot evidence that the mechanism identifies the body (K4, seed 0,
80,000 developmental ticks, no sparsity penalty). This is **one seed of one
world and is not the result**; it is why the run is worth doing:

| quantity | discovered | world's true value |
|---|---:|---:|
| held-out sensory RMSE | **0.0158** | null 0.1513 |
| food uptake, small / large | 0.195 / 0.614 | 0.20 / 0.60 |
| water uptake, small / large | 0.189 / 0.588 | 0.20 / 0.60 |
| energy uptake, small / large | 0.208 / 0.608 | 0.20 / 0.60 |
| food depletion per tick | 0.0076 | 0.008 |
| water depletion per tick | 0.0115 | 0.012 |
| energy depletion per tick | 0.0158 | 0.016 |

An oracle carrying the world's true constants reaches 0.0032 per-tick rollout
error on the same stream, so 0.0158 is within a small factor of the achievable
floor rather than near the null.

Also settled and locked:

- Hyperparameters, selected on held-out sensory error alone on seed 0 / K4:
  shooting schedule A, learning rate **0.03**. The sparsity coefficient was
  being selected when this handover was written -- read
  `runs/organism/probe61_discovered_self/selection.json`.
- `src/homesocial/organism/discovered_self.py` -- the whole mechanism, the
  audits, the deployment battery, the gate evaluation, and a CLI. Default off,
  own module, no `OrganismConfig` knob.
- `tests/test_discovered_self.py` -- 14 guards, including that the frozen
  ecology is bit-identical by default, that freezing one axis leaves every other
  axis's random stream untouched, that the ground truth actually moves across
  the sweep, that `--seed` actually varies this stage's model, that the two
  sensations are permutation-symmetric across latent dimensions, and that the
  planner's choice is invariant to relabelling the discovered dimensions.
- `ReportConfig.frozen_needs`, `ReportWorld.death_need`, and
  `ReportWorld.force_next_help_need` -- all additive, all default-inert.

### The immediate next action

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.discovered_self --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --run-dir runs/organism/probe61_discovered_self --seeds 5 --lives 100 --worlds K1,K2,K3,K4 --learning-rate 0.03 --sparsity <selection.json chosen_sparsity> --controls
```

Roughly four to five hours. Then write
`docs/decisions/2026-08-02-discovered-self-structure-result.md` against the
locked gates F0 and G1--G5, and rewrite this file.

Two things to expect and report rather than patch:

1. **The planner may go degenerate in K4.** `safety` is unreachable -- no word
   requests it and no help restores it -- so once it becomes the running
   minimum, every token scores the same expected `min(next latent)` and the
   preregistered tie-break picks token 0. This is a predicted failure mode of
   probe60's promoted objective in a world with an unreachable variable. Measure
   it, report it, do not special-case it.
2. **G1 is the gate most at risk**, because at zero sparsity the fit sometimes
   splits one bodily axis across two latent dimensions, which the elimination
   criterion would then count twice.

## Executive handover (probe57--60, unchanged)

The repository now has its first positive, falsifiable causal self-report
result in the minimal report ecology.

> A parameter-persistent embodied organism learns a public word lexicon, a
> persistent causal belief over its own hidden food/water/energy state, and a
> full-vocabulary model of how its words change caregiver help. It composes
> those models to communicate its inferred need with 94.99% fidelity and 92%
> survival on held-out lives, while matched causal controls fail.

The final probe57 result passes every locked gate:

| Endpoint | Result | Gate |
|---|---:|---:|
| Hidden-need balanced accuracy | **92.07%** | >=90% |
| Mean absolute body error | **0.0158** | <=0.03 |
| Grounded report fidelity | **94.99%** | >=60% |
| Grounded survival | **92.0%** | >=80% |
| Scrambled-listener survival | 3.5% | control |
| Grounded minus scrambled | **88.5 points** | >=15 |
| Belief-fork following | **100%** | >=80% |
| Report-fork following | **82.14%** | >=60% |
| Observation-only balanced decoder | 33.39% | chance 33.33% |

This warrants a narrow claim of learned, causally grounded, persistent bodily
self-modeling and self-report **in this ecology**. It does not license claims
of consciousness, sentience, phenomenal experience, unrestricted reflection,
human-like identity, or a metaphysically privileged self.

No larger compute or external/generated corpus is needed now. Replication and
online adaptation tests come before scale.

### Independently verified and replicated

A second agent re-audited probe57 without having implemented it, reproduced it
from the frozen checkpoint, attacked the leakage surface, and replicated the
causal stage across five independent developmental seeds (n=100 lives each):

| Endpoint | mean | sd | seeds passing |
|---|---:|---:|---:|
| Balanced hidden-need accuracy | 0.9181 | 0.0066 | 5/5 |
| Grounded survival | 0.9060 | 0.0182 | 5/5 |
| Grounded report fidelity | 0.9471 | 0.0034 | 5/5 |
| Belief fork following | 1.0000 | 0.0000 | 5/5 |
| Scrambled-listener survival | 0.0420 | 0.0179 | control |
| Zero-belief survival | 0.0000 | 0.0000 | control |
| Full promotion gate | — | — | 5/5 |

A disjoint earlier seed set agreed: 0.9183 +/- 0.0086 accuracy, 0.9120 +/-
0.0084 survival, 0.9510 +/- 0.0054 fidelity.

Two audit findings changed the picture and two defects were fixed:

- The belief is a **contracting observer**, not a dead-reckoner. Error falls
  over life (0.0184 -> 0.0131), and corrupting it mid-life at sd 0.25 leaves
  survival unchanged. Homeostatic clipping plus the help loop re-anchors it.
- The birth interoception reading is worth only a few survival points and
  almost nothing in fidelity. With the population mean, giving zero per-life
  body information, the organism still reaches 0.81 survival and 0.906
  fidelity; a random birth level gives 0.79 and 0.906. Fidelity never falls
  below 0.90 anywhere in that sweep.
- `--seed` did not vary this experiment at all; the harness passed a constant
  `seed_base`. Fixed and guarded by a test. Multi-seed replication before the
  fix would have retrained bit-identical models.
- The natural 1,000,000 seed stride puts `--seed 2`'s developmental worlds on
  the planner battery's evaluation base. The stride is now 10,000,000 and a
  guard rejects any overlap with the evaluation band.

Full audit: `docs/decisions/2026-07-26-causal-self-report-independent-verification.md`

### Probe59: the self-model is no longer the bottleneck

Online adaptation after a body-rule change (metabolism and portions x1.5):
recalibration works and does not help.

- Body error 0.0879 -> **0.0164**; learned-constant error 0.0978 -> **0.0175**,
  toward the new truth on **5/5** seeds.
- Survival 0.480 -> 0.500 (sd 0.095). **Both survival gates fail.**
- A **perfect** body model survives only 0.840, and the adapted system's own
  planner drags that to 0.504.

The legacy planner is the constraint. `causal_social_token` maximizes the minimum
predicted axis, so it abandons the real deficit whenever two needs are close:
need-word rate 100% under a clear deficit, 49.6%/42.2% when the two lowest are
within 0.05. Probe60 tests the preregistered repair below; A2 (structure
discovery) remains promoted.

Binding on all later phases: state a **belief-side endpoint** alongside any
behavioural one, and **check the oracle ceiling before locking a gate** -- if a
perfect model cannot reach the threshold, the gate measures something else.

Result: `docs/decisions/2026-07-26-online-adaptation-result.md`

### Probe60: planner repaired; continual self-model still not load-bearing

The outcome-aware planner preserves the learned listener consequence
distribution until after homeostatic utility: `E[min(next body)]`, not the
legacy `min(E[next body])`.

- Adapted survival rises **0.500 -> 0.904** (sd 0.009, range 0.89--0.91);
  G2 passes 5/5.
- Scrambled-listener and zero-belief survival are both **0.000**; report
  fidelity is 0.911 and the lexical gate remains intact 5/5.
- Body error remains **0.0158** and causal constant error moves 0.0978 ->
  **0.0175** on 5/5 seeds.
- But frozen belief reaches **0.890** and the stale analytic belief reaches
  **0.904**, exactly matching adapted survival. G3 fails (+1.4 and +0.0 points
  against locked +5-point gates).

The planner defect is real and repaired. The stronger Phase A1b claim is
falsified: continual recalibration is more accurate, but the repaired
three-way help policy is insensitive to the remaining numerical error. The
outcome-aware planner can be used as a controlled instrument; it is not
evidence that self-model adaptation controls behaviour.

Result: `docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`

### What this result is still missing

Stated plainly so the next agent does not overclaim it:

1. The self-model's **structure is stipulated, not discovered** — three axes,
   linearity, and every effect sign were given. 658 scalars were fit into a
   correct hand-built template. A fourth hidden bodily variable is
   unrepresentable.
2. The **hand-coded probe53 filter is still better** (99.96%, zero error). This
   is a lossy approximation of a closed form the designer already had. Nothing
   yet shows the learned model doing what the analytic filter cannot.
3. There is **no reflection**: no uncertainty, no reasoning about the model as
   opposed to the state, no past or future self. The improved planner
   enumerates learned one-step consequences but still only allocates one of
   three kinds of immediate help.
4. Emergent signaling of private state is a **populated literature**. The
   distinctive asset here is the control battery and preregistration
   discipline, not the signaling behavior itself.

## What the successful organism learned

### Public lexical development

Probe52 used one uniformly random, body-independent guided joint-attention
event per childhood round. The caregiver labeled only an external resource and
never named the child's current need or a correct future report.

After exactly 60,000 ticks, paired held-out word interventions produced:

- 100% intact food/water/energy resource choice;
- 0% cyclic-word correctness;
- 100% paired action change.

After 200,000 adult ticks and all later causal development, the same parameter
path still scores 100% intact, 1.67% cyclic, and 98.33% paired action change.
The public lexicon is learned, causally effective, and persistent.

### Structured causal self-model

The successful probe57 module is a learned constrained state-space model, not
an exact simulator and not a body-reading shortcut. Its parameters are:

- non-positive per-tick depletion and movement costs;
- non-negative public-surface-conditioned uptake effects;
- non-positive newly visible surface-conditioned shock effects; and
- a 60-token model of visible help-surface versus no-help consequences.

Development used exactly 80,000 primitive ticks, 2,283 one-pass sequence
updates, and 12,247 public listener outcomes. Tokens were sampled uniformly
from the entire vocabulary, independently of body. True developmental body
values supervised dynamics only; no need class, correct word, listener parse,
report score, hidden kind/event, or adult current body entered the model.

Every earlier motor, lexical, recurrent, world-model, and legacy-mouth
parameter remained bit-identical. The causal model identified all three
effective token consequences and all 49 no-help tokens correctly.

### Social-consequence utterance planning

At execution the planner enumerates all 60 tokens. It combines each learned
word-to-help distribution with learned surface uptake and depletion, then
chooses the token maximizing the predicted future minimum of its own belief.
There is no fixed effective-token list in this path.

The final causal controls are decisive:

- mute listener and mute organism: 0% survival;
- every fixed need word: 0% survival;
- zero/frozen self-belief: 0% survival;
- shuffled belief: 35.67% fidelity and 10% survival;
- scrambled listener: 3.5% survival despite 97.96% truthful reporting;
- held-out births: 95.07% fidelity, 87.5% survival;
- held-out portions: 84.88% fidelity, 93% survival; and
- perceptible portion forks: reports change 100%, follow both bodies 82.14%.

Fidelity remains 94.3%-95.6% across the full 400-tick life. This is neither a
birth echo nor a fixed rhythmic code.

## Evidence ladder and closed lines

| Probe | Mechanism | Result |
|---|---|---|
| 48 | consequence-only neural mouth | report fail |
| 49 | unified help uptake | report fail; latent body decodable 64.04% |
| 50 | COMA token critic | report fail |
| 51 | tied lexicon, voluntary inspection | comprehension fail |
| 52 | need-independent guided joint attention | lexical pass; neural report fail |
| 53 | exact visible-history epistemic filter | feasibility pass, not learning |
| 54 | recurrent continuous self-belief | one identity gate fail |
| 55 | ranked continuous belief | identity fail |
| 56 | separate neural urgency head | identity fail; recurrent line closed |
| 57 | structured learned causal self + social planner | **all local gates pass** |
| 58 | causal-stage replication, 5 seeds | **5/5 gates pass**, sd <= 0.009 |
| 59 | online adaptation after a body-rule change | recalibration passes 5/5; survival gates **fail**; planner identified as the bottleneck |
| 60 | outcome-aware realized-consequence planner | planner repair passes (+40.4 survival points); continual-model load-bearing gate **fails** |
| 61 | discovered bodily structure from two sensations | preregistered, built, guarded, hyperparameters selected; **treatment run not yet run** |

Do not reopen without contrary evidence:

- report entropy, head width, vocabulary size, replay, sparse-return loss
  weights, horizons, or counterfactual token-credit variants;
- more guided lexical exposure after comprehension reaches ceiling;
- black-box recurrent self-belief width/loss/head variants;
- a larger legacy neural mouth on the same state; and
- compute scale as a substitute for causal structure.

## Exact next work

Phase A1 established online parameter recalibration on the belief side;
Phase A1b repaired the policy but falsified the claim that the recalibration is
behaviourally load-bearing in this ecology. Do not retune either result.

1. **Phase A2 structure discovery is under way as probe61.** See "Where the next
   agent should start" at the top of this file: it is preregistered, built,
   guarded and hyperparameter-selected, and the five-seed treatment run is the
   next thing to execute. Its endpoints are belief-side and survive the policy
   insensitivity exposed by probes59--60.
2. Design B1 evidence integration only after A2. Before using survival, prove
   that the correction intervention changes the action selected; otherwise a
   robust policy can flatten a real belief improvement again.
3. Test catastrophic interference by alternating body regimes and measuring
   parameter/belief recovery, not survival alone.
4. Replicate the full probe52 childhood-to-adult pipeline; all causal-stage
   replications still share one lexical/motor parent.
5. Add future-self and uncertainty communication only after A2 and B1 pass,
   with listener-mediated consequences and same-present/different-future
   branch controls.

Do not request an external corpus or generated data for these steps. Larger
compute becomes reasonable only after multi-seed local replication shows the
same architecture, thresholds, and causal controls survive.

## Artifacts and records

- `runs/organism/probe52_guided_report_lexicon/gate/`
- `runs/organism/probe52_guided_report_lexicon/adult/`
- `runs/organism/probe53_observable_history_filter/feasibility/`
- `runs/organism/probe54_explicit_self_belief/treatment/`
- `runs/organism/probe55_ranked_self_belief/treatment/`
- `runs/organism/probe56_relational_urgency_belief/treatment/`
- `runs/organism/probe57_structured_causal_self/treatment/`
- `runs/organism/probe57_structured_causal_self/planner_battery/`
- `runs/organism/probe58_causal_stage_replication/`
- `runs/organism/probe59_online_adaptation/`
- `runs/organism/probe60_outcome_aware_self_planner/`
- `docs/decisions/2026-07-26-guided-report-lexicon-result.md`
- `docs/decisions/2026-07-26-observable-history-filter-result.md`
- `docs/decisions/2026-07-26-explicit-self-belief-result.md`
- `docs/decisions/2026-07-26-ranked-self-belief-result.md`
- `docs/decisions/2026-07-26-relational-urgency-belief-result.md`
- `docs/decisions/2026-07-26-structured-causal-self-model-result.md`
- `docs/decisions/2026-07-26-causal-self-report-independent-verification.md`
- `docs/decisions/2026-07-26-online-adaptation-result.md`
- `docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`

Latest verification: **333 tests passed**. New behavior is default off.
Simulator event/kind metadata remains audit-only and never enters causal
belief, listener learning, utterance planning, policy, or replay.

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
