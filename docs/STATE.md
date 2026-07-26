# STATE

Last rewritten: 2026-07-26. Rewrite this file, never append.

## Executive handover

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
   opposed to the state, no past or future self. Utterance choice is an argmin
   over three numbers resolving to one of three words.
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

Do not reopen without contrary evidence:

- report entropy, head width, vocabulary size, replay, sparse-return loss
  weights, horizons, or counterfactual token-credit variants;
- more guided lexical exposure after comprehension reaches ceiling;
- black-box recurrent self-belief width/loss/head variants;
- a larger legacy neural mouth on the same state; and
- compute scale as a substitute for causal structure.

## Exact next work

The local proof is complete and the causal stage is replicated. The next phase
is robustness, continual adaptation, and scope—not another mechanism tweak.

**Step 2 is now the decisive experiment.** It is the only one on this list that
can show the learned model doing something probe53's exact filter cannot: the
filter's constants are baked in and must fail under changed dynamics, while a
learner can re-identify them. Until it passes, this work remains a lossy
re-derivation of a closed form the designer already had.

1. ~~Replicate the causal stage across five independent seeds.~~ **Done**
   (probe58, 5/5 gates, sd <= 0.009). Still outstanding: replicate the full
   probe52 childhood-to-adult pipeline, which all five runs currently share.
2. Change metabolic rates, shock magnitudes, help periods, surface remappings,
   and portion distributions after development. Continue only the causal model
   online and measure adaptation versus frozen, reset, and exact-filter
   controls. Note that the belief self-corrects through homeostatic clipping,
   so the frozen control must be measured, not assumed to collapse.
3. Test catastrophic interference: alternate regimes and require recovery of
   earlier regimes without lexical loss.
4. Expand the latent self-state beyond declared homeostatic axes only after
   the above passes. New dimensions must earn causal intervention evidence.
5. Add reflective communication about predicted future self-change and model
   uncertainty, with receivers acting on those reports and counterfactual
   branch tests preventing templated narration.

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
- `docs/decisions/2026-07-26-guided-report-lexicon-result.md`
- `docs/decisions/2026-07-26-observable-history-filter-result.md`
- `docs/decisions/2026-07-26-explicit-self-belief-result.md`
- `docs/decisions/2026-07-26-ranked-self-belief-result.md`
- `docs/decisions/2026-07-26-relational-urgency-belief-result.md`
- `docs/decisions/2026-07-26-structured-causal-self-model-result.md`
- `docs/decisions/2026-07-26-causal-self-report-independent-verification.md`

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
