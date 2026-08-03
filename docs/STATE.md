# STATE

Last rewritten: 2026-08-03. Rewrite this file, never append.

## Where the next agent should start

Both of the last two probes are **finished and both close their mechanism.**

1. **Probe61 (structure discovery) is complete: four of five locked gates fail.**
   Result: `docs/decisions/2026-08-02-discovered-self-structure-result.md`. The
   mechanism is closed by the preregistered failure rule. Its passing parts (F0,
   G2, the recovered rates) are real but uncontrolled, because the preregistered
   controls were **not run** -- that is the one piece of unfinished business on
   this probe and it is described under "Probe61" below.
2. **Probe62's ceiling survey closes the uncertainty mechanism** before it was
   built. See "Probe62" below. It also opens the one comparison this repository
   has never been able to make -- learned self-model against hand-written filter
   -- and names the experiment.

**The next experiment is the one probe62 opened**, described at the end of the
probe62 section. Phase A2 is closed; do not reopen it by re-tuning probe61.

## Probe61: complete, closed, and what is left undone

Result: `docs/decisions/2026-08-02-discovered-self-structure-result.md`.
Artifacts: `runs/organism/probe61_discovered_self/full/discovered_self.json`,
20 fits.

| gate | result | |
|---|---|---|
| F0 body tracking | K3 error 0.0161--0.0209, mean **0.0177** vs gate 0.05 | **pass** |
| G1 dimension | per-K matches 2/5, 3/5, 2/5, 5/5; means 1.6 < 2.6 < 3.6 < 4.0 | **fail** |
| G2 mapping | **5/5 in K3 and 5/5 in K4** | **pass** |
| G3 rates | 3/5 where 4/5 required | **fail** |
| G4 lesion | 0/5, as worded | **fail** |
| G5 deployment | K3 fidelity clears 0.80 on 2/5; K4 degenerates 5/5 | **fail** |

Four of five fail, so the mechanism is closed by the preregistered failure rule.
Do not rescue it by raising `L`, changing the sparsity family, re-tuning per
world, or shrinking the sweep; the preregistration forbids each by name.

**Three things worth carrying forward.**

1. **The feared falsification did not occur.** The preregistration said constant
   `d_eff` across `K` would convict the sparsity coefficient of setting the
   answer. Mean `d_eff` is strictly increasing in `K`, so the count is genuinely
   responsive to a moving body. G1 fails on accuracy, not on responsiveness.
2. **The seed-0 K3 warning does not replicate.** `silent_variable_found` is False
   on **5/5** K3 seeds and True on **3/5** K4 seeds. The discriminating clause is
   causal quietness, not drift: K3's spurious extra dimensions never fall below
   an uptake ratio of 0.125, K4's reach 0.078. The old caution was a hand-read of
   drift alone. This still does not license the discovered-`safety` claim -- 3/5
   is below the locked 4/5, the drift overshoots by 35--170%, and nothing in 20
   fits ever exceeded four, so K4's 5/5 stays consistent with a capped upward
   bias. A true `K = 5` world would discriminate; this ecology has none.
3. **The split-by-world execution was verified, not assumed.** Every scalar field
   of all four seed-0 records is bit-identical to the earlier single-process run.

**Unfinished: the preregistered controls were never run.** Shuffled feelings,
mean-channel-only, minimum-channel-only, and the labeled three-dimensional
reference. They exist to void a positive result, so they do not change a verdict
that is negative on four gates -- but **F0 and G2 are uncontrolled and must not
be cited as standalone positives until they are run.** The CLI runs the treatment
loop before controls, so this needs a small driver calling `run_world_seed`
directly with `shuffle_feelings=True` / `channels="mean"` / `channels="min"`.

## Probe62: the uncertainty mechanism is closed before it was built

Full record: `docs/decisions/2026-08-03-self-uncertainty-ceiling-survey.md`.
Artifacts: `runs/organism/probe62_uncertain_self/ceiling_survey.json`. Module:
`src/homesocial/organism/uncertain_self.py`. Guards:
`tests/test_uncertain_self.py` (9).

Reflection -- reporting the reliability of one's own model rather than one's
state -- was the next rung. Following probe60's binding instruction to check the
oracle ceiling *before* locking a gate, the precondition was measured first. It
fails, and it fails at the ceiling, which is the strongest form the failure could
take.

**The lever.** `ReportConfig.silent_shock_probability`, default `0.0`,
default-inert. A silent shock changes the body exactly as a loud one does, from
the same stream; only its perceptible marker is withheld, and the silence draw
uses its own generator so nothing else moves. Guarded by a full-life test: with a
forced identical action sequence the body trajectory at `q=1.0` is bit-identical
to `q=0.0` while every shock flips from loud to silent.

**What was found**, on one shared history, 40 lives, 48 particles:

| q | oracle | naive (probe53 filter) | posterior_vote (Bayes rule) | spread~error r | coverage |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.945 | 0.945 | 0.945 | -0.116 | 0.942 |
| 0.50 | 0.957 | 0.803 | 0.804 | +0.432 | 0.931 |
| 1.00 | 0.959 | 0.693 | 0.692 | +0.415 | 0.923 |

1. The lever opens a **26.6-point** gap at `q=1.0` on naming the truly lowest
   need -- the first substantial headroom on the report endpoint in this
   repository.
2. **None of it is recoverable.** `posterior_vote` is the Bayes-optimal rule
   given the observable history and it ties the biased point filter at every
   silence rate. The oracle's advantage is information the silent shocks
   destroyed, not better inference.
3. The posterior **is** genuinely calibrated: coverage 0.92--0.94 against nominal
   0.90, and the least-certain quartile of ticks carries about 2.4x the error of
   the most-certain. The organism does know when it does not know.
4. **That calibration is actionable, but barely.** At matched inspection budget,
   uncertainty-timed beats rate-matched random by +0.0 to **+2.7** points, while
   inspecting at all is worth up to **+22.1**. Timing by the *true* error is not
   better and is often worse (-6.2 to +3.3), so it is a poor policy rather than
   an upper bound -- the best timing rule is not established, only the size of
   the prize. The +2.7 costs 34.3 inspections per life against ~66 help windows,
   and this test grants inspections for free.

**Why**, measured not inferred: the naive filter's signed bias at `q=1.0` is
+0.051 and **flat across the whole life** against ~0.68 of cumulative hidden
loss. The body is bounded in [0,1] and the filter saturates at the ceiling on
~9.7% of ticks; every saturation erases the accumulated offset. The bound is an
unmodelled evidence channel -- *I know I am not above full* -- and it does the
self-model's job for free. This is `CLAUDE.md`'s "environment doing the model's
job" trap with a number on it, and it retrospectively explains probes 59--60: a
self-model whose error is capped at 0.05 by homeostasis cannot be load-bearing
however accurate it becomes.

**Binding consequence.** Do not preregister uncertainty communication in this
ecology. Finding 2 is the strong reason and is a measured impossibility; finding
4 is a judgement that ~2.5 points at an unaffordable budget does not justify a
listener response, a token and a rent structure, recorded as a judgement. For
self-uncertainty to pay rent the uncertainty must be **bursty**, and a bounded,
frequently-saturated state variable cannot produce that.

### What probe62 opens

The silent-shock lever is the first thing in this repository that makes the
probe53 hand-written filter **strictly wrong** rather than merely redundant.
That is the standing obstacle recorded below under "what this result is still
missing", item 2: *nothing yet shows the learned model doing what the analytic
filter cannot.*

The filter is hand-coded to the visible events, so it cannot represent an
unobservable drift at all. A model fitted to how the body *actually behaves* --
which is exactly what probe61 does, from two scalars -- should absorb that drift
into its learned parameters. The comparison is now runnable and **has not been
run**: fit a probe61 discovered self-model in a `q > 0` ecology and score its
body error and named-need accuracy against `naive` on the same shared history,
with `q = 0` as the null where the two must tie. That is the highest-value
experiment now available, it needs a preregistration, and its endpoints are
belief-side so probes 59--60's policy insensitivity cannot flatten them.

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
| 61 | discovered bodily structure from two sensations | **4 of 5 locked gates fail**; mechanism closed. F0 and G2 pass (uncontrolled); rates within 11%; `d_eff` responsive but inaccurate |
| 62 | self-uncertainty ceiling survey (feasibility, no gates) | calibrated posterior **ties** the point filter at every silence rate; oracle-timed inspection does not beat random; mechanism **closed**, lever kept |

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

**Phase A2 is closed.** Probe61 failed four of five locked gates and probe62
closed the uncertainty rung above it. Neither is reopened by retuning.

1. **Run the learned-versus-analytic comparison probe62 opened.** Preregister it
   first. Fit a probe61-style discovered model in a `q > 0` ecology and score it
   against probe53's hand-written filter on a shared history, with `q = 0` as the
   null where the two must tie. This is the first available test of the standing
   obstacle -- that nothing yet shows the learned model doing what the analytic
   filter cannot -- and its endpoints are belief-side.
2. **Run probe61's missing controls** if F0 or G2 are ever to be cited. Small
   driver, cheap, and they are already specified in the preregistration.
3. Design B1 evidence integration only after (1). Before using survival, prove
   that the correction intervention changes the action selected; otherwise a
   robust policy can flatten a real belief improvement again. Probe62 gives the
   quantitative reason this keeps happening: homeostatic bounding caps
   self-model error at ~0.05 however much of the body is hidden.
4. Test catastrophic interference by alternating body regimes and measuring
   parameter/belief recovery, not survival alone.
5. Replicate the full probe52 childhood-to-adult pipeline; all causal-stage
   replications still share one lexical/motor parent.
6. Future-self report is the remaining reflection rung. Before building it, run
   its ceiling survey the way probe62 did -- it costs hours and it saved building
   an entire mechanism.

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
- `runs/organism/probe61_discovered_self/full/`
- `runs/organism/probe62_uncertain_self/`
- `docs/decisions/2026-08-02-discovered-self-structure-result.md`
- `docs/decisions/2026-08-03-self-uncertainty-ceiling-survey.md`
- `docs/decisions/2026-07-26-guided-report-lexicon-result.md`
- `docs/decisions/2026-07-26-observable-history-filter-result.md`
- `docs/decisions/2026-07-26-explicit-self-belief-result.md`
- `docs/decisions/2026-07-26-ranked-self-belief-result.md`
- `docs/decisions/2026-07-26-relational-urgency-belief-result.md`
- `docs/decisions/2026-07-26-structured-causal-self-model-result.md`
- `docs/decisions/2026-07-26-causal-self-report-independent-verification.md`
- `docs/decisions/2026-07-26-online-adaptation-result.md`
- `docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`

Latest verification: **361 tests passed**. New behavior is default off.
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
