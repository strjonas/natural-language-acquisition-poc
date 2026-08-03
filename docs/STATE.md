# STATE

Last rewritten: 2026-08-03. Rewrite this file, never append.

## Where the next agent should start

**The repository's oldest open obstacle is closed.** Probe63 is a full treatment
result, preregistered, five seeds, all seven locked gates passed by the recursive
arm. Read `docs/decisions/2026-08-03-individual-self-calibration-result.md`
first, then the ceiling survey it rests on.

The next work is in "Exact next work" at the bottom. **Do not reopen probes 61 or
62**; both are closed and probe63 does not reopen either.

## Probe63: the standing obstacle, and why it was never about the mechanisms

Every previous probe here tried to make a *learned* self-model beat probe53's
hand-written filter, and none could. `docs/STATE.md` carried the obstacle for
five probes:

> The hand-coded probe53 filter is still better (99.96%, zero error). ... Nothing
> yet shows the learned model doing what the analytic filter cannot.

The reason was one level below any mechanism. **Every organism in this ecology
burned fuel at exactly the species rate.** `food_metabolism` was 0.008 for all of
them, in every life, forever. So what was being called a self-model was a model
of *bodies in general* -- a physics whose constants the designer knows in closed
form. That is why the hand-written filter was exact, and it means no learner
could ever have beaten it. **There was nothing individual to learn.** The
organism was also blind to itself: the body is masked on every tick after birth,
so it got one reading and then flew 400 ticks on dead reckoning.

Probe63 changes both, with three default-inert levers guarded over full lives
(`tests/test_individual_self.py`): `metabolic_spread` (this body's own burn
rates), `uptake_spread` (its own absorption), `interoception_probability` (the
only channel through which either could be found out; readings are delivered in
`info`, never in the packet, so the motor path is bit-identical at any rate).

### The result

Treatment world `metabolic_spread` 0.60, reading rate 0.03, 5 seeds x 40 lives,
every tier on **one shared history**.

| tier | body error | named-need accuracy |
|---|---:|---:|
| `population` -- species filter, *the old repository* | 0.0785 +/- 0.0093 | 0.6649 +/- 0.0172 |
| `snap` -- corrected to truth at every reading | 0.0454 +/- 0.0045 | 0.8148 +/- 0.0127 |
| `learned` -- NLMS self-calibration | 0.0191 +/- 0.0040 | 0.9353 +/- 0.0080 |
| `recursive` -- RLS self-calibration | **0.0133 +/- 0.0023** | **0.9464 +/- 0.0059** |
| `individual` -- born knowing its rates, no readings | 0.0171 +/- 0.0031 | 0.9327 +/- 0.0111 |

- Against the hand-written filter: **83% less body error, +28.2 points** on
  naming the truly lowest need.
- Against **identical evidence with no self-model** (`snap`, the control phase B1
  demands because its first gate is "satisfiable by clipping"): **70.7% less
  error, +13.2 points**.
- `recursive` **beats `individual`** on all five seeds. Being able to see
  yourself occasionally and having to work out what you are beats being born
  knowing your own constants and then blinded. The model is what carries the
  evidence between readings.

Closed loop (context, never gated), 5 seeds x 40 lives: survival `population`
0.535 -> `recursive` **0.640**, fidelity 0.672 -> **0.926**. Survival separates
the species filter from everything else and then saturates -- `snap` 0.620,
`individual` 0.610, `learned` 0.605, `recursive` 0.640 -- exactly as the survey
predicted, which is why it was not gated. Fidelity separates cleanly (+14.6
points for `recursive` over `snap`).

| gate | NLMS | RLS |
|---|---|---|
| G1 corrigibility | pass 5/5 | pass 5/5 |
| G2 report | pass 5/5 | pass 5/5 |
| G3 localization, metabolism world | pass 5/5 | pass 4/5 |
| G3 localization, absorption world | **fail 0/5** | pass 5/5 |
| G4 no false discovery | pass 5/5 | pass 5/5 |
| G5 shuffled readings | pass 4/5 | pass 5/5 |
| G6 rate recovery | pass 4/5 | pass 5/5 |

### The two findings worth carrying forward

1. **Prediction and attribution come apart.** The greedy arm is corrigible and
   confidently wrong about *what* it is: it answers "metabolism" at 0.877 mass
   when that is true and 0.765 when the truth is absorption. Its accuracy gives
   no signal that its self-attribution is wrong -- and in the absorption world
   that misattribution makes it *worse* than having no self-model (0.0590 against
   `snap`'s 0.0428). A self-model can be well calibrated about its own state
   while being confidently wrong about its own nature.
   NLMS's *higher* metabolic share is not better localization; an estimator that
   always answers "metabolism" scores well in the world its bias matches. Only
   the pair of worlds separates them, which is why the ground truth had to move.
2. **A self-model can only localize a fact its own parameter set can express, and
   when it cannot it does not fail loudly -- it produces a confident wrong
   answer.** The first version gave the organism one absorption parameter for its
   whole body, matching `ReportConfig`'s single `portion_small`. In an absorption
   world it then blamed metabolism at 0.703. Giving it per-need parameters fixed
   it without touching the world.

### The reflexive quantity

The organism keeps one copy of its body filter per constant it could be wrong
about, each with exactly that constant perturbed, and reads off
`J[need, parameter] = d(predicted body)/d(log parameter)`. That is not a fact
about where its body is; it is a fact about how its own *model* would respond if
a particular belief about itself were wrong. `test_the_self_jacobian_is_the_real_derivative`
checks it against a filter genuinely rebuilt with the scaled constant, to
**1e-12**, for all seven parameters over full lives.

### A methodological correction, now binding

Probe60 made it binding to check the oracle ceiling before locking a gate.
Probe63 adds: **check it at the operating point the gate will be scored at**, and
**check the ceiling instrument is not undersampled**. G3's thresholds came from a
ceiling measured at reading rate 0.10 while the treatment ran at 0.03;
remeasuring at 0.03 put the locked gate *above* its own apparent ceiling. That
turned out to be an artifact -- the ceiling instrument discards lives with too
few usable rows and had discarded five of eight -- and the online arm beat it.
An undersampled ceiling is its own trap: it can close a mechanism that works.

## Probe62: the uncertainty mechanism, closed

Full record: `docs/decisions/2026-08-03-self-uncertainty-ceiling-survey.md`.
Module: `src/homesocial/organism/uncertain_self.py`.

The `silent_shock_probability` lever opens a 26.6-point gap at `q=1.0` on naming
the lowest need, and **none of it is recoverable**: `posterior_vote` is the
Bayes-optimal rule on the observable history and ties the biased point filter at
every silence rate. The posterior is genuinely calibrated (coverage 0.92--0.94
against nominal 0.90) but uncertainty-timed inspection beats rate-matched random
by only +0.0 to +2.7 points, at budgets consuming half of all help.

Why: the body is bounded in [0,1] and the filter saturates on ~9.7% of ticks, so
its bias equilibrates at +0.051 instead of accumulating. **Do not preregister
uncertainty communication in that ecology.** The lever is kept.

Note that probe63's ecology is *not* that ecology -- being wrong about your own
rate compounds with elapsed time and is not erased by saturation, which is why
the same homeostatic bound no longer caps the error at 0.05. Probe62's closure
was explicitly scoped to its own ecology; a self-uncertainty rung over probe63's
individuality is not foreclosed by it, and is named below.

## Probe61: closed, and what is still undone

Result: `docs/decisions/2026-08-02-discovered-self-structure-result.md`. Four of
five locked gates fail; the mechanism is closed by the preregistered failure
rule. Do not rescue it by raising `L`, changing the sparsity family, re-tuning
per world, or shrinking the sweep -- the preregistration forbids each by name.

Worth carrying: mean `d_eff` is strictly increasing in `K`, so the count is
genuinely responsive to a moving body; G1 fails on accuracy, not responsiveness.
The seed-0 K3 warning does not replicate. The split-by-world execution was
verified bit-identical.

**Still undone:** probe61's preregistered controls (shuffled feelings,
mean-channel-only, minimum-channel-only, labeled three-dimensional reference)
were never run. They void a positive result, so they do not change a verdict
negative on four gates -- but **F0 and G2 remain uncontrolled and must not be
cited as standalone positives until they are run.**

## Where the five properties now stand

From `docs/DIRECTION_2026-07-26.md` section 2.

1. **Discovered** -- *partial*. Probe63 discovers the **values** of its own
   causal constants and **which** of them differ from its species, by
   intervention across worlds whose ground truth moves. It does **not** discover
   its state variables; probe61 tried that and is closed.
2. **Corrigible** -- **yes**, and this is phase B1. Evidence updates the model,
   the improvement survives the `snap` clipping control by 70.7% of error, and
   shuffled readings destroy it.
3. **Load-bearing across uses** -- *partial*, unchanged. The model drives the
   report; survival is reported as context and was never gated.
4. **Productive under novel demand** -- **no**. Untouched.
5. **Reflexive** -- *partial*. The organism computes and acts on
   `d(its own prediction)/d(its own parameter)`, which is a representation of its
   model rather than of its state. It does not yet *report* any of that.

## Executive handover (probe57--60, unchanged)

The v1 result stands and is independently verified and replicated:

> A parameter-persistent embodied organism learns a public word lexicon, a
> persistent causal belief over its own hidden food/water/energy state, and a
> full-vocabulary model of how its words change caregiver help. It composes those
> models to communicate its inferred need with 94.99% fidelity and 92% survival
> on held-out lives, while matched causal controls fail.

Five-seed replication: balanced accuracy 0.9181 +/- 0.0066, grounded survival
0.9060 +/- 0.0182, fidelity 0.9471 +/- 0.0034, belief fork 1.0000, scrambled
listener 0.0420, zero belief 0.0000, full promotion gate 5/5. Full audit:
`docs/decisions/2026-07-26-causal-self-report-independent-verification.md`.

- **Probe59**: online adaptation after a body-rule change. Recalibration works
  (body error 0.0879 -> 0.0164, 5/5) and buys no survival; the legacy planner is
  the bottleneck. `docs/decisions/2026-07-26-online-adaptation-result.md`
- **Probe60**: outcome-aware planner `E[min(next body)]` raises adapted survival
  0.500 -> 0.904 (5/5), but frozen belief reaches 0.890 and stale analytic 0.904,
  so G3 fails. The planner repair is real; continual recalibration is not
  behaviourally load-bearing *in that ecology*.
  `docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`

Binding on all later phases: state a **belief-side endpoint** alongside any
behavioural one, and **check the oracle ceiling before locking a gate** -- now
extended by probe63's clause above.

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
| 59 | online adaptation after a body-rule change | recalibration 5/5; survival gates **fail** |
| 60 | outcome-aware realized-consequence planner | planner repair passes; load-bearing gate **fails** |
| 61 | discovered bodily structure from two sensations | **4 of 5 gates fail**; closed |
| 62 | self-uncertainty ceiling survey | calibrated posterior **ties** the point filter; closed, lever kept |
| 63 | individual body + online self-calibration | **RLS passes 7/7 locked gates**; NLMS corrigible but fails localization. Standing obstacle closed |

Do not reopen without contrary evidence:

- report entropy, head width, vocabulary size, replay, sparse-return loss
  weights, horizons, or counterfactual token-credit variants;
- more guided lexical exposure after comprehension reaches ceiling;
- black-box recurrent self-belief width/loss/head variants;
- a larger legacy neural mouth on the same state;
- compute scale as a substitute for causal structure; and
- probe61's structure discovery and probe62's uncertainty communication.

## Exact next work

1. **Make the self-model pay rent in speech.** Probe63's organism knows something
   no listener can hear: *how fast it burns*. The vocabulary has 60 tokens and 3
   carry meaning, because the listener can only grant three things. A listener
   that could grant a **large or small** portion on request, or grant **early**,
   would make "I burn fast" worth saying -- and it is a fact about the self, not
   about the state, so it is the first genuine candidate for property 4
   (productive under novel demand). Preregister; ceiling-survey it first at the
   operating point, per probe63's clause.
2. **Self-uncertainty, reopened only here.** Probe62 closed uncertainty in *its*
   ecology because the bounded body made error equilibrate. In probe63's ecology
   a newborn is genuinely lost about itself and converges as readings arrive --
   the "bursty" uncertainty probe62 said was required. Run the ceiling survey
   before building anything, and do not treat probe62's closure as either
   permission or prohibition; it was scoped to its ecology.
3. **Run probe61's missing controls** if F0 or G2 are ever to be cited. Small
   driver, cheap, already specified in that preregistration.
4. **Cross-life self-knowledge.** Probe63 learns within one life from the species
   prior. Whether an organism should carry a prior about *itself* across
   regime changes, and whether that helps or produces catastrophic interference,
   is untested. Alternate body regimes and measure parameter/belief recovery,
   not survival alone.
5. **Replicate the full probe52 childhood-to-adult pipeline**; all causal-stage
   replications still share one lexical/motor parent.
6. **Make the learner neural if and only if a task demands it.** Probe63's rules
   are 21 scalars in numpy. That is a feature, not a gap -- the claim is about
   what an online self-model recovers. Do not swap in a network for its own sake.

Do not request an external corpus or generated data for these steps. Larger
compute becomes reasonable only after multi-seed local replication shows the same
architecture, thresholds, and causal controls survive.

## Artifacts and records

- `runs/organism/probe63_individual_self/` -- `ceiling_survey.json`,
  `treatment.json`, `localization_rate010.json`, `closed_loop.json`,
  `shuffled_clamped.json`
- `runs/organism/probe62_uncertain_self/`, `probe61_discovered_self/full/`
- `runs/organism/probe57_structured_causal_self/`, `probe58_causal_stage_replication/`,
  `probe59_online_adaptation/`, `probe60_outcome_aware_self_planner/`
- `docs/decisions/2026-08-03-individual-self-ceiling-survey.md`
- `docs/decisions/2026-08-03-individual-self-calibration-preregistration.md`
- `docs/decisions/2026-08-03-individual-self-calibration-result.md`
- `docs/decisions/2026-08-03-self-uncertainty-ceiling-survey.md`
- `docs/decisions/2026-08-02-discovered-self-structure-result.md`
- `docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`
- `docs/decisions/2026-07-26-causal-self-report-independent-verification.md`
- `docs/decisions/2026-07-26-online-adaptation-result.md`

Latest verification: **378 tests passed**. All new behavior is default off; the
null world confirms the three probe63 levers are inert at their defaults.
Simulator event/kind metadata remains audit-only and never enters causal belief,
listener learning, utterance planning, policy, or replay. Interoceptive readings
are delivered in `info` and never in the packet, so no policy input changed.

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
