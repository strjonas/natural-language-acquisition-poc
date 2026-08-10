# Direction: From Bodily Observer to Genuine Self-Model

Date: 2026-07-26
Status: ACTIVE. Supersedes `DIRECTION_2026-07-12.md` and `PLAN_ORGANISM.md`,
both of which are frozen as historical records. `docs/STATE.md` remains the
rolling handover; this file is the destination that handover moves toward.

## 1. Where we actually are

Tagged `v1-causal-bodily-self-report`. Verified and replicated across five
seeds: the organism learns a constrained model of its own hidden
food/water/energy state (91.8% +/- 0.7% balanced accuracy) and a model of how
its words change a caregiver's help, and composes them to ask for what it needs
(94.7% +/- 0.3% fidelity, 90.6% +/- 1.8% survival) while every causal control
fails as required.

That is a real result and it is smaller than it sounds. Stated without
flattery:

- The self-model is **658 scalars** fitted into a template a human wrote. The
  three axes, the linearity, and the sign of every effect were given. The agent
  discovered none of its own structure.
- Probe53's hand-coded exact filter still **beats** it, 99.96% with zero error.
  So today the learned model is a lossy approximation of a closed form we
  already possessed. Nothing yet shows learning buying anything.
- Utterance selection is an `argmin` over three numbers resolving to one of
  three words. There is **no reflection** anywhere in the system.
- The belief survives without any evidence-correction pathway only because
  homeostatic clipping re-anchors it. That is the ecology rescuing the model,
  not the model working.

So: a working bodily observer and a working signaling policy. Not yet a self.

## 2. What "finished" means

The terminal claim this project is trying to earn, stated so it can be attacked:

> An embodied agent that **discovers** the variables of its own hidden state
> without being told them, keeps that model accurate **online** as its body
> changes, **knows and reports how uncertain** it is, can describe states it
> **has not yet been in**, and uses one and the same model for acting,
> predicting, and speaking -- with every one of those established by
> intervention rather than by fluency.

That claim is novel, falsifiable, and does not mention consciousness. It is the
finish line. Nothing below is worth doing except as a step toward it.

### The five properties that separate a self-model from a parrot

Each is a gate, not a vibe. Current status in brackets.

Status updated 2026-08-03 after probe63.

1. **Discovered** -- the agent finds its own state variables. [**partial**:
   probe63 discovers the *values* of its own causal constants and *which* of them
   differ from its species, across worlds whose ground truth moves. It does not
   discover its state variables; probe61 tried and is closed.]
2. **Corrigible** -- evidence updates the model, including evidence it was
   wrong. [**yes**: probe63, phase B1, surviving the clipping control by 70.7%
   of error, destroyed by shuffled readings.]
3. **Load-bearing across uses** -- one model drives action, prediction, and
   speech, so damaging it damages all three together. [partial]
4. **Productive under novel demand** -- it can say things it was never trained
   to say, because a listener needs them. [no -- untouched, and now the binding
   constraint. Probe63's organism knows something no listener can hear.]
5. **Reflexive** -- it represents facts about the model itself (uncertainty,
   staleness, error), not only about the state. [**partial**: probe63 computes
   and acts on `d(its own prediction)/d(its own parameter)`, which is about the
   model rather than the state. It does not yet *report* any of it.]

## 3. The ladder

Every phase: preregister before implementing, lock gates before running,
multi-seed from the start (>= 5), and require the causal controls, not the
average. A phase that fails closes its mechanism; do not tune thresholds to
rescue it.

### Phase A -- earn the word "learned"

**A1. Online adaptation under changed body dynamics.** **RUN 2026-07-26. Gates
1 and 2 failed; gate 3 passed 5/5.** Recalibration works -- body error fell
5.4x and every learned constant moved toward the new truth on every seed -- but
it bought no survival, because survival in this ecology is not controlled by
the self-model. A *perfect* body model survives 0.840, and the adapted system's
own planner pulls that to 0.504. See
`decisions/2026-07-26-online-adaptation-result.md`.

Two lessons are now binding on every later phase:

- **State a belief-side endpoint** (body error, constant recovery, calibration)
  alongside any behavioural one. A survival endpoint measures the policy.
- **Check the oracle ceiling before locking a gate.** If a perfect model cannot
  reach the threshold, the gate is measuring something else.

**A1b. Repair the utterance planner. RUN 2026-07-30. G2 passes; G3 fails.**
Probe60 replaced `min(E[next body])` with the outcome-aware
`E[min(next body)]`, retaining the learned distribution over listener
consequences until after nonlinear viability utility. Survival rose from 0.500
to **0.904 +/- 0.009**, all five seeds cleared 0.89, and scrambled-listener and
zero-belief controls both scored 0.000. The lexical gate remained intact.

The full Phase A1b claim nevertheless fails. Frozen belief survived 0.890 and
the stale analytic belief survived **0.904**, exactly matching the adapted
belief despite 4.2x worse body error. The repaired three-way decision is robust
to numerical state error, so continual recalibration is not behaviourally
load-bearing here. The exact one-step mechanism is closed as a route to that
claim; do not rescue it with objective temperatures, deficit weights, tie
rules, vocabulary restrictions, or ecology retuning. The planner may remain a
controlled instrument in later probes. See
`decisions/2026-07-30-outcome-aware-self-planner-result.md`.

Binding addition: before a later corrigibility experiment uses survival as a
gate, its belief intervention must first be shown to change the selected
action. A policy-invariant belief improvement is still scientifically useful,
but it cannot establish load-bearing self-correction.

*Historical framing, kept because it explains why A1 was run:*

After development, change metabolic rates, shock magnitudes, and portion
sizes. Continue **only** the causal parameters online. Controls: frozen model,
reset-and-relearn, and the probe53 exact filter carrying its now-stale
constants.

Gates: adapted beats frozen on survival and body error; and adapted beats the
stale exact filter. That second gate is the point of the whole experiment --
it is the first result in this repository where learning buys something a
closed form cannot deliver.

Warning from the v1 audit: the belief self-corrects through clipping, so the
frozen control **must be measured**, not assumed to collapse. If frozen does
not degrade, the regime shift was too weak; strengthen it and rerun rather
than reporting a null as a pass.

**A2. Structure discovery.** **Promoted by A1's gate-2 failure**, as that
preregistration specified. Its endpoints are belief-side, so it does not
depend on the survival instrument A1 showed to be unreliable here.
**Preregistered and run as probe61 on 2026-08-03. Four of five locked gates
failed; the mechanism is closed.** It did recover a responsive latent structure
from two scalar viability signals, but it did not reliably recover dimensional
count or survive the causal lesion/deployment gates. See
`decisions/2026-08-02-discovered-self-structure-result.md` and `STATE.md`.

Stop telling the agent it has three axes. Give it
a latent state larger than the body (say 8 dims) plus a sparsity or rank
penalty, and require it to recover the true dimensionality.

Gates: recovered effective dimensionality is 3; each discovered dimension maps
to exactly one true need under intervention; and ablating a discovered
dimension causes the *specific* bodily failure it encodes, not a general one.

Two things the preregistration adds to that sketch, because as sketched the
first gate is not falsifiable on its own:

- **The ground truth has to move.** A sparsity coefficient tuned to return "3"
  will return "3" whatever the body is. Probe61 therefore freezes bodily axes to
  build worlds whose true dimension is 1, 2, 3 and 4, and runs the same learner
  with the same frozen hyperparameters on all four. The recovered dimension must
  track the true one.
- **The supervision has to go too.** Recovering three dimensions is not
  discovery if the training target is still the true three-vector. Probe61
  replaces it with two scalars the world already computes for its own purposes:
  the mean of the bodily variables, which is literally the organism's reward,
  and their minimum, which is what kills it. Neither names an axis or a count.

The fourth world is the payoff: the unmodified frozen ecology's true bodily
dimension is four, because `safety` depletes and enters both viability signals,
and it is exactly the variable this repository's hand-written template cannot
represent.

This is the single largest step toward "not parroted". Until it passes, the
self-model's content is authored, not learned.

### Phase B -- earn corrigibility

**B1. Evidence integration. RUN 2026-08-03 as probe63. All seven locked gates
pass on the recursive arm.** See
`decisions/2026-08-03-individual-self-calibration-result.md`.

The sketch below was implemented almost literally: interoception returns
intermittently and unpredictably, and the systematically biased body is an
*individual* one -- each organism's metabolism and absorption drawn at birth, so
the species constants are wrong about it in a need-specific direction that
persists for the whole life.

That change also resolved section 1's second bullet, which had stood since v1.
Probe53's hand-written filter is not merely worse here; its constants are **not
knowable at design time**, because they are facts about an individual that does
not exist until it is born. A designer can supply the form of an estimator but
not its content. At `metabolic_spread` 0.60 and reading rate 0.03, over 5 seeds:
body error 0.0785 (species filter) -> 0.0454 (`snap` -- identical readings, no
self-model) -> **0.0133** (learned); report accuracy 0.665 -> 0.815 -> **0.946**.

The clipping warning below was justified and is why `snap` is the control the
result is stated against: correcting to each reading and modelling nothing
already recovers a large part of the gap. What survives that baseline is 70.7%
of the remaining error and +13.2 points of report accuracy.

Two findings worth carrying into C: **prediction and attribution come apart** --
a greedy rule is fully corrigible while being confidently wrong about *which* of
its own constants differs, and its accuracy gives no signal that it is wrong; and
**a self-model can only localize a fact its own parameter set can express**, and
when it cannot it does not fail loudly, it produces a confident wrong answer.

*Original sketch, kept because it is what was built:*

Return interoception intermittently and
unpredictably, and add a learned correction gain. Introduce a *systematically
biased* body the current model cannot represent.

Gates: after corruption, error decays faster than the clip-only baseline; and
under systematic bias, the corrigible model recovers where open-loop cannot.
Both are needed -- the first alone is satisfiable by clipping.

### Phase C -- earn reflection

**C1. Calibrated uncertainty.** The agent maintains and reports a confidence.
The listener has a **limited help budget** and must triage, which is what makes
uncertainty worth saying.

Gates: reported confidence is calibrated (predicted error tracks actual error);
lesioning uncertainty costs survival in the budgeted world but **not** in the
unlimited one. That second gate is what proves the report is about the model
rather than decoration.

**C2. Future-self reports.** "I will need X" before it is true.

Gates: counterfactual branch test -- reports must diverge when the *future*
diverges while the present is identical. This is the specific control that
kills templated narration.

### Phase D -- earn scale

Only after A-C. See section 4.

## 4. On ramping up vocabulary and compute

The instinct is right about the destination and wrong about the mechanism, so
this is stated as a binding principle:

> **Complexity is pulled by the task, never pushed by the parameters.**

The vocabulary is already 60 tokens. Exactly 3 carry meaning. Adding tokens
does not add complexity, it adds unused symbols -- because the listener can
only grant three things, so there is nothing else worth saying. Scaling the
vocabulary now would produce a larger table of mostly-zero logits and no new
behavior. `STATE.md` already closed "compute scale as a substitute for causal
structure" as a line, and that judgment stands.

The way to genuinely increase complexity is to make the world **require
distinctions the current protocol cannot express**:

- a listener with a limited budget (forces urgency and triage) -- Phase C1;
- delayed help (forces prediction and future-tense) -- Phase C2;
- a body whose dynamics drift (forces online learning) -- Phase A1;
- hidden state the agent was not told it has (forces discovery) -- Phase A2;
- multiple listeners with different competences (forces addressing).

Each of those *forces* a larger effective vocabulary rather than merely
permitting one. When Phase C passes, the task will demand more symbols than 3
and more compute than 80,000 ticks, and at that point scale is justified by
need and is the obvious next move. Requesting it before then buys a bigger
version of a solved toy.

Concretely: **no external corpus, no generated data, and no larger compute
until A1, A2, and B1 have passed.** Development currently costs ~1.5 minutes.
That is a feature -- it is what makes five-seed replication routine, and losing
it should be a deliberate, earned decision.

## 5. Outstanding debt

- The full probe52 childhood-to-adult pipeline is **not** replicated; all five
  v1 seeds share one lexical/motor parent.
- `train.py` is 9,481 lines and `OrganismConfig` carries 69 knobs, 55 of them
  at default and many belonging to explicitly closed lines. New work should not
  add knobs to it; new mechanisms go in their own module, as `causal_self.py`
  did.

## 6. Frozen

Historical records. Do not append, do not treat as current:

- `docs/DIRECTION_2026-07-12.md`, `docs/PLAN_ORGANISM.md` (superseded here)
- `docs/grand_architecture_roadmap.md`, `docs/evaluation2045.md`
- `docs/results_2026_06_23.md`, `docs/phase_1_spec.md`, `preregistration.md`
- `docs/HighLevelPlan.md` (the original framing conversation)
- the ~39 pre-pivot modules in `src/homesocial/` outside `island/`,
  `organism/`, and `creole/` -- kept as evidence of what was tried

`docs/decisions/` is append-only and is the durable record; `runs/` is
gitignored and is not.
