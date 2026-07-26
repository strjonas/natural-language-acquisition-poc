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

1. **Discovered** -- the agent finds its own state variables. [no]
2. **Corrigible** -- evidence updates the model, including evidence it was
   wrong. [no: there is no correction pathway at all]
3. **Load-bearing across uses** -- one model drives action, prediction, and
   speech, so damaging it damages all three together. [partial]
4. **Productive under novel demand** -- it can say things it was never trained
   to say, because a listener needs them. [no]
5. **Reflexive** -- it represents facts about the model itself (uncertainty,
   staleness, error), not only about the state. [no]

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

**A1b. Repair the utterance planner.** Newly identified and now on the critical
path. `causal_social_token` maximizes the *minimum* predicted axis. A need word
concentrates help on one axis and stops improving `min` as soon as that axis
passes the second-lowest; a word with diffuse listener mass lifts every axis a
little and so raises `min` directly. When the two lowest needs are close, the
diffuse word wins and the organism stops naming what it lacks -- need-word rate
falls from 100% under a clear deficit to 49.6% (developed) and 42.2% (adapted)
when the two lowest are within 0.05.

Preregister a replacement objective -- expected deficit reduction, or survival
probability under the learned dynamics, rather than max-min -- gated against
the oracle-planner ceiling of 0.840. Do not bolt a fix onto A1.

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

Stop telling the agent it has three axes. Give it
a latent state larger than the body (say 8 dims) plus a sparsity or rank
penalty, and require it to recover the true dimensionality.

Gates: recovered effective dimensionality is 3; each discovered dimension maps
to exactly one true need under intervention; and ablating a discovered
dimension causes the *specific* bodily failure it encodes, not a general one.

This is the single largest step toward "not parroted". Until it passes, the
self-model's content is authored, not learned.

### Phase B -- earn corrigibility

**B1. Evidence integration.** Return interoception intermittently and
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
