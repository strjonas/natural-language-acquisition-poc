# Vision-plus-interim-result paper outline

## Recommended genre

A **workshop position paper with an empirical case study** or a short technical
report is honest now. A conventional full robotics paper is premature because
there is no physical robot, the language is constrained, the body architecture
is hand-specified, and the learned parent is not independently replicated.

## Working title

**From Regulation to Report: A Causal Research Program for Developmentally
Grounded Machine Self-Models**

Alternative, more empirical title:

**Finding Out What Body You Have: Online Individual-Body Calibration for
Grounded Self-Report in a Homeostatic Gridworld**

## One-sentence thesis

Embodied language should be evaluated not by whether an agent sounds
self-aware, but by whether one persistent, corrigible model of the agent's own
individual dynamics is causally necessary for prediction, action, and useful
communication.

## Draft abstract

Language-capable agents can produce fluent first-person descriptions without
maintaining a model that is causally involved in their behavior. We outline a
developmental research program in which socially acquired language becomes
grounded because it improves the regulation of an individual, partially
observed body. As an interim case study, we introduce a small homeostatic
gridworld where organisms are born with unknown metabolic parameters and
receive intermittent interoceptive readings. A recursive least-squares
calibrator maintains an online belief over 21 body parameters and is evaluated
against a species prior, a same-evidence snapshot control, a simpler normalized
LMS learner, and an individual-parameter control. Across five seed blocks of 40
lives, the recursive model reduces body-state error from 0.0454 to 0.0133
relative to the snapshot control and increases correct lowest-need reports from
0.8148 to 0.9464, passing seven preregistered gates including moved-ground-truth,
shuffled-evidence, null-world, and rate-recovery tests. Closed-loop report
fidelity improves, while the survival difference remains unresolved. The
result does not demonstrate natural-language development or consciousness; it
establishes a controlled first rung and motivates experiments comparing
grounded, absent, and matched decoupled language in a unified recurrent learner.

## Contributions that can be claimed now

1. A concrete research program connecting individual bodily regulation,
   socially grounded language, and causal self-model evaluation.
2. An operational distinction between a generic body model and knowledge about
   one individual body.
3. A preregistered same-evidence comparison showing that persistent online
   parameter learning improves state estimation and constrained reporting.
4. A negative mechanistic comparison: normalized LMS is corrigible but fails to
   distinguish correlated causal sources that RLS separates.
5. A public record of failed gates and explicit nonclaims.

Do not claim a novel RLS algorithm, human-like selfhood, language emergence,
natural-language learning, or state-of-the-art robotics.

## Paper structure

### 1. Motivation

- Fluent self-report is not evidence that a self-model controls anything.
- A model of universal body physics is not yet a model of *this* body.
- Development gives language a possible causal role: asking, warning, and
  explaining can change help and therefore viability.

### 2. Positioning

- developmental joint attention and language games;
- symbol emergence in robotics;
- homeostatic motivation;
- recurrent language-conditioned world models;
- continuous robot self-modeling and system identification.

The gap should be phrased as a synthesis: these literatures rarely combine an
individual hidden-body identification problem with productive social language
and causal report evaluation in one developmental loop.

### 3. Research program

Three research questions:

- **RQ1:** Does situated language that improves viability yield better
  self/world representations than token- and proposition-matched decoupled
  language?
- **RQ2:** Can one persistent latent model be made causally load-bearing for
  prediction, control, and report?
- **RQ3:** Does a generated miniature language remain productive under novel
  combinations, listener demands, and newly taught words?

### 4. Interim environment and method

- small partially observed homeostatic social gridworld;
- individual metabolic and absorption constants;
- intermittent readings and designer-specified causal sensitivities;
- five conditions: population, snap, NLMS, RLS, and known-individual/no-reading;
- locked gates, five seed blocks x 40 lives, shared histories for belief-side
  comparisons.

### 5. Interim results

- main body-error and report-accuracy table;
- moved-ground-truth localization table;
- shuffled and null controls;
- closed-loop fidelity and explicitly inconclusive survival contrast;
- RLS-versus-NLMS attribution lesson.

### 6. Limitations

- custom toy simulator;
- standard RLS, hand-authored 21-parameter form;
- exact, cost-free interoceptive readings when they occur;
- three predefined report words, not generated language;
- one inherited trained parent checkpoint;
- five seed blocks and no external environment;
- substantial AI-assisted implementation.

### 7. Ethical position

- functional operationalization, no consciousness claim;
- anthropomorphic language treated as a confound;
- transparent AI contribution disclosure;
- physical deployment requires external safety systems and staged validation;
- future human interaction data requires consent and privacy review.

### 8. Next experiment

Introduce generated sequences describing need, magnitude, timing, and action.
Hold out combinations. Compare grounded teacher interaction with silent and
matched decoupled exposure. Require mediation through listener behavior and
body outcome, plus latent ablation/interchange tests.

## Figures

1. **Program diagram:** body/world/social loops feeding one recurrent state and
   three heads (prediction, action, language).
2. **Probe63 causal diagram:** hidden individual constants -> body trajectory ->
   sparse readings -> persistent estimator -> report -> help.
3. **Main result:** paired per-seed error and accuracy for snap versus RLS;
   avoid bars that hide pairing.
4. **Research ladder:** current individual-body result, productive language,
   grounded-vs-decoupled comparison, physical robot transfer.

## What is missing before submission

For a workshop or arXiv technical report:

- independently reproduce the committed summary and full default run;
- add paired plots and exact uncertainty intervals;
- perform a systematic related-work pass and citation audit;
- explain the post-ceiling-survey addition of RLS chronologically;
- ask one robotics/embodied-AI researcher to challenge the claim boundary;
- replace repository placeholder URLs after creating the remote.

Estimated focused effort: about **one to two weeks** for a defensible workshop
draft using the current result. A stronger paper containing the first productive
language experiment is more plausibly **one to three months**, depending on
training stability and replication—not merely writing time.

## Blog-post route available immediately

Suggested title: **The Body Was Identical, So There Was No Self to Learn**

Narrative:

1. Start with the failed attempt to make a learned self-model beat an analytic
   filter.
2. Explain the diagnostic realization: every organism had the same body.
3. Introduce individual hidden constants and intermittent evidence.
4. Show why `snap` is the fair control.
5. Present the 70.7% error reduction and 13.2-point report gain.
6. Explain the RLS/NLMS attribution split.
7. End with the open question: can this individual fact become productive,
   compositional language that helps a listener act?

That is a complete short post with a genuine result and an honest request for
research direction. It does not need to wait for the grand vision.
