# Direction: From Probe Collection to Organism

Date: 2026-07-12
Status: ACTIVE. This supersedes the "Immediate Implementation Change" sections of
`grand_architecture_roadmap.md` and the "Current Experimental State" log in
`evaluation2045.md`. Those two files are frozen as historical records; do not
append to them.

## 1. Verdict

The project has genuinely established its core mechanism at miniature scale,
and then spent roughly the last two weeks of session time polishing a
sub-sub-subsystem whose returns are now third-decimal-place noise.

What is established (each with causal audits and at least partial replication):

- **In-loop self-model.** A recurrent agent with masked interoception learns to
  infer its own hidden body state from action/outcome history, beating
  shuffled/latest/random controls (M7/M8). This is the crown jewel: a self-state
  representation that exists because the loop demanded it, not because it was
  labeled.
- **Action-conditioned self-prediction.** The option-world model predicts the
  agent's own future body state 6 steps ahead (final-need MSE ~0.13 -> ~0.014,
  replicated across seeds and base bodies).
- **Self-state communication that pays rent.** Compact learned messages derived
  from the self-model causally mediate another agent's body-relevant choices.
  Interventions on the learned self-change channel (negate/reverse-rank)
  collapse performance to below-chance. Replicated across checkpoints.
- **Grounded self-reply improves actual online survival.** In
  `situated_partner_dialogue` with the extended renewable substrate, dialogue
  reaches mean viability 0.740 / min viability 0.217 vs the partial partner's
  0.704 / 0.013, near oracle, and zero-outcome ablation destroys the gain.

What is NOT established, measured against the actual goal (a self that develops
in-loop and is truly communicated):

- **No language.** What exists is 2-4 discrete symbols in 2-3 slots, trained as
  discriminative sender/receiver probes on frozen features. Conventions do not
  transfer robustly (population and turnover experiments were negative). Nothing
  generates utterances. Nothing acquires words.
- **No organism.** The system is ~30 separate scripts training separate heads on
  frozen checkpoints. There is no single agent that perceives, predicts, acts,
  and speaks from one persistent state in one ongoing life. Cross-episode
  continuity is listed "not passed" and was never attempted.
- **No in-loop policy learning.** From-scratch RL (tabular, A2C, PPO) never
  learned ask-then-act. All competence enters via behavior cloning from a
  scripted teacher-follower. The developmental story currently applies to the
  self-*model* (genuinely learned in-loop) but not to the *behavior*.
- **No reflection.** Self-report is structured classification, not generated
  language about the self.

## 2. Diagnosis of the drift

Two failure patterns caused the last ~40 commits (risk gates, recovery
policies, floor calibrations) to produce deltas like 0.6854 -> 0.6861 mean
viability:

1. **Chasing a floor the environment does not permit.** In the current audits
   the ORACLE terminates (dies) in 33-38% of episodes. No amount of recovery
   calibration on the learner side can produce a reliable safety floor in a
   world where perfect play dies a third of the time. The recovery/calibration
   program was structurally unable to succeed.
2. **Loss of altitude.** Each negative result spawned a fix one level deeper
   (risk gate -> learned critic -> rollout labels -> recovery policy -> visited
   risk -> constrained selector -> floor calibrator -> LCB variant), instead of
   propagating the negative result upward to the substrate. The frontier of the
   project became "cross-model floor calibration for constrained temporal
   recovery," five levels below the goal.

The audit discipline (interventions, matched controls, preregistration) is
excellent and must be kept. The experimental topology (ever-deeper offline
probes) is exhausted and must stop.

## 3. The pivot: build the organism

A self is the integration, not the parts. One persistent state that is
simultaneously used for control, for predicting one's own future, and for
communication, across a continuing life. Every ingredient has now been
validated separately; none of them live together. The next phase builds ONE
agent, in ONE environment, trained by ONE online loop:

```text
tokens (heard) + egocentric obs + interoception
        -> encoder -> persistent recurrent latent (carried across episodes)
        -> world-model heads: next-obs, next-interoception, reward,
           caregiver-utterance prediction
        -> actor-critic (motor actions + utterance tokens out)
```

Design commitments:

- **MLX, 5-30M params**, GRU/SSM core, optional small attention block. This
  fits M4/32GB comfortably for online RL.
- **Model-based or strongly model-auxiliary RL** (Dreamer-lite imagination, or
  model-free + auxiliary predictive losses if imagination is unstable —
  decide by experiment, timeboxed). This is the principled fix for the
  from-scratch exploration failure, together with an intrinsic bonus for
  information gain about own body dynamics.
- **BC warmstart is allowed as curriculum bootstrap only**, and every claim
  must state whether it survives without it.
- **Long lives, persistent state.** Hidden state and an episodic memory buffer
  persist across episodes. Difficulty recalibrated so a competent adult policy
  survives indefinitely (oracle survival >= 95%). Death should be a learnable
  consequence of neglect, not a coin-flip of the ecology.
- **Language is a token stream, both directions.** Caregiver utterances arrive
  as token sequences (not one-hot slots). The agent gets an utterance action
  head emitting token sequences from the same closed vocabulary. Language is
  never rewarded directly — it pays rent only through consequences (help
  arrives, warnings avert harm, questions get answered).

### The environment: a creole island

Keep the grid family (interfaces + audits exist), but upgrade it into a world
worth talking about:

- Closed vocabulary ~50-150 words; utterances 1-5 tokens; compositional
  templates: naming, requests, spatial relations ("water near tree"),
  warnings, questions — including questions about the agent's state
  ("you hungry?").
- Hidden object kinds, out-of-sight resources, and delayed effects, so
  utterances carry information perception cannot supply.
- A caregiver NPC with joint attention: reacts to pointing, labels what the
  agent attends to, warns before harm, answers questions, asks questions, and
  helps — but only as well as the agent's communication is informative and
  truthful. Wrong self-reports cause wrong help and real viability cost. This
  is the rent mechanism for truthful self-communication.
- The caregiver's language is produced from a large pregenerated, cached bank
  of situated utterance variants (LLM-generated offline within the closed
  vocabulary, curriculum-staged). No live LLM calls inside the training loop.
  The LLM supplies world content, never the agent's mind — otherwise we are
  back to a simulated self.

### The audits carry over unchanged

A generated self-report counts as "truly communicated self" only if it:

1. tracks hidden interoceptive state the observation does not contain,
2. collapses under interventions on the internal channel (the existing
   negate/shuffle/zero machinery),
3. generalizes to novel states/objects,
4. is causally load-bearing: the caregiver/partner's help, and the agent's own
   downstream viability, depend on it.

This battery is this project's methodological signature. It is the answer to
"not parroted."

## 4. Phases, gates, kill criteria

**C0 — Consolidate and freeze (this week).**
Commit pending work, tag the repo `v0-probe-era`. Write
`docs/results_synthesis.md`: one table — claim, strongest audit, replication
status — covering M0-M15 and the option/dialogue line. Create `docs/STATE.md`
(<= 2 pages, rewritten each session, never appended). Declare the
recovery/calibration line closed.

**C1 — Organism v0 on the current grid (~2 weeks of sessions).**
One process: encoder -> persistent latent -> world-model heads -> actor-critic,
trained online on the extended renewable substrate with the difficulty
recalibration (oracle survival >= 95%).
GATE G1: matches or beats the best patchwork online result (mean viability
>= 0.74 equivalent under the recalibrated ecology) with grounded > silent >
shuffled separation on held-out seeds, and survives a BC-ablation statement.
KILL: if after 2 weeks it cannot beat the partial-partner baseline, stop and
rethink architecture at the top level — do not tune hyperparameters past that
point.

**C2 — Token language in (comprehension).**
Creole-island upgrade + cached utterance bank. Agent consumes token sequences;
caregiver-utterance prediction is an auxiliary loss.
GATE G2: in-loop acquisition — ask-then-act discovered in-loop; comprehension
of dozens of words measured by held-out-world performance and fast-mapping
probes (novel word -> correct behavior after k exposures); grounded vs
silent/shuffled separation.

**C3 — Token language out (generative self-report that pays rent).**
Utterance action head; caregiver help conditional on informative, truthful
communication; partner cannot see agent internals.
GATE G3: generated (not selected) self-reports pass the full audit battery
(section 3) and improve survival vs silent/scrambled-report controls.

**C4 — Self-battery + autobiographical continuity on the organism.**
Re-run the M4-style battery (self/other attribution, k-step self-forecasting,
latent centrality, threat generalization) on the unified agent. Add
cross-episode tests: does persistent memory inform behavior and report
("avoided the thing that hurt it yesterday, and can say so")?

**C5 — Scale.** Only after G1-G3: request GPU compute for 10-100M+ models,
longer lives, populations, richer worlds (Crafter-scale). Also the right point
to run the original antagonism experiment (grounded vs decoupled corpus) at a
scale where it is publishable.

## 5. Operating rules (anti-drift, binding for future sessions)

1. **Altitude rule.** Before any experiment, write one sentence naming which
   goal-level metric it moves (survival, language acquisition, self-report
   fidelity, self-battery). If the connection needs more than two levels of
   indirection, do not run it.
2. **Timebox rule.** Any single approach gets at most 3 negative probes or 2
   days, whichever comes first. Then the negative result propagates UP one
   level (question the substrate), never down (add a patch).
3. **One harness.** All headline results flow through one benchmark command
   with fixed seeds and the standard audit battery. No new one-off scripts;
   the existing ~30 probe scripts are frozen.
4. **Docs.** `docs/STATE.md` is the single live status document, rewritten not
   appended. Individual experiment records go to `docs/decisions/` as small
   dated files. The two mega-logs stay frozen.
5. **Session end ritual.** Update STATE.md: where the frontier is, what is
   next, what is currently forbidden (closed lines).

## 6. Resource requests (to Jonas)

1. **LLM API key: requested NOW.** Needed for C2 — offline pregeneration of the
   caregiver utterance bank (cheap model is fine; everything is cached, no
   in-loop calls) and later as an evaluation judge for free-form reports.
2. **GPU compute: deferred.** Hold until G1 + G2 pass locally. The ask will
   come with a concrete spec (model size, run count, estimated hours/cost).
   Local M4/32GB is sufficient for C0-C3 at the stated scale.
