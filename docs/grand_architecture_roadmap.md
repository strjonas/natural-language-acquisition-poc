# Grand Architecture Roadmap

Date: 2026-06-23

## Reset

The Q-learning baseline was only a plumbing probe. It is not the vision.

The real target is an online embodied/social learner with:

- persistent recurrent state,
- interoceptive/homeostatic variables,
- action-contingent world feedback,
- social language learned inside the loop,
- a world model that predicts both external and internal consequences,
- enough language capacity to later communicate about its own state.

The core hypothesis is not that "a transformer cannot ever have a self." The
better hypothesis is:

> Self-like organization is most likely to emerge when a system must maintain a
> persistent model of its own action-conditioned body state in order to preserve
> viability, and language becomes useful when it helps that control problem.

Static objective text is not expected to create this by itself because it does
not force the model to distinguish action-caused change, body-state change, and
world-caused change in a persistent control loop.

## What Existing Work Already Covers

Existing systems cover important parts of the space:

- BabyAI: grounded language in a simple world with a teacher-like expert.
- MIA / Playhouse: rich situated language and interaction in a 3D world.
- Gato: transformer sequence model as a multimodal action policy.
- AdA: large-scale RL agent with attention-based memory and fast adaptation.
- DreamerV3: recurrent world-model RL that scales across many domains.
- MineDojo / Voyager / VPT: open-ended Minecraft agents, internet-scale
  knowledge, skill libraries, and behavior pretraining.
- Crafter: tractable open-world survival benchmark with food, water, shelter,
  tools, hazards, and achievements.

The missing combination is:

1. homeostatic/interoceptive stakes,
2. grounded social language that serves those stakes,
3. a persistent world model of external and internal consequences,
4. self-battery evaluation independent of self-report,
5. later narrative/reporting evaluation once the minimal control layer exists.

## Recommended Architecture: Interoceptive World-Model Agent

### Environment

Short term:

- continue with `HomeostaticSocialGrid` until interfaces stabilize.

Medium term:

- fork or wrap Crafter-like survival dynamics,
- add teacher NPCs and joint-attention actions,
- add explicit interoceptive variables if the base environment does not expose
  enough internal state.

Long term:

- Minecraft/MineDojo-style environment if compute and tooling support it.

### Observation Stream

Each timestep should include:

- egocentric visual or object observation,
- proprioceptive state: position, orientation, held object, recent action,
- interoceptive state: hunger, thirst, fatigue, injury, temperature/safety,
- social stream: teacher utterances, agent utterances, nearby entities,
- optional memory/query channel later.

The agent should not receive perfect semantic labels forever. Early curricula
can expose object categories, but real language learning requires periods where
teacher utterances carry information that perception alone does not directly
name.

### Action Space

The action space should mix motor and social acts:

- move/turn/look,
- pick up/use/eat/drink/rest/build,
- point at object/location,
- ask/answer,
- emit short utterance tokens.

Language is not rewarded directly. It is useful only when it improves viability,
prediction, coordination, exploration, or recovery.

### Model Core

Use a recurrent world-model agent, not tabular RL.

Initial serious version:

- encoder for observation/interoception/social text,
- recurrent latent state,
- dynamics model predicting next latent state from action,
- decoder/prediction heads for external observation,
- interoception head predicting next internal variables,
- teacher-language head predicting grounded utterances,
- actor-critic trained from imagined latent rollouts.

This is Dreamer-style in spirit, but modified so interoception is first-class.

Scale-up version:

- recurrent state-space model or Mamba/RWKV-style streaming backbone for long
  persistent state,
- attention or transformer blocks for content-based reasoning and language,
- language decoder coupled to the latent body/world state,
- optional frozen or slowly trained LLM-like module for higher-level planning,
  but not as the source of the minimal self.

The practical architecture is probably hybrid:

```text
perception + interoception + social text
        -> recurrent latent world model
        -> actor-critic for embodied action
        -> language head for grounded speech
        -> reflective head for later self-report tests
```

## Training Plan

### Current Position: 2026-06-23

The project is still in Stage 0, with the first pieces of Stage 1 now entering
the recurrent baseline.

Completed:

- homeostatic/social grid loop with grounded teacher and silent control,
- randomized object placement and masked object-kind observations,
- tabular Q-learning plumbing baseline,
- recurrent actor-critic baseline with PPO-style clipped updates,
- richer egocentric visible-object observation slots,
- previous-event observation features,
- deeper recurrent latent core,
- action-conditioned latent transition head predicting next observation,
  interoceptive state, reward delta, and teacher utterance,
- denser viability-weighted training objective and normalized policy
  advantages,
- padded batched rollout updates for longer local sweeps,
- GAE value targets and bootstrapping for truncated episodes,
- CLI knobs for larger hidden sizes, PPO epochs, batch size, and grid
  dimensions.
- language-necessity diagnostic mode where hidden kind changes independently of
  object position, learner-facing object names are ambiguous, and unsafe
  guessing is costly,
- scripted ask-then-act probe for checking that grounded teacher language is
  actionable before blaming recurrent learner capacity.
- behavior-cloning warmstart from closed-loop teacher-following trajectories,
- counterfactual branch training for action-conditioned interoceptive
  consequence heads,
- viability-rank training over branched action choices,
- causal-attribution probe over frozen recurrent/consequence features,
- initial structured self-report head for need, cause, and consequence reports.
- anti-parrot calibration with majority/shuffled-label controls, random-model
  comparison, and recurrent/consequence feature ablations,
- restricted self-report path that excludes current needs, observed outcomes,
  action identity, and teacher-presence shortcuts.
- report-mediated social triage where emitted need reports causally change
  teacher advice, action, and homeostatic outcome.
- masked-interoception training where exact need values are absent from model
  input but recurrent action/outcome history supports next-need inference,
- deterministic structured need reports generated directly from the learned
  hidden-state consequence estimate,
- hidden-report mediation with latest-only, shuffled-history, and random-model
  controls.
- stochastic nonlinear body dynamics with hidden episode metabolism,
  persistent hunger/thirst pressure, action-order-dependent strain, and visible
  bodily events,
- compositional reports for inferred need, severity, trend, and calibrated
  confidence; latest cause remains an explicit symbolic field.
- emergent two-slot discrete messages trained only through another agent's
  body-relevant use/avoid decision, with balanced social intents and causal
  message-slot interventions.
- convention transfer to fresh receivers trained against a frozen sender from
  small outcome-labeled interaction sets.
- population-level pressure with multiple senders and one shared receiver,
  including a failed use/avoid shortcut audit and a harder self-request game.
- a stronger masked-interoception stochastic body checkpoint whose compressed
  self-estimate supports substantially better food/water/rest/avoid requests.
- shared self-request code where four senders use aligned dominant symbols for
  `food`, `water`, `rest`, and `avoid`, and a receiver chooses aid from message
  alone.
- multi-aspect current self-state language where a shared receiver reconstructs
  continuous needs, low-need flags, dominant need, and severity from a compact
  learned message.
- trend-balanced temporal self-state language using learned self-estimate
  deltas, passing majority and latest-only controls on worsening/steady/
  improving states.
- temporal delta intervention audit where zeroing, shuffling, or negating only
  the learned self-change channel degrades communicated trend.
- counterfactual action-branch trend language where the sender communicates
  action-conditioned future body trend and state from learned imagined
  consequences.
- option-level counterfactual language where short seek/rest/wait branches
  produce a clearer trained-over-random temporal self-state signal than
  one-step primitive action branches.
- direct multi-step option-branch training for the transition model, improving
  horizon-6 final need MSE from `0.111690` to `0.005782` and option-trend
  prediction from `0.2602` to `0.8587`.
- exploratory state collection plus current/future/delta option reports,
  improving balanced option trend communication margin to `0.6213` versus
  `0.4513` random.
- renewable-resource probe that improved water branch coverage and trend margin
  slightly, but failed to solve food coverage or full-state reconstruction.
- rich resource ecology with multiple food/water/shelter/danger objects,
  improving balanced option coverage and absolute trend communication to
  `0.6606` versus `0.5079` random.
- option feature-intervention audit showing the rich-ecology trend message is
  sensitive to the learned self-change features: trained trend drops from
  `0.6606` to `0.5060` when the delta block is shuffled and `0.3527` when it
  is negated.
- three-seed rich-ecology replication of the option-world plus intervention
  pipeline: trained trend averages `0.6585`, random control averages `0.5185`,
  option-majority remains `0.3333`, and negated-delta trained trend averages
  `0.3431`.
- option self-change mediation where compact messages guide a receiver's
  option choice: trained choice accuracy `0.7827` versus `0.5522` random and
  `0.2000` target-majority, with negated-delta trained accuracy collapsing to
  `0.1916`.

Not yet passed:

- free-form reflective language generation,
- stable hidden-report quality across training seeds,
- learned multi-step cause attribution for bodily events rather than symbolic
  rendering of the latest event,
- zero-shot convention alignment between independently initialized agents,
- replicated self-request language across body-model seeds,
- replicated multi-aspect self-state language across body-model seeds,
- wider counterfactual environments where action-conditioned trend cannot be
  partly inferred from action/time regularities,
- option-level communication across independent body-model checkpoints, larger
  balanced trend samples, and less latent-feature shortcut leakage,
- replicated option mediation across body-model seeds and a faster grouped
  option collector,
- longer autobiographical memory and cross-episode continuity,
- reliable ask-then-act discovery in the randomized hidden-kind setting,
- recurrent PPO learning of the language-necessity diagnostic without
  curriculum or imitation warmstart,
- replicated latent option-world and option-language results across seeds,
- richer self-battery tests beyond one-step consequences and attribution.

Interpretation:

Prewarming and curriculum learning are not inherently hacks when they use real
closed-loop trajectories. They are standard ways to make sparse embodied RL
tractable. But they should come after the environment, observation stream,
objective, and baseline learner can pass fixed-world and scripted sanity checks.
The immediate bar is therefore: keep the current grounded-language and
self-battery effects under matched controls, add anti-parrot report calibration,
then decide whether to deepen the grid world or move to a richer survival
environment before scaling randomized PPO training.

### Stage 0: Environment Validation

Goal: prove that teacher utterances are actionable and that homeostatic survival
requires non-trivial behavior.

Baselines:

- random agent,
- scripted teacher-following agent,
- weak tabular baseline,
- small recurrent PPO/A2C baseline.

Passing condition:

- scripted teacher-following agent benefits from language,
- recurrent learner can discover some ask-then-act routines,
- no-language condition is meaningfully worse under matched conditions.

### Stage 1: World Model Pretraining From Online Episodes

Generate episodes from:

- scripted agents,
- noisy teacher-following agents,
- random/exploratory agents,
- early learned agents.

Train the model to predict:

- next observation,
- next interoceptive state,
- reward/viability change,
- action consequences,
- teacher utterance from situated context.

This is not fake self-dialogue. It is predictive training on real closed-loop
trajectories.

### Stage 2: Online Homeostatic RL

Train the actor using:

- viability/homeostatic reward,
- curiosity or information gain,
- empowerment/controllability bonus if needed,
- auxiliary losses for predicting interoceptive consequence and teacher
  response.

The agent should learn that "I am thirsty" is not a prompt string. It is a
predictive/control-relevant internal condition.

### Stage 3: Grounded Language Curriculum

Curriculum examples:

- naming: point at object -> teacher labels it,
- affordance: ask near object -> teacher says "drink water" or "avoid danger",
- spatial: "water near tree",
- hidden property: visually similar objects with different effects,
- social correction: teacher warns before harmful action,
- tool use: "use cup for water", "use axe on tree",
- novel word fast-mapping.

The language data is generated by the world and teacher policy, not by
prewritten transcripts of the agent having a self.

### Stage 4: Minimal Self Battery

Do not evaluate self by self-report first.

Tests:

- self-caused versus externally caused sensory change,
- prediction of own future interoceptive state under planned actions,
- body/tool boundary: when does a controllable object become treated as part of
  action space,
- threat generalization: novel object threatens internal state,
- social distinction: teacher action versus own action,
- latent intervention: identify whether a compact internal state controls
  policy and prediction,
- memory continuity: preserve commitments and internal needs across delayed
  tasks.

### Stage 5: Narrative / Reflective Layer

Only after Stage 4, test whether the agent can truthfully communicate:

- what it needs,
- why it acted,
- what would happen if it acted differently,
- whether an event was caused by itself or another agent,
- whether its report tracks hidden internal variables better than chance.

This is where the "rational/narrative" layer becomes relevant.

### Stage 6: Antagonism / Decoupled Knowledge

Compare:

1. no language,
2. grounded in-loop language,
3. decoupled factual corpus,
4. grounded language followed by decoupled corpus,
5. decoupled corpus followed by re-grounding.

Measure whether decoupled training improves, leaves intact, or damages:

- survival,
- interoceptive prediction,
- self/other distinction,
- grounded language,
- calibrated self-report.

## Transformer / Mamba Question

A pure frozen transformer in a context window can simulate an agent-like self
description, and transformer policies can act in environments. But if the system
has no online learning, no persistent internal state outside context, and no
private interoceptive dynamics that matter for control, then any "self" is
likely episodic and role-like.

The right use of transformer-like models is not to reject them. It is to embed
them in the loop:

- observations and actions are tokens,
- interoceptive state is part of the stream,
- the policy acts online,
- recurrent or external memory persists,
- training includes action-conditioned consequence prediction.

Mamba/RWKV/RetNet-style models are interesting because they provide streaming
recurrent state with scalable sequence modeling. They are not automatically more
self-enabling, but they are architecturally closer to persistent online agents
than a frozen context-only language model.

## Why Existing RL Agents Did Not Obviously Develop A Strong Self

Most RL agents lack several ingredients:

- their reward is external and task-specific, not an internal viability problem,
- they do not need to model hunger, injury, fatigue, or bodily persistence,
- they often reset every episode with no autobiographical continuity,
- they do not have social language as a grounded tool,
- their evaluations reward task success, not self/other distinction or
  calibrated self-report,
- their policies often do not maintain a rich, inspectable persistent world
  model.

They may learn agency-like structures. They usually are not trained or evaluated
for a reflective self that can communicate truthfully about its own state.

## Immediate Implementation Change

Stop treating the current tabular experiment as the research path.

Next concrete implementation:

1. Add a recurrent neural baseline against the current environment. Done:
   MLX-backed GRU actor-critic is implemented, but from-scratch A2C does not
   yet discover resource-use behavior in the hard randomized hidden-kind setup.
2. Add a scripted teacher-following evaluation to verify language usefulness.
   Done at the unit-test and aggregate diagnostic level.
3. Add a language-necessity diagnostic where object position cannot solve the
   task and unsafe guessing is costly. Done.
4. Add curriculum or imitation warmstart so the recurrent policy can discover
   ask-then-act routines before scaling. Done for BC warmstart; not solved for
   from-scratch PPO.
5. Add a world-model module that predicts observation, interoception, reward,
   and teacher utterance from action-conditioned trajectories. Partially done
   with one-step consequence heads plus counterfactual branch training.
6. Add minimal self-battery and report heads. Partially done with
   action-consequence ranking, causal attribution, and structured self-report.
7. Decide whether to scale the environment to Crafter before adding large
   language generation.

The first publishable experiment is not "we created a self." It is:

> Grounded social language improves homeostatic regulation and produces
> measurable self-like control representations in an online recurrent agent,
> while decoupled text does not.
