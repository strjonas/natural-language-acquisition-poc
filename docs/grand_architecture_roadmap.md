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
- matched-source three-seed mediation replication on the same option-world
  checkpoint: trained choice averages `0.7936`, random `0.5812`,
  target-majority `0.2000`, and negated-delta trained choice `0.2120`.
- independent option-world mediation replication over two fresh option-world
  heads from the same base body checkpoint: trained choice averages `0.7853`,
  random `0.5597`, target-majority `0.2000`, and negated-delta trained choice
  `0.1864`.
- independent base-body mediation replication on older m8 stochastic body
  checkpoints: trained choice averages `0.7472`, random `0.5858`,
  target-majority `0.2000`, and negated-delta trained choice `0.1590`.
- delta-only option mediation where the receiver sees only predicted
  future-current self-change: trained choice averages `0.7790`, random drops
  to `0.5161`, shuffled-delta trained choice `0.2027`, and negated-delta
  trained choice `0.0738`.
- noisy-branch delta-only mediation with `0.3` option action noise across two
  fresh noisy option-world seeds: trained choice averages `0.5020`, random
  drops to `0.2974`, target-majority remains `0.2000`, and negated-delta
  trained choice `0.0908`.
- calibrated lower-noise branch mediation at `0.15` action noise: trained
  choice improves to `0.5670` over two seeds and negated-delta trained choice
  stays low (`0.1117`), but random rises to `0.3883`.
- direct predicted-future self-model rank control: at `0.15` noise trained
  rank reaches `0.5268` versus message `0.5536`; at `0.3` noise trained rank
  reaches `0.4940` versus message `0.4701`. Random rank is low (`0.2321` /
  `0.1726`) even when random message receivers remain high.
- intervention-aware rank fine-tuning on grouped noisy branches: direct trained
  self-model choice rises to `0.6436` over two seeds at `0.15` noise, random
  self-model rank stays near majority (`0.2062`), and message choice improves
  modestly to `0.5871`.
- rank fine-tuning with dynamics replay: direct trained self-model choice
  remains high (`0.6207`), world-trend accuracy improves from `0.5130` to
  `0.6175`, and negated-delta trained message choice falls to `0.0660`.
- self-model-targeted communication: training messages to communicate the
  model's own predicted best future self-state keeps trained choice useful
  (`0.5631`) while dropping random message choice to `0.1775`; direct trained
  self-model rank remains `0.6288`.
- configurable discrete message capacity for option mediation; initial 3-slot,
  8-symbol, and longer-training probes did not beat the 2-slot/4-symbol
  self-model-target baseline.
- soft-message training support for option mediation; first temperatures
  (`0.8`, `0.35`) underperformed hard straight-through training when evaluated
  with hard messages.
- hard-message score reconstruction support; first weights (`0.1`, `0.03`)
  kept random controlled but did not beat the self-model-target baseline.
- staged hard-message score pretraining, where discrete messages are first
  trained to reconstruct predicted self-model option scores before choice
  training. Paired controls averaged `0.5677` selected-pretrain choice versus
  `0.5547` without pretraining, with a same-seed sweep peaking at 15 epochs.
- pairwise score-rank message shaping via `--score-rank-weight`; first probes
  roughly matched but did not robustly beat staged score pretraining.
- receiver score-distribution distillation via `--score-distillation-weight`;
  first probes also stayed below scorepre15 on the tested seed.
- frozen-code receiver warmup via `--frozen-receiver-epochs`; first probes
  underperformed scorepre15, so receiver scheduling alone is not enough.
- light hard-message commitment pressure via `--message-commitment-weight`;
  weight `0.005` improved all three paired seeds, raising trained choice from
  `0.5677` to `0.5776` while lowering random choice from `0.1602` to `0.1494`.
- independent-body check of light commitment on two older m8 stochastic bodies;
  trained choice improved from `0.5382` to `0.5459` and random choice fell from
  `0.2208` to `0.2131`, but direct self-rank remained `0.6295`.
- capacity-plus-commitment probes: on m11 seed `10043`, 3x4 and 2x8 messages
  both underperformed the 2x4 committed baseline.
- commitment-timing probe: pretrain-only commitment reached the best single
  compact-message result (`0.6031`) but tied constant commitment on three-seed
  mean accuracy and had weaker negated-delta collapse.
- message-code diagnostics for mediation CSVs, showing that commitment schedule
  differences are not explained by simple code collapse or target-code mutual
  information on the best m11 seed.
- row-pattern reuse diagnostics showing that full message tuples are mostly
  one-off sample patterns, so high row-pattern MI is not yet evidence of a
  stable shared convention.
- branch-code target auxiliary loss via `--code-target-weight`, which failed to
  improve the best pretrain-only compact-message seed.
- score-codebook replay via `--message-replay-weight`, which pinned
  score-pretrained hard codes during choice training but did not beat the
  pretrain-only baseline.
- held-out receiver transfer for option mediation, showing that a fresh
  receiver can learn the frozen trained sender above random-sender control but
  still trails the co-trained receiver.
- multi-receiver convention pressure via `--receiver-copies`, which did not
  improve held-out receiver transfer and reduced primary receiver accuracy.
- simpler-convention probes with lower balance pressure and a single-slot `1x8`
  message; both improved or preserved some reuse metrics but lost too much
  self-model choice accuracy.
- score-rank code pressure via `--score-rank-code-weight`, which forced one
  slot toward self-model rank buckets but still reduced primary and held-out
  receiver accuracy.
- receiver bottleneck, held-out receiver score-distillation, sender value-code,
  and sender score-distillation probes; all failed to close the held-out
  transfer gap, narrowing the next convention step to interaction-structure
  changes rather than more single sender/receiver auxiliary losses.
- receiver-turnover option mediation via `--receiver-turnover-interval`; both
  frequent and mid-run receiver resets underperformed, so the next convention
  test should be a true multi-sender/shared-receiver option game rather than
  resetting one receiver pathway.
- true option population mediation with multiple independent senders and one
  shared receiver; naive simultaneous training, explicit sender agreement, a
  smaller two-sender population, and longer population training all failed to
  improve held-out transfer, so the next variant should stage sender/receiver
  introduction instead of training all partners from scratch together.
- staged option population mediation via `--staged-population-epochs`; this
  preserved the base convention exactly and let a new sender use the fixed
  receiver, but the new sender still transferred poorly to a fresh receiver, so
  staged sender introduction needs explicit transfer pressure.
- staged sender introduction with auxiliary transfer receivers; a two-receiver
  ensemble improved the new sender's held-out transfer from `0.5359` to
  `0.5516`, while stronger or larger ensembles hurt, leaving staged transfer
  pressure as the current most promising convention path.
- base-sender token imitation for staged new senders; this improved use of the
  fixed receiver but did not independently solve fresh-receiver transfer, while
  combining imitation with transfer pressure gave the best accuracy/transfer
  balance so far for the new sender.

Not yet passed:

- free-form reflective language generation,
- stable hidden-report quality across training seeds,
- learned multi-step cause attribution for bodily events rather than symbolic
  rendering of the latest event,
- zero-shot convention alignment between independently initialized agents,
- stable option-message conventions under partner turnover, cross-sender
  training, or staged population training,
- replicated self-request language across body-model seeds,
- replicated multi-aspect self-state language across body-model seeds,
- wider counterfactual environments where action-conditioned trend cannot be
  partly inferred from action/time regularities,
- option-level communication across independent body-model checkpoints, larger
  balanced trend samples, and less latent-feature shortcut leakage,
- higher-accuracy noisy option mediation where the trained self-change channel
  stays strong after route regularity is reduced,
- a better noisy-world objective: larger value-gap filtering, doubled world
  training, and split world/mediation noise did not recover high choice without
  reintroducing control leakage or weakening the negated-delta gate,
- message compression improvements so self-model-targeted messages close the
  gap to direct self-model rank (`0.5631` message versus `0.6288` direct),
- better discrete message training, not just larger vocabularies or more slots,
  because naive capacity increases did not improve self-model-targeted choice,
- auxiliary semantic losses for hard messages, because pure soft-message
  training introduced a soft-to-hard evaluation gap,
- a stronger message-shaping objective, because staged score pretraining gives
  only a modest seed-variable gain and still trails direct self-model rank,
- staged or discrete-code-specific sender/receiver optimization, because
  pairwise rank pressure and receiver-logit distillation did not close the
  message bottleneck as simultaneous auxiliary losses,
- stronger compact-message expressivity/stability across independent bodies,
  because m8 direct self-rank remains high while committed messages stay near
  `0.546`,
- better discrete-code training rather than raw capacity increases, because
  extra committed slots/vocabulary hurt the current receiver,
- code semantics and receiver/code alignment diagnostics, because active code
  usage is already broad while compact-message accuracy still trails direct
  self-rank,
- receiver-useful codebook objectives that preserve self-model rank semantics,
  because direct ordinal bucket supervision did not close the transfer gap,
- explicit code stability diagnostics across seeds and independently trained
  bodies, because the current semantics check is still a single-seed comparison,
- stronger rank-plus-dynamics training that recovers the old world-trend metric
  closer to pre-rank levels while keeping direct branch choice above `0.62`,
- message bottleneck improvements so compact communication can express the
  stronger rank-finetuned self-model signal,
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
   Current learned-message work is probing whether option-level self-consequence
   messages survive new speakers and new listeners. The strongest result so far
   is staged sender introduction after the base convention was trained with two
   receiver copies: new-speaker held-out listener accuracy improved modestly
   from mean `0.5334` to `0.5464` across seeds `10043-10045`. Receiver-logit
   and score-head distillation were negative.
   New diagnostics now measure whether messages carry positive body-state delta
   and relative option-value fields. The best staged run does carry those
   fields, but fixed-slot supervision hurt transfer, so the next architecture
   change should create compositional pressure through interaction rather than
   hand-assigned code labels. A first auxiliary field-receiver version also
   hurt transfer, which strengthens the case for making field reports causally
   useful in the environment instead of adding another offline decoding loss.
   A first query-conditioned field-use receiver now measures that directly:
   current messages support relative-value queries and degrade under delta
   interventions. Opportunity-aware accounting shows positive body-improvement
   choices are partially communicated when available, but still recover only
   part of the oracle improvement margin. Auxiliary field-action receiver
   pressure improved the narrow decoder metric but hurt held-out listener
   transfer, so the next step should shift the main interaction/state
   distribution toward positive self-improvement opportunities. That shift is
   now partially implemented with `--min-positive-delta`: on two m11 seeds,
   natural-frequency opportunity states produced strong positive chosen body
   deltas for both a staged new sender and a fresh receiver, while delta
   interventions collapsed below zero. The remaining weakness is target
   imbalance, so the next version should use stratified or weighted opportunity
   sampling rather than hard balancing or raw natural frequencies. A first
   stratified training version (`target_option_resample` for training, natural
   evaluation) removes the majority-target shortcut and preserves positive
   held-out receiver deltas, but reduces raw accuracy and leaves negated-delta
   controls imperfect. A stricter row-wise `reverse_delta_rank` intervention
   now closes that audit gap: reversing option-specific self-change rank drives
   both trained and held-out receivers to harmful choices. The next improvement
   should broaden opportunity diversity and move this causal self-change
   communication into richer, longer-horizon interaction. A first mixed-ecology
   source-training path (`standard` plus `rich`) improves intervention
   robustness but slightly lowers rich held-out transfer, so the problem is now
   less about proving local self-change dependence and more about preserving it
   as the interaction format becomes broader. A first two-turn dialogue
   prototype now exists, but its initial diagnostic is negative: it can correct
   the first proposal somewhat, yet does not recover positive body-delta choice
   without the stronger rank-finetuned mediation pipeline. The next dialogue
   version should make the second turn part of the main embodied/social task.
   The prototype now supports that stronger pipeline: with option-world
   training plus rank finetuning, two-turn dialogue recovers positive
   body-improvement choice and intervention collapse. Forced-proposal
   evaluation now shows the second turn is useful when it is required: after a
   deliberately second-best or worst first proposal, the original reply channel
   recovers positive body-improvement choice, while shuffled/reversed
   self-change channels choose harmful options. The next gap is to make this
   repair pressure arise naturally inside the environment and to scale beyond a
   compact option-message protocol. Repair-proposal training is now a first
   bridge: auxiliary second-best/worst proposal losses improve two-seed repair
   accuracy without losing the ordinary proposal path, but the bad proposal is
   still oracle-injected rather than produced by a situated learned partner.
   A first learned-partner proxy, `model_runner_up`, now audits repair against
   the proposal receiver's own second-most likely choice. It is useful as an
   evaluation mode, but training on it directly was negative/mixed because the
   runner-up is not consistently a homeostatically meaningful mistake. The next
   partner should have partial observations or limited message access, making
   its mistakes learned, situated, and consequential.
7. Decide whether to scale the environment to Crafter before adding large
   language generation.

The first publishable experiment is not "we created a self." It is:

> Grounded social language improves homeostatic regulation and produces
> measurable self-like control representations in an online recurrent agent,
> while decoupled text does not.
