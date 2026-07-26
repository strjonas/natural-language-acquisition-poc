> **FROZEN HISTORICAL RECORD.** Superseded by `docs/DIRECTION_2026-07-26.md`.
> Kept as evidence of what was tried. Do not append to it and do not treat any
> plan, status, or number in it as current. Current state: `docs/STATE.md`.


Context:
think in humans. many religons, especially christianity emphasize 'the flesh' and 'the spriit'. There seem to be two parts of us. The agentic part of us, that is in a way, an animal, and the rational part. both come together. The animal part gives us the illusion of the self that our rational part builds stories around (not sure if this is correct, but swamping over it, thats roughly how i understand it). The same should be possible to achieve in artificial systems also, right? 
and yeah, stuff is only modeled when its useful. Well, why did humans develop language? it was very powerful abstraction taht allowed us to talk about compmlex things with each other. So, in the training data, we need dialogue, or, when we do it from scratch, multiple agents (tho from scratch might be too long). but yeah, don't need many, just two maybe. Could also be artifical in a way? like in some movies a baby being raised by animals or a robot. but when the robot poitns to something in a 'baby state', it should be told that this is called 'x' for example. Idk something like that, but you have seen that already, well done. but question would be how to harcode that, or if we need to build a small (but capable enough) llm in the model that runs, but that would slow trainign run down... but we can pregenerate enough maybe like in video games. huge dialogues in a huge world, starting baby like and progressing. that could work. 
sure, this might need a very comprehensive enviornment. But the frontier keeps pushing whats possible. And, thinking again of brain in the vat, we might be able to similate it or take shortcuts. Just think hard about what we can do, there has to be a solution. Believe in yourself, reserach, think, maybe trial and error. If something could work. lets try it. Pushing the frontier is exactly about this brave process.
as i read on, i really think you have a good grasp of what is needed all together. building the environment, the architectue, setup. I can provide you with a cheaper model api key that you can use to let it generate all the dialogues, etc. that is needed. 
anyway, lets start with a plan how to do this. Think deep. We might need an overaching plan first, and then subplans for the different thing. The overaching plan is the thing to get right, it stands and falls with that. So reserach and think deeply. I will then route your plan to chatgpt 5.5 xhigh to critique and expand. I mean you two can collaborate. You can also be open about any questions. I mean you two roughly play in the same era, you're like two super intelligent super experienced colleagues who figure this out together. I think ping pong a few times and a plan stands. so you go ahead and startthe closed-loop-coupling is genuinely much cleaner than my thesis i think. Yeah, that makes sense, if the agent takes actions but somehow they don't change his condition, why keep it as a state, it just isn't useful. Good reframing!
regarding the dialouge idea as poison, i think its a misunerstanding maybe: what i mean is this: when a baby is born into this world, many characters appear in its world, in addition to all the object. the objects can be done by a physics engine, we have that kind. But we need real characters who already possess language, because that significantly decreseases the needed time from millions of years of language development into a few years of language learning in humans, and thus, hopefully also decreseases the training compute necessary by a factor of millions. I don't mean pregenerate a dialogue including the agnet where a self should develop, i mean the agent can act, and based on that action, like in the video game where the human is the play,er the world acts, and the people in the world do. For example, the agent outputs an action that could be related to pointing to a tree. The ficitonal person says 'tree'. Just like a mother would. This is not faking a self. This is just how people learn and so should the robots. You know? what i am trying to do is to create a simple human, as simple as possible so that it still works, and a simple enviornment, as simple as possible that it still works. So that, we get an artifical agent that works in a simplified way, just like a human.. You knwo what i mean? I mean this universe could be much simpler, with simpler language, like a creole, just expressive enough to get what i want, simpler world, like people living on a stranded island with just enough objects to actually leanr. youk now?  that has to be possible. And lets forget what i wanted to do before - partly. This - what i just described - is what i want to build now. 
so now try it with that again. Reserach, think, deeply. How could this be done. Think big. Think pushing the frontier.

---

I would build in that direction, but I would tighten it.

**What The Agent Got Right (the output isn't avaialble in this file)**

1. **Closed-loop language learning is the right core.**  
   Your idea is much closer to child language learning than corpus pretraining: the agent acts, the world responds, a social teacher labels or corrects something grounded in the shared situation.

2. **BabyAI / MiniGrid is a good starting substrate.**  
   BabyAI was explicitly built for grounded language learning with synthetic compositional language, curriculum levels, and a teacher-like bot. It is not your full idea, but it is a strong base. Sources: [BabyAI paper](https://arxiv.org/abs/1810.08272), [MiniGrid docs](https://minigrid.farama.org/), [MiniGrid GitHub](https://github.com/Farama-Foundation/Minigrid).

3. **A grammar teacher is better than an LLM teacher at first.**  
   I agree with the agent here. At runtime, an LLM teacher would be slow, expensive, noisy, and hard to analyze. A deterministic teacher with access to world state is cleaner. Use an LLM later to generate varied surface language or curriculum templates, not as the core world loop.

4. **The “three conditions” idea is genuinely good.**  
   Compare:
   - no language/objective knowledge,
   - grounded in-loop teacher knowledge,
   - decoupled corpus/objective knowledge.

   That gives you a clean test of your real thesis: not “knowledge is bad for self,” but “decoupled knowledge may weaken self-like organization because it does not pay rent in the organism’s loop.”

**Where The Agent Overreached**

The plan is directionally right, but it makes a few things sound easier than they are.

BabyAI is not really “a mother teaching a child.” It is mostly instruction-following in a gridworld. It gives you objects, partial observability, synthetic language, and tasks, but not rich social development.

DreamerV3 is powerful, but I would not start there. It is a strong world-model RL algorithm, and the current Dreamer line is impressive across many domains: [DreamerV3 paper](https://arxiv.org/abs/2301.04104), [Nature version](https://www.nature.com/articles/s41586-025-08744-2), [DreamerV3 code](https://github.com/danijar/dreamerv3). But adding Dreamer, language, homeostasis, social teaching, and self-metrics all at once creates too many failure modes.

Also, “self dissolves” is too strong as an engineering target. You need to operationalize it as measurable self-like organization: agency detection, body/action ownership, internal-state prediction, persistence under threat, and whether a latent “self-state” becomes causally central for control.

**Should We Build It Like That?**

Yes, but as a staged research program:

**Phase 1: Homeostatic Social MiniGrid**

Build a tiny world with:
- hunger/thirst/energy/safety internal variables,
- objects like food, water, shelter, danger, tools,
- partial observability,
- actions like move, look, point, pick up, eat, drink, ask/gesture,
- teacher NPC that labels objects/actions only when contextually appropriate,
- survival/viability reward based on homeostatic regulation.

This is where Keramati and Gutkin’s homeostatic RL idea is relevant: reward can be derived from reducing deviation from viable internal states, not from arbitrary external tasks. Source: [Homeostatic reinforcement learning](https://elifesciences.org/articles/04811), [RL theory for homeostatic regulation](https://papers.neurips.cc/paper/4437-a-reinforcement-learning-theory-for-homeostatic-regulation).

**Phase 2: Language As Survival Tool**

Do not reward “understanding words” directly. Make language useful because it helps the agent survive.

Example:
- agent points at unknown object,
- teacher says “water”,
- later agent hears “water near blue rock,”
- agent can use that to restore thirst.

Then test whether language improves viability, generalization, and planning.

**Phase 3: Self Battery**

Before using the word “self,” define tests:
- Can the agent distinguish self-caused sensory changes from world-caused changes?
- Does it learn that some variables belong to its own body/state?
- Does it protect future viability, not only immediate reward?
- Does it generalize “danger to me” across new objects?
- Does a compact internal self-state emerge that predicts action consequences and survival?

**Phase 4: Antagonism Experiment**

Run the three conditions:
1. no teacher language,
2. grounded teacher language,
3. decoupled corpus or abstract factual pretraining.

Prediction I would actually trust enough to test:

Grounded language should improve self-like control because it helps the agent regulate itself. Decoupled language may either do nothing, distract learning, or create brittle representations unless it is later coupled back into action.

**My Main Alternative Idea**

Do not frame the architecture as “animal part plus rational part” too literally. That is a useful human metaphor, but architecturally I would build:

- **body loop:** internal needs, action costs, viability,
- **world loop:** perception, objects, physics, affordances,
- **social loop:** teacher, labels, correction, joint attention,
- **memory/world model:** predicts consequences,
- **language layer:** maps symbols onto useful distinctions inside the loop.

The “self” should not be a module. It should be an emergent control structure that appears because the system must model “what happens to this body if I act.”

So: the agent’s plan is good, but I would simplify the first build. Start with Homeostatic Social MiniGrid, deterministic teacher, small recurrent policy, strict metrics. Add Dreamer/world models only after the environment and tests prove useful.

---

## Implementation Log: 2026-06-23

Built the first Homeostatic Social Grid prototype:

- homeostatic variables: food, water, energy, safety,
- situated objects: water, food, shelter, danger, tree, rock,
- actions: turning, moving, pointing, asking, consuming, resting, waiting,
- deterministic grounded teacher,
- silent-teacher control condition,
- randomized object placement for experiments,
- masked object-kind encoding for the learner,
- tabular Q-learning baseline,
- unit tests and ASCII demo.

First result:

The tabular Q-learning learner does not show a reliable grounded-teacher
advantage after 3000 training episodes across initial seeds. This is currently
interpreted as a learner/capacity/exploration limitation, not as evidence
against the closed-loop language idea. The social loop emits grounded,
action-contingent utterances, but the tabular learner does not reliably discover
the ask-then-act routine under randomized layouts.

Immediate next step:

Add a teacher-following sanity baseline to verify that the teacher utterances
are actionable in principle, then move to a small recurrent trainable policy.

---

## Roadmap Correction: Grand Version

The tabular Q-learning baseline was only a plumbing probe. It is not the vision.
The real architecture should be an online recurrent world-model agent with
interoceptive state, grounded social language, and later a reflective language
head. The guiding document is now `docs/grand_architecture_roadmap.md`.

Updated target:

- build an Interoceptive World-Model Agent,
- train it in a closed embodied/social loop,
- make language useful for viability rather than rewarding language directly,
- evaluate minimal self-like control before evaluating self-report,
- compare grounded language against decoupled text only after the agent has a
  stable self-relevant control structure.

Immediate implementation direction:

1. Keep the current grid only as a fast interface testbed.
2. Add a recurrent neural baseline. Done: MLX GRU actor-critic is in place.
   Current result: from-scratch A2C still does not learn resource use in the
   randomized hidden-kind setting after 200 local episodes.
3. Add curriculum/imitation warmstart and batched rollouts so the recurrent
   policy can discover ask-then-act routines.
4. Add a world-model prediction module for observation, interoception, reward,
   and teacher utterance.
5. Evaluate whether to scale the environment toward Crafter before adding
   richer language.
