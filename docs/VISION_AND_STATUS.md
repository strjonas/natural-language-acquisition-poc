# Vision and research status

Date: 2026-08-10

## What the original documents actually propose

`HighLevelPlan.md` and `grand_architecture_roadmap.md` describe a “simple human
in a simple world”:

- an embodied agent with hunger, thirst, energy, safety, and action costs;
- a partially observed world whose dynamics respond to the agent;
- social caregivers who already possess a small language;
- joint attention through pointing, asking, labeling, and correction;
- a persistent recurrent/world model predicting external and internal effects;
- language rewarded only indirectly through prediction, coordination, and
  viability;
- later reporting and reflection, evaluated only after nonverbal self-relevant
  control exists;
- a comparison among no language, grounded in-loop language, and matched but
  decoupled text.

Its best idea is still the sentence: **the self should not be a module**. A
self-relevant structure should become load-bearing because the learner needs to
predict “what happens to this body if I act.”

## Acquisition is not emergence

The caregiver world is primarily a theory of **language acquisition and
grounding**. The language convention already exists in the caregivers; the
agent must connect forms to referents, actions, needs, consequences, and social
intent. This is a sensible shortcut: it studies a developmental timescale
rather than asking two agents to recreate cultural evolution.

**Language emergence** is a different experiment. Multiple initially naive
agents would have to invent and stabilize a convention without a pre-existing
teacher lexicon. The repository has tested small emergent communication codes,
but not the development of an open-ended language.

For this project, “real language” should be defined operationally and modestly
as a **miniature productive language**:

- the agent generates token sequences rather than selecting one of three words;
- known words are combined in novel need/action/object/property combinations;
- a listener acts correctly on combinations withheld during training;
- reference depends on shared attention and dialogue history;
- the agent can learn a new word from a small number of situated interactions;
- utterances change outcomes, and shuffled or decoupled language loses that
  benefit.

This would be real progress on language development without pretending to have
reproduced adult natural language.

## Novelty map

| Part of the vision | Closest established work | Assessment |
|---|---|---|
| caregiver labeling and joint attention | developmental robotics has studied robot-caregiver joint attention since at least [Nagai et al. (2003)](https://www.jstage.jst.go.jp/article/tjsai/18/2/18_2_122/_article/-char/en) | established |
| embodied symbol systems | the [Symbol Emergence in Robotics survey](https://arxiv.org/abs/1509.08973) frames symbols as socially organized through physical and semiotic interaction | established field |
| lexicons and compositional conventions through interaction | [grounded language games](https://arxiv.org/abs/2004.09218) cover naming, compositionality, populations, and robots | established field |
| synthetic grounded language curriculum | [BabyAI](https://openreview.net/forum?id=rJeXCo0cYX) combines gridworld tasks, compositional instructions, and a synthetic teacher | close substrate |
| scripted social agents in gridworlds | [SocialAI](https://arxiv.org/abs/2107.00956) studies social cognition and language with scripted partners | close environment family |
| language inside a predictive world model | [Dynalang](https://proceedings.mlr.press/v235/lin24g.html) learns from future prediction of text and sensory observations and can generate grounded language | close architecture |
| needs grounding words with a caregiver | [Markelius et al. (2023)](https://arxiv.org/abs/2310.13377) study hunger, thirst, curiosity, affect, and human caregiver feedback in a virtual robot | direct conceptual overlap |
| continuous robot self-modeling | [Bongard et al. (2006)](https://pubmed.ncbi.nlm.nih.gov/17110570/) show damage recovery through ongoing self-modeling | established |
| this individual body's hidden constants carried into self-report | Probe63; no exact match found in the targeted search | narrow empirical contribution |
| grounded vs. matched decoupled language, judged through body regulation and causal self-model use | proposed, not yet run; no exact match found in the targeted search | strongest prospective contribution |

The conclusion is not “the whole vision is novel.” It is: **the research
program has a defensible seam between several mature areas, and Probe63 gives
that seam an initial empirical foothold.**

## Did the project take the right path?

Partly.

It took the right path by starting with a fast custom simulator, making needs
and social consequences explicit, building causal controls, preserving failed
experiments, and refusing to infer a self from fluent output. Probe63 also found
a real design principle: individuality must exist before an individual
self-model has anything to discover.

It drifted away from the grand language vision by accumulating many specialized
offline probes, discrete codes, and hand-structured reports. The current report
mouth selects among three predefined need words. The RLS model has 21 scalar
parameters in a designer-specified causal template. These are useful scientific
instruments, but continuing to enlarge them will not gradually turn them into a
language-developing child.

The existing repository should therefore remain a **microscope and baseline**,
while the next language learner is built as a small, clean v2 path. Rewriting
the 9,000-line experimental training module would obscure both stories.

## Recommended v2 path

### Research question

Does a productive social language become more grounded and more useful when its
distinctions help a persistent learner predict and regulate its own individual
body, compared with no language and a matched decoupled-language control?

### Minimal architecture

```text
vision/object slots + interoceptive evidence + dialogue tokens
                         |
                  recurrent latent state
                  /          |           \
       next-world/body     actor/value    token decoder
          prediction          head       (generated speech)
```

The key constraint is shared state: prediction, action, and speech must depend
on the same persistent representation. Separate heads are acceptable; separate
facts supplied directly to the report head are not.

### Curriculum

1. **Joint attention:** point/look; caregiver emits object and action words.
2. **Need-action grounding:** ask for water/food/rest and experience the bodily
   consequence.
3. **Properties and relations:** small/large portions, near/far, before/after,
   safe/dangerous.
4. **Novel composition:** hold out combinations such as “large water now” while
   training the component meanings elsewhere.
5. **Dialogue:** clarification, correction, and fast mapping of a new symbol.
6. **Matched decoupling:** expose a control agent to the same token counts and
   propositions at times when they cannot guide action.

A procedural teacher should determine semantics from simulator state. A cached
paraphrase bank can add surface variation. A live LLM should not be the hidden
source of the agent's competence or the arbiter of correctness.

### Decisive evaluations

- productivity on withheld word combinations;
- few-shot mapping of a new word in a shared-attention episode;
- viability and calibration with grounded, silent, shuffled, and decoupled
  language;
- mediation: language changes listener action, which changes the body;
- necessity: ablate or interchange the shared latent and measure effects on
  prediction, control, and report;
- independent training seeds, including the parent developmental run.

## Distance to the vision

| Milestone | Present state | Plausible focused effort |
|---|---|---|
| honest public repository and interim story | ready | now |
| small blog post / project application | enough evidence already | days |
| generated miniature language with one compositional holdout | not built | roughly 2–4 focused weeks |
| credible grounded-vs-decoupled empirical paper | missing v2 learner, controls, and independent replications | roughly 1–3 months |
| full “simple human in a simple world” developmental system | far away; open research | many months to years |

These are engineering estimates, not guarantees. The current result should be
published as an interim result rather than held hostage to the full vision.
