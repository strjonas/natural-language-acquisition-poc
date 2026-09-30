# What this agent actually is, and the route to a talking self

Date: 2026-08-18. Status: **audit and route.** Not a result. No gate is claimed
and nothing here is evidence about the organism; sections 1 and 2 are an audit of
the code as it stands on 2026-08-18, and sections 3 to 5 are a route derived from
results that already exist. Every number in section 1 was read out of the code,
not out of a document.

Written because the repository has an accurate record of *what was measured* and
no current record of *what the thing being measured is*, and because the two
questions a reviewer asks first -- "what is the architecture" and "why not put a
language model in as the caregiver" -- both have answers that follow from
measurements already in `docs/decisions/`.

## 1. The audit

### The shared core

One GRU. `src/homesocial/organism/model.py`.

    vision/interoception vector + heard tokens
        tokens: Embedding(60 -> 32) -> GRU(32)
        Linear -> LayerNorm -> GRU(256, 256) -> Linear -> LayerNorm
                              |
    policy (act) . value (critic) . next_vector / next_needs / drift_needs
    (predict world and body) . reward_head . next_tokens (predict what the
    caregiver will say) . report heads . episodic binding

Every capability hangs off one recurrent state. This is the "the self should not
be a module" idea from the original plan, and it is implemented.

### The self-model is not in that GRU

It is 658 scalars beside it, and the count is exact:

| parameter | count | meaning |
|---|---:|---|
| `causal_drift_raw` | 3 | per-need metabolic rate |
| `causal_move_extra_raw` | 1 | extra energy cost of moving |
| `causal_uptake_raw` | 27 | 9 surfaces x 3 needs |
| `causal_shock_raw` | 27 | 9 surfaces x 3 needs |
| `causal_listener_logits` | 600 | 60 vocab x (9 surfaces + 1) |
| **total** | **658** | |

`len(VOCAB) == 60` and `len(SURFACES) == 9`, both read from the code;
`3 + 1 + 27 + 27 + 600 = 658`, which is the number
`md/archive/DIRECTION_2026-07-26.md` section 1 already carries. The document and
the code agree.

One update rule, `model.causal_self_transition`:

    clip( current
          + duration * drift
          + moves * move_extra
          + surfaces @ uptake
          + shocks @ shock,
          0, 1 )

**The self-model is one clipped linear equation.** The three axes, the linearity,
and the sign of every effect were written by a person. Only the coefficients are
learned. Probe63 adds an online RLS calibration of those coefficients per
individual; it does not change the form.

### The mouth has no parameters

`future_request.need_scores` computes `E[min(next body)]` for each of the three
need words under the believed consequence of saying it, and the speaker takes the
argmax. `portion_request` adds a size word from a Beta-Binomial posterior over
which word the caregiver treats as "large".

**Total expressible messages: 3 needs x 2 sizes = 6.**

### The algorithm, in one line each

- **Core:** reinforcement learning plus multi-head prediction on the
  developmental task, from the frozen probe52 lexical parent.
- **Self-model:** Adam on a constrained causal template over public experience
  (probe57), plus per-individual online RLS (probe63).
- **Mouth:** no parameters. Arithmetic over the belief.
- **Language is never rewarded directly.** It pays only through consequence.

## 2. What this architecture cannot become, stated plainly

Three blockers. None is vague and none is a matter of scale.

1. **The mouth cannot generate.** It is an argmax over an enumerated set. There is
   no sense in which it could emit a sentence. Enlarging it produces a wider table
   of mostly-zero logits, which is what `DIRECTION` section 4 already forbids.
2. **The self-model cannot represent what its template lacks.** It is linear in
   three axes. Probe63 recorded the consequence in its own words: *a self-model
   can only localize a fact its own parameter set can express, and when it cannot
   it does not fail loudly -- it produces a confident wrong answer.* A self that
   can only be wrong in ways its designer anticipated is not discovering itself.
3. **The world affords six meanings.** A perfect speaker would have nothing else
   to say.

**None of this says the approach is wrong.** It says three components are
placeholders. What surrounds them -- causal grounding, evidence by intervention
rather than fluency, and the four ecology laws below -- is the part that is hard,
and it is the part that works. Probe69 removed the strongest apparent argument
against replacing the placeholders with learned components: the black-box family's
nine failures were measured at one budget, and the family ties the structured one
given compute.

## 3. Why a small LLM caregiver is right later and wrong now

The proposal is to put a small language model in as the caregiver, so the agent
has a competent speaker to learn from. The intuition is correct about human
development and it is correct about this project's eventual needs. It is wrong
about the current binding constraint, and probe68 says why with a number rather
than a preference:

> **The vocabulary an agent can acquire is bounded by the caregiver's action
> space, not by its lexicon.**

The caregiver can do six distinguishable things. Probe68 established that a word
changes an outcome only when its consequence exceeds
`margin = E_grant[min(grant * uptake, g)]`. So there are exactly six things worth
saying. A 0.8B model with a 50,000-token vocabulary attached to a body with six
consequences **will teach six words**; the other 49,994 have no gradient behind
them, because nothing in the world changes when the agent says them.

That is `DIRECTION` section 4 -- complexity is pulled by the task, never pushed by
the parameters -- with a measurement underneath it.

### When it becomes the right tool

- Past roughly 100 meanings, hand-authoring the caregiver *is* the bottleneck, and
  a language model is the practical way to write it.
- It offers one thing no script does, which expands the **consequence** space and
  not merely the surface: **repair and negotiation.** "Do you mean the water or
  the berries?" A clarification exchange is itself a consequence-bearing action,
  and the repository has no dialogue structure at all.

### The trap in that, which this repository has already named

If clarification is free, **the environment does the model's job**: the agent need
not know itself, it waits to be asked. That is the third trap in `CLAUDE.md`.
Clarification must therefore **cost ticks**, with the body draining while the
negotiation runs. Then self-knowledge still pays -- and pays *more* the worse the
agent is at expressing itself.

### The line, and the control that enforces it

- The model **may** generate the caregiver's *surface*: how a decision is phrased.
- The model **may not** decide what help is granted. That is the reward signal.
- The model **may not** judge whether the agent spoke correctly. That is the
  arbiter of correctness, and it would void every intervention-based claim in this
  repository -- a fluency machine grading a fluency claim. `VISION_AND_STATUS.md`
  states this independently.

**The control:** scramble the model's phrasing while holding the simulator's
grants bit-identical, and require survival not to move. If it moves, the model has
leaked into the consequence path and the experiment is contaminated. This makes
the proposal falsifiable rather than a matter of taste, and it is the same shape
as the scrambled-listener control that every probe since 57 has run.

## 4. The route

### The observation the route turns on

**Every report this organism has ever made is about its state** -- "I need water".
Probe65's future-tense report is a state report displaced in time. But *talking
about yourself* means reporting your **constitution**: "I am the kind of body that
burns water fast."

Probe63 gives the organism exactly that knowledge -- individual metabolic and
uptake constants, learned online, verified by intervention, provably different
from its species. **Nothing in the world can hear it.** `STATE.md` has said so
since 2026-08-03: *"Probe63's organism knows something no listener can hear."*

That is an **ecology** gap, and probes 63, 65 and 68 all found that ecology gaps
are the ones that move.

### The first experiment, and the objection that shaped it

**Naive form:** a caregiver configured once per life by a constitution report,
delivering a tailored ration thereafter.

**The objection that kills the naive form.** The organism *already* asks for what
it needs at every grant. A whole-life pre-configuration is a strictly coarser
version of a channel that already works, so it should lose. A design has to name
something the per-tick channel **structurally cannot do**, or it is measuring
nothing.

**First repair, which was itself wrong.** The obvious answer is *arrive on
time*: probe65 made the caregiver answer the request it heard eighteen ticks ago,
so a reactive caregiver eats the delay on every grant, and one that knows your
rates could schedule ahead. **That is also dead**, and the code says so:
`answer_horizon` returns the lag and `need_scores` projects the body forward by
it, so **probe65's speaker already anticipates the delay itself**. Telling the
caregiver your rates so that *it* can anticipate duplicates work the speaker
already does. A design has to name something neither the per-tick channel nor the
existing speaker can reach.

**What is actually out of reach: the caregiver's own store.** Probe64 gave the
caregiver a basket — `caregiver_store`, the total portion mass it has for the
whole life, spent per grant, with a grant it cannot afford **refused**. Two
properties make it unreachable from the per-tick channel:

- it is the caregiver's **private state** — nothing the organism perceives
  reports `_store_remaining`;
- its depletion is a **whole-life** quantity, and every existing message is
  scoped to one request.

So a caregiver told your rates can **budget its own basket across your life** —
hold portions back for when you will actually need them instead of spending early
on whatever was asked for first. No sequence of per-tick requests can produce that
allocation, because the allocation is a decision about a resource the speaker
cannot see and cannot address.

That inherits four measured results rather than hoping:

- Probe64 built the store and found **survival hears none of it** — which is what
  a per-request size word should do, since it cannot budget.
- Probe65 measured that the value of knowing *what you are* rather than *where you
  are* grows with the horizon (+0.1017 at lag 18, sign-reversing between lag 9 and
  12). A whole-life budget is the longest horizon in the system.
- Probe68 measured that an effect reaches a word only above the margin. A
  **refused** grant is the largest effect available in body units — the organism
  gets nothing — and `diagnose_probe68.py` can check it before anything is locked.
- Probe63 already recovers the rates, and its lesions already work.

And it **forces vocabulary the task pulls**: "I burn water fast" cannot be said
with the word "thirsty". It requires a predicate about the speaker -- a different
word class from anything in the current six. That is `DIRECTION` section 4's
prescription aimed at the self instead of the state.

It would move property 5 (reflexive -- reporting a fact about the model rather
than the state) and property 4 (productive -- saying something never trained),
which are the two that probe69 explicitly did not touch.

### The ceiling check that must come first

Per probe68's binding rule and probe57's: **before any gate is locked**, measure
whether a caregiver that budgets its store with *oracle* rates beats one that
spends it first-come at all — and at what store size, since at
`caregiver_store = 0` the basket is unlimited and there is nothing to allocate.
If the oracle budget does not beat first-come, the mechanism closes before it is
built. Two designs died that way inside probe68's survey without costing a run,
and two more died here, in this document, before reaching a survey.

### The order after that

1. **Constitution report** -- make self-description pay. Uses machinery that
   exists; needs no new architecture and no language model.
2. **Grow the consequence space** -- more bodily axes (STATE item 2, which also
   serves probe61's `K = 5` world, probe66's "more needs before more model", and
   probe68's unmoved axis-gap term), more portion levels, tense, multiple
   caregivers with different competences. Message space 6 -> ~100.
3. **Measure the composition law** at that size: where a lookup table stops
   covering and factorization starts paying. Derive it in closed form and check
   it, as probe68 did with the margin; **do not fit it** -- probe69's G6 is what
   fitting near the noise floor costs. That crossover is what *licenses* a
   generative mouth rather than assuming it.
4. **Then** the generative mouth, and **then** the language-model caregiver, at a
   consequence space rich enough that both are pulled rather than pushed.

## 5. What this document is not

- **Not evidence.** Sections 3 to 5 are derivations from existing results, and a
  derivation has been overturned by measurement in this repository at least four
  times -- probe68's shift-equivariance clause, probe69's G6, and the two probe68
  designs killed by their own surveys. Each step above needs its own survey,
  preregistration and gates.
- **Not a claim that the route is short.** Step 2 forces retraining the frozen
  probe52 parent, which nothing since probe57 has had to do.
- **Not a revision of any closed line.** Nothing here reopens a mechanism.
- **Section 1 is an audit and will drift.** It is true of the code on 2026-08-18.
  Check it against the code before relying on it, which is the fifth trap in
  `CLAUDE.md`.
