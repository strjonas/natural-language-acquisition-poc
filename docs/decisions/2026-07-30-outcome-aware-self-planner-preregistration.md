# Outcome-aware causal self-planner preregistration

Date: 2026-07-30
Status: locked before implementation or treatment runs
Probe: 60 / Phase A1b

## Why this experiment exists

Probe59 established a real continual-learning result on the belief side and a
failure on the behavioural side. After the body's metabolism and portions
changed, the causal self-model reduced body error from 0.0879 to 0.0164 and
moved every learned constant toward the new truth on 5/5 seeds, but survival
remained 0.500. With a perfect body belief, the adapted model's planner still
survived only 0.504.

The failure is localized to `causal_social_token`. It first averages the
listener's possible help outcomes and then applies the nonlinear homeostatic
utility:

```
score_old(word) = min(E[next_body | word])
```

That is not the expected utility of saying the word. A diffuse listener
distribution can appear to raise every bodily axis even though each realized
branch grants only one resource. The error is largest when the two lowest
needs are close, exactly where probe59 measured need-word use falling to 42.2%
after adaptation.

This probe tests one mechanism and no variants:

```
score_new(word) = E[min(next_body) | word]
```

The expectation is over every learned listener outcome: each visible help
surface plus no help. Each branch is advanced with the same learned depletion
and uptake model already used for belief tracking. This is the smallest
decision-theoretically coherent repair: preserve the learned consequence
distribution until after applying viability utility.

The motivation is consistent with homeostatic reinforcement learning, which
defines value through movement toward physiological stability, and with the
distributional perspective on reinforcement learning, which warns that a
random outcome distribution contains decision-relevant information erased by
its expectation:

- Keramati & Gutkin (2014), *Homeostatic reinforcement learning for
  integrating reward collection and physiological stability*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4270100/
- Bellemare, Dabney & Munos (2017), *A Distributional Perspective on
  Reinforcement Learning*: https://arxiv.org/abs/1707.06887
- Agarwal, Kakade & Yang (2020), *Model-Based Reinforcement Learning with a
  Generative Model is Minimax Optimal*:
  https://proceedings.mlr.press/v125/agarwal20b.html

These papers motivate the class of computation; they do not establish this
repository's claim. Only the interventions below can do that.

## Locked mechanism

For all 60 vocabulary tokens:

1. infer `P(listener_outcome | token)` from the existing learned listener
   logits;
2. predict the bodily state immediately before the next help boundary using
   the existing learned drift;
3. create one realized branch for every help surface and one no-help branch;
4. apply that surface's existing learned uptake to its branch;
5. clip each branch to the body's public `[0, 1]` bounds;
6. score the token by listener-probability-weighted mean of the branch's
   minimum bodily axis; and
7. choose the highest-scoring token with the existing deterministic token-ID
   tie break.

The planner may read only the persistent causal belief, public step count,
help period, learned drift, learned surface uptake, and learned listener
distribution. It may not read true needs, need labels, `HELP_SURFACES`,
`NEED_TO_REPORT_WORD`, hidden object kinds/events, future randomness, or audit
metadata when choosing a token.

The legacy max-min-of-mean planner remains unchanged as the primary lesion.
No planner parameter is trained. No reward, report label, or correct word is
introduced.

## Fixed data and protocol

- Parent checkpoint:
  `runs/organism/probe57_structured_causal_self/treatment/organism_causal_self_seed1.npz`
- Body shift: probe59's frozen shift, metabolism and portions both multiplied
  by 1.5; all other factors unchanged.
- Adaptation: exactly 40,000 primitive ticks at learning rate `3e-3`, using
  probe59 seed bases `82,100,000 + 10,000,000 * seed_index`.
- Replication: five adaptation seeds, indices 0 through 4, from the start.
- Evaluation: 100 lives per condition and seed, paired world seeds beginning
  at 7,100,000.
- All pre-existing model parameters remain bit-identical. Only the already
  declared causal parameters may adapt.
- The new planner lives in its own module. No `OrganismConfig` knob is added.

## Factorial conditions

The harness must keep belief source, planning model, objective, and real
listener separately named.

1. **adapted/outcome-aware**: adapted causal belief, adapted causal parameters
   for planning, new objective, grounded listener;
2. **adapted/legacy**: same adapted belief and parameters, old objective;
3. **frozen/outcome-aware**: developed-but-frozen belief and planning model,
   new objective;
4. **stale-belief/adapted-planner**: probe53 analytic belief with pre-shift
   constants, adapted planning model, new objective;
5. **oracle-belief/adapted-planner**: analytic belief with true shifted
   constants, adapted planning model, new objective;
6. **adapted-belief/frozen-planner**: adapted belief, frozen planning model,
   new objective;
7. **scrambled listener**: condition 1's internal computation, but the real
   listener replaces every utterance with a random need word;
8. **zero belief**: condition 1 with the planner belief clamped to zero;
9. **privileged truthful speaker**: says the world's true lowest-need word;
   feasibility ceiling only, never evidence for learning.

Conditions 4--6 are the contamination audit. They prevent a better belief, a
better planning model, and an objective change from being silently credited to
one another.

## Locked feasibility and promotion gates

All aggregate differences below are paired across the fixed evaluation lives.
No threshold is changed after any treatment result is observed.

### F0 -- ecology ceiling

The privileged truthful speaker must survive at least 0.90 in the shifted
ecology. If it does not, behavioural gates are invalid and the experiment
stops without retuning the world.

### G1 -- defect-specific unit intervention

In the synthetic two-deficit case pinned by probe59, the legacy planner must
choose the deliberately diffuse token and the outcome-aware planner must
choose the token whose learned listener outcome deterministically repairs the
lowest axis. With a clear single deficit both must select the targeted token.

### G2 -- behavioural repair

Across five seeds, adapted/outcome-aware must:

- reach mean survival `>= 0.80` and survival `>= 0.75` on 5/5 seeds;
- exceed adapted/legacy mean survival by at least 15 percentage points; and
- retain mean report fidelity `>= 0.90`.

The 0.80 survival threshold is below the already measured 0.840 perfect-belief
ceiling under the *broken* planner and below F0's required ceiling. It is not a
claim of optimality.

### G3 -- continual self-model becomes behaviourally load-bearing

Adapted/outcome-aware must exceed both frozen/outcome-aware and
stale-belief/adapted-planner mean survival by at least 5 percentage points.
Independently, its mean body error must be `<= 0.03`, and learned-constant error
must move toward the shifted truth on 5/5 seeds.

This gate is what probe59 could not pass. If the belief endpoint passes but
either survival contrast fails, the exact one-step planner is useful but does
not establish that continual self-model improvement controls behaviour.

### G4 -- causal necessity

Relative to adapted/outcome-aware, both the real scrambled listener and the
zero-belief lesion must reduce mean survival by at least 30 percentage points.
The internal listener model is not scrambled: the mismatch between predicted
and actual social consequence is the intervention.

### G5 -- shared-model ceiling and persistence

- oracle-belief/adapted-planner survival must be no lower than full
  adapted/outcome-aware survival (within one paired life, 0.01 tolerance);
- adapted-belief/frozen-planner is reported separately and not folded into any
  gate;
- every non-causal parent tensor remains bit-identical through adaptation; and
- the probe52 lexical intervention gate remains intact on the adapted models.

## Failure rule and claim boundary

All G1--G5 gates must pass. A failure closes this exact one-step
`E[min(next_body)]` mechanism. It is not rescued by trying soft-min
temperatures, hand-weighted deficits, special-casing tied needs, restricting
the candidate vocabulary, or changing the ecology. Any such proposal requires
a new preregistration with its own lesion.

A pass would show that one learned causal self-model can be corrected online
and that its improved beliefs and social-consequence predictions jointly
control grounded self-report and survival. It would not show discovered latent
structure, evidence-based belief correction, calibrated uncertainty,
multi-step reflection, sentience, consciousness, or natural-language
understanding.

No external corpus, generated data, vocabulary growth, or larger compute is
authorized by this probe.
