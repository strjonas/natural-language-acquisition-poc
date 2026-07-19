# STATE

Last rewritten: 2026-07-19. Rewrite this file, never append.

## Where the frontier is

The project now has a replicated, unified organism with a causally used bodily
self-model. It does **not** yet have demonstrated language comprehension or
self-report. The active direction remains `DIRECTION_2026-07-12.md`; historical
probe logs stay frozen.

The MLX organism contains one recurrent perception/token core with policy,
value, world, bodily, reward, and caregiver-token heads. It learns online in the
island, retains a reservoir of real episodic segments, trains two-step latent
dynamics from replay, and uses a two-decision receding planner. Kind-blind
visible-object options bind contemplated actions to perceived object features,
never hidden kind. Counterfactual simulator branches are audit-only. Planning
removal and score reversal are standard harness conditions.

## Strongest result

Two full 200k-tick grounded runs (hidden 256, no adult offers) independently
retain accurate and causally useful bodily prediction:

- Seed 1 normal/removed/reversed adult survival: 3%/2%/0%; lifespan
  133.75/118.25/62.65; resources 29.54/25.78/11.44; within-state predicted vs
  actual action-score correlation 0.426; predicted-best advantage 0.039.
- Seed 2: 1%/0%/0%; lifespan 104.89/84.42/55.60; resources
  34.18/28.03/13.73; correlation 0.825; predicted-best advantage 0.103.
- Oracle survives 99%; random survives 0% and dies around tick 54.

This is the first organism result where a learned model of future bodily state
remains accurate after long training and behavior degrades in the predicted
direction when that self-model is removed or reversed. Replay and two-step
ablations show complementary effects: replay improves resource competence;
multi-step modeling improves counterfactual ranking.

The result is still far from G1: survival is sparse and mean viability
(0.696/0.683) is below the 0.74 target. It establishes an early causal
self-model, not a human-like self, authentic desire, reflection, self-report,
or consciousness.

## Negative results retained

- Token-masked behavior cloning, in-hand offers, and proximal-to-distal offers
  do not yield unsupported survival without sensorimotor options.
- The earlier on-policy one-step self-model collapsed under long training:
  0/50 survival and negative counterfactual ranking despite low MSE. This line
  is closed; no scale/loss tuning.
- Current language is not load-bearing. Acute grounded/silent/shuffled
  inference gives 3%/3%/4% survival and essentially matched lifespans.
  Separately trained grounded/silent/shuffled controls give 3%/0%/2% survival
  and 133.75/102.10/130.44 ticks. Grounded does not meaningfully beat shuffled.

## What is next

The next goal-level bottleneck is information rent, not model size:

1. Remove stable surface-kind shortcuts. Randomly assign consumable surface
   meanings per life while guaranteeing viable food/water ecology.
2. Add a kind-blind visible-object inspect/ask option alongside consume. It
   performs embodied joint attention and returns the caregiver's label; no
   language or question reward is added.
3. Preserve per-life recurrent memory so a label can change the predicted
   bodily consequence of consuming that surface later.
4. Recalibrate oracle/random and preregister grounded/silent/shuffled plus
   acute token interventions. Language evidence requires semantic tokens to
   improve survival and to causally alter self-model predictions/actions.
5. Only after comprehension (G2) add generated utterance actions and truthful
   self-report rent (G3), followed by autobiographical continuity.

## Currently forbidden / deferred

- No more BC, offer-distance, planner-scale, token-loss, or one-step-score
  tuning.
- No counterfactual simulator outcomes in training; audit only.
- No live LLM mind/training calls and no direct language reward.
- No appending to frozen mega-logs or new top-level one-off probe scripts.
- No GPU request yet: G1/G2 have not passed locally, and the next bottleneck is
  structural rather than compute-limited.
