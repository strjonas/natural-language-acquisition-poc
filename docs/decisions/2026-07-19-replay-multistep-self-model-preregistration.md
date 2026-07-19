# Replay-backed multi-step self-model preregistration

Date: 2026-07-19
Goal-level metrics: stable unsupported survival and causal bodily planning.

## Architectural hypothesis

The closed one-step line trains each experienced transition once, jointly with
an on-policy actor, and uses only an immediate consequence to bias policy
logits. The full probe's low absolute error but inverted action ranking is
consistent with rare-effect forgetting and closed-loop model exploitation.

The next learner will add two missing organism-level mechanisms:

1. **Episodic replay.** Retain real experienced segments across lives and make
   additional world/self-model updates from diverse past transitions. Replay
   never changes policy with stale off-policy advantages.
2. **Multi-step latent dynamics.** The action-conditioned transition state is
   trained through two consecutive experienced actions. At decision time,
   receding-horizon planning evaluates every available first action through a
   second imagined decision and chooses using predicted future bodily state.

Only actual chosen transitions enter either loss. Simulator copies remain
strictly diagnostic. Language still has no direct reward.

## First smoke

- Grounded, seed 1, hidden 64, 40k actual ticks.
- Offer childhood and planning transition at 20k; distance 0 -> 2.
- Kind-blind object-bound consume options.
- Replay capacity 256 segments, one auxiliary replay update per online update.
- Planning horizon 2, scale 6, held-out 20 x 400 adult evaluation, 200-state
  counterfactual audit, matched removed/reversed interventions.

## Gates and ablations

Promote only if normal planning beats reversed and within-state score
correlation plus predicted-best advantage are positive. Target the prior smoke
thresholds (>0.20 and >0.005), but report a miss rather than retuning.

If it passes, separately ablate replay and horizon 2 before a full run. If it
still exhibits confident counterfactual errors, the second probe in this line
adds uncertainty-aware pessimistic planning. Three failures close the
replay/MPC architecture, not merely a learning-rate setting.

## Seed-1 smoke result and locked ablations

The smoke passed its prediction thresholds: needs MAE 0.0233, within-state
score correlation 0.2929, predicted-best accuracy 46%, and predicted-best
actual advantage 0.0125. The original 20-life intervention was noisy. A
read-only expansion to 100 fixed lives gave normal/removed/reversed survival of
10%/8%/3%, mean lifespan 106.32/92.03/64.01, and resources
16.56/15.32/10.12.

Two matched 40k ablations are now fixed:

1. no replay, while retaining two-step latent loss and horizon-2 planning;
2. replay retained, but both latent loss and planning returned to horizon 1.

The architecture is promoted only if the integrated run is better than at
least one ablation on causal lifespan and no ablation dominates it across
survival, resources, and prediction ranking.

## Ablation result and replication rule

On 100 held-out adult lives, integrated replay+horizon-2 achieved 10% survival,
106.32 mean ticks, 16.56 resources, and 0.2929 score correlation. Removing
replay gave 7%, 103.77, 11.87, and 0.4119. Keeping replay but returning to
horizon 1 gave 9%, 113.28, 11.32, and 0.1690. Reversal reduced mean life to
64.01, 60.10, and 59.31 respectively. Neither ablation dominates the integrated
architecture: replay improves resource competence, while two-step learning
improves counterfactual ranking.

Before a full run, repeat the integrated 40k configuration at seed 2. Require
positive score correlation and predicted-best advantage, and normal planning
better than reversed in survival or mean lifespan. This rule is fixed before
seeing seed 2.

Seed 2 passed: 8% adult survival, 108.61 mean ticks, 0.4271 score correlation,
0.0123 predicted-best advantage, and normal/removed/reversed lifespans of
108.61/86.41/71.34. Promotion criteria are satisfied on two seeds.
