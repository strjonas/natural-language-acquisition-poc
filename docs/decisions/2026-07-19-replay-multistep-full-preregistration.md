# Replay-backed multi-step full-probe preregistration

Date: 2026-07-19
Goal-level metrics: unsupported adult survival and stable causal self-model use.

## Configuration

- Grounded, seed 1, hidden 256, 200k actual environment ticks.
- Offer childhood threshold 0.75 and distance 0 -> 2, ending at 100k.
- Object-bound residual bodily model, two-step latent rollout loss weight 1.
- Reservoir replay of 256 real segments; one auxiliary-only replay update per
  online update.
- Two-decision receding planning, scale 6, activated at 100k.
- Evaluation: 100 held-out adult lives x 1,000 ticks; 500 audited states;
  normal, planning-removed, and score-reversed conditions.

## Gates

Pass only if:

- at least one unsupported adult life reaches 1,000 ticks;
- normal planning beats reversed in survival, or in mean lifespan without a
  survival regression;
- within-state one-step score correlation and predicted-best actual advantage
  remain positive after the long run; and
- both food and water are consumed nontrivially.

The comparison target is probe 11, which collapsed to 0/50 survival and
negative ranking after 200k ticks. If this passes, replicate the full grounded
run before spending full runs on silent/shuffled language controls. If it fails,
the next replay/MPC probe adds uncertainty-aware pessimism rather than tuning
replay capacity, scale, or loss weights.

## Seed-1 result

The probe passed all gates. Oracle/random survival was 99%/0%. The organism's
normal/removed/reversed survival was 3%/2%/0%, mean lifespan
133.75/118.25/62.65, and successful resources 29.54/25.78/11.44. It consumed
both food (15.61) and water (13.93).

Unlike probe 11, counterfactual quality survived the long run: needs MAE
0.0244, within-state score correlation 0.4263, predicted-best accuracy 55.6%,
and predicted-best actual advantage 0.0389 across 5,335 outcomes. This is the
first full-scale result in the organism line where accurate bodily prediction
is causally load-bearing under removal and reversal.

The required next action is the same full configuration at seed 2. Do not begin
silent/shuffled language training unless seed 2 retains nonzero adult survival,
positive score correlation/advantage, and normal behavior better than reversed.

## Seed-2 replication result

Seed 2 passed the replication rule. Normal/removed/reversed survival was
1%/0%/0%, mean lifespan 104.89/84.42/55.60, and resources
34.18/28.03/13.73. Counterfactual score correlation was 0.8249 and
predicted-best advantage 0.1030. Accurate, causally used bodily prediction is
therefore replicated across two full 200k runs.

This does not yet pass G1: mean viability is 0.696/0.683 across the two seeds,
below 0.74, and language necessity has not been measured. Proceed to the fixed
grounded/silent/shuffled comparison without weakening those statements.
