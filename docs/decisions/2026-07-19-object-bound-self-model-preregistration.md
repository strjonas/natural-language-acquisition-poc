# Object-bound bodily self-model preregistration

Date: 2026-07-19
Goal-level metrics: counterfactual bodily ranking and causal planning.

## Hypothesis

The transition model currently receives a different one-hot action for every
visible-object slot, but no explicit representation of the object selected by
that action. It must rediscover the relationship between action slot 0 and
visual slot 0 independently from slot 1, and so on, from sparse chosen actions.
Residual prediction reduced absolute error without learning this binding.

## Intervention

Factor consume-option consequences into:

- one shared `consume perceived object` action identity for every slot; and
- the raw learner-visible feature vector of the selected object (presence,
  egocentric displacement, and surface identity).

Primitive actions retain separate identities. Hidden environmental kind is not
included. The object feature is already in the organism's observation; this
change binds its contemplated act to its percept rather than adding privileged
information. All consequence targets still come only from experienced chosen
actions, and counterfactual simulator copies remain audit-only.

## Smoke gate

Use the residual probe's 40k-tick schedule and its fixed held-out seeds. Require:

- within-state action-score correlation above 0.20;
- predicted-best over predicted-worst actual advantage above 0.005;
- normal planning better than reversed planning on lifespan or balanced
  resource use; and
- no regression above 0.06 counterfactual needs MAE.

Passing the prediction gate permits a longer behavioral probe, but is not by
itself evidence of unsupported survival, language necessity, self-report, or
consciousness.

## Seed-1 result and replication rule

The matched 40k smoke passed: counterfactual needs MAE 0.0314, within-state
score correlation 0.3725, predicted-best accuracy 48.5%, and predicted-best
actual advantage 0.0173. Normal/removed/reversed planning yielded respectively
111.35/95.15/79.20 mean adult ticks on identical seeds; reversed planning also
halved 400-tick survival from 10% to 5%.

Before the full run, repeat the fixed smoke on seeds 2 and 3. Promote if at
least one additional seed passes both ranking thresholds and normal planning
beats reversed planning in mean lifespan or successful resource use. Report all
seeds; do not select only the best replication.

Seed 2 passed both ranking thresholds (correlation 0.2276, advantage 0.0131)
and normal planning beat reversed in lifespan (75.4 vs 67.2) and resources
(12.1 vs 10.8). Seed 3 failed the ranking thresholds (0.0572, -0.0012), while
normal planning still beat reversed in lifespan (112.4 vs 65.7), survival
(10% vs 0%), and resources (11.9 vs 6.8). Thus two of three seeds pass the
prediction gate and all three show behavior in the predicted intervention
direction. Promotion is warranted, with robustness explicitly unresolved.
