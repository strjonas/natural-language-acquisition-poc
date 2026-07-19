# STATE

Last rewritten: 2026-07-19. Rewrite this file, never append.

## Where the frontier is

The active direction remains `DIRECTION_2026-07-12.md`: build one online
organism whose persistent state supports control, self-prediction, and later
language. The historical probe era stays frozen.

The unified MLX organism now has:

- one recurrent perception/language core with policy, value, world, bodily,
  reward, and caregiver-token heads;
- an adult island where hidden kinds are never observed, oracle survival is
  98% on the current 50-life gate, and random survival is 0%;
- in-loop caregiver resource offers that fade to zero, including a
  proximal-to-distal childhood;
- kind-blind visible-object consume options with semi-Markov duration-aware
  GAE and strict invalid-slot masking;
- a residual bodily model that predicts need changes from sensed current needs;
- object-bound option prediction: every slot shares one abstract consume act
  applied to the selected learner-visible object feature, never hidden kind;
- counterfactual simulator-copy audits used only for evaluation; and
- matched self-model removal/reversal interventions in the one canonical
  harness.

All 178 tests pass. Headline runs are under `runs/organism/probe4` through
`probe11`; concise preregistrations/results are in `docs/decisions/`.

## What the experiments established

Childhood bootstrap alone is insufficient:

- token-masked oracle behavior cloning reached high imitation/consume recall
  but zero unsupported survival;
- in-hand offers produced long supported lives but collapsed after fade;
- moving offers from distance 0 to 2 also collapsed after fade.

Motor abstraction was necessary and useful. Kind-blind visible-slot options
raised successful adult resource interactions by more than an order of
magnitude. They do not expose bodily kind, so language necessity remains
testable.

The self-model result is promising at smoke scale but not robust:

- the original absolute one-step head had counterfactual score correlation
  -0.055; reversing its advice did not hurt;
- residual prediction improved next-needs MAE from 0.137 to about 0.040 but did
  not fix action ranking;
- object binding passed the 40k smoke gate on seeds 1 and 2 (correlations 0.373
  and 0.228) and all three seeds behaved worse when self-model advice was
  reversed;
- the 200k/256-hidden full probe failed: 0/50 unsupported 1,000-tick survivals.
  It achieved 111.8 mean ticks and 30.4 resources versus random's 54.5 and
  0.42. Normal/removed/reversed planning produced 111.8/100.5/75.9 ticks, but
  the 500-state audit collapsed to -0.050 correlation and -0.0095 actual
  predicted-best advantage.

Verdict: the organism learned substantial resource-seeking behavior and its
prediction signal causally gates option use, but the long-run one-step head is
not an accurate counterfactual self-model. G1 is not passed. There is no basis
yet for a language, self-report, authentic-desire, or consciousness claim.

## What is next

The one-step on-policy auxiliary planner line has reached its three-probe
timebox. The negative result propagates upward to the training architecture.

Next build a real model-based learner rather than another policy-logit bias:

1. Add persistent, diverse episodic replay for bodily/world transitions so
   rare consequences are retained rather than learned once and forgotten.
2. Learn multi-step latent dynamics and use short-horizon receding planning;
   action selection must depend directly on predicted future bodily state.
3. Represent epistemic uncertainty (ensemble or equivalent) so planning is
   pessimistic about unsupported counterfactuals instead of exploiting model
   errors; information gain about body dynamics may drive exploration, never
   language reward.
4. Keep the same held-out survival, counterfactual ranking, removal/reversal,
   and grounded/silent/shuffled gates in `organism.harness`.

Only after grounded adult survival and causal self-model use are robust should
we spend runs on language comprehension (G2), then generated self-report (G3).

## Currently forbidden / deferred

- No more BC epoch, offer-distance, planner-scale, or one-step-score tuning.
- No counterfactual simulator outcomes in training; branches are audit-only.
- No appending to frozen mega-logs or new top-level one-off probe scripts.
- No live LLM mind/training calls and no direct reward for language.
- No GPU request yet: G1 and G2 have not passed locally.
