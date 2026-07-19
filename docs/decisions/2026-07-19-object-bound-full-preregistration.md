# Object-bound self-model full-probe preregistration

Date: 2026-07-19
Goal-level metrics: unsupported adult survival and causal self-model use.

## Configuration

- Grounded language, seed 1, hidden size 256, 200k actual environment ticks.
- Caregiver offer threshold 0.75 and distance 0 -> 2, both ending at 100k.
- Kind-blind visible-slot consume options with object-bound residual bodily
  prediction.
- Planning scale 6 and reward weight 0.5, activated when childhood support ends
  at 100k.
- Evaluation: 50 fixed held-out adult lives x 1,000 ticks, no offers.
- Counterfactual audit: 500 held-out states, simulator copies used only for
  measurement.
- Automatic matched evaluations with self-model planning removed and reversed.

## Gates

The grounded probe passes only if:

- at least one of 50 unsupported lives reaches 1,000 ticks;
- normal planning exceeds reversed planning in survival, or in mean lifespan
  with no survival regression;
- within-state predicted/actual action-score correlation is positive and
  predicted-best actual advantage is positive; and
- successful food and water consumption are both nontrivial.

If it passes, replicate and train silent/shuffled controls. If it fails, do not
spend three times the compute on language controls. Nothing in this probe tests
self-report or warrants a consciousness claim.

## Result

The probe failed. The oracle survived 98% and random 0%, confirming the ecology
gate. The organism survived 0/50 adult lives. It nevertheless learned a strong
but unstable resource policy: 111.8 mean ticks and 30.4 successful resources
per life versus random's 54.5 and 0.42. Normal/removed/reversed planning yielded
111.8/100.5/75.9 mean ticks and 30.4/25.8/18.9 resources.

The larger model's counterfactual audit failed despite low absolute error:
needs MAE 0.0191, score correlation -0.0496, predicted-best accuracy 3.2%, and
predicted-best actual advantage -0.0095 across 5,494 candidate outcomes. The
planner had become a coarse option-use bias that the policy co-adapted to, not
an accurate action-by-action bodily model. This closes the one-step
on-policy auxiliary planner line after three negative probes. The next change
must be at the training/architecture level, not another score or scale tweak.
