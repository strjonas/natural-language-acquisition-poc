# Homeostatic Social Grid

This is the first prototype for a minimal embodied/social learning environment.

The goal is not to build a full agent yet. The goal is to define the smallest
closed loop where language can become useful because it helps an agent regulate
its own viability.

## Core Idea

The environment contains:

- a partially observed grid world,
- resources such as water, food, shelter, danger, trees, and rocks,
- internal homeostatic variables: food, water, energy, and safety,
- actions such as moving, pointing, asking, consuming, and resting,
- a deterministic teacher that speaks only in response to situated agent action.

This gives us the first testbed for comparing:

1. no social language,
2. grounded in-loop teacher language,
3. later: decoupled/objective language injected outside the loop.

## Run

```bash
python3 -m homesocial.demo
```

If running from a fresh checkout without installing the package:

```bash
PYTHONPATH=src python3 -m homesocial.demo
```

Run the first teacher/no-teacher Q-learning comparison:

```bash
PYTHONPATH=src python3 -m homesocial.experiment --episodes 500 --eval-episodes 50
```

Run the recurrent actor-critic baseline:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 500 --batch-size 16 --ppo-epochs 3 --eval-episodes 20 --hidden-size 128 --log-every 50
```

Run a cheaper fixed-world diagnostic before spending time on randomized hidden
object-kind sweeps:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 120 --batch-size 8 --ppo-epochs 2 --eval-episodes 10 --hidden-size 64 --max-steps 80 --fixed-world --conditions grounded_teacher silent_teacher --log-every 40
```

Run the stricter language-necessity sanity check before larger PPO sweeps. In
this mode object positions do not identify object kind, visible objects share
the same learner-facing name, and unsafe guessing is costly:

```bash
PYTHONPATH=src python3 -m homesocial.experiment --episodes 0 --eval-episodes 20 --fixed-world --diagnostic-mode language_necessary --scripted-probe --conditions grounded_teacher silent_teacher
```

Run a tiny recurrent smoke test against the same diagnostic mode:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 16 --batch-size 4 --ppo-epochs 1 --eval-episodes 2 --hidden-size 32 --max-steps 40 --diagnostic-mode language_necessary --conditions grounded_teacher silent_teacher --log-every 8
```

Run a behavior-cloning warmstart from closed-loop teacher-following episodes:

```bash
PYTHONPATH=src python3 -m homesocial.imitation --expert-episodes 200 --epochs 5 --batch-size 32 --eval-episodes 20 --hidden-size 128 --fixed-world --diagnostic-mode language_necessary --checkpoint runs/bc_homegrid.weights.npz
```

Run the first self-battery consequence probe on a trained checkpoint:

```bash
PYTHONPATH=src python3 -m homesocial.self_battery --checkpoint runs/bc_homegrid.weights.npz --rollout-policy model --teacher-modes grounded masked shuffled wrong
```

Train counterfactual branch consequences for the self-battery:

```bash
PYTHONPATH=src python3 -m homesocial.counterfactual --checkpoint runs/bc_homegrid.weights.npz --output-checkpoint runs/cf_homegrid.weights.npz --episodes 400 --max-decisions 1200 --epochs 6 --batch-size 128 --rollout-policy teacher --branch-selection probe --rank-weight 1.0
```

Train and evaluate the structured self-report head:

```bash
PYTHONPATH=src python3 -m homesocial.report_head --checkpoint runs/cf_homegrid.weights.npz --eval-teacher-modes grounded masked shuffled wrong
```

The default report input is restricted to recurrent state plus the model's own
action-conditioned need/reward predictions. Run the anti-shortcut calibration
suite before interpreting report accuracy:

```bash
PYTHONPATH=src python3 -m homesocial.report_calibration --checkpoint runs/cf_homegrid.weights.npz
```

Compare against an untrained recurrent/consequence representation with:

```bash
PYTHONPATH=src python3 -m homesocial.report_head --checkpoint runs/cf_homegrid.weights.npz --random-model-control
```

Test whether truthful need reports are causally useful to a teacher that cannot
inspect the agent's needs:

```bash
PYTHONPATH=src python3 -m homesocial.report_mediation --checkpoint runs/cf_homegrid.weights.npz
```

This triage gate validates causal communication, not learned-model necessity.
Use `--random-model-control` to expose the current direct-interoception
shortcut.

Train with exact need values removed from the observation while retaining
next-need supervision:

```bash
PYTHONPATH=src python3 -m homesocial.imitation --expert-episodes 500 --epochs 10 --hidden-size 96 --next-needs-weight 5 --diagnostic-mode language_necessary --interoception-mode masked --checkpoint runs/bc_hidden_interoception.weights.npz
```

Evaluate temporal self-state inference and report-mediated action:

```bash
PYTHONPATH=src python3 -m homesocial.interoception --checkpoint runs/bc_hidden_interoception.weights.npz --random-model-control
PYTHONPATH=src python3 -m homesocial.hidden_mediation --checkpoint runs/bc_hidden_interoception.weights.npz --random-model-control
```

Train sequence-dependent hidden body dynamics:

```bash
PYTHONPATH=src python3 -m homesocial.imitation --expert-episodes 800 --epochs 12 --hidden-size 128 --next-needs-weight 6 --diagnostic-mode language_necessary --interoception-mode masked --body-dynamics-mode stochastic --checkpoint runs/bc_stochastic_body.weights.npz
```

Evaluate ordered-history inference and compositional reports:

```bash
PYTHONPATH=src python3 -m homesocial.interoception --checkpoint runs/bc_stochastic_body.weights.npz --history-modes full latest shuffled reversed --random-model-control
PYTHONPATH=src python3 -m homesocial.compositional_report --checkpoint runs/bc_stochastic_body.weights.npz --random-model-control
```

Train a two-slot discrete message protocol only through receiver decisions:

```bash
PYTHONPATH=src python3 -m homesocial.emergent_language --checkpoint runs/bc_stochastic_body.weights.npz --feature-mode self_estimate --random-model-control
```

Test whether the learned convention transfers to new receivers or independent
protocol initializations:

```bash
PYTHONPATH=src python3 -m homesocial.convention_transfer --checkpoint runs/bc_stochastic_body.weights.npz --feature-mode self_estimate
```

Train a population of senders against a shared receiver to test whether symbols
become interoperable rather than private:

```bash
PYTHONPATH=src python3 -m homesocial.population_language --checkpoint runs/bc_stochastic_body.weights.npz --agreement-weights 0 0.05 0.2
```

Train a harder self-request protocol where the receiver must choose the aid
category from the message alone:

```bash
PYTHONPATH=src python3 -m homesocial.self_request_language --checkpoint runs/bc_stochastic_body.weights.npz --agreement-weights 0 0.05 0.2
```

Train a richer self-state protocol where the receiver reconstructs continuous
needs, low-need flags, dominant need, severity, and trend from the message
alone:

```bash
PYTHONPATH=src python3 -m homesocial.self_state_language --checkpoint runs/bc_stochastic_body.weights.npz --agreement-weights 0 0.05 --history-modes full latest --random-model-control
```

For a stricter temporal-reflection audit, balance the dataset across worsening,
steady, and improving states and expose only learned self-estimate deltas:

```bash
PYTHONPATH=src python3 -m homesocial.self_state_language --checkpoint runs/bc_stochastic_body.weights.npz --feature-mode self_estimate_delta --balance-target trend --agreement-weights 0.05 --history-modes full latest --random-model-control
```

Intervene on the learned self-estimate delta channel while holding current
self-state fixed:

```bash
PYTHONPATH=src python3 -m homesocial.temporal_counterfactual --checkpoint runs/bc_stochastic_body.weights.npz --random-model-control
```

Train and test trend communication on counterfactual action branches from the
same current state:

```bash
PYTHONPATH=src python3 -m homesocial.counterfactual_trend_language --checkpoint runs/bc_stochastic_body.weights.npz --random-model-control
```

Run the stricter action-balanced branch audit, then a short option-level
counterfactual audit where branches execute seek/rest/wait policies for several
steps:

```bash
PYTHONPATH=src python3 -m homesocial.counterfactual_trend_language --checkpoint runs/bc_stochastic_body.weights.npz --balance-target action_trend --random-model-control
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/bc_stochastic_body.weights.npz --horizon 6 --random-model-control
```

Train the transition head directly on multi-step option branches, then test
latent option-state communication from the updated world model:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world.weights.npz --horizon 6
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world.weights.npz --horizon 6 --rollout-mode latent --random-model-control
```

Use broader exploratory state collection and an explicit current/future/delta
self-estimate feature when testing trend language:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world_random.weights.npz --horizon 6 --state-policy random
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world_random.weights.npz --horizon 6 --state-policy random --rollout-mode latent_current --trend-weight 2.0 --random-model-control
```

Run the same probe with renewable resources to test whether repeated food/water
encounters improve temporal self-report coverage:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world_renewable.weights.npz --horizon 6 --state-policy random --renewable-resources
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world_renewable.weights.npz --horizon 6 --state-policy random --rollout-mode latent_current --trend-weight 2.0 --random-model-control
```

Use richer resource ecology when probing less rest-dominated option reports:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --rollout-mode latent_current --trend-weight 2.0 --random-model-control
```

Audit whether the option self-trend message depends on current/future/delta
self-estimate features rather than only option/time regularities:

```bash
PYTHONPATH=src python3 -m homesocial.option_feature_intervention --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --trend-weight 2.0 --balance-target option_trend --random-model-control
```

Replicate the rich-ecology option-world plus feature-intervention pipeline
across seeds, including random-model and option-majority controls:

```bash
PYTHONPATH=src python3 -m homesocial.option_seed_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9901 9902 9903 --horizon 6 --state-policy cycle --resource-ecology rich --trend-weight 2.0 --balance-target option_trend --random-model-control
```

Test whether compact option self-change messages can mediate a useful option
choice, rather than only reconstruct labels. The command reuses the same
grouped option states for trained and random controls:

```bash
PYTHONPATH=src python3 -m homesocial.option_mediation --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --random-model-control
```

Force the mediation receiver to see only predicted future-current self-change
instead of current and future absolute self-estimates:

```bash
PYTHONPATH=src python3 -m homesocial.option_mediation --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --random-model-control
```

Add a direct self-model rank control that chooses by predicted future self-state
without training a message receiver:

```bash
PYTHONPATH=src python3 -m homesocial.option_mediation --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --random-model-control --self-model-rank-control
```

Stress option identity shortcuts by adding branch action noise while keeping the
actual sampled action sequence as both the true outcome and the model input:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9921 9922 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.3 --random-model-control
```

Option-world and mediation source noise can be split for curriculum probes:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9961 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --world-option-action-noise 0.3 --mediation-option-action-noise 0.15 --random-model-control
```

Fine-tune the self/world model directly on grouped branch ranking before
training the message receiver:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9981 9982 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --rank-finetune-epochs 4 --self-model-rank-control --random-model-control
```

Add dynamics replay during rank fine-tuning to preserve ordinary option-world
prediction quality:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9991 9992 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Train the message receiver to communicate the model's own predicted best
self-future instead of the oracle branch label:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10001 10002 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Vary the discrete message capacity for compression probes:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10011 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --message-slots 3 --message-vocabulary 4 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Train with soft message probabilities but still evaluate hard discrete messages:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10021 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --soft-message-training --message-temperature 0.8 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Add an auxiliary hard-message loss that reconstructs predicted self-model
option scores from the emitted symbols:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10031 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-reconstruction-weight 0.1 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Pretrain the hard discrete message code to reconstruct the self-model's option
scores before normal choice training:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Run paired no-pretrain controls on the same seeds before interpreting a
score-pretraining gain:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10041 10042 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Add pairwise self-score rank pressure to the hard-message score head:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --score-rank-weight 0.05 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Distill the receiver's hard-message choice logits toward the self-model score
distribution:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --score-distillation-weight 0.2 --score-distillation-temperature 0.7 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Warm up the receiver on a fixed hard message code after score pretraining:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --frozen-receiver-epochs 5 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Add light sender commitment pressure so each slot becomes more confident in its
current hard symbol while balance pressure keeps code usage spread out:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --message-commitment-weight 0.005 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Apply commitment only during score pretraining, then fine-tune choice without
extra commitment:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --score-pretrain-commitment-weight 0.005 --message-commitment-weight 0.0 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Mediation CSVs include `message_codes_used`, `message_code_entropy`,
`dominant_message_code_fraction`, `target_code_mutual_information`,
`choice_code_mutual_information`, `message_patterns_used`,
`reused_message_pattern_fraction`, and row-pattern mutual information fields to
distinguish code collapse, weak sender/receiver semantics, and one-off pattern
memorization.

`--message-replay-weight` snapshots score-pretrained hard message tokens and
penalizes later sender drift during choice training.
`--heldout-receiver-epochs` trains a fresh receiver against a frozen sender to
test whether the compact self-code transfers beyond the co-trained receiver.
`--receiver-copies` trains multiple co-receivers against the same sender as a
convention-pressure probe.
`--score-rank-code-weight` pushes one message slot to encode self-model rank
buckets as a reusable ordinal-symbol probe.

Replicate the strict self-model-targeted commitment setting across independent
base body checkpoints:

```bash
PYTHONPATH=src python3 -m homesocial.option_body_mediation_replication --checkpoints runs/bc_m8_stochastic_persistent_seed12.weights.npz runs/bc_m8_stochastic_persistent_seed13.weights.npz --seeds 10112 10113 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --message-commitment-weight 0.005 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Train a fresh option-world model per seed, then test whether option mediation
still works on those independently trained world-model heads:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9901 9902 --horizon 6 --state-policy cycle --resource-ecology rich --random-model-control
```

Run the same option-world plus mediation pipeline across multiple independently
trained base body checkpoints:

```bash
PYTHONPATH=src python3 -m homesocial.option_body_mediation_replication --checkpoints runs/bc_m8_stochastic_persistent_seed12.weights.npz runs/bc_m8_stochastic_persistent_seed13.weights.npz --seeds 9912 9913 --horizon 6 --state-policy cycle --resource-ecology rich --random-model-control
```

Run a larger local sweep once the fixed-world diagnostic is healthy:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 2000 --batch-size 32 --ppo-epochs 3 --eval-episodes 50 --hidden-size 256 --max-steps 160 --conditions grounded_teacher silent_teacher --log-every 100
```

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
