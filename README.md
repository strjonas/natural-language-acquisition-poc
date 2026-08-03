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

## Unified organism self-report

The current frontier is the report island: interoception is visible once at
birth and then masked, and a caregiver who cannot inspect the body grants help
only for generated full-vocabulary reports. Start with the exact-dynamics
feasibility seal:

```bash
PYTHONPATH=src python3 -m homesocial.island.report_calibrate --lives 400 --rhythm-lives 100 --rhythm-max-length 4 --unified-uptake
```

Reproduce probe52's need-independent guided lexical childhood and stop at its
frozen comprehension gate:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --report-unified-uptake --report-lexical-childhood-steps 60000 --report-lexical-guided-labels --report-lexical-gate-only --train-steps 200000 --segment-length 64 --hidden-size 256 --report-slots 2 --report-entropy-weight 0.02 --report-audit-lives 200 --seed 1 --run-dir runs/organism/probe52_guided_report_lexicon/gate --log-every-lives 100
```

Continue the exact gate-passed model into the scaffold-free adult report task:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --report-unified-uptake --report-adult-from-lexical-checkpoint runs/organism/probe52_guided_report_lexicon/gate/organism_report_lexical_child_seed1.npz --train-steps 200000 --segment-length 64 --hidden-size 256 --report-slots 2 --report-entropy-weight 0.02 --report-audit-lives 200 --seed 1 --run-dir runs/organism/probe52_guided_report_lexicon/adult --log-every-lives 100
```

The child reaches 100% paired causal word comprehension, and the adult retains
it, but adult self-report still fails at 31.46% fidelity and 3% survival. Run
the read-only state diagnostics without changing the organism:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-audit-lives 200 --report-self-state-diagnostic-only --run-dir runs/organism/probe52_guided_report_lexicon/diagnostic_self_state
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-audit-lives 200 --report-supervised-state-upper-bound-only --run-dir runs/organism/probe52_guided_report_lexicon/diagnostic_supervised_upper_bound
```

The supervised decoder is an audit-only upper bound trained on frozen states,
not an organism capability. Verify that the permitted observation/action
history is epistemically sufficient before learning a causal self-model:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-audit-lives 200 --report-observable-history-filter-only --run-dir runs/organism/probe53_observable_history_filter/feasibility
```

Train the successful structured causal body and full-vocabulary listener
models. Random developmental tokens are independent of body, and every
pre-existing organism parameter remains frozen:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-structured-causal-development-steps 80000 --report-audit-lives 200 --seed 1 --run-dir runs/organism/probe57_structured_causal_self/treatment --log-every-lives 200
```

After the self-model and lexical gates re-pass, run the sealed learned
social-consequence planner battery:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe57_structured_causal_self/treatment/organism_causal_self_seed1.npz --report-causal-social-planner-battery --report-audit-lives 200 --report-lexical-gate-lives 300 --seed 1 --run-dir runs/organism/probe57_structured_causal_self/planner_battery
```

Probe57 reaches 94.99% report fidelity and 92% survival versus 3.5% with a
scrambled listener. Belief lesions, mute/fixed-word controls, held-out births
and portions, and perceptible body forks all support the causal interpretation.
See `docs/STATE.md` for the exact boundary: this is learned causal self-report
in the minimal ecology, not a consciousness claim.

Replicate the causal stage across independent developmental seeds and degrade
the one privileged birth reading. Note that `--seed` alone cannot vary this
stage, because the causal parameters initialize deterministically; the
developmental world stream is what must move:

```bash
PYTHONPATH=src python3 -m homesocial.organism.causal_self_replication --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --run-dir runs/organism/probe58_causal_stage_replication --seeds 5 --lives 100
```

All five seeds pass every promotion gate: balanced accuracy 0.9181 +/- 0.0066,
survival 0.9060 +/- 0.0182, fidelity 0.9471 +/- 0.0034, belief fork following
1.00, zero-belief survival 0.00. Replacing the birth reading with the
population mean, which gives zero per-life bodily information, still yields
0.81 survival and 0.906 fidelity, so the belief is a contracting observer
rather than a dead-reckoner. See
`docs/decisions/2026-07-26-causal-self-report-independent-verification.md` for
the full re-audit and for what this result still does not show.

Reproduce the post-shift outcome-aware planner experiment. It preserves the
listener outcome distribution until after applying homeostatic utility, and
factorially separates the adapted belief, planning model, objective, and real
listener:

```bash
PYTHONPATH=src python3 -m homesocial.organism.outcome_aware_self --parent runs/organism/probe57_structured_causal_self/treatment/organism_causal_self_seed1.npz --run-dir runs/organism/probe60_outcome_aware_self_planner --seeds 5 --lives 100 --adaptation-ticks 40000 --lexical-lives 90
```

The planner repair raises survival from 0.500 to 0.904 and both scrambled
listener and zero-belief lesions score 0.000. The full gate still fails:
frozen and stale beliefs survive 0.890 and 0.904, so continual recalibration is
more accurate but not behaviourally load-bearing in this three-choice ecology.
See
`docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`.

Ask whether the organism can discover the variables of its own body instead of
being handed them. The learner gets an overcomplete eight-dimensional latent and
only two scalar sensations per transition -- the mean of its bodily variables,
which is the world's own reward signal, and their minimum, which is what kills
it -- and no axis count, no axis order, no axis name, and no per-life birth
reading. The true number of bodily variables is varied in the world by freezing
axes, so the recovered dimension is scored against a ground truth that moves:

```bash
PYTHONPATH=src python3 -m homesocial.organism.discovered_self --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --run-dir runs/organism/probe61_discovered_self --seeds 5 --lives 100 --worlds K1,K2,K3,K4 --learning-rate 0.03 --sparsity 1e-5 --controls
```

`K4` is the unmodified frozen report ecology, whose true bodily dimension is
four: `safety` depletes at 0.002 per tick and enters both viability signals, and
it is the variable probe57's hand-written three-axis template cannot represent.
Pass `--sparsity` the value recorded in
`runs/organism/probe61_discovered_self/selection.json`, which is chosen by
held-out sensory error alone. Preregistration and locked gates:
`docs/decisions/2026-08-02-discovered-self-structure-preregistration.md`.

The twenty treatment fits may be split by world, because `run_world_seed` is
pure in `(world, seed)`. Run one process per world into
`runs/organism/probe61_discovered_self/full/<world>`, then merge and evaluate the
locked gates over the pooled records; the merge refuses a partial or ragged
sweep:

```bash
PYTHONPATH=src python3 scripts/probe61_merge_worlds.py --run-dir runs/organism/probe61_discovered_self/full
```

Ask whether self-uncertainty is worth anything before building a mechanism to
communicate it. `silent_shock_probability` withholds a shock's perceptible
marker while leaving the body change and every random stream identical, so it
moves what the organism can know about itself and nothing about what happens to
it. Four bodies -- the true one, probe53's exact visible-history filter, that
filter debiased, and a particle cloud conditioned on being alive -- are scored on
one shared history:

```bash
PYTHONPATH=src python3 -m homesocial.organism.uncertain_self --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --run-dir runs/organism/probe62_uncertain_self --lives 40 --particles 48 --skip-closed-loop
```

Silence opens a 26.6-point gap to the oracle on naming the truly lowest need,
and none of it is recoverable: the Bayes-optimal rule ties the biased point
filter at every silence rate. The cloud is genuinely calibrated -- coverage
0.92--0.94 against a nominal 0.90 -- but at matched inspection budget that
calibration is worth only +0.0 to +2.7 points over rate-matched random, against
up to +22.1 for inspecting at all. The body is bounded in [0,1] and the filter
saturates on ~9.7% of ticks, so its bias equilibrates at +0.051 instead of
accumulating. The ecology carries the self-model. Full record:
`docs/decisions/2026-08-03-self-uncertainty-ceiling-survey.md`.

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
