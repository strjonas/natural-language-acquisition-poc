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

Run a larger local sweep once the fixed-world diagnostic is healthy:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 2000 --batch-size 32 --ppo-epochs 3 --eval-episodes 50 --hidden-size 256 --max-steps 160 --conditions grounded_teacher silent_teacher --log-every 100
```

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
