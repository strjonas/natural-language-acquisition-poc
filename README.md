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
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 500 --eval-episodes 20 --hidden-size 128 --log-every 50
```

Run a cheaper fixed-world diagnostic before spending time on randomized hidden
object-kind sweeps:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 120 --eval-episodes 10 --hidden-size 64 --max-steps 80 --fixed-world --conditions grounded_teacher --log-every 40
```

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
