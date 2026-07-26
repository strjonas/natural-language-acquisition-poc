> **FROZEN HISTORICAL RECORD.** Superseded by `docs/DIRECTION_2026-07-26.md`.
> Kept as evidence of what was tried. Do not append to it and do not treat any
> plan, status, or number in it as current. Current state: `docs/STATE.md`.

# Phase 1 Spec: Homeostatic Social Grid

This phase should answer one question:

Can we build a minimal closed loop where social language improves an agent's
ability to regulate its own viability?

## Build Target

The first environment is intentionally small:

- 2D grid world with partial observation.
- Agent body has internal variables: food, water, energy, safety.
- Resources affect internal variables through action.
- Teacher utterances are contingent on situated actions.
- Reward is derived from viability change, not instruction following.

The teacher is part of the environment response function. It should not narrate
the agent's inner life, and it should not provide dialogue that pretends the
agent already has a self. It labels and advises only from shared situation.

## Initial Conditions

Condition A: no teacher.

Condition B: grounded teacher.

Condition C: decoupled text/objective knowledge. This comes later, after the
grounded loop is working and we have non-linguistic metrics.

## Minimal Metrics

The first metrics are behavioral and non-linguistic:

- survival length,
- mean viability over time,
- successful resource use,
- avoidance of danger,
- recovery from low need states,
- whether language improves any of the above under matched training budgets.

Self-like organization is not measured by self-talk. Later tests should measure:

- self-caused versus world-caused sensory change,
- prediction of own internal state under planned action,
- protection of future viability,
- generalization from known danger to novel threats,
- whether a compact internal state is causally useful for control.

## Current Prototype

The code currently provides:

- `HomeostaticSocialGrid`,
- deterministic `SituatedTeacher`,
- homeostatic reward,
- partial observation,
- a random agent,
- a small hand-coded need-seeking baseline,
- unit tests and an ASCII demo.

## Next Engineering Step

Current result: the tabular Q-learning baseline does not reliably benefit from
grounded teacher language, even when object positions are randomized and object
kinds are hidden from the learner. This is a learner/capacity warning, not yet a
negative result about the environment.

Next, add a teacher-following sanity baseline to prove that teacher utterances
are actionable in principle. After that, add a trainable baseline with a small
recurrent policy. Keep it simple:

- tabular or tiny neural policy first,
- teacher/no-teacher comparison,
- fixed seeds,
- logged episode summaries,
- no world-model architecture until the environment and metrics are stable.
