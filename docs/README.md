# docs/ index

Read these three, in order. Everything else in this directory is frozen.

| File | What it is |
|---|---|
| `DIRECTION_2026-07-26.md` | **The destination.** Terminal claim, the five properties a real self-model needs, and the phase ladder to earn them. |
| `STATE.md` | **The rolling handover.** Where things stand right now. Rewritten, never appended. |
| `decisions/` | **The durable evidence record.** Append-only. One preregistration and one result per experiment. |

`runs/` is gitignored, so run outputs are not durable. Any number worth keeping
belongs in a `decisions/` document.

## Current position

Tagged `v1-causal-bodily-self-report`. A learned bodily observer and a
three-word signaling policy, replicated over five seeds with every causal
control failing as required. Not yet a self-model in the sense the project is
aiming at -- its structure is stipulated rather than discovered, it has no
evidence-correction pathway, and it does not reflect. See
`DIRECTION_2026-07-26.md` section 1 for the honest accounting and section 3 for
what closes each gap.

## Frozen history

Kept as evidence of what was tried and refuted. Each carries a header saying
so. Do not treat any plan, status, or number in them as current.

- `DIRECTION_2026-07-12.md`, `PLAN_ORGANISM.md` -- the probe-to-organism pivot
  and its engineering plan, both executed
- `grand_architecture_roadmap.md`, `evaluation2045.md` -- pre-pivot architecture
- `results_2026_06_23.md`, `results_synthesis.md`, `phase_1_spec.md` -- probe-era
  results and specs
- `../preregistration.md` -- the original M0 preregistration
- `HighLevelPlan.md` -- the original framing conversation

## Naming

- `decisions/<date>-<experiment>-preregistration.md` -- mechanism and locked
  gates, written **before** implementing
- `decisions/<date>-<experiment>-result.md` -- outcome, including failures,
  and the explicit claim boundary
