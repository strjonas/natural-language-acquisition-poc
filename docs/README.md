# Documentation index

For a public overview, read these first:

| File | What it is |
|---|---|
| `VISION_AND_STATUS.md` | Original vision, novelty boundary, acquisition-vs-emergence distinction, and the route to productive language. |
| `STATE.md` | Current evidence, failures, and exact next work. Rewritten rather than appended. |
| `decisions/2026-08-03-individual-self-calibration-result.md` | The strongest current empirical result and its claim boundary. |
| `../paper/OUTLINE.md` | Honest vision-plus-interim-result workshop paper route. |
| `INTERVIEW_GUIDE.md` | Short and technical explanations for a robotics interview. |

For continued research, read these three in order. Most other planning files
are frozen historical records.

| File | What it is |
|---|---|
| `DIRECTION_2026-07-26.md` | **The destination.** Terminal claim, the five properties a real self-model needs, and the phase ladder to earn them. |
| `STATE.md` | **The rolling handover.** Where things stand right now. Rewritten, never appended. |
| `decisions/` | **The durable evidence record.** Append-only. One preregistration and one result per experiment. |

Most of `runs/` is gitignored. A roughly 3 MB allow-listed publication bundle
contains Probe63's JSON records and the parent checkpoint required for a smoke
rerun. Any other number worth keeping belongs in a `decisions/` document.

## Current position

Tagged `v1-causal-bodily-self-report`, then extended by Probe63. The repository
now contains a three-word signaling policy plus an online RLS body calibrator
that learns individual bodily constants from intermittent evidence and passes
all seven preregistered gates over five seeds. It remains far short of the grand
vision: bodily structure is largely stipulated, the language is selected rather
than productively generated, and there is no reflective or narrative learner.

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
