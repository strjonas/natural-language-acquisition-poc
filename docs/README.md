# Documentation index

For a public overview, read these first:

| File | What it is |
|---|---|
| `VISION_AND_STATUS.md` | Original vision, novelty boundary, acquisition-vs-emergence distinction, and the route to productive language. |
| `STATE.md` | Current evidence, failures, and exact next work. Rewritten rather than appended. |
| `decisions/2026-08-16-discovered-convention-result.md` | The newest result: the organism finds out for itself whether a word means the same thing whatever it is asking for -- generalizing where that is licensed at no cost, and declining where it is not. |
| `decisions/2026-08-16-future-request-result.md` | The result before it: whether it is better to know where your body is or what your body is depends on how far ahead the world makes you think -- with the crossover predicted before it was measured. |
| `decisions/2026-08-14-portion-request-result.md` | A fact about the self that perfect state knowledge cannot supply, and the behavioural null that bounds it. |
| `decisions/2026-08-03-individual-self-calibration-result.md` | The individual-body calibration result and its claim boundary. |
| `../paper/OUTLINE.md` | Honest vision-plus-interim-result workshop paper route. |
| `INTERVIEW_GUIDE.md` | Short and technical explanations for a robotics interview. |

For continued research, read these three in order. Most other planning files
are frozen historical records.

| File | What it is |
|---|---|
| `../md/archive/DIRECTION_2026-07-26.md` | **The destination.** Terminal claim, the five properties a real self-model needs, and the phase ladder to earn them. |
| `STATE.md` | **The rolling handover.** Where things stand right now. Rewritten, never appended. |
| `decisions/` | **The durable evidence record.** Append-only. One preregistration and one result per experiment. |

Most of `runs/` is gitignored. A small allow-listed publication bundle contains
Probe63's, Probe64's, Probe65's and Probe66's JSON records and the parent checkpoint required
for a smoke rerun. Any other number worth keeping belongs in a `decisions/` document.

## Current position

Tagged `v1-causal-bodily-self-report`, then extended by Probes 63, 64 and 65. The
repository contains a six-signal request protocol (need x portion size) plus an
online RLS body calibrator that learns individual bodily constants from
intermittent evidence and passes every preregistered gate over five seeds, three
times running.

Probe65 is the current frontier and it changes how the earlier results read. When
the caregiver answers a request eighteen ticks after hearing it, every word
becomes a prediction, and an error about *where you are* enters that prediction
once while an error about *what you are* enters multiplied by the horizon. The
ordering between those two kinds of self-knowledge therefore **reverses** with
the horizon, at a crossover the preregistration computed in advance and got right
to within a tick. Probe65 also passes the control that phase C2 names as the one
that kills templated narration -- reports must diverge when the future diverges
while the present is identical -- and carries the first behavioural gate a belief
has ever carried here.

It remains far short of the grand vision. Bodily structure is largely stipulated,
the listener factorization is given rather than discovered, the decision rule is
designer-supplied, utterances are selected rather than generated, the vocabulary
is three need words plus two size words, and there is no reflective or narrative
learner. Probe65's own behavioural claim is bounded: the *horizon* reaches
survival, the individual *rate* does not yet, and the result states the sample
size that would settle it.

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
