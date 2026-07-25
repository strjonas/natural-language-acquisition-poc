# Resolving the forecast/memory tension: result

Date: 2026-07-25
Decision: the budget resolves the tension completely and **P4 passes at
99.78%** — the gate that blocked three successive planners is solved. **P5
fails** on four of its six gates, three of them narrowly, so the behavioral
training pair is not licensed and this stops here per the locked rule.

## The chain of elimination

`2026-07-25-metabolic-drift-supervision-result.md` left a sharp tension. The
drift correction repairs the bodily forecast completely and carries gate 6 from
18.00% to 87.11%, but it degrades the lexical memory the headline result rests
on, from 100.00% cross-round reuse to 63.33%.

Four candidate causes were tested, each with its own control. Three were
refuted.

| Manipulation | Forecast | Memory | Verdict |
|---|---:|---:|---|
| Sealed baseline, 30k ticks | 18.00% | 100.00% | the starting tension |
| Drift term, shared head | 100.00% | 63.33% | tension appears |
| **Split drift/event head** | 100.00% | 57.83% | **not the output path** |
| Split head, no drift term (control) | 50.33% | 100.00% | architecture is harmless |
| Hidden 128 (diagnostic) | 100.00% | 50.28% | **not capacity** |
| Hidden 256 (diagnostic) | 100.00% | 54.39% | **not capacity** |
| `max_grad_norm` 10.0 | 100.00% | 62.56% | **not the gradient budget** |
| Clip 10.0, no drift term (control) | 29.33% | 98.89% | raised clip is harmless |
| **60,000 ticks** | **100.00%** | **100.00%** | **the budget** |
| 120,000 ticks (diagnostic) | 100.00% | 100.00% | no further gain |

Each negative was recorded as a negative and none of the locked stop rules was
weakened to keep an explanation alive. The capacity-based compute request that
the split-head stop rule pointed at was **withdrawn as refuted by its own
diagnostic** rather than carried forward.

## The measurement that looked right and was wrong

The gradient-norm instrumentation was the most persuasive wrong answer of the
session, and is worth recording as such. Over matched 3,000-tick runs, raw
gradient norm before clipping against `max_grad_norm = 1.0`:

| Condition | Median | p90 | Updates clipped |
|---|---:|---:|---:|
| Baseline | 1.271 | 4.541 | 59.4% |
| Drift term | 7.500 | 20.208 | 100.0% |
| Drift term + split head | 8.235 | 20.598 | 100.0% |

The drift term really does inflate the gradient sixfold and really does
saturate the trust region on every update, and that really would rescale the
sparse lexical gradients. It explained all three observations, including why
head separation and capacity both failed, since clipping is global. It was
still wrong: raising the clip to 10.0 left the memory at 62.56%, while the
control confirmed the raised clip is harmless on its own.

A mechanism that explains every observation is not thereby the cause.

## What the resolution actually is

The two objectives were never in competition for parameters, output range, or
step size. They were in competition for **experience**. Metabolism is present
in every transition; the binding-conditioned consumption jump occurs about
twice per life. At 30,000 ticks there is enough experience to fit either one
well, and adding a dense objective displaces the sparse one. At 60,000 ticks
there is enough for both, and the interference disappears entirely rather than
partially — 100.00%, not 85%.

That the fix is more online life, in a project about a continually learning
embodied organism, is the least surprising possible answer and was the last one
tested.

## Confirmatory run

Retrained into `runs/organism/probe33_online_budget/` at the preregistered
60,000-tick budget, which independently replicates the diagnostic:

| Quantity | Result |
|---|---:|
| Urgent-index survival, predicted post-return | **100.00%** |
| Urgent-index survival, predicted post-inspect | 100.00% |
| Cross-round reuse, each of rounds 3-8 | **100.00%** |
| Acute silence | 30.94% |
| Acute write suppression | 31.17% |
| Valid memory rows, grounded / controls | 2.00 / 0.00 |

Forecast and lexical memory at ceiling on one checkpoint, with both ablations
at chance. No prior checkpoint in this project has held both.

## The blind endpoints: P4 passes, P5 fails

Protocol-branch feasibility, 300 fixed contexts, reuse count 7. The sealed
baseline column is the 30,000-tick checkpoint that failed gate 6 three times.

| Gate | Requirement | Sealed baseline | 30k + drift | **60k** | Decision |
|---|---:|---:|---:|---:|---|
| 1 Mean intact inspect advantage | > 0 | +0.1735 | -0.0652 | **+0.0205** | **Pass** |
| 2 Positive-advantage contexts | >= 75% | 99.00% | 7.33% | 46.33% | **Fail** |
| 3 Write-suppression advantage drop | >= 0.05 | 0.2371 | 0.0570 | 0.0466 | **Fail** |
| 4 Collapsed-label advantage drop | >= 0.05 | 0.2371 | 0.0570 | 0.0466 | **Fail** |
| 5 Label-contingent terminal choices | >= 60% | 99.89% | 64.11% | **85.11%** | Pass |
| 6 Matching label selects the target | >= 60% | **18.00%** | 87.11% | **99.78%** | **Pass** |
| 7 Danger label avoids the target | >= 90% | 100.00% | 74.44% | 85.33% | **Fail** |

**P4 is the headline.** Gate 6 — failed by the mean-observation planner, the
observation-branching planner and the protocol branch at 19.11%, 18.00% and
18.00% — is now **99.78%**. The oracle bound predicted this and the trained
model delivers it. Branch contingency is fully write-caused: 85.11% intact
against exactly 0.00% under both write suppression and collapsed labels.

Gate 1 also passes for the first time under a repaired forecast: the mean
inspect advantage is positive, so acquiring a word is on average worth its
metabolic and delay cost. That was the operation `docs/STATE.md` named as the
one still missing.

Gates 3, 4 and 7 fail narrowly (0.0466 against 0.05; 85.33% against 90%). Gate
2 fails widely and carries a specific anomaly worth naming: the
positive-advantage context rate is **identical at 46.33%** across intact, write
suppressed and collapsed conditions, while the mean advantages differ
(+0.0205 against -0.0262). Whether a context has positive inspect advantage is
therefore being decided by something the write does not touch, even though the
*size* of the advantage is fully write-caused. That dissociation is the precise
next object of study, and it is a property of the branch construction rather
than of the forecast.

Per the locked interpretation rule for "P4 passes, P5 fails", the forecast
repair is confirmed, the branch battery is the remaining obstacle, and this
stops here. No fifth planner, no gate weakened, no grid reopened.

## Provenance and integrity

The 60,000-tick budget was found by a **labelled diagnostic**, and
`2026-07-25-online-budget-preregistration.md` says so explicitly: the forecast,
memory and control numbers above were seen before that document was written and
are reported as replications, not as blind endpoints. Only the feasibility
battery and the write-disabled control were locked blind.

Artifacts: `runs/organism/probe31_split_drift_head/`,
`runs/organism/probe32_gradient_budget/`, `runs/organism/probe33_online_budget/`.

## Claim boundary

One seed. No write-disabled training pair yet, no seed 2, no freshly trained
silent or shuffled controls, no open-island transfer, no generated speech.
Nothing here is reflection, self-report, or consciousness. The organism still
does not produce language, and its needs and objective remain engineered.

No compute request is justified: the working budget is 60,000 ticks, which
trains locally in under a minute.
