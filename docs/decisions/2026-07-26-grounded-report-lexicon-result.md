# Grounded report-lexicon development result

Date: 2026-07-26  
Decision: comprehension gate failed; adult report training was not started

## Integrity and provenance

The first three attempts aborted before checkpoint or evaluation because the
new energy condition exposed two fixed-option pose bugs and one missing
`consumed_energy` round-terminal event. Regression tests now cover all three
targets in all eight rounds, the inspect-return-next-option path, and the
energy transition.

The first completed run then exposed a separate accounting bug: consuming on
the exact global cutoff cleared the grid's truncation flag and recorded 60,002
ticks. Its artifact is retained under
`runs/organism/probe51_grounded_report_lexicon/gate/` and is not the interpreted
endpoint. The gate failure from that run was already visible before the fix.

The interpreted artifact is therefore a non-blind deterministic integrity
rerun under the unchanged locked thresholds and seed:

`runs/organism/probe51_grounded_report_lexicon/gate_exact/`

It contains exactly 60,000 summed life ticks across 1,363 lives. The harness
stopped after the frozen gate and wrote no adult checkpoint or adult update.

## Frozen comprehension gate

| Endpoint | Locked gate | Exact-budget result | Decision |
|---|---:|---:|---|
| Intact correctness | >= 80% | **37.67%** | Fail |
| Food correctness | >= 70% | 69.32% | Fail |
| Water correctness | >= 70% | 42.61% | Fail |
| Energy correctness | >= 70% | 3.09% | Fail |
| Cyclic-word correctness | <= 45% | 36.33% | Pass alone |
| Intact minus cyclic | >= 30 points | **1.33 points** | Fail |
| Paired action change | >= 60% | **1.67%** | Fail |

The cyclic score is low only because both conditions are near chance. It is not
evidence for word use: changing the heard content word almost never changes the
selected surface.

## Predeclared diagnosis

Training accumulated 622 voluntary labels in 1,363 lives. Exposure collapsed
as learning proceeded:

| Primitive-tick window | Labels per life | Choice correctness |
|---:|---:|---:|
| 0-20k | 1.287 | 35.07% |
| 20-40k | 0.157 | 34.20% |
| 40-60k | **0.074** | 32.25% |

A read-only explicit-write geometry audit on 300 held-out contexts localized
the break:

- mean pairwise lexical-value L1 distance: **0.0969** (the three inputs are not
  collapsed);
- mean pairwise word-conditioned target-delta L1: **0.00305**;
- word-conditioned predicted-kind accuracy: **33.78%**;
- matching-word predicted-kind accuracy: 40.00%;
- any-word selection contingency: **4.00%**; and
- cyclic-word selection change: **2.67%**.

Every word predicts almost the same food-biased consequence: mean predicted
food restoration is 0.1274-0.1287, water 0.0528-0.0544, and energy is negative
(-0.0496 to -0.0479), regardless of whether the controlled word is hungry,
thirsty, or tired.

The failure is therefore upstream of production and downstream of bare token
separation: voluntary information acquisition vanishes, so the lexical values
do not become differentiated bodily-consequence models. The action reducer
cannot use a distinction the dynamics model never learned.

## Claim boundary and decision

This run does not show lexical comprehension, grounded production, or
self-report. It does show that weight tying alone is not a bridge from a hidden
self-state to language, and it falsifies the hypothesis that ordinary sparse
actor-critic exploration will preserve enough voluntary report-word exposure
in this childhood.

Per the locked rule, adult training, report evaluation, seed 2, sweeps, larger
compute, and generated data are blocked. The next mechanism-level test is
caregiver-guided joint attention to a uniformly random external resource. It
changes exposure, not the report loss, thresholds, output vocabulary, or adult
task, and is preregistered separately before implementation.
