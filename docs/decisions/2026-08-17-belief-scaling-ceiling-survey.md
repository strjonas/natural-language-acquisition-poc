# Structure against scale: a ceiling survey

Date: 2026-08-17. Status: **feasibility survey.** No gate is locked here and none
is claimed; the gates are in
`docs/decisions/2026-08-17-belief-scaling-preregistration.md`. This document
records what the survey established, which stands whatever the treatment
returns.

Artifacts: `runs/organism/probe69_belief_scaling/scaling_survey.json`.
Module: `src/homesocial/organism/belief_scaling.py`.
Bands: survey development 1,500,000,000, survey audit 1,700,000,000 — both
disjoint from the treatment's 1,900,000,000 / 2,100,000,000 and from every band
previously claimed (the highest was probe68's 1,400,000,000).

## 0. This touches a closed line. Read this section first.

`docs/STATE.md` closes **"black-box recurrent self-belief width/loss/head
variants"** and **"compute scale as a substitute for causal structure"**. A
reviewer's first challenge will be that this is budget-shopping on a failed
gate, and that challenge is correct about what would make this illegitimate. So,
precisely:

- **Probe54's gate is failed and stays failed.** Nothing here re-runs it and
  reports a pass. The harness lock at `harness.py:1220` pinning development to
  exactly 80,000 ticks is deliberate, is the right guard, and is **untouched**.
- **The endpoint is not a threshold.** It is the *shape* of two curves and the
  gap between them. A flat black-box curve would have *strengthened* the
  existing closure, and the module was built to be able to report that.
- **The contrary evidence that justifies asking** is inside the closing document
  itself. Probe55 recorded that a disposable ridge readout of its 64-unit hidden
  state reached **72.42%** — above the 70% gate its output head missed at
  62.79% — and wrote: "The state still contains gate-level identity information
  that the shared magnitude-and-order output does not express." That was a
  diagnosis, and it was never followed up, because the line was closed on head
  variants rather than on budget.

What had never been asked is whether the family failed because it is the wrong
family or because it was given 80,000 ticks and 738 gradient updates. Measured
on this machine, that budget is **90 seconds**.

## 1. The instrument is matched, and it reproduces the record

Both arms are scored by `_balanced_accuracy` over `world.lowest_need()` —
literally the same function object, imported by `causal_self` from
`self_belief`, with the same truth and the same `argmin` prediction rule. The
only difference between the arms' audits is which module holds the belief.

Two calibration checks against published numbers, on a different audit seed
base, so agreement is expected to be close rather than exact:

| cell | here | published | source |
|---|---:|---:|---|
| structured @ 80k | 0.9261 | ~0.918 | probe57 / probe58 five-seed |
| black-box w64 @ 80k | 0.5988 | 0.6219 | probe54 |

The instrument reproduces the record on both families.

## 2. The two families have opposite compute curves

One seed, 200 audit lives per cell, 114 minutes total.

| budget | structured | black-box w64 | black-box w256 | gap (w64) | gap (w256) |
|---:|---:|---:|---:|---:|---:|
| 80,000 | **0.9261** | 0.5988 | 0.7402 | **0.3273** | **0.1859** |
| 320,000 | 0.9183 | 0.7825 | 0.8290 | 0.1358 | 0.0893 |
| 1,280,000 | **0.9419** | 0.8654 | **0.8928** | 0.0764 | **0.0491** |

- **The structured model is saturated at the locked budget.** Sixteen times the
  compute buys **+0.0158**, and the curve is *non-monotonic* (0.9261 → 0.9183 →
  0.9419), so the move is at noise level. Probe57's 80,000 was, for this family,
  a well-chosen budget: it was already converged.
- **The black-box is not saturated anywhere in this range.** +0.1526 at width
  256, monotonic, still climbing at 1,280,000.
- **Width is free.** 87.6s at width 256 against 90.8s at width 64 — the loop is
  world-simulation-bound, not network-bound. The entire "was it simply too
  small" axis costs nothing to sweep, which is why no previous probe's cost
  argument applies to it.

### The consequence, which is the finding of this survey

> **The locked budget of 80,000 sits where the two families are maximally
> separated.** The gap runs 0.327 → 0.076 (width 64) and 0.186 → 0.049 (width
> 256) across the range. Every comparison in this repository between a
> structured mechanism and a black-box one was measured at the single budget
> most favourable to structure.

That is not a claim that structure is worthless — the opposite reading is the
better-supported one. Structure reaches 0.926 in **90 seconds** and never
improves again; the black-box needs roughly 16x that to come within 0.05 of it.
**Structure buys sample efficiency, not a ceiling.** But the pattern that
organizes the repository's evidence ladder — five structured passes against nine
black-box failures — is confounded with a budget, and no document had recorded
that.

## 3. Probe55's readout bottleneck is a small-budget artifact

Probe55's diagnosis was that the hidden state held gate-level information the
head could not express. Measured against budget, that inversion is real at
80,000 and **reverses**:

| cell | head | ridge | head − ridge |
|---|---:|---:|---:|
| w64 @ 80k | 0.5988 | 0.7319 | **−0.1331** |
| w64 @ 1.28M | 0.8654 | 0.8478 | **+0.0176** |
| w256 @ 80k | 0.7402 | 0.8178 | **−0.0776** |
| w256 @ 1.28M | 0.8928 | 0.8836 | **+0.0092** |

The head catches up with the linear readout of its own state and passes it. So
the readout limit probe55 identified was correctly identified and is not a
property of the architecture; it is what an under-trained head looks like. The
head variants that line was closed on were the wrong lever, which is consistent
with them having failed.

## 4. Both lesions hold at every budget

The control that the accuracy is causal in the belief rather than an artifact of
the audit, at all nine cells:

| | min | max |
|---|---:|---:|
| zero-belief balanced accuracy | 0.3333 | 0.3333 |
| shuffled-belief balanced accuracy | 0.3311 | 0.3338 |

Chance is 0.3333. Both lesions sit at chance everywhere, including the
high-compute black-box cells where accuracy is 0.89. Nothing here is the audit
scoring itself.

## 5. The prediction this survey fixes in advance

The gap between the black-box (width 256) and the structured model shrinks by a
strikingly constant factor per 4x of compute:

    0.0893 / 0.1859 = 0.480
    0.0491 / 0.0893 = 0.550        mean 0.515

Extrapolated at that rate, and **locked here before the treatment runs**:

| budget | predicted gap | predicted black-box |
|---:|---:|---:|
| 5,120,000 | **0.025** | ~0.917 |
| 20,480,000 | **0.013** | ~0.929 |

So the prediction is neither "scale wins" nor "structure wins". It is
**convergence without crossing**: the black-box approaches the structured
model's level and does not overtake it inside this range. That is a specific,
falsifiable claim, and probe65's practice — fix the crossover from already-published
numbers before the run — is what makes it worth stating.

## 6. What the survey cannot settle

- **It is one seed.** Probe68's survey warning is explicit and was earned twice:
  single-seed readings on a balanced-accuracy endpoint have already pointed the
  wrong way. The structured arm's own non-monotonicity here puts the noise floor
  at no better than +/-0.02. Nothing in section 2 is a result until the treatment
  returns.
- **It is belief-side only.** Balanced lowest-need accuracy says nothing about
  whether any of it reaches a word. Probe68's margin law governs that and is
  unaffected: a better belief that does not exceed the ecology's margin still
  changes nothing. Probes 59, 60, 62, 64 and 67 all improved a belief and bought
  no behaviour, and a black-box that matches the structured model's accuracy
  would inherit exactly that problem.
- **It says nothing about survival**, about language, about the mouth, or about
  the four properties. It is one estimator's accuracy against one budget.
- **One architecture family, one parent.** A GRU of two widths, developed from
  the single frozen probe52 adult that every causal-stage result in this
  repository shares. The replication debt in `DIRECTION` section 5 applies here
  unchanged.
- **The top of the curve is unmeasured.** 1,280,000 is where the black-box was
  still climbing, not where it stopped.
