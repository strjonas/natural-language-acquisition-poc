# Structure against scale: result

Date: 2026-08-18. Decision: **six of seven locked gates pass. G6 fails and is
not rewritten.**

Preregistration: `docs/decisions/2026-08-17-belief-scaling-preregistration.md`.
Survey: `docs/decisions/2026-08-17-belief-scaling-ceiling-survey.md`.
Module: `src/homesocial/organism/belief_scaling.py`. Grader:
`scripts/grade_probe69.py`.
Artifacts: `runs/organism/probe69_belief_scaling/{treatment,extrapolation}/scaling_survey.json`.
Bands: development 1,900,000,000, audit 2,100,000,000 — disjoint from the
survey's and from every band previously claimed. 24.4 hours of compute over 51
cells.

## 1. What was tested

Whether the pattern that organizes this repository's evidence ladder — five
structured mechanisms passing, nine black-box ones failing — is confounded with
the 80,000-tick development budget at which all of those comparisons were made.

Probe54's gate is **not** re-run here and is **not** promoted. It stays failed.
The harness lock at `harness.py:1220` is untouched. The endpoint is the shape of
two curves on a matched instrument.

## 2. The curve

Five seeds, 200 audit lives per cell, `_balanced_accuracy` over
`world.lowest_need()` — the same function object both audits already import.

| budget | structured | black-box w64 | black-box w256 | gap (w256) |
|---:|---:|---:|---:|---:|
| 80,000 | 0.9234 | 0.5906 | 0.7447 | **0.1788** |
| 320,000 | 0.9338 | 0.7818 | 0.8390 | 0.0948 |
| 1,280,000 | 0.9330 | 0.8695 | 0.9028 | 0.0301 |
| 5,120,000 | 0.9351 | — | **0.9349** | **+0.0003** |

`gap` is `structured − black-box` at the same budget and seed. The 5,120,000 row
is three seeds; the rest are five.

| gate | locked | measured | |
|---|---|---:|---|
| G1 | structured 1.28M − 80k **< +0.05** | **+0.0095** | PASS |
| G2 | blackbox256 1.28M − 80k **> +0.10**, CI excludes 0, >= 4/5 | **+0.1582** [+0.1419, +0.1744], 5/5 | PASS |
| G3 | gap(1.28M) **< 0.5x** gap(80k), >= 4/5 | **0.168x**, 5/5 | PASS |
| G4 | every lesion **<= 0.40**, all cells | max **0.3368** | PASS |
| G5 | gap 80k > 320k > 1.28M, >= 4/5 | 5/5 | PASS |
| G6 | gap(5.12M) in **[0.005, 0.045]** and no crossing 3/3 | **+0.0003**, crossing 1/3 | **FAIL** |
| G7 | blackbox_w64@80k within **+/-0.05** of probe54's 0.6219 | **0.5906** (d 0.0313) | PASS |

## 3. The finding

**The structured model is saturated at the locked budget.** Sixteen times the
compute buys +0.0095, and the curve is non-monotonic (0.9234 → 0.9338 → 0.9330).
Probe57's 80,000 ticks — 90 seconds on this machine — was, for that family,
already converged.

**The black-box is not saturated until it has caught up.** +0.1582 over the same
range on 5/5 seeds, and the gap collapses monotonically on every seed:

    0.1788  ->  0.0948  ->  0.0301  ->  +0.0003

> **The locked budget of 80,000 sits where the two families are maximally
> separated, and the separation is gone by 5,120,000.** Every comparison in this
> repository between a structured mechanism and a black-box one was measured at
> the single budget most favourable to structure.

**This is not a demotion of structure.** The better-supported reading is the
opposite one, and it is the durable output of this probe:

> The structured advantage is a **sample-efficiency** advantage, and it is large:
> 0.9234 in ninety seconds, against a black-box that needs roughly **64x** the
> compute to reach the same number. It is **not a ceiling**. The two families
> tie at 0.935.

Both statements matter for a scaling decision and they point opposite ways. A
budget-constrained programme should build the structured mechanism. A programme
that intends to scale cannot cite this repository's evidence ladder as a reason
not to.

## 4. G6 failed, and what it was wrong about

The survey fixed a gap-closure rate of 0.515 per 4x of compute (measured 0.480
and 0.550) and predicted a **residual gap of 0.025** at 5,120,000 with no
crossing. Locked before the run. Measured:

| seed | black-box | structured | gap |
|---|---:|---:|---:|
| 0 | 0.9324 | 0.9437 | +0.0113 |
| 1 | 0.9270 | 0.9148 | **−0.0122** |
| 2 | 0.9451 | 0.9468 | +0.0017 |
| **mean** | **0.9349** | **0.9351** | **+0.0003** |

Both halves fail: +0.0003 is below the [0.005, 0.045] band, and one seed of
three crosses.

**The direction was right and the magnitude was wrong.** The gap did not shrink
to a residual; it closed to zero. The error is diagnosable and is worth
recording as a method note: the 0.515 ratio was fitted on **one seed**, and its
last two points were already inside the noise floor. The per-seed standard
deviation of the structured arm at 1,280,000 is **0.0153**, which is fifty times
the family difference being extrapolated. **A ratio fitted between two points
that are both inside the noise cannot constrain a third.**

**This is a failed prediction and is reported as one**, per the preregistration's
own instruction, rather than folded into the success of G1–G5. What it is *not*
is evidence of crossing: with a seed spread of 0.0093–0.0153 against a family
difference of 0.0003, the honest word is **tie**. The gate that would have
detected a genuine crossing correctly declines to call one.

## 5. Probe55's readout bottleneck was a small-budget artifact

Probe55 recorded that a ridge readout of its hidden state reached 72.42% while
its output head reached 62.79%, and concluded that the state held gate-level
information the head could not express. Measured against budget, `head − ridge`:

| cell | head | ridge | head − ridge |
|---|---:|---:|---:|
| w64 @ 80k | 0.5906 | 0.7254 | **−0.1349** |
| w64 @ 320k | 0.7818 | 0.8091 | −0.0273 |
| w64 @ 1.28M | 0.8695 | 0.8574 | **+0.0121** |
| w256 @ 80k | 0.7447 | 0.8050 | −0.0603 |
| w256 @ 1.28M | 0.9028 | 0.8904 | +0.0125 |
| w256 @ 5.12M | 0.9349 | 0.8951 | **+0.0397** |

The inversion is real at 80,000, reverses, and keeps widening — the head ends up
**0.04 above** a linear readout of its own state, which is what a head that has
learned a nonlinear decoding looks like. Probe55's diagnosis was correct and its
*inference* was not: the limit it found was an under-trained head, not an
architecture. The head variants the line was subsequently closed on were the
wrong lever, which is consistent with all of them having failed.

## 6. Controls

Every cell, every seed, both arms — 51 cells:

| | max over all cells |
|---|---:|
| zero-belief balanced accuracy | 0.3368 |
| shuffled-belief balanced accuracy | 0.3368 |

Chance is 0.3333. Both lesions sit at chance everywhere, including where
accuracy is 0.935. Nothing here is the audit scoring itself.

G7 is the guard against the documentation-drift trap: the instrument reproduces
probe54's published 0.6219 to within 0.031 on a different band, so the cells
here are commensurable with the record they are being compared to.

## 7. What this does not show

- **Belief-side only.** Balanced lowest-need accuracy. **Nothing here shows a
  better belief reaching a word**, and probe68's margin law says it need not: a
  belief error changes an utterance only if it exceeds
  `E_grant[min(grant · uptake, g)]`. Probes 59, 60, 62, 64 and 67 each improved a
  belief and bought no behaviour. **A black-box that ties the structured model on
  accuracy inherits that problem whole**, and this probe does not touch it.
- **Nothing about survival, language, the mouth, or the five properties.** No
  property advances. The mouth is still `argmax` over three scores.
- **One architecture family, one loss, one parent.** A GRU at two widths,
  supervised on developmental body values, developed from the single frozen
  probe52 adult that every causal-stage result in this repository shares. The
  replication debt in `DIRECTION` section 5 is untouched and now matters more,
  not less.
- **The comparison is at equal ticks, not equal wall-clock or equal
  parameters.** Width is free here because the loop is world-simulation-bound;
  on hardware where it is not, the black-box's 64x is an underestimate.
- **Above 5,120,000 is unmeasured.** The claim is that the gap closes, not that
  anything happens beyond it.
- **`train_explicit_self_belief` is supervised on true body values.** It is not
  learning the body from consequence, and neither arm here speaks or acts. This
  is an estimator comparison.

## 8. Consequence for the record

`docs/STATE.md` lists "black-box recurrent self-belief width/loss/head variants"
and "compute scale as a substitute for causal structure" as closed. Neither
closure is overturned by this result and both should be **restated** rather than
deleted:

- the *head/width/loss variant* line stays closed — this probe moved budget, and
  §5 shows head variants were the wrong lever;
- *compute as a substitute for causal structure* stays closed as a **claim about
  substitution**, and is now qualified by a measurement: compute is not a
  substitute at 80,000 ticks, is most of the way there at 1,280,000, and is
  indistinguishable at 5,120,000.

What must be added to the record is that **no comparison between the families in
this repository was ever run at more than one budget**, and that the one budget
chosen is the point of maximum separation.
