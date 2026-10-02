# Probe79: observable-error resetting fails the selectivity and control gates

Recorded: 2026-10-02. Preregistration:
`2026-10-02-evidence-reset-preregistration.md`.
Module: `src/homesocial/organism/evidence_reset.py`.

**G1 passes; G2 and G3 fail. This fixed detector is closed.** Two consecutive
large, same-sign prediction errors improve rate estimates and counterfactual
requests after hidden body changes, but trigger too often in unchanged bodies
and do not reliably outperform randomized error signs. No closed-loop detector
survival treatment is licensed, and no threshold or streak was retuned.

## Construction and endpoints

Five paired blocks x 28 identities x five test episodes in each of stationary
and changed cohorts: 1,400 test episodes, plus 140 calibration episodes.
Retained memory drives every physical trajectory. Adaptive, muted,
sign-scrambled, oracle-reset and true-rate readouts share that history and
current-state/uptake belief. Only their rate estimates differ. Therefore the
common driver's survival cannot establish any detector benefit.

The detector uses real, pre-fit prediction errors at ordinary sparse readings.
Two successive eligible errors above 0.05 body units with the same sign on one
channel clear rate evidence; the triggering real reading is then fitted.
The sign-scrambled control randomizes only detection signs, preserving error
magnitudes and fitting the actual readings. Muted rates and words match the
retained driver exactly at every decision, asserted during execution.

Means below first weight decisions within each block, then weight the five
blocks equally. They are not pooled episode averages.

| changed-history rate readout | rate MAE | word regret |
|---|---:|---:|
| retained / muted | 0.002141 | 0.013392 |
| adaptive reset | 0.001099 | 0.008812 |
| sign-scrambled reset | 0.001246 | 0.009277 |
| oracle reset | 0.000763 | 0.006756 |
| true rate | 0 | 0.004701 |

Adaptive reductions versus retained are **48.68% MAE** and **34.20% regret**.
Regret is a matched counterfactual request endpoint, not observed survival.
Even true-rate regret is nonzero because the common state/uptake estimates can
still be wrong.

## Locked gates

| paired contrast, positive favors adaptive | mean [95% interval] | positive blocks |
|---|---|---:|
| retained minus adaptive MAE | +0.001042 [+0.000864, +0.001221] | 5/5 |
| retained minus adaptive regret | +0.004581 [+0.003295, +0.005867] | 5/5 |
| sign-scrambled minus adaptive MAE | +0.000147 [-0.000061, +0.000355] | 4/5 |
| sign-scrambled minus adaptive regret | +0.000466 [-0.000972, +0.001903] | 4/5 |

Intervals use five paired blocks and Student-t critical 2.776.

- **G1 changed improvement passes:** both lower bounds are positive, all five
  blocks improve, and both reductions exceed the locked 20% minimum.
- **G2 stationary preservation fails:** adaptive resets **24/140 unchanged
  identities (17.14%)**, exceeding the locked 10% budget. By-block counts are
  **5, 3, 5, 5, 6**; these identities receive 33 resets. The other two clauses
  pass: stable MAE is 0.000421 versus retained 0.000428, and mean regret cost is
  +0.000175, below 0.001. A stable-body reset is a false alarm about physical
  change under this gate; it can nevertheless correct an initially inaccurate
  estimate. These results do not say every such reset is harmful.
- **G3 controls fails:** muted parity passes, but both adaptive-versus-scrambled
  intervals cross zero. Four favorable blocks do not satisfy the required
  positive lower bounds. This is insufficient evidence for directional
  persistence, not proof that the two detectors are equivalent.

In changed bodies, adaptive resets 77/140 identities, with 90 total resets;
sign-scrambled resets 69/140, with 84 total resets. These are descriptive reset
counts, not detection sensitivities: episodes before change also contribute.
In this sample 72/140 changes occur within the scheduled life, and 68 occur at
the next birth after death. All identities remain in the endpoints.

## Verification and reproducibility

All 420 progress units completed. The two-identity diagnostic took 9.46 seconds;
the full screen took **332.02 seconds (5.53 minutes)**, below the estimated
12 +/- 4 minutes. The repository suite passed **610 tests and 13 subtests** in
80.01 seconds while the screen ran. Tests cover pre-fit scoring, one transition
per update, streak and episode semantics, no truth access in the estimator,
sign corruption isolated from fitting, real muted parity, binding gates,
seed isolation and source/configuration-pinned resume.

`scripts/summarize_probe79.py` independently reconstructs the endpoints,
paired intervals and gates using only the standard library. Raw extraction
checks all calibration/episode seeds, complete paired cohorts, muted episode
scores and pre-change equality. Both raw extraction and reconstruction from
the compact public evidence agree with the saved summary. The checker does
not independently simulate the physics or reimplement the learner. Compact
records retain block sums; per-episode checks require the local raw artifact.

```bash
.venv/bin/python scripts/summarize_probe79.py \
  --input runs/organism/probe79_evidence_reset/evidence.json
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.evidence_reset \
  --out /tmp/probe79-survey.json
```

The compact record retains the source, parent and preregistration fingerprints
and raw-artifact hash. A new source version requires a fresh run output path.

## Decision and claim boundary

Probe78 establishes physical value for correctly timed forgetting. Probe79
does not establish a selective observable change detector, calibrated
uncertainty, learned reflection, or an adaptive survival benefit. Its positive
matched-history improvements do not override the failed gates.

Close this threshold-and-streak mechanism before a survival treatment. A
different proposal must explain how it distinguishes ordinary estimation error
from stale evidence, use separate calibration and held-out evaluation, and
retain stationary false-alarm and causally informative controls. Further work
is a new proposal, not a rescue of this run.
