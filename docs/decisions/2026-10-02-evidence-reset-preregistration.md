# Probe79: an observable-error trigger for obsolete rate memory

Date: 2026-10-02. Preregistered before implementation or new data.

## Rationale and bounded first screen

Probe78 passes all reset-ceiling clauses. Replace its hidden change cue with
a simple evidence-driven rule: two consecutive large, same-direction prediction
errors for the same bodily channel trigger full rate-evidence reset. The rule
is authored; neither a learned architecture nor calibrated uncertainty is claimed.

First run a **matched-history feasibility screen**, not a closed-loop survival
treatment. Retained memory drives every physical episode. Adaptive, muted and
sign-scrambled detectors see that exact observable history. They can reset
their own rate evidence, but cannot change its physical trajectory. Only if
all screen gates pass is a separate closed-loop preregistration warranted.

## Detector, fixed before seeing errors

At a public interoceptive reading, first advance the detector's observable
Jacobian through the actual transition. Compute reading minus the model's
clipped bodily prediction **before fitting that reading**. Use only channels
whose reading is strictly between 1e-6 and 1-1e-6, as in existing RLS. A channel
is surprising when absolute error exceeds **0.05 body units**, one quarter of
the fixed small portion. At two successive eligible readings, the same channel
must have surprising errors with the same sign. A small error, opposite sign,
or clipped reading clears that channel's streak. A trigger clears all rate
coefficients, covariance and cumulative burn statistics exactly as Probe78,
without changing the current-state estimator or revealing any new observation.
Fit the triggering reading normally after the reset, using its existing
observable-history Jacobian. Do not process a transition twice.

Clear streaks at episode births and after each reset; retain rate evidence
across births. No warmup, time-since-change cue, body identity metadata, true
constants, true state, future observation or known number of physical changes
enters the detector. The threshold and two-reading rule are not swept.

Controls: muted trigger (compute the same errors but never reset), and
sign-scrambled trigger (independent fair sign per eligible channel/reading,
preserving error magnitudes and reading times). The latter destroys directional
persistence, not the presence of a surprising magnitude, and is not assumed
in advance to fail. Every shadow estimator fits the real readings; corrupted
observations never enter its RLS fitting. The driver is the unchanged retained
memory. Include oracle-reset and true-rate readouts as reference instruments.

## Population and budget

Same physical distributions, frozen parent, change schedule family and one
calibration + five test episodes as Probe78. Fresh identities and episode
streams: main 4.4B/4.5B, diagnostic 4.6B/4.7B; stride 2,000,000 per block and
16 per identity's episodes. Five blocks x 28 identities, each paired stationary
and changed. Diagnostic one block x two identities, never graded. Python's
motor RNG accepts these integer seeds; no uint32 conversion is introduced.
Retain the original 0.03 readings and death truncation; no extra sensing or
training evidence. Source, parent, configuration and preregistration pin resume.

## Endpoints and locked screen gates

On retained physical histories, record each readout's rate MAE, matched
true-state word regret and word changes over all scored decisions, with an
after-change panel descriptive only. Every identity remains in the analysis.
Record detector resets, timing, and stable identity-level false alarms.
Survival is the common driver's survival and cannot establish detector benefit.

All clauses required:

1. On changed histories, adaptive lowers all-decision rate MAE and word regret
   versus retained with positive paired 95% lower bounds across five blocks
   (critical 2.776), >=4 positive blocks, and >=20% mean reduction for each.
2. On stable histories, adaptive mean rate MAE <=1.10x retained, mean regret
   increase <=0.001, and <=10% of identities have any adaptive reset. This
   explicit false-alarm budget prevents labelling indiscriminate relearning
   selective correction. These are feasibility bounds, not equivalence claims.
3. Muted rates and requests match retained exactly on every scored decision;
   adaptive's changed-history MAE and regret each beat sign-scrambled with
   positive paired 95% lower bounds and >=4 positive blocks. No required
   survival collapse is imposed on this matched-history control.

If any clause fails, close this fixed detector before a survival treatment.
Do not rescue it by changing the threshold, streak, budget, reporting only
changed bodies, or treating counterfactual word regret as observed survival.
A different evidence model would need a distinct argument and preregistration.

## Verification

Tests must show pre-fit scoring, one transition per update, false-alarm/streak
semantics, no simulator access, sign corruption only in detection, cue isolation,
episode-boundary handling, muted parity, source-pinned resume and tick-weighted
gates. Save atomic per-identity/cohort records and a compact evidence record.
Time the separate diagnostic before the locked run. No global configuration or
legacy learning path changes.
