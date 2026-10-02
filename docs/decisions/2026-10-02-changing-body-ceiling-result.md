# Probe78: correctly timed forgetting has physical value after bodily change

Recorded: 2026-10-02; run began 2026-09-30 and finished before this review.
Preregistration: `2026-09-30-changing-body-ceiling-preregistration.md`.
Module: `src/homesocial/organism/changing_body.py`.

**All three locked continuation clauses pass** (four scored contrasts).
An oracle timing cue that discards obsolete rate evidence adds **18 survivors
in 700 episodes** after hidden changes: **+0.025714 [+0.002590, +0.048838]**,
four positive seed blocks and one tie. The cue supplies neither new constants
nor a state reading. Stationary oracle-reset and retained episodes are exactly
identical, including their counterfactual scores.

This licenses an evidence-driven detector experiment. It does not demonstrate
that the organism recognizes a change, estimates staleness, or reports uncertainty.

## Physical endpoints

Five paired blocks x 28 identities x five test episodes per cell/cohort.
Stationary and changed cohorts share identity, calibration and episode streams.
All 140 calibration episodes are excluded from survival endpoints.

| rate readout | stationary survivors / 700 | changed survivors / 700 | changed mean life ticks |
|---|---:|---:|---:|
| retain evidence | 155 | 149 | 138.02 |
| reset each episode | 132 | 147 | 133.61 |
| forget gradually (0.9 per reading) | 156 | 156 | 140.69 |
| **oracle-timed full reset** | **155** | **167** | **148.22** |
| current true rates | 153 | 173 | 151.99 |

Changed retained survivors by block: **28, 30, 30, 34, 27**; oracle reset:
**35, 35, 33, 34, 30**. The primary endpoint includes every test episode,
including episodes before change and those that die before its nominal tick.

| locked contrast | mean [paired 95% interval] | positive blocks | verdict |
|---|---|---:|---|
| true rates minus retained survival | +0.034286 [+0.008893, +0.059679] | 4/5, one tie | pass |
| oracle reset minus retained survival | +0.025714 [+0.002590, +0.048838] | 4/5, one tie | pass |
| retained minus oracle-reset rate MAE, matched history | +0.001508 [+0.000886, +0.002130] | 5/5 | pass |
| retained minus oracle-reset word regret, matched history | +0.006777 [+0.004565, +0.008989] | 5/5 | pass |

Intervals use the five paired blocks (Student-t critical 2.776), not 700
independent episode replicates. The primary survival lower bound is narrow.
The gradual-forgetting and episodic arms are descriptive; neither is promoted
to a confirmed adaptation treatment by selecting it after these outcomes.

## Matched evidence and change exposure

On retained-arm histories, all-decision rate MAE falls **0.002249 → 0.000741**,
and word regret **0.012939 → 0.006162**. Replacing only the rate readout changes
**11.01%** of scored request words. In the descriptive after-change panel,
rate MAE falls **0.003515 → 0.001009**, regret **0.019355 → 0.007944**, and
**18.50%** of words change. State/uptake belief, movement estimate, request
ledger and decision time are held fixed within these comparisons.

In every cell, **62/140** bodies encounter their physical change and cue
within the scheduled episode. **78/140** die before that boundary and receive
their new constants and cue at the next birth. These identities remain in all
endpoints. The design therefore mixes within-life and between-episode changes;
it is not solely a damage-recovery experiment within uninterrupted lives.

The reset clears coefficients, covariance and cumulative burn statistics,
while preserving the observable-history Jacobian, current-state estimator and
actual evidence counters. An interval spanning the cue is not secretly snapped
to truth. The oracle is an instrument for the value of correctly timed resets,
not a mathematical upper bound on every possible adaptation algorithm.

## Verification and reproducibility

All **1,540** progress units completed: 140 calibration episodes and 1,400
identity/cohort/cell units containing 7,000 test episodes. Recorded runtime:
**2,195.76 seconds (36.6 minutes)**, including checkpoint writes and contention
from the full regression suite. The runtime exceeded the initial 35-minute
upper estimate while this chat was interrupted; the run itself completed.

`scripts/summarize_probe78.py` imports only the standard library. It reconstructs
the survival and tick-weighted score endpoints, paired intervals and every gate
from raw episode records; verifies all episode seeds, calibration and unit
counts; checks stationary sham parity and equality before physical changes;
and agrees with the saved summary. Compact block evidence reproduces the same
numbers without the raw episode records. Its limits are explicit: individual
seed and episode checks were performed during extraction, not reconstructed
from omitted records.

The completed implementation passed **602 tests plus 13 subtests** repository
wide. Focused independent-checker tests passed again after adding pre-change
cohort parity. Real interruption/resume matches uninterrupted episode records.
No legacy model or training configuration changed.

```bash
python3 scripts/summarize_probe78.py
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.changing_body \
  --out /tmp/probe78-survey.json
```

The raw artifact and source/parent/preregistration fingerprints are preserved
locally; `runs/organism/probe78_changing_body/evidence.json` is the public compact
record. New source versions require a fresh output path.

## Decision

The stationary-memory result is now challenged in a regime where retaining
obsolete evidence is physically costly and timely forgetting helps. The next
authorized step is to test detection from observable prediction errors, with
stationary false-alarm controls. It must not receive the simulator's change cue.
Structure discovery, motor use of the self-model, generative language, and
independent parent development remain separate gaps.
