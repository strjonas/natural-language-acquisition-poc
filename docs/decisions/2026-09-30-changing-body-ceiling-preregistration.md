# Probe78: does forgetting obsolete rate evidence have physical headroom?

Date: 2026-09-30. Preregistered before implementation or data.

## Purpose

Probe75 establishes persistent rate memory for stationary bodies. Before
building a learned change detector, test whether an oracle timing cue that
discards obsolete rate evidence can improve the same organism on its existing
reading stream. This is a headroom instrument, never learned reflection.

## Construction

Use Probe75's frozen parent, original help lag 24, help period six, independent
body multipliers in [0.4,1.6], 0.03 interoception, unlimited caregiver basket,
five test episodes after one shared calibration episode. Five independent
blocks x 28 identities. Each identity appears in paired stationary and changed
cohorts. The changed body redraws all four metabolic multipliers independently
from the same distribution once, then retains them through later episodes.
Randomize the change episode uniformly among test episodes 2,3,4 and the
nominal tick uniformly among 24,48,72. Apply at the first decision boundary at
or after that tick. If death prevents that boundary, the new constants apply
at the next episode birth. Do not remove these identities or condition survival
on witnessing a change. Stationary counterparts carry an identical sham schedule.

No true state is revealed by a change. Birth state, episode randomness, motor
hidden state, state/uptake estimator, movement estimate and request ledger
reset exactly as in Probe75. Motor actions remain from the frozen parent.
The rate change cue is visible only to the oracle-reset rate estimator and
audit; it is not supplied to state estimation, motor or request selection.

Every cell runs the same rate estimators and all five counterfactual request
readouts on its own physical history. Only the supplied request-rate vector
differs. Cells:

- retained: Probe75's persistent estimator and cumulative rate readout;
- episodic: discard rate evidence at every episode birth;
- forgetting: persistent RLS with forgetting factor 0.9 per nonempty reading;
  decay cumulative burn numerator/denominator at those readings too, so the
  readout forgets consistently (fixed a priori, no sweep);
- oracle_reset: retained estimator until an actual physical change; clear rate
  coefficients, covariance and cumulative burn evidence on the cue, preserving
  its observable-history Jacobian and every episodic state belief;
- true_rate: current true bodily rates, solely as an additional ceiling.

The oracle does not receive new constants, a body reading, or extra samples.
The first interval spanning the cue remains in observable history; no secret
state snap is allowed. At a birth cue, start from the public birth packet.
Evidence tick/reading counters remain cumulative and are not fabricated by
forgetting; store reset counts separately. Stable oracle_reset and retained
must be bit-identical under a sham schedule. More frequent forgetting is a
comparison, not a mechanism selected after viewing outcomes.

## Endpoints and continuation

Primary closed-loop endpoint: mean survival over all five test episodes,
paired across the five independent blocks. On retained-arm histories record
matched rate MAE, true-state word regret and word changes for every readout,
holding current state/uptake belief, ledger and movement estimate identical.
Report both all scored decisions and decisions after the scheduled physical
change, but gate only the all-decision endpoints to avoid post-survival selection.
Stationary and changed cohorts are reported separately; do not average them.

All three clauses must pass before building an evidence-driven reset detector:

1. True_rate minus retained survival in changed bodies has positive paired
   95% Student-t lower bound (df4, critical2.776) and >=4 positive blocks.
2. Oracle_reset minus retained survival satisfies the same criterion.
3. On retained-arm changed histories, oracle_reset lowers rate MAE and matched
   word regret with positive paired 95% lower bounds and >=4 positive blocks.

Stable oracle_reset must match retained exactly; otherwise the construction is
invalid, not negative. If a clause fails, close *cue-triggered complete reset
on this construction* as the immediate route. A true-rate ceiling cannot rescue
a failed reset ceiling. Simple forgetting and episodic arms are descriptive;
promoting one based on these outcomes needs an independently preregistered
confirmation. No adaptive mechanism or confidence report is claimed here.

If all clauses pass, separately preregister an evidence-driven detector and
its false-alarm control before implementing it. Its continuation must require
improvement after changes and preserved utility in stationary individuals.

## Independence, runtime, verification

Use 4.0B identity and 4.1B episode bands, block stride 2,000,000 and identity
episode stride16. Diagnostic uses 4.2B identity and 4.22B episode bands, one
block and two identities, disjoint from all scored data. Motor RNG is Python
Random; these bands are safe for inherited seed handling. Tagged SHA-256
domains separate physical change draws and timing from episode RNGs. Two
cohorts intentionally share episode randomness; five cells intentionally share
initial calibration and the predeclared schedule. Parent/source/configuration
hashes pin atomic resumable identity/cell units. Final gates require exactly
5 x 28 x 5; diagnostics never receive gates.

Tests: hidden-truth access denied to real learner updates, cue gives no new
state, reset clears all rate evidence, forgetting includes readout statistics,
change persists across death/birth, sham parity, constant parent, matched
readouts, seed isolation, summary weighting, gate binding and resume.
Time the separate diagnostic before estimating the full run. Persist compact
block evidence and a durable decision, with actual change exposure and evidence
counts. Existing stationary mechanisms and global configuration stay unchanged.
