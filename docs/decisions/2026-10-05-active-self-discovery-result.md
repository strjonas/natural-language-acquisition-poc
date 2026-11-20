# Probe80: a self-posterior discovers its own body while living in it — result

Date: 2026-10-05. Preregistration:
`2026-10-04-active-self-discovery-preregistration.md`. Direction:
`2026-10-04-self-in-the-action-loop-direction.md`. Mechanism:
`src/homesocial/organism/self_posterior.py`; runner:
`src/homesocial/organism/active_self.py`; independent checker:
`scripts/summarize_probe80.py`.

**All nine locked gates pass**, and an independent standard-library
reconstruction from the raw per-life records agrees with the runner on every
gate. Five blocks × 12 identities × K1–K5 = 300 identities, 1,800 lives per
arm, survey band 5,000,000,000, run once.

> When the organism's own self-model chooses what it does, an online posterior
> fed by two scalar sensations and its own actions discovers its actuator
> structure while alive (exactly, after six lives, in 100/100/93/67% of bodies
> with one to four hidden resources); its survival depends on that model beyond
> structure-free action values, beyond another body's evidence and beyond
> forgetting between lives; uncertainty about itself is what makes it discover;
> and its reported confidence about itself is calibrated because it checks
> itself against its own sensations.

## Survival (six-life mean, block-weighted)

| K | random | structure-free | **posterior** | uncritical | reset | certainty | yoked | true structure | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.436 | 0.969 | **0.978** | 0.983 | 0.864 | 0.978 | 0.225 | 1.000 | 1.000 |
| 2 | 0.164 | 0.417 | **0.864** | 0.861 | 0.661 | 0.756 | 0.047 | 0.906 | 0.906 |
| 3 | 0.047 | 0.039 | **0.675** | 0.656 | 0.469 | 0.583 | 0.006 | 0.747 | 0.747 |
| 4 | 0.017 | 0.006 | **0.472** | 0.353 | 0.297 | 0.336 | 0.000 | 0.589 | 0.594 |
| 5 | 0.006 | 0.000 | 0.272 | 0.169 | 0.158 | 0.253 | 0.003 | 0.558 | 0.572 |

The posterior closes **96, 94, 90 and 80%** of the gap between random action
and acting on its true structure at K1–K4 (48% at K5). Its survival rises from
the first to the sixth life as it learns itself: 0.88→1.00, 0.70→0.92,
0.50→0.77, 0.32→0.58, 0.18→0.32. Tracking state from the two scalars costs
nothing once structure is known: true-structure equals oracle within 0.014 at
every K.

## Discovery (exact actuator partition, after life 1 / after life 6)

| K | posterior | uncritical | reset | certainty | yoked (own body) | reported count, life 6 |
|---|---|---|---|---|---|---:|
| 1 | 0.85 / **1.00** | 0.83 / 1.00 | 0.87 / 0.90 | 0.00 / 0.00 | 0.02 / 0.02 | 1.00 |
| 2 | 0.67 / **1.00** | 0.67 / 1.00 | 0.60 / 0.67 | 0.00 / 0.02 | 0.00 / 0.00 | 2.00 |
| 3 | 0.57 / **0.93** | 0.53 / 0.87 | 0.47 / 0.48 | 0.00 / 0.03 | 0.00 / 0.00 | 3.00 |
| 4 | 0.32 / **0.67** | 0.15 / 0.47 | 0.28 / 0.20 | 0.00 / 0.03 | 0.00 / 0.00 | 3.98 |
| 5 | 0.10 / 0.25 | 0.10 / 0.18 | 0.10 / 0.10 | 0.02 / 0.05 | 0.00 / 0.00 | 4.98 |

Probe77 recovered 4–36% at K1 and 0–6% at K2–K5 from the same lived budget with
uniform random exploration. The difference is who chooses the actions: a
self-model that explores because it is uncertain about itself.

## Locked gates

Intervals: paired over the five blocks, Student-t critical 2.776.

| gate | requirement | observed | verdict |
|---|---|---|---|
| G1 discovery | exact after life 6 ≥ 0.80 at K1–K3, ≥ 0.40 at K4; count strictly increasing K1–K5 | 1.00, 1.00, 0.93, 0.67; counts 1.00 < 2.00 < 3.00 < 3.98 < 4.98 | **pass** |
| G2 action uses the self-model | `posterior − random` survival positive at each K1–K4 | +0.542 [+0.481, +0.603]; +0.700 [+0.615, +0.785]; +0.628 [+0.509, +0.747]; +0.456 [+0.366, +0.546]; all 5/5 | **pass** |
| G3 structure, not action values | `posterior − structure-free` positive at K2–K4 | +0.447 [+0.338, +0.557]; +0.636 [+0.530, +0.742]; +0.467 [+0.400, +0.533]; all 5/5 | **pass** |
| G4 its own body | `posterior − yoked` K1–K4 positive; yoked exact on own body ≤ 0.10 | +0.678 [+0.634, +0.722], 5/5; 0.004 | **pass** |
| G5 uncertainty drives discovery | `posterior − certainty` exact after life 6, K1–K4: positive, ≥ 0.30 | +0.879 [+0.823, +0.935], 5/5 | **pass** |
| G6 persistence | `posterior − reset` survival lives 2–6, K1–K4, positive | +0.202 [+0.151, +0.253], 5/5 | **pass** |
| G7 calibrated self-report | K1–K4: Brier ≤ 0.10; accuracy ≥ 0.90 at confidence ≥ 0.9; ≤ 0.10 at ≤ 0.1 | 0.0526; 0.965 (1,161 reports); 0.017 (116) | **pass** |
| G8 self-criticism load-bearing | `uncritical − posterior` Brier, K1–K4, positive | +0.0748 [+0.0440, +0.1056], 5/5 | **pass** |
| G9 relabeling | 100 offline replays transport the partition exactly | 100/100 | **pass** |

## Descriptive, as preregistered

- **K1 tie with the structure-free learner, as predicted**: +0.008 [−0.027,
  +0.044]. With one hidden resource, knowing which actuator raises the minimum
  is enough. With two or more the structure-free learner collapses (0.417, 0.039,
  0.006) because the right action depends on *which* hidden variable is lowest.
- **Other-body evidence is worse than none**: yoked survives below random at
  every K ≥ 2 and recovers 0.4% of its own structure.
- **Self-criticism, beyond calibration**: discovery +0.067 [+0.038, +0.095] 5/5
  (K1–K4 exact after life 6); survival +0.034 [−0.0005, +0.069] with all five
  blocks positive but the lower bound below zero — **not established**. It
  matters most where the self-model is weakest: Brier at K4 0.142 against
  0.390 uncritical, at K5 0.288 against 0.535. Rebuilds by life 6 average 0,
  0, 0.33, 1.08, 4.77 at K1–K5; falsified (confidence-withdrawn) reports are
  0, 0, 1.7, 4.7 and 26% of reports.
- **Acting on uncertainty also bought survival here**: `posterior − certainty`
  survival K1–K4 +0.084 [+0.057, +0.111], 5/5. The development band (20 bodies
  per K) showed only +0.028 with an interval spanning zero; this larger sample
  resolves it. The certainty lesion still discovers almost nothing (≤ 0.05).
- **K5 is the stress condition it was declared to be**: discovery 0.25,
  calibration Brier 0.288 (accuracy 0.39 at confidence ≥ 0.9), survival 0.272
  against 0.558 with true structure, though still above random and
  structure-free on 5/5 blocks (+0.267, +0.272).

## Execution record

- One survey run. No gate, threshold, particle count, falsification threshold,
  arm or sample changed after the preregistration was written; it was edited
  once before launch, to record the timed diagnostic's runtime.
- The machine slept for about six hours mid-run (workers accumulated ~20
  minutes of CPU over that wall time). Units resumed; nothing was rerun.
- The 10 units not yet started were rescheduled from an eight- to a
  ten-worker pool; idle sleep was prevented until the run finished.
- The last two K5 units were long poles (some identities took 15–27 minutes:
  repeated self-criticism rebuilds in the certainty and yoked lesions). Their
  remaining identities were computed in parallel processes with the runner's
  own code and seeds and spliced in identity order. Five identities computed
  both sequentially and in parallel (block 3 K5 identities 3–4, block 4 K5
  9–11) were **bit-identical** apart from wall-clock seconds; the splice refuses
  any mismatch, manifest difference or gap. The interrupted sequential process
  is recorded as exit 143.
- Compute: 77,662 identity-seconds (~21.6 CPU hours) for 300 identities × 9
  arms × 6 lives plus 100 permutation replays.
- The descriptive "worst tick evidence" field jumps on rebuild ticks (a rebuild
  replaces the evidence total); it enters no gate.
- Full regression after adding the modules: 632 tests and 13 subtests pass.

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.active_self --mode survey --grade
python3 scripts/summarize_probe80.py --mode survey
```

Raw unit files, logs and the splice record are in
`runs/organism/probe80_active_self/` (ignored); these numbers are the durable
record.

## What this establishes, against the five properties

1. **Discovered** — advanced. The count of hidden resources and which opaque
   actuator restores which are inferred online from two scalars, without forks,
   with shocks and death, and track a moved ground truth across K1–K4. Still
   within an authored family (bounded, linearly depleting, saturating
   resources) and with the sensor width given.
2. **Corrigible** — unchanged here: the body is stationary. The self-criticism
   that Probe80 validates is the mechanism Probe81 will test under change.
3. **Load-bearing across uses** — advanced substantially in this substrate:
   one posterior chooses every action, predicts every sensation and produces
   the report; lesioning its structure (structure-free), its evidence (yoked),
   its memory (reset), its use of uncertainty (certainty) or its self-checking
   (uncritical) each costs a pre-named endpoint. Still nonspatial; the island
   organism's motor parent is untouched.
4. **Productive under novel demand** — not addressed. No language in Probe80.
5. **Reflexive** — advanced. Its confidence about its own structure is
   calibrated (Brier 0.053), and this is caused by acting on a fact about its
   own model — that its own sensations were impossible under it — not by a
   property of the body; the lesion loses it (0.127).

## What it does not show

- No language, no listener, nothing said aloud; the "report" is a readout scored
  against truth. Words enter at Probe82.
- No changing body; self-criticism under change is Probe81's question.
- Not spatial and not the island organism; the motor here is actuator choice.
- Discovery is of reachable restorative structure within an authored family.
  Unreachable resources, nonlinear dynamics and a body outside the family are
  untested; self-criticism checks consistency with its own model class only.
- K5 is not solved: every actuator active, discovery 0.25, overconfident.
- Self-criticism's survival effect is not established (lower bound −0.0005).
- No consciousness or sentience claim.
