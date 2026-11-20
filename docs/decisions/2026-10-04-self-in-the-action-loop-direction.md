# The self belongs in the action loop: direction after Probes 57–79

Date: 2026-10-04. Status: **direction proposal with exploratory pilots.** The
pilots below are development evidence on seed bands disjoint from every locked
test (probe77-tagged bands 7.0M–9.2M and probe80's development band 5.2B).
They chose the mechanism; they are not results. The first locked test is
Probe80 (`2026-10-04-active-self-discovery-preregistration.md`). This document
does not revise any closed line and does not reopen Probe61's latent pruning.

**Update 2026-10-05: Probe80 passed all nine locked gates** on 300 survey
identities, independently reconstructed
(`2026-10-05-active-self-discovery-result.md`). Exact self-structure after six
lives 1.00/1.00/0.93/0.67 at K1–K4; calibrated self-report (Brier 0.053);
survival over random, structure-free, other-body and reset controls on 5/5
blocks. The pilots below remain development evidence.

## 1. Diagnosis: why twenty probes bought a few survival points each

Since Probe57 the self-model has chosen exactly one thing: which of three to six
request words to say. Survival is decided by a frozen motor parent and a fixed
caregiver clock. Every self-model property has therefore had to be detected as
a change in how often a *word* changes a *grant*:

| probe | what the self-model changed | survival effect |
|---|---|---:|
| 73 | learned vs species rates in the request rule | +0.0143 [+0.0003, +0.0283] |
| 74 | retained vs reset rate evidence | +0.0300 [+0.0141, +0.0459] |
| 75 | same, independent confirmation | +0.0357 [+0.0031, +0.0683] |
| 78 | oracle-timed forgetting after hidden change | +0.0257 [+0.0026, +0.0488] |

These are real and replicated, and they sit at the edge of five-block
significance. The ceiling is set by the fraction of survival a word controls,
not by the quality of the self-model. Probe64 said it outright: *the world cannot
hear the difference.*

**Pilot A** measures the same quantity — the survival value of knowing your own
body — in Probe76/77's body family when the self-model chooses the *action*
instead of a word. Ten opaque actuators, K hidden bounded resources, mean/min
sensations, shocks, one action every six ticks, 400-tick lives, 100 bodies per K:

| hidden resources K | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| uniform actuator choice (Probe77's policy) | 0.48 | 0.10 | 0.01 | 0.00 | 0.00 |
| knows its structure and state (oracle) | 1.00 | 0.94 | 0.72 | 0.68 | 0.59 |

Knowing yourself is worth **0.5–0.9 survival** when the self-model acts, versus
0.01–0.04 when it chooses a word. That is the largest lever in this repository,
and it was unused because the self-model was never given the motor.

Moving it there also changes which properties are *necessary*. A self-model that
only picks a word never has to discover its structure, never pays for being
uncertain and never needs to notice that it is wrong. One that acts must do all
three or die. The three placeholders named in the 2026-08-18 audit — authored
template, argmax mouth, six-meaning world — are downstream of that one choice.

## 2. The mechanism family

One online **self-posterior** (`src/homesocial/organism/self_posterior.py`):

- **Discrete self-structure is sampled**: which of the sensor channels are bodily
  resources, and which resource each opaque actuator restores (assigned at the
  actuator's first use from the locally optimal proposal).
- **Continuous self-parameters are exact Gaussians** per particle — levels,
  depletion rates, portion sizes, shock size — updated by Kalman steps under
  sampled modes. Probe63's RLS calibrator is the special case with the structure
  fixed; this is its generalization to a structure that is not given.
- **Implied inequalities enter by moment-matched truncation**: every other
  resource lies above the observed minimum, a saturating grant filled its
  resource, levels never exceed one. Their probabilities enter the evidence.
- **Action by posterior sampling**: imagine one self consistent with the
  evidence, including what untried actuators might do, and act as that self
  would. Uncertainty about the self is what makes it explore.
- **Reports**: the posterior mode of its own actuator partition with a
  confidence that includes what it has not yet tested, and its running log
  predictive probability of its own sensations.
- **Self-criticism**: when one tick's sensations are impossible under the
  whole posterior (below −30 nats), the self-model is falsified; it withdraws
  its confidence and rebuilds by replaying its remembered experience (Section 3).

Its inputs are the actuator it chose and two scalars. It is told that the sum
sensation covers five channels (as Probes 76–77 publicly fixed the mean's
denominator) and that resources deplete, saturate and are shocked. It is not
told how many resources exist, which actuator does what, how much, how fast, or
where anything is.

## 3. Pilot evidence (exploratory, development seeds)

### Discovery while alive, without forks (Pilot B, 10 bodies per K, six lives)

| K | first-life survival | six-life survival | exact partition after life 1 | after life 6 | Probe77 random, life 1 / 6 |
|---|---:|---:|---:|---:|---|
| 1 | 0.90 | 0.98 | 0.90 | 1.00 | 0.04 / 0.36 |
| 2 | 0.80 | 0.87 | 0.80 | 1.00 | 0.00 / 0.06 |
| 3 | 0.20 | 0.58 | 0.20 | 0.90 | 0.00 / 0.01 |
| 4 | 0.30 | 0.38 | 0.30 | 0.60 | 0.00 / 0.00 |
| 5 | 0.00 | 0.10 | 0.10 | 0.10 | 0.00 / 0.00 |

Probe77 closed *random exploration plus predecessor contrasts*; it explicitly
left directed exploration open. A posterior that explores because it is
uncertain recovers the exact structure where random exploration recovered
almost nothing, at the same lived budget, with shocks and death.

### What the self-model is load-bearing for (Pilot C, 5 bodies per K; Pilot bandit, 20)

Six-life mean survival, paired bodies:

| K | random | structure-free learner | posterior | reset each life | other body's evidence (yoked) | true structure, tracked state | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.20 | 0.97 | 0.97 | 0.80 | 0.13 | 1.00 | 1.00 |
| 2 | 0.00 | 0.40 | 0.87 | 0.57 | 0.00 | 0.93 | 0.93 |
| 3 | 0.03 | 0.02 | 0.53 | 0.53 | 0.00 | 0.57 | 0.57 |
| 4 | 0.00 | 0.00 | 0.50 | 0.33 | 0.00 | 0.83 | 0.83 |
| 5 | 0.00 | 0.00 | 0.17 | 0.10 | 0.00 | 0.47 | 0.47 |

(The structure-free learner column is from the 20-body bandit pilot on a
different band; random/oracle there were 0.50/0.17/0.01/0.02/0.00 and
1.00/0.96/0.82/0.77/0.53.)

Four observations shaped Probe80:

1. **Tracking state from two scalars is free once structure is known**: the
   true-structure arm equals the oracle at every K. The whole remaining gap is
   the cost of discovering structure.
2. **Structure matters exactly when there is structure.** With one hidden
   resource, learning which actuator raises the minimum is enough (0.97). With
   two or more it collapses (0.40, 0.02, 0, 0), because the right action depends
   on *which* hidden variable is lowest — a fact only a model with separate
   variables can hold.
3. **A self-model of someone else is worse than none.** Fed a donor body's
   sensations under its own actions, the agent dies more often than random.
4. **Acting on uncertainty is what discovers.** A first certainty-equivalent
   lesion was unfair (it fell through to restoring a non-urgent resource instead
   of exploring); corrected, it still never recovers its full partition at K1
   while surviving (0.93 vs 0.97). Probe80 locks the corrected lesion.

### Calibration and the inference failures found on the way (Pilots D–H)

The posterior's confidence in its own full structure is calibrated at K1–K4 and
was overconfident at K5. Three development findings, each kept as a standing
lesson rather than tuned away:

- **Static structure collapses under resampling.** Wrong, confident partitions
  at K5 come from particles losing independent ambiguities together. More
  particles help (K4 exact 0.23 → 0.70 at N=400 → 1000); a structural change
  hazard (0.02, 0.05) and island SMC (4 × 250) did **not** help and are not
  used.
- **Inequalities must be in the filter.** Without truncation, a wrong merged
  structure out-scored the true one by ~5 nats of evidence on recorded K5 lives;
  with truncation the truth scores at or above the mode. Truncation is in.
- **Surprise is a falsification signal, but not yet a clean one.** A collapsed
  posterior's own predictive log-probability falls to values impossible under
  its noise model (worst ticks −6,000 to −235,000) while it reports confidence
  1.0. But a filter *given the true structure* also collapses on some K5 lives
  (shock attribution), so self-surprise cannot yet serve as a false-alarm-free
  change detector. Moment-matching the shock attribution did not fix it.

At N=1000, development calibration (Pilot H, three lives × 10 bodies): K3 exact
0.73 / confidence 0.77 / Brier 0.012; K4 0.70 / 0.68 / 0.018; reports at
confidence ≥ 0.9 were correct 40/40 at K3–K4. K5: 0.20 / 0.44 / 0.190.

### Probe80's development band (5,200,000,000; 20 bodies per K, six lives)

With every arm in place and the corrected certainty lesion, before
self-criticism:

| K | random | structure-free | posterior | reset | certainty | yoked | true structure | oracle | exact, life 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.38 | 0.96 | 0.99 | 0.89 | 0.99 | 0.21 | 1.00 | 1.00 | 1.00 |
| 2 | 0.18 | 0.43 | 0.88 | 0.72 | 0.83 | 0.03 | 0.96 | 0.96 | 1.00 |
| 3 | 0.02 | 0.03 | 0.58 | 0.50 | 0.57 | 0.00 | 0.68 | 0.68 | 0.85 |
| 4 | 0.00 | 0.00 | 0.35 | 0.33 | 0.40 | 0.01 | 0.61 | 0.61 | 0.35 |
| 5 | 0.00 | 0.00 | 0.22 | 0.14 | 0.09 | 0.00 | 0.54 | 0.53 | 0.30 |

A nuance the pilots hid: the certainty lesion survives about as well as the
posterior at K1–K4 while recovering almost none of its structure (+0.78 exact
for the posterior). Forced, undirected tries find *a* working actuator for the
urgent resource without mapping the self. Uncertainty-driven exploration is
what *discovers*; in this body it is not what keeps the organism alive.

The calibration grade failed (K1–K4 Brier 0.155; confidence 0.90 against
accuracy 0.75). The diagnosis: **81 of 103 confidently wrong reports came from
lives in which the posterior's own sensations had been impossible under it
(below −30 nats on one tick), against 2 of 390 correct reports.** The organism
had the information that its self-model was broken and did not use it.

**Self-criticism** uses it: a falsified self-model withdraws its confidence
and re-interprets its remembered experience from the prior, keeping the
best-evidenced rebuild. Re-measured on the same bodies:

| K | survival | exact, life 6 | Brier |
|---|---|---|---:|
| 1 | 0.99 → 0.99 | 1.00 → 1.00 | 0.000 |
| 2 | 0.88 → 0.88 | 1.00 → 1.00 | 0.007 |
| 3 | 0.58 → 0.63 | 0.85 → 0.90 | 0.078 |
| 4 | 0.35 → 0.45 | 0.35 → 0.50 | 0.175 |
| 5 | 0.22 → 0.32 | 0.30 → 0.55 | 0.096 |

K1–K4 pooled Brier 0.155 → 0.065; reports at confidence ≥ 0.9 correct 94.5%
(381). Paired over blocks, self-criticism improves calibration on 5/5 blocks
(+0.090, lower bound +0.028); its gains in discovery and survival are positive
on average with intervals spanning zero. Probe80 gates only the calibration
claim. This is the first mechanism in the repository that acts on a fact about
its own model rather than about the body — Property 5 — and Probe80 tests it
against its lesion.

## 4. The ladder this opens

Each step keeps this repository's method: preregistration, locked gates,
destructive controls, five blocks.

| probe | question | property |
|---|---|---|
| **80** | Does a self-posterior discover its own actuator structure while staying alive, and does survival, discovery and calibrated self-report depend on it? | 1, 3, 5 |
| 81 | After a hidden remapping or rate change, does the same posterior notice and relearn — with stationary false alarms controlled — where Probe79's threshold failed? Requires first solving the true-structure collapse in Section 3. | 2 |
| 82 | **Social actuators.** Some resources can be restored only by a caregiver who acts on words. Words become causes the same posterior learns over *discovered* variables; a caregiver who can see the body names a resource the agent discovered but cannot restore. Scrambled-label and wrong-listener controls. | 4, 5 |
| 83 | **Productive and future speech.** A compositional caregiver protocol (which × how much × when), held-out combinations, and Probe65's counterfactual-branch test for "I will need" — over discovered variables. "I don't know" buys a caregiver check, so uncertainty is something said that has a consequence. | 4, 5 |
| 84 | Integration into the spatial organism and independent developmental parents. | 3 |

Steps 82–83 are where this line meets the original plan: caregivers teach words,
but the words now attach to variables the agent found in itself rather than to
axes a designer wrote. The 2026-08-18 audit's rule still binds: the vocabulary
an agent can acquire is bounded by the listener's action space, so 82–83 must
grow the listener's consequences, not the lexicon.

### Pilots toward Probes 81 and 82 (development, very small, not evidence)

**Change (Probe81).** Probe80's mechanism plus one inference: when no rebuild
of the whole remembered past survives its own replay, conclude that the body
changed and rebuild from the current life only. Development band, K2–K3, four
bodies with a stationary twin each; at a scheduled tick of life three the
targets of two large actuators are swapped:

| after the hidden change | exact structure after life 6 | confidence after the change | stationary twins concluding "changed" |
|---|---:|---|---:|
| uncritical | 0/4 | 1.00 — confidently wrong | — |
| self-critical, keeps its best model | 0/4 | 0.00 — knows, cannot repair (11 rebuilds) | 0/4 |
| self-critical, concludes the body changed | **4/4** | recovers to 0.74–1.00 in 1–3 lives | **0/4** |

Probe79's threshold detector reset 17% of stable bodies. Two design lessons:
Probe76/77's remap moves a *small* alias, which a self-model preferring large
portions never uses again, so no sensation ever contradicts it — change
*detection* needs a change that touches what the organism does; knowledge about
unused parts of itself going *stale* is a separate property (a change hazard
whose confidence decays with time since last use). And rebuild cost grows with
remembered history: the falsified-but-unrepaired arm rebuilt 11 times.

**Words for a self it discovered (Probe82).** One discovered resource has no
actuator the organism controls; a caregiver restores any resource when asked
with the right word, under an unknown convention (six words, one empty). The
posterior treats words as social actuators; a caregiver who can see the body
sometimes names the lowest resource, and the posterior treats the name as
evidence linking the word to whichever *of its own discovered variables* it
believes is lowest. Eight bodies per K, six lives:

| K2 | survival | first-life survival | knows the unreachable need's word after life 1 | report fidelity |
|---|---:|---:|---:|---:|
| labels (true lowest) | **0.85** | **0.62** | **0.62** | **0.97** |
| silent (trial and error) | 0.67 | 0.12 | 0.12 | 0.76 |
| labels naming a random *active* resource | 0.81 | 0.50 | 0.75 | 0.91 |
| oracle | 0.90 | 0.88 | — | — |

Report fidelity: asked "what do you need?", the organism answers with the
caregiver's word for its truly most urgent resource. K3 points the same way,
smaller (labels 0.69 survival / 0.91 fidelity; silent 0.67 / 0.81). By life six
every arm knows the word for the need it cannot meet itself.

The pilot's own control was flawed and that is its most useful result: naming a
random *active* resource still tells the organism which words matter for this
body (two of six at K2), which is why it helps almost as much as true labels.
Probe82 must scramble reference while preserving vocabulary statistics, and add
a labels-without-self-model arm (a learner that hears names but has no
discovered variables to attach them to).

## 5. What this is not, and the risks

- **The family is authored.** Bounded resources that deplete linearly and
  saturate under restoration. Discovered: count, actuator partition, sizes,
  rates, levels. A misspecified-body arm belongs in a later probe.
- **The sensor width (five channels) is given**, as it was in Probes 76–77.
- **Nonspatial.** The motor here is actuator choice. Probe84 must put it back in
  a body that moves.
- **K5 is not solved.** Every actuator is active, the oracle survives about half
  the time, and the inference is overconfident there. Probe80 preregisters K5
  as a descriptive stress condition, not a pass.
- **Survival here measures a policy too.** Every comparison in Probe80 is paired
  and changes only the self-model or how it is used; the oracle and
  true-structure arms bound what the action rule can reach.
- No consciousness, sentience or language claim. Language enters at Probe82.
