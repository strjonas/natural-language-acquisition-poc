# Probe80: a self-posterior discovers its own body while living in it

Date: 2026-10-05. Status: **preregistered before any run on the survey band
(5,000,000,000).** Direction and pilot record:
`2026-10-04-self-in-the-action-loop-direction.md`. Mechanism:
`src/homesocial/organism/self_posterior.py`. Runner:
`src/homesocial/organism/active_self.py`.

## Development history, disclosed

The mechanism and these gates were developed on seed bands disjoint from the
survey: probe77-tagged pilot bands 7.0M–9.2M and Probe80's development band
5,200,000,000 (100 identities, recorded in
`runs/organism/probe80_active_self/development_*`). In that order:

1. A plain particle filter with jittered continuous parameters lost the true
   hypothesis at saturation events. Replaced by Rao-Blackwellized Gaussians.
2. The first certainty-equivalent lesion was unfair (it restored a non-urgent
   resource instead of exploring). Corrected before any development summary
   was graded.
3. Without inequality truncation a wrong merged structure out-scored the true
   one on recorded lives. Truncation added.
4. Rejected after testing: structural change hazard 0.02/0.05, island SMC,
   moment-matched shock attribution, sensor sigma 0.006, rarer resampling.
   Kept: N = 1000 particles (N = 400 failed at K4).
5. The development grade of the uncritical mechanism failed calibration
   (K1–K4 Brier 0.155; confidence 0.90 against accuracy 0.75): 81 of 103
   confidently wrong reports came from lives in which the posterior's own
   sensations had been impossible under it, against 2 of 390 correct reports.
   Self-criticism (below) was added, and re-measured on the same development
   bodies (Brier 0.065). It is therefore a development-motivated mechanism; the
   survey is its first independent test, and G8 tests it against its lesion.

No survey-band seed has been run. The 5,100,000,000 diagnostic band (one
identity per K) is timed before the survey and never enters a gate.

## Question

When the self-model, not a frozen motor parent, chooses what the organism does,
does an online posterior over the organism's own unnamed structure:

1. discover that structure while alive, without forks;
2. become the thing survival depends on — beyond action values, beyond another
   body's evidence, beyond forgetting between lives;
3. explore *because* it is uncertain about itself;
4. know how sure it is, and know when its model of itself is broken?

## Mechanism (locked)

`SelfPosterior` with `PosteriorConfig(particles=1000, hazard=0.0,
sensor_sigma=0.003, shock_hazard=0.025, ess_fraction=0.5,
prior_level=(0.6,0.2), prior_rate=(0.012,0.01), prior_portion=(0.45,0.3),
prior_shock=(0.3,0.15), shock_attribution="sample", falsification_nats=30,
rebuild_attempts=3)`.

- Particles sample which of the five sensor channels are resources and which
  channel each actuator drives (lazily, at first use, locally optimal
  proposal). Each particle carries an exact Gaussian over levels, rates,
  portion sizes and shock size, updated by Kalman steps under sampled
  saturation, shock and minimum modes, with moment-matched truncation for the
  inequalities the sensations imply.
- **Action** — posterior sampling: draw one self (structure, parameters,
  levels, imagined effects of untried actuators) and restore the resource that
  self would lose first with the largest portion it believes available.
- **Self-criticism** — if one tick's sensations have log predictive
  probability below −30 nats under the whole posterior, the self-model is
  falsified. It replays its remembered experience from the prior three times
  and keeps the best-evidenced rebuild not itself falsified; else it keeps the
  best and marks itself falsified, reporting confidence 0 until a later rebuild
  succeeds (attempted at the next birth). At most two in-life rebuilds per life.
- **Report** after every life: the posterior mode of the full actuator
  partition, its probability (untried actuators imagined from the prior), the
  mode resource count, and the falsified flag.

Inputs: only the organism's own actuator choice and two scalars per tick, the
sum and the minimum of five channel levels. It is told the sensor covers five
channels and that resources deplete, saturate and are shocked; nothing else.

## Substrate and sample

Probe77's lived substrate unchanged: Probe76 body family via
`make_physical_world`, ten opaque actuators (each resource has one 0.20 and one
0.60 actuator; inactive targets are inert), K1–K5, birth levels
0.35–0.85, individual rates ×U[0.4,1.6] per identity, shocks 0.025/tick of 0.25
on a uniform slot (inactive slots unaffected), one actuator every six ticks
absorbed next tick, 400-tick lives, six lives per persistent identity with new
births and shocks.

Five blocks × 12 identities × K1–K5 = **300 identities**, 1,800 lives per arm.
Blocks are the replication units; intervals are paired over blocks, Student-t
critical 2.776. Identities within a block are not replicates.

## Arms (paired: same identities, births, rates, shocks)

| arm | what differs from the treatment |
|---|---|
| `random` | Probe77's uniform actuator choice; no self-model |
| `bandit` | structure-free learner: optimistic action values on the change in the minimum; ε = 0.1 |
| **`posterior`** | **treatment** |
| `uncritical` | self-criticism disabled (falsification 0) |
| `reset` | posterior re-initialized at every birth |
| `certainty` | acts on its single most probable self; untried actuators chosen uniformly only when no known actuator restores the most urgent believed resource |
| `yoked` | senses a donor body (independent map, rates, births, shocks) that receives the same actions; lives in its own body |
| `oracle_structure` | every particle given true structure, sizes, rates, shock size; tracks levels from the same sensations; self-criticism off |
| `oracle` | acts on true state and structure |

## Locked gates

K1–K4 are the claim. K5 (every actuator active; development oracle survival
0.53) is a stress condition reported descriptively, except where a gate names
it. Development values are the 100-identity development band (gates G1, G7, G8
from the self-critical re-measurement; the rest from the uncritical grade, with
self-criticism expected not to reduce them).

| gate | requirement | development |
|---|---|---|
| G1 discovery | exact partition after life 6 ≥ 0.80 at each of K1, K2, K3 and ≥ 0.40 at K4; mean reported resource count after life 6 strictly increasing K1 < … < K5 | 1.00, 1.00, 0.90, 0.50; counts 1.00, 2.00, 3.05, 4.00, 4.90 |
| G2 the action uses the self-model | `posterior − random` six-life survival positive at each of K1–K4 | +0.62, +0.70, +0.56, +0.35; 5/5 each |
| G3 structure, not action values | `posterior − bandit` survival positive at each of K2–K4 | +0.45, +0.55, +0.35; 5/5 each |
| G4 its own body | `posterior − yoked` survival positive, K1–K4 pooled; yoked exact partition of its own body after life 6 ≤ 0.10 | +0.56; 0.01 |
| G5 uncertainty drives discovery | `posterior − certainty` exact partition after life 6, K1–K4 pooled: positive and mean ≥ 0.30 | +0.78 |
| G6 persistence | `posterior − reset` survival over lives 2–6, K1–K4 pooled, positive | +0.11 [+0.06, +0.16] |
| G7 calibrated self-report | posterior end-of-life reports, K1–K4 pooled: Brier ≤ 0.10; accuracy ≥ 0.90 among confidence ≥ 0.9; accuracy ≤ 0.10 among confidence ≤ 0.1 | 0.065; 0.945 (381); 0.000 (63) |
| G8 self-criticism is load-bearing | `uncritical − posterior` Brier, K1–K4 pooled, positive | +0.090 [+0.028, …], 5/5 |
| G9 relabeling | offline replay of the posterior arm's own history with all actuator IDs consistently relabeled transports the reported partition exactly, for the first four identities of every block × K (100 controls) | 100/100 (development) |

"Positive" means paired mean > 0, lower 95% bound > 0 and at least four of five
blocks positive.

**All nine must pass** for the claim: *an online self-posterior, fed by two
scalars and its own actions, discovers its own actuator structure while alive
at K ≤ 4, its survival depends on it beyond action values, another body's
evidence and cross-life forgetting, uncertainty drives its discovery, and its
reported confidence about itself is calibrated because it checks itself.*

## Descriptive, not gated

K5 on every endpoint; `posterior − certainty` and `posterior − uncritical`
survival and discovery (development: positive on average, intervals spanning
zero); the oracle and true-structure ceilings and the share of the gap closed;
first-life discovery; rebuild counts and falsified reports; K1 `posterior −
bandit` (predicted tie: one hidden resource needs no structure).

Pre-stated K5 expectations from development: exact discovery after life 6
about 0.5, survival above random, calibration worse than K1–K4.

## Failure rule

A failed gate is reported as failed and its claim is not made. If G1, G2 or G7
fails the mechanism is closed for the integrated claim at this budget. No
threshold, particle count, falsification threshold, rebuild count, sample size
or arm definition changes after the survey starts; no rerun with more
identities; no K moved between core and descriptive.

## Seeds, budget, artifacts

Survey base 5,000,000,000; diagnostic 5,100,000,000; development
5,200,000,000; block stride 2,000,000, K stride 100,000, identity stride 100,
donor identity +50. All randomness is drawn through SHA-256 domains tagged
`probe80:` (map, donor-map, birth, scale, shock, agent-arm, uniform-actions,
permutation, offline). All bands lie above every earlier probe's.

Runtime: the diagnostic (one identity per K, all nine arms, five processes)
took 83–135 s per identity. The survey runs as 25 resumable block × K units on
eight workers; estimate 100 ± 30 minutes. Each unit pins the
source of both modules, the substrate and this file; a changed source refuses
to resume. Raw records stay in `runs/` (ignored); the result document carries
the headline numbers.

```bash
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.active_self --mode survey --block B --k K
PYTHONPATH=src .venv/bin/python -m homesocial.organism.active_self --mode survey --grade
```

## Claim boundaries, stated now

- The hypothesis family is authored (bounded resources, linear depletion,
  saturating restoration, uniform shocks); discovered are count, partition,
  sizes, rates and levels. The sensor width is given, as in Probes 76–77.
- Nonspatial: the motor is actuator choice, not navigation; the frozen island
  parent is not involved and not replicated.
- Survival is a policy endpoint; all arms share the action rule except
  `random`, `bandit`, `certainty` and `oracle`.
- Self-criticism tests consistency with the agent's own model class. It does not
  detect a body outside that class, and it is not yet a change detector
  (Probe81).
- No language, consciousness or sentience claim.
