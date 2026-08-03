# Discovered bodily structure: result

Date: 2026-08-03. Preregistration:
`docs/decisions/2026-08-02-discovered-self-structure-preregistration.md`.
Artifacts: `runs/organism/probe61_discovered_self/full/discovered_self.json`,
20 fits = 4 ground-truth worlds x 5 seeds, 100 held-out lives per deployment
condition. Hyperparameters were frozen before any treatment run: schedule A,
learning rate 0.03, sparsity 3e-5.

**Verdict: four of the five locked gates fail. By the preregistered failure rule
-- "all of G1--G5 must pass" -- this mechanism is closed.** It is not rescued by
raising `L`, changing the sparsity family, re-tuning per world, or weakening the
sweep. What it did establish is recorded below with equal care, because some of
it is strong and one of it reverses a warning carried in `docs/STATE.md`.

| gate | requirement | result | |
|---|---|---|---|
| F0 | K3 mapped body error `<= 0.05`, 5/5 seeds | 0.0161--0.0209, mean **0.0177** | **pass** |
| G1 | `d_eff == K` on `>= 4/5` seeds for every `K`, mean strictly increasing | 2/5, 3/5, 2/5, 5/5; means **1.6 < 2.6 < 3.6 < 4.0** | **fail** |
| G2 | one-to-one map, 3 channels agree, concentration `>= 0.80`, `>= 4/5` in K3 **and** K4 | **5/5 and 5/5** | **pass** |
| G3 | rates within 25%, movement on energy `>= 0.80`, silent variable, `>= 4/5` in K4 | **3/5** | **fail** |
| G4 | lesion drops the lesioned need's recall `>= 30` points | **0/5** | **fail** |
| G5 | survival `>= 0.75` and fidelity `>= 0.80` in K3 **and** K4 | K3 partial, K4 fails outright | **fail** |

## Execution note

The twenty fits were run as four processes, one per world. `run_world_seed` is
pure in `(world, seed)`, and this was verified rather than assumed: every scalar
field of all four seed-0 records is **bit-identical** to the earlier
single-process seed-0 run in `runs/organism/probe61_discovered_self/seed0/`. The
merge (`scripts/probe61_merge_worlds.py`) refuses partial or ragged sweeps and
is the only place the locked gates are evaluated, because nearly all of them are
defined over the pooled record set.

## What passed, and it is not nothing

**F0.** The discovered latent tracks a body it was never shown, from two scalars,
with no per-life birth reading: mean absolute error 0.0177 in K3 against a gate
of 0.05 and probe57's supervised 0.0158 -- and probe57 was handed the true
three-vector at every developmental transition.

**G2, cleanly, on 10/10 seeds.** In both K3 and K4 the need-to-dimension map is
one-to-one and three independent causal channels -- learned uptake, learned
shock, and a live forced-grant intervention in a held-out life -- agree on the
same map at concentration `>= 0.80`. Nothing told the learner there were three
reachable needs, which dimension was which, or what any of them was called.

**The rates themselves are recovered to within 11%, on every seed.** In K4, all
fifteen need-by-seed ratios of learned to true metabolism:

| need | true | recovered range | ratio range |
|---|---:|---|---|
| food | 0.008 | 0.00714--0.00778 | 0.89--0.97 |
| water | 0.012 | 0.01148--0.01285 | 0.93--1.07 |
| energy | 0.016 | 0.01565--0.01596 | 0.978--0.998 |

**And `d_eff` moves with a ground truth that moves.** Mean recovered dimension is
1.6, 2.6, 3.6, 4.0 against true 1, 2, 3, 4 -- strictly increasing, which is the
second clause of G1 and it passes. The preregistration named the specific
falsification it feared: *"constant `d_eff` across `K` falsifies discovery and
convicts the sparsity coefficient of setting the answer."* **That falsification
did not occur.** The count is genuinely responsive to the body.

## What failed

**G1, on accuracy.** The per-`K` requirement is `>= 4/5` and only K4 meets it.
Recovered dimensions: K1 (1,1,2,2,2), K2 (2,2,2,3,4), K3 (4,4,3,3,4), K4
(4,4,4,4,4). The bias is upward everywhere -- no fit in the sweep ever
under-counted -- and no fit ever exceeded four despite an eight-dimensional
latent. Held-out sensory RMSE is uniformly good on the misses as well as the
hits (0.0088--0.0233), so this is the dimension criterion, not the fit.

**G3, at 3/5 where 4/5 was required.** Two sub-clauses cost it. Movement
concentrates on the energy dimension on 4/5 seeds, but the concentration itself
clears 0.80 on only 3/5 (0.983, 0.799, 0.812, 0.837, 0.531). And the silent
variable is found on 3/5.

**G4, 0/5, as already known from seed 0 and unchanged by four more.** The gate is
worded as a blinding prediction and the phenomenon is the opposite: freezing a
discovered dimension *fixates* the organism on that need rather than blinding it
to it, because the frozen birth prior sits below where help keeps the real axes.
That description remains post hoc. **The locked gate stands as failed and is not
rewritten to match it.**

**G5.** K3 deployment is strong -- survival 0.90/0.95/0.91/0.92/0.91, scrambled
listener 0.04--0.07, zero-belief and mute-organism 0.00 on every seed -- but
report fidelity is 0.714--0.876 and clears the 0.80 bar on only 2/5. K4
deployment degenerates on 5/5 exactly as predicted from seed 0: survival
0.56--0.58, fidelity 0.42--0.45. `safety` is unreachable, its discovered drift
runs fast, so it becomes the running minimum in every branch and probe60's
`E[min]` objective goes indifferent across the vocabulary.

## The seed-0 K3 warning does not replicate, and the reason is quantitative

`docs/STATE.md` carried a strong caution from seed 0: that K3, where no fourth
bodily variable exists, produced an extra drift-only dimension at 0.0022 with
"the same signature" as K4's, making the discovered-`safety` clause confounded.

**Across all five seeds the coded criterion has zero false positives.**
`silent_variable_found` is **False on 5/5 K3 seeds** and **True on 3/5 K4 seeds**
(drifts 0.0027, 0.0029, 0.0028 against a true 0.002). It was already False for
K3 seed 0 in the original run; the seed-0 caution was a hand-read of the extra
dimension's drift alone, and the full criterion also requires the dimension to be
causally quiet.

That is exactly the clause that separates them. Both worlds produce extra
dimensions. What differs is how silent they are -- uptake as a share of the
smallest mapped dimension's:

| | extra-dimension uptake ratio | drift |
|---|---|---|
| K3 (nothing to find) | 0.125, 0.174, 0.219 | 0.0011--0.0022 |
| K4 (`safety` is real) | **0.078, 0.080, 0.091**, 0.111, 0.182 | 0.0027--0.0054 |

K3's spurious dimensions are never quieter than 0.125; K4's reach 0.078. The
10% threshold was locked in advance and it separates them 0/5 against 3/5.

**This does not license the discovered-`safety` claim.** Three of five is below
the locked 4/5, the recovered drift overshoots the true 0.002 by 35--170%, and
K4's perfect `d_eff` remains consistent with an upward bias that cannot exceed
four rather than with detection. The honest statement is narrower and worth
having: **the criterion is specific but not sensitive enough.** A world with true
`K = 5` would discriminate the two readings and this ecology does not contain
one.

## Controls: not run

The preregistration requires shuffled-feelings, mean-channel-only,
minimum-channel-only, and the labeled three-dimensional reference reported for
every seed. **They were not run.** They exist to void a *positive* result, and
this result is negative on four of five gates, so their absence does not change
the verdict. It does mean the passing gates F0 and G2 are uncontrolled here and
should not be cited as standalone positives without them. The CLI runs the
treatment loop before controls, so a controls-only pass needs a separate driver.

## Claim boundary

This experiment does **not** show that the organism discovers how many bodily
variables it has. G1 fails and that was the central claim.

It does show, within one frozen ecology and without the controls that would seal
it, that from only the mean and minimum of its own bodily variables the organism
recovers three metabolic rates to within 11%, a one-to-one need-to-dimension map
confirmed by three independent causal channels on 10/10 seeds, and a body it was
never shown to within 0.0177 -- and that its recovered dimension count responds
monotonically to a ground truth that moves.

It shows nothing about evidence-based belief correction, uncertainty, reflection,
future-self report, non-linear or non-stationary bodily structure, consciousness,
sentience, or natural-language understanding. The homeostatic-resource readout
prior remains authored, and probe62 separately established that this ecology's
bounded body caps self-model error at ~0.05 regardless of how much is hidden.
