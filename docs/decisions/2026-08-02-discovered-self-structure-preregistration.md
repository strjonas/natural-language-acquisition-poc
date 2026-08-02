# Discovered bodily self-structure preregistration

Date: 2026-08-02
Status: locked before implementation or treatment runs
Probe: 61 / Phase A2

## Why this experiment exists

`DIRECTION_2026-07-26.md` section 2 names five properties that separate a
self-model from a parrot. The first is **discovered**: the agent finds its own
state variables. It is currently `no`, and A2 was promoted by A1's gate-2
failure.

The probe57 self-model is 658 scalars fitted into a template a human wrote:

- the latent is exactly three-dimensional (`causal_drift_raw` has shape `(3,)`);
- latent axis `i` **is** true need `i`, because the training target is the true
  body vector `packet.needs[:3]` and the deployed readout is
  `REPORT_NEEDS[argmin(belief)]`;
- movement is hardcoded to cost the third axis
  (`mx.array([0.0, 0.0, move_extra])`);
- a fourth bodily variable is unrepresentable.

Every one of those is authored, not learned. This probe removes all four.

## The information the learner is given, and what is taken away

Probe57's causal stage is supervised by the **true per-axis body vector** at
every developmental transition. This probe replaces that with the two scalars
the world already computes for its own purposes:

| channel | source | what it already is |
|---|---|---|
| `mean_viability` | `env.Needs.mean_viability` | the world's own **reward**: `reward = mean_viability_after - mean_viability_before` (`env.py`) |
| `viability` | `env.Needs.viability` | the **death** signal: `terminated = viability <= 0` |

Neither names an axis, an ordering, a count, or a unit. Both are already
available to this organism in other learning paths, so this is a strict
**reduction** of privilege, not a new channel. Informally: the organism is told
only *how well it is overall* and *how close to death it is*, and must discover
from those two feelings plus its own public action/observation history that it
is made of several separate bodily variables, how many there are, what each one
does, and which of the world's resources restores which.

Also removed: the per-life birth body reading. The deployed belief starts from a
single learned prior identical in every life. Probe58 measured the cost of
exactly this (population-mean birth: 0.81 survival, 0.906 fidelity), so it is
paid knowingly.

Not removed, and stated as remaining structural priors: the latent is bounded in
`[0, 1]`, its transition is linear-plus-clip in the same public features probe57
used (duration, move count, uptake surface, shock surface), depletion is
non-positive, uptake is non-negative, shocks are non-positive, and the two
feelings are the weighted sum and the minimum of the latent. The last is the
homeostatic-resource prior; it is permutation-symmetric across latent dimensions
and therefore cannot hand over an axis identity, a count, or an order.

## Locked mechanism

A new module `src/homesocial/organism/discovered_self.py`. No `OrganismConfig`
knob. No pre-existing tensor is read or written by fitting.

**Latent.** `L = 8` dimensions, overcomplete against every ground truth tested.

**Parameters** (`raw` denotes an unconstrained tensor):

- `drift_raw (L,)` -> `drift = -softplus(raw)`, per-dim per-tick depletion;
- `move_raw (L,)` -> `move = -softplus(raw)`, per-dim extra cost per move;
- `uptake_raw (S, L)` -> `uptake = +softplus(raw)`, `S = 9` surfaces;
- `shock_raw (S, L)` -> `shock = -softplus(raw)`;
- `gate_raw (L,)` -> `q = sigmoid(raw)`, per-dim participation, initialized at
  `sigmoid(1.0) = 0.731` so the learner starts believing it has eight variables;
- `birth_raw (L,)` -> `z0 = sigmoid(raw)`, the learned birth prior;
- `mean_scale_raw`, `mean_offset`: scalar affine readout for the mean channel;
- `listener_logits (60, S+1)`: the probe57 listener model, relearned here from
  the same stream so the module is self-contained.

`drift_raw`, `uptake_raw`, `shock_raw`, `gate_raw`, `birth_raw` are initialized
with seed-dependent noise. Symmetric initialization cannot break the latent
permutation symmetry, so `--seed` varies this stage's model as well as its world
stream. This is the trap `test_structured_causal_development_actually_varies_with_its_seed`
was written for.

**Transition.** For each transition with duration `d`, move count `m`, uptake
one-hot `U`, shock one-hot `H`:

```
z' = clip(z + d*drift + m*move + U @ uptake + H @ shock, 0, 1)
```

**Readout.** With `u_j = q_j * z_j + (1 - q_j)` (an unparticipating dimension
reads as a full resource and can never be the minimum):

```
mean_hat = mean_offset + mean_scale * sum_j (q_j * z_j)
min_hat  = min_j u_j                      # exact minimum, no temperature knob
```

**Objective.** Over the fitted lives:

```
MSE(mean_hat, mean_viability) + MSE(min_hat, viability)
  + cross_entropy(listener_logits[token], observed help surface / no-help)
  + lambda * sum_j q_j
```

Adam, gradient-norm clip 1.0, minibatches of 32 lives, rolled out from `z0`
through the whole life with backpropagation through the rollout. There is no
teacher forcing: the latent is never reset to a truth it is not given.

**Deployment.** Belief starts at `z0` and is advanced by the same transition on
`_public_transition_features` only. The utterance planner is probe60's promoted
outcome-aware objective, in latent space:

```
score(token) = E_{listener outcome} [ min_j ( u_j of the realized next latent ) ]
```

with probe60's deterministic token-ID tie break. It reads no need label, no
`HELP_SURFACES`, no `NEED_TO_REPORT_WORD`, and no true body. The map from a
discovered dimension to a need is used **only by audits, never by the organism**.

Feelings are development-time only. The deployed belief is open-loop, exactly as
probe57's was. Turning the feelings on at deployment is Phase B1 and is not
tested here.

## Fixed data and protocol

- Parent checkpoint (motor and lexical parameters only):
  `runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz`.
- Development: one fixed stream of **80,000 primitive ticks** per seed and per
  ground truth, matching probe57's budget exactly. Fitting makes repeated passes
  over that one stream; no additional experience is collected.
- Tokens during development are drawn from a fixed, **body-independent**
  distribution: each of the three need words with probability 0.10, the other 57
  tokens sharing 0.70 uniformly. This differs from probe57's uniform draw for
  one declared reason -- it lengthens developmental lives so that slow bodily
  variables are observable at all -- and it is fixed before any body is seen.
- Lives are split by index 80/20. Fitting uses the first 80%. Every fit-quality,
  dimension, and ablation metric uses the held-out 20%.
- Replication: **5 seeds** from the start, indices 0-4.
- Developmental seed bases: `210,000,000 + 10,000,000 * seed_index +
  1,000,000 * K`. Evaluation bases: mapping interventions `300,000,000`,
  deployment `320,000,000`. A guard rejects any overlap, per the seed-stride
  trap.
- All pre-existing parent tensors must remain bit-identical.

### Ground-truth dimension sweep

The number of bodily variables is varied **in the world**, by freezing axes, and
the same learner with the same frozen hyperparameters is run on each. A frozen
axis has zero metabolism, birth level 1.0, and no shocks; its random draws are
still consumed so unfrozen streams stay bit-identical.

| condition | frozen | live variables | true `K` |
|---|---|---|---:|
| K1 | food, water, safety | energy | 1 |
| K2 | food, safety | water, energy | 2 |
| K3 | safety | food, water, energy | 3 |
| K4 | none | food, water, energy, **safety** | 4 |

K4 is the **unmodified frozen report ecology**. Its true bodily dimension is
four, because `safety` starts at 1.0, depletes 0.002/tick, and enters both
`viability` and `mean_viability` -- and it is exactly the variable probe57's
hand-written template cannot represent. K3 is that ecology with the vestigial
axis switched off.

This sweep is what makes the dimensionality result non-tunable. A sparsity
coefficient chosen to produce a particular count on one world must reproduce
three other counts on three other worlds without further adjustment.

### Hyperparameter selection, and what it may not touch

Learning rate and epochs are chosen once, on **seed index 0, K4 only**, by
held-out feeling RMSE at `lambda = 0`, from the grid `lr in {3e-3, 1e-2}`,
`epochs in {200, 400, 800}`.

`lambda` is then chosen by a rule fixed here before any dimension is computed:
**the largest `lambda` on the grid `{0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`
whose held-out feeling RMSE is within 5% of the best RMSE on that grid** -- the
standard one-standard-error style rule, expressed in fit terms only.

No gate quantity -- recovered dimension, mapping, drift recovery, ablation
specificity -- may be inspected during selection. The selected values are then
frozen across all 5 seeds and all 4 ground truths.

## Amendment, 2026-08-02, before any treatment run

Recorded while implementing, before any gate quantity was computed on any seed.

**What changed.** Single-shot fitting of a whole life could not reach the loss
the model class permits. Measured on an 8,000-tick pilot stream: an oracle model
carrying the world's true constants reaches a per-tick rollout error of 0.0032
on both sensations, while plain full-batch Adam plateaued around 0.09 -- three
orders of magnitude of loss apart. The cause is the conditioning of open-loop
trajectory fitting: an error in a depletion rate is multiplied by every tick
that follows it.

The fit therefore uses **Bock's multiple shooting**: each life is cut into short
segments, each segment is shot from its own free state, and the segments are
stitched by a continuity penalty; the segment length is then annealed until the
final stage is the ordinary single-shot trajectory. Per-life initial conditions
were already part of the design; multiple shooting is the same idea applied
within a life.

- Bock & Plitt (1984), *A multiple shooting algorithm for direct solution of
  optimal control problems*: https://doi.org/10.1016/S1474-6670(17)61205-9

**What this does not change.** The mechanism, the latent size, the two
sensations, the ground-truth sweep, every gate, every threshold, and every
control are exactly as locked above. The segment states are an optimizer device
and are discarded; **every held-out number reported, including the entire
dimension criterion, is computed by single shooting** -- one initial condition
per life and then open loop.

Two further implementation facts, also fixed before any gate was computed:

1. The word-to-consequence table is a categorical likelihood with no coupling to
   the latent, so it is estimated in closed form by its Laplace-smoothed maximum
   likelihood rather than by gradient descent. This is the same model probe57
   fitted, estimated exactly instead of approximately, and it keeps the table's
   gradient from consuming the body's clipped gradient budget.
2. The learning-rate grid is extended to `{3e-3, 1e-2, 3e-2}` and the epoch grid
   is replaced by two shooting schedules. Selection is still by held-out
   sensory RMSE alone, on seed 0 and K4 only.

## Second amendment, 2026-08-02, before any treatment run

Also recorded while implementing, before any gate quantity was computed.

**Concentration is measured over the effective dimensions.** G2 and G3 both
require an effect to concentrate `>= 0.80` on one dimension. As first written,
that share was over all eight. That is not measurable: a dimension the two
sensations cannot reach has *no identified parameters at all* -- its uptake,
shock and movement entries stay wherever initialization put them, and
initialization puts `softplus(-3) = 0.049` in every uptake entry. Five such
dimensions contribute 0.245 of spurious mass, which dilutes a correctly
recovered 0.80 uptake to 0.66 without any fact about the model changing.

Concentration is therefore computed over the dimensions the dimension criterion
retained. This is the only reading under which the quantity measures discovery
rather than initialization. The all-eight value is reported alongside it in
every case, and the retained set is itself an outcome of a locked criterion, not
a choice made per seed.

Nothing else moves: the thresholds, the required three-way agreement, the
one-to-one requirement, and the seed fractions are exactly as locked.

## Third amendment, 2026-08-02, before any treatment run

**The sparsity grid was written at the wrong scale and is extended downward.**

The locked grid was `{0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`. Running it on the
selection stream shows the sensory mean-squared error at the optimum is about
`1.7e-4`. The penalty is `lambda * sum_j q_j` with eight gates in `[0, 1]`, so
even the grid's *smallest* nonzero value contributes up to `8e-4` -- roughly
five times the entire data term. Every nonzero point on the grid therefore
swamps the fit, and the preregistered selection rule can only return `lambda =
0`, which deletes the sparsity penalty the mechanism is defined to include.

The grid is extended to
`{0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`. The original
points are all still run and reported, so the record shows why they were
dominated.

The **selection rule itself does not change**: the largest coefficient whose
held-out sensory RMSE is within 5% of the best on the grid, chosen on seed 0 and
K4 only, reading held-out sensory error and nothing else. No dimension, mapping,
rate, or lesion quantity is inspected during selection. This is a correction to
a badly scaled grid, made before any treatment run, of the same kind as the
learning-rate extension above.

## Recovered effective dimension

Defined interventionally, by greedy backward elimination on held-out lives.
`RMSE_full` is the fitted model; `RMSE_null` predicts each channel's held-out
mean; `R = RMSE_null - RMSE_full`. Repeatedly delete the dimension whose
deletion costs least. `d_eff` is the smallest number of retained dimensions
whose held-out RMSE is `<= RMSE_full + 0.10 * R`.

Gate threshold `0.10` is locked here. The full elimination curve is reported so
the margin is visible.

## Locked gates

All of G1-G5 must pass. `n = 5` seeds per ground truth, 100 held-out evaluation
lives per condition.

### F0 -- the discovered model tracks the body at all

In K3, after mapping dimensions to needs by the audit map, mean absolute error
between mapped latent and true need is `<= 0.05` on 5/5 seeds. Looser than
probe57's `0.03` because the per-life birth reading has been removed. If F0
fails, nothing below is interpretable.

### G1 -- the recovered dimension tracks the ground truth

`d_eff == K` on `>= 4/5` seeds for each `K in {1, 2, 3, 4}`, and mean `d_eff`
strictly increasing in `K`.

Constant `d_eff` across `K` falsifies discovery and convicts the sparsity
coefficient of setting the answer.

### G2 -- each discovered dimension is one bodily variable

For the three interventionally reachable needs, on `>= 4/5` seeds in **both** K3
and K4, all three of the following agree on the same one-to-one need-to-dimension
map, each with concentration `>= 0.80` on its dimension:

- **uptake parameters**: the learned uptake row of that need's two help surfaces;
- **shock parameters**: the learned shock row of that need's shock surface;
- **live intervention**: the change in the deployed latent when a grant of that
  need is forced in a held-out life.

Concentration is the mapped dimension's share of the absolute effect summed over
all 8 dimensions. Three independent causal channels must name the same
dimension. A random assignment satisfying one-to-one-ness plus three-way
agreement at 0.80 concentration has probability far below 0.01.

### G3 -- the discovered rates are the body's real rates

On `>= 4/5` seeds in K4:

- each mapped dimension's learned per-tick depletion is within **25%** of the
  true metabolism it corresponds to (food 0.008, water 0.012, energy 0.016);
- the learned extra movement cost concentrates `>= 0.80` on the energy
  dimension -- which probe57 was told and this learner is not;
- exactly one further effective dimension exists whose total uptake and total
  shock magnitudes are each `<= 10%` of the smallest mapped dimension's, and
  whose learned per-tick depletion lies in `[0.001, 0.004]`.

The last clause is the sharp one: the true value is 0.002, the variable is
`safety`, nothing in the world announces it, no word requests it, no help
restores it, and probe57's self-model has no slot for it.

### G4 -- lesioning a discovered dimension causes its own specific failure

Deployment in K3 with the discovered planner. For each mapped dimension `j`,
freeze `z_j` at its birth prior for the whole life and measure per-need recall
(the rate of saying need `k`'s word while `k` is truly lowest):

- the largest recall drop is on `map(j)` for **3/3** dimensions, on `>= 4/5`
  seeds;
- recall on `map(j)` drops `>= 30` points and recall on each other need drops
  `<= 10` points;
- freezing a dimension `d_eff` did **not** retain changes every recall by
  `<= 5` points.

### G5 -- the deployed report is causally grounded

In K3 and in K4, relative to the intact discovered planner:

- grounded survival `>= 0.75` and report fidelity `>= 0.80`;
- grounded minus scrambled-listener survival `>= 15` points;
- zero-belief and mute-organism survival lower by `>= 30` points;
- the probe52 lexical intervention gate remains intact on the parent tensors.

Survival `0.75` is set below probe58's measured 0.81 population-mean-birth
ceiling because this organism has no per-life birth reading at all.

## Controls that must fail

Reported for every seed; a positive result is void if any of these also passes.

1. **Shuffled feelings**: both channels permuted across ticks within each life.
   G1 and G2 must fail.
2. **Mean channel only**: the minimum is withheld. Prediction: the axes are not
   separable from a single linear projection, so G2 fails.
3. **Minimum channel only**: reported for symmetry.
4. **Labeled 3-dimensional reference**: probe57's supervision on the same
   stream, as the privilege ceiling for F0's body error.

## Failure rule and claim boundary

A failed gate closes this mechanism. It is not rescued by raising `L`, changing
the sparsity family, adding readout channels, re-tuning `lambda` per world,
softening the minimum, lengthening development, or weakening the ground-truth
sweep to fewer values of `K`. Any of those requires a new preregistration with
its own controls.

A pass would show that this organism **discovers how many bodily variables it
has, what each one's dynamics are, and which of the world's resources restores
which**, from nothing but how good and how bad it feels, and that damaging one
discovered variable produces exactly that variable's failure in what it says.

It would **not** show evidence-based belief correction (Phase B1), calibrated
uncertainty or any reflexive representation (C1), future-self report (C2),
non-linear or non-stationary bodily structure, discovery of variables with no
consequence for the two feelings, consciousness, sentience, or natural-language
understanding. The homeostatic-resource readout prior remains authored.

No external corpus, generated data, vocabulary growth, or larger compute is
authorized by this probe.

## Research basis

- Hyvärinen & Morioka (2016), *Unsupervised feature extraction by time-contrastive
  learning and nonlinear ICA*: https://arxiv.org/abs/1605.06336
- Ahuja et al. (ICML 2023), *Interventional causal representation learning*:
  https://proceedings.mlr.press/v202/ahuja23a.html
- Keramati & Gutkin (2014), *Homeostatic reinforcement learning*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4270100/
- Seth & Tsakiris (2018), *Being a beast machine: the somatic basis of selfhood*:
  https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(18)30188-1

These motivate the mechanism class and the identifiability argument. Only the
interventions above are evidence about this organism.
