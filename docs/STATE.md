# STATE

Last rewritten: 2026-07-25. Rewrite this file, never append.

## Executive handover

The organism still has the prior session's strongest result:

- a hypothetical label naming the demanded resource selects that object
  **99.78%** of the time;
- ten-tick bodily forecasts preserve its own urgent-need ordering **100%**;
- acquired word-to-object bindings are reused **100%** in every later round,
  against about 31% under acute silence or write suppression; and
- acquiring a word has positive mean value (+0.0205).

This session localized and mostly repaired the remaining value-amplitude
failure. The strongest new checkpoint, probe44 at the single analytically
derived bound-event weight 0.0231, achieves final replay valid-bound event MAE
of **0.1227 food and 0.0508 water**, versus 0.3182 / 0.2580 in its fresh zero
control.

It is not promoted. Its last 10,000-tick online MAE is 0.1326 food versus
0.0569 water, a 0.0757 gap against the locked <= 0.03 balance gate. The late
on-policy stream contains 190 bound food events and 300 bound water events.
Per-segment balancing cannot balance a need absent from a segment.

Current best calibration checkpoint:
`runs/organism/probe44_gradient_matched_bound_event/treatment/`.

## The causal chain established this session

### 1. The apparent branch alias was not the cause

Probe34 tested the prior handover's proposed return-origin repair. Chaining the
learned inspect latent into the exact return carrier made prediction much
worse, because the latent had been trained with a decoder-predicted vector and
the intervention paired it with an exact vector—an out-of-distribution mixed
state.

The grouped control found the real sign determinant:

- all 139 food-demand contexts had positive inspect advantage;
- all 161 water-demand contexts had negative inspect advantage;
- write suppression changed magnitude but changed **zero signs**.

The old 46.33% gate was exactly the fraction of food contexts, not branch
aliasing.

### 2. Terminal bodily magnitude was compressed

Probe35 compared learned and simulator-scored terminal consumption on 300 fixed
contexts:

- realized demanded-minus-wrong margin: exactly 0.4000;
- learned margin: 0.0276 overall;
- learned/realized: 4.78% food, 8.74% water;
- demanded ranking remained 99.67% correct.

The model knew *which* object helped but severely underestimated *how much*.

### 3. A generic event loss learned exogenous resets

Probe36's 0.01 scale-balanced event term failed real calibration. Probe37
showed the failure already exists on the real acquired-label path, not only in
explicit counterfactual writes.

Probe38 instrumented two fresh 60,000-tick runs:

- 6,010 resource restorations in treatment;
- 39.60% had a valid selected-surface binding;
- replay preserved that fraction (1.007x);
- event/base gradient ratio was 16.50% globally and 16.44% lexically;
- final bound replay MAE remained 0.3397 food / 0.3253 water.

The magnitude-only event mask contained 10,499 forced between-round body-reset
entries—56.18% of all selected entries—while bound restorations were only
12.74%. Reset MAE improved far faster than grounded restoration MAE.

### 4. Causal consume scoping helped, but was insufficient

Probe39 restricted the extra objective to public consume actions and, at
horizon two, to chains whose final action is consume. It never used hidden
kind or simulator event name.

Final bound replay MAE moved to 0.2303 food / 0.2534 water. Food met the locked
25% improvement gate; water improved 22.08% and failed. The treatment was
rejected rather than promoted directionally.

### 5. Representation geometry is healthy under causal scoping

Probe40 held body, recurrent core, geometry, selected surface, and consume
action fixed while swapping only `this food`, `this water`, and `this danger`
memory values.

The generic event checkpoint had a real lexical collision:

- food-water cosine 0.980;
- minimum/maximum lexical distance 0.140.

The scoped checkpoint did not:

- food-water cosine 0.631;
- distance ratio 0.379;
- settled consequence second/first singular ratio 0.787;
- intended-label ordering 100% for food and water;
- water contrast per lexical distance was 91.7% of food.

Architecture or lexical separation is not the remaining scoped blocker.

### 6. Bound, per-need credit improved both resources

Probe41 selected only consume events whose selected surface already had a valid
lexical row, then averaged loss per active own-body dimension.

Against a fresh exact control:

- food replay MAE: 0.2303 -> 0.1935 (15.99%, pass);
- water: 0.2534 -> 0.2271 (10.40%, fail).

The online bound-count gap shrank from 31.9% to 14.5%, but water still lagged.
Count imbalance contributes; it is not the whole amplitude failure.

### 7. Neither gradient cancellation nor input aliasing remains

Probe42 computed base/event gradients on 160 fresh bound segments per fixed
checkpoint without updating:

- treatment water-present global cosine: +0.483;
- treatment water-present lexical cosine: +0.407;
- negative fractions: 4.35% / 7.61%;
- lexical event/base norm: 0.434.

The signal is aligned and substantial. Gradient surgery is not licensed.

Probe43 collected 1,000 lived bound restorations and probed the exact vector
entering the transition MLP:

- full held-out food/water accuracy: 100%;
- lexical-only accuracy: 100%;
- nearest-neighbor agreement: 100%;
- predicted/target magnitude: 40.47% food, 28.20% water.

The target is perfectly identifiable and still attenuated.

### 8. One analytic scale solves endpoint fit, not online balance

The water-present lexical gradient ratio 0.4336 fixed one value before
training:

`0.01 / 0.4336 = 0.0231`.

Probe44 ran fresh zero, 0.01, and 0.0231 conditions—no grid.

Final replay:

| Resource | Zero | 0.01 | 0.0231 |
|---|---:|---:|---:|
| Food MAE | 0.3182 | 0.2532 | **0.1227** |
| Water MAE | 0.2580 | 0.1520 | **0.0508** |

Both endpoint gates pass decisively. Both first-to-last online improvements
also pass (0.216 food, 0.290 water). The last-window balance gate fails:
0.1326 versus 0.0569.

Per the locked stop rule, terminal, forecast, memory, and feasibility audits
were not run on probe44.

## Exact next step

Do not change weight 0.0231, architecture, budget, or representation.

The next mechanism test is cross-update need balancing:

1. Maintain learner-side cumulative counts or stratified replay for
   valid-bound consume events per own-body target dimension.
2. Use only public consume action, selected-binding validity, and lived body
   delta; never hidden kind, correctness, event name, or counterfactual target.
3. Keep the bound objective and 0.0231 scale exactly fixed.
4. Train a fresh unbalanced 0.0231 control and a balanced treatment.
5. Require both final replay MAEs <= 0.10, both last-window online MAEs <=
   0.10, and their last-window gap <= 0.03 before terminal evaluation.

The observed failure is across-update nonstationarity. Another per-segment
normalizer cannot answer it.

## Sealed capabilities that still stand

- **Forecast repair / gate 6:** probe33 and
  `2026-07-25-forecast-memory-tension-result.md`.
- **Persistent cross-round lexical reuse:** 100% with matched acute ablations.
- **Causally load-bearing external lexical memory:** writes determine
  object-local choice; no-write and collapsed-language branches remove it.
- **Oracle drift substitution:** true drift alone takes the old planner from
  18% to 100%, and the trained scale-balanced drift model realizes it.
- **Homeostatic utility correction and persistent information economics:** the
  prior probe series remains sealed.

## Closed explanations

Do not reopen these without contrary evidence:

- return-origin aliasing as the magnitude cause;
- explicit-write transfer as the primary failure;
- raw event or binding scarcity;
- replay dilution;
- generic output-head sharing;
- fourfold hidden capacity;
- global gradient clipping;
- lexical collision on the causal scoped/bound checkpoint;
- downstream semantic-rank collapse;
- food/water identity aliasing at the transition input;
- gradient cancellation;
- a weight grid.

## Reproducibility and controls

- Fixed-seed evaluation is deterministic; training is not guaranteed
  bit-reproducible. Fresh paired controls remain mandatory.
- All new loss paths default to zero and are independently selectable.
- Simulator event/kind metadata appears only in declared post-hoc audit
  grouping, never in a model input, loss selector, policy, replay sampler, or
  planner.
- All experiments train locally on Apple Metal in about one minute per
  60,000-tick run.

## Claim boundary

Do not call this system conscious, sentient, authentically desiring,
reflective, or a real `me`. It does not yet generate language or report an
internal state.

The defensible claim is:

> Online embodied experience creates a persistent, object-local,
> causally load-bearing lexical memory; that memory makes lived bodily outcome
> direction perfectly identifiable, and an analytically scaled grounded-event
> objective can fit outcome magnitude locally. Continual calibration remains
> imbalanced across a nonstationary on-policy stream.

Seed expansion, write-disabled training pairs, open-island transfer, generated
self-report, and communication remain blocked until calibration balance,
terminal transfer, forecast/memory preservation, and the full feasibility
battery pass in that order.

No larger compute or generated-data request is justified.
