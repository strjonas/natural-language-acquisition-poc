# STATE

Last rewritten: 2026-07-25. Rewrite this file, never append.

## Executive handover

The strongest sealed capabilities remain:

- a hypothetical label naming the demanded resource selects that object
  **99.78%** of the time;
- a ten-tick bodily forecast preserves the organism's own urgent-need ordering
  **100%** of the time;
- acquired word-to-object bindings are reused **100%** in every later round,
  versus about 31% under acute silence or write suppression; and
- acquiring a word has positive mean value (**+0.0205**).

The new intermediate result is a clean negative one. Probe45 implemented
learner-visible need-balanced replay exactly as preregistered. It alternated
food/water requests 511/510, found an eligible segment on 1,021/1,021 updates,
and kept active-need replay exposure close (737 food, 750 water). It still made
continual calibration *less* balanced:

| Metric | Uniform control | Balanced treatment | Gate |
|---|---:|---:|---:|
| Final replay food MAE | 0.1227 | **0.1414** | <= 0.10 |
| Final replay water MAE | 0.0508 | **0.0370** | <= 0.10 |
| Last-window food MAE | 0.1326 | **0.1627** | <= 0.10 |
| Last-window water MAE | 0.0569 | **0.0388** | <= 0.10 |
| Last-window gap | 0.0757 | **0.1239** | <= 0.03 |

The sampler mechanism passes; its learning gates fail. Segment-count balance
is not the remaining repair.

Probe46 then preregistered a read-only food-versus-water gradient audit on 128
segments containing both outcomes. The fixed uniform policy produced only 20
such segments in 20,044 ticks. That integrity gate failed, so the treatment
checkpoint was not run and the 20 samples were not interpreted.

Current best calibration checkpoint remains probe44:
`runs/organism/probe44_gradient_matched_bound_event/treatment/`.

Latest artifacts:

- `runs/organism/probe45_need_balanced_replay/`
- `runs/organism/probe46_cross_need_gradient_geometry/control/`

## What probe45 establishes

### 1. The implementation tested the intended mechanism

Eligibility uses only:

- public consume action;
- selected external lexical-memory validity; and
- positive lived food/water delta above 0.175.

It never reads simulator event name, hidden object kind, correctness,
counterfactual outcome, or current demand. Food/water replay requests alternate
deterministically. Selection is uniform within the requested eligible pool and
falls back to the original uniform sampler only when that pool is empty.

The 6,000-tick real-loop smoke had both pools from the first replay, request
counts 34/33, eligible hits 67/67, and zero fallbacks. Unit tests separately
verify causal scope, alternation, fallback, and the disabled RNG path.

### 2. The sealed default path is exact

The fresh probe45 uniform control reproduced probe44 exactly to the stored
precision:

- final replay food/water MAE: 0.122666 / 0.050826;
- last-window food/water MAE: 0.132611 / 0.056932.

This is stronger than a distributional control: the default-off code path
preserved the seeded training trajectory and its single `choice()` RNG path.

### 3. The sampler removed absent-need credit imbalance

Treatment accounting:

- replay updates: 1,021 for 1,021 collected lives/segments;
- post-pool requests: food 511, water 510;
- eligible hits: 1,021/1,021 (**100%**, gate >=95%);
- fallbacks: zero;
- selected segments food-active: 737;
- selected segments water-active: 750.

Every selected segment hit at least its requested need. Inclusion-exclusion
therefore gives 271 food-only, 284 water-only, and 466 both-active updates.
Active-need exposure differs by only 1.76%. A need being absent from too many
replay updates was real but not sufficient to explain calibration asymmetry.

### 4. Closed-loop stream feedback dominates simple replay balance

The bound water/food event-count ratio moved from 1.326 in the uniform control
to 1.670 in treatment overall, and from 1.579 to 1.728 in the final window.
The replay loss updates shared recurrent/dynamics parameters, which also feed
the policy that generates the next on-policy stream. Equal auxiliary exposure
does not hold the future experience distribution fixed.

Both treatment resources still improve from first to last window:

- food: 0.353858 -> 0.162708, improvement 0.191150;
- water: 0.351709 -> 0.038837, improvement 0.312872.

But food misses the absolute threshold and the gap grows to 0.123871. Per the
locked stop rule, terminal calibration, forecast, memory, and feasibility were
not run on probe45.

## Probe46: diagnostic integrity failure

Probe42 had shown that the combined bound-event objective is aligned with the
ordinary bodily objective. It did not compare food-specific and water-specific
gradients. Probe46 attempted that comparison on the same segments so context
would be matched exactly.

The code computes the existing horizon-one plus horizon-two bound constituent
separately per need and reports geometry for:

- all reached parameters;
- lexical parameters; and
- the shared recurrent/dynamics path excluding independent final need rows.

It also reports `gradient_norm / sqrt(objective)` as a local Jacobian-norm
proxy. The implementation produced finite nonzero values in a two-segment
smoke and passed the full test suite.

The preregistered population was infeasible: only 20 both-resource segments
appeared within 20,044 ticks, versus 128 required. Do not extend the budget or
interpret those samples after seeing this result. The audit remains useful for
small feasibility checks, but it did not answer cross-need geometry.

## Exact next step

Do **not** train another replay sampler, increase replay capacity or update
count, change weight 0.0231, increase model size, or run downstream capability
gates yet.

The next agent should preregister a **population cross-need gradient audit**:

1. Use the fixed fresh probe45 uniform and treatment checkpoints.
2. Collect food-present and water-present learner-visible valid-bound segments
   independently, rather than requiring both in one segment.
3. Compute the already implemented resource-specific bound objective and
   gradient for each eligible segment without optimizer updates.
4. Average gradients within resource, then compare food versus water mean
   gradients in lexical and shared-dynamics scopes. Use deterministic bootstrap
   resamples if a negative-fraction statistic is desired.
5. Lock achievable per-resource counts and a tick ceiling before running. A
   read-only count-only feasibility pass may set that design; the failed 20
   paired samples are not evidence.

Interpretation must remain conditional:

- strong negative cross-need geometry may license projection or parameter
  separation;
- aligned, similarly sensitive gradients rule cancellation out and point next
  to a fixed-stream matched update-response audit, isolating replay learning
  from the on-policy feedback loop;
- mixed geometry licenses no optimizer intervention.

Only after a local learning intervention passes both endpoint MAE <=0.10,
both last-window MAE <=0.10, and gap <=0.03 should the unchanged sequence run:

1. real terminal calibration;
2. forecast and cross-round memory preservation;
3. the seven-gate feasibility battery;
4. only then write-disabled training pairs, seed expansion, open-island
   transfer, or communication work.

## Prior causal chain that still stands

### Branching and magnitude

- Probe34 ruled out return-origin aliasing as the sign cause. Demand identity,
  not binding writes, determined the old sign split.
- Probe35 found correct terminal ranking but severe amplitude compression:
  learned demanded-minus-wrong margin was only 4.78% of realized for food and
  8.74% for water.
- Probes36-38 showed a generic event loss mostly learns exogenous round resets.
- Probe39's consume scoping helped but did not meet the water gate.

### Representation and optimization

- Probe40 showed healthy lexical separation under causal scoping.
- Probe41's bound per-need objective improved both resources but was too weak.
- Probe42 found calibration/base gradients aligned, not canceled.
- Probe43 decoded exact food/water resource identity from transition input at
  100% while magnitude stayed attenuated.
- Probe44 derived the single scale 0.0231 from measured lexical gradient ratio;
  it solved endpoint magnitude but not online balance.
- Probe45 rules out simple active-need replay-count imbalance as the remaining
  mechanism.

## Closed explanations

Do not reopen these without contrary evidence:

- return-origin aliasing as the magnitude cause;
- explicit-write transfer as the primary failure;
- raw event or binding scarcity;
- generic replay dilution;
- generic drift/output-head sharing;
- fourfold hidden capacity;
- global gradient clipping;
- lexical collision under causal scoping;
- food/water identity aliasing at transition input;
- calibration versus base-loss gradient cancellation;
- insufficient bound-event objective scale;
- absent-need replay exposure as the whole online imbalance;
- a loss-weight, replay-size, or update-count grid.

Cross-need gradient cancellation is **unanswered**, not closed: probe46 failed
sampling integrity before measuring it.

## Research basis

- Chrysakis and Moens, class-balancing reservoir sampling for imbalanced online
  streams: https://proceedings.mlr.press/v119/chrysakis20a.html
- GRASP rehearsal comparison, including uniform class-balanced sampling:
  https://proceedings.mlr.press/v274/harun25a.html
- Optimizing Class Distribution in Memory, on coupled balance for multi-label
  replay: https://arxiv.org/abs/2209.11469
- Gradient Surgery for Multi-Task Learning, on negative task-gradient geometry:
  https://papers.nips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html

These papers motivated measurements; none is treated as evidence that a method
works in this organism.

## Repository state

Relevant commits, in order:

- `d693dc5` preregisters need-balanced replay;
- `8241bb0` implements and tests it;
- `5a58305` records the failed learning gates;
- `404f1fa` preregisters paired cross-need geometry;
- `0735bf5` implements the read-only audit;
- `c923f55` records its sampling-integrity failure.

Latest verification before this rewrite: **271 tests passed and 13 subtests
passed**. All new training paths are default off. Simulator event/kind metadata
remains audit-only and never enters model input, loss selection, replay, policy,
or planning.

Training remains local on Apple Metal; no larger compute, external data, or
generated-data request is justified.

## Claim boundary

Do not call the system conscious, sentient, authentically desiring,
reflective, or a real `me`. It does not yet generate language or report an
internal state.

The defensible intermediate claim is:

> Online embodied experience creates persistent, object-local,
> causally load-bearing lexical memory. That memory makes lived bodily outcome
> identity perfectly decodable and supports locally calibrated consequence
> magnitude. Simple need-balanced replay is mechanically successful but does
> not stabilize continual calibration inside the closed perception-learning-
> action loop.
