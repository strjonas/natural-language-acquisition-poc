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

This handoff adds two clean negative results and one completed diagnostic:

1. **Probe45:** learner-visible need-balanced replay works mechanically but
   makes continual food/water calibration less balanced.
2. **Probe46:** a same-segment cross-need gradient audit stops on its sampling
   integrity gate; no undersized result is interpreted.
3. **Probe47:** independently sampled population gradients show mild lexical
   anti-alignment in both checkpoints but positive shared-path geometry.
   Treatment makes both cosines slightly *more* positive, not >=0.20 more
   negative. Treatment-induced cross-need conflict is not the failure cause.

Probe45 headline:

| Metric | Uniform control | Balanced treatment | Gate |
|---|---:|---:|---:|
| Final replay food MAE | 0.1227 | **0.1414** | <= 0.10 |
| Final replay water MAE | 0.0508 | **0.0370** | <= 0.10 |
| Last-window food MAE | 0.1326 | **0.1627** | <= 0.10 |
| Last-window water MAE | 0.0569 | **0.0388** | <= 0.10 |
| Last-window gap | 0.0757 | **0.1239** | <= 0.03 |

Current best calibration checkpoint remains probe44:
`runs/organism/probe44_gradient_matched_bound_event/treatment/`.

Latest artifacts:

- `runs/organism/probe45_need_balanced_replay/`
- `runs/organism/probe46_cross_need_gradient_geometry/control/`
- `runs/organism/probe47_population_cross_need_gradient/`

The exact next experiment is a **fixed-stream matched update-response audit**.
It is intentionally not started in this session. No larger compute, generated
data, or full training condition is justified.

## Probe45: need-balanced replay

### Mechanism integrity passes

Replay eligibility uses only:

- public consume action;
- selected external lexical-memory validity; and
- positive lived food/water delta above 0.175.

It never reads simulator event name, hidden object kind, correctness,
counterfactual outcome, or current demand. Food/water requests alternate and
selection is uniform within the requested eligible pool. The original uniform
sampler is the default and its single-`choice()` RNG path is preserved.

Treatment accounting:

- replay updates: 1,021 for 1,021 collected lives/segments;
- requests after both pools exist: food 511, water 510;
- eligible hits: 1,021/1,021 (**100%**, gate >=95%);
- fallbacks: zero;
- selected segments food-active: 737;
- selected segments water-active: 750.

Every selected segment hit its requested need. Inclusion-exclusion gives 271
food-only, 284 water-only, and 466 both-active updates. Active-need exposure
differs by only 1.76%.

The fresh uniform control reproduced probe44 exactly to stored precision:

- final replay food/water MAE: 0.122666 / 0.050826;
- last-window food/water MAE: 0.132611 / 0.056932.

The default-off implementation therefore preserves the sealed trajectory, not
just its approximate distribution.

### Learning gates fail

Treatment still improves both resources from first to last window:

- food: 0.353858 -> 0.162708, improvement 0.191150;
- water: 0.351709 -> 0.038837, improvement 0.312872.

But food misses the absolute gate and the final gap grows to 0.123871. Final
replay food also misses 0.10 at 0.141372.

The on-policy bound water/food event-count ratio rises from 1.326 in control to
1.670 in treatment overall, and from 1.579 to 1.728 in the final window.
Replay updates shared recurrent/dynamics features that also feed the policy,
so changing auxiliary sampling changes the policy and hence the future stream.
Equal replay exposure does not hold experience fixed.

Per the locked stop rule, terminal calibration, forecast, memory, and
feasibility were not run.

## Probe46: paired-gradient integrity failure

Probe46 preregistered 128 fresh segments containing both valid-bound food and
water restorations within 20,000 ticks. The fixed uniform policy produced only
20 in 20,044 ticks. The first integrity gate fails.

The treatment checkpoint was not run, the 20 samples were not interpreted, and
the budget was not extended after seeing the result. The resource-specific
horizon-one plus horizon-two objective and scoped-gradient implementation did
produce finite nonzero smoke measurements and remains useful as audit support.

## Probe47: population cross-need gradients

### Feasibility and integrity

A count-only pass measured independent resource populations without computing
an objective or gradient:

| Checkpoint | Total segments | Food | Water | Both | Ticks |
|---|---:|---:|---:|---:|---:|
| Uniform control | 356 | 113 | 147 | 20 | 20,044 |
| Balanced treatment | 367 | 105 | 144 | 21 | 20,041 |

This locked 96 food and 96 water segments per checkpoint. The completed audit
reached all samples in 17,382 control ticks and 17,804 treatment ticks.

For each population it computes the exact existing resource-specific bound
objective, takes gradients without an update, averages within resource, and
compares mean food versus water gradients in global, lexical, and shared-
dynamics scopes. It uses 256 deterministic multinomial bootstrap resamples.

All values are finite. A repeated two-resource smoke followed identical ticks,
had bit-identical objectives, and identical reported summaries. Raw Metal
gradient norms varied by at most 4.89e-10 absolute / 5.03e-9 relative; that
precision is retained in artifacts.

### Geometry is mixed, not treatment-induced conflict

| Scope | Control cosine | Treatment cosine | Control negative bootstrap | Treatment negative bootstrap | Shift |
|---|---:|---:|---:|---:|---:|
| Lexical | -0.1108 | **-0.0851** | 100% | 100% | **+0.0257** |
| Shared dynamics | +0.1339 | **+0.1522** | 0% | 0% | **+0.0183** |

Strong conflict required treatment cosine <0 in both scopes, >=95% negative
bootstrap in both, and a <=-0.20 shift from control in both. Shared dynamics is
positive and treatment moves both scopes upward. This criterion fails.

Full alignment required >=+0.20 in both scopes. Mild lexical anti-alignment
therefore makes the locked classification **mixed/inconclusive**. It is stable,
but it predates the sampler and weakens under treatment; it cannot explain why
probe45 balance worsens.

### Sensitivity does not explain the failure

Median bootstrap error-normalized water/food sensitivity:

| Scope | Control | Treatment | Relative shift |
|---|---:|---:|---:|
| Lexical | 0.7186 | **0.6412** | -10.77% |
| Shared dynamics | 0.7441 | **0.6575** | -11.64% |

The treatment ratios cross the absolute 0.67 line, but the locked mechanism
criterion also required a same-direction >=25% shift from control. It fails.
Water is also the better-calibrated resource despite lower local sensitivity,
the opposite of an explanation for food's slower learning.

Mean objectives were 0.01749 food / 0.01368 water in control and 0.01717 /
0.01094 in treatment. Error normalization prevents residual scale from being
misread as sensitivity.

No projection, resource-specific head, parameter separation, replay-count
change, or loss-weight change is licensed.

## Exact next step for a later session

Do **not** train another replay sampler, increase replay capacity or update
count, change weight 0.0231, increase model size, or run downstream capability
gates yet.

Preregister a **fixed-stream matched update-response audit** that isolates
learning from policy feedback:

1. Start from one fixed probe45 uniform-control checkpoint.
2. Collect one frozen learner-visible training buffer and a disjoint heldout
   buffer with the checkpoint policy and no parameter updates. Run a count-only
   feasibility pass before locking per-resource heldout counts.
3. Clone the same initial weights. Initialize identical fresh optimizer states;
   checkpoint optimizer state is not saved and must not be reconstructed
   differently between conditions.
4. Apply the same locked number of world-model-only updates to both clones from
   the same buffer. The sole manipulation is uniform versus the already tested
   alternating need-balanced selector. No policy action or new experience may
   occur after updates begin.
5. Measure food/water valid-bound heldout MAE before and after, along with
   cross-resource regressions, using identical heldout segments and hidden
   states.
6. Lock a stop rule before implementation. Do not turn update count, buffer
   size, learning rate, or loss weight into a grid.

Interpretation:

- if balanced selection improves food without regressing water on a fixed
  stream, probe45's failure is specifically closed-loop behavioral feedback;
- if it does not, segment-balanced sampling fails even when exposure is held
  fixed and should be abandoned;
- either result is diagnostic and does not by itself license full training.

Only after a later local learning intervention passes both endpoint MAE <=
0.10, both last-window MAE <=0.10, and gap <=0.03 should the unchanged sequence
run:

1. real terminal calibration;
2. forecast and cross-round memory preservation;
3. the seven-gate feasibility battery;
4. only then write-disabled training pairs, seed expansion, open-island
   transfer, or communication work.

## Prior causal chain that still stands

### Branching and magnitude

- Probe34 ruled out return-origin aliasing as the sign cause.
- Probe35 found correct terminal ranking but severe amplitude compression.
- Probes36-38 showed a generic event loss mostly learns exogenous resets.
- Probe39's consume scoping helped but did not meet the water gate.

### Representation and optimization

- Probe40 showed healthy lexical separation under causal scoping.
- Probe41's bound per-need objective improved both resources but was too weak.
- Probe42 found calibration/base gradients aligned, not canceled.
- Probe43 decoded exact food/water identity from transition input at 100% while
  magnitude stayed attenuated.
- Probe44 derived the single scale 0.0231 from measured lexical gradient ratio;
  it solved endpoint magnitude but not online balance.
- Probe45 rules out absent-need replay exposure as the whole imbalance.
- Probe47 rules out treatment-induced cross-need conflict or sensitivity shift
  as the mechanism of probe45's regression.

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
- treatment-induced cross-need gradient conflict or sensitivity shift;
- a loss-weight, replay-size, or update-count grid.

Mild baseline lexical food/water anti-alignment is measured, not erased. It is
not mechanism-specific and does not license surgery under the locked result.

## Research basis

- Class-balancing reservoir sampling for imbalanced online streams:
  https://proceedings.mlr.press/v119/chrysakis20a.html
- GRASP rehearsal comparison:
  https://proceedings.mlr.press/v274/harun25a.html
- Coupled distribution balance in multi-label replay:
  https://arxiv.org/abs/2209.11469
- Negative task-gradient geometry and gradient surgery:
  https://papers.nips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html

These papers motivate measurements; none is evidence that its method works in
this organism.

## Repository state

Recent commits, in order:

- `d693dc5` preregisters need-balanced replay;
- `8241bb0` implements and tests it;
- `5a58305` records the failed learning gates;
- `404f1fa` preregisters paired cross-need geometry;
- `0735bf5` implements that read-only audit;
- `c923f55` records its sampling-integrity failure;
- `080077d` preregisters population cross-need geometry;
- `604b69e` implements the population audit;
- `cbe7dab` records its mixed negative result.

Latest verification before this rewrite: **272 tests passed and 13 subtests
passed**. New training behavior is default off; probe47 adds only a read-only
audit. Simulator event/kind metadata remains audit-only and never enters model
input, loss selection, replay, policy, or planning.

Training and audits run locally on Apple Metal. No larger compute, external
data, or generated-data request is justified.

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
> action loop, and this failure is not explained by treatment-induced
> food/water gradient conflict.
