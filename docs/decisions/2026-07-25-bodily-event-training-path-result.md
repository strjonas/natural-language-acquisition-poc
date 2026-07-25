# Bodily-event training-path result

Date: 2026-07-25
Decision: **training-distribution underfit with adequate data, conditioning,
replay coverage, and gradient**. The generic event term spends most of its
declared signal on forced between-round body resets, not grounded consumption.

## Integrity

The exact paired 60,000-tick runs, instrumentation, thresholds, and
interpretation were committed in
`2026-07-25-bodily-event-training-path-preregistration.md` before the audit was
implemented and before either run was started. The audit changes no model
input, loss, optimizer update, action, replay sample, or planner decision.

Artifacts:
`runs/organism/probe38_bodily_event_training_path/`.

All 81 organism tests pass.

## Locked classifications

| Diagnosis | Locked criterion | Treatment | Decision |
|---|---:|---:|---|
| Target scarcity | < 500 resource restorations | **6,010** | no |
| Semantic-conditioning scarcity | < 25% bound | **39.60%** | no |
| Replay dilution | replay/online bound fraction < 0.5 | **1.007** | no |
| Global optimization starvation | event/base gradient < 5% | **16.50%** | no |
| Lexical optimization starvation | event/base gradient < 5% | **16.44%** | no |
| Training-distribution fit | bound replay MAE < 0.05 | food **0.3397**, water **0.3253** | no |
| Training-distribution underfit | bound replay MAE >= 0.05 | both above | **yes** |

The treatment has 175 unique
category/surface/outcome-kind/body-bin/validity contexts. Its selected lexical
transition feature is nonzero and substantial on bound replay restorations
(mean norm 1.019 food, 1.279 water). The semantic pathway is present at the
transition head; the trained output remains near zero.

## What the event mask actually contains

Treatment online entries above the sealed 0.175 threshold:

| Entry class | Count | Share of all event entries |
|---|---:|---:|
| Forced between-round body reset | **10,499** | **56.18%** |
| Food restoration | 2,844 | 15.22% |
| Water restoration | 3,166 | 16.94% |
| Poison | 2,179 | 11.66% |
| Bound food or water restoration | **2,380** | **12.74%** |

The “bodily event” selector is purely magnitude-based. It therefore calls the
protocol's exogenous `semantic_choice_round_transition` a bodily event even
though the agent did not cause it and no selected object or lexical binding can
condition it. Those reset entries outnumber all resource restorations by 1.75
to one and bound restorations by 4.41 to one.

This is not a hidden-kind leak in learning. The audit may read simulator event
names to classify rows after the fact, but the loss sees only deltas. The
defect is precisely that the loss has no causal action selector.

## Learning curves identify where the capacity went

Pre-update MAE from the first to last 10,000-tick window:

| Target | First window | Last window | Improvement |
|---|---:|---:|---:|
| Bound food restoration | 0.3560 | 0.3417 | 0.0143 |
| Bound water restoration | 0.3587 | 0.3265 | 0.0321 |
| Forced round reset | 0.3450 | **0.2386** | **0.1064** |

The explicit event term is not inert. It learns the majority reset target much
faster than the grounded restoration target it was introduced to calibrate.

On the final replay reservoir, the treatment is not better than its fresh
zero-event control:

| Bound replay MAE | Zero control | Generic event treatment |
|---|---:|---:|
| Food | **0.3125** | 0.3397 |
| Water | **0.2556** | 0.3253 |

This agrees with probe37's held-out real-path failure and now locates it inside
the data received by the loss rather than at explicit counterfactual transfer.

## Gradient result

Six fixed every-100th-update samples contained a bound restoration. The
already-weighted event term had median gradient norm 0.1160 against 0.7159 for
the original bodily loss (ratio 0.1650). On
`token_embedding`/`binding_value`/`binding_read`, the norms were 0.00295
against 0.01842 (ratio 0.1644).

The 0.01 term has a measurable, nontrivial lexical gradient. Increasing it is
not licensed: the problem is what its mask selects.

## Smallest causal correction

The next test should preserve the scale, threshold, model, budget, and
experience while changing only event eligibility:

- one-step calibration applies only when the learner-visible action is a
  primitive consume or consume-object option;
- horizon-two calibration applies only to a rollout whose **final** action is
  consume, so the supervised endpoint is the terminal consequence rather than
  the subsequent forced reset.

This selection uses the agent's own action, never hidden object kind or
simulator event metadata. Ordinary bodily prediction and multi-step prediction
still train on all transitions through their sealed losses; only the extra
calibration term becomes causally scoped.

Do not increase or sweep the generic event weight. No larger compute or
generated data is justified.

## Claim boundary

This result shows that an acquired lexical memory reaches a learned transition
model but a magnitude-only auxiliary objective allocates most of its
calibration to exogenous body resets. It is not reflection, self-report,
generated language, consciousness, or subjective experience.
