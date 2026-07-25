# Protocol return-origin result

Date: 2026-07-25
Decision: **the chained inspect latent is rejected by its mechanism gate**. It
makes the six-tick return forecast worse in every one of 300 paired contexts.
The feasibility treatment was therefore not run.

## Integrity

The manipulation, paired mechanism measurements, unchanged feasibility gates,
and stop rule were committed in
`2026-07-25-protocol-return-origin-preregistration.md` before implementation or
checkpoint evaluation.

Checkpoint, unchanged:
`runs/organism/probe33_online_budget/organism_grounded_choice60000x3h40r6n0.55q8_options_inspect_bind16_replay256x1_drift0.01split_clip10_seed1.npz`.

Artifacts: `runs/organism/probe34_protocol_return_origin/`.

No parameter was updated.

## Control reproduction

The selectable aliased mode reproduces every sealed feasibility number to
printed precision:

| Quantity | Sealed | Instrumented control |
|---|---:|---:|
| Mean intact advantage | +0.0205 | +0.0205 |
| Positive contexts | 46.33% | 46.33% |
| No-write / collapsed advantage | -0.0262 | -0.0262 |
| Write / collapsed advantage drop | 0.0466 | 0.0466 |
| Branch contingency | 85.11% | 85.11% |
| Matching-label target selection | 99.78% | 99.78% |
| Danger-label avoidance | 85.33% | 85.33% |

R1 passes. The instrument did not alter the sealed path.

## Mechanism result

The single treatment used the already-computed, action-conditioned inspect
latent only as the input to the learned return transition. The external-memory
write, exact public carrier, observation-driven recurrent state, and subsequent
settling path were held fixed.

| Quantity | Aliased choice origin | Chained inspect origin |
|---|---:|---:|
| Worst per-need return MAE | **0.1017** | **0.3333** |
| Urgent-index survival | 100.00% | 100.00% |
| Contexts with lower error | — | **0 / 300** |

Per-need mean prediction against the realized six-tick return:

| Need | Realized | Aliased | Chained |
|---|---:|---:|---:|
| Food | -0.0600 | -0.0296 | **+0.0444** |
| Water | -0.0840 | -0.0190 | **+0.0423** |
| Energy | -0.1100 | -0.0083 | **+0.2233** |
| Health | -0.0120 | -0.0076 | -0.1089 |

R2 fails maximally: the treatment improves no paired context and worsens the
worst error by 0.2316. R3 happens to hold, but cannot rescue the failed
mechanism.

Per the locked stop rule, the treatment feasibility battery was not run. No
latent mixture, duration scale, or endpoint threshold was tuned.

## Why the attractive diagnosis was wrong

The code really did sever the first latent edge, and the checkpoint really was
trained with a two-decision open-loop loss. That was not sufficient to make the
proposed composition in-distribution.

During multi-step training, the inspect latent is paired with the world model's
own decoded next vector. The treatment instead paired that latent with the
protocol branch's exact public return carrier. Earlier audits measured mean L1
error 13.55 on the decoder's visual/pose reconstruction; avoiding that decoded
scene is the reason the protocol branch exists. The treatment therefore joined
one half of the reconstructive branch to one half of the protocol branch and
created a state/vector combination the model had not learned.

As with the gradient-budget diagnosis, a real defect that explains the
observed code path was not thereby the cause of the failing endpoint.

## Exploratory sign decomposition

The preregistered treatment failed, so the next permitted action was to identify
the context factor without changing the planner. The aggregate 46.33% sign rate
is exactly 139/300. The fixed seed set contains exactly 139 food-demand and 161
water-demand contexts. Grouping the unchanged aliased control confirms identity:

| Condition | Food advantage / positive | Water advantage / positive |
|---|---:|---:|
| Intact | +0.1034 / **100.00%** | -0.0511 / **0.00%** |
| Write suppressed | +0.0563 / **100.00%** | -0.0974 / **0.00%** |
| Labels collapsed | +0.0563 / **100.00%** | -0.0974 / **0.00%** |

The write never changes a sign: paired sign agreement is 100.00% and the rate
at which a write turns a nonpositive context positive is 0.00%. It adds a real
mean advantage of 0.0466, but a resource-specific, write-independent offset is
larger.

This refutes the prior claim that return-state aliasing determines Gate 2. The
sign is determined exactly by which bodily resource is urgent. More strongly,
the no-write control predicts that physically inspecting with no information is
beneficial in every food context. Gate 2 is therefore currently dominated by
bodily-value calibration, not lexical information.

## Next falsifiable boundary

Before changing a loss or value rule, decompose the organism's predicted value
of a correct versus incorrect terminal consumption by demanded resource and
compare it with the simulator-scored consequence. The specific question is
whether the self-model preserves the correct *ranking* at 99.78% while
compressing the resource-dependent outcome magnitude enough that a lexical
write is worth only 0.0466.

If that predicted correct-minus-wrong bodily gain is accurately calibrated,
Gate 2 is an invalid aggregate threshold for an asymmetric metabolism and must
be replaced only through a new substrate preregistration, not weakened after
the fact. If it is compressed or cross-contaminated, the next intervention
belongs in consequence calibration, not in branch construction.

No compute or data request is justified.

## Claim boundary

This is a controlled negative result about composing learned bodily
counterfactuals. It is not self-report, reflection, generated language, or
evidence of consciousness.
