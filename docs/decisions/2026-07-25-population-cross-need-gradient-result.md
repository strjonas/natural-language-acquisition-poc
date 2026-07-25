# Population cross-need gradient result

Date: 2026-07-25
Decision: **integrity passes; geometry is mixed and does not license an
optimizer intervention**.

## Integrity

The count feasibility, 96-per-resource populations, fixed checkpoints,
gradient scopes, 256 bootstrap resamples, thresholds, and stop rule were
committed before the audit implementation or run.

Both checkpoints supplied the full population within the 20,000-tick ceiling:

| Checkpoint | Food | Water | Elapsed ticks |
|---|---:|---:|---:|
| Uniform control | 96 | 96 | 17,382 |
| Balanced treatment | 96 | 96 | 17,804 |

All objectives and geometry values are positive/finite where required. A
repeated two-food/two-water smoke followed identical ticks and produced
bit-identical objective values and identical reported summaries. Raw Metal
gradient norms differed by at most 4.89e-10 absolute / 5.03e-9 relative; this
precision is retained rather than rounded away.

Artifacts: `runs/organism/probe47_population_cross_need_gradient/`.

## Locked geometry result

| Scope | Control cosine | Treatment cosine | Control negative bootstrap | Treatment negative bootstrap | Treatment-control shift |
|---|---:|---:|---:|---:|---:|
| Lexical | -0.1108 | **-0.0851** | 100% | 100% | **+0.0257** |
| Shared dynamics | +0.1339 | **+0.1522** | 0% | 0% | **+0.0183** |

Strong conflict required treatment cosine <0 in *both* scopes, >=95% negative
bootstrap in both, and a <=-0.20 shift from control in both. The shared path is
positive, and treatment moves both scopes slightly upward. The criterion fails
decisively.

Aligned geometry required >=+0.20 and <=5% negative in both scopes. The lexical
path is mildly negative in both checkpoints, so that criterion also fails.
The locked classification is **mixed/inconclusive**.

The negative lexical population geometry is real and stable, but it predates
the balanced sampler and becomes less negative under treatment. It cannot
explain why treatment worsened food/water balance.

## Sensitivity result

Median bootstrap error-normalized water/food sensitivity:

| Scope | Control | Treatment | Relative shift |
|---|---:|---:|---:|
| Lexical | 0.7186 | **0.6412** | -10.77% |
| Shared dynamics | 0.7441 | **0.6575** | -11.64% |

Treatment is below the absolute 0.67 boundary in both scopes, but the locked
criterion also required a same-direction >=25% shift from control. It fails.
Moreover, water is the better-calibrated resource despite its *lower* local
sensitivity; this direction does not explain food's slower learning.

Mean resource-specific objective values were 0.01749 food / 0.01368 water in
control and 0.01717 / 0.01094 in treatment. The audit's normalization prevents
that residual-error difference from masquerading as sensitivity.

## Consequence

No gradient projection, parameter separation, resource-specific head, replay
count change, or loss-weight change is licensed. Probe45's mechanism-specific
failure is not caused by treatment-induced food/water gradient conflict or a
large treatment-induced sensitivity shift.

The live remaining explanation is the closed learning-action loop: replay
selection changes shared features, which changes the policy and therefore the
future event stream. The next diagnostic must hold experience fixed and apply
matched uniform versus balanced replay updates from the same checkpoint and
same buffer. That directly tests whether balanced replay helps calibration on a
fixed distribution before on-policy behavior can feed back. It requires a new
preregistration and is not a full training run.

## Claim boundary

This is an optimization result for a grounded bodily predictor. It is not
evidence of reflection, communication, consciousness, or subjective
experience, and it does not justify larger compute or generated data.
