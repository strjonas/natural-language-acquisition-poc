# Cross-need gradient geometry preregistration

Date: 2026-07-25
Status: locked before the audit is implemented or run.

## Motivation

Probe45 made replay requests exactly need-balanced and delivered valid eligible
segments on every update, yet food calibration worsened while water improved.
Selected-segment active-need exposure was already close: food 737 and water
750. Another count sampler cannot answer the residual asymmetry.

Multi-label replay makes marginal class balance coupled because one sample can
belong to several classes; Optimizing Class Distribution in Memory formulates
that coupling explicitly rather than treating samples as single-label
(https://arxiv.org/abs/2209.11469). Here the more immediate question is
optimization, not another memory update rule. Gradient Surgery for Multi-Task
Learning identifies negative task-gradient cosine as a direct signature of
detrimental interference
(https://papers.nips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

Probe42 measured calibration gradients against the ordinary bodily objective.
It did not compare food calibration with water calibration. This audit fills
that exact gap without changing a parameter.

## Fixed-checkpoint audit

Run the same read-only audit on the fresh probe45 uniform-control and balanced-
treatment checkpoints.

For each checkpoint, use its fixed policy to collect 128 fresh segments that
contain **both** learner-visible valid-bound food and water restorations, within
20,000 primitive ticks. Eligibility uses public consume action, selected
binding validity, and positive lived own-body delta above 0.175. Simulator
event and kind metadata are not used.

For each segment, compute the exact existing horizon-one plus horizon-two
bound-event objective separately for food and water. Take gradients without an
optimizer step and report food-versus-water geometry in three scopes:

1. all parameters reached by either objective;
2. lexical parameters (`token_embedding`, `binding_value`, `binding_read`);
3. the shared recurrent/dynamics path, excluding the four independent final
   need-output rows.

For each scope report median cosine, negative fraction, raw norms, and the
water/food ratio of `gradient_norm / sqrt(objective)`. The latter is a local
Jacobian-norm proxy that removes the first-order scale effect of one resource
having larger residual error at the checkpoint.

No model parameter, optimizer state, replay buffer, or environment state from
training is mutated. Raw per-segment measurements are retained.

## Locked integrity gates

- Both checkpoints must yield all 128 paired segments within 20,000 ticks.
- Every included segment must have nonzero food and water objectives.
- All geometry and normalized-sensitivity values must be finite.

## Locked interpretation

The treatment has **strong cross-need conflict** only if both lexical and
shared-path median cosine are < 0, both negative fractions are >= 60%, and
each treatment median is at least 0.20 below the matched control. Only that
outcome licenses a gradient-projection or parameter-separation intervention.

The treatment has **aligned cross-need gradients** if both lexical and shared-
path median cosine are >= +0.20 and both negative fractions are <= 25%. That
rules cross-need cancellation out.

A resource-sensitivity asymmetry is large enough to explain the failure only
if the treatment's median error-normalized water/food ratio is > 1.50 or <
0.67 in both lexical and shared scopes, and shifts by at least 25% in the same
direction relative to control.

Anything else is mixed/inconclusive and licenses no optimizer intervention.
If gradients are aligned and sensitivity is not asymmetric, the next test is
a fixed-stream matched update-response audit, isolating replay learning from
the on-policy feedback loop. It is not another full training run.

## Stop rule and claim boundary

Do not modify training from this diagnostic unless its corresponding locked
criterion passes. Do not run another replay capacity, update-count, loss-
weight, or model-size condition. This is an optimization audit of a grounded
body model, not reflection, communication, consciousness, or subjective
experience.
