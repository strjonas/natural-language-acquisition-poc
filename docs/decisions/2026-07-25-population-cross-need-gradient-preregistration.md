# Population cross-need gradient preregistration

Date: 2026-07-25
Status: locked after count-only feasibility and before any population gradient
is computed.

## Feasibility basis

Probe46's same-segment design failed because only 20 both-resource segments
appeared in 20,044 ticks. A read-only count-only pass, explicitly allowed by
the handoff, measured independent resource populations over the same fixed
20,000-tick ceiling:

| Checkpoint | Total segments | Food eligible | Water eligible | Both |
|---|---:|---:|---:|---:|
| Probe45 uniform control | 356 | 113 | 147 | 20 |
| Probe45 balanced treatment | 367 | 105 | 144 | 21 |

No objective, gradient, optimizer update, replay insertion, or training was
computed in this feasibility pass. These counts lock **96 food segments and 96
water segments per checkpoint**. The failed probe46 samples are not reused.

## Question

Probe45 equalized active-need replay exposure but food remained underfit. Does
the mean food calibration gradient conflict with the mean water calibration
gradient in the lexical/shared pathway, or does one resource have a materially
different local sensitivity?

Multi-task gradient methods compare gradients of task minibatches; they do not
require both tasks to occur in the same example. Negative task-gradient cosine
is the relevant interference signature in Gradient Surgery for Multi-Task
Learning
(https://papers.nips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).
Multi-label replay research likewise treats overlapping membership as a
coupled distribution problem, rather than requiring paired examples
(https://arxiv.org/abs/2209.11469).

## Fixed-checkpoint audit

Run the same read-only measurement on the fresh probe45 uniform-control and
balanced-treatment checkpoints.

1. Follow each fixed checkpoint policy for at most 20,000 primitive ticks.
2. Retain the first 96 segments with a learner-visible valid-bound food
   restoration and independently the first 96 with a valid-bound water
   restoration. A both-eligible segment enters both populations, as it did in
   the lived training objective.
3. For each retained segment, compute the already implemented exact
   horizon-one plus horizon-two bound-event constituent for only that resource.
4. Take gradients without an optimizer step. Average gradient vectors within
   resource, then compare mean food versus mean water gradients in:
   - all reached parameters;
   - lexical parameters (`token_embedding`, `binding_value`, `binding_read`);
   - the shared recurrent/dynamics path, excluding independent final need rows.
5. Report aggregate cosine, norms, and the water/food ratio of
   `mean-gradient norm / sqrt(mean objective)`, a local sensitivity proxy.

Use 256 deterministic multinomial bootstrap resamples with seed derived only
from the fixed experiment seed. Resample the 96 food and 96 water populations
independently, preserve the same bootstrap indices across checkpoints, and
report cosine median, 5th/95th percentiles, negative fraction, and median
normalized-sensitivity ratio. Bootstrap outputs assess stability; they do not
alter thresholds or select hyperparameters.

Raw per-segment tick, objective, gradient norm, and normalized sensitivity are
retained. Full gradient vectors are transient and are not serialized.

## Integrity gates

- Each checkpoint must yield all 96 food and 96 water samples within 20,000
  ticks.
- Every included objective must be positive and every reported value finite.
- No model parameter, optimizer state, replay buffer, or training environment
  state is mutated.
- Fixed checkpoint evaluation must remain deterministic across a repeated
  two-sample smoke.

## Locked interpretation

The balanced treatment has **strong cross-need conflict** only if:

- aggregate cosine is < 0 in both lexical and shared scopes;
- bootstrap negative fraction is >= 95% in both scopes; and
- each treatment aggregate cosine is at least 0.20 below the matched control.

Only that outcome licenses a gradient-projection or parameter-separation
training preregistration.

The treatment has **aligned cross-need gradients** if aggregate cosine is >=
+0.20 and bootstrap negative fraction is <= 5% in both lexical and shared
scopes. That rules cross-need cancellation out.

A resource-sensitivity asymmetry is large enough to explain the treatment
failure only if its median bootstrap water/food normalized-sensitivity ratio is
> 1.50 or < 0.67 in both lexical and shared scopes, and the ratio shifts by at
least 25% in the same direction relative to control.

Anything else is mixed/inconclusive and licenses no optimizer intervention.
If gradients are aligned and sensitivity is not asymmetric, the next test is
the handoff's fixed-stream matched update-response audit, isolating learning
from the on-policy feedback loop. If integrity fails, stop without extending
the tick ceiling or reducing the locked sample count.

## Claim boundary

This is a read-only optimization diagnostic of a grounded body model. It is not
a reflection, communication, consciousness, or subjective-experience test. It
licenses no larger compute, generated data, or full training condition by
itself.
