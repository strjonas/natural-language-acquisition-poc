# Probe71: expanded-body composition-law ceiling preregistration

Date: 2026-08-22. Status: **construction and ceiling-survey
preregistration.** No treatment result, parent retraining or property promotion is
claimed here. This document fixes the five-axis body, the parameter-free
composition predictor, controls, continuation rule and seed bands before either
instrument is implemented or any survey life is run.

Planned modules:
`src/homesocial/island/expanded_body.py` and
`src/homesocial/organism/composition_law.py`. Planned artifact:
`runs/organism/probe71_expanded_body/construction_survey.json`.

## 1. Why construction comes before another learner

Probe70 closed fixed lifetime quotas before a constitution word was built. The
remaining route in `docs/STATE.md` starts by adding bodily axes, for four
independent reasons: it moves probe68's untouched axis-gap term, supplies the
true `K = 5` world requested by probe61's claim boundary, gives probe66 more
than three needs on which to distinguish structures, and grows the current six
need-by-size messages.

That route is expensive. The probe52 lexical/motor parent and every structured
self-model since probe57 assume the current axes and surfaces. Retraining them
before showing that the new body is viable and that its larger message space
actually creates measurable compositional rent would reverse the repository's
oracle-ceiling rule.

So Probe71 begins with two questions and stops there unless both return:

1. Can an oracle keep the fixed five-axis body alive without retuning its
   portions, life length or metabolic spread?
2. Does a closed-form occupancy law predict a non-degenerate first-use cost for
   an enumerative message learner, on natural request streams rather than a
   held-out cell chosen by the experimenter?

## 2. This is a new construction, not a rescue of Probe61

Probe61's sparsity-penalized latent is closed: four of five gates failed. It is
not reopened, widened or refitted here. The five-axis world addresses a claim
boundary that Probe61 itself recorded: its `K = 1..4` sweep could not
discriminate a learner whose recovered dimension merely saturated at four from
one that tracked bodily dimension beyond it.

If this construction passes, a later mechanism may aim probe66's exact Bayesian
model comparison at bodily partitions. That would be a **new mechanism for the
failed discovery phase**: exact comparison among declared structural
hypotheses in a moved `K = 3`/`K = 5` ground truth, not another attempt to rescue
the failed sparse latent. Nothing in this ceiling survey implements or credits
that later mechanism.

## 3. The five-axis body, fixed before implementation

The reportable axes are, in order:

| axis | report word | species resting depletion per tick | origin |
|---|---|---:|---|
| food | `hungry` | 0.008 | existing report axis |
| water | `thirsty` | 0.012 | existing report axis |
| energy | `tired` | 0.016 | existing report axis |
| safety | `safe` | 0.002 | existing real but unreportable body axis |
| health | `hurt` | 0.010 | one new body axis |

The first four rates are already in `ReportConfig`. Health is placed between
food and water, rather than copied from an endpoint, and is never swept. Energy
retains the existing moving cost of 0.035; the ceiling oracle's motor trace is
shared across body conditions, so action-dependent burden cannot favour an arm.

Everything else is inherited unchanged from the report ecology:

- life length 400 ticks, help every 6 ticks;
- small and large portions 0.20 and 0.60;
- global shock probability 0.025 and shock size 0.25;
- birth levels `(0.35, 0.45, 0.55, 0.65, 0.75, 0.85)`;
- individual metabolic spread 0.60, uptake spread zero;
- one deterministic size-word convention shared across needs in the factoring
  world.

At `K = 5`, safety and health receive their own birth draw, individual metabolic
scale, help surfaces and shock surface. Shocks remain one global Bernoulli event
per tick and choose uniformly among the live axes, so adding axes does not add
shock probability. Every help/size and shock consequence has a distinct public
surface identity. New surfaces extend only the expanded world's schema; the
legacy nine-surface packet and every old checkpoint keep their existing widths.

The `K = 3` control is the first three rows, matching Probe61's K3 world where
the unreportable safety axis is frozen. Its report words, rates and six help
surfaces are exactly the existing request subspace. There is no `K = 4`
treatment cell: live-but-unreportable safety already made Probe61's unmodified
body `K = 4`, while the present question is whether the claim and the reportable
message space continue past that boundary.

## 4. The composition law, derived rather than fitted

Let `C` be the set of need-by-size messages, `T` the number of natural request
opportunities in a life, and `p_c` the probability that the oracle needs
combination `c` at an opportunity. An enumerative learner must see a whole cell
before using it; a factorized learner can use a new cell once its need and size
parts are grounded elsewhere.

For the deliberately smallest first-use model, a combination can charge rent at
most once: on its first natural use. Let `q_c` be the probability that using the
correct composed message rather than the enumerative fallback changes realised
help by more than the probe68 margin in that state. Under an exchangeable i.i.d.
request stream, the expected consequential cold-start rate is exactly

```
R_T = (1 / T) * sum_c q_c * (1 - (1 - p_c) ** T)
```

because `1 - (1 - p_c) ** T` is the probability that cell `c` is needed at
least once. There is no fitted coefficient, slope, threshold or extrapolated
ratio. The two factors in `STATE.md` are present explicitly: unseen occupancy
and consequence beyond the ecology's margin.

The enumerative fallback is not chosen after seeing the stream. It is
Probe64's existing `ListenerModel("tabular")`: an unseen cell tries the first
size word in fixed vocabulary order, `more`. In the factoring convention this
already produces the right large portion and the wrong small portion. Thus an
unseen large cell has `q_c = 0`; an unseen small cell compares the useful uptake
of `not` (small) against `more` (large), including clipping at the body's current
headroom. This is exactly the cold-start difference the later treatment will
make, not an oracle-picked worst alternative.

The i.i.d. assumption is not hidden. Homeostatic requests are serially
dependent. The survey therefore reports three quantities separately:

1. the formula from the natural cell frequencies;
2. its empirical first-use/consequence rate on the live ordered stream;
3. the same empirical rate after requests are permuted within life, preserving
   every cell count and consequence label while removing order.

The permuted stream is the instrument control. The live-minus-permuted residual
is the cost of homeostatic serial dependence, not silently folded into a fitted
parameter.

### Pre-survey amendment: distinguish fixed-count permutation from i.i.d. sampling

Recorded during unit implementation, before any survey life or gate quantity
was run. A fixed-count permutation guarantees that every cell observed in the
life remains present. The equation above instead describes i.i.d. sampling with
replacement, under which a rare cell may be absent. Treating those as the same
control would make clause 4 fail for a mathematical reason even on a perfect
instrument.

The survey therefore carries **both** controls:

- `permuted`: shuffle the observed `(combination, consequence)` pairs without
  replacement. This preserves every count and measures serial-order effects.
- `iid_resampled`: draw `T` pairs with replacement from that life's empirical
  pair distribution. This is the direct Monte Carlo control for the closed-form
  equation.

Nothing about the body, formula, thresholds or continuation rule moves. In
clause 4, "permuted control" below is corrected to "i.i.d.-resampled control".
The permutation remains reported and remains the only basis of the serial
residual.

`q_c` is measured with paired counterfactual grants at the request state. The
only allowed consequential rule is the one already licensed by Probe68:

```
margin = min(grant * uptake, gap between the two lowest projected axes)
```

The counterfactual may read the simulator body for audit, as Probe68's diagnostic
does; no learner, speaker or policy receives it.

## 5. Locked construction survey

Conditions: `K = 3` and `K = 5`, factoring size convention only. Five seeds x
100 lives per condition. Survey seed base **2,500,000,000**, stride 2,000,000.
Any later treatment starts no lower than **2,600,000,000**. Paired seed/life
indices are used wherever the two bodies share a stream; new-axis draws come
from dedicated generators and cannot perturb legacy draws.

The ceiling requester knows the true current body, true individual rates and
true convention. The motor trace is body-blind and shared. There is no learned
component and no comparison between estimator families, so Probe69's budget
condition is satisfied by construction: equal simulation ticks and constant
work per axis, reported alongside wall time. A later learned-family comparison
must budget-sweep independently.

Primary construction endpoints:

- oracle survival and death axis;
- number of natural request opportunities and occupied combinations;
- per-life first-use rate;
- axis-gap and Probe68 margin distributions;
- formula, live empirical rent, permuted empirical rent, i.i.d.-resampled rent
  and their absolute errors;
- per-axis request, shock and grant counts.

## 6. Continuation rule

This is a feasibility rule, not a treatment gate. Parent retraining and a
factorized-versus-enumerative treatment are licensed only if all of the
following hold:

1. **Legacy inertness:** the full pre-Probe71 test suite passes, and the `K = 3`
   schema exposes exactly the existing three report axes, words, rates and six
   help surfaces.
2. **Five-axis viability:** mean oracle survival is at least 0.75 and at least
   four of five seeds are at or above 0.70. No help period, portion, rate, birth
   distribution or shock constant may be moved to rescue a failure.
3. **The fifth axis is real:** health accounts for at least 5% of deaths or at
   least 5% of oracle requests. Otherwise it is a decorative state variable and
   the body does not license a larger vocabulary.
4. **Instrument validity:** on the within-life i.i.d.-resampled control, mean
   absolute formula error is at most 0.015 in both `K` conditions. Fixed-count
   permutation is reported separately as the serial-order control.
5. **Natural range:** across the pooled five-axis lives, the 90th-minus-10th
   percentile of predicted consequential cold-start rate is at least 0.03. A
   predictor with no natural range cannot license a law or a larger learner.
6. **The axis-gap lever moved:** the `K = 5` median axis gap differs from `K = 3`
   by at least 0.02, and the direction is stated from the measurement rather
   than preregistered as favourable.

If any clause fails, no neural parent is retrained and no speaker treatment is
built. A changed help clock, hand-picked holdout, longer life or additional
portion level is a new ecology and requires a new argument rather than an
amendment after the result.

## 7. Controls required of the later treatment

These are fixed now so a positive construction cannot weaken them later:

- `factorized`: shares evidence across needs only where its own structural
  posterior licenses sharing;
- `enumerative`: the same observations and counts, one need-by-size table per
  cell;
- `permuted_parts`: preserves token frequency and table capacity while assigning
  learned size content to the wrong need labels;
- `oracle_composition`: knows the convention and bounds the payoff available;
- a moved tangled-convention world, where always factoring must fail and the
  structural posterior must reverse;
- survival and a belief/listener endpoint together; neither substitutes for the
  other.

No family advantage may be called a ceiling unless its development budget is
swept. No gate may use a contrast that mixes state and rate errors.

## 8. What even a positive survey would not show

- No organism discovers an axis, a partition, a word or a grammar.
- The five axis names, dynamics, report words and surface identities are
  designer-supplied.
- The composition equation is a first-use occupancy law, not a theory of syntax
  or open-ended generation.
- A five-axis oracle ceiling says nothing about whether the existing recurrent
  parent can learn the expanded motor/lexical world, which is why that expensive
  step remains conditional.
- One homeostatic ecology, two portion sizes and at most ten consequence-bearing
  messages. This can license the next experiment; it cannot license a language
  model caregiver or the v2 architecture by itself.
