# Probe71: expanded-body composition-law construction result

Date: 2026-08-22. Preregistration:
`docs/decisions/2026-08-22-expanded-body-composition-law-preregistration.md`.
Modules: `src/homesocial/island/expanded_body.py` and
`src/homesocial/organism/composition_law.py`. Artifact:
`runs/organism/probe71_expanded_body/construction_survey.json`.

Five seeds x 100 lives in each of `K = 3` and `K = 5`, seed band
2,500,000,000 with stride 2,000,000. No learned component and no parent
checkpoint.

## Verdict

**Four of six continuation clauses pass. C5 and C6 fail and are not rewritten.** The
five-axis body is viable, its new health axis is load-bearing, the closed-form
composition instrument reproduces its i.i.d. control, and natural individual
bodies vary. But the pooled natural range is **0.02570**, below 0.030, and the
axis gap moves by **-0.01707**, short of the preregistered absolute movement of
0.020.

The continuation rule requires all six clauses. Therefore this construction
does **not** license retraining the probe52 parent or building the
factorized-versus-enumerative speaker treatment.

| clause | locked requirement | result | |
|---|---|---|---|
| C1 legacy inertness | old suite passes; K3 request schema exact | **504 tests + 13 subtests**, exact 3 axes/9 surfaces | **pass** |
| C2 five-axis viability | mean >= 0.75; >= 4/5 seeds >= 0.70 | **0.976**, seed range 0.96-0.99 | **pass** |
| C3 health is real | >= 5% of deaths or requests | **20.70% requests**, 8.33% deaths | **pass** |
| C4 instrument validity | i.i.d. formula MAE <= 0.015 in K3 and K5 | **0.00697**, **0.00350** | **pass** |
| C5 natural range | K5 predicted p90-p10 >= 0.03 | **0.02570** | **fail** |
| C6 axis-gap lever moved | absolute median step >= 0.020 | **-0.01707** | **fail** |

### Scoring correction before the final record

The first generated artifact incorrectly took a quantile of five per-seed
quantiles for C5 and a median of five per-seed medians for C6. The
preregistration says **pooled five-axis lives**. The final scorer pools the
underlying per-life predicted rates and request-level gaps, removes those raw
internal values from the public artifact, and reruns the identical seed band and
sample count. No threshold, body, seed or observation changed.

This correction changes C5 from an erroneous pass at 0.03195 to a fail at
0.02570. C6 remains failed, with the correctly pooled step -0.01707. All numbers
below are from the corrected artifact.

## 1. The body itself works

The construction reuses food, water and energy exactly, promotes the existing
safety variable into the request space, and adds health at a fixed resting
depletion of 0.010. Portions, help period, life length, shocks, birth levels and
individual spread were not retuned.

| body | oracle survival | mean steps | median axis gap | median margin |
|---|---:|---:|---:|---:|
| K3 | 0.950 | 383.08 | 0.08325 | 0.08325 |
| K5 | **0.976** | **392.73** | **0.06618** | **0.06618** |

The higher survival is not a treatment claim: shocks still occur once globally
and are spread over more axes, and the two bodies do not have equal total
burden. Viability here is only a ceiling check. Its useful conclusion is the
weaker one the preregistration asked for: **adding the axes did not make the
fixed help ecology uninhabitable.**

Health is not decorative. It receives **20.70% of all oracle requests**, close
to food's 19.00% and water's 22.49%, and accounts for one of twelve K5 deaths.
Safety receives 11.25% of requests. The ten-message space is naturally used
rather than merely declared.

## 2. The closed-form occupancy law is valid on the process it names

For cell probabilities `p_c`, consequential probabilities `q_c`, and `T`
request opportunities, the preregistered first-use rate is

```
R_T = (1 / T) * sum_c q_c * (1 - (1 - p_c) ** T)
```

The pre-survey amendment separated two controls that are mathematically
different: fixed-count permutation for serial order, and i.i.d. resampling with
replacement for the equation itself. On 31,586 K3 and 32,392 K5 natural request
opportunities:

| body | formula | i.i.d. resample | fixed-count permutation | live order |
|---|---:|---:|---:|---:|
| K3 | 0.01400 | **0.01380** | 0.01425 | **0.02245** |
| K5 | 0.02345 | **0.02318** | 0.02337 | **0.02680** |

The aggregate formula/resample differences are 0.00020 and 0.00026. C4 uses
the stricter mean absolute **per-life** errors, 0.00697 and 0.00350, and both are
inside the locked 0.015 band.

The live stream is not i.i.d., especially in K3: a need that was just helped is
temporarily unlikely to recur, so first uses arrive in an order different from
sampling with replacement. That is why the live-minus-permuted residual is
0.00820 in K3 and 0.00343 in K5. The residual is reported rather than fitted
away. The law is an exchangeable occupancy ceiling, not a full model of
homeostatic request order.

## 3. Natural compositional range exists, but C5 says there is not enough

K5's predicted consequential cold-start rate has a correctly pooled p90-p10
range of **0.02570**, below the required 0.03. Its opportunity-weighted predicted
level is 0.02345, against 0.01400 in K3. The live empirical levels are 0.02680
and 0.02245.

This is the result the old `composition_holdout` could not provide. No cell is
withheld by fiat: individual birth levels, rates, shocks and homeostatic order
determine which need-by-size combinations occur and when. It does **not** show a
factorized learner winning, and C5 says the variation is too narrow to license
the law treatment on this body.

## 4. C6 also fails, narrowly and decisively

The pooled median axis gap falls from 0.08325 to 0.06618, a step of
**-0.0170660**. The
direction is the useful one for a finer decision surface, and the magnitude is
85.3% of the locked requirement. It is still a failure.

The 0.020 clause was written before the body or survey existed, specifically to
prevent a more expensive learner from being justified by a nominal change to
the ecology. Rounding -0.01707 to -0.02, replacing the median with a favourable
quantile, or relaxing the band after seeing the result would do exactly that.
None is done.

Together C5 and C6 close the fixed Probe71 construction as a bridge to parent
retraining. A different fifth-axis rate, help clock, birth distribution, portion
level or message horizon is a different ecology and needs a new argument; it is
not a repair to this result.

## 5. What this does and does not change

What is now established:

- a default-off, axis-generic K5 body can be represented without widening any
  legacy packet or checkpoint;
- the fixed K5 body is viable and both new reportable axes enter requests;
- the closed-form first-use law is correctly instrumented for i.i.d. streams;
- homeostatic serial order is a measurable residual, not ignorable noise;
- the natural ten-message space supplies measurable but sub-threshold predictor
  range.

What is not established:

- No organism discovers or even estimates an axis.
- No factorized speaker beats an enumerative one.
- No word is learned, generated or made load-bearing for survival.
- Probe61 remains closed; no part of its sparse latent was reused.
- No self-model property advances.
- The K5 substrate is an oracle construction layer, not yet the spatial report
  world or a recurrent learner.

## 6. Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.composition_law \
  --lives 100 --seeds 5 \
  --out runs/organism/probe71_expanded_body/construction_survey.json
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The construction survey took under two seconds after import on this machine.
All Probe71 behavior is isolated in new default-off modules. The full repository
passes **504 tests and 13 subtests**.
