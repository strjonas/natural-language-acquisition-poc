# Online adaptation under changed body dynamics result

Date: 2026-07-26
Decision: **gates 1 and 2 fail, gate 3 passes 5/5.** Recalibration works; it does
not buy survival. The binding constraint is the planner, not the self-model.

Artifact: `runs/organism/probe59_online_adaptation/`

## Outcome against the locked gates

| Gate | Requirement | Result | |
|---|---|---:|---|
| 1 | adapted beats frozen by >= 10 survival points and lower body error | +2.0 pts (sd 9.5), 1/5 seeds | **fail** |
| 2 | adapted beats stale analytic filter by >= 10 points | **-9.0 pts** (sd 9.5), 0/5 seeds | **fail** |
| 3 | learned constants move toward the new truth | 0.0978 -> 0.0175, 5/5 seeds | **pass** |
| — | shift is informative (frozen drops >= 15 pts) | 44.0 pts, 5/5 | pass |

Per the preregistration, gate 2 was the gate this experiment existed for, and
failing it closes the line. No threshold, budget, learning rate, or shift
magnitude was adjusted after seeing these numbers.

## What actually happened

| Condition | Survival | Body error | Fidelity |
|---|---:|---:|---:|
| adapted | 0.500 +/- 0.095 | **0.0164** | **0.962** |
| frozen | 0.480 | 0.0879 | 0.898 |
| reset | 0.546 +/- 0.093 | 0.0555 | — |
| stale analytic filter | 0.590 | 0.0884 | — |
| refit analytic filter | 0.416 +/- 0.018 | 0.1703 | — |
| **oracle filter, frozen planner** | **0.840** | 3e-10 | — |
| **oracle filter, adapted planner** | **0.504 +/- 0.158** | 3e-10 | — |

Recalibration succeeded decisively. Body error fell 5.4x, mean absolute
constant error fell 5.6x, and every learned constant moved toward its new true
value on all five seeds. The unshifted baseline was 0.92 survival and the shift
cost the frozen model 44 points, so the manipulation was strong and the model
genuinely re-identified the new body.

None of that reached survival.

## The diagnostic that explains it

The last two rows are the result. Holding belief **perfect** (error 3e-10) and
changing only which model selects words: the frozen planner survives 0.840, the
adapted planner survives 0.504.

Adaptation improved the belief and **damaged the planner**. Since both share the
same causal parameters, improving those parameters for prediction made them
worse for decision.

The mechanism is `causal_social_token`, a greedy max-min planner:

```
futures = clip(belief + ticks_to_help * drift + expected_uptake, 0, 1)
score   = min(futures, axis=-1);  token = argmax(score)
```

Maximizing the *minimum* axis is the flaw. A need word concentrates listener
mass on one surface, so it delivers a large gain to a single axis -- but once
that axis passes the second-lowest, `min` stops improving and the remaining
gain is wasted. A word whose listener mass is spread across surfaces delivers a
small gain to *every* axis, which raises `min` directly. Whenever the two
lowest needs are close together, the spreading word wins and the organism stops
asking for what it actually lacks.

Measured on the checkpoints, 2,000 sampled beliefs per cell:

| Belief regime | frozen emits a need word | adapted |
|---|---:|---:|
| one clear deficit | **100.0%** | **100.0%** |
| two lowest needs within 0.05 | 49.6% | **42.2%** |

Both planners are perfect when one need is unambiguously lowest. Both collapse
to near chance when two are tied, and adaptation makes it worse -- learned
uptake rises from 0.600 to 0.901, which amplifies the diffuse residual mass
that spreading words exploit. Aggregated over the full belief distribution the
adapted organism asks for help 83.7% of ticks against the frozen model's 90.9%.

This also explains the paradox of the adapted system's *higher* fidelity:
fidelity is scored only on ticks where a need word was said, so a planner that
falls silent exactly when its objective degenerates looks more truthful while
dying more.

`test_maxmin_planner_abandons_the_deficit_when_two_needs_are_close` pins this.

An earlier draft of this document attributed the failure to clipping saturating
the targeted axis. That explanation was wrong -- a direct test of it failed, and
the belief-regime breakdown above is what the effect actually depends on. The
wrong version is recorded here rather than quietly replaced.

## What this closes and what it does not

It does **not** show that online adaptation fails. Gate 3 passed on every seed
and the belief improved 5x; continual recalibration of a structured causal body
model works, and works fast (40,000 ticks).

It **does** close the framing of A1. The experiment was built to show learning
beating analysis on survival, and survival in this ecology is not controlled by
the self-model. A perfect body model survives 0.840, not 1.0, and the gap
between a perfect model and a good one (0.840 vs the adapted system's 0.500) is
smaller than the gap opened by the planner alone. Aiming a self-model
experiment at a survival endpoint measures the policy.

The `refit_exact_filter` upper bound behaved unexpectedly and is reported as
measured: 0.416 survival at 0.170 body error, worse than the stale filter. The
least-squares refit recovers drift well but estimates portions from percentiles
of observed gains, which the clipping at 1.0 biases downward. It is a weak
estimator, not evidence about the learner, and it should not be cited as an
upper bound in its current form.

## Design errors made and corrected during this experiment

Both are recorded because both would have produced a false pass.

1. **An unsurvivable regime.** The first locked shift (metabolism x2.0,
   portions x0.6) put every condition at 0% survival including the oracle. A
   world nothing can survive cannot discriminate belief sources. The shift was
   re-chosen to be difficulty-neutral using only the oracle and frozen
   conditions, never the adapted one; see the preregistration for the full
   record.
2. **Planner contamination.** The analytic-filter conditions initially used the
   *adapted* model as their planner, so those baselines inherited adaptation's
   effect. The tell was that "oracle" varied 0.35-0.74 across adaptation seeds
   when it should be constant, tracking `adapted_survival`. With the planner
   stated explicitly the oracle is constant at 0.840 and gate 2 flips from an
   apparent 4/5 pass at +18.2 points to a 0/5 fail at -9.0 points.

The uncorrected numbers are the ones that would have been reported as success.

## Next

Two things follow, and neither is a tuning of this mechanism.

1. **The planner is the bottleneck and needs its own preregistered experiment.**
   A max-min objective over a clipped state is degenerate whenever the remedy is
   large relative to the deficit. Scoring on unclipped headroom, or on expected
   deficit reduction, is the obvious repair -- but it is a new mechanism and is
   not to be bolted on to rescue this result.
2. **A2 (structure discovery) is promoted**, as the preregistration specified
   for a gate-2 failure. It also does not depend on survival as its endpoint,
   which this experiment shows is the wrong instrument for self-model claims in
   this ecology.

Future phases should state a **belief-side endpoint** (body error, constant
recovery, calibration) alongside any behavioural one, and should check the
oracle ceiling before locking gates. If a perfect model cannot reach the gate,
the gate is measuring something else.
