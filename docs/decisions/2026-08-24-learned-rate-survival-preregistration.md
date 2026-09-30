# Probe73: does the learned individual rate improve survival? — preregistration

Date: 2026-08-24. This document fixes the construction, endpoints, cells,
sample, seed band, gates and failure rule before the treatment runner is
implemented and before any treatment life is generated.

Parent results:

- `docs/decisions/2026-08-16-future-request-result.md` (Probe65), which
  predicted the regret crossover before measuring it and fixed roughly 700
  lives per arm as the sample needed to resolve a survival step near 0.03;
- `docs/decisions/2026-08-22-granularity-survival-result.md` (Probe72), which
  established a clean positive physical ceiling for individual-rate knowledge
  while closing the finer-clock interaction;
- `runs/organism/probe73_learned_rate_survival/granularity_diagnostic.json`,
  generated before this preregistration on the diagnostic band below.

## 1. The claim under test

Probe65's recursive self-model beat perfect state knowledge on named-need
accuracy and regret at long horizons. At lag 18 it also survived 0.345 against
0.315, but 200 lives per arm could not distinguish that +0.030 from zero. The
comparison was not clean: `recursive - state_oracle` changes both state and
rate error.

Probe72 repaired the physical half of that ambiguity with oracles. In its fine
ecology, true individual rates survived +0.0214 better than species rates with
current state perfect in both arms, on 700 lives per arm. That is a physical
ceiling rather than a learned result. The proposed help-clock amplifier failed,
so this experiment returns to the unchanged baseline ecology and asks the
remaining one-factor question:

> Holding the recursive self-model's filtered state and uptake belief fixed,
> does allowing request planning to read its learned metabolic rates improve
> survival over forcing that same planner to use species rates?

This is the first survival gate in the repository placed on the learned rate
itself.

## 2. Fixed ecology and the two preconditions

The ecology is Probe65's metabolic world with no Probe68 quantum treatment:

- `help_delay = 24`, past the preregistered regret crossover;
- `help_period = 6`, portion scale `1.0`, unlimited caregiver store;
- metabolic spread `0.60`, uptake spread `0.00`, interoception probability
  `0.03`;
- frozen Probe52 motor parent, 400-tick maximum life, unchanged shocks, body,
  request ledger and `E[min(next body)]` word rule.

Lag 24 was named by Probe65 and by `docs/STATE.md` before this experiment. It is
not selected from a new survival sweep. Probe67 independently measured the
recursive point arm at 0.2100 survival and the true-rate `individual` arm at
0.2375 over 400 lives per arm at this lag. Those arms do not share the matched
state used below, so that +0.0275 supplies a direction and scale, not the answer
to this probe.

### Granularity check

The required Probe68 diagnostic was run before this document on eight lives at
lag 24, seed base `2,900,000,000`. At the unchanged portion scale it measured:

- median consequential decision margin `0.13936`;
- recursive point-error median differential `0.05208`, p90 `0.18334`;
- recursive point-error reach on consequential decisions `0.2552`;
- species-rate state-oracle reach `0.7558`.

The belief error can therefore reach the existing word surface without changing
the help clock. The treatment does not halve the quantum again; Probe72 closed
that as a survival-value amplifier.

### Budget check

Every cell runs the same recursive calibrator and the same frozen motor parent
on every decision. The lesion computes the learned rate and then withholds it
from the request rule, so estimator and inference compute are matched. Nothing
trains and no mechanism family is compared. This probe therefore makes no
ceiling claim of the kind Probe69 requires a budget sweep to support.

## 3. The one-factor cells

All three cells instantiate `SelfModelTier("recursive")`. They use its exact
current point estimate, exact uptake belief, readings, update rule, ledger and
horizon. Only the rate vector passed to the already-existing request rule
changes:

| cell | request-planning rates | state and uptake |
|---|---|---|
| `species_rate` | species rates | recursive tier |
| `learned_rate` | recursive tier's learned rates | recursive tier |
| `true_rate` | this life's true rates | recursive tier |

Thus the primary contrast changes one input:

    LEARNED_RATE_VALUE = survival(learned_rate) - survival(species_rate)

The ceiling contrast is:

    TRUE_RATE_VALUE = survival(true_rate) - survival(species_rate)

The true-rate cell is not a learned self-model. It asks whether rate knowledge
has physical value when the state machinery is matched exactly, rather than
reusing Probe72's perfect-state ceiling.

Each cell also scores both learned-rate and species-rate counterfactual words on
its current state and ledger before the actual word steers the life. Their cost
is measured with Probe65's true `need_scores`:

    MATCHED_REGRET_VALUE = regret(species-rate word) - regret(learned-rate word)

The locked belief-side panel is the value on `learned_rate` histories. It holds
trajectory, state, uptake, outstanding requests and decision time fixed; only
the rates used to choose the two counterfactual words differ.

Rate recovery is measured on the same histories as mean absolute per-tick error
between the learned and true rate vectors, against the corresponding error of
the species vector.

## 4. Sample, seed isolation and intervals

Five paired seed blocks x 140 consecutive lives per cell = **700 lives per
arm**, 2,100 treatment lives total. All cells see the same life seeds.

- granularity diagnostic band: `2,900,000,000`;
- treatment band: `3,000,000,000`;
- seed-block stride: `2,000,000`;
- maximum treatment seed used: `3,008,000,139`.

The bands are disjoint from Probe72's latest treatment band at 2.8 billion and
from one another. Treatment order is `seed -> cell -> chunk`, completing one
paired block before starting the next. Chunks contain 20 lives and exist only
for durable progress; the five seed blocks remain the statistical units.

Means and two-sided 95% Student-t intervals are computed over five paired block
values. Positive gates also require at least 4/5 blocks to have the predicted
sign. No interval treats the 700 lives as 700 independent model replications.

## 5. Locked gates

All gates must pass.

| gate | requirement |
|---|---|
| **G1: learned rate reaches survival** | `LEARNED_RATE_VALUE > 0`, its paired 95% interval excludes zero, and at least 4/5 blocks are positive. |
| **G2: matched physical ceiling exists** | `TRUE_RATE_VALUE > 0`, its paired 95% interval excludes zero, and at least 4/5 blocks are positive. |
| **G3: matched regret predicts the sign** | `MATCHED_REGRET_VALUE > 0` on `learned_rate` histories, its paired 95% interval excludes zero, and at least 4/5 blocks are positive. |
| **G4: the rate was learned** | `species rate MAE - learned rate MAE > 0` on `learned_rate` histories, its paired 95% interval excludes zero, and at least 4/5 blocks are positive. |
| **G5: the construction remains viable** | Mean `true_rate` survival is at least `0.15`. |

G1 is the probe. G2 prevents a learned null from being blamed on a construction
with no one-factor headroom. G3 is the belief-to-decision endpoint in the
currency survival is denominated in. G4 prevents a behavioural difference from
being attributed to a rate estimator that did not improve its rate estimate.
G5 prevents a near-dead ecology from turning a handful of survivors into an
epistemic claim; its floor is below the independent Probe67 value of 0.2375.

## 6. Failure and continuation rules

- If G1 fails while G2-G5 pass, the learned rate has no demonstrated survival
  value at the precommitted 700-life resolution in this fixed lag-24 ecology.
  Do not rescue it by increasing the lag, halving the help clock, adding a
  store, moving the spread or relaxing the interval.
- If G2 fails, this matched-state construction lacks physical rate headroom.
  Probe72's separate perfect-state ceiling remains true, but cannot be promoted
  into this learner result.
- If G3 fails, regret does not provide the predicted bridge from the matched
  rate correction to the live outcome, even if G1 happens to pass.
- If G4 fails, no learned-rate claim is made.
- If G5 fails, the construction is invalid and no epistemic result is claimed.

No gate, arm, sample, lag, seed band, ecology constant or contrast changes after
seeing a treatment life. A cross-life self prior is a later mechanism and is not
added here regardless of the result.

## 7. Claim boundary fixed in advance

Passing would show that one existing online bodily self-calibrator learns an
individual metabolic rate whose use in a fixed three-word request rule improves
survival over a rate lesion with matched state, uptake, evidence, compute and
planner. It would advance the load-bearing property from an oracle ceiling to a
learned physical use.

It would not show a learned or optimal planner, discovered bodily structure,
cross-life self-knowledge, uncertainty reporting, language generation,
compositional expansion, generalization to another ecology, consciousness or
sentience. The body is still three needs and the mouth still selects one of
three designer-provided need words.
