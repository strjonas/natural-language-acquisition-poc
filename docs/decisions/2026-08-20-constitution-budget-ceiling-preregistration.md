# Constitution report: oracle-budget ceiling preregistration

Date: 2026-08-20. Status: **ceiling-survey preregistration.** No treatment gate
is locked here and no language mechanism exists yet. This document fixes the
ceiling instrument, sweep, controls, continuation rule and seed band before the
instrument is implemented or any survey life is run.

Planned module: `src/homesocial/organism/constitution_budget.py`. Planned
artifact: `runs/organism/probe70_constitution_budget/ceiling_survey.json`.

## 1. The question that comes before a mechanism

Probe63's organism recovers its individual metabolic rates, but every report it
can make is still about state: which need is lowest now or will be lowest when a
delayed grant arrives. The route in
`docs/decisions/2026-08-18-architecture-audit-and-route.md` proposes the first
report about constitution instead: *I am the kind of body that burns water
fast*.

That report is worth building only if a listener can do something with it that
the existing per-request protocol cannot. The candidate is the listener's
private finite basket. Probe64's speaker can choose the need and portion on each
grant, but it cannot perceive `ReportWorld._store_remaining` and cannot reserve
that store across a whole life.

So the ceiling question is deliberately prior to learning and language:

> Does a caregiver given this life's **oracle individual rates** allocate one
> fixed basket across bodily needs better than the existing first-come
> caregiver, and at what basket size?

If not, a word for the rates has no consequence to ground it and the mechanism
is not built.

## 2. What stays fixed

- Probe64's metabolic world: `metabolic_spread = 0.60`, uptake spread zero,
  interoception probability 0.03.
- Probe64's portions, help period, shocks and 400-tick life. None is retuned.
- The frozen probe52 parent and its motor policy.
- An **oracle speaker** in every arm: true current body, true individual rates,
  the same need rule and the same portion rule. This is the strongest existing
  first-come requester and isolates the caregiver-side allocation.
- Probe64's already-established size-word convention is initialized as known in
  every arm. Spending the first two grants rediscovering an old word would mix
  listener exploration into a ceiling about basket allocation, especially when
  one arm can refuse the very grant that supplies the lexical evidence.
- Paired life seeds across policies and store cells. Trajectories may diverge
  only after the caregiver makes a different allocation.

There is no learned component in this comparison. Both allocators make one
constant-time decision per grant, so the budget requirement introduced by
probe69 is satisfied by construction: equal simulation ticks, equal motor
compute, and no losing training curve whose saturation could be in question.

## 3. The allocator, fixed before implementation

The caregiver divides its lifetime store into one account per reportable need.
For need `i`, its predicted burden is

    burden_i = metabolic_rate_i + shock_probability * shock_size / 3

Energy uses the public 0.5 moving / 0.5 resting mixture already used to
initialize `MoveFraction`. Uptake is one in the treatment world; the general
expression divides burden by uptake. The account is

    quota_i = store * burden_i / sum(burden)

At a grant boundary, a requested portion is approved while that need's charged
mass is still below its quota. The final discrete portion may cross the quota;
the next one is refused. This is the unique no-free-parameter rounding rule that
does not make a fractional quota impossible to spend. The existing global store
check remains authoritative, so the caregiver can never overspend the basket.

This is a **lifetime share allocator**, not a state oracle. It reads the
constitution supplied at birth, the need and size currently requested, its own
private accounts, and the public ecology constants. It never reads the true
body, future shocks, future actions or a simulator death label.

## 4. Arms and controls

| arm | account weights | purpose |
|---|---|---|
| `first_come` | none | probe64's existing caregiver; spends until empty |
| `oracle_rates` | this life's true per-need burden | ceiling candidate |
| `species_rates` | species-average burden | budgeting without individual self-knowledge |
| `permuted_rates` | oracle burden cyclically reassigned across need labels | content lesion |

The permuted control preserves the exact multiset and sum of oracle burdens. It
therefore gives the caregiver the same total demand, the same quota sizes and
the same arithmetic while making the claim *which need burns how fast* false.
If it performs like `oracle_rates`, a constitution-specific claim fails even if
budgeting in general helps.

At `caregiver_store = 0`, the basket is unlimited and allocation is bypassed.
All four arms must be tick-identical there; this is the inertness control.

## 5. Locked survey

Stores: **0, 34, 30, 28, 26, 24, 22**. The finite anchors 34, 30, 26 and 22 are
probe64's published sweep; 28 and 24 fill its two four-unit gaps rather than
moving either endpoint after seeing this allocator.

Five seeds x 40 lives per arm and cell. Survey seed base **2,300,000,000**,
stride 2,000,000. The band is disjoint from probe69's latest 2,100,000,000 audit
band and is paired across arms. A future treatment, if licensed, starts no lower
than 2,400,000,000.

Primary endpoint: paired survival difference `oracle_rates - first_come`.
Secondary endpoints: mean life steps, store spent, useful uptake per unit store,
refusals, grants by need and death cause. The latter explain a difference; they
do not substitute for survival.

## 6. Continuation rule

This is a feasibility rule, not a treatment gate. The constitution mechanism is
licensed only if, at a finite store where first-come survival is neither a floor
nor a ceiling:

1. `oracle_rates - first_come >= 0.10` in mean survival;
2. the paired difference is positive on at least four of five seeds;
3. `oracle_rates` beats both `species_rates` and `permuted_rates` at that cell;
4. the advantage is not an isolated spike: an adjacent finite store also has a
   positive mean oracle advantage; and
5. the unlimited-store inertness control is exact.

If no cell satisfies all five, this allocator closes before a constitution word
is built. A different allocation algorithm would be a different mechanism and
would need a new argument and survey; a failed result is not repaired by moving
the store, changing the speaker, lengthening the life or retuning the portions.

## 7. Probe68's granularity check

The intervention is approval versus refusal of the requested portion. Its
realized body effect is the useful uptake of that portion, bounded above by 0.20
or 0.60 body units and below by zero when the body is full. The survey records
useful uptake and refusal timing so the result can state whether the allocator's
effect actually exceeds the ecology's decision margin rather than assuming that
every refusal matters. A treatment gate will not be locked unless the operating
cell clears that check.

## 8. What a positive survey would not show

- No organism reports a rate. The caregiver is handed oracle values by the
  instrument.
- No rate is discovered, generalized or linguistically composed here.
- The allocation rule is designer-supplied and is one rule, not an optimality
  proof over all possible rationing policies.
- A survival advantage would license building the report mechanism. It would not
  establish any of the five self-model properties by itself.
- One frozen parent, one three-need ecology and one finite-store family.
