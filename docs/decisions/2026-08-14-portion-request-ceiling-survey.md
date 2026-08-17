# Ceiling survey: is a portion size worth asking for, and is it a fact about the self?

Date: 2026-08-14
Status: survey. No gate is locked here. Instruments only.
Module: `src/homesocial/organism/portion_request.py`
Artifact: `runs/organism/probe64_portion_request/ceiling_survey.json`

Probe60 made it binding to check the oracle ceiling before locking a gate.
Probe63 added two clauses: check it **at the operating point the gate will be
scored at**, and check the ceiling instrument is not undersampled. This survey
does both, and adds a third precaution of its own -- it is run on a seed band
(`950,000,000`) disjoint from the treatment band, so no threshold is read off a
life a gate then scores.

## What is being surveyed

`docs/STATE.md`'s first item of exact next work:

> **Make the self-model pay rent in speech.** Probe63's organism knows something
> no listener can hear: *how fast it burns*. [...] A listener that could grant a
> **large or small** portion on request, or grant **early**, would make "I burn
> fast" worth saying.

Two default-inert levers on `ReportConfig` implement that listener:

| lever | what it does |
|---|---|
| `portion_requests` | the caregiver hears a size word beside the need word and grants that class instead of drawing one. The draw is still consumed, so a world where nothing is asked for is bit-identical. |
| `caregiver_store` | a basket rather than a spring: total portion mass for the whole life. A large portion spends three times a small one, and a grant the caregiver cannot afford is not made. Zero is the old unlimited caregiver. |

Neither is meaningful alone. Without the first there is nothing to ask; without
the second "large" dominates and the size word is decoration.

Two questions have to be answered before anything can be preregistered.

1. **Does the size question discriminate?** How often does believing you are a
   typical member of your species give a different answer from the truth? If
   rarely, a self-model has nothing to contribute and the gate would measure
   noise.
2. **Does the ration bite without flooring the endpoint?** A basket so large
   that always-large wins makes the word decoration; one so small that nothing
   survives makes survival a floor.

## The size rule

Every tier speaks by one rule, shared bit-identically, with `interval` fixed a
priori at `help_period * 3 = 18` -- the caregiver's own clock, not a parameter
of any organism:

```text
required  = believed_rate * interval
headroom  = 1 - believed_level
delivered(size) = min(portion(size) * believed_uptake, headroom)
                 -> cheapest size whose delivered >= required
                 -> else the size that delivers most, ties to the cheap one
```

`required` is a fact about the **rate**. `headroom` is a fact about the
**state**. That the two enter separately is the whole design: it makes it
possible to ask which of them the decision actually needs.

An earlier version of this rule fell back to *large* whenever nothing sufficed,
which pays triple for a portion that falls off the top of a full body. That is
not a defensible rule under a ration and it was corrected before any measurement
was recorded. The correction moved `state_oracle` from 0.744 to 0.890 on the
first exploratory run -- it made the state term matter *more*, against the
interest of this probe's hypothesis, which is the direction an honest correction
is allowed to go.

## Result 1: the question discriminates, and most of it is a state fact

Treatment world `metabolic_spread` 0.60, reading rate 0.03, store 30, 40 lives,
every tier on one shared history with the need named by one shared driver, so
the only thing varying is the belief that sets the size.

| tier | what it knows | size accuracy |
|---|---|---:|
| `population` | species rates, filtered state. *Today's repository.* | 0.7338 |
| `snap` | species rates, state corrected at every reading | 0.7838 |
| `state_oracle` | **the true body every tick**, species rates | 0.9115 |
| `recursive` | probe63's RLS self-calibration | 0.9274 |
| `individual` | **its own true rates**, and no readings at all | 0.9925 |
| `oracle` | true body and true rates | 1.0000 |

Species and truth disagree on 26.6% of scored ticks, so 0.734 is exactly the
headroom `population` leaves. Reading the decomposition off that:

- **Perfect state knowledge buys 17.8 of the 26.6 available points.** Most of
  the size decision is a state fact, and this survey says so before any gate
  claims otherwise.
- **8.1 points remain that perfect state knowledge does not reach**, and knowing
  your own rates while being *blind* to your state reaches them: `individual`
  0.9925 against `state_oracle` 0.9115. That gap is between two oracles and does
  not depend on any learner working.
- The learner captures 1.6 of those 8.1 points (`recursive` 0.9274). That is the
  quantity a gate can be locked on, and it is small enough that a gate on it may
  well fail.

In the `both` world -- metabolic and absorption spread together -- the same
ordering holds with wider gaps: `population` 0.618, `snap` 0.683, `state_oracle`
0.779, `recursive` 0.842, `individual` 0.992, discrimination 0.382. The
treatment world stays `metabolic`, matching probe63, because it was chosen
before these numbers existed; `both` is reported and not gated.

Null world (`metabolic_spread` 0): discrimination 0.000, every tier 1.000,
`recursive` 0.9998. There is nothing to find and nothing is found.

Shuffled readings (another organism's body, matched in count and timing):
`recursive` falls to 0.587, *below* `population`'s 0.733, with rate error 1.396.
The control destroys the mechanism, as it must.

## Result 2: the ration bites, and the operating point is a store of 30

Oracle speaker, need pinned at oracle so only the size channel varies, 40 lives.

| store | speaker | survival | overflow | uptake efficiency | refusals/life |
|---:|---|---:|---:|---:|---:|
| unlimited | sized | 0.750 | 7.46 | 0.691 | 0.0 |
| unlimited | always large | 0.750 | 21.38 | 0.436 | 0.0 |
| unlimited | always small | 0.175 | 0.14 | 0.981 | 0.0 |
| 34 | sized | 0.725 | 7.42 | 0.691 | 0.1 |
| 34 | always large | 0.150 | 18.84 | 0.440 | 5.6 |
| 30 | sized | 0.725 | 7.27 | 0.693 | 0.7 |
| 30 | always large | **0.000** | 16.38 | 0.446 | 7.5 |
| 30 | always small | **0.175** | 0.14 | 0.981 | 0.0 |
| 26 | sized | 0.500 | 6.76 | 0.698 | 2.1 |
| 22 | sized | 0.350 | 5.98 | 0.706 | 3.4 |

A store of 30 is the operating point. Both constant policies fail there and for
opposite reasons -- always-small starves the body, always-large bankrupts the
caregiver -- while the sized speaker keeps all of the unlimited-store survival.
At 26 and below everything degrades toward a floor; at 34 the ration barely
constrains the sized speaker.

## Result 3: the negative that shapes the whole preregistration

Same conditions, store 30, need pinned at oracle, only the size word varying.

| speaker | survival | large share | overflow |
|---|---:|---:|---:|
| `population` sizes | 0.675 | 0.41 | 6.62 |
| `snap` sizes | 0.750 | 0.44 | 7.30 |
| `state_oracle` sizes | 0.750 | 0.45 | 7.68 |
| `recursive` sizes | 0.725 | 0.45 | 7.60 |
| `individual` sizes | 0.725 | 0.43 | 6.98 |
| `oracle` sizes | 0.725 | 0.44 | 7.27 |
| **random size word** | **0.750** | 0.49 | 8.75 |
| **no size word at all** | **0.725** | 0.49 | 8.90 |

**Survival does not order the tiers, and a random size word does exactly as well
as a correct one.** The size word has rent -- both constant policies collapse --
but its *content* buys nothing this ecology can feel. The whole spread across
six tiers is 0.075, and the best of them is `snap` and `state_oracle`, neither
of which knows what it is.

The reason is visible in the same table. Every sized speaker asks large on
0.41--0.45 of grants and the random speaker on 0.49; they spend almost the same
basket and differ only in *which* windows get the large portion. A need served
every 18 ticks is served again before a mis-sized portion can kill it, so
placement is precisely what the body's homeostasis absorbs. This is the same
wall probes 59, 60 and 62 hit, measured here from a new direction.

Two consequences, both binding on the preregistration that follows:

1. **Survival is not gated on tier ordering.** Doing so would gate a quantity
   this survey has already shown to be flat. What *is* gated behaviourally is
   the one behavioural effect that is real and large: the sized speaker against
   the two constant speakers.
2. **The claim boundary is fixed in advance.** Probe64 can claim that the
   self-model changes what is said and that no amount of state knowledge
   supplies it. It cannot claim the world hears the difference, and the result
   document must say so in those words.

## Exploratory runs spent during construction

Recorded because they are how the thresholds were chosen, and because the lives
they touched are therefore not available to a gate.

- Seed band `960,000,000`, 40 lives, run while the module was being built. It
  reproduces Result 1 closely on a different band -- `population` 0.733, `snap`
  0.787, `state_oracle` 0.921, `recursive` 0.946, `individual` 0.994,
  discrimination 0.267 -- and adds the two conditions the definitive survey does
  not carry:
  - shuffled readings: `recursive` 0.587 against `population` 0.733, rate error
    1.396. The control destroys the mechanism, as it must.
  - the `both` world: `population` 0.618, `snap` 0.683, `state_oracle` 0.779,
    `recursive` 0.842, `individual` 0.992, discrimination 0.382. Wider gaps than
    the treatment world, which is exactly why the treatment world was not
    changed to it after the fact.
  - composition holdout: factored 1.000, tabular 0.500, tabular evidence 0.000
    of lives.
- An earlier six-life run under the uncorrected size rule, superseded.

The 1.6-point margin `recursive` holds over `state_oracle` on the definitive
band is smaller than the 2.5 points the construction band showed. G2 is
therefore locked at a threshold this survey does not comfortably clear, and it
is locked there anyway because it is the claim.

The treatment band therefore starts at `970,000,000`, above everything spent
here, with the 2,000,000 stride probe63 established. `950,000,000` carries this
survey. No band overlaps another.

## What this survey does not establish

- Nothing about a learner. Every number above is an instrument.
- Nothing about whether the size rule is the *right* rule. It is a rule a
  rational planner would use under a ration, it is designer-supplied, and it is
  identical across tiers. The physical endpoints in Result 3 do not depend on
  it, and they are the ones that came out flat.
- Nothing about worlds other than this ecology at these frozen constants. The
  `help_period` was not retuned to make placement matter, and must not be: it is
  frozen by the feasibility run recorded in `runs/organism/report_feasibility/`.
