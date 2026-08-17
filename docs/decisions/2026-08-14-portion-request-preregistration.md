# Preregistration: probe64, a fact about the self that state knowledge cannot supply

Date: 2026-08-14
Status: PREREGISTRATION. Written before the treatment is run. Gates below are
locked and are not adjusted after seeing results. A failed gate closes its
mechanism.
Module: `src/homesocial/organism/portion_request.py`
Survey this rests on: `2026-08-14-portion-request-ceiling-survey.md`

## 1. The claim under test

`docs/DIRECTION_2026-07-26.md` section 2 lists five properties. Property 4,
*productive under novel demand*, is the binding constraint and is marked **no**:

> it can say things it was never trained to say, because a listener needs them.
> [no -- untouched, and now the binding constraint. Probe63's organism knows
> something no listener can hear.]

Probe64 gives the listener something to hear. The caregiver can be asked for a
large or a small portion and carries a finite basket, so the size of a request
becomes a decision with a cost. The claim this puts up to be attacked:

> There is a fact about an embodied agent's own body -- how fast it burns --
> that changes what it should say, that reading its own state perfectly does not
> supply, and that an online self-model recovers from intermittent evidence. And
> the agent can produce a request it has never once uttered, correctly the first
> time it needs it, because its model of the listener is factored into what it
> is asking for and how much.

## 2. Why the state cannot substitute, stated so it can fail

One grant arrives per `help_period` and three needs compete for it, so a portion
must carry its need about `help_period * 3 = 18` ticks. What must be covered is
`rate * interval`. Two organisms whose bodies are in identical states, one
burning fast and one slow, should ask for different portions.

The control that makes this falsifiable is `state_oracle`: a tier that **reads
the true body every tick** and believes the species rates. If a self-model does
not beat it, then everything a self-model buys here is state estimation under
another name and the mechanism is closed.

The survey already establishes that most of this decision *is* a state fact --
perfect state knowledge recovers 18.8 of the 26.7 available points -- and that
7.3 points remain that it does not reach. Probe64 is a claim about those 7.3
points and nothing larger. That boundary is fixed here, before the run.

## 3. Mechanism

Nothing is trained. The self-model is probe63's RLS calibrator, the motor policy
is probe52's frozen parent, and the size rule is one expression shared
bit-identically by every tier:

```text
required        = believed_rate * 18
headroom        = 1 - believed_level
delivered(size) = min(portion(size) * believed_uptake, headroom)
                  -> cheapest size whose delivered >= required
                  -> else the size that delivers most, ties to the cheap one
```

Tiers differ in exactly one thing, what they believe they are:

| tier | body belief | rate belief |
|---|---|---|
| `population` | species filter | species |
| `snap` | species filter, corrected at every reading | species |
| `state_oracle` | **true body** | species |
| `recursive` | RLS self-calibration | **learned** |
| `individual` | species filter plus true corrections | **true** |
| `oracle` | true body | true |

The listener model is separate and learns from perception only: the organism
knows what it said and can see which surface appeared beside it (`roots` rather
than `berry`). It never reads `granted_need`, `granted_large`, or any other
simulator metadata, which `docs/STATE.md` bars from listener learning.
`factored` pools one belief per size word across needs; `tabular` keeps one cell
per (need word, size word) from identical observations.

## 4. Operating point, fixed by the survey

- treatment world `metabolic`: `metabolic_spread` 0.60, `uptake_spread` 0.00
- `interoception_probability` 0.03
- `caregiver_store` 30.0, `portion_requests` on
- 5 seeds x 40 lives, seed band `970,000,000` with stride `2,000,000`
- checkpoint `runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz`

The `both` world (`uptake_spread` 0.60 as well) is run and **reported, not
gated**. Its numbers were seen during construction and the treatment world was
chosen before they existed; gating the world with the larger effect after seeing
both would be selection.

## 5. Locked gates

Each is scored per seed and passes at **4 of 5**.

**G1 -- a self-model beats the same evidence without one.**
`recursive` size accuracy minus `snap` size accuracy `>= 0.05`, treatment world.
Phase B1's warning that a corrigibility gate is "satisfiable by clipping" made
`snap` the control probe63 stated its result against; it is the control here too.

**G2 -- the decisive one. A rate belief beats perfect state knowledge.**
`recursive` minus `state_oracle` `>= 0.01`, treatment world.
Margin at the survey point is 0.015. This gate is deliberately tight because it
is the claim. If it fails, probe64's central assertion is false and the result
document says so.

**G3 -- the headroom is real and is not about the learner.**
`individual` minus `state_oracle` `>= 0.03`, treatment world.
An oracle-against-oracle gate: it asks whether this ecology contains rate
information that state knowledge cannot supply, independently of whether
anything recovers it. G2 without G3 would be a learner beating a control by
accident; G3 without G2 would be headroom nothing reaches.

**G4 -- no false discovery.**
Null world, `|recursive - population| <= 0.005`.

**G5 -- the readings are what does it.**
Shuffled readings (another organism's body, matched in count and timing):
`recursive - population <= 0.01`. The self-model must be destroyed, not merely
slowed.

**G6 -- a combination never uttered, correct on first use.**
The speaker never says a size word while asking about energy for a whole life,
so neither model ever sees what a size word does to an energy grant. Both are
then asked for both sizes of energy. `factored >= 0.90` **and**
`tabular <= 0.60`. Asking both sizes is what stops a two-word vocabulary being
solved by elimination: a model with no evidence scores exactly one half whatever
its tie-break is.

**G7 -- the size word pays rent at all.**
At the operating point, with the need pinned at oracle so only the size channel
varies, the sized speaker's survival exceeds **both** `always_large` and
`always_small` by `>= 0.10`.

## 6. What is deliberately not gated, and why

**Survival does not gate the tier ordering.** The survey measured it and it is
flat: every tier lands between 0.700 and 0.767, and a speaker saying a **random**
size word scores 0.733 -- the same as one saying the right one. Gating a
quantity already shown to be flat would be theatre. Probe60's binding addition
is satisfied in the direction it asks about (the belief intervention does change
the selected word), and the behavioural consequence is measured and reported
rather than assumed.

The prediction recorded in advance: **the five tiers will not separate on
survival, and `random_size` will tie the sized speakers.** If the treatment
contradicts this, that is a surprise and is reported as one.

## 7. Failure conditions

- G2 fails: the central claim is false. Do not rescue it by changing the size
  rule, the service interval, the store, the reading rate, the spread, or the
  treatment world. Each is named here so that none can be adjusted afterwards.
- G3 fails: there is no rate headroom in this ecology and the probe is closed
  regardless of G2.
- G5 fails: whatever `recursive` gained was not the readings.
- G6 fails: the factored listener model is not productive and property 4 remains
  untouched.
- G7 fails: the ration does not make the size word worth saying and the whole
  device is decoration.

`help_period`, `portion_small`, `portion_large`, `life_steps` and every other
frozen ecology constant stay frozen. The survey names the temptation explicitly:
lengthening the help period would make per-grant placement matter and would
manufacture a behavioural result. It is forbidden here by name.

## 8. Claim boundaries, written before the result

Whatever happens, probe64 will **not** show:

- that the size rule is optimal, or learned. It is designer-supplied, one
  expression, identical across tiers. What varies is only the belief fed into
  it.
- that the factorization of the listener model is discovered. It is given. What
  is tested is that a factored model plus a self-model utters a correct novel
  combination while an unfactored model with identical evidence cannot.
- that the self-model is load-bearing for survival. The survey says it is not,
  and section 6 says so in advance.
- anything about a productive grammar. Two words became meaningful, from six to
  a possible twelve, and they became meaningful because a listener could act on
  them. That is the mechanism `docs/DIRECTION_2026-07-26.md` section 4 asks for
  and it is a small instance of it.
