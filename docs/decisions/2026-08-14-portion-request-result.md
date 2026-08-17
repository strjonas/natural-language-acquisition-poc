# Result: probe64, a fact about the self that state knowledge cannot supply

Date: 2026-08-14
Preregistration: `2026-08-14-portion-request-preregistration.md`
Ceiling survey: `2026-08-14-portion-request-ceiling-survey.md`
Module: `src/homesocial/organism/portion_request.py`
Artifact: `runs/organism/probe64_portion_request/treatment.json`
Summary: `.venv/bin/python scripts/summarize_probe64.py`

## Verdict

**All seven locked gates pass.** 5 seeds x 40 lives, thresholds fixed before the
run against a survey on a disjoint seed band.

| gate | what it asks | result |
|---|---|---|
| G1 self-model over evidence | `recursive - snap >= 0.05` | **pass 5/5** |
| G2 rate over perfect state | `recursive - state_oracle >= 0.01` | **pass 4/5** |
| G3 rate headroom exists | `individual - state_oracle >= 0.03` | **pass 5/5** |
| G4 no false discovery | null world, `\|recursive - population\| <= 0.005` | **pass 5/5** |
| G5 shuffled readings | `recursive - population <= 0.01` | **pass 5/5** |
| G6 first-use composition | `factored >= 0.90` and `tabular <= 0.60` | **pass 5/5** |
| G7 the size word pays rent | sized beats both constant speakers by `>= 0.10` | **pass 5/5** |

G2 is the decisive gate and it is the one that nearly failed. Its per-seed
margins are +0.0140, +0.0439, **+0.0068**, +0.0463, +0.0643; seed 2 falls below
the locked 0.01 and the gate passes on the other four. That is what a gate
locked at the edge of a surveyed effect looks like when the effect is real but
small, and it is reported rather than smoothed.

## The headline

The caregiver can now be asked for a large or a small portion and carries a
finite basket, so a large portion empties it three times as fast. The right size
is the smallest that covers what this body will burn before that need is served
again -- `rate * 18 ticks`. Every tier speaks by that one rule and differs only
in what it believes it is.

| tier | what it knows | size accuracy | body error |
|---|---|---:|---:|
| `population` | species rates, filtered state. *Today's repository.* | 0.7235 +/- 0.0238 | 0.0931 |
| `snap` | species rates, state corrected at every reading | 0.7839 +/- 0.0227 | 0.0417 |
| `state_oracle` | **the true body every tick**, species rates | 0.9178 +/- 0.0206 | **0.0000** |
| `recursive` | probe63's RLS self-calibration | **0.9528 +/- 0.0241** | 0.0139 |
| `individual` | **its own true rates**, and no readings at all | 0.9930 +/- 0.0015 | 0.0418 |
| `oracle` | true body and true rates | 1.0000 | 0.0000 |

Paired per-seed contrasts, 95% CI over five seeds:

| contrast | difference |
|---|---|
| a learned rate over a **perfect state** | **+0.0350 [+0.0053, +0.0648]** |
| a true rate over a perfect state | +0.0753 [+0.0477, +0.1029] |
| a self-model over the same evidence (`snap`) | +0.1689 [+0.1429, +0.1949] |

Every interval excludes zero. The claim, stated so it can be attacked:

> **An organism that knows what it is and is unsure where it is chooses what to
> say better than one that reads its own body perfectly and believes it is a
> typical member of its species.** `individual` carries body error 0.0418 and
> `state_oracle` carries 0.0000, and `individual` is 7.5 points better at
> choosing the portion.

The decomposition matters as much as the headline, and it closes exactly.
Species and truth disagree on 27.7% of scored ticks. Perfect state knowledge
recovers **19.4** of those points; knowing your own rates while being blind to
your state recovers a further **7.5**; the last **0.7** need both. **Most of this
decision is a state fact.** Probe64 is a claim about the 7.5, and the 7.5 is
where the self-model lives.

## The controls

**Null world** (`metabolic_spread` 0): discrimination 0.000, every tier 1.000,
`recursive` 0.9997 +/- 0.0002. There is nothing to find and nothing is found.

**Shuffled readings** (another organism's body, matched in count and timing):
`recursive` falls to 0.5584 +/- 0.0384 against `population` 0.7234, with rate error
1.109 against the species prior's 0.300. The self-model is not slowed by bad
evidence, it is destroyed by it -- which is what says the readings are what did
the work.

**Rate recovery**: 0.0944 against a species prior error of 0.300, so the
organism ends each life with a materially better estimate of its own burn rate
than "I am typical".

**The `both` world** (metabolic and absorption spread together, reported not
gated): `population` 0.7147, `snap` 0.7568, `state_oracle` 0.8065, `recursive`
**0.8605**, `individual` 0.9907. The ordering holds and the rate gap widens to
5.4 points learned and 18.4 points at the ceiling.

## Property 4: a request it had never made

For a whole life the speaker never says a size word while asking about energy --
132.5 such requests per life -- so the caregiver never once demonstrates what a
size word does to an energy grant. Both listener models watch the identical
transcript. Then each is asked for **both** sizes of energy, which is what stops
a two-word vocabulary being solved by elimination.

| model | first-use accuracy | had any energy evidence |
|---|---:|---:|
| `factored` (one belief per size word, pooled over needs) | **0.9850 +/- 0.0094** | 0.98 |
| `tabular` (one cell per need x size, identical evidence) | **0.5000 +/- 0.0000** | 0.00 |

The table scores exactly chance on every seed, with zero variance, because it
has two empty cells and nothing to fill them from. The factored model answers
correctly from the needs it did ask about.

This is the first thing in this repository that meets property 4's wording --
*it can say things it was never trained to say, because a listener needs them* --
and the boundary is stated in section "What this does not show": the
factorization is given, not discovered.

## The negative, and it is the important half

Under the ration, with the need word pinned at oracle so only the size channel
varies:

| speaker | survival | useful uptake per unit of basket | overflow | large share |
|---|---:|---:|---:|---:|
| sized by `recursive` | 0.605 +/- 0.080 | 0.708 | 6.51 | 0.43 |
| **random size word** | **0.640 +/- 0.077** | 0.656 | 8.42 | 0.51 |
| **no size word at all** | **0.630 +/- 0.083** | 0.655 | 8.31 | 0.50 |
| always large | 0.005 +/- 0.010 | 0.449 | 15.76 | 0.98 |
| always small | 0.145 +/- 0.058 | 0.978 | 0.12 | 0.03 |

G7 passes on a wide margin: a speaker that always says the same thing dies, for
opposite reasons -- always-small starves the body, always-large bankrupts the
caregiver. The size word has rent.

**Its content does not.** A speaker saying a uniformly random size word survives
0.640 against the informed speaker's 0.605, and saying nothing at all survives
0.630. The 3.5-point gap is inside one standard deviation and points the wrong
way. This was predicted in the preregistration from the survey and is confirmed
at five seeds.

The mechanism is visible in the same table. The informed speaker asks large on
0.43 of grants and the random speaker on 0.51; they spend almost the same basket
and differ in *which* windows get the large portion. The informed speaker is
measurably better at the physical thing it is optimizing -- 0.708 useful uptake
per unit of basket against 0.656, and 23% less overflow -- and that advantage
does not become survival, because a need served again every 18 ticks recovers
before a mis-sized portion can kill it. The body's homeostasis absorbs exactly
the quantity the self-model improves.

So the honest summary of probe64 is two sentences that have to be said together:

> The self-model changes what the organism says, by a margin that perfect
> knowledge of its own state cannot reach, and the improvement is real in the
> physical currency the caregiver's basket is denominated in. The world cannot
> yet hear the difference.

## What this does not show

- **That the size rule is optimal or learned.** It is designer-supplied, one
  expression, identical across tiers. What varies is only the belief fed into
  it. The rule-independent endpoints (useful uptake per basket, overflow) move
  in the same direction, which is evidence the rule is not merely self-serving,
  but they are not a proof it is right.
- **That the listener-model factorization is discovered.** It is given. What is
  tested is that a factored model plus a self-model utters a correct novel
  combination while an unfactored model with identical evidence cannot.
- **That the self-model is load-bearing for survival.** It is not, here, and the
  section above says so with the numbers.
- **That anything grammatical happened.** Two words in the existing 60-token
  vocabulary became meaningful because a listener could act on them, taking the
  meaningful protocol from three signals to six. That is
  `md/archive/DIRECTION_2026-07-26.md` section 4's mechanism -- complexity pulled
  by the task -- at its smallest honest scale.
- **Anything about ecologies other than this one.** `help_period` was not
  retuned. Lengthening it would make per-grant placement matter and would
  manufacture a behavioural result; the preregistration forbade it by name and it
  was not done.

## Where the five properties now stand

From `md/archive/DIRECTION_2026-07-26.md` section 2.

1. **Discovered** -- *partial*, unchanged.
2. **Corrigible** -- **yes**, unchanged. Reconfirmed here by G5.
3. **Load-bearing across uses** -- *partial*, unchanged. The model now drives two
   channels of speech rather than one, and damaging it damages both together;
   it still does not reach action.
4. **Productive under novel demand** -- **no -> partial.** The organism produces
   a request it has never made, correctly on first use, because a listener needed
   it, and the unfactored control with identical evidence cannot. Partial rather
   than yes because the factorization is given.
5. **Reflexive** -- *partial*, unchanged.

## What this hands to the next rung

Probe64 converts a vague obstacle into a measured one. The reason a self-model
buys no survival here is not that its content is wrong -- it is 95.3% correct --
but that **the consequence horizon is shorter than the correction interval**. A
need is re-served every 18 ticks, and homeostasis erases a mis-sized portion
before it matters.

That names the next world precisely, and it is a world change rather than a
mechanism change:

1. **Make one wrong request unrecoverable.** A caregiver that serves a need
   rarely -- or one whose help arrives after a delay the organism must predict --
   would put a mis-sized portion beyond the reach of the next grant. This is
   phase C2's territory (`docs/DIRECTION` section 3) and it should be surveyed
   before it is built, at the operating point, per probe63's clause.
2. **Do not reach for it by retuning `help_period`.** That constant is frozen by
   the feasibility run and retuning it to rescue a behavioural gate is the exact
   move this repository forbids. A new lever in a new module, default off, is the
   route.
3. **Self-uncertainty is now more attractive than it was.** G2's seed-2 near-miss
   is a life where the organism's rate estimate was not good enough to beat
   perfect state knowledge, and it had no way to say so. `docs/STATE.md` item 2
   already reopens uncertainty for probe63's ecology; probe64 adds a use for it.
