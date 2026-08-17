# Result: probe66, discovering that a word's meaning transfers

Date: 2026-08-16
Preregistration: `2026-08-16-discovered-convention-preregistration.md`
Ceiling survey: `2026-08-16-discovered-factorization-ceiling-survey.md`
Module: `src/homesocial/organism/discovered_convention.py`
Artifact: `runs/organism/probe66_discovered_convention/treatment.json`

## Verdict

**All six locked gates pass.** 5 seeds x 40 lives, thresholds fixed before the
run against a survey on a disjoint seed band.

| gate | what it asks | result |
|---|---|---|
| G1 generalizes where licensed, at no cost | speaks `>= 0.95`, right within 0.02 of `factored` | **pass 5/5** |
| G2 declines where it is not | tangled world, speaks `<= 0.75` | **pass 5/5** |
| G3 the payoff | confidently wrong `>= 0.10` less often than `factored` | **pass 5/5** |
| G4 **the recovered structure tracks the world** | posterior gap `>= 0.12` | **pass 5/5** |
| G5 the given factorization fails | tangled, `factored <= 0.60` | **pass 4/5** |
| G6 the tabular control is chance | exactly 0.500 in both worlds | **pass 5/5** |

G5 is the gate that says there is anything to discover, and seed 3 is the one
that misses it: `factored` scored 0.650 there, above the locked 0.60, because
that seed's draws happened to leave the three needs mostly agreeing. It passes
4/5 and that is reported rather than smoothed.

## What was given, and what is now found

Probe64's `factored` listener holds one belief per size word pooled over needs --
*this word means large, whatever I am asking for* -- and that pooling is what
lets it utter a combination it has never uttered. The designer chose to pool.
`tangled_size_words` lets the caregiver draw its word-to-size orientation **per
need**, so "more" may be large for food and small for water, and then asks
whether the organism can tell the difference.

The mechanism has no free parameters. Before generalizing what a word does to a
need it has never used that word for, the organism asks whether the needs it
*has* used it for agree with each other -- two exact Beta-Binomial marginal
likelihoods under a uniform prior, equal prior odds. A need with no evidence
contributes the same factor to both sides, so **the cell being asked about cannot
influence the judgement about whether meaning transfers to it**. And where the
posterior says it does not transfer, the organism **declines**: it says the need
word with no size word, which is what every organism before probe64 did.

## The headline

Both worlds, five seeds, forty lives each. `spoke` is how often the model was
willing to answer at all; `right` is how often it was correct when it did.

| world | model | spoke | right when spoken | **confidently wrong** |
|---|---|---:|---:|---:|
| factoring | `factored` | 1.0000 | 0.9900 +/- 0.0050 | 0.0100 |
| factoring | `tabular` | 1.0000 | 0.5000 +/- 0.0000 | 0.5000 |
| factoring | **`discovered`** | 1.0000 | **0.9925 +/- 0.0061** | **0.0075** |
| tangled | `factored` | 1.0000 | 0.5025 +/- 0.0899 | **0.4975** |
| tangled | `tabular` | 1.0000 | 0.5000 +/- 0.0000 | 0.5000 |
| tangled | **`discovered`** | **0.5350 +/- 0.0860** | 0.4954 +/- 0.1082 | **0.2650 +/- 0.0544** |

> **Where pooling is right, discovering it costs nothing** -- 0.9925 against
> 0.9900, inside one standard deviation. **Where pooling is wrong, discovering
> that halves the damage** -- confidently wrong falls from 0.4975 to 0.2650, a
> 47% reduction, per-seed deltas -0.175, -0.188, -0.337, -0.137, -0.325.

## The gate that makes it discovery

A learner that always answers "it factors" would score exactly `factored`'s
numbers and would be indistinguishable from one that found out. G4 is what
separates them: the organism's own posterior that the word transfers, measured in
both worlds.

| world | mean posterior that the word transfers |
|---|---:|
| factoring | **0.6856 +/- 0.0169** |
| tangled | **0.4499 +/- 0.0460** |

Per-seed gaps 0.232, 0.227, 0.292, 0.160, 0.268 -- above the locked 0.12 on every
seed. **The recovered structure tracks the true structure of the world**, which
is probe61's moved-ground-truth standard and the reason this is a discovery
rather than a lucky default.

The posterior sits near 0.69 rather than near 1.0 in the factoring world, and
that is correct rather than weak. When every observed cell is deterministic, the
per-cell hypothesis fits the data exactly as well as the pooled one; all that
favours pooling is the parameter it does not spend, and that Occam factor grows
only logarithmically in the evidence. A posterior of 0.69 is what the arithmetic
licenses, and it is enough to act on.

## The prediction that held

The preregistration recorded, in advance, that **accuracy-when-speaking in the
tangled world would stay at chance and must not be gated**: when the two observed
needs happen to agree, the third is genuinely a coin flip and the information is
not in the transcript. Measured: **0.4954 +/- 0.1082**. It is chance, and the
claim was never that the organism could be right there.

That is the honest ceiling of this mechanism and it is worth stating plainly.
With three needs, two observed and one held out, an organism that sees its two
agree cannot know whether the third does. What it *can* know is when its two
**disagree** -- and that is exactly when it stops guessing. Declining on about
half the tangled lives is not a partial success; it is the whole of what the
evidence supports.

## What this does not show

- **That the hypothesis space is discovered.** The organism chooses between two
  structures a designer wrote down -- pooled and per-cell. What it is not told is
  which of them this world implements. That is a strictly smaller claim than
  probe61 attempted for bodily structure, and it is smaller precisely because
  probe61 showed the larger one to be out of reach in this ecology.
- **That the vocabulary grew.** Three need words and two size words, unchanged.
  What changed is when the organism is willing to use them.
- **That declining is free.** It is not scored against survival here. Probe64's
  measurement that a sizeless request is survivable (`no_size` 0.630 against
  `always_large`'s 0.005) is what licenses declining as a move, and that
  measurement was made in probe64's ecology, not re-run in this one.
- **That anything grammatical happened.** The question is whether one lexical
  item's meaning is constant across the frames it appears in. That is a real
  question about a convention and this is its smallest honest version.

## Where property 4 now stands

`md/archive/DIRECTION_2026-07-26.md` property 4 -- *productive under novel demand:
it can say things it was never trained to say, because a listener needs them.*

Probe64 moved it from **no** to **partial**, with one reason recorded for why it
was not **yes**: the factorization was given. **That reason is now removed.** The
organism produces a request it has never made, correctly on first use, because it
has worked out from its own evidence that the word's meaning carries -- and in a
world where the meaning does not carry, it works that out too and stops.

What keeps the property short of an unqualified **yes** is no longer the
factorization but the hypothesis space: the organism selects between two
structures rather than inventing them. That is the boundary stated in the section
above, and it is the honest remaining gap.

## What this hands to the next rung

1. **Score declining against survival.** Probe64 measured that a sizeless request
   is survivable in its ecology; probe66 assumes it. A closed-loop run in the
   tangled world would show whether the organism that declines actually lives
   longer than the one that commits, which is the difference between a better
   belief and a better life.
2. **Widen the hypothesis space.** Two structures were written down. A learner
   that entertained partial pooling -- some needs sharing a convention and others
   not -- would be discovering more, and with three needs the design is already at
   the edge of what is identifiable, so this wants more needs before more model.
3. **The same question about the need words.** Nothing here tests whether the
   organism knows that *"hungry" means food whatever size it is asking for*. That
   is the transposed version of the same convention question and it is free to
   run.
