# Preregistration: probe66, discovering that a word's meaning transfers

Date: 2026-08-16
Status: PREREGISTRATION. Written before the treatment is run. Gates below are
locked and are not adjusted after seeing results. A failed gate closes its
mechanism.
Module: `src/homesocial/organism/discovered_convention.py`
Survey this rests on: `2026-08-16-discovered-factorization-ceiling-survey.md`

## 1. The claim under test

Probe64 moved property 4, *productive under novel demand*, from **no** to
**partial**, and named the one reason it is not **yes**:

> That the listener-model factorization is discovered. It is given.

The claim this puts up to be attacked:

> An organism can find out **whether a word means the same thing whatever it is
> asking for**, from the same evidence that teaches it what the word does; it can
> generalize to a combination it has never uttered where that generalization is
> licensed, and **decline** where it is not -- while a model handed the
> factorization generalizes in both worlds and is confidently wrong in one.

## 2. Why there is something to discover

The ceiling survey moved the ground truth with `tangled_size_words`, and measured
that a pooling model does **not** shrug when the world stops factoring. Probe64's
`factored` listener scores 0.988/1.000/1.000 where the convention factors and
**0.350/0.375/0.500** where it does not -- at or below the 0.500 a model with no
evidence scores. It commits, and commits wrongly. Being handed the factorization
is therefore a bet, and the world can settle it against you.

## 3. Mechanism

Nothing is trained and there are no free parameters. Before generalizing what a
word does to a need it has never used that word for, the organism asks whether
the needs it *has* used it for agree with each other:

```text
H_factors    every need shares one meaning for this word
H_tangled    every need has its own
```

Both are scored by their exact Beta-Binomial marginal likelihood under a uniform
prior with equal prior odds; the posterior is the organism's answer. A need with
no evidence contributes the same factor to both sides, so **the held-out cell
cannot influence the judgement about whether meaning transfers to it** -- which
is what makes this a judgement about transfer rather than about the cell.

The organism speaks when that posterior is at or above even odds, and otherwise
**declines**: says the need word with no size word, which is what every organism
before probe64 did and which probe64 measured as survivable (`no_size` survives
0.630 against `always_large`'s 0.005). Declining is a real move in this ecology,
not a scoring dodge.

Three listener models watch one identical transcript, driven by `factored`
exactly as in probe64, so the discovered model can never be starved or fed by a
policy difference of its own:

| model | what it does |
|---|---|
| `factored` | probe64's. Pools across needs, always. |
| `tabular` | probe64's. Never pools. |
| `discovered` | finds out whether pooling is licensed, and may decline. |

## 4. Operating point, fixed by the survey

- world `metabolic_spread` 0.60, `uptake_spread` 0.00,
  `interoception_probability` 0.03, `caregiver_store` 30.0, `portion_requests` on
- both `tangled_size_words` settings, held out need `energy`
- 5 seeds x 40 lives, seed band `1,060,000,000` with stride `2,000,000`,
  disjoint from the survey band `1,040,000,000`
- checkpoint `runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz`

## 5. Locked gates

Each is scored per seed and passes at **4 of 5**. Surveyed values in brackets.

**G1 -- it generalizes where generalization is licensed, and pays nothing for
the ability to decline.** Factoring world: `discovered` speaks `>= 0.95` **and**
is right when it speaks within 0.02 of `factored`. [1.000; +0.008]

**G2 -- it declines where generalization is not licensed.** Tangled world:
`discovered` speaks `<= 0.75`. [0.510]

**G3 -- the payoff, and the reason the first two are not enough on their own.**
Tangled world: `discovered` is confidently wrong -- speaks and is wrong -- at
least 0.10 less often than `factored`. [-0.187 to -0.338, mean -0.230]

**G4 -- the recovered structure tracks the world.** The mean posterior that the
word transfers is at least 0.12 higher in the factoring world than in the tangled
one. [0.228 to 0.289]. This is probe61's moved-ground-truth standard: a learner
that always answers "it factors" fails here whatever else it does.

**G5 -- the given factorization fails when the world stops factoring.**
Tangled world: `factored` first-use accuracy `<= 0.60`. [0.485]

**G6 -- the tabular control is exactly chance in both worlds.** [0.500, sd 0.000]

## 6. What is deliberately not gated, and the prediction recorded in advance

**Accuracy-when-speaking in the tangled world is not gated, and must not be.**
When the two observed needs happen to agree, the third is genuinely a coin flip
-- the information is not in the transcript, and no learner can have it. The
survey measures `discovered` at 0.441 there, which is chance, and the
preregistration predicts it will stay at chance. **The claim is about declining,
not about being right in a world where being right is impossible.** If the
treatment shows accuracy-when-speaking materially above 0.5 in the tangled world,
that is a surprise and is reported as one, because it would mean the mechanism is
reading something the design did not intend to leave available.

## 7. Failure conditions

- **G1 fails**: the discovery costs the generalization it was meant to protect,
  and the mechanism is worse than probe64's given factorization. Closed.
- **G2 or G3 fails**: it does not decline, or declining buys nothing. Closed.
- **G4 fails**: the recovered structure does not track the world, so whatever G1
  through G3 showed was not discovery. This is the decisive gate and its failure
  closes the mechanism regardless of the others.
- **G5 fails**: there was nothing to discover and the survey's measurement does
  not replicate.

Do not rescue any failure by moving the speak threshold off even odds, changing
the prior, weighting the hypotheses unequally, changing the held-out need, or
adjusting the store, the spread or the reading rate. Each is named here so that
none can be adjusted afterwards.

## 8. Claim boundaries, written before the result

Whatever happens, probe66 will **not** show:

- **that the hypothesis space is discovered.** The organism chooses between two
  structures a designer wrote down -- pooled and per-cell. What it is not told is
  which of them this world implements. That is a strictly smaller claim than
  probe61 attempted for bodily structure, and it is smaller precisely because
  probe61 showed the larger one to be out of reach here.
- **that the vocabulary grew.** It is still three need words and two size words.
  What changes is when the organism is willing to use them.
- **that declining is free.** It is not scored against survival here at all;
  probe64's measurement that `no_size` is survivable is what licenses declining
  as a move, and that measurement was made in probe64's ecology.
- **that anything grammatical happened.** The question is whether one lexical
  item's meaning is constant across the frames it appears in. That is a real
  question about a convention, and it is the smallest honest version of it.
