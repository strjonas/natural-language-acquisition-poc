# caregiver-v1: a world in which a second word pays? Ceiling survey

Date: 2026-11-21. Status: **development instrument; oracle speakers, no
learner, no capability claim.** Follows
`2026-11-20-learned-self-budget-experiment.md`, whose `caregiver-v0` survey
showed a one-slot world: the resource word was worth +0.42 to +0.90 survival,
the size word +0.02 to +0.07, and the size choice did not depend on the body's
own rates.

## Why v0 failed, mechanically

Levels cap at 1.0 and nothing penalizes asking for too much, and the caregiver
could be asked at any decision. So a speaker can always wait until a large
portion fits and ask for it: the second word carries nothing a timing rule does
not. A second slot needs (a) a reason the right amount varies from request to
request and (b) a cost for the wrong amount.

## Design shared by all candidates (written before any v1 survey ran)

Probe80's body family through `vec_body` (no new body physics). As in v0, some
active resources are social: no own actuator restores them.

- **Visits.** The caregiver is present at every fourth decision (every 24
  ticks), starting with the first decision of a life. Speech at other decisions
  is not heard. The organism can see whether the caregiver is present.
- **Speech and action together.** At each decision the organism uses an
  actuator (or none) *and* may say one two-word utterance. This removes v0's
  opportunity cost between acting and speaking, which confounded word value.
- **Private convention.** Per identity, six resource words name the five
  channels in random order (one names nothing), and the second slot's words
  are randomly permuted. Drawn from the identity's own seed streams.
- **Delivery** lands on a later tick and is capped at 1.0; anything above the
  cap is lost.

Three candidates differ only in what the second word does:

| candidate | first word | second word (3 or 6 words) | cost of a wrong second word |
|---|---|---|---|
| `v1-size` | resource | portion 0.2 / 0.4 / 0.6, delivered next tick | per-life ration: every unit asked is charged, overflow wasted |
| `v1-when` | resource | 0.6 delivered after 1 / 13 / 25 ticks | same per-life ration |
| `v1-basket` | resource | a second resource word: 0.4 first + 0.2 second; same or empty word: 0.6 to the first | none charged; the visit's 0.6 basket is the limit |

Ration (size, when): per life, 1.2 x the species' expected need of the social
resources (species rate plus expected shock loss, over 400 ticks). A body with
fast individual rates is short; a slow one has slack.

## Reference speakers

All know true structure, levels, rates and convention. Own resources are
handled by Probe80's oracle rule (large actuator of the own resource that runs
out first), identical in every arm. At a visit, a speaker projects every
candidate utterance (and silence) to the next visit's delivery plus one window,
with expected shock loss, and picks the cheapest utterance whose projected
lowest social level stays at or above 0.1; if none does, the utterance with
the highest projected lowest level. Basket utterances cost nothing.

Arms, on identical identities, births and shocks:

- `silent`; `composed` (full vocabulary);
- `single_<m>`: the second word fixed to one meaning *m* (each meaning is one
  arm; the best of them is "best single");
- `species`: `composed`, but projected with species rather than individual
  rates (the self-knowledge contrast);
- `scrambled`: `composed` naming the next channel's resource word(s).

## Pass rule (locked before results)

Cells (K, social): (2,1), (3,1), (3,2), (4,1), (4,2), (5,2); `v1-basket` only
in cells with two social resources, where a split is possible. Five blocks x
100 identities x six lives, band 6,300,000,000. Paired over blocks, two-sided
95% t-intervals.

A candidate passes if, in its cells:

1. speech matters: `composed` - `silent` >= +0.20 in every cell;
2. the first word matters: `composed` - `scrambled` >= +0.20 in every cell;
3. **the second word matters**: `composed` - best single >= +0.08 with the
   interval above zero in at least two thirds of cells, and >= +0.03 in all
   (v0's best was +0.07);
4. **self-knowledge reaches speech**: `composed` - `species` >= +0.04 with the
   interval above zero in at least half the cells;
5. combinations are needed: every second-word meaning is >= 10% of
   `composed` requests in at least two thirds of cells;
6. headroom: `composed` <= 0.97 in every cell.

Among passing candidates, prefer the largest worst-cell margin on criterion 3.
If none passes: record it, and change one named mechanism with its reason
before any rerun. Parameter changes after seeing results are development
iterations and are counted below; a chosen world gets a fresh band before any
learner is evaluated on it.

## Development iterations before the full survey (disclosed)

Both were made on 2-block x 30-identity smoke runs of band 6.3B, which overlap
the full survey's first identities. Neither changed a world parameter or a
threshold.

1. **Reference speaker.** The rule above ("cheapest utterance whose projection
   stays above 0.1") is short-sighted. With two social resources and one
   request per visit it starved one of them and lost to always-0.6 by 0.26
   (K4, two social). Replaced, for every arm alike, by: keep every social
   resource above 0.1 until the next visit's delivery first; then absorb the
   most; a spilled unit of ration counts 1.5 against (0 for the basket); ties
   go to the resource that runs out first; silent when a request would add
   less than 0.1.
2. **Self-knowledge contrast.** A speaker that reads true levels needs its
   rates only for a 31-tick projection, so `species` matched `composed` by
   construction. The contrast now compares two speakers that *track* their
   social levels by dead reckoning, one with the individual's rates
   (`reckoned`) and one with the species' (`species`). Both restart from the
   true levels at each life's first decision and see which channel each shock
   hit and which deliveries landed, which is more than the organism senses.
   Criterion 4 is `reckoned` - `species`. In this deterministic body, own-rate
   reckoning is exact (tested), so `reckoned` equals `composed`.

## Result (full survey, 2026-11-21)

`PYTHONPATH=src .venv/bin/python -m homesocial.organism.caregiver_v1`;
evidence `runs/organism/caregiver_v1/ceiling_survey_{size,when,basket}.json`.

**`v1-size` passes all six criteria**, 5/5 blocks in every contrast:

| cell | silent | composed | best single | composed - best single | reckoned - species | shares 0.2 / 0.4 / 0.6 |
|---|---:|---:|---:|---:|---:|---|
| K2, 1 social | 0.015 | 0.773 | 0.552 (0.4) | +0.221 [+0.162, +0.279] | +0.054 [+0.022, +0.086] | 0.53 / 0.33 / 0.14 |
| K3, 1 social | 0.013 | 0.694 | 0.494 (0.4) | +0.200 [+0.174, +0.226] | +0.048 [+0.031, +0.066] | 0.50 / 0.35 / 0.15 |
| K3, 2 social | 0.000 | 0.577 | 0.398 (0.6) | +0.179 [+0.134, +0.224] | +0.211 [+0.187, +0.235] | 0.20 / 0.35 / 0.45 |
| K4, 1 social | 0.015 | 0.602 | 0.429 (0.4) | +0.173 [+0.149, +0.197] | +0.048 [+0.030, +0.066] | 0.51 / 0.34 / 0.14 |
| K4, 2 social | 0.000 | 0.496 | 0.322 (0.6) | +0.174 [+0.130, +0.218] | +0.176 [+0.149, +0.203] | 0.20 / 0.35 / 0.45 |
| K5, 2 social | 0.000 | 0.480 | 0.321 (0.4) | +0.158 [+0.133, +0.183] | +0.157 [+0.134, +0.180] | 0.21 / 0.37 / 0.42 |

A wrong resource word is as bad as silence (`scrambled` within 0.004 of
`silent`). Which single size is best changes with the cell, and no single size
comes within 0.15 of composed requests: the second word carries a choice that
neither the cell nor a fixed habit settles.

**`v1-when` fails** criterion 3 (second word +0.015 to +0.070) and criterion 5
(the 25-tick delay is 62-90% of requests). **`v1-basket` fails** criterion 3
(+0.054 to +0.056). Both pass the self-knowledge contrast.

Reading: v0's failure was the world, not the idea. A caregiver who visits at
intervals and rations what it gives makes the amount a real second slot, and
the right amount depends on tracking this body's own rates. A reference
speaker is still not a ceiling: these margins are lower bounds on what the
second word can be worth.

**Decision:** `caregiver-v1-size` is the world for the speaking learner
(`speaking_self.py`). Its development evaluation uses band 6,400,000,000; any
confirmatory test needs a further fresh band and a preregistration.

## First speaking-learner runs (added 2026-11-21, evening)

Development runs of `speaking_self.py` in `caregiver-v1-size`, three arms run
side by side for 10 h each (about 1.02 billion decisions, 29,000 decisions per
second per arm): warm start from the learned-self checkpoint with seeds 0 and
1, and from scratch with seed 0. Final checkpoints, 100 development identities
per cell (band 6.4B, monitored during training). Evidence:
`runs/organism/speaking_self_dev/evidence.json`.

| cell | silent* | best single size* | composed* | learned (warm s0 / s1 / scratch) | muted | memory reset | second word scrambled |
|---|---:|---:|---:|---|---:|---:|---:|
| K1, 0 social | 1.000 | 1.000 | 1.000 | 0.983 / 0.988 / 0.965 | 0.975 | 0.868 | 0.983 |
| K2, 1 social | 0.022 | 0.637 | 0.865 | 0.623 / 0.617 / 0.633 | 0.018 | 0.228 | 0.577 |
| K3, 1 social | 0.007 | 0.470 | 0.650 | 0.473 / 0.473 / 0.488 | 0.007 | 0.140 | 0.417 |
| K3, 2 social | 0.000 | 0.398 | 0.643 | 0.148 / 0.142 / 0.225 | 0.000 | 0.020 | 0.115 |
| K4, 1 social | 0.007 | 0.428 | 0.597 | 0.412 / 0.358 / 0.398 | 0.005 | 0.085 | 0.387 |
| K4, 2 social | 0.000 | 0.368 | 0.548 | 0.167 / 0.123 / 0.202 | 0.000 | 0.027 | 0.122 |
| K5, 2 social | 0.000 | 0.272 | 0.423 | 0.067 / 0.072 / 0.095 | 0.000 | 0.013 | 0.048 |

\* Reference speakers reading true levels, same identities. Lesion columns
are warm s0. "Best single size" here is the 0.4 arm.

What the learner acquired, from bodily consequence alone and with meanings
private to each identity:

- **Resource words.** Heard requests name a social resource 65-90% of the time
  (chance about 17-33%). Muting the learner removes all of its advantage over
  silence: speech is load-bearing.
- **When to speak.** It speaks at 52-92% of visits when it has a social
  resource. With none it still speaks about half the time, which costs nothing
  here.
- **Memory.** Emptying the state at each birth drops survival by 0.10-0.40,
  so word meanings learned in earlier lives are used in later ones.

What it did not acquire:

- **The size word is mostly idle.** Scrambling it costs 0.00-0.08; it says 0.2
  in only 5-15% of requests (the reference speaker: about half, with one social
  resource). With one social resource it matches the best fixed-size speaker,
  not the composed one; with two it is far below both.
- **No evidence of productive composition.** New pairs of known words (1-2 per
  identity) name a social resource about as often as its other requests
  (0.46-0.66).
- **The warm start bought nothing** by 1 billion decisions: scratch is equal or
  slightly better and relearns the body.

The two warm seeds agree closely. All arms were still rising (for example
K3/2 social 0.03 -> 0.15 -> 0.23 over the scratch run).

Reading: the one-slot result of Probe64/70 and v0 reappears, now in a world
whose second slot is demonstrably worth +0.16 or more. The blocker has moved
from the world to the learner. Two causes are open: the state may not encode
how full each social resource is, or it does but the size choice is not
reached by the delayed, noisy reward (a policy-gradient credit problem).
