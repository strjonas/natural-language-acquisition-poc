# A learned self on Probe80's benchmark, and a ceiling check for words

Date: 2026-11-20. Status: **development build and budget experiment. No
confirmatory result, no gate, no capability claim.** Nothing here revises
Probe80 or any closed line.

## Blocker

The target in STATE is one learner that regulates its body and acquires
productive words about it. The repository has two halves that cannot be joined
by extension: the island organism speaks through a parameterless six-message
rule, and Probe80's self-model is a filter hand-derived for physics its designer
wrote, so it cannot take in words. Whether a *learned* sequence model can do
Probe80's job was never tested. Probes 48-56 closed the neural route at 80,000
ticks; Probe69 later showed that budget was the one most favourable to
structure. The 2026-10-21 review also noted Probe80 has no trained baseline
with history-dependent decisions.

## What was built

1. `src/homesocial/organism/vec_body.py` -- Probe80's body family over NumPy
   arrays. It adds no physics. For any identity seed it reproduces
   `active_self.LivedBody` exactly (map, rates, births, shocks), checked tick
   by tick across deaths and rebirths in `tests/test_vec_body.py`. Measured
   on a shared machine: about 640,000 decisions (3.9 million ticks) per second
   for 1,024 bodies, against about 80,000 ticks per second for the reference
   object. Training speed is set by the network, not the body.
2. `src/homesocial/organism/learned_self.py` -- one GRU (256 units) trained
   across bodies with recurrent PPO. It receives what `SelfPosterior` receives:
   two scalars per tick, its own actuator choice, birth events. One recurrent
   state feeds the actuator policy, a value estimate, and a predictor of its
   own next six ticks of sensation. Memory persists across the six lives of an
   identity and is emptied between identities.
3. `src/homesocial/organism/caregiver_world.py` -- scenario `caregiver-v0` and
   its oracle ceiling survey (below).

Learning sources: bodily reward only (0.5/66 per decision survived, 0.5 for
completing a life); next-sensation prediction as an auxiliary objective. No
linguistic signal exists in this experiment.

Privileged information: a **structure probe** is trained on the true actuator
partition. It sits behind a stop-gradient (tested: its loss sends zero gradient
into the recurrent state) and is a measuring instrument, not the organism's
report. "All 55 probe facts right" equals Probe80's "exact partition" (tested).
It says whether structure is decodable from the state, not that the organism
reports it or is calibrated.

Seeds: training bodies come from a NumPy stream; evaluation uses identity band
6,100,000,000, built by the exact `LivedBody` generator, guarded against every
Probe80 band. Probe80's survey band stays untouched. Comparison with Probe80's
posterior is against its published table (other bodies, same family), which is
adequate for development and not a paired control.

## Sanity check (not evidence about the outcome)

Six minutes, 15.3 million decisions, 60 development identities per K, machine
shared with other jobs (about 42,000 decisions per second end to end):

| K | random | learned | learned, memory emptied each life | Probe80 bandit | Probe80 posterior |
|---|---:|---:|---:|---:|---:|
| 1 | 0.453 | 0.894 | 0.772 | 0.969 | 0.978 |
| 2 | 0.136 | 0.483 | 0.406 | 0.417 | 0.864 |
| 3 | 0.019 | 0.203 | 0.153 | 0.039 | 0.675 |
| 4 | 0.006 | 0.017 | 0.031 | 0.006 | 0.472 |
| 5 | 0.008 | 0.006 | 0.006 | 0.000 | 0.272 |

The learning signal is real and still rising steeply; retained memory already
helps at K1-K3. The probe decoded no exact partition yet. This run establishes
that the implementation learns, nothing about where it ends.

## The budget experiment

One sustained run, default configuration, about nine hours (roughly 1.3 billion
decisions at the measured rate; unverified under different machine load).
Evaluation every 20 million decisions on 200 development identities per K, with
the memory-reset lesion; curves in `curve.jsonl` and `train.jsonl`; resumable
checkpoint.

```bash
PYTHONPATH=src caffeinate -i .venv/bin/python -m homesocial.organism.learned_self train --out-dir runs/organism/learned_self_dev --max-hours 9
```

`--resume` continues it. Weights, optimizer, counters and the training stream
are restored; the bodies alive at the save are redrawn.

Next decision, read from the curve and not from one endpoint:

- Survival near the posterior's at K1-K3 and still rising: the learned route
  can carry bodily knowledge. Design the confirmatory comparison (own band,
  preregistration, paired posterior arm) and give the same learner words.
- Flat for the last third of the run and clearly below the posterior: test one
  named limitation at a time (capacity, input encoding, exploration), or keep
  the posterior as the belief and learn only the speech on top.

Not built: the yoked-body control, the uncertainty lesion, any report by the
organism itself, bodies outside the family, change, space.

## Ceiling check for words: `caregiver-v0`

Scenario: some resources are social (no own actuator restores them). The
organism may say a resource word and a size word instead of acting. The
convention is private to each identity. A small portion arrives next tick; a
large one arrives 12 ticks later and the caregiver hears nothing meanwhile.

Oracle speakers on identical identities and shocks, 5 blocks x 100 identities
x 6 lives per cell, band 6,200,000,000:

| cell | silent | small only | large only | composed | size from species rates | wrong resource word |
|---|---:|---:|---:|---:|---:|---:|
| K2, 1 social | 0.016 | 0.843 | 0.890 | 0.920 | 0.917 | 0.016 |
| K3, 1 social | 0.019 | 0.737 | 0.783 | 0.807 | 0.806 | 0.019 |
| K3, 2 social | 0.000 | 0.561 | 0.551 | 0.623 | 0.609 | 0.000 |
| K4, 1 social | 0.016 | 0.579 | 0.650 | 0.668 | 0.671 | 0.016 |
| K4, 2 social | 0.000 | 0.374 | 0.463 | 0.510 | 0.508 | 0.000 |
| K5, 2 social | 0.000 | 0.216 | 0.376 | 0.422 | 0.422 | 0.000 |

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.caregiver_world
```

Reading:

- **The resource word is load-bearing**: +0.42 to +0.90 over silence, and the
  wrong word is exactly as bad as silence.
- **The size word is nearly idle**: choosing it buys +0.02 to +0.07 over always
  saying "large" (5/5 blocks, small). The composed speaker says "small" in 2-8%
  of requests.
- **Self-knowledge does not reach the size word**: judging size from species
  rates instead of this body's costs at most +0.014.

So `caregiver-v0` would teach a one-slot vocabulary. It is not yet a world in
which composition pays, and no learner should be trained for productivity in
it. This is the result Probe64 and Probe70 found in other forms.

Longer delays (24, 36, 60 ticks) do not rescue it: the hand-written composed
rule then loses to always-small by up to 0.48, so that rule is not an upper
bound there and the large option is simply bad. A reference speaker is not a
ceiling proof; a second slot needs a different consequence, not a retuned
delay. Candidates for `v1`, each to be put through this same survey first: a
basket divided among several named resources per visit; a "when" slot under
delayed answers (Probe65's horizon effect); a costly "check me" request.

## Result of the sustained run (added 2026-11-21)

Development result: one training seed, `git` source `754493a`, default
configuration. The run stopped at the default `--max-decisions` cap
(2.0 billion decisions, 7.25 h of training, about 75,000 decisions per second
under overnight load), before the nine-hour cap. Evaluation: 200 development
identities per K, six lives each, final checkpoint (no checkpoint selection).
Evidence: `runs/organism/learned_self_dev/evidence.json` (curve, final
evaluation, checkpoint hashes); the checkpoint itself stays local.

| K | random | myopic oracle | Probe80 posterior* | learned | learned, memory reset each life | probe exact, life 6 | Probe80 exact, life 6* |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.394 | 1.000 | 0.978 | 0.962 | 0.873 | 1.000 | 1.00 |
| 2 | 0.129 | 0.910 | 0.864 | 0.938 | 0.754 | 0.995 | 1.00 |
| 3 | 0.039 | 0.782 | 0.675 | 0.887 | 0.598 | 0.590 | 0.93 |
| 4 | 0.010 | 0.675 | 0.472 | 0.796 | 0.443 | 0.080 | 0.67 |
| 5 | 0.004 | 0.552 | 0.272 | 0.747 | 0.315 | 0.000 | 0.25 |

\* Probe80's published survey: other bodies of the same family, not paired.

Survival by life at K5: 0.33, 0.62, 0.78, 0.89, 0.94, 0.94 (memory reset:
flat, 0.29-0.34). K4: 0.34 to 0.96. Life-one survival is no better than the
reset arm, so the gain is built from this body's own earlier lives, not from
leaked identity information.

Curve: most of the gain came in the first 200 million decisions (K5 0.66).
Over the last third, the slope is +0.000 (K1), +0.009, +0.013, +0.018, +0.026
(K5) survival per billion decisions, against an evaluation noise of about
0.01. K1-K2 have plateaued; K4-K5 are still rising slowly.

**The "oracle" is not a ceiling.** It always grants the large portion to the
unblocked resource that would run out first, from true levels and rates. The
learner, which sees only the two scalars, beats it at K2-K5. At K3-K5 in life
six only 31% of its choices equal the oracle's; 50% target another resource,
7% use the other portion, and 13% use an actuator that restores nothing
(`scripts/learned_self_actions.py`). Why that control is better is untested.
One hypothesis is that "do nothing" and timing avoid overshoot; this has not
been checked. Probe80's oracle numbers should be read as a reference policy
only, as the method guidance already warns.

**What this establishes (development):** one learned recurrent state, given
only what `SelfPosterior` gets, finds out its body across lives and keeps it
alive at least as well as the hand-derived posterior at K2-K5 on this family.
At K1 it is 0.02 below. It survives better while decoding the exact partition
less often at K3-K5 (per-fact accuracy at life six: 0.93-1.00). Good control
does not need an exactly decodable map here.

**What it does not establish:** a paired comparison with the posterior (other
bodies, one seed); replication across training seeds; the yoked-body and
uncertainty controls; generalization outside the family; anything about
words. The development identities were monitored throughout, so they cannot
serve for confirmation.

**Decision:** the learned route passes its development branch condition ("near
the posterior and still rising"). Keep this learner as the body-knowledge
carrier for the language stage. A confirmatory comparison on a fresh band
(paired posterior arm, five training seeds, yoked and reset controls) is now
worth preregistering, at about 7 h per seed at this rate. The language stage is
still blocked by `caregiver-v0`, which supports only a one-slot vocabulary.
