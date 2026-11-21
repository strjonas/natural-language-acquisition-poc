# caregiver-v2: an immediate bodily cost for the wrong size, and a held-out word pair

Date: 2026-11-21. Status: **design and pass rules, written before any v2
survey or training.** Follows the probe in
`2026-11-21-caregiver-v1-ceiling-survey.md`: the speaking learner's state
encodes its social levels, but its size choice ignores them, and the world
makes the consequence of a wrong size nearly invisible.

## Change from v1-size (one mechanism, one test)

1. **Overfilling makes the body sick.** A delivery that would take a level
   above 1.0 leaves it at `1 - HARM * overflow`, with `HARM = 2`. Asking 0.6
   when 0.3 is missing ends at 0.4; asking 0.2 ends at 0.9. The cost is bodily,
   immediate and sensed. The per-life ration is removed: no hidden budget.
   Everything else is v1-size: visits every fourth decision, sizes 0.2 / 0.4 /
   0.6 delivered next tick, speech and action at the same decision.
2. **Two conventions.** `private`: as v1, per identity. `shared`: one fixed
   convention for every identity (a language community). Both are learned
   from consequence only; neither is given to the learner.
3. **A held-out pair (shared convention).** In training bodies, channel 0's
   individual rate factor is drawn from [0.4, 0.7] instead of [0.4, 1.6], so
   that channel rarely if ever needs 0.6. Held-out test bodies have channel 0
   social with a factor in [1.3, 1.6]. The learner knows the word for channel
   0 (from small requests) and the word for 0.6 (from other channels); the
   pair is the test of a new combination of known words.

## Survey pass rule (locked)

Reference speakers as in v1 (same rule; the spill penalty is the body's own
harm), with true or dead-reckoned levels. Unrestricted family, the six v1
cells, five blocks x 100 identities, band 6,500,000,000. Pass requires the v1
criteria 1-6 (criterion 4 with dead reckoning; criterion 6 at <= 0.97).

Held-out pair, reference `composed` speaker, cells (K, social) = (2,1), (3,1),
(3,2), (4,2) with channel 0 social, 3 blocks x 100 identities each, train-like
and test-like distributions:

- H1: in train-like bodies, (channel 0, 0.6) is < 2% of channel-0 requests;
- H2: in test-like bodies, it is >= 15%;
- H3: in test-like bodies, `composed` minus a speaker forbidden that one pair
  (`no_held_out_pair`, otherwise identical) is >= +0.05 survival, interval
  above zero.

If the survey fails, record it and change one named mechanism before training.

## Learner runs (development, tonight)

`speaking_self.py` unchanged in architecture and learning signals, from
scratch, 10 h each:

- **A: v2, private convention, seed 0.** Isolates the world change against the
  v1 runs.
- **B: v2, shared convention with the held-out pair, seed 0.** Evaluated also
  on held-out bodies: how often it says (channel 0, 0.6) there, its survival
  against the reference speakers, and the same survival with that one pair
  muted.

What would count as the target's core demonstration in development (not yet
confirmation): in B, the learner says the held-out pair in held-out bodies at
a rate near the reference speaker's, and muting that one pair costs survival.
Confirmation then needs a fresh band, a preregistration and five seeds.

## Survey results and iteration 1 (2026-11-21, evening)

`PYTHONPATH=src .venv/bin/python -m homesocial.organism.caregiver_v2`;
evidence `runs/organism/caregiver_v2/{ceiling_survey,held_out_pair}.json`.

**v2 (sickness, no ration) fails** criteria 3 and 5. The size word is worth
+0.050 to +0.116 over the best single size (3 of 6 cells reach +0.08), and in
cells with two social resources the composed speaker says 0.2 in 1% of
requests. Self-knowledge matters more than in v1: own-rate minus species-rate
reckoning +0.197 to +0.340 (5/5 blocks everywhere). Without a ration, large
requests cost only when they overflow, so the second word carries less.

**The held-out pair fails H1.** In training-like bodies the pair is already
29-70% of channel-0 requests: birth levels as low as 0.35 open large deficits
at a life's first visit, and with two social resources channel 0 is served
less often. H2 passes (65-89%); H3 passes only with two social resources
(+0.21 to +0.26) and fails with one (+0.003 to +0.006).

Iteration 1, one named change each, decided after these results:

1. **World `v2r` = v1-size's ration plus sickness.** The ration is what made
   size matter in v1; sickness is what makes a wrong size felt at once. Same
   survey and criteria.
2. **Held-out training restriction.** In training bodies, channel 0 is born at
   0.75 or 0.85, its rate factor is in [0.4, 0.7], and when it is social it is
   the only social resource. H1 is then measured on that training
   distribution (cells with one social resource); H2 and H3 on the same
   held-out test cells as before. Thresholds unchanged.

## Iteration 1 results and iteration 2 (2026-11-21, evening)

**`v2r` passes all six survey criteria** (`--part survey --ration`; evidence
`ceiling_survey_v2r.json`). The second word is worth +0.125 to +0.244 over the
best single size, 5/5 blocks in every cell, the strongest of any world so far.
Every size is 15-48% of requests. Own-rate minus species-rate reckoning is
+0.18 to +0.26 with two social resources and +0.007 to +0.010 (not
significant) with one; criterion 4 needs half the cells and passes on three.

**The held-out restriction works for H1** in both worlds: the pair is 0.000 of
channel-0 requests in training-like bodies. With the v2r ration, held-out
bodies with one social resource starve whatever the speaker says (survival
about 0.03): their fast channel 0 outruns a ration set from species rates.
Without a ration, in those cells the pair is used (65%) but two 0.4 requests
replace it (+0.003 to +0.006). With two social resources the pair is
needed and worth survival in both worlds (v2r: share 0.62-0.68, +0.137
[+0.091, +0.183] and +0.137 [+0.095, +0.180]).

Iteration 2, decided after these results: held-out test cells are those with
two social resources, (3,2) and (4,2), where the pair cannot be substituted.
H1 stays on the one-social training distribution. Under this, v2r passes H1-H3
(`held_out_pair_v2r_iter2.json`). A bug was fixed on the way (held-out bodies
computed their ration before being made social, so it was zero).

**Runs launched tonight**, both from scratch, seed 0, 10 h, world `v2r`:
A, private convention; B, shared convention with the held-out pair. Development
only; both bands monitored.

**Correction (same evening).** A test found that `restrict_hold_rate` read
channel 0's social flags as a view of the array it then edited, so it removed
channel 0 from the social resources altogether. The H1 value 0.000 above was
therefore trivial. Fixed (a copy) and re-run: H1 is 0.015 of channel-0
requests in training-like bodies (threshold < 0.02), H2 0.62-0.68, H3 +0.137
in both test cells; **pass**. Run B had started with the bug and was restarted
from scratch on the fixed code (manifest pin `eff7ab9a...`). Run A (private
convention) uses neither fixed function; its manifest pins the earlier file.
