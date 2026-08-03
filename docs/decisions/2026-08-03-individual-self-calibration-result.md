# Probe63: an organism that finds out what its own body is

Date: 2026-08-03. Status: **treatment result.** Preregistration:
`docs/decisions/2026-08-03-individual-self-calibration-preregistration.md`, gates
locked there before implementation and not adjusted. Feasibility:
`docs/decisions/2026-08-03-individual-self-ceiling-survey.md`.

Artifacts: `runs/organism/probe63_individual_self/`. Module:
`src/homesocial/organism/self_calibration.py`. Instruments:
`src/homesocial/organism/individual_self.py`. Guards:
`tests/test_individual_self.py` (10), `tests/test_self_calibration.py` (6).

## Verdict

Two learning rules faced one set of locked gates.

| gate | NLMS (`learned`) | RLS (`recursive`) |
|---|---|---|
| G1 corrigibility, error <= 0.75x `snap` | **pass** 5/5 | **pass** 5/5 |
| G2 report, accuracy >= `snap` + 4.0 | **pass** 5/5 | **pass** 5/5 |
| G3 localization, metabolism world | **pass** 5/5 | **pass** 4/5 |
| G3 localization, absorption world | **fail 0/5** | **pass** 5/5 |
| G4 no false discovery | **pass** 5/5 | **pass** 5/5 |
| G5 shuffled readings | **pass** 4/5 | **pass** 5/5 |
| G6 rate recovery | **pass** 4/5 | **pass** 5/5 |

**The recursive arm passes every locked gate. The greedy arm is corrigible but
not discriminating**, and its G3 failure is closed by the preregistered rule: it
learns *that* it is atypical and is confidently wrong about *how*.

## What was claimed and what it rests on

> An organism born not knowing its own body, given intermittent readings of
> itself, learns online which of its own causal constants differ from its
> species -- and thereby knows where its body is and names its true lowest need
> better than the same evidence without a self-model, better than a
> species-accurate hand-written filter, and better than being born knowing its
> own rates but blind.

Treatment world: `metabolic_spread` 0.60, reading rate 0.03, 5 seeds x 40 lives,
all tiers scored on **one shared history** so no contrast is a policy difference.

| tier | body error | named-need accuracy |
|---|---:|---:|
| `population` -- species filter, *today's repository* | 0.0785 +/- 0.0093 | 0.6649 +/- 0.0172 |
| `snap` -- corrected to truth at every reading | 0.0454 +/- 0.0045 | 0.8148 +/- 0.0127 |
| `learned` -- NLMS self-calibration | 0.0191 +/- 0.0040 | 0.9353 +/- 0.0080 |
| `recursive` -- RLS self-calibration | **0.0133 +/- 0.0023** | **0.9464 +/- 0.0059** |
| `individual` -- born knowing its rates, no readings | 0.0171 +/- 0.0031 | 0.9327 +/- 0.0111 |

Four things in that table matter.

1. **Against the hand-written filter**: error falls 0.0785 -> 0.0133, an 83%
   reduction, and the report gains **28.2 points**.
2. **Against evidence without a self-model**: `snap` receives the identical
   readings and corrects its belief to each one. Error falls a further **70.7%**
   and the report gains **13.2 points**. Phase B1 warned its first gate is
   "satisfiable by clipping"; this is the margin that survives the clipping
   baseline.
3. **`recursive` beats `individual`.** An organism that has to find out what its
   body is, but can see itself occasionally, ends up *more accurate* than one
   born knowing its own constants exactly and then blinded. Self-knowledge and
   evidence are not substitutes; the model is what carries the evidence forward
   between readings.
4. Every seed separation is far larger than its spread. The per-seed error gap
   between `recursive` and `snap` is 0.0324 / 0.0342 / 0.0343 / 0.0325 / 0.0274,
   against standard deviations of 0.0023 and 0.0045.

## Why this is not the old result again

`docs/STATE.md` has carried one obstacle since v1:

> The hand-coded probe53 filter is still better (99.96%, zero error). ... Nothing
> yet shows the learned model doing what the analytic filter cannot.

It is no longer true, and not by a margin. The reason is structural rather than
numerical: **the filter's constants are no longer knowable at design time.** They
are drawn when the organism is born. A designer can still supply the *form* of an
estimator -- and the ceiling survey's analytic instrument is exactly that -- but
its *content*, this body's rates, exists only in this organism's own history.

That is the first content in this repository's self-model that is about a self
rather than about a species. Every previous "self-model" here was a physics with
the same constants for every organism that ever lived, which is precisely why
nothing learned could beat a hand-written one.

## "You changed the ecology so your mechanism would win"

That is the first objection and it deserves a direct answer rather than a
footnote, because changing the world until something passes is exactly the sin
this repository's method exists to prevent.

1. **The change was diagnosed, not chosen.** Probe62 measured why probes 59--62
   all went flat: the body is bounded in [0,1], the filter saturates on ~9.7% of
   ticks, and every saturation erases the accumulated error, capping self-model
   error at ~0.05. That is `CLAUDE.md`'s "environment doing the model's job"
   trap with a number attached. The diagnosis came before this ecology existed.
2. **It is the repository's own next phase, already written down.**
   `DIRECTION_2026-07-26.md` phase B1 says: *"Return interoception intermittently
   and unpredictably ... Introduce a systematically biased body the current model
   cannot represent."* That is precisely what the three levers do. Section 4 of
   the same document makes it binding that complexity is *pulled by the task*,
   and lists making the world require distinctions the current protocol cannot
   express as the sanctioned way to do it.
3. **Nothing previously established was disturbed.** All three levers are
   default-inert and guarded over full lives; the whole suite of **377 tests**
   passes, and the null world reproduces `population`, `snap`, `learned`,
   `recursive` and `individual` at identical body error 0.0000 with total learned
   correction exactly 0.000.
4. **The gates were locked before the mechanism was implemented, and one arm
   failed one of them.** An ecology tuned to produce a pass would not have
   produced a 0/5.
5. **The change makes the problem harder, not easier.** The oracle survives ~0.70
   here against 0.92 in v1, because individuality makes some bodies
   unsurvivable, and `population` -- the system this repository shipped as v1 --
   loses 33 points of report accuracy the moment bodies stop being identical.

What the change does *not* get to claim is comparability. No number here is
comparable to probe57's, and this document does not compare them.

## Does it find out *what* about itself is different?

This is the discovery claim, and it is where the two arms come apart.

The organism keeps one copy of its body filter per constant it could be wrong
about, each with exactly that constant perturbed, and reads off
`J[need, parameter] = d(predicted body)/d(log parameter)`. That is not a fact
about where its body is; it is a fact about how its own *model* would respond if
a particular belief about itself were wrong. Each residual is then attributed
across its own parameters. The guard
`test_the_self_jacobian_is_the_real_derivative` checks that quantity against a
filter genuinely rebuilt with the scaled constant, to **1e-12**, for all seven
parameters over full lives.

Share of the learned correction's mass, by world:

| world | rate | NLMS metabolic / uptake | RLS metabolic / uptake |
|---|---:|---|---|
| metabolism only | 0.03 | **0.877** / 0.099 | **0.733** / 0.206 |
| absorption only | 0.03 | 0.765 / 0.201 | 0.286 / **0.531** |
| metabolism only | 0.10 | **0.927** / 0.056 | **0.890** / 0.086 |
| absorption only | 0.10 | 0.548 / 0.413 | 0.095 / **0.845** |
| both | 0.03 | 0.818 / 0.153 | 0.484 / 0.431 |
| neither | 0.03 | total correction **0.000** | total correction **0.000** |

Per seed at rate 0.03, which is where the gates were scored. Recursive metabolic
share in the metabolism world: 0.755 / 0.770 / 0.742 / **0.649** / 0.748 -- the
fourth seed is the one that misses the locked 0.70, giving 4/5. Recursive uptake
share in the absorption world: 0.503 / 0.611 / 0.478 / 0.583 / 0.481, all clear,
5/5. Greedy uptake share in the same world: 0.160 / 0.311 / 0.204 / 0.238 /
0.092, none clear, 0/5.

At reading rate 0.10 the recursive arm's attributions are nearly disjoint: 0.890
metabolic mass when the truth is metabolic, 0.845 uptake mass when the truth is
absorptive, and in the world where both differ it lights both (0.484 / 0.431).
In a world where nothing is individual it recovers **exactly zero** -- no false
discovery, and the shares are reported as undefined rather than as a smear.

The greedy arm does not discriminate. It reports "metabolism" at 0.877 when that
is true and at 0.765 when it is false. Note that its metabolic share in the
metabolism world is *higher* than the recursive arm's (0.877 against 0.733), and
that this is not evidence of better localization -- an estimator that always
answers "metabolism" scores well in the world its bias happens to match. Only
the pair of worlds separates them, which is why the ground truth had to move. **It is right about where its body is and
wrong about what it is.** In the absorption world that misattribution is not
free: NLMS body error is 0.0590 against `snap`'s 0.0428, so an organism confident
about the wrong parameter does *worse* than one with no self-model at all.

The mechanism is structural, not incidental. NLMS moves along the current
sensitivity direction, splitting a residual in proportion to how sensitive each
parameter is at that moment, and cannot separate two constants whose effects
arrive together. Help arrives on a clock here, so "I absorb less per portion" and
"I burn faster between them" both look like a deficit proportional to elapsed
time. RLS keeps the accumulated second-order statistics that tell them apart.

**Prediction and attribution come apart, and accuracy gives no signal that the
attribution is wrong.** A self-model can be well calibrated about its own state
while being confidently wrong about its own nature. That is the most transferable
finding here.

## The controls

- **Shuffled readings** (G5). Readings replaced by another organism's body,
  matched in count and timing. Body error: `population` 0.0785, `snap` 0.1998,
  NLMS 0.4865, RLS 0.3054; accuracy falls to 0.381 and 0.452 against
  `population`'s 0.665. Every tier that touches the evidence is severely harmed,
  including `snap`. Nothing here is exploiting the act of being corrected.
- **Null world** (G4). Both spreads zero. Total learned correction is **0.000**
  for both arms and body error is 0.0000 for every tier. An organism whose body
  is typical finds nothing and loses nothing. This is also the check that the
  three new ecology levers are inert at their defaults.
- **`snap`** is the control the whole result is stated against, not a
  convenience baseline. It receives identical evidence.
- **`individual`** receives no readings and is context, not a bound.

## Rate recovery

RLS recovers the effective burn multiplier to mean absolute error **0.0696**
(per seed 0.064 / 0.055 / 0.070 / 0.088 / 0.071) against a species prior error of
0.300 -- a 4.3x improvement on assuming it is typical, 5/5 seeds.

The NLMS arm gives 0.126 / 0.218 / **15.253** / 0.110 / 0.094: four seeds
recover the rate well and one diverges. That is the same defect as its
misattribution, seen through a different endpoint -- a greedy rule that puts a
residual on whichever parameter is most sensitive can walk a long way in the
wrong direction before anything contradicts it.

`recursive` also beats `individual` on named-need accuracy on every seed:
0.943 / 0.942 / 0.951 / 0.941 / 0.955 against 0.926 / 0.919 / 0.933 / 0.934 /
0.952.

## Closed loop, as context and never as a gate

Each tier speaking for itself, so each lives its own life. 5 seeds x 40 lives,
metabolism world, reading rate 0.03.

| tier | survival | report fidelity |
|---|---:|---:|
| `population` | 0.535 | 0.672 |
| `snap` | 0.620 | 0.780 |
| `learned` (NLMS) | 0.605 | 0.894 |
| `recursive` (RLS) | **0.640** | **0.926** |
| `individual` | 0.610 | 0.797 |

The belief-side gain does reach behaviour: **+10.5 survival points and +25.4
fidelity points** over the species filter. But the survey's warning holds exactly
as stated -- survival separates `population` from everything else and then
saturates, ranking `snap` 0.620, `individual` 0.610, `learned` 0.605 and
`recursive` 0.640 within a band that five seeds cannot resolve. Fidelity
separates cleanly (+14.6 points for `recursive` over `snap`). This is why
survival was not gated, and it is reported here rather than omitted because a
belief-side result that moved nothing at all would be worth knowing about.

One artefact worth flagging: `individual` reaches fidelity 1.000 in the ceiling
survey but only 0.797 here. The survey's version substituted this life's exact
constants into the filter, while this one expresses the same knowledge as a
frozen correction in the Jacobian parameterization, which carries a small
clipping-induced floor (open-loop body error 0.0171 rather than 0.0000). The
parameterization the learner uses is therefore slightly lossy, and every learned
number above is measured against that same lossy baseline rather than flattered
by it.

## Numerical divergence under the shuffled control, disclosed

In the preregistered run the NLMS arm's correction overflowed to NaN on one
shuffled-control seed. The comparison scored NaN as *not* exceeding
`population`, so G5 was recorded as 4/5 -- which still passes, and which is
conservative against NLMS, since a divergence is the control succeeding rather
than failing.

A sanity clamp on self-belief (no organism concludes its metabolism is a thousand
times its species rate: |log correction| <= 6.9 per entry) was added afterwards so
the divergence is reported as a number rather than a NaN.

It is verified not to bind anywhere the organism reads its own body. Total
absolute correction across all 21 entries is 1.083 / 1.034 (metabolism world,
NLMS / RLS), 1.879 / 1.614 (absorption), 1.332 / 2.938 (both) and **0.000 /
0.000** (null), against a clamp budget of 21 x 6.9 = 144.9. The largest observed
is 2% of the budget.

With the clamp the shuffled control gives NLMS 0.472--0.536 and RLS 0.277--0.319
against `population`'s 0.063--0.091, and **G5 passes 5/5 for both arms**. The
gate verdict in the table above is the original preregistered run's 4/5, which
passes either way; the clamped rerun is
`runs/organism/probe63_individual_self/shuffled_clamped.json`. Nothing else in
this document is affected, because the clamp is inactive in every other
condition.

## A methodological correction this probe owes

Probe60 made it binding to *check the oracle ceiling before locking a gate*.
Probe63 adds a clause it learned the hard way: **check it at the operating point
the gate will be scored at.**

G3's thresholds were taken from a pooled least-squares ceiling measured at
reading rate 0.10, while the treatment runs at 0.03. On noticing this, the
ceiling was remeasured at 0.03 and came back at 0.790 / 0.312 -- putting the
locked absorption gate of 0.45 *above* its own ceiling. That looked like the
exact defect probe60 warns about.

It was not, and the reason matters. The ceiling instrument requires a minimum
number of usable rows per need per life and discards lives without them; at rate
0.03 it discarded five of eight, leaving a three-life estimate. The online
recursive arm, pooling forty lives per seed, reached 0.531 -- above both the gate
and the "ceiling". **The ceiling was undersampled, not real.** Reported here
because the worry was genuine and acted on, and because an undersampled ceiling
is its own trap: it can close a mechanism that would have worked.

## Claim boundaries

- This is **parameter** self-discovery, not **structure** discovery. The organism
  finds the values of its own bodily constants and which of them differ from its
  species. It is still told that it has three needs and what kinds of thing act
  on them. Probe61 attempted structure discovery, failed four of five gates, and
  remains closed; nothing here reopens it.
- **Interoception is free and unprompted.** The organism never chooses to look at
  itself, so nothing here bears on the value of self-inspection -- which probe62
  measured in its own ecology and found nearly worthless.
- **The learning rules are analytic, not neural.** NLMS and RLS are standard
  online estimators over 21 scalars, run in numpy. The claim is about what an
  online self-model recovers, not about a network architecture.
- **The ecology is changed.** No number here is comparable to probe57's, and the
  oracle itself survives only ~0.70 at this spread because individuality makes
  some bodies unsurvivable. The null world exists to show the levers are inert
  when off, and it does.
- **Within-life only.** Each organism starts from the species prior at birth and
  learns about itself over one life. Nothing is carried across lives, so this is
  not a claim about developmental or evolutionary self-knowledge.
- **Survival is context and was never gated**, for the reason the survey gave:
  this help loop separates the species filter from everything else but does not
  separate evidence from a self-model.
- Nothing here licenses claims of consciousness, sentience, phenomenal
  experience, or a metaphysically privileged self.
