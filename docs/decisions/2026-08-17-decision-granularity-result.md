# Probe68: the granularity of an ecology's decisions — result

Date: 2026-08-17. Preregistration:
`docs/decisions/2026-08-17-decision-granularity-preregistration.md`.
Ceiling survey: `docs/decisions/2026-08-17-decision-granularity-ceiling-survey.md`.
Module: `src/homesocial/organism/decision_granularity.py`. Diagnostic:
`scripts/diagnose_probe68.py`. Guards: `tests/test_decision_granularity.py`.
Artifacts: `runs/organism/probe68_granularity/treatment.json` and the survey files.

Treatment: 8 seeds x 30 lives per cell, band 1,400,000,000, stride 2,000,000,
disjoint from the diagnostic (1,260,000,000) and survey (1,320,000,000) bands.

**Verdict: four of the five locked gates pass. G5 fails and is not rewritten.**
The failure is informative and is recorded in section 5 with the same care as the
passes, because the quantity G5 was *meant* to test passed to four decimals while
the gate written on it failed by five times its band.

| gate | requirement | result | |
|---|---|---|---|
| **G6** | rate value rises with grain, state value does not | rate **+0.0855**, 8/8 seeds; state **+0.0101**, spans zero | **pass** |
| G2 | nutrition null inside +/-0.02, half-width < 0.02 | **+0.0186** [+0.0109, +0.0264], half-width 0.0077 | **pass** |
| G3 | grain response > 0, CI clear of zero, >= 6/8 seeds | **+0.0335** [+0.0079, +0.0591], 6/8 | **pass** |
| G4 | second step smaller than first | +0.0335 then **+0.0166** | **pass** |
| G5 | lag-0 null inside +/-0.02 | **+0.1072** [+0.0906, +0.1238] | **fail** |

## 1. The result: same food, finer grain, and knowing what you are is worth twice as much

Three rows. The caregiver's **rate** of help is identical in all of them to the
last decimal -- `grant / help_period` is 0.0667 per tick throughout -- and only the
**quantum** differs: smaller portions delivered proportionally more often.

| quantum | help_period | grant | grant/tick | margin | **rate value** | **state value** |
|---|---:|---:|---:|---:|---|---|
| 1 | 6 | 0.4000 | 0.0667 | 0.1749 | **+0.1272** [+0.1038, +0.1506] | +0.2271 [+0.1934, +0.2607] |
| 1/2 | 3 | 0.2000 | 0.0667 | 0.0878 | **+0.2127** [+0.1846, +0.2409] | +0.2371 [+0.2152, +0.2590] |
| 1/3 | 2 | 0.1333 | 0.0667 | 0.0669 | **+0.2775** [+0.2404, +0.3145] | +0.2293 [+0.2091, +0.2495] |

    rate value   =  oracle - state_oracle        both hold the true body; they
                                                 differ only in whose rates.
    state value  =  state_oracle - population    both hold species rates; they
                                                 differ only in read vs filtered.

> Halving the grain of the help, with the amount of help held exactly fixed,
> **more than doubles** the value of knowing what kind of body you are -- and
> leaves the value of knowing where that body is where it was.

### G6, the gate this probe exists for

Paired per seed, `quantum_half` against `nutrition_1x`:

| step | mean | 95% CI | seeds agreeing |
|---|---|---|---|
| **rate value** | **+0.0855** | [+0.0753, +0.0956] | **8/8** |
| state value | +0.0101 | [-0.0088, +0.0289] | 5/8 |

The rate step is **8.5x** the state step, every seed agrees on its sign, and the
state step's interval contains zero. Per-seed rate steps: +0.0696, +0.0735,
+0.0810, +0.0824, +0.0863, +0.0882, +0.0948, +0.1081.

**State value is the lesion, and it holds.** If a finer decision surface simply
made every decision harder, both would have risen together and this claim would be
empty. It does not: across the three compensated rows state value reads +0.2271,
+0.2371, +0.2293 -- flat inside its own intervals while rate value goes +0.1272 ->
+0.2127 -> +0.2775.

### G4: the response follows the reach, not the quantum

Reach -- the share of consequential decisions the rate error is large enough to
reach -- was measured **before** the treatment at 0.4407, 0.7600, 0.7972 for
quanta 1, 1/2, 1/3. It rises steeply then flattens, and so does the response:
first step **+0.0335**, second step **+0.0166**. A mechanism whose value were
linear in the quantum would have passed G3 and failed here.

## 2. Why, in one line

State error enters the projected body `level - rate * H` **once**. Rate error
enters it **multiplied by H**. At `H = 18` a rate error is therefore a large
quantity -- and the coarse margin was **swallowing** it. Making the margin finer
does not create the information; it stops the ecology from discarding it.

That is probe65's sentence with the missing half supplied. Probe65 showed the
horizon puts the rate error *into* the decision variable. Probe68 shows the grain
decides whether the decision variable can *express* it.

## 3. The 400-fold range, from two ecology parameters

The same self-model object, the same arithmetic, the same frozen parent:

| ecology | rate value |
|---|---|
| lag 0, grant 0.40 | **+0.0007** [+0.0005, +0.0010] |
| lag 0, grant 0.80 | **+0.0009** [+0.0003, +0.0015] |
| lag 18, quantum 1 | +0.1272 |
| lag 18, quantum 1/3 | **+0.2775** [+0.2404, +0.3145] |

At lag 0, `state_oracle` scores **0.9995** named-need accuracy: a body read
perfectly and believed to be typical is essentially never wrong, so there is
nothing for a self-model to add. **The value of knowing what you are ranges over
roughly four hundredfold across two ecology parameters, and the granularity
formula orders it.** No property of the organism changed.

## 4. G2 passes, and it is not a null

`nutrition_2x` doubles the grant rate to 0.1333 per tick. Granularity is
arithmetically **identical** at that grant -- measured 0.14837 at both scale 1.0
and 2.0 on fixed trajectories, to five decimals.

The mixed contrast moved **+0.0186 [+0.0109, +0.0264]**, half-width 0.0077. By the
preregistered rule that is a pass: inside the +/-0.02 band and resolved. **It is
not zero.** The interval excludes zero and 8/8 seeds agree, so the honest
statement is "+0.019, smaller than the band I locked", not "nothing happened".

And the preregistration's own warning about this cell is confirmed: doubling the
grant does **not** hold the realised decision surface fixed. State value collapses
from **+0.2271 to +0.0857** and the consequential share falls from 0.753 to 0.229.
A larger grant changes the body distribution and so the axis gap -- the formula's
second term -- even while leaving its first term saturated.

> **G2 is a weak null and G6 is a strong positive, and they should not be read as
> equal evidence.** What the pair supports is that grain matters; it does not
> establish that resources do not.

## 5. G5 fails, the gate stands, and the prediction it was written for passed

G5 required the lag-0 contrast to be flat across grant size. Measured
**+0.1072 [+0.0906, +0.1238]** against a +/-0.02 band -- five times outside, 8/8
seeds. The contrast is negative at both scales, so the sign clause held; the
equivalence clause failed outright.

**The gate is not rewritten.** The reason it failed is that G5 was the one gate
left on the *mixed* contrast `individual - state_oracle`, and at lag 0 that
contrast contains no rate signal at all -- so it is pure state value, which the
grant moves a great deal (+0.4142 -> +0.2424).

The prediction G5 was written to test -- that at horizon 1 the rate is too small
to matter at any grant size -- is confirmed about as sharply as this repository
can confirm anything: rate value **+0.0007** and **+0.0009**, both intervals
excluding zero and both under a thousandth of the body.

So: **a locked gate failed, and the thing it was aimed at is true.** That is a
measurement of the gate's construction, not of the world, and section 6 of the
preregistration already says why the mixed contrast was abandoned for exactly this
reason. G5 was the residue of the abandoned design and it should have gone with it.

## 6. What this changes about how to plan the next probes

**Probe67's floor is now a formula with a knob in it.** `STATE.md` carried
"anything under ~0.12 cannot change a consequential word" as a precondition. The
replacement is

    margin = E_grant[ min(grant * uptake, axis gap) ]

reproduced with MAE 0.000000 at two of four lags, and the 0.12 is its saturated
branch. The check to run before building is no longer "is my effect bigger than
0.12" but "**what is the margin of the ecology I am proposing, and is my effect
bigger than that**".

**And the floor is not freely choosable.** The ceiling survey found oracle
survival **0.000** at half the grant: the ecology's grain is pinned from below by
viability. Five separate negatives -- probes 59, 60, 62, 64, 67, each a belief
that improved and a behaviour that did not hear it -- now have one structural
account rather than five. **The grain of a decision surface is downstream of how
much help a body needs to stay alive**, and the way past it is not a bigger
grant but a **smaller, more frequent** one.

## 7. Claim boundaries

- **The endpoint is what is said, not how long anything lives.** Every number in
  sections 1 through 5 is open-loop named-need accuracy on a shared history. The
  closed loop appears only in the viability check. Probe65 remains the only place
  in this repository where a belief has reached survival.
- **`rate value` is arithmetically `1 - state_oracle` accuracy**, because `oracle`
  scores exactly 1.0000 by construction. That is the cleanest isolation available
  -- perfect state with true rates against perfect state with species rates -- and
  it is not more than that.
- **Reach is an upper bound, not a predictor of level.** It ordered the response
  correctly and over-predicts `state_oracle`'s error by 0.16.
- **The axis-gap term was never moved.** Only the grant was. Moving the gap means
  a body with more axes than this simulator has, and that is the next lever
  rather than a result here.
- **Probes 59 and 60 remain unretrodicted.** The shift-equivariance clause that
  would have explained them was refuted before the treatment ran; the homeostatic
  cap binds on 82.6% of lag-18 ticks and converts common-mode error into
  differential error.
- **`nutrition_2x` is confounded** in the direction section 4 states, and was
  preregistered as such.
- One ecology, one frozen parent checkpoint, one motor policy, three needs. A
  granularity law measured in a world with three bodily axes is **not** shown to
  hold in a world with thirty.
- Nothing here is about discovery, uncertainty reporting, bodily structure,
  language productivity, consciousness, or sentience.

## 8. Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_probe68.py --lives 8 --delays 0,18,24,30
PYTHONPATH=src .venv/bin/python scripts/diagnose_probe68.py --quantum --lives 28 --delays 18
PYTHONPATH=src .venv/bin/python -m homesocial.organism.decision_granularity --ceiling --lives 30
PYTHONPATH=src .venv/bin/python -m homesocial.organism.decision_granularity --quantum --lives 25
PYTHONPATH=src .venv/bin/python -m homesocial.organism.decision_granularity \
    --treatment --lives 30 --seeds 8 --out runs/organism/probe68_granularity/treatment.json
```

The treatment is roughly ninety minutes on this machine. Everything else is
minutes. `portion_scale` and `help_period` are default-off on
`run_open_loop`/`run_closed_loop`, and `tests/test_decision_granularity.py`
asserts inertness at the default rather than trusting it.
