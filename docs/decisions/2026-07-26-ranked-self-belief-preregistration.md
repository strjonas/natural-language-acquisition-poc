# Ranked continuous self-belief preregistration

Date: 2026-07-26  
Status: locked after probe 54 diagnostics and before implementation

## Failure-selected mechanism

Probe 54's learned recurrent self-belief failed its sole endpoint at 62.19%
balanced lowest-need accuracy versus the locked 70% gate. It passed survival,
change ordering, lesion, observation-control, and causal portion-fork gates.
Accuracy was stable across all four life buckets, ruling out late-life drift.

On a frozen whole-life 140/60 split, an affine calibration of the three
continuous outputs remained at 62.28%, but a disposable ridge readout of the
module's 64-dimensional recurrent state reached 73.53%. Thus the recurrent
state retains enough need-identity information and the MSE-trained continuous
readout loses its ordering.

## Sole change

Repeat the exact fresh probe-54 developmental run from the same probe-52 adult
checkpoint. Add weight 0.1 on a pairwise logistic rank loss over the same three
continuous body predictions and the same perceptible developmental food,
water, and energy targets:

`log(1 + exp(-sign(true_i - true_j) * (pred_i - pred_j)))`

Average over the three need pairs and all lived steps, then add it to the
unchanged continuous MSE. This teaches relational body geometry, not a lowest-
need class, report word, listener action, or correct utterance.

Everything else is fixed: seed 1, 64 recurrent units, Adam 3e-4, exactly 80,000
primitive ticks, one online update per lived sequence, uniformly random
body-independent grants, frozen pre-existing organism parameters, no replay,
no report credit, and no current body in recurrent input after birth.

## Gates and stop rule

Use the unchanged 200-life probe-54 gates:

- balanced lowest-need accuracy >=70%;
- current-observation balanced accuracy <=45%;
- ten-tick realized self-change ordering >=80%;
- zero and shuffle lesions each reduce balanced accuracy by >=20 points; and
- causal portion-fork following >=60%.

The existing 60% fidelity and 80% survival values may be reported but cannot
replace any gate. If any gate fails, do not enable the social word planner and
do not tune rank weight, margin, optimizer, width, duration, or seed. If all
pass, recheck the frozen lexical gate and only then implement the separately
declared listener-consequence planner. No scale or generated data is licensed.
