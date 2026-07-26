# Relational urgency self-belief preregistration

Date: 2026-07-26  
Status: locked after probe 55 and before implementation

## Failure-selected mechanism

Probes 54 and 55 show the same separation: a disposable linear readout of the
64-dimensional recurrent belief state exceeds the 70% identity gate, while a
three-value continuous body head remains near 62% even after rank pressure.
The one head is being asked both for calibrated magnitude and categorical
ordering.

## Sole change

Repeat the exact fresh probe-54 developmental run from the same probe-52 adult
checkpoint. Keep its three-value continuous body head and MSE unchanged. Add a
separate three-axis relational urgency head from the same recurrent belief
state. Train that head only with the pairwise logistic ordering of perceptible
continuous body values, weight 1.0:

`log(1 + exp(-sign(true_i - true_j) * (urgency_i - urgency_j)))`

At evaluation, the lowest urgency axis names the inferred need. Continuous
values still supply magnitude MAE and realized-change metrics. No lowest-need
class, correct report word, listener response, or utterance target is present.

Everything else remains identical: fresh initialization, seed 1, width 64,
Adam 3e-4, exactly 80,000 ticks, random body-independent grants, one update per
sequence, no replay, body masked from recurrent input after birth, and all
pre-existing organism parameters frozen.

## Gates

The unchanged 200-life probe-54 gates apply. The urgency head must reach >=70%
balanced identity; observation <=45%; continuous ten-tick ordering >=80%;
zero/shuffle urgency lesions each cost >=20 points; and urgency portion-fork
following >=60%.

If any gate fails, close this developmental supervision architecture. Do not
tune urgency loss, width, budget, optimizer, seed, or thresholds. If all pass,
recheck lexical comprehension and implement the already declared learned
listener-consequence planner. Passing is not self-report by itself and does
not authorize scale.
