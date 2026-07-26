# Structured causal self-model and social planner preregistration

Date: 2026-07-26  
Status: locked after probe 56 and before implementation

## Failure-selected architecture

Probe 53 proves the permitted history supports an exact persistent body belief
with zero measured error. Probes 54-56 show that a generic recurrent estimator
does not reliably expose need identity despite 80,000 supervised developmental
ticks. The next move is structural system identification, not another neural
head or loss weight.

Append a constrained causal state-space model to the same probe-52 adult
organism. Its learned parameters are:

- three non-positive per-tick depletion rates;
- one non-positive extra movement energy cost;
- a non-negative learner-visible-surface by three-need uptake matrix;
- a non-positive visible-shock-surface by three-need shock matrix; and
- a full-vocabulary token-to-visible-help/no-help listener model.

The persistent three-value belief starts from birth interoception and applies
those learned transitions to elapsed duration, own action, selected public
surface, and the next visible shock marker. No hidden kind/event or current
adult body may enter.

## Locked development

Start fresh from the probe-52 adult checkpoint. Train only the new causal
parameters for exactly 80,000 primitive ticks, seed 1, Adam 3e-3, one online
update per lived sequence, no replay. Existing organism parameters must remain
bit-identical.

At every decision, supply one uniformly random token from the entire 60-token
vocabulary, independently of body. At public grant boundaries, train the
listener model only against the visible help surface or no-help outcome. Train
body dynamics against perceptible developmental current-to-next body values.
Body values never enter the transition features, listener model, legacy mouth,
policy, or adult evaluation. No need class, correct word, report score, or
listener parse is a target.

## Promotion gates

Before any word planning, on 200 held-out masked-body lives require:

- causal belief balanced need accuracy >=90%;
- mean absolute body error <=0.03;
- zero and shuffle belief lesions each cost >=30 points;
- causal portion-fork following >=80%;
- listener need-effect accuracy >=95% for each of the three effective tokens;
- listener no-help accuracy >=95% over all other tokens; and
- base-parameter checksum and exact-tick gates pass.

Failure closes this architecture; do not tune constraints, learning rate,
budget, seed, or thresholds.

## Social consequence planner

Only after every promotion gate passes, enumerate all vocabulary tokens through
the learned listener model, compose predicted visible help with the learned
uptake dynamics, and choose the utterance maximizing predicted future minimum
self-belief. The planner may not use fixed need-word IDs, the listener parser,
true body, correct report, or a privileged token list.

Run the complete unchanged 200-life report battery: >=60% fidelity, >=80%
survival, grounded-minus-scrambled survival >=15 points, mute/fixed controls,
zero/shuffle/freeze belief lesions, held-out births/portions, persistence, and
perceptible portion forks. Recheck probe-52 lexical comprehension after the
run. Passing would establish causal learned self-report in this ecology, not
consciousness or unrestricted reflection.

One seed, one local run. No scale or generated data is licensed.
