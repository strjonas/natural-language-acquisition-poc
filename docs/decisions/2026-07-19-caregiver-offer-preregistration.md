# In-loop caregiver-offer childhood preregistration

Date: 2026-07-19
Goal-level metric: adult survival learning, prerequisite of G1.

## Diagnosis propagated upward

Token-masked BC on 100 privileged oracle lives reached 0.81 imitation accuracy
and 0.82 consume recall, but both sampled and greedy held-out policies died at
~55 steps before online learning. After 200k online steps the organism died at
exactly 54 steps and consumed only 0.8 times per evaluation life. The failure
is not merely catastrophic forgetting: the oracle acts on a global kind-aware
map that the learner cannot observe, so its policy is not fully imitable from
the learner stream. More BC replay would preserve an unrepresentable mapping.

## Developmental substrate change

During infancy, when food or water drops below a threshold, the caregiver puts
a renewable resource in the learner's hand. The offered surface is an ordinary
learner-visible percept at egocentric `(0, 0)`, and grounded caregiver tokens
use the existing `OFFER(kind)` speech act. The organism must choose `consume`;
no reward is added beyond the resource's actual effect on its body. Ignored
offers persist and are repeated. Care fades linearly to threshold zero, and
held-out adult evaluation has no offers.

This creates dense in-loop examples of:

`private need -> social offer/percept -> own consume action -> bodily recovery`

without cloning actions, exposing hidden kinds, or rewarding language.

## Probe 5

- Zero BC.
- Grounded language, seed 1, otherwise probe-1 defaults.
- Offer threshold starts at 0.75 and fades to 0 over 100k steps.
- 200k total online steps; 20 held-out 1000-step adult lives with no caregiver
  offers.
- Exact no-offer comparison: probe 1.

Pass: nonzero adult survival and successful resource-consume retention after assistance is
fully absent. Supporting evidence: life length/consume counts remain above the
metabolic clock in the second 100k steps, not only during care.

If the agent consumes while offers exist but collapses as they fade, make one
slower competence-contingent fade. If it never learns offered consumption,
stop changing curricula and replace primitive navigation/control with learned
options or a model-based planner; the current A2C substrate is then the wrong
level.
