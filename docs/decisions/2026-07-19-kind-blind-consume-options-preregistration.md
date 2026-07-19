# Kind-blind sensorimotor option preregistration

Date: 2026-07-19
Goal-level metric: independent adult survival, then grounded-language benefit.

## Why the control level changes

Probe 6 closes primitive-action curricula. Hand and distance-one support
produced long lives, but distance-two support collapsed to 57.6 steps. In the
unsupported adult phase the agent achieved 1.10 unoffered successful resource
uses per life and 0% survival. Flat A2C did not compose orientation, movement,
target tracking, and consumption before the delayed homeostatic payoff.

The organism now gains one sensorimotor option per visible object slot. Choosing
that option navigates to the egocentric target location and attempts consume.
The controller may read only location and proprioceptive orientation; it cannot
read the object's hidden kind, caregiver semantics, or future bodily effect.
Thus it supplies a reach/grasp-like motor primitive while leaving the
self-relevant decision intact: the recurrent organism must decide *which*
surface to use from its private need, history, percepts, and heard language.
Choosing poison remains harmful.

Options take multiple environment ticks. Training and evaluation use summed
environment reward, summed viability rent, minimum viability across the option,
and duration-discounted semi-Markov GAE. Model action-consequence heads predict
the option's final observation/body/reward from the same shared recurrent state.

## Probe 7

First run a short mechanics/learning smoke. Then, if options are actually used,
run 200k grounded steps with the proven proximal-to-distal caregiver childhood
(threshold 0.75, distance 0 -> 2, both ending at 100k), seed 1, hidden 256, and
20 held-out unsupported adult lives.

Primary pass: nonzero held-out adult survival. Required diagnostics: successful
resource uses, harm, option duration accounting, and comparison to primitive
probe 6. This establishes motor competence only. No language claim is made
until the exact option architecture is run grounded/silent/shuffled and the
heard-token channel is intervened on.

If options are learned but survival remains zero, the next architecture is a
latent planner (TD-MPC/Dreamer-lite style), not another reward or curriculum
change.
