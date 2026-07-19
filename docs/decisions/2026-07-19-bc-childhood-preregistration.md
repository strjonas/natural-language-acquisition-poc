# BC childhood preregistration

Date: 2026-07-19
Goal-level metric: adult survival learning, prerequisite of G1.

## Why this probe exists

Three from-scratch probes failed because the organism never learned the local
approach -> face -> consume motor sequence. This probe tests the next sanctioned
mitigation from `PLAN_ORGANISM.md`: a small behavior-cloning childhood followed
by the unchanged online lifelong loop.

The oracle reads hidden object kinds. Its demonstrations therefore cannot count
as evidence that the learner acquired language, even though the oracle itself
never reads caregiver tokens. To prevent the strongest contamination, every
caregiver token in the imitation data is replaced by padding. BC can teach
learner-visible sensorimotor sequences and body consequences, but it cannot
directly teach a word-to-action mapping. All language claims still require a
zero-BC control and grounded/silent/shuffled adult comparisons.

Rare action labels are square-root inverse-frequency weighted (capped at 8x)
so `consume` is not erased by the oracle's many `wait`/`rest` steps. The
world-model observation, need, and reward heads share the BC trajectories; the
caregiver-token prediction head does not, because the BC language is masked.

## Probe 4

- Childhood: 100 token-masked oracle lives, 2 BC epochs, seed 1.
- Adulthood: 200k grounded online steps, unchanged adult metabolism and A2C
  settings from probe 1.
- Held-out evaluation: 20 lives, seeds starting at 900000, 1000-step horizon.
- Mandatory comparison: probe 1's exactly matched zero-BC grounded run.

Primary pass: held-out survival > 0% and a clear increase in mean life length
beyond the ~54-step thirst clock. Supporting diagnostics: BC consume recall,
online consume-event counts, and persistence or erosion of competence during
RL.

Interpretation rules:

1. If BC itself has near-zero consume recall, the bootstrap implementation—not
   online RL—is the failed level; change the demonstration curriculum once.
2. If BC imitates consume but adulthood immediately erases it, test one staged
   BC/RL mixing schedule. Do not tune entropy again.
3. If survival works, run grounded/silent/shuffled with identical token-masked
   childhoods. A grounded advantage is not credited until it survives that
   comparison and an internal token-channel intervention.
4. If the warm-start policy survives equally under silent input before online
   learning, BC has supplied more than motor competence. Report that
   contamination and switch to caregiver OFFER-based in-loop childhood.
