# Proximal-to-distal caregiver childhood preregistration

Date: 2026-07-19
Goal-level metric: independent adult resource seeking and survival.

Probe 5 showed a sharp dependency boundary: hand offers supported 309.5-step
mean lives and 18.9 successful resource consumes while available, but the
second 100k steps without offers fell to 55.5 steps and 1.18 successful
resource consumes per life. A slower time-only fade would postpone feeding
without teaching the missing approach/face sequence.

Probe 6 makes the one permitted competence-oriented curriculum change. Offered
resources begin in hand, then move linearly to Manhattan distance two while
remaining ordinary learner-visible surface percepts. The caregiver repeats the
same grounded `OFFER(kind)` utterance. The agent must learn consume-in-hand,
turn-and-consume, then approach-turn-consume before offers disappear. No action
labels, kind percepts, or added rewards are introduced.

Setup: grounded, seed 1, zero BC, threshold 0.75 fading over 100k steps,
offer distance 0 -> 2 over the same interval, 200k total steps, then 20 held-out
adult lives without offers. Compare probe 5 and probe 1.

Pass: nonzero held-out adult survival, with unoffered successful resource uses
remaining above the metabolic requirement in the second 100k steps. Kill: if
the agent learns distal offered resources but again collapses at zero support,
the flat primitive-action A2C control substrate is closed; implement latent
model-based planning or explicit learned options rather than another childhood.
