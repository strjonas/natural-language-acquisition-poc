# Inspect--remember--choose childhood preregistration

Date: 2026-07-19
Goal-level metric: acquire correct grounded label-to-body meanings in-loop
before transferring them to unsupported island life.

## Why the developmental topology changes

The balanced open-island probe creates information rent and the organism asks
for labels, but correct label-to-consequence learning is sparse. A label only
receives useful credit if the organism later consumes the same surface, and
long navigation, other objects, metabolism, and death intervene. Two grounded
seeds show causal token sensitivity without replicated consequence correctness;
one shuffled seed can satisfy the token-shift diagnostic by chance.

The next substrate is an infancy, not a supervised pretraining task. The same
organism, losses, homeostatic reward, body, tokens, inspect/consume options, and
online loop remain. Each short life simply concentrates the natural causal
sequence: jointly inspect an object, retain its label, choose, experience the
bodily result.

## Paired choice trial

- At reset, randomly choose hunger or thirst as the low need.
- Uniformly reshuffle the same five surface-to-kind meanings with the exact
  2-food/2-water/1-poison quota.
- Place two randomly ordered, learner-visible surfaces at distance two: one
  whose hidden kind repairs the current low need and one poison.
- Any food/water/poison consumption ends the trial. A correct choice improves
  the real body; poison harms it. There is no correctness, inspect, question,
  language, or token reward.
- A trial lasts at most 20 primitive ticks. The caregiver labels only through
  the existing embodied inspect/ask interaction. Grounded, padding, and
  independently per-event shuffled modes keep identical bodies, objects,
  observation shapes, action shapes, and budgets.
- Record whether the agent inspected the ultimately consumed surface, the
  chosen hidden kind, correct need-matched choice, poison choice, and timeout.

This is a configuration of the island and the one organism harness, not a new
offline probe or a scripted teacher policy.

## First grounded probe

- Seed 1, hidden 64, 20k actual primitive ticks entirely in paired childhood.
- Consume and inspect visible-slot options; no caregiver offers and no BC.
- Replay reservoir 256, one auxiliary update per online update, horizon-two
  latent loss weight 1.
- Planning scale 6, horizon two, activated at 10k ticks.
- Evaluate 500 fixed held-out paired trials with no parameter updates, plus 200
  label-to-self-model interventions. Report stochastic and greedy choice
  behavior separately.

Promotion requires all of:

- at least 60% overall correct need-matched choices (chance is 50% conditional
  on consuming one of the pair);
- at least 65% correct choices among trials where the chosen surface was first
  inspected;
- nontrivial voluntary inspection and at least 100 inspected held-out choices;
- true-label bodily-consequence classification exceeds padding by at least 10
  percentage points and true-label consequence MAE is lower than padding;
- substituting `danger` for a resource label or a needed resource for `danger`
  moves predicted bodily outcome and consume preference in the substituted
  direction; and
- body-only action ranking remains positive.

If the policy never inspects, the next probe adds learned expected information
gain about bodily consequences to planning. It may not reward tokens or asking.
If it inspects but fails the delayed binding gate, add a learned dual-code
episodic key/value memory and compare it to the GRU under the same trials. Do
not tune token loss, planning scale, or vocabulary after seeing this result.

## Replication and transfer gate

Only if the grounded probe passes, run seed 2 and matched silent/shuffled
training. Grounded must exceed both controls across seeds and pass acute
silence/counterfactual-token interventions. Then train a fresh organism with
20k paired childhood ticks followed, without resetting weights or optimizer,
by 20k unsupported open-island ticks. The transfer claim requires retained
paired-choice semantics plus improved open-island survival over matched
silent/shuffled developmental histories.

This developmental task can establish a correctly grounded comprehension
mechanism. It cannot establish generated language, self-report, reflection,
autobiographical identity, authentic human-like desire, consciousness, or a
human-like `me`.
