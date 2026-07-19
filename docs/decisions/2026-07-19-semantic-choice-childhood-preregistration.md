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
- true-label bodily-consequence classification is at least 60%, exceeds
  padding by at least 10 percentage points, and has lower consequence MAE;
- substituting `danger` for a resource label or a needed resource for `danger`
  achieves at least 60% counterfactual-kind classification, changes the
  predicted bodily delta by at least 0.02 mean L1, and moves consume preference
  in the substituted direction in at least 60% of cases; and
- on the 100 resource contexts, swapping `food` and `water` achieves at least
  60% counterfactual-kind classification, moves both substituted- and
  true-kind bodily scores in the intended directions, changes the predicted
  bodily delta by at least 0.02 mean L1, and lowers consume probability in at
  least 60% of cases. This blocks a merely binary `safe`/`danger` code.

If the policy never inspects, the next probe adds learned expected information
gain about bodily consequences to planning. It may not reward tokens or asking.
If it inspects but fails the delayed binding gate, add a learned dual-code
episodic key/value memory and compare it to the GRU under the same trials. Do
not tune token loss, planning scale, or vocabulary after seeing this result.

## Replication and transfer gate

Only if the grounded probe passes, run seed 2 and matched silent/shuffled
training. Grounded must exceed both controls across seeds and pass acute
silence/counterfactual-token interventions. Because paired behavior itself can
be solved with a binary safe/danger distinction, a three-way
needed-resource/wrong-resource/poison trial must then demonstrate behavioral
food/water/danger use before open-island transfer. Only after that test, train a
fresh organism with 20k paired childhood ticks followed, without resetting
weights or optimizer, by 20k unsupported open-island ticks. The transfer claim
requires retained semantics plus improved open-island survival over matched
silent/shuffled developmental histories.

This developmental task can establish a minimal danger/safety behavioral use
plus a distinct word-to-bodily-consequence mechanism. It cannot by itself
establish three-way behavioral semantics, generated language, self-report,
reflection, autobiographical identity, authentic human-like desire,
consciousness, or a human-like `me`.

## Pre-result validity hardening

Before any paired-choice training result was run, a code audit found defects in
the preceding open-island label audit and a construct review found that paired
behavior is only a binary safe/danger test. The behavioral thresholds are
unchanged; the causal gate is strengthened with the preregistered food/water
swap above. The implementation is hardened as follows:

- grid generation and caregiver sampling use deterministic but distinct RNG
  streams;
- paired childhood permits only object-ahead ASK/POINT labels; empty-space ASK
  cannot invoke the open island's need-matched `answer_where` speech act, and
  terminal praise/correction is suppressed;
- grounded and shuffled choice labels are both the two-word form
  `this <kind>`; shuffled kinds are independently drawn from the pair-matched
  25% food / 25% water / 50% danger marginal, removing utterance-length and
  template-family shortcuts;
- the label audit takes at most one target from each held-out trial and
  alternates exactly between the need-matched resource and poison, yielding 200
  unique contexts with a 100/100 class balance;
- true and counterfactual branches use the in-distribution controlled forms
  `this food`, `this water`, and `this danger`, changing one functional token
  while keeping the visual referent fixed;
- direct token-to-policy probability shifts and the incremental contrast when
  the nonlinear planner is enabled are reported separately (the latter is not
  claimed as pure mediation); bodily-delta correctness remains the primary
  self-model endpoint;
- the option servo routes around visible blockers and moves off a selected
  under-agent object before asking or consuming; the audit rejects any label
  whose reported surface/kind does not match the selected target; and
- held-out worlds inherit the model's visible-slot layout, with regression
  tests for non-default layouts.
- the broad all-action body-ranking audit is labeled `prelabel` and treated
  only as model-health calibration, not promotion evidence; the balanced
  post-inspect label-to-bodily-delta audit carries the causal gate.
- the open island's additive per-tick survival bonus is disabled only inside
  fixed-horizon choice trials, where it would reward delaying until timeout;
  the real homeostatic change reward remains and telescopes across movement,
  inspection, correct repair, poison, and waiting.
- per-segment advantage normalization is disabled inside short choice lives;
  otherwise a successful early inspect receives a negative normalized
  advantage solely because the later consume return is larger. Ordinary long
  island segments retain normalization.
- the final choice life is shortened at the primitive boundary when necessary,
  so the 20,000-tick childhood and later childhood-to-island transition are
  exact rather than overshooting by up to one trial.
- horizon-two auxiliary windows that cross a newly arriving non-padding label
  are masked: the current open-loop transition cannot encode that intermediate
  observation, so fitting the later outcome would otherwise train an
  irreducible average. One-step post-label body learning remains active.

These changes reduce false positives; none adds language reward, privileged
kind input to learning or acting, demonstrations, extra training data, or a
new promotion criterion.

Pre-training calibration over 5,000 fixed held-out seeds gives 49.96% correct
for a random side, 49.40% for a deterministic lowest-surface-index policy, and
100.00% for a scripted policy that uses only the first inspected canonical
label plus remembered learner-visible coordinates. The task is therefore
solvable through the intended channel while the tested nonlinguistic priors are
at chance.
