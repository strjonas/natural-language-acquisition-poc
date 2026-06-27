# Preregistration: M0 Experimental Validity

Date: 2026-06-23

## Purpose

M0 validates the Homeostatic Social Grid before learner claims are made. The
goal is to prove that teacher language has causal value in the diagnostic setup
and that comparisons across teacher conditions are not confounded by input
shape or hidden-kind leakage.

## Fixed Configuration

- Environment: `HomeostaticSocialGrid`
- Diagnostic mode: `language_necessary`
- World placement: fixed positions unless explicitly labeled randomized
- Learner-facing object kinds: hidden
- Language channel: included for matched-shape comparisons
- Teacher modes: `grounded`, `silent`, `masked`, `shuffled`, `wrong`
- Diagnostic seeds: `1..12` for unit-level audit, `1..40` for command-line M0
  audit
- Shuffled teacher seed: same as run seed

## Metrics

- `mean_viability`
- `min_viability`
- `resource_uses`
- `danger_hits`
- `teacher_utterances`

The primary M0 metric is paired mean viability across diagnostic seeds. Resource
use and danger hits are secondary behavioral checks.

## Gates

M0 passes only if all gates hold:

1. Observation vector size is invariant across teacher modes when
   `include_language_channel=True`.
2. Masked language uses the null/OOV language token while preserving the full
   language-channel shape.
3. Hidden object kind does not change the masked observation vector in the fixed
   diagnostic layout.
4. The grounded teacher-following oracle beats the best blind diagnostic
   control by at least `0.01` mean viability.
5. The grounded oracle produces more useful resource actions than silent,
   masked, and blind controls.
6. Shuffled and wrong teacher modes score below grounded on mean viability.

## Allowed Deviations

Allowed before M1:

- Increase diagnostic seeds if estimates are noisy.
- Tighten gate margins after results are stable.
- Add more blind controls if a new positional shortcut is found.
- Fix environment bugs that leak hidden kind or make teacher utterances
  non-actionable.

Not allowed without recording a new preregistration section:

- Changing teacher utterance semantics after seeing learner results.
- Rewarding language directly.
- Comparing grounded and silent learners with different input shapes.
- Treating self-report as evidence before the self-battery exists.

## Post-Run Log

No deviations yet.

## M7 Hidden-Interoception Audit

Date: 2026-06-25

Purpose:

- remove exact need values from model input without changing observation size,
- test whether recurrent action/outcome history supports body-state inference,
- make the inferred state causally useful through a teacher-mediated report.

Primary inference controls:

- trained full history,
- trained latest observation only,
- trained shuffled prior history,
- random weights with the same architecture.

Primary inference metric: next-need MSE. Secondary metric: lowest-need accuracy.

Primary mediation metrics:

- report accuracy,
- helpful-resource use rate,
- irrelevant-resource use rate,
- helpful-use precision,
- target-need delta.

Protocol deviations recorded during development:

1. The first long-horizon mediation metric was rejected after a constant
   `need_rest` report exploited repeated shelter use.
2. Balanced one-step triage validated the communication protocol but was
   insufficient because random representations retained directly observed
   needs.
3. Exact needs were then masked and the final hidden-state audit used natural
   action/outcome histories.
4. Overall mediation decision accuracy was demoted to secondary because
   non-helpful object/need pairs dominate. Selective helpful versus irrelevant
   use is the primary causal measure.

## M8 Stochastic Body And Compositional Report Audit

Date: 2026-06-25

Primary inference gate:

- full-history next-need MSE below shuffled and reversed history,
- full-history MSE below latest-only and random controls,
- replicate on two training seeds.

Primary report gate:

- score need, severity, and trend separately,
- report confidence coverage and conditional need accuracy,
- keep symbolic cause accuracy separate from learned state fields.

Primary mediation gate:

- full-history helpful-use precision above latest-only and random controls,
- irrelevant-use rate remains below `0.02` on both trained seeds.

Recorded deviation:

- the first stochastic body used additive shocks and did not create a strong
  order requirement; it was rejected and replaced by persistent metabolic
  pressures before the final runs.

## M9 Emergent Message Audit

Date: 2026-06-25

Message constraints:

- two categorical slots,
- three symbols per slot,
- no direct message-label supervision,
- sender receives only the learned predicted body state,
- receiver receives only message and object kind.

Gates:

- balanced decision accuracy above `0.80`,
- all four social intents represented in the learned codebook,
- shuffling both slots reduces decision accuracy by at least `0.10`,
- natural-world mediation precision above `0.75` on two body-model seeds,
- latest-only history mediation precision below full history.

Recorded deviation:

- natural-frequency training omitted a stable food symbol. Intent-balanced
  sender/receiver training was introduced; natural-world mediation remained
  unchanged.

## M10 Convention Transfer Audit

Date: 2026-06-26

Transfer constraints:

- train two independent sender/receiver protocols from different random seeds,
- keep the sender input restricted to the learned predicted body state,
- test zero-shot sender/receiver swaps without remapping symbols,
- freeze one sender and train fresh receivers only from message, object, and
  use/avoid outcomes.

Gates:

- original protocol pair remains above `0.85` decision accuracy,
- independent zero-shot swaps should be reported as a failure unless they match
  original-pair balanced accuracy,
- frozen-sender receivers should approach original-pair balanced accuracy with
  small labeled interaction sets.

Recorded result:

- zero-shot swaps did not converge to a shared lexicon;
- a fresh receiver reached `0.8454` balanced accuracy from 32 sender examples
  and `0.8960` from 256 examples on the seed13 body checkpoint.

## M11 Population And Self-Request Language Audit

Date: 2026-06-26

Population constraints:

- multiple independently initialized senders,
- one shared receiver that never observes sender identity,
- sender input restricted to the learned internal self-state representation,
- no direct message-label supervision.

Primary gates:

- all senders remain useful to the shared receiver,
- dominant intent-symbol mappings align across senders,
- four request intents use distinct dominant symbols,
- request receiver chooses aid category from message alone.

Recorded deviations and outcomes:

1. The first shared-receiver use/avoid game aligned senders but allowed a lossy
   shortcut: food and water often shared one dominant message because object
   kind was still visible to the receiver.
2. The harder self-request game removed object-kind input. It failed on the old
   stochastic checkpoints because the compressed self-estimate itself only
   supported about `0.71-0.74` food/water/rest/avoid probe accuracy.
3. A stronger masked-interoception stochastic body model improved full-history
   next-need MSE to `0.002136` and lowest-need accuracy to `0.9165`.
4. With that body model, four senders plus one shared receiver reached `0.8384`
   request accuracy, `1.0000` intent-pair agreement, and `1.0000` dominant
   intent distinctness under mild agreement pressure.

## M12 Multi-Aspect Self-State Language Audit

Date: 2026-06-26

Communication constraints:

- sender input is still restricted to the learned self-state representation,
- message is three categorical slots with four symbols each,
- one shared receiver reconstructs state from message alone,
- no direct message-symbol labels are provided.

Primary targets:

- continuous four-need vector,
- low-need flags for each need,
- dominant need/request,
- severity,
- trend.

Gates:

- full-history messages beat latest-only and random-model controls on current
  state fields,
- dominant need codes align across senders under mild agreement pressure,
- trend is counted as passed only if it materially beats a majority trend
  baseline.

Recorded outcome:

- current-state communication passed: need MSE `0.005698`, low-flag accuracy
  `0.9538`, dominant accuracy `0.8327`, severity accuracy `0.8598`, sender
  code agreement `1.0000`, and dominant-code distinctness `1.0000`.
- trend did not pass: best trend accuracy was `0.6851` against a `0.667`
  majority baseline.

## M13 Trend-Balanced Temporal Self-State Audit

Date: 2026-06-26

Problem:

- natural trajectories are heavily skewed toward worsening states;
- the first trend metric barely beat a majority baseline;
- a temporal self-report should be tested on worsening, steady, and improving
  states separately.

Added constraints:

- balance train/eval samples across trend labels,
- sender input uses only learned self-estimates and learned estimate deltas,
- exact current or previous needs remain masked,
- compare full history, latest-only, and random-model controls.

Recorded outcome:

- on a 372-sample balanced eval set, trained full-history
  `self_estimate_delta` messages reached `0.8360` trend accuracy;
- latest-only control fell to `0.2883`;
- random full-history control reached `0.7742`, so the learned model improves
  the trend signal but environmental regularities still explain some of it.

## M14 Temporal Delta Intervention Audit

Date: 2026-06-26

Intervention:

- train the trend-balanced `self_estimate_delta` message protocol,
- evaluate the same balanced states after modifying only the learned delta half
  of the sender input,
- keep the current learned self-estimate fixed,
- compare original, zeroed, shuffled, and negated deltas.

Gate:

- trend accuracy should fall when the learned delta is removed or contradicted,
  while current-state reconstruction should degrade less.

Recorded outcome:

- trained trend accuracy fell from `0.8468` original to `0.6720` with shuffled
  deltas and `0.4825` with negated deltas;
- random trend accuracy also fell from `0.7419` to `0.3972` shuffled and
  `0.4449` negated, so random temporal features still carry trajectory signal;
- trained original remained better than random original on trend and all
  current-state metrics.

## M15 Counterfactual Action-Branch Trend Audit

Date: 2026-06-27

Intervention:

- collect multiple valid action branches from the same current state,
- label the actual next-body trend for each branch,
- sender input is the model's action-conditioned learned next-state estimate
  plus learned branch delta,
- train/evaluate on balanced worsening, steady, and improving branch outcomes.

Gate:

- trained branch messages should beat random branch messages on trend and
  current-state reconstruction,
- branch evaluation should remain balanced across trend classes.

Recorded outcome:

- branch eval set contained `1130/1130/1130` worsening/steady/improving
  samples;
- trained branch trend accuracy reached `0.7605` versus random `0.6726`;
- trained branch current-state reconstruction was much stronger than random:
  need MSE `0.007946` versus `0.022587`, dominant `0.8569` versus `0.7211`.
