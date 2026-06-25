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
