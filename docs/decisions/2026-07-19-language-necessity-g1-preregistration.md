# Language-necessity G1 preregistration

Date: 2026-07-19
Goal-level metric: in-loop language comprehension that improves adult survival.

## Conditions

Use the replicated full replay+horizon-2 organism configuration at seed 1.
Train separate silent and shuffled organisms for 200k actual ticks, with the
same childhood, replay, model, planner, and held-out 100 x 1,000 adult worlds as
the grounded run. Shuffled preserves token shapes/frequencies while breaking
situational meaning. Silent removes token content.

Before those training controls, evaluate the already trained grounded
checkpoint at inference under grounded, silent, and shuffled caregiver modes on
the same seeds. This is a token-reliance intervention, not a replacement for
matched training controls.

## Claim gate

Evidence for language comprehension requires grounded to exceed *both* silent
and shuffled training controls in survival and mean lifespan, or to exceed both
by at least 15% in mean lifespan with no survival regression. The grounded
checkpoint should also degrade when meaningful tokens are removed or shuffled
at inference. Otherwise report no language effect and redesign the information
rent; do not tune token loss or call the utterance predictor comprehension.

Self-model removal/reversal and counterfactual audits remain mandatory in all
conditions. This probe cannot establish generated language or self-report.

## Result

The language claim failed. Acute inference conditions on the grounded
checkpoint produced grounded/silent/shuffled survival of 3%/3%/4% and mean
lifespans of 133.75/134.23/130.29. Meaningful tokens were not load-bearing.

Separately trained grounded/silent/shuffled organisms produced survival of
3%/0%/2% and lifespans of 133.75/102.10/130.44. Grounded clearly exceeds
silent, but differs only marginally from shuffled, which preserves token
traffic while breaking semantics. The trained controls all learned bodily
prediction, so generic tokens/optimization and seed variance remain sufficient
explanations.

The current ecology leaks a nonlinguistic strategy: the `water` surface is
always water and most candidate food surfaces are safe. The consume option also
lets the policy skip the existing primitive ask/point interaction. The next
substrate must randomize consumable surface meanings per life and expose a
kind-blind inspect/ask option. Do not tune token-prediction weight; meaningful
answers must pay rent through safer subsequent bodily outcomes.
