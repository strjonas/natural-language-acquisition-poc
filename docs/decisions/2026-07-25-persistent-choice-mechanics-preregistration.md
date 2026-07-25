# Implemented persistent-choice mechanics preregistration

Date: 2026-07-25  
Status: locked after implementation unit tests and before the first aggregate
mechanics audit

## Purpose

Verify that the implemented multi-round environment realizes the positive
semantic-rent substrate measured by the prior audit, without leaking hidden
kind or erasing consumption consequences.

## Fixed audit

Run 2,500 matched eight-round lives beginning at seed `2_000_000` through the
actual public `IslandWorld` and `execute_agent_action` interfaces.

Configuration:

- three fixed surfaces with one food, one water, and one poison;
- low need 0.55, alternating food/water;
- six-tick fixed return after inspection;
- one forced zero-credit round-transition decision after consumption;
- persistent surface-kind mapping, recurrent state opportunity, and lexical
  memory opportunity across all eight rounds.

Compare:

1. blind first-surface consumption;
2. a public-label persistent policy that reads only packet tokens and visible
   surface identities, inspects unknown surfaces until it can identify the
   needed kind, and retains those labels; and
3. a hidden-kind clairvoyant ceiling used only for audit validation.

Report final minimum need per round, correct/wrong/poison rates, ticks per
round, inspections per life, and paired utility gain over blind.

## Gates

- all policies complete exactly eight rounds per life;
- blind correct rate remains within [30%, 37%];
- public-label persistent policy is at least 99% correct;
- it averages no more than two inspections per life;
- it exceeds blind by at least 0.07 final-min-need units per round; and
- the clairvoyant ceiling exceeds the public-label policy.

Additionally, unit tests must show that:

- the post-consumption packet is emitted before the next demand reset;
- the round transition is forced and excluded from policy credit;
- mapping and surface identities persist;
- demand alternates; and
- the final round terminates normally.

Failure blocks learned training and requires repairing the environment, not
tuning the agent.
