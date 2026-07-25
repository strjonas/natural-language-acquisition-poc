# Persistent-mapping semantic-rent upper-bound preregistration

Date: 2026-07-25  
Status: locked before implementing or running the persistent-mapping audit

## Result selecting the redesign

The exact one-shot delayed task makes information irrational. Across 10,000
contexts, one truthful inspection reduced final minimum bodily need by 0.1147
relative to blind immediate consumption; two inspections reduced it by 0.2129
despite reaching 100% correct choices. The label costs ten primitive ticks and
the learned mapping is discarded after one consumption.

## Goal-level metric

This diagnostic asks whether persistent object-local semantic memory can pay
embodied rent when it is reused across recurring hunger and thirst in one
continuing life. It is an exact-dynamics substrate calibration, not a learned
agent result.

## Fixed audit

Evaluate 2,500 matched hidden mappings beginning at seed `1_900_000`.
For each mapping, retain one fixed food, one fixed water, and one fixed poison
surface across a life. Evaluate life lengths of 1, 2, 4, and 8 choice rounds.

At each round:

- the same three learner-visible surfaces occupy the same three positions;
- the low bodily need is 0.55 and alternates food/water from a seeded random
  starting need;
- the other needs start at 0.75;
- inspect, six-tick return, and consume use the exact existing dynamics;
- final utility is the minimum exact bodily need after consumption; and
- primitive-tick accounting restarts for the next round only as an audit
  convenience, while ticks and utilities are accumulated across the life.

No label or mapping is supplied except through the public grounded token packet.

Compare:

1. **Blind persistent:** consume the first visible surface every round; retain
   no labels.
2. **Persistent Bayes memory:** retain public surface-to-label observations
   across rounds. If a known surface matches the current need, consume it.
   Otherwise inspect unknown surfaces in learner-visible order until the
   matching label is heard. After two distinct labels, infer the third from the
   public one-food/one-water/one-danger task structure. Never read hidden kind
   for action selection.
3. **Clairvoyant ceiling:** consume the correct surface immediately each round,
   using hidden kind only for the unattainable ceiling.

Report by round count:

- mean final minimum need per round;
- paired cumulative and per-round utility gain over blind;
- correct/wrong-resource/poison rates;
- mean primitive ticks per round; and
- mean inspections per life.

## Decision rule

The persistent redesign is mechanically viable if, at either 4 or 8 rounds:

- persistent Bayes memory exceeds blind by at least 0.02 mean utility per
  round;
- its correct rate is at least 90%;
- it uses no more than two inspections per life on average; and
- the clairvoyant ceiling exceeds blind.

If viable, implement a multi-round semantic childhood in the environment with
an explicit non-agent round-transition observation so consumption consequences
remain visible to the learned world model before the next bodily demand. The
episodic bank must persist across rounds.

If no round count passes, abandon this delayed object-label substrate rather
than reducing inspection cost post hoc or adding language reward.
