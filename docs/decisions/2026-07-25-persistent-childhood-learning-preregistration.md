# Persistent-childhood seed-1 learning preregistration

Date: 2026-07-25  
Status: locked before any learned multi-round training run

## Goal-level metric

Test whether one continually trained embodied organism will acquire truthful
object labels because one persistent lexical memory improves repeated bodily
choices later in the same life.

This is the first behavioral test in the repository where the exact
environment makes delayed semantic acquisition rational. It targets
comprehension and autobiographical continuity, not generated self-report or
consciousness.

## Result permitting training

In the implemented eight-round environment, an exact public-label policy gains
0.0813 final-min-need units per round over blind action, reaches 100% correct
choices, and needs exactly two inspections per life. All mechanics gates and
231 local tests pass before training.

## Locked pair

Train exactly two seed-1 conditions for 30,000 primitive ticks:

1. write-enabled object-local lexical bank;
2. identical architecture and initialization with every lexical write
   suppressed.

Shared configuration:

- grounded language;
- hidden size 64, token embedding 32, binding width 16;
- eight rounds per life, one persistent food/water/poison mapping;
- low need 0.55, horizon 40 per round, six-tick post-label return;
- four-tick kind-blind inspect and consume options;
- segment length 64;
- two-step world-model loss, weight 1.0;
- replay capacity 256, one replay update per live update;
- default optimizer, losses, entropy, and reward shaping;
- no BC warmstart;
- no self-model planner; and
- exact 30,000 primitive-tick budget.

The self-model planner is intentionally disabled. The sealed mean-observation
planner cannot value information, and the new observation-branching planner
correctly assigns negative value within a single round because it does not yet
back up reuse across later rounds. Adding either to this pair would suppress
the very cross-round actor-critic credit being tested.

Literal write-enabled command:

```bash
.venv/bin/python -m homesocial.organism.harness \
  --train-steps 30000 \
  --segment-length 64 \
  --hidden-size 64 \
  --semantic-choice-childhood-steps 30000 \
  --semantic-choice-horizon 40 \
  --semantic-choice-objects 3 \
  --semantic-choice-low-need 0.55 \
  --semantic-choice-rounds 8 \
  --semantic-choice-return-duration 6 \
  --consume-options \
  --inspect-options \
  --episodic-binding-size 16 \
  --planning-horizon 2 \
  --multi-step-model-weight 1 \
  --replay-capacity 256 \
  --replay-updates 1 \
  --semantic-choice-eval-episodes 300 \
  --semantic-choice-acute-modes silent shuffled \
  --semantic-choice-acute-disable-binding-writes \
  --label-self-model-audit-inspections 300 \
  --label-referent-audit-contexts 300 \
  --eval-episodes 0 \
  --baselines \
  --language-modes grounded \
  --seed 1 \
  --train-max-steps 40 \
  --run-dir runs/organism/probe23_persistent_childhood
```

The write-disabled command is identical plus:

```text
--disable-episodic-binding-writes
```

Run write-enabled first, then write-disabled. Do not change gates or parameters
between conditions.

## Evaluation

Use 300 fixed held-out lives beginning at the harness evaluation seed, giving
2,400 choice rounds per condition. Report aggregate and per-round choice,
correctness, and inspection rates.

For the write-enabled checkpoint also run acute silent, independently shuffled,
and write-suppressed evaluations on identical held-out lives. Run the existing
crossed label-to-bodily-model and delayed referent-locality audits.

## Promotion gates

The write-enabled seed passes only if all hold:

1. aggregate correct rate is at least 60%;
2. each of rounds 3-8 is at least 65% correct;
3. mean inspection rate in rounds 1-2 exceeds mean inspection rate in rounds
   5-8 by at least 20 percentage points, demonstrating acquisition then reuse;
4. acute write suppression reduces aggregate correctness by at least 15
   percentage points;
5. both acute silence and acute shuffle reduce aggregate correctness by at
   least 15 points;
6. the write-disabled trained control is no higher than 45% correct;
7. the write-enabled model exceeds the write-disabled control by at least 15
   points;
8. true-label bodily-kind accuracy remains at least 75%; and
9. delayed labeled-target accuracy remains at least 70%, same-kind broadcast
   no more than 40%, and key-reassignment directionality at least 60%.

Inspection is not directly rewarded and labels, kinds, correct actions, and
simulator counterfactuals never enter learning.

## Stop rule

If any gate fails, stop before seed 2 and before fresh language-control
training. Diagnose whether the failure is cross-round credit, memory use,
representation, or evaluation coverage. Do not tune on seed 1 past one
mechanism-level follow-up selected by that failure topology.

No GPU-cloud, API, or data-generation request is authorized by this
preregistration.
