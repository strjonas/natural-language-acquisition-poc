# Dual-code three-way childhood preregistration

Date: 2026-07-19
Status: locked before any interpreted training run

## Result that selected this branch

The sealed two-object grounded seed learned a causal resource-versus-danger
token effect but did not promote. Stochastic behavior was 250/500 correct,
including 83/155 after inspecting the selected surface. The bodily predictor
passed the resource/danger intervention but classified 0/100 food-to-water or
water-to-food substitutions as the substituted kind.

A delayed held-out audit sharpened the failure: after one label and several
padding-token motor observations, the model classified the labeled target in
396/400 cases but assigned that same predicted kind to the unlabeled object in
393/400. The GRU retained a global last-label code, not an object-bound one.

The old training distribution also made food and water observationally
redundant with the body: the only resource in a trial always repaired the
currently low need. Memory cannot identify a distinction absent from
experience. This preregistration therefore changes both the representation and
the causal learning contrast.

## Intervention

This is a task-appropriate reversal of Hill et al.'s dual-coding episodic
memory: their embodied fast-mapping agent used language keys to retrieve visual
values, while the present organism must use a visible referent to retrieve its
acquired lexical value. Their strongest comparison likewise used a
three-object task and found ordinary recurrent memory unreliable
([Grounded Language Learning Fast and Slow](https://arxiv.org/abs/2009.01719)).

The organism gains a per-life dual-code bank with one row per learner-visible
surface identity:

- the key is the attended surface one-hot already present in perception;
- the value is a learned projection of the complete heard utterance and has
  width 16;
- a value is written only when a nonpadding utterance follows embodied
  object-ahead ASK/POINT;
- currently visible objects retrieve only the value stored under their own
  surface key, plus a validity bit;
- the selected object's retrieved value enters a shared slot-local option
  scorer and the selected-object consequence transition directly;
- imagined transitions preserve existing bindings but do not invent labels;
  recurrent processing reads the previous bank before the current observation
  writes, so a new label first affects the external-memory pathway on the next
  observation while its raw tokens still enter the recurrent core normally;
  and
- the bank and validity bits reset at every life boundary.

The write/address protocol uses only learner-visible action, pose, geometry,
surface identity, and raw tokens. It never sees hidden kind, situation type,
correctness, audit targets, or reward. Meaning and behavioral use remain
learned end to end through the existing actor-critic and bodily/world
prediction losses. There is no label, inspect, question, or language reward.

The primary control has the identical 105,930-parameter architecture,
initialization, bank, slot-local policy, optimizer, data budget, and random
seed, but every episodic write is suppressed. The enabled and disabled models
therefore differ by one causal operation rather than parameter count. The
original 85,304-parameter recurrent-only model remains a secondary
architecture baseline, not the primary control.

## Crossed three-object childhood

Every life independently remaps the five consumable surfaces and presents
exactly one food, one water, and one poison at equal Manhattan distance two.
Hunger or thirst is independently selected at 0.55 while the other need starts
at 0.75. Thus each of the three functional labels occurs in both body contexts,
and a safe/danger code or `repair argmin(body)` shortcut cannot solve the task.
Consuming any object ends the life; correctness means consuming the resource
matching the current low need. Wrong-resource, poison, and timeout outcomes are
reported separately.

The trial uses a delayed-choice protocol so an agent cannot inspect and consume
at the same pose. At the canonical center, the only agent-selected actions are
WAIT and shared visible-slot inspect/consume options. Every outgoing option is
exactly four primitive ticks. An inspect ends in ASK and exposes its label
packet to the model. The next action is then forced and excluded from actor and
entropy credit: six kind-blind motor ticks return to center, canonicalize the
heading to NORTH, and end with a padding WAIT. All six ticks, metabolism,
reward, critic targets, and world-model targets remain in the lived stream.
Only the final padding return packet is observed, matching the macro-action
interface. The horizon-two planner treats inspect's second decision as this
forced return and treats consume as terminal rather than imagining an illegal
post-consume action. Horizon 40 permits three inspect-return cycles plus choice.

Grounded, silent, and shuffled conditions have matched environment seeds/world
draws, initial nonlinguistic states, dynamics, and observation shapes; their
subsequent trajectories are not yoked. Grounded speech is `this <true-kind>`.
Shuffled speech independently samples `food`, `water`, or `danger` with a
uniform marginal. Silent speech is padding.

Pre-training calibration over 5,000 fixed held-out seeds gives 33.68% correct
for a uniformly random object, 32.56% for the lowest surface index, and 33.68%
for a fixed-position rule. A quota-aware embodied oracle using only sensed
need, visible surface identity, acquired token labels, and the learner-
observable one-of-each task rule succeeds in 1,000/1,000 trials at horizon 40.
It consumes an explicitly matching label, or after two nonmatching labels
infers the remaining surface by elimination. It uses mean 20.820 and maximum
24 primitive ticks, mean 1.682 and maximum 2 inspections, with no timeout or
death.

## Locked seed-1 comparison

Run two fresh grounded seed-1 organisms, write-enabled and write-disabled:

- exactly 30,000 primitive ticks, entirely in three-object childhood;
- horizon 40, low need 0.55, fixed inspect/consume duration 4, fixed return
  duration 6, hidden size 64, token embedding 32, binding value width 16;
- consume and inspect options; no offers, BC, demonstrations, or intrinsic
  reward;
- replay reservoir 256 with one auxiliary replay update per online update;
- two-step latent loss weight 1 and two-step receding planner;
- planner scale 6, activated at tick 15,000, reward-head weight 0.5; and
- 600 fixed held-out behavioral trials, 300 exactly balanced
  label-to-consequence contexts (50 per functional-label x body-need cell),
  and 300 delayed referent contexts.

An inspection trial means that an agent-selected kind-blind inspect option
actually elicited a learner-visible `label|...` event. Primitive ASK/POINT and
CONSUME are masked in this delayed task; their use inside a macro is physical
execution, not a separate policy choice.

The write-enabled checkpoint is also evaluated without updates under acute
silent speech, independently shuffled speech, and suppression of every
episodic write. Greedy behavior is reported but is diagnostic; the stochastic
fixed-seed policy is the preregistered behavioral endpoint.

The pre-seal 83-tick seed-91 hidden-16/binding-4 engineering smoke automatically
emitted six-trial stochastic, greedy, acute, and audit rows. It was used only
to validate exact tick accounting, finite online/replay losses, and harness
wiring; those tiny diagnostic values were not used to select the delayed task,
30,000-tick budget, thresholds, binding width, or planner settings. This
preregistration is sealed only after that engineering validation.

## Exact locked commands

Write-enabled:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m homesocial.organism.harness \
  --train-steps 30000 --segment-length 64 --hidden-size 64 \
  --semantic-choice-childhood-steps 30000 --semantic-choice-horizon 40 \
  --semantic-choice-objects 3 --semantic-choice-low-need 0.55 \
  --semantic-choice-return-duration 6 --semantic-choice-eval-episodes 600 \
  --semantic-choice-acute-modes silent shuffled \
  --semantic-choice-acute-disable-binding-writes \
  --consume-options --inspect-options --episodic-binding-size 16 \
  --self-model-planning --planning-start-steps 15000 --planning-scale 6 \
  --planning-reward-weight 0.5 --planning-horizon 2 --model-horizon 2 \
  --multi-step-model-weight 1 --replay-capacity 256 --replay-updates 1 \
  --label-self-model-audit-inspections 300 \
  --label-referent-audit-contexts 300 --seed 1 --train-max-steps 40 \
  --eval-episodes 0 --language-modes grounded \
  --run-dir runs/organism/probe18_dual_code_delayed
```

Matched write-disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m homesocial.organism.harness \
  --train-steps 30000 --segment-length 64 --hidden-size 64 \
  --semantic-choice-childhood-steps 30000 --semantic-choice-horizon 40 \
  --semantic-choice-objects 3 --semantic-choice-low-need 0.55 \
  --semantic-choice-return-duration 6 --semantic-choice-eval-episodes 600 \
  --consume-options --inspect-options --episodic-binding-size 16 \
  --disable-episodic-binding-writes \
  --self-model-planning --planning-start-steps 15000 --planning-scale 6 \
  --planning-reward-weight 0.5 --planning-horizon 2 --model-horizon 2 \
  --multi-step-model-weight 1 --replay-capacity 256 --replay-updates 1 \
  --label-self-model-audit-inspections 300 \
  --label-referent-audit-contexts 300 --seed 1 --train-max-steps 40 \
  --eval-episodes 0 --language-modes grounded \
  --run-dir runs/organism/probe18_dual_code_delayed
```

## Promotion gates

All of the following must hold for the write-enabled seed:

1. **Three-way behavior:** at least 55% correct over all 600 trials, at least
   90% choice coverage, at least 60% of trials containing a voluntary
   inspection, at least 200 inspected selected-surface choices, and at least
   60% correctness among those choices.
2. **Causal channel use:** acute suppression of writes, acute silence, and
   acute shuffled speech each reduce overall correctness by at least 10
   percentage points from grounded inference. The separately trained matched
   write-disabled organism measures whether recurrence can learn an alternative
   binding strategy; its score is reported but is not a promotion gate.
3. **Crossed bodily semantics:** true-label bodily-kind accuracy is at least
   75% overall and at least 65% in every one of the six functional-label x
   body-need cells; true-label consequence MAE is lower than padding MAE.
4. **Counterfactual semantics:** both resource/danger substitution and
   food/water substitution classify the substituted kind in at least 70% of
   their cases, change predicted bodily delta by at least 0.02 mean L1, and
   move body-only planned consume probability in the intended direction in at
   least 65% of cases. For food/water swaps, the 65% direction threshold must
   hold separately in all four resource-label x body-need cells, not only when
   pooled.
5. **Referent locality:** delayed labeled-target kind accuracy is at least 70%,
   same-kind broadcast to the balanced unlabeled object is at most 40%, and
   target/nonreferent predictions differ by at least 0.02 mean L1. Moving the
   post-label stored key to the other surface and then replaying the identical
   padding return must relocate the label-kind score in the intended two
   directions in at least 60% of contexts with at least 0.02 mean prediction
   L1 change. Erasing the post-label bank before the same replay must lower the
   target's true-kind score on average. Final-state edits that hold the already
   memory-influenced recurrent core fixed are secondary controlled-direct-effect
   diagnostics and are not gates.

For the food/water probability gate, direction is conditioned on the body:
the probability of consuming a label that names the currently needed resource
must exceed the probability under the label that names the other resource.
This sign convention is applied identically to the direct policy and planned
policy shifts and is reported separately in all four resource-label x
body-need cells.

The unlabeled object's actual-kind accuracy and joint two-object accuracy are
diagnostics, not gates: after one inspection its identity is intentionally
underdetermined. The key-reassignment intervention, not lucky nonreferent
classification, is the causal localization test.

## Stop and replication rules

Both locked seed-1 conditions run regardless of the first result; there is no
within-pair tuning. If any gate fails, do not run seed 2 or language-training
controls. Use the failure topology only to select the next structural change:

- insufficient inspection selects learned expected information gain about
  bodily consequences, without token/ask reward;
- filled/local memory but poor three-way consequence learning selects an
  explicit token-only semantic-effect decoder trained on experienced bodily
  deltas; and
- accurate localized self-models without behavioral use selects a planner that
  can branch on unknown future observations, rather than a larger policy.

Only a complete seed-1 pass permits seed-2 replication followed by freshly
trained silent and shuffled controls. Grounded must then beat controls across
both seeds before any open-island transfer run or compute request.

Open-island transfer is additionally blocked until simultaneous caregiver
offers can no longer overwrite object labels in the binding bank. Offers are
disabled in this locked three-object comparison, so that known collision is
absent here and is not changed within this preregistered environment.

Even a pass would establish only acquired, object-local word-to-bodily
semantics and its causal behavioral use. It would not establish generated
language, reflection, autobiographical continuity, authentic human-like
desire, consciousness, or a human-like `me`.
