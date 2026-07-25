# STATE

Last rewritten: 2026-07-25. Rewrite this file, never append.

## Executive handover

The organism's bodily forecast is repaired, and with it the gate that had
blocked this project for three successive planners. On one checkpoint, at once:

- **A branch whose label names the demanded resource selects that object
  99.78% of the time** (gate 6; the sealed baseline is 18.00%, and the
  mean-observation, observation-branching and protocol-branch planners failed it
  at 19.11%, 18.00% and 18.00%).
- **Acquiring a word is now worth its cost**: the mean intact inspect advantage
  is positive (+0.0205) for the first time. That was the one operation the last
  handover named as missing.
- **The persistent lexical memory is untouched at ceiling**: 100.00%
  cross-round reuse in every one of rounds 3-8, against 30.94% under acute
  silence and 31.17% under acute write suppression.
- **The ten-tick forecast preserves the organism's own need ordering** 100.00%
  of the time, against 18.00% before.

Checkpoint: `runs/organism/probe33_online_budget/`.

What blocks promotion is now a different thing than before: four of the seven
feasibility gates still fail, three of them narrowly. See "The current
blocker".

## How the forecast was repaired

The last handover localized the failure to a systematic per-need metabolic
drift bias. That was correct, and the fix had two parts.

1. **The loss ignored slow metabolism.** Drift entries are 82.7% of all entries
   but carry about one percent of the delta head's gradient, because they are
   both smaller and less heavily weighted than consumption events. What the
   head learned instead was the consumption confound: predicted delta
   correlated -0.923 with the *current* food need, because consumption happens
   when a need is low, so the fitted line extrapolated to a large spurious
   decay whenever a need was high. A scale-balanced drift term, stratified per
   need and normalized by the world's per-tick metabolic scale, fixes it.
2. **The organism needed twice as much life.** At 30,000 ticks the dense
   metabolic objective displaces the sparse lexical one and reuse falls to
   63.33%. At 60,000 it does not, and both sit at 100.00%.

The decisive measurement that licensed all of this was the **oracle
substitution** in `audit_metabolic_drift_forecast`: replacing only the
predicted drift with the simulator's true drift took the planner's own chain
from 18.00% to 100.00%, proving forecast error was the whole of the failure
rather than one contributor.

## Three refuted explanations, kept refuted

Before the budget was tried, three cheaper causes were tested and each was
killed by its own control. None of these should be reopened.

| Candidate | Test | Memory | Verdict |
|---|---|---:|---|
| The shared output head | split drift/event head | 57.83% | not the cause; the architecture-only control holds 100.00% |
| Capacity | hidden 128 / 256 | 50.28% / 54.39% | not the cause |
| The gradient budget | `max_grad_norm` 1.0 -> 10.0 | 62.56% | not the cause; the raised-clip control holds 98.89% |

The gradient measurement is the one to remember as a caution. The drift term
genuinely inflates the raw gradient sixfold and genuinely saturates the trust
region on 100% of updates, which explained every observation including why head
separation and capacity both failed. It was still not the cause. **A mechanism
that explains every observation is not thereby the cause.**

The capacity-based compute request that the split-head stop rule pointed at was
withdrawn as refuted by its own diagnostic.

## The current blocker, measured

| Gate | Requirement | Sealed baseline | Now | Decision |
|---|---:|---:|---:|---|
| 1 Mean intact inspect advantage | > 0 | +0.1735 | +0.0205 | Pass |
| 2 Positive-advantage contexts | >= 75% | 99.00% | 46.33% | **Fail** |
| 3 Write-suppression advantage drop | >= 0.05 | 0.2371 | 0.0466 | **Fail** |
| 4 Collapsed-label advantage drop | >= 0.05 | 0.2371 | 0.0466 | **Fail** |
| 5 Label-contingent terminal choices | >= 60% | 99.89% | 85.11% | Pass |
| 6 Matching label selects the target | >= 60% | 18.00% | **99.78%** | Pass |
| 7 Danger label avoids the target | >= 90% | 100.00% | 85.33% | **Fail** |

Branch contingency is fully write-caused: 85.11% intact against exactly 0.00%
under both write suppression and collapsed labels.

## Exact next step

Gate 2 carries a specific, diagnosable anomaly and should be taken first,
because it is the widest failure and the other three are near misses that may
move with it.

**The positive-advantage context rate is identical at 46.33% across intact,
write-suppressed and collapsed conditions**, while the mean advantages differ
(+0.0205 against -0.0262). Whether a context has positive inspect advantage is
therefore decided by something the write does not touch, even though the *size*
of the advantage is entirely write-caused. Find that context-level factor
before changing anything: it is a property of the branch construction, not of
the forecast, and it is measurable read-only on the existing checkpoint.

A known, unfixed defect of that construction is the likeliest suspect and is
already documented: the planner asks the model what happens if it waits from
the *choice pose* and expects the answer for the six-tick return, but at the
choice pose a wait is one tick. The query is structurally aliased, and no loss
over lived transitions can resolve it, because the two situations carry the
same action label. The repaired model answers something between the one-tick
and six-tick quantity (predicted food -0.0188 against a realized -0.0600); the
need *ordering* survives because the shortfall is proportional across needs,
which is why gate 6 passes anyway. The advantage *magnitude* has no such
protection, and gates 2, 3 and 4 are all magnitude gates.

Do not add a fifth planner. Do not reopen the drift weight grid, the 0.175
regime bound, the 0.02 scale, the gradient clip, or the split head. Do not
sweep the budget further; 120,000 ticks was no better than 60,000 on any
measured quantity.

## Sealed results that stand

- **Forecast repair and the budget resolution (headline).**
  `docs/decisions/2026-07-25-forecast-memory-tension-result.md`; artifacts in
  `runs/organism/probe31_split_drift_head/`, `probe32_gradient_budget/`,
  `probe33_online_budget/`.
- **The drift-supervision tension.**
  `docs/decisions/2026-07-25-metabolic-drift-supervision-result.md`. The clean
  monotone dose-response (100.00% / 63.33% / 46.11% / 9.89%) with three
  `drift_weight = 0` retrains all at 100.00% as the control.
- **Cross-round reuse.**
  `docs/decisions/2026-07-25-protocol-branch-planner-result.md`.
- **Homeostatic utility correction.**
  `docs/decisions/2026-07-25-homeostatic-terminal-utility-result.md`.
- **Information economics of the substrate.**
  `docs/decisions/2026-07-25-persistent-mapping-rent-result.md` and
  `-persistent-choice-mechanics-result.md`.
- **Delayed dual-code representation.** The 2026-07-19 sealed pair.

## Negative results that are now closed

- One-shot delayed choice as a substrate for language acquisition: closed by
  exact upper bound.
- The minimum-need utility as a planning score after a delay: superseded, kept
  selectable and default-off.
- The split drift/event head, a fourfold capacity increase, and the gradient
  trust region: all three refuted as causes of the forecast/memory tension,
  each by its own control.
- The claim that the forecast is unfixable, or that fixing it cannot rescue
  gate 6: refuted twice, by the oracle bound and then by a trained model.

## Architecture notes that matter

- Training is **not bit-reproducible at a fixed seed**; evaluation on a fixed
  checkpoint is. Every effect must be read against a retraining noise band, so
  controls are retrained rather than compared to a single sealed number. A unit
  test, not a checkpoint hash, guards the zero-weight reproduction path.
- An externally injected memory row reaches the recurrent core only through
  observation steps: 0 steps gives 18.67%, 3 gives 66.67%, 5 gives 81.33%.
- The observation decoder has a mean L1 error of 13.55 on the visual/pose part.
  Any planner that decodes and re-encodes an observation inherits it;
  `write_binding_into_state` exists to avoid fabricating a scene.
- Continual learning must stay live. Replay carries are detached at segment
  boundaries.

## Claim boundary and research ethic

Do not call this system conscious, sentient, human-like, authentically
desiring, reflective, or a real `me`. Its needs and objective are engineered,
and it does not generate language or report an internal state.

The defensible claim is now stronger than the last handover's and is still
narrow: online embodied experience produced a persistent, object-local,
causally load-bearing word-to-own-body memory that determines correct action
for the remainder of a life; matched write and language ablations remove it;
and the organism's own model of its body is now accurate enough that a
hypothetical word almost always selects the right object and that acquiring one
is, on average, worth its cost. It is one seed, with no write-disabled training
pair yet.

Seed 2, freshly trained silent and shuffled controls, the write-disabled
training pair, open-island transfer and generated speech all remain blocked
until the feasibility battery passes in full.

No live LLM training/data calls, hidden-kind training targets, simulator
counterfactuals in learning, or direct language rewards are authorized or
needed. **No compute request is justified**: the working configuration trains
locally in under a minute, and the one compute request this project came close
to making was withdrawn this session as refuted by its own diagnostic.
