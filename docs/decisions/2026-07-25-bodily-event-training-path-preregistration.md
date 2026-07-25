# Bodily-event training-path preregistration

Date: 2026-07-25
Status: locked before instrumentation is implemented and before either
instrumented retrain is run.

## Question

Probe36 put an independently normalized loss on lived bodily events, yet
probe37 found no calibration even on the real acquired-label path. Raw target
scarcity is not enough as an explanation: the two probe36 life logs each
contain about 6,100 consumptions. The missing measurement is whether those
events reach the loss with the consumed surface's acquired lexical binding
valid and read by the recurrent core.

This experiment changes no learning rule. It instruments the exact probe36
zero control and `bodily_event_loss_weight = 0.01` treatment during fresh,
matched 60,000-tick retrains.

## Locked runs

Both runs use seed 1, hidden size 64, 64-decision segments, three-object
40-tick semantic-choice childhood for the full 60,000 ticks, six-tick returns,
eight rounds per life, low need 0.55, consume and inspect options, 16-wide
episodic bindings, horizon-two model loss at weight 1, replay reservoir 256
with one update per segment, drift weight 0.01, split drift head, and gradient
clip 10.

1. Fresh control: event weight 0.
2. Fresh treatment: event weight 0.01.

The optimizer, data stream, action sampling, loss terms, architecture, and
budget are otherwise unchanged. The rejected event weight is neither
increased nor swept.

## Instrumentation

For every online segment before its update, record:

- every target entry above the sealed 0.175 event threshold, split into food
  restoration, water restoration, poison, and other bodily events;
- the selected surface, hidden outcome kind from simulator metadata used only
  by this audit, start-body bin, and whether the selected surface's binding was
  already valid at the action state;
- whether that binding was therefore included in the visible-slot read into
  the recurrent core, and the norm of its lexical transition feature;
- one-step pre-update absolute delta error, split by event kind and binding
  validity, in 10,000-tick windows.

At fixed every-100th-update samples that contain a bound resource event,
measure the raw L2 gradient norm of:

- the original change-boosted bodily loss;
- the added, already-weighted event term by itself; and
- each signal restricted to the lexical path
  (`token_embedding`, `binding_value`, and `binding_read`).

At the end, score every segment retained in the replay reservoir with the final
model and report the same event and valid-binding composition and error. This
compares the online and replay distributions without adding data or updating
parameters.

The audit metadata is forbidden from entering the model, policy, loss, replay
sampling, or planner.

## Locked interpretation

The following diagnoses are not mutually exclusive; report all that apply.

- **Target scarcity:** fewer than 500 food-plus-water restoration transitions
  occur in a run.
- **Semantic-conditioning scarcity:** fewer than 25% of resource restorations
  have a valid selected-surface binding at the action state.
- **Replay dilution:** the valid-bound resource fraction in the final replay
  reservoir is less than half its online fraction.
- **Optimization starvation:** in the treatment, the median already-weighted
  event-gradient norm is below 5% of the original bodily-loss gradient norm,
  either globally or on the lexical path.
- **Training-distribution fit / held-out failure:** the treatment's final
  replay MAE on valid-bound resource entries is below 0.05 while probe37's
  real-immediate terminal MAE remains above 0.05.
- **Training-distribution underfit:** final replay MAE on valid-bound resource
  entries is at least 0.05.

Food and water are reported separately even when the combined gate decides the
classification. Counts, unique surface-kind-body-bin contexts, raw norms, and
all windowed errors are retained, not only threshold decisions.

## Stop rule

This is an instrumentation experiment, not a license to tune. If conditioning
is scarce, the next intervention must alter sampling or credit assignment
specifically for bound events. If gradients are starved despite adequate
conditioning, the next intervention must first restore the declared event
term's effective scale without changing its target distribution. If the
training distribution is fit, the next intervention must address
generalization. If it is underfit with adequate gradients and coverage, inspect
conflicting contexts and representation before changing a weight.

No larger compute or generated data is justified: both exact runs are local and
complete in under a minute each.

## Claim boundary

This can identify how a grounded lexical memory does or does not reach a bodily
event model. It is not reflection, self-report, generated language,
consciousness, or a claim about subjective experience.
