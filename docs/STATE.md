# STATE

Last rewritten: 2026-07-25. Rewrite this file, never append.

## Executive handover

The organism now has a **persistent lexical self-model that is behaviorally
sufficient**. Two words heard once each during ordinary embodied experience
determine the correct consumption choice in every one of six subsequent
recurring bodily demands within the same life, at 100.00% across 1,800 measured
rounds. Acute silence gives 34.00%; acute write suppression, with the language
still audible, gives 34.11%. The effect is carried by the persistent
object-local lexical bank, not by an utterance being audible at decision time.

Two things had to be fixed to see this, and both are now understood:

1. **The environment had to pay for knowing.** The one-shot task made
   information irrational: an exact-dynamics audit showed one truthful
   inspection *loses* 0.1147 bodily units against blind consumption. The
   eight-round persistent-mapping task pays +0.0813 per round instead.
2. **The planner had to ask its self-model the right question.** Every planner
   scored actions by the predicted *minimum* need. After a ten-tick detour that
   minimum is fixed by energy, which no consumption choice affects, and it
   rewards an uncertain smeared prediction over a correct concentrated one.
   Scoring the need the organism currently observes as most urgent moves
   post-label choice from 31.50% to 100.00% on identical states.

What is still missing is one operation: the organism does not **choose** to
acquire words. Its deployed inspection rate is 14% and flat across rounds.

## The current blocker, measured

Three planners have now failed to make inspection worth its cost, and the
diagnosis has converged on a cause that is not a planner. The corrected utility
reads whichever need the organism predicts will be most urgent, so it is only
as good as that prediction:

| Body state the urgent need is read from | Names the demanded resource |
|---|---:|
| The real current observation | 100.00% |
| The model's predicted post-inspect body | 60.67% |
| The model's predicted post-return body | 18.44% |

The last planner's failing gate came in at 18.00%. The organism's semantics are
intact; its ten-tick metabolic forecast is not. This matches the training
weights: next-observation prediction carries weight 0.1 while the bodily-change
loss is boosted 20-fold on consumption events, so slow metabolic drift over a
long detour is the least-supervised quantity in the model.

## Exact next step

Preregister and run a **world-model loss correction** that supervises slow
metabolic drift over multi-tick option rollouts, and take the same gate 6 as
its endpoint: in a hypothetical branch whose label names the resource the body
needs, the terminal choice must select the labeled object at least 60% of the
time. Success criterion at the mechanism level: the urgent-need index survives
a ten-tick forecast.

Do not implement a fourth planner. Do not adjust the settling count, the reuse
count, the planning scale, or the utility rule. If the forecast correction
fails, the remaining option is to let the organism learn the value of knowing
directly from experienced cross-round reuse under a longer training budget,
which is the first thing in this project that would justify a compute request.

## Sealed results that stand

- **Cross-round reuse (headline).**
  `docs/decisions/2026-07-25-protocol-branch-planner-result.md`;
  artifacts `runs/organism/probe28_cross_round_reuse/`.
- **Homeostatic utility correction.**
  `docs/decisions/2026-07-25-homeostatic-terminal-utility-result.md`. Real
  post-label choice 100.00%/0.00%/0.00% by case, against 31.50% under the old
  rule. Includes the one-factor bridge that localizes the branch failure to the
  decoder-reconstructed observation.
- **Information economics of the substrate.**
  `docs/decisions/2026-07-25-persistent-mapping-rent-result.md` and
  `-persistent-choice-mechanics-result.md`. The implemented eight-round
  environment gives a public-label policy +0.0813 per round over blind at 100%
  correct with exactly two inspections per life.
- **Delayed dual-code representation.** The 2026-07-19 sealed pair remains
  valid: crossed true-label bodily-kind accuracy 100% versus 32.33% for a
  matched write-disabled control.

## Negative results that are now closed

- One-shot delayed choice as a substrate for language acquisition: closed by
  exact upper bound, not by tuning.
- The mean-observation planner, the observation-branching planner, and the
  protocol branch: all three fail the same gate, for the reason above. Do not
  reopen without a fixed forecast.
- The minimum-need utility as a planning score after a delay: superseded, kept
  selectable and default-off so every sealed artifact reproduces.

## Trained-behavior status

The corrected persistent-childhood seed-1 pair (`probe24`) reaches 38.17%
held-out correctness write-enabled against 32.42% write-disabled and 32.62%
under acute write suppression, with 100% counterfactual word-to-bodily-kind
accuracy and 100% delayed target localization. It fails the behavioral gates of
`2026-07-25-persistent-childhood-learning-preregistration.md`, which remain the
standing bar. Seed 2, freshly trained silent and shuffled controls, open-island
transfer, generated speech, and any compute or data request stay blocked until a
fresh pair clears them.

## Architecture notes that matter

- An externally injected memory row reaches the recurrent core only through
  observation steps: 0 steps gives 18.67%, 3 gives 66.67%, 5 gives 81.33%. Real
  acquisition reaches 100% because the protocol supplies those steps.
- The world model's observation decoder has a mean L1 error of 13.55 on the
  visual/pose part. Any planner that decodes and re-encodes an observation
  inherits that error. `write_binding_into_state` exists so a counterfactual
  word can be considered without fabricating a scene.
- Continual learning must stay live. Replay carries are detached at segment
  boundaries, so long-life grounding will eventually need replay that includes
  the write history.

## Claim boundary and research ethic

Do not call this system conscious, sentient, human-like, authentically
desiring, reflective, or a real `me`. Its needs and objective are engineered,
and it does not generate language or report an internal state. The defensible
claim is now stronger than in July but still narrow: online embodied experience
produced a persistent, object-local, causally load-bearing word-to-own-body
memory that determines correct action for the remainder of a life, and matched
write and language ablations remove it.

No live LLM training/data calls, hidden-kind training targets, simulator
counterfactuals in learning, direct language rewards, or compute requests are
currently authorized or needed. Simulator branches stay audit-only; sealed
artifacts and decision records are preserved.
