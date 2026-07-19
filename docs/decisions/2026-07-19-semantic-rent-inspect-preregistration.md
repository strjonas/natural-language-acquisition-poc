# Per-life semantic rent and inspect-option preregistration

Date: 2026-07-19
Goal-level metric: G2 in-loop grounded comprehension that changes the
organism's own predicted bodily consequences and improves unsupported life.

## Diagnosis fixed before the probe

The prior grounded/silent/shuffled comparison is correctly negative, but its
substrate makes that outcome uninformative about learnable word meaning:

- A position-aware policy that never reads hidden kinds or tokens can treat
  the stable `water` surface as water, treat the three food-candidate surfaces
  according to their 0.75 food prior, and learn the rare poison by bodily
  outcome. It survives 98/100 held-out lives versus the full-kind oracle's
  99/100. Language therefore has essentially no ecological information rent.
- The shuffled caregiver is permuted once for an entire training run, making
  it a stable substitution cipher, but evaluation constructs a new permutation
  each life. This train/evaluation mismatch is removed. Shuffled label forms
  will be sampled independently of the true label on every label event, from
  the same speech-act family, in both training and evaluation.
- The published planning intervention combines predicted bodily need and
  predicted external reward. A post-hoc body-only re-audit with reward weight
  zero retains the main direction on both saved checkpoints. Seed 1
  normal/removed/reversed lifespan is 123.43/118.25/78.30 ticks and seed 2 is
  96.53/84.42/64.40. Body-only within-state correlations are 0.494 and 0.642,
  with predicted-best actual advantages 0.079 and 0.054. New causal claims
  will report body-only and reward-inclusive planning separately.

## Structural hypothesis

Semantic tokens can be acquired in-loop when they supply otherwise unavailable
information about the consequences of contemplated actions for this body.
The same recurrent state must bind a jointly attended surface to a heard kind,
retain that binding within the life, use it to predict the body's outcome of
consuming the surface, and choose accordingly.

This is information rent, not a token reward. Inspecting, asking, correct token
prediction, and speaking receive no direct reward.

## Substrate change

1. The five consumable surfaces (`water`, `spring`, `berry`, `roots`, and
   `mushroom`) are assigned exactly two food kinds, two water kinds, and one
   poison kind by a seeded uniform shuffle at every life reset. Thus every
   surface has the same 0.4/0.4/0.2 marginal bodily meaning. Multiple instances
   of a surface share its within-life meaning. Shelter, contact danger, and
   blocking landmarks remain stable.
2. The creole bank covers food, water, and danger labels for every consumable
   surface. Caregiver label utterances always reveal the grounded kind in the
   grounded condition.
3. Alongside each visible-slot consume option, the organism gains a visible-slot
   inspect option. Its kind-blind servo may read only the selected perceived
   displacement and proprioceptive orientation; it approaches/faces the object
   and executes `ASK`. It never reads kind, bodily effect, caregiver situation,
   or simulator outcomes.
4. Consume and inspect are distinct action identities in the learned transition
   model and share object-binding features. The GRU state persists across the
   inspect-to-consume delay within a life and resets at death. No hand-written
   semantic memory is supplied in this first probe.
5. Shuffled label events draw a seeded random label variant from the label
   speech-act family independently of the true situation on each event. This
   preserves label-token traffic and marginal vocabulary while eliminating a
   stable cipher. Silence remains all padding.

## Required mechanics and ecology gates

- Across a large deterministic seed sample, every consumable surface must have
  identical empirical kind marginals up to sampling error and every life must
  contain the exact 2-food/2-water/1-poison quota.
- Oracle survival must remain at least 95% and random survival at most 10% over
  100 held-out 1,000-tick lives.
- Unit tests must prove that the inspect servo reaches the selected learner-
  visible object, emits a label without consuming it, exposes no hidden kind to
  the option controller, and uses semi-Markov duration accounting.
- Grounded, silent, and shuffled modes must have identical observation and
  action shapes.

## First learning probe

Run the integrated replay plus horizon-two organism at seed 1:

- hidden 64, 40k actual ticks, 1,000-tick training lives;
- caregiver offer threshold 0.75 and distance 0 -> 2, both ending at 20k;
- consume and inspect visible-slot options;
- two-step latent loss weight 1, replay reservoir 256 with one auxiliary
  update per online update;
- planning scale 6 beginning at 20k, with body-only and reward-inclusive
  diagnostics reported separately;
- 100 held-out unsupported adult lives of 400 ticks and 200 audited states.

Exact command is recorded with the result. This first run is a structural
smoke, not a language comparison.

Promotion requires all of:

- inspect options are used after support ends and lead to nontrivial label
  exposures;
- food and water are both consumed nontrivially;
- normal bodily planning beats reversed bodily planning in lifespan without a
  survival regression;
- within-state bodily score correlation and predicted-best actual advantage
  are positive after training; and
- the label-to-self-model audit below contains enough held-out inspected
  states to be meaningful.

If inspect is not selected, the next architectural probe is uncertainty-aware
information value for inspect (expected reduction in bodily-consequence
uncertainty), not an ask/token reward. If labels are selected but not retained,
the next probe adds a learned dual-code episodic binding memory. Do not tune the
token-prediction weight.

## Decisive label-to-self-model audit

At held-out states, execute a kind-blind inspect on a consumable surface and
fork only the recurrent update that receives the resulting observation:

1. true grounded label tokens;
2. padding tokens;
3. a matched counterfactual label with the kind token changed while vision,
   body, action history, recurrent pre-state, and attended surface are fixed.

For the subsequent consume option, measure predicted bodily delta, bodily
score, and action probability. Also branch an audit-only simulator copy to
record the actual consequence; simulator outcomes never enter training.

Comprehension evidence requires that true tokens improve held-out consequence
classification over padding, and that counterfactual kind substitution moves
the predicted bodily consequence and consume preference in the substituted
semantic direction. Aggregate survival alone is insufficient.

## Full comparison gate

Only after the structural smoke passes, replicate grounded at seed 2 and train
matched silent and shuffled controls. A G2 claim then requires:

- grounded exceeds both controls in survival, or by at least 15% in lifespan
  with no survival regression, across the fixed held-out worlds;
- acute silence and per-event shuffled labels degrade the grounded checkpoint;
- true/counterfactual label interventions causally alter its bodily predictions
  and choices in the expected direction; and
- the effect replicates across training seeds and held-out surface assignments.

This probe cannot establish generated language, self-report, reflection,
authentic desire, autobiographical identity, consciousness, or a human-like
`me`.
