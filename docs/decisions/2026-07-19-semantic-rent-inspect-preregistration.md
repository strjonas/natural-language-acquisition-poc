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

## Result

The structural hypothesis passed, but the full G2 claim did not.

The ecology/mechanics gates passed before learning: oracle/random survival was
99%/0% over 100 x 1,000 ticks; all 188 tests passed; and a 1,024-tick mechanics
run used inspect, received labels, and retained finite losses and audits.

Two independent grounded 40k runs passed every structural promotion rule:

| seed | survival | lifespan | inspect/life | labels/life | body-only normal/reversed life | body score corr. | best/worst advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6% | 97.19 | 14.72 | 14.19 | 86.46 / 59.51 | 0.200 | 0.029 |
| 2 | 5% | 98.15 | 2.67 | 3.68 | 103.25 / 66.13 | 0.211 | 0.039 |

The matched training controls show a real but noisy grounded advantage:

| seed | grounded survival/life | silent survival/life | shuffled survival/life |
|---|---:|---:|---:|
| 1 | 6% / 97.19 | 3% / 81.70 | 1% / 87.19 |
| 2 | 5% / 98.15 | 5% / 81.29 | 3% / 89.38 |

Grounded therefore exceeds both controls on seed 1. On seed 2 it lives 20.7%
longer than silent and survives more often than shuffled, but it neither
strictly exceeds silent survival nor beats shuffled lifespan by the locked 15%
margin. The aggregate replication gate is not met. Acute inference is also
mixed: seed 1 grounded/silent/shuffled survival is 6%/6%/3% with lifespans
97.19/98.72/94.16; seed 2 is 5%/4%/7% and 98.15/85.17/94.79. Meaningful tokens
are not consistently load-bearing at inference.

The matched label intervention found a narrower positive mechanism. On the two
grounded models, changing only functional-kind tokens moved subsequent consume
probability in the substituted semantic direction by +0.0116 and +0.0136,
with directional hit rates 87.0% and 77.5%. Seed-1 silent/shuffled controls had
near-zero directed shifts and 36%/35% hit rates. However, the seed-2 shuffled
model also produced a +0.0114 shift and 83% hit rate despite never receiving
grounded meanings. Arbitrary token embeddings can therefore satisfy this shift
metric. The stricter correctness measures do not replicate: true labels improve
kind classification over padding only slightly (17.0% vs 13.5%; 31.0% vs
27.0%), stay at or below chance, and improve actual-consequence MAE on seed 1
but worsen it on seed 2.

Verdict: per-life information rent and inspect actions improve the developmental
substrate, and the organism is token-sensitive in its own bodily model, but it
does not yet reliably bind the *correct* heard meaning to the contemplated
bodily outcome. Do not call this comprehension. Labels are voluntarily sampled
but their consequence association is too sparse/noisy in the open island. The
next probe is the preregistered paired inspect--remember--choose childhood,
not token-loss tuning or larger models.

## Post-result audit correction

A subsequent code-and-data audit invalidated the narrower causal interpretation
of the label intervention. The 200 nominal rows per checkpoint came from only
1--9 policy-driven episodes and 3--14 unique object targets, with radically
different resource/danger mixtures. Some zero-distance options also labeled or
consumed the object ahead instead of the selected object; shuffled seed 2 had 28
wrong-referent rows. The consume-probability shift was mostly a direct
token-to-policy effect: removing planning retained 0.00835/0.01156 of seed 1's
reported shift, 0.01088/0.01357 for grounded seed 2, and 0.00952/0.01143 for
shuffled seed 2. The planning-specific increment had only 52--58% directional
hits. Finally, grid generation and shuffled speech used separately constructed
PRNGs but the same seed, leaving a small measurable hidden-kind association.

Therefore withdraw the statement that these runs establish a positive
label-to-self-model mechanism. They establish only (a) replicated use of an
action-conditioned bodily model, (b) voluntary inspection, and (c) a
descriptive grounded-training lifespan advantage that failed its locked
replication gate. The headline negative conclusion is unchanged. The next
probe splits the RNG streams and uses balanced, unique, fixed-context causal
audits with direct-policy and planning-mediated effects reported separately.
