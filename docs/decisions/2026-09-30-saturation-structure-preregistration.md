# Probe76: hidden restoration groups from saturation interactions

Date: 2026-09-30. Preregistration, written before implementing or generating
the new survey. Probe75 has completed and independently passed its six original
gates. That confirms persistent rate memory, but does not discover its variables.
Probe61's latent fitting/pruning mechanism remains closed. This experiment asks
a different identification question using interventions rather than latent
dimension penalties. Probe71's composition/retraining route remains closed.

## Question and explicit instrument privileges

Can scalar bodily sensations identify which unnamed restorative actuators
target the same hidden bounded resource, and how many *reachable* resources
exist, without supplying axes, names, dimensions, portions or a target map to
inference?

Use the existing five-slot ExpandedBody physical family, unchanged birth levels
(0.35 through 0.85), per-axis metabolic spread 0.60, resting rates, and small
0.20 / large 0.60 portions. Every world exposes the same ten opaque actuator
IDs. Independently shuffle their physical assignments per block/world. In K1
through K5 select K active slots randomly. Other slots stay at 1, have zero
metabolism and inert grants. The scalar mean always divides by **five**, so
reciprocal mean scaling cannot supply K. Minimum and mean alone are exposed;
schema, true state, rates, portions and target identities are scorer-only.

This is an **identification instrument, not an online organism treatment**.
It deliberately allows matched forks from the same initial state, and disables
shocks. Both privileges must be reported as limitations. It retains six-tick
grant opportunities and next-tick absorption. Each branch costs thirteen
physical resting ticks; the learner never gets an instantaneous post-grant
reading. With shocks absent the smallest birth 0.35 exceeds thirteen times
the largest resting depletion 0.0256, so no branch needs a post-death reading.
The viability assertion is checked in the runner.

## Interaction and inference fixed in advance

For each context run the common-control branch 00, ten A0 branches, ten 0B
branches and all hundred AB branches. A is requested at tick 6 and absorbed
at tick 7; B is requested at tick 12 and absorbed at tick 13. Branches share
birth and metabolic draws. Read scalar sensations after tick 13.

`C[a,b] = mean(AB) - mean(A0) - mean(0B) + mean(00)`.

An actuator is active if any single-action increase is above 1e-8. Connect
active IDs when **either orientation** has C below -1e-8 in any developmental
context. Count connected components; do not assume two aliases per group,
known sizes, a fixed count, an axis order, or mean/min inverse formulas.
The threshold is a numerical tolerance fixed before results, not a statistical
threshold fitted to a body. Different independent axes have zero interaction.
Within an axis, clipping can make two restorations compete. Small/small
interactions need not be detectable; varying contexts and large/small overlap
can connect their aliases. Select each group's representative by the largest
mean single-action effect in development, using only transcripts.

## Sample, seeds and held-out evidence

Five independent paired seed blocks × K1–K5, each with forty developmental
contexts and forty **disjoint** held-out contexts. No model, parent checkpoint,
corpus, motor retraining or global configuration change is introduced. Freeze
opaque mappings and active subsets within each fitted world; vary birth and
individual rates across contexts. Contexts are evidence about species-level
causal grouping, not forty independent learner replications.

Use experiment seed base 3,800,000,000 and block stride 2,000,000, world
stride 100,000. Within a world use separate offsets for mapping (0), development
(1,000), held-out (10,000), permutation control (20,000), shuffled pairing
(30,000), remapping (40,000), and unreachable-axis challenge (50,000).
Each context uses its offset plus its index. All draws use local generators;
no developmental, Probe74 or Probe75 episode stream is reused. A separately
seeded two-context timing diagnostic at 3,850,000,000 is excluded from gates.
Save progress atomically after each complete block/world. Pin source and
configuration; reject partial resume under changed code. Report per-world
counts/partitions, actual branch and tick budgets, and elapsed time.

## Locked continuation gates

All six clauses must pass. No count, tolerance, context budget, control or
threshold is revised after generating a survey context.

1. **G1 count:** recover exactly K reachable components on at least four of
   five blocks **for every K**, with strictly increasing mean recovered count.
2. **G2 partition:** recover exactly the active actuator set and its physical
   target-equivalence partition on at least four of five blocks for every K.
3. **G3 held-out prediction:** predict which actuator/representative pairs can
   occlude in forty held-out contexts from group membership alone. Score whether
   any held-out interaction is below -1e-8. Require accuracy at least 0.95 and
   recall at least 0.90 for both occluding and non-occluding pairs on at least
   four of five blocks for every K. Include inactive actuators as non-occluding;
   report confusion counts. Held-out data never enters grouping or representative
   selection. This predicts causal overlap, not a next-state vector.
4. **G4 ID permutation:** independently relabel the complete development and
   test transcripts. Counts must stay identical and inferred partitions must
   transport exactly on all twenty-five worlds.
5. **G5 pairing lesion:** shuffle scalar outputs among intervention assignments
   within each developmental context, preserving their multiset. This must fail
   the exact active-set/partition conjunction on at least four of five blocks
   in every K. Report count separately; a constant count of one can accidentally
   match K1 and is not the control criterion.
6. **G6 moved causal mapping:** in each K2–K5 world, move one randomly selected
   active small alias to a different active axis, preserving its portion.
   Generate new disjoint development and held-out contexts. The stale partition
   must mispredict held-out overlap for the moved alias with an old or new group
   representative; newly inferred groups must recover the changed active-set/
   partition exactly on at least four of five blocks for every applicable K.
   The number of axes stays K because the old large alias still restores its
   original axis. This is a physical consequence change, not a token rename.

## Unreachable-variable challenge and claim boundary

For K1–K4 also activate one additional slot whose two actuators remain inert.
Report the discovered reachable count and the actual larger bodily count.
Do **not** promote a reachable count into total dimension or invent a residual
drift rule that guesses one extra axis. Unactuated variables can be observationally
indistinguishable from each other through aggregate drift and minimum alone;
this mechanism has no general total-dimension identification guarantee.

Failure of any gate closes this fixed causal-grouping instrument as a bridge
at this budget. Passing all gates licenses an online exploration experiment
that obtains grouping evidence without matched forks or shock-free contexts.
It does not establish arbitrary state-variable discovery, online state/rate
estimation, shared motor/prediction/speech use, uncertainty reports, generative
language, survival benefit, consciousness, sentience, or completion of the
repository vision. The bounded-resource physical family remains authored.

## Pre-run RNG clarification (2026-09-30, before diagnostic/survey generation)

Independent code review found that directly calling ExpandedBody.reset with the
additive context seeds above would reuse a generator seed: its scale salt minus
birth salt is 6,000,002, equal to three block strides plus two context indices.
That would correlate block-0 metabolic draws with block-3 birth draws. No
diagnostic or survey contexts had been generated when this was identified.

The new adapter must use tagged SHA-256-derived local seeds separately for birth,
metabolism, mapping and each control. Birth still uses uniform choice from the
existing six levels; metabolism still uses uniform [0.4,1.6] in the same slot
order. The physical distributions, context bands, sample, thresholds and all
six gates are unchanged. Existing ExpandedBody remains untouched. Verify
distinct derived seeds over all scheduled experiment/context/domain tuples,
including held-out, controls and diagnostic, before running. Matched branches
within one context deliberately share the same birth/metabolic seeds.
