# Probe77: sequential randomized occlusion at a lived budget

Date: 2026-09-30. Status: preregistered before implementation or data.

## Question and mechanism

Can a finite stream of ordinary scalar sensations replace Probe76's matched
forks? Test one deliberately bounded candidate: randomize opaque restorative
actions and estimate whether the previous action suppresses the next action's
absorbed effect. Same-resource saturation can produce suppression; different
independent resources cannot in expectation under randomized actions.

Ten opaque IDs, K1–K5, fixed five-slot mean, original birth/rate distributions,
0.025 shocks, help every six ticks with absorption on the next tick. Inactive
slots remain inert and full, including under shocks. No matched forks, state
vectors, rates, portion sizes, axis names, target maps or shock labels enter
the estimator. The action schedule is i.i.d. uniform over ten IDs, independent
of the body and sensations. This tests randomized exploration, not an adaptive
exploration policy or a motor controller.

For a grant tick, estimate effect as its mean increment minus the median of
up to five preceding no-grant mean increments. Require at least three such
increments; do not consume post-grant observations to estimate this baseline.
Shocks stay in these observations. A grant-effect record contains only the
current ID, previous granted ID (within the episode), and estimated effect.

An ID is active when at least three observations have median effect > 1e-6.
For each ordered pair of distinct active IDs a,b, compare b's effects when
preceded by a against b's effects following any other ID. Require three
samples in each group. Call suppression only if the mean difference exceeds
two estimated standard errors and 1e-6. Use sample variance with a floor of
1e-12. An undirected edge needs suppression in either direction; connected
components are proposed restorative groups. These thresholds are fixed, not
tuned on the survey. Return supported-pair coverage as well as a partition;
an unsupported pair is not evidence of different resources.

## Experience, sample, controls

Five independent blocks, 20 bodies per K per block: 500 body identities.
Each identity keeps its rates and ID map over six independent 400-tick
episodes. Primary budget is episode one; secondary budget is the first six.
Death ends that episode, with no extra samples credited. Cross-episode pooling
is explicitly a persistent-body assay, not learning within one uninterrupted
life. Record nominal and actual ticks, observations, active-set recall and
precision, pair coverage, exact counts/partitions and survival.

For every identity after development, run two additional independent episodes
with the grouping frozen. Held-out endpoint: predict scalar residual effect
for each action using its development mean and the mean under
the preceding action when available; compare squared error to a per-action
mean baseline. Report this prediction endpoint even if grouping fails.
Structural balanced accuracy is scored over distinct actionable ID pairs
against the hidden map; K1 has no negative pairs and is excluded from that
balanced-accuracy endpoint. Count recovery never establishes state estimation.

Controls use the same observations: (1) shuffle effect values across records,
preserving action IDs and episode boundaries; (2) permute all IDs consistently,
requiring exact partition transport. For K2–K5, physically remap one small
alias and collect six *new* episodes without carrying grouping evidence;
score new recovery and whether the old grouping matches the changed truth.
For K1–K4, activate an extra resource but block its actions, and repeat six
episodes. Total dimension must always be marked unresolved; score only the
reachable partition, and include early deaths. Neither challenge consumes
development or held-out observations of the original learner.

Tagged SHA-256 RNG domains separate identity, birth, shocks, actions, mapping
and controls. Main seed base 3,900,000,000; diagnostic base 3,950,000,000;
block stride 2,000,000, K stride 100,000, identity stride 100, episode offsets
0–7 development/test, 10–15 remap, 20–25 unreachable. Diagnostic: one block,
one identity per K, never enters survey scoring. Five blocks are replication
units; bodies within them are not additional seed replicates.

## Locked continuation rule

Evaluate the primary and secondary budgets separately; the secondary cannot
rescue a claim about one episode. For a budget to license a subsequent online
estimator treatment, all clauses must pass:

1. Exact reachable active-set/partition recovery >= 0.80 in every K pooled
   across the 100 bodies, and >= 0.70 in at least four blocks at every K.
2. Structural balanced accuracy >= 0.80 at every K2–K5; count mean tracks K
   strictly. Include missing/unsupported active IDs as prediction errors.
3. Own-pairing exact-partition rate exceeds shuffled pairing by >= 0.20 in
   every K, and ID permutation transports the entire prediction exactly.
4. Held-out effect MSE is lower than the per-action mean baseline on at least
   four blocks and the mean reduction is >= 10% (zero baseline error fails).
5. At the six-episode challenge budget, remapped exact recovery >= 0.80 at
   every K2–K5 and stale partitions are wrong in >= 0.80 of those worlds.

Unreachable challenges are descriptive and enforce the claim boundary, not
a requirement to recover inaccessible dimensions. Survival is an exploration
cost, not a success gate: no survival benefit is claimed by this instrument.
If either budget fails, close this randomized predecessor-contrast mechanism
at that budget. Do not change thresholds, add episodes, choose a favourable K,
or infer that another sequential estimator or directed policy cannot work.

## Verification and deliverable

Test delayed absorption, death truncation, no hidden input, pairing/permutation,
disjoint seeds, held-out exclusion, gate binding, and resume source pins.
Write atomic per-world progress, source/configuration hashes, compact results
and a result decision. First time the separate diagnostic to estimate the full
run; no expensive learner or parent retraining is licensed by a failed screen.
