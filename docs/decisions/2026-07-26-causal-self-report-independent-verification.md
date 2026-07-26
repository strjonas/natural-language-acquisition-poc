# Independent verification and multi-seed replication of the causal self-report result

Date: 2026-07-26
Decision: probe57 verified and replicated; claim boundary tightened

This is an adversarial re-audit of the probe57 result by a second agent that
did not implement it. It re-derives every headline number from artifacts,
attacks the leakage surface, runs two controls the sealed battery omitted, and
replicates the causal stage across five independent developmental seeds.

## What held

Reproduction from the frozen checkpoint, 50 lives, seed base 7,100,000:
survival 0.9200, fidelity 0.9538. The sealed 200-life battery CSV contains
every number quoted in the result document.

Parameter integrity is exact. Diffing the probe52 adult parent against the
probe57 checkpoint: 44 pre-existing tensors bit-identical, five new tensors,
**658 trainable causal scalars** in total.

The leakage surface is clean.

- `_public_transition_features` reads only step count, own action index, and
  visible surfaces. It never touches `packet.needs`.
- The grounded listener grants on the word it heard, not on the true need
  (`ReportWorld.hear` -> `_heard_need` -> `HELP_SURFACES`).
- Fidelity is scored against `lowest_need()` sampled before the action.
- The privileged speech inside `audit_structured_causal_self_model` regulates
  the audit world only and never enters belief or measurement.
- `_balanced_accuracy` is macro-averaged recall, not raw accuracy.

System identification genuinely recovered the world's constants:

| Quantity | Learned | True |
|---|---:|---:|
| food drift/tick | -0.0077 | -0.008 |
| water drift/tick | -0.0117 | -0.012 |
| energy drift/tick | -0.0158 | -0.016 |
| roots -> food uptake | 0.576 | 0.60 |
| spring -> water uptake | 0.584 | 0.60 |
| mushroom -> energy uptake | 0.600 | 0.60 |
| thorn/tree/rock shock | -0.251/-0.250/-0.250 | -0.25 |

The extra movement energy cost is the one poorly identified parameter:
learned -0.0179 against a true 0.035.

## Multi-seed replication

`runs/organism/probe58_causal_stage_replication/`

Five independent developmental seed streams from the same probe52 adult
parent, 80,000 ticks each, 100 evaluation lives each. Seed index 0 reuses the
original probe57 stream; the rest are new.

| Endpoint | mean | sd | min | max |
|---|---:|---:|---:|---:|
| Balanced hidden-need accuracy | 0.9181 | 0.0066 | 0.9082 | 0.9251 |
| Mean absolute body error | 0.0169 | 0.0018 | 0.0151 | 0.0190 |
| Grounded survival | 0.9060 | 0.0182 | 0.8900 | 0.9300 |
| Grounded report fidelity | 0.9471 | 0.0034 | 0.9414 | 0.9503 |
| Scrambled-listener survival | 0.0420 | 0.0179 | 0.0200 | 0.0600 |
| Zero-belief survival | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Belief fork following | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Report follows forked body | 0.9077 | 0.0843 | 0.8462 | 1.0000 |
| Listener no-help identification | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

Every promotion gate passed on 5/5 seeds, every causal sign held on 5/5, and
all three listener effect identifications were correct on 5/5. The variance is
small enough that the single-seed report was not a lucky draw.

A prior five-seed set drawn from a different stride (6.1M, 11M, 22M, 33M, 44M)
independently produced balanced accuracy 0.9183 +/- 0.0086, survival 0.9120 +/-
0.0084, and fidelity 0.9510 +/- 0.0054. Two disjoint seed sets agree.

This replicates the **causal stage** only. All runs share one probe52
lexical/motor parent, so it is seed-robustness of the new mechanism, not of
the full childhood-to-adult pipeline. That remains outstanding.

## Harness defect found and fixed

`--seed` did not vary this experiment at all. The causal parameters initialize
deterministically through `mx.full`/`mx.zeros`, so `mx.random.seed` cannot
perturb them, and the harness passed a constant
`seed_base=EVAL_SEED_BASE + 5_200_000`. Re-running probe57 under any `--seed`
would have retrained a bit-identical model while appearing to replicate.

`seed_base` now comes from `causal_replication_seed_base(args.seed)`, so seed 1
reproduces the original stream exactly and other seeds genuinely differ.
`test_structured_causal_development_actually_varies_with_its_seed` guards it.

The obvious 1,000,000 stride is itself wrong and was caught during this audit:
it places `--seed 2`'s developmental worlds at 7,100,000, exactly the planner
battery's evaluation base, so that seed would have developed on the lives it is
scored against. The stride is 10,000,000 and `_check_seed_isolation` rejects any
developmental stream overlapping the 6,500,000-7,600,000 evaluation band.
`test_replication_seed_streams_never_touch_the_evaluation_bands` guards it.

The vocabulary is 60 tokens, not 52. The code always used `len(VOCAB)` and is
unaffected; five documentation references were wrong and are corrected.

## Two controls the sealed battery omitted

**Birth-anchor dependence.** Belief is initialized from the true body at birth
(`reveal_birth_needs=True`, preregistered). Corrupting that anchor, 100 lives
on the probe57 stream:

| Birth belief | Survival | Fidelity | Body MAE |
|---|---:|---:|---:|
| true | 0.92 | 0.9484 | 0.0175 |
| + Gaussian noise sd 0.05 | 0.85 | 0.9415 | 0.0182 |
| + Gaussian noise sd 0.10 | 0.87 | 0.9325 | 0.0197 |
| + Gaussian noise sd 0.20 | 0.85 | 0.9159 | 0.0236 |
| random birth level | 0.79 | 0.9058 | 0.0284 |
| population mean, zero per-life information | 0.81 | 0.9062 | 0.0278 |

Fidelity degrades only 4 points across the whole sweep and stays far above the
0.60 gate everywhere, including with no per-life bodily information at all.
Survival is the more sensitive endpoint: the population-mean anchor still
clears the 0.80 gate at 0.81, while a random birth level lands marginally
below it at 0.79. State this precisely rather than claiming the anchor is
irrelevant -- it is worth a few points of survival and almost nothing in
fidelity.

It follows that most of the belief is reconstructible from public history
alone, which probe53 had already established analytically.

**Is the belief open-loop?** No. Mean absolute error *falls* across life,
0.0184 -> 0.0131 over 50-tick buckets. Corrupting the belief in place at tick
100 leaves survival unchanged at 0.917 and costs little fidelity: sd 0.10 ->
0.939, sd 0.25 -> 0.904, with post-shock error settling at 0.0280 against a
0.0158 baseline.

Clipping at the homeostatic bounds, combined with the speak/help/consume loop
driving needs to saturation, makes the belief a **contracting observer**
rather than a dead-reckoner. This is a real property of the system that the
original battery neither measured nor claimed. It is also a warning: the
architecture has no evidence-correction pathway, and it survives without one
only because the ecology's saturation re-anchors it.

## Tightened claim boundary

The result is genuine, reproducible, well-controlled, and now replicated. What
it is *not*:

1. **The self-model's structure is stipulated, not discovered.** Three axes,
   linearity, and the sign of every effect were specified by the designer. The
   learner fit 658 scalars into a correct hand-built template. A fourth hidden
   bodily variable could not be represented at all.
2. **The hand-coded solution is still better.** Probe53's exact filter reaches
   99.96% with zero error; the learned model reaches 91.8% +/- 0.9%. This is a
   lossy approximation of a closed form the designer already possessed. No
   experiment yet shows the learned model doing something the analytic filter
   cannot.
3. **There is no reflection.** No uncertainty representation, no reasoning
   about the model as opposed to the state, no past or future self, no
   counterfactual self-description. Utterance selection is an argmin over three
   numbers resolving to one of three words.
4. **The signaling result sits inside a populated literature.** Emergent
   communication of private state is well studied. The distinctive asset here
   is the intervention battery and the preregistration discipline, not the
   signaling behavior.

## Consequences for sequencing

Multi-seed replication is discharged for the causal stage. The next
experiment should be **online adaptation**: change metabolic rates, shock
magnitudes, or portion distributions after development and continue only the
causal model, against frozen, reset, and exact-filter controls.

That is the decisive one. It is where a learned model can beat the analytic
filter, whose constants are baked in and must fail. Passing it converts this
work from a lossy re-derivation of a known solution into evidence for
something the closed form cannot do.

After that, in order: latent structure discovery (over-parameterize the state
and require sparsity to recover three axes rather than being told three);
evidence integration (return interoception intermittently and require
correction, which the current architecture cannot perform); and only then
reflective communication about predicted future state and uncertainty.
