# Working in this repository

## What this project is

An attempt to build an embodied agent with a causally grounded, persistent
self-model that it can reflect on and communicate -- established by
intervention, never by fluency. Read in this order:

1. `md/archive/DIRECTION_2026-07-26.md` -- the destination and the phase ladder.
2. `docs/STATE.md` -- the rolling handover. Where things stand right now.
3. `docs/decisions/` -- the durable evidence record, append-only.

Everything in `docs/` not listed there is frozen history. `DIRECTION_2026-07-26.md`
section 6 names it explicitly.

## Environment

- Use `.venv/bin/python`. The pyenv python on this machine has no pytest, so a
  bare `python -m pytest` fails misleadingly.
- `PYTHONPATH=src` for module invocations.
- MLX on Apple Silicon. Everything here trains locally in minutes; there is no
  cluster and none is needed yet.

```bash
.venv/bin/python -m pytest -q
```

## Method (binding)

1. **Preregister before implementing.** Write the mechanism, the locked gates,
   and the failure condition into `docs/decisions/<date>-<name>-preregistration.md`
   first. Then implement. Then write the result doc. Gates are not adjusted
   after seeing results -- a failed gate closes the mechanism.
2. **Controls, not averages.** Every positive claim needs the lesion that
   should kill it: shuffled, zeroed, frozen, muted, scrambled-listener, fixed
   word. A result without a failing control is not a result.
3. **Multi-seed from the start**, >= 5. See the seed trap below.
4. **Language is never rewarded directly.** It pays rent only through
   consequence -- help arrives, harm is averted.
5. **New mechanisms go in their own module.** `train.py` is 9,481 lines and
   `OrganismConfig` already carries 69 knobs, 55 at default and many belonging
   to closed lines. Do not add knobs to it. `causal_self.py` is the pattern to
   follow: separate module, default off, existing parameters left bit-identical.
6. **Claim boundaries are part of the result.** State what the experiment does
   not show, in the result doc, every time.

## Traps this repository has actually fallen into

- **The seed that does nothing.** The causal parameters initialize
  deterministically, so `mx.random.seed` cannot vary that stage -- only the
  developmental world stream (`seed_base`) can. A harness that holds
  `seed_base` fixed retrains a bit-identical model for every `--seed` and
  reports perfect "replication". Guarded by
  `test_structured_causal_development_actually_varies_with_its_seed`.
- **The seed stride that leaks.** A 1,000,000 stride puts `--seed 2`'s
  developmental worlds on the planner battery's evaluation base, so that seed
  develops on the lives it is later scored against. Use
  `causal_replication_seed_base`; `_check_seed_isolation` rejects overlaps.
- **The environment doing the model's job.** The v1 belief survives with no
  evidence-correction pathway only because homeostatic clipping re-anchors it.
  Before crediting a mechanism, check whether the ecology is carrying it --
  corrupt the state mid-life and see whether anything degrades.
- **A mechanism that explains everything and is still not the cause.** From the
  probe33 era: the drift term really did inflate gradients 6x and saturate the
  clip on every update, and explained every observation, and was still not the
  cause. Run the control anyway.
- **Documentation drifting from code.** The v1 write-up said "52 tokens" in five
  places; the vocabulary is 60 and the code always used `len(VOCAB)`. Check
  numbers against the code before putting them in a doc.

## Artifacts

- `runs/` is **gitignored**. Run outputs are not durable; the numbers in
  `docs/decisions/` are. Always write the headline numbers into a decision doc.
- Checkpoints are `.npz` plus a sibling `.npz.json` config, loaded by
  `load_organism_checkpoint`.
- Reproduction commands live in `README.md` under "Unified organism
  self-report". Keep them runnable.

## Reporting

Report what happened, including when a gate failed. A negative result that
closes a mechanism is worth more here than a positive one that was not
controlled -- probes 48 through 56 are all negative and they are why probe57 is
credible. Do not describe a partial result as complete, and do not soften a
failed gate into a "mid result".
