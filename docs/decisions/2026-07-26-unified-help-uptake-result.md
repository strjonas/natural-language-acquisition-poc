# Unified-help uptake result

Date: 2026-07-26
Status: complete negative capability result with positive mechanism result
Preregistration: `2026-07-26-unified-help-uptake-preregistration.md`
Artifacts: `runs/organism/probe49_unified_help_uptake/`

## Feasibility seal

With no ecology retuning, unified voluntary `CONSUME` uptake passes F1-F5:

| Policy | Survival | Mean viability | Mean steps |
|---|---:|---:|---:|
| Truthful oracle | 0.9675 | 0.7404 | 389.61 |
| Mute | 0.0000 | 0.0390 | 29.34 |
| Random word | 0.0275 | 0.1810 | 102.78 |
| Best blind rhythm (`ewf`) | 0.5400 | 0.5453 | 296.18 |

Oracle survival is >=0.95, both language-necessity margins exceed 0.20, and
the oracle viability margin over every blind policy exceeds 0.05.

## Locked learning result

The sole training manipulation was unified uptake.  All learner settings,
seed, hidden-body constraints, vocabulary, and the 200,000-step budget matched
probe48.

| Metric | Result | Gate |
|---|---:|---:|
| Held-out report fidelity | 0.3435 | >=0.60 |
| Held-out survival | 0.0500 | >=0.80 |
| Grounded mean life steps | 131.39 | — |
| Grounded mean viability | 0.2299 | — |
| Scrambled survival | 0.0200 | grounded >= control +0.15 |
| Mute survival | 0.0000 | grounded >= control +0.15 |
| Held-out birth fidelity | 0.3453 | preservation required |
| Held-out portion fidelity | 0.3435 | preservation required |
| Perceptible-fork report change | 0.0345 | causal direction required |
| Perceptible-fork body following | 0.0044 | causal direction required |

T1 and T2 fail.  Grounded exceeds scrambled survival by only 0.03, not 0.15.
Fidelity is stable across time buckets but near chance, so this is persistent
non-truth rather than persistent reflection.

## Mechanism diagnostics

The uptake barrier is removed exactly:

| Need | Uptake/grant under oracle words |
|---|---:|
| Food | 0.9994 |
| Water | 0.9969 |
| Energy | 0.9983 |

Oracle-listener survival is 0.9550.  The substrate can support the capability
and the frozen motor policy can execute it.

A decoder trained only for read-only diagnosis, split by whole held-out life,
finds:

| Input | Accuracy | Balanced accuracy |
|---|---:|---:|
| Shared recurrent state feeding the mouth | 0.6922 | **0.6404** |
| Current masked observation | 0.3827 | **0.3285** |
| Majority | 0.3757 | — |
| Chance | 0.3333 | 0.3333 |

The true label trained only disposable ridge readouts and never updated the
organism.  This is evidence that online embodied experience formed a usable,
persistent hidden self-state representation inside the unified organism.  It
is not evidence that the organism itself can communicate that representation.

## Interpretation

Unified uptake is a substantive mechanism success and a capability failure.
It removes the action convention, lengthens life, broadens need-word use, and
reveals a linearly accessible self-state.  Consequence-only REINFORCE still
does not bind that self-state to truthful generated language.

