# Need-report rent result

Date: 2026-07-26
Status: complete negative result
Preregistration: `2026-07-25-need-report-rent-preregistration.md`
Artifacts: `runs/organism/probe48_need_report_rent/`

## Result

The locked seed-1, 200,000-step consequence-only report run fails T1-T3.

| Metric | Result | Gate |
|---|---:|---:|
| Held-out report fidelity | 0.0045 | >=0.60 |
| Held-out survival | 0.0000 | >=0.80 |
| Mean life steps | 33.52 / 400 | — |
| Grounded mean viability | 0.0579 | — |
| Scrambled survival | 0.0000 | grounded >= control +0.15 |
| Mute survival | 0.0000 | grounded >= control +0.15 |
| Perceptible-fork report change | 0.0138 | causal direction required |
| Perceptible-fork body following | 0.0058 | causal direction required |

Tick-200 persistence is unmeasurable because no evaluated life reaches that
regime.  The organism mostly emits food/water words and essentially never
reports energy, despite energy being the fastest-depleting need.

## Epistemic audit repair

Before the locked run, the counterfactual-body audit was corrected.  The old
implementation silently edited the true portion after consumption while
keeping the visible portion surface identical, asking a history-based agent to
know an unobserved intervention.  The corrected paired audit fixes social and
motor streams, changes exactly one perceptible small/large portion and its
lived bodily consequence, and compares later reports only after uptake.  The
training path is unchanged.

The harness was also repaired so a saved report checkpoint can be audited
read-only.  Full verification after the repair passed.

## Post-hoc substrate diagnostic

An oracle-listener diagnostic ignored the frozen organism's reports, supplied
the true lowest-need word, and left its motor policy in control:

| Need | Grants | Uptakes | Uptake/grant |
|---|---:|---:|---:|
| Food | 218 | 218 | 1.0000 |
| Water | 271 | 271 | 1.0000 |
| Energy | 1,189 | 190 | 0.1598 |

Survival remained 0%.  Energy alone required `REST`; food and water used the
already learned `CONSUME` action.  This supports a circular discovery barrier:
`tired` has no value until rest-at-shelter is learned, and rest-at-shelter has
no training examples until `tired` is said.

This diagnostic was declared post-hoc and is not a capability gate.

## Claim boundary

Probe48 establishes that the full-vocabulary mouth receives consequence
gradients and learns nontrivial traffic, but not truthful self-report.  It is
not evidence of reflection or communicated self-knowledge.

