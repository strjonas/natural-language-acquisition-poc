# Implemented persistent-choice mechanics result

Date: 2026-07-25  
Decision: environment mechanics pass; permit a locked local learning pair

## Integrity

The 2,500-life, eight-round audit and its gates were locked in
`2026-07-25-persistent-choice-mechanics-preregistration.md`.

Result:

`runs/organism/probe22_persistent_choice_mechanics/harness_seed1_grounded-silent-shuffled.csv`

## Results

| Policy | Rounds/life | Final min need/round | Gain vs blind | Correct | Poison | Ticks/round | Inspections/life |
|---|---:|---:|---:|---:|---:|---:|---:|
| Blind | 8.00 | 0.5400 | 0.0000 | 33.14% | 33.72% | 4.875 | 0.00 |
| Public-label persistent | 8.00 | 0.6212 | **+0.0813** | 100.00% | 0.00% | 7.375 | 2.00 |
| Clairvoyant ceiling | 8.00 | 0.6700 | +0.1300 | 100.00% | 0.00% | 4.875 | 0.00 |

All policies completed exactly eight rounds. Blind accuracy is inside the
preregistered 30-37% interval. The public-label policy is 100% correct, uses
exactly two inspections, and exceeds blind by 0.0813 per round, passing the
0.07 gate. The ceiling remains higher.

Unit tests additionally establish that the consumption-consequence packet
precedes the new demand, the transition is forced, the mapping and surface
identities persist, demand alternates, and only the final round terminates.

## Decision

The implemented environment preserves the positive information economics of
the audit and is ready for a local learned write-enabled/write-disabled pair.
This permits learning evidence, not promotion: seed 2, language controls,
open-island transfer, speaking, and larger compute remain blocked.
