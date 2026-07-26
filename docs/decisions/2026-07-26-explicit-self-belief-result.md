# Explicit learned self-belief result

Date: 2026-07-26  
Decision: one gate failed; social word planning was not enabled

## Integrity

A 64-unit recurrent self-belief module was appended to the frozen probe-52
adult organism. It trained for exactly 80,000 primitive ticks and 738 one-pass
lived-sequence updates. Only the new module changed; every pre-existing policy,
world-model, lexical, and mouth parameter remained bit-identical.

After birth, the module input forcibly zeroed body channels and contained only
masked observation, own last action, and elapsed primitive duration. True body
values were developmental teaching targets only. Grants were uniformly random
and independent of body; no correct report word, need class, listener target,
or utterance reward existed. Mean loss fell from 0.0704 in the first quarter to
0.0164 in the last.

Artifacts:

- `runs/organism/probe54_explicit_self_belief/treatment/`
- `runs/organism/probe54_explicit_self_belief/diagnostic_detail/`
- `runs/organism/probe54_explicit_self_belief/diagnostic_calibration/`

## Frozen 200-life gate

| Endpoint | Locked gate | Result | Decision |
|---|---:|---:|---|
| Balanced lowest-need accuracy | >=70% | **62.19%** | Fail |
| Observation balanced accuracy | <=45% | 33.33% | Pass |
| Ten-tick change ordering | >=80% | **86.68%** | Pass |
| Zero lesion drop | >=20 points | **28.86 points** | Pass |
| Shuffle lesion drop | >=20 points | **28.67 points** | Pass |
| Portion-fork following | >=60% | **70.89%** | Pass |

The belief-controlled listener loop nevertheless survived 91.5% of lives and
reached 374.83 mean ticks. That does not override the failed identity gate.

## Read-only diagnosis

Balanced accuracy was stable across life: 62.76%, 62.16%, 62.54%, and 61.51%
for ticks 25-99, 100-199, 200-299, and 300-399. This is not accumulating
long-horizon drift. Recall was 57.36% food, 56.01% water, and 73.20% energy.

On a frozen whole-life 140/60 split:

- an affine map of the three predicted body values reached only 62.28%; but
- a disposable ridge readout of the 64-dimensional belief state reached
  **73.53% balanced accuracy**.

Those diagnostic labels never updated the organism. The result localizes the
failure to the continuous MSE readout's need ordering, not lack of relevant
information in recurrent belief state.

Per the stop rule, the social word planner was not implemented or evaluated.
The next single local mechanism is a pairwise rank loss over the same
continuous developmental body targets, preregistered separately. No
categorical need label or report word will be supervised.
