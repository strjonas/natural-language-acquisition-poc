# Persistent-mapping semantic-rent upper-bound result

Date: 2026-07-25  
Decision: implement the multi-round childhood substrate

## Integrity

The mechanism, fixed round counts, 2,500-life seed set, policies, and gates were
locked in `2026-07-25-persistent-mapping-rent-preregistration.md`.

The result is stored in:

`runs/organism/probe21_persistent_mapping_rent/harness_seed1_grounded-silent-shuffled.csv`

This was an exact-dynamics audit. It did not instantiate a learned checkpoint,
update parameters, or inject simulator information into learning.

## Results

| Rounds/life | Policy | Final min need/round | Gain vs blind/round | Correct | Inspections/life |
|---:|---|---:|---:|---:|---:|
| 1 | Blind | 0.5364 | 0.0000 | 32.96% | 0.00 |
| 1 | Persistent labels | 0.3443 | -0.1922 | 100.00% | 1.67 |
| 2 | Blind | 0.5356 | 0.0000 | 32.46% | 0.00 |
| 2 | Persistent labels | 0.4750 | -0.0606 | 100.00% | 2.00 |
| 4 | Blind | 0.5356 | 0.0000 | 32.46% | 0.00 |
| 4 | Persistent labels | 0.5725 | **+0.0369** | 100.00% | 2.00 |
| 8 | Blind | 0.5356 | 0.0000 | 32.46% | 0.00 |
| 8 | Persistent labels | 0.6212 | **+0.0857** | 100.00% | 2.00 |
| 8 | Clairvoyant ceiling | 0.6700 | +0.1344 | 100.00% | 0.00 |

The preregistered gate passes at both four and eight rounds. At four rounds the
persistent policy exceeds blind by 0.0369 per round, has 100% correct choices,
and uses exactly two inspections per life. The eight-round margin is 0.0857.
The clairvoyant ceiling remains higher than both.

## Interpretation

The prior one-shot failure was not evidence that the acquired lexical memory
was inherently useless. It was evidence that the task destroyed the memory
before it could repay acquisition cost.

With recurring hunger and thirst under one persistent hidden mapping:

- the first one or two rounds pay the physical cost of acquiring labels;
- the same surface-keyed values are then reused without new language;
- the break-even point occurs before round four; and
- utility approaches the clairvoyant ceiling as the number of reuse rounds
  grows.

This is aligned with the project goal in a way a cheaper inspection would not
be: value comes from autobiographical continuity of a learned mapping, not from
an externally subsidized sensing action.

## Implementation selected

Add `semantic_choice_rounds`, defaulting to one for backward compatibility.
For multi-round lives:

1. keep the same surface identities, kinds, and positions;
2. alternate the low food/water demand;
3. after consumption, expose the true post-consumption body packet;
4. force one non-agent WAIT-coded transition into the next demand, with actor
   and entropy weight zero;
5. restore the three objects and canonical center/NORTH pose;
6. preserve recurrent state and the episodic lexical bank; and
7. terminate only after the final round or death/timeout.

The explicit transition is necessary: resetting the demand in the consumption
packet would erase the very bodily consequence that grounds the word.

Before training, an exact public-label policy and blind control must reproduce
the positive rent inside the implemented multi-round environment. No compute
or data request is justified.
