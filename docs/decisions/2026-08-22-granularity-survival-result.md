# Probe72: does a finer decision surface carry individual-rate knowledge to survival? — result

Date: 2026-08-22. Preregistration:
`docs/decisions/2026-08-22-granularity-survival-preregistration.md`.
Module: `src/homesocial/organism/granularity_survival.py`. Guards:
`tests/test_granularity_survival.py`. Artifact:
`runs/organism/probe72_granularity_survival/treatment.json`.

Five paired seed blocks x 140 lives per cell = 700 lives per arm, treatment
band 2,800,000,000, lag 18. The frozen parent, body, metabolic spread, expected
help per tick and inference budget are identical between ecologies. The fine
ecology halves each portion and the caregiver period together.

**Verdict: three of five locked gates pass. G1 and G2 fail and are not
rewritten.** True individual-rate knowledge has a small positive survival value
in the fine ecology, but refining the help quantum does not increase it. The
clean interaction moves in the wrong direction on average and only 2/5 seed
blocks have the predicted positive sign.

| gate | requirement | result | |
|---|---|---|---|
| **G1** | rate step > 0, interval clear of zero, >= 4/5 positive | **-0.0229 [-0.0869, +0.0412], 2/5 positive** | **fail** |
| **G2** | `abs(state step) < rate step` | state **+0.0286** vs rate **-0.0229** | **fail** |
| G3 | fine rate value > 0, interval clear, >= 4/5 | **+0.0214 [+0.0089, +0.0340], 5/5** | pass |
| G4 | oracle survival >= 0.30 in both ecologies | **0.3643, 0.5057** | pass |
| G5 | expected grant/tick identical within 1e-12 | **0.0666667 = 0.0666667 exactly** | pass |

## 1. Survival

| ecology | oracle | state oracle | population |
|---|---:|---:|---:|
| baseline, portion 1 / period 6 | 0.3643 | 0.3200 | 0.2557 |
| fine, portion 1/2 / period 3 | 0.5057 | 0.4843 | 0.3914 |
| fine - baseline | **+0.1414** | **+0.1643** | **+0.1357** |

The paired 95% intervals on those ecology-wide lifts are [+0.0900, +0.1929],
[+0.1204, +0.2082] and [+0.0871, +0.1843], respectively. Smaller, more
frequent help at the same expected mass per tick improves survival for every
arm. It improves `state_oracle` more than `oracle`, which is the opposite of the
selective rate-value mechanism G1 predicted.

Mean life length tells the same story: oracle 219.0 -> 272.2 ticks,
state-oracle 203.4 -> 262.5, and population 174.1 -> 228.6. The construction is
not collapsed and the null is not caused by survival saturation.

## 2. The clean one-factor contrasts

    RATE_VALUE  = oracle - state_oracle
    STATE_VALUE = state_oracle - population

| contrast | baseline | fine | fine - baseline |
|---|---:|---:|---:|
| rate value | +0.0443 [-0.0098, +0.0984] | **+0.0214 [+0.0089, +0.0340]** | **-0.0229 [-0.0869, +0.0412]** |
| state value | +0.0643 [+0.0218, +0.1068] | +0.0929 [+0.0751, +0.1106] | +0.0286 [-0.0228, +0.0799] |

Per-block rate steps are +0.0357, -0.0571, +0.0071, -0.0929 and -0.0071.
Probe68's open-loop rate step was +0.0855 on 8/8 seeds. The same construction's
survival step is therefore not merely unresolved: its point estimate reverses
and its locked replication count fails.

The fine ecology still supplies the physical endpoint Probe65 left unresolved.
With state perfect in both arms, using this individual's true rates rather than
species rates buys **2.14 survival points**, every seed block agrees, and the
interval excludes zero at the preregistered 700 lives per arm. That is G3, and
it matters. It says rate knowledge can reach survival here. It does not say the
finer quantum made it do so: baseline rate value is numerically twice as large,
although its much wider interval includes zero.

## 3. What failed, exactly

Probe68's formula remains a correct account of the **word-decision margin**:

    margin = E_grant[min(grant * uptake, axis gap)]

Halving the quantum moves that margin and raises open-loop consequential need
accuracy. Probe72 shows that this does not imply a larger survival contrast.
The ecology intervention changes the timing of every grant and lifts every arm
by 13.6 to 16.4 points; the survival benefit is larger for the species-rate
state oracle than for the individual-rate oracle. Whatever converts the finer
clock into longer life is broader than the rate-error decision margin.

This is not a post-hoc objection that invalidates the construction. Equal
expected nutrition, viability and a clean positive fine rate contrast all pass.
It is the result: a mechanism can enlarge the accuracy value of a self fact
without enlarging that fact's survival value.

G2 fails with it. State value moves +0.0286 while rate value moves -0.0229, so
the finer ecology does not selectively expose rate knowledge at the physical
endpoint. The preregistered lesion does its job.

## 4. Claim boundary

- This closes **smaller, more frequent help as a way to amplify the survival
  value of individual-rate knowledge** in this fixed lag-18 ecology. The
  open-loop Probe68 result stands; its transfer to survival does not.
- The positive G3 endpoint is a clean oracle ceiling, not a learned self-model.
  It licenses asking whether the recursive learner can capture that 2.14-point
  value in a separately controlled experiment; it does not answer that question.
- The result does not identify why all arms benefit from the finer clock. Grant
  timing, clipping, outstanding requests, death hazards and motor trajectories
  are all inside the live loop and were not separately intervened on here.
- No model trains, so there is no family saturation claim. Every arm uses the
  same frozen 80,000-tick parent checkpoint and matched inference budget.
- No self-model property advances. The only belief-like quantity that reaches
  survival here is oracle true-rate knowledge, not a learned belief. Probe65
  remains the learned system's behavioural result.
- Nothing here establishes discovery, uncertainty reporting, structure,
  language productivity, consciousness or sentience.

## 5. Reproduction

The runner checkpoints after every 20 lives and resumes an interrupted artifact
without repeating completed chunks. The treatment took 70.8 minutes on this
machine.

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.granularity_survival \
  --out runs/organism/probe72_granularity_survival/treatment.json
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_granularity_survival.py \
  tests/test_decision_granularity.py tests/test_future_request.py
```
