# Metabolic drift supervision preregistration

Date: 2026-07-25
Status: locked before the loss is changed and before any output of the changed
loss is read. The measuring instrument
(`audit_metabolic_drift_forecast`) was written first, against the unchanged
sealed checkpoint, and its baseline numbers are recorded below.

## Why this is the right target, and why it is not another planner

`2026-07-25-protocol-branch-planner-result.md` closed three planners on one
diagnosis: the corrected homeostatic utility scores whichever need the organism
predicts will be most urgent when it returns, and that prediction decays over
the ten-tick acquisition detour until it no longer ranks the organism's own
needs correctly. The failing gate 6 came in at 18.00%; the rate at which the
predicted post-return body still identifies the demanded resource was 18.44%.

The new committed audit adds the measurement that was missing, the **oracle
substitution**: the identical chain with the simulator's true drift in place of
the predicted drift. On the unchanged sealed checkpoint, 300 fixed contexts
from seed `1_700_000`:

| Body the urgent need is read from | Names the demanded resource |
|---|---:|
| Real current observation | 100.00% |
| Model's predicted post-inspect body | 62.00% |
| Model's predicted post-return body | **18.00%** |
| **Oracle drift, same chain, same steps** | **100.00%** |

This is the fact that justifies the experiment. Every other component of the
forecast chain — the write, the settling observations, the terminal utility,
the return construction — is already sufficient. Replacing only the predicted
drift with the true drift takes the chain from 18.00% to 100.00%. The forecast
error is therefore not merely correlated with the failure; it is the whole of
it.

The precision required is also now measured rather than guessed. The margin
between the demanded need and the runner-up is a constant **0.2000** in every
one of the 300 contexts, while the true food/water drift differential over the
whole detour is only 0.04. The ordering is robust to any forecast whose
per-need error stays well inside that margin. The current per-need mean
absolute errors are 0.0959 (inspect food), 0.0867 (inspect water), 0.0899
(return water) and 0.1080 (return energy).

## The defect in the loss, measured

The bodily-delta head is not slightly biased on long options. It has largely
not learned metabolism at all. One-tick primitive predictions at the choice
pose, against a true per-tick drift of food -0.010 and water -0.014:

| Action | Predicted food | Predicted water | True food | True water |
|---|---:|---:|---:|---:|
| TURN_LEFT | -0.1386 | -0.0665 | -0.0100 | -0.0140 |
| MOVE_FORWARD | -0.0958 | -0.1744 | -0.0100 | -0.0140 |
| WAIT | -0.0897 | -0.0365 | -0.0100 | -0.0140 |
| REST | -0.0653 | -0.0214 | -0.0100 | -0.0140 |

Two measurements identify the cause.

1. **The drift regime receives about one percent of the head's gradient.**
   Over the implemented task, entries with `|target delta| <= 0.175` are 82.7%
   of all entries but carry a mean magnitude of 0.058 against 0.330 for event
   entries, so squared error alone favours events by a factor of about 32; the
   existing `1 + 20 * max|target delta|` transition weighting adds a further
   2.6 (72.1% of weight mass on 48.9% of transitions).

2. **What the head learned instead is the consumption confound.** The
   correlation between the predicted delta and the *current* need level is
   -0.923 for food and -0.951 for water. Consumption happens when a need is
   low, so in the training data a low need predicts a large positive delta; the
   head fits that need-anticorrelated line and extrapolates it to a large
   spurious *negative* delta whenever a need is high and no consumption occurs.
   This is exactly the regime the acquisition detour sits in.

A loss that is dominated by events cannot separate the two regimes, because the
regression that fits the events is the one that breaks the drift.

## Mechanism: a scale-balanced drift term

One term is added to the bodily-delta loss, and only that term. The existing
change-boosted term is preserved verbatim, so the consumption predictions the
current headline result depends on keep their supervision unchanged:

```python
per_entry     = (predicted - target) ** 2                 # (T, 4), per need
per_transition= per_entry.mean(axis=-1)                   # (T,)
weights       = 1.0 + change_boost * max|target|          # (T,)   unchanged
event_term    = (per_transition * weights).sum() / weights.sum()   # unchanged

drift_mask    = (|target| <= drift_threshold)             # (T, 4), per entry
drift_term    = (per_entry * drift_mask).sum() / drift_mask.sum()
loss          = event_term + drift_weight * drift_term / drift_scale ** 2
```

Three properties make this the minimal correction rather than a knob:

- The stratification is **per entry, not per transition**, so the water drift
  on a food-consumption transition is supervised as drift. That is precisely
  the entry the confound corrupts.
- The `drift_scale ** 2` denominator converts the drift regime to a *relative*
  error. A squared loss on quantities of size 0.01 is otherwise invisible
  beside one on quantities of size 0.5, whatever the counts.
- The term is **self-limiting**. It is large only while the drift fit is bad
  and falls below the event term once the residual approaches the drift scale,
  so it cannot trade away the consumption fit at convergence.

Constants, fixed now and not searched:

- `drift_threshold = 0.175`. Chosen from the empirical delta histogram over the
  implemented task, which has an **empty bin** at `[0.175, 0.200)`: metabolic
  drift over the longest option tops out near 0.14 and consumption events begin
  near 0.225. It is a gap in the data, not a tuned cut.
- `drift_scale = 0.02`. The per-tick metabolic scale of the world
  (food 0.010, water 0.014, energy 0.015 walking).

The same term is applied to the two-step open-loop rollout loss, which carries
the identical `1 + boost * max|target|` weighting and is the loss that actually
supervises the multi-step forecast the planner performs.

`drift_weight = 0.0` is the default and reproduces every sealed artifact
bit-for-bit. It is the single manipulated variable of this experiment.

## Calibration, and why it does not contaminate the endpoint

`drift_weight` is selected over the fixed grid **{0.01, 0.03, 0.1, 0.3}** by
the following rule, decided now:

> Train all four at the identical probe24 seed-1 configuration. Select the
> **smallest** weight whose `worst_need_absolute_error` across both steps and
> all four needs is below 0.05. If none reaches 0.05, select the weight with
> the smallest `worst_need_absolute_error`. Ties go to the smaller weight.

The selection criterion is **forecast accuracy**, which is the quantity the
added term directly optimizes. The endpoint is the **planner's gate 6**, a
different quantity produced by a different mechanism, which the loss never
sees. Selecting a regularization weight on the loss's own target and then
testing a downstream behavioral consequence is a two-stage design, not a search
on the endpoint. No gate below is adjusted after any run.

Nothing else changes: architecture, hidden size 64, optimizer, learning rate,
entropy, reward shaping, replay, the 30,000-tick budget, the planner, the
utility rule, the settling count, the reuse count and the planning scale are
all held at their sealed values.

## Locked gates

### Mechanism gates, judged first

On the selected checkpoint, `audit_metabolic_drift_forecast` over the same 300
fixed contexts from seed `1_700_000`:

- **M1.** `worst_resource_drift_bias` (the larger of the absolute food and
  water signed biases, over both the inspect and return steps) is below
  **0.01**. Baseline 0.0663.
- **M2.** `worst_need_absolute_error` across all four needs and both steps is
  below **0.05**. Baseline 0.1080. This is the gate the measured 0.2000 margin
  makes operative.
- **M3.** `urgent_index_survives_predicted_post_return` is at least **90%**.
  Baseline 18.00%.

### Endpoint gate, judged only if M1–M3 pass

- **E1.** The protocol-branch feasibility audit's gate 6 — a branch whose label
  names the resource the body demands selects the labeled object — is at least
  **60%** on the same 300 contexts at reuse count 7. Baseline 18.00%.
- **E2.** Gates 1, 2, 3, 4, 5 and 7 of that same locked seven-gate feasibility
  battery continue to pass at their preregistered thresholds. They passed on
  the baseline and must not be bought back.

### Non-regression gates, judged on the selected checkpoint regardless

The headline sealed result must survive a change to the world-model loss.

- **N1.** Cross-round label reuse: each of rounds 3–8 selects the needed object
  at least **90%** of the time. Baseline 100.00% in every round.
- **N2.** The matched controls stay at chance: acute write suppression and
  acute silence each at most **45%**. Baseline 34.11% and 34.00%.

If N1 or N2 fails, the drift term is rejected outright whatever M and E say,
and the sealed probe24 checkpoint remains the reference.

## If everything passes

Only then run the behavioral pair: one fresh matched write-enabled /
write-disabled training pair at the same configuration with the planner
activated after 15,000 ticks, judged by the nine unchanged promotion gates of
`2026-07-25-persistent-childhood-learning-preregistration.md`.

Seed 2, freshly trained silent and shuffled controls, open-island transfer,
generated speech, and any compute or data request remain blocked until that
pair passes.

## Stop rule

If M1–M3 fail, the loss correction is a negative result. Do not add a fifth
planner, do not widen the grid, do not move `drift_threshold` or `drift_scale`,
and do not weaken any gate. The remaining option named in `docs/STATE.md`
stands: let the organism learn the value of knowing from experienced
cross-round reuse under a longer training budget, which is the first thing in
this project that would justify a compute request.

If M1–M3 pass but E1 fails, the conclusion is that the forecast was necessary
but not sufficient, the oracle bound above is contradicted by the deployed
chain, and the discrepancy itself is the next object of study — not another
planner.
