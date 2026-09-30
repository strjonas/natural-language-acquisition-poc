# Probe75: cross-episode individual rate memory replicates on disjoint bodies

Date: 2026-09-30. Preregistration:
`docs/decisions/2026-09-30-cross-life-replication-preregistration.md`.
Runner: `src/homesocial/organism/cross_life_self.py`. Independent reconstruction:
`scripts/summarize_probe75.py`. Raw local artifact:
`runs/organism/probe75_cross_life_replication/replication.json`; compact evidence:
`runs/organism/probe75_cross_life_replication/evidence.json`.

**All six unchanged confirmation clauses pass.** On new bodies and episode
streams, retained rate evidence improves test survival from **0.2100 to
0.2457**: **+0.0357 [+0.0031, +0.0683]**, positive on **5/5 paired blocks**.
That is **25 additional survivors in 700 test episodes**. This confirms Probe74's
local memory benefit under the same frozen lexical/motor parent. It does not
complete the repository's self-model or language-acquisition vision.

## Locked confirmation endpoints

Intervals are two-sided 95% Student-t intervals over the five paired blocks,
using the original df=4 critical value **2.776**. No identities or episodes are
treated as independent model replications. Probe74's survey observations are
excluded from every confirmation calculation.

| clause | independently reconstructed endpoint | verdict |
|---|---|---|
| C1: persistent-identity physical ceiling | true − reset survival **+0.037143 [+0.009525, +0.064761], 5/5 positive** | pass |
| C2: retained evidence reaches survival | retained − reset survival **+0.035714 [+0.003133, +0.068296], 5/5 positive** | pass |
| C3: matched regret bridge | reset-word − retained-word regret **+0.002082 [+0.001205, +0.002959], 5/5 positive** | pass |
| C4: less rediscovery at birth | reset − retained birth-rate MAE **+0.003927 [+0.003360, +0.004494], 5/5 positive** | pass |
| C5: memory belongs to the individual | donor − own birth-rate MAE **+0.002411 [+0.002059, +0.002763], 5/5 positive** | pass |
| C6: viable construction | true-rate survival **0.247143**, floor 0.15 | pass |

Every gate uses the exact original `grade` function. The independent standard
library script reconstructs those same gates from underlying counts and score
sums and agrees with the stored runner summaries and gates. No thresholds,
calibration duration, sample size, help clock or ecology were changed after
confirmation began.

## Physical effect on the new population

Five blocks × 28 identities × five test episodes give 700 episodes per arm.
There are 140 shared calibration episodes, excluded from survival endpoints.

| rates supplied to the fixed request rule | survivors / 700 | survival | mean episode steps |
|---|---:|---:|---:|
| episode-reset recursive rates | 147 | 0.210000 | 136.06 |
| **retained individual rate evidence** | **172** | **0.245714** | **152.88** |
| true individual rates | 173 | 0.247143 | 154.44 |
| retained evidence initialized from another body | 151 | 0.215714 | 142.06 |

| block | reset | retained | true | swapped | retained − reset |
|---|---:|---:|---:|---:|---:|
| 1 | 0.271429 | 0.307143 | 0.314286 | 0.285714 | +0.035714 |
| 2 | 0.121429 | 0.135714 | 0.135714 | 0.121429 | +0.014286 |
| 3 | 0.278571 | 0.314286 | 0.328571 | 0.285714 | +0.035714 |
| 4 | 0.171429 | 0.185714 | 0.185714 | 0.164286 | +0.014286 |
| 5 | 0.207143 | 0.285714 | 0.271429 | 0.221429 | +0.078571 |

Retained memory adds 16.82 mean episode ticks. Water deaths fall from 174 to
151 and food deaths from 95 to 85; energy deaths stay at 234 and safety deaths
rise from 50 to 58. The endpoint is the net 25 additional survivors. The
survival interval is wide despite all five blocks having the same sign; its
lower bound is +0.003133.

The ungated retained-minus-swapped survival contrast is **+0.030000
[+0.005393, +0.054607]**, positive on 5/5 blocks. The swapped arm may update
and correct its donor prior; it is not frozen. Together with C5, this supports
specificity to the individual rather than a benefit from arbitrary prior-body
calibration.

True-minus-retained survival is **+0.001429 [−0.011724, +0.014581]**. This does
not establish equivalence or exhausted headroom. No evidence-budget sweep was
run, and the supplied request rule remains myopic.

## Evidence, prediction and speech

On retained-arm histories, the counterfactual panel holds the exact episodic
state belief, uptake belief, movement estimate, request ledger and time fixed.
Only the supplied rate readout changes. Birth error is measured before any new
transition or reading; regret and tick-rate error use the scored decisions
after the inherited fidelity warmup.

| endpoint on retained histories | reset-rate readout | retained own-memory readout | donor-memory readout |
|---|---:|---:|---:|
| birth-rate MAE | 0.00499226 | **0.00106531** | 0.00347678 |
| rate MAE per scored decision | 0.00083025 | **0.00039502** | 0.00156269 |
| word regret per scored decision | 0.00688038 | **0.00479859** | 0.00912294 |

Own memory removes **78.7%** of birth-rate error, **52.4%** of scored-decision
rate error, and **30.3%** of matched regret relative to episode reset. Its
rate-only replacement changes **5.81%** of scored need words. Regret is
reconstructed by summing the paired score differences within a block and
dividing by that block's total scored decisions, then averaging the five block
values. It is not an average of per-episode regret ratios.

Survival by successive test episode is:

| episode | reset | retained | true | swapped |
|---|---:|---:|---:|---:|
| 1 | 0.235714 | 0.235714 | 0.257143 | 0.221429 |
| 2 | 0.235714 | 0.257143 | 0.257143 | 0.214286 |
| 3 | 0.157143 | 0.235714 | 0.221429 | 0.178571 |
| 4 | 0.221429 | 0.257143 | 0.257143 | 0.242857 |
| 5 | 0.200000 | 0.242857 | 0.242857 | 0.221429 |

The aggregate improvement does not imply a monotonic survival curve as
evidence accumulates. Each new episode also has its own birth, layout, shocks,
reading schedule, help randomness and motor randomness.

The nominal evidence budgets remain at most **400 ticks** for reset readouts
and **2,400 ticks** for retained rates: one calibration plus five test episodes.
Actual calibration totals are **19,328 physical ticks and 571 readings**,
averaging 138.06 ticks and 4.08 readings per identity. On retained histories,
test episodes average 152.88 physical ticks and 4.56 processed readings. By
the end of test episode five, own memory carries mean cumulative evidence of
**896.44 nonterminal ticks and 26.86 readings**, including calibration.
Terminal transitions are not fitted, matching Probe74.

All arms compute the same episodic estimator, own-memory estimator,
donor-memory estimator and four rate-word counterfactuals. More relevant prior
evidence is the intended treatment, not additional per-tick estimator compute.
Wrong-identity priors match the nominal calibration episode budget but can
have different actual episode lengths and reading counts.

## Disjoint streams and verification

Confirmation identities use **3,600,000,000 + block × 2,000,000 + identity**;
episodes use **3,700,000,000 + block × 2,000,000 + identity × 16 + episode**,
where calibration is episode zero. The preceding runtime diagnostic used
3.4B/3.5B and is excluded from every confirmation endpoint. Probe74's
3.2B/3.3B survey and Probe73's 3.0B lives were not reused.

The full replication took approximately **11 minutes 40 seconds** locally.
Atomic progress contains all 140 calibrations and 560 complete identity/cell
units. Resume pins seed bands, configuration, parent weights and configuration
contents, plus all Python source under `src/homesocial`. An unfinished legacy
artifact without a source fingerprint cannot resume. A completed historical
Probe74 artifact can be re-summarized without new episodes or assigning it an
unverified modern source hash.

Reproducibility fingerprints:

- Parent weights and sibling configuration SHA-256:
  `ddd7c5a9e1cdb4274a63a98c078065e55702b6116457c1cbc659a2b8c32135d4`.
- Mechanism package SHA-256:
  `d7dcad13136215ca79546dfdf75f7778105dc42651e5e240b8d05ea21341f4de`.
- Completed raw artifact SHA-256:
  `85365d1a17d22f66439fb0e72ad4f6c3eaf8ddced32bf9ba65d2e4f978d83d24`.

The independent summary imports no `homesocial`, NumPy or MLX code. Against
the raw artifact it checks completion, 140 calibration records, 560 distinct
identity/cell units, five episodes per unit, every expected episode seed,
cumulative evidence arithmetic and all stored summaries/gates. Its compact
evidence preserves configuration, fingerprints, actual counts, calibration
totals, per-block underlying score sums and per-episode survivor counts.
Recalculation from that approximately 80 KB record also matches every endpoint.
The compact path checks block sums; individual seed checks occurred when the
raw record was reduced and cannot be repeated from omitted episode records.

Artifact verification has **18 dedicated tests**, including unequal episode
lengths, duplicate/incomplete records, incorrect seeds, inconsistent evidence,
non-finite values, altered stored summaries, and compact reconstruction. Real
learner updates are guarded against any simulator-property read, and a small
real interrupted survey must exactly match uninterrupted execution after
resume.

```bash
# Recalculate from committed compact evidence with standard Python only.
python3 scripts/summarize_probe75.py

# On a local checkout containing the complete raw artifact, also validate
# every individual episode record and regenerate the compact evidence.
python3 scripts/summarize_probe75.py \
  --input runs/organism/probe75_cross_life_replication/replication.json \
  --out runs/organism/probe75_cross_life_replication/evidence.json

# Re-run the unchanged construction on its preregistered confirmation bands.
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.cross_life_self \
  --identity-seed-base 3600000000 --episode-seed-base 3700000000 \
  --out runs/organism/probe75_cross_life_replication/replication.json
```

## Claim boundary and next direction

The result now replicates a narrow causal claim: retaining relevant evidence
about the same stationary simulated body's rates improves its next-episode
birth estimate, changes speech, lowers same-history regret and improves
survival on a disjoint population and episode stream. Episode current-state
and uptake beliefs still reset, as do the ledger, movement estimate and motor
hidden state.

It remains one shared frozen lexical/motor parent. The organism does not
discover its axes, recognize identity, model bodily changes, speak uncertainty
or generate language. Memory assignment is maintained externally. Its motor
policy does not read the self-model. Body structure, causal parameterization,
request rule and three need words are supplied. Resetting the simulated body
after death is not biological persistence through death. No consciousness or
sentience claim follows.

This closes the immediate replication debt for persistent rate memory under
this parent, not the full parent-development or architecture debt. Next work
should confront a remaining property with a distinct controlled mechanism,
using this confirmed memory result as a baseline. An evidence-budget sweep,
changing-body study or motor-use extension would each require its own
preregistration and relevant oracle/lesion checks; the present result does not
substitute for them.

Reproduction note after later source additions: the confirmation pins the
whole package, so its completed local output intentionally refuses a new source
fingerprint. Recalculate historical evidence with `scripts/summarize_probe75.py`.
For a fresh run under the unchanged rate mechanism use the fixed wrapper and a
new output path:

```bash
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.cross_life_replication \
  --out /tmp/probe75-replication.json
```
