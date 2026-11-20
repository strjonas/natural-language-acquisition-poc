# STATE

Last guidance update: 2026-10-21. Experimental frontier: 2026-10-05.
Latest build: 2026-11-20 (development; a budget experiment is ready to run, no
confirmatory result). Rewrite this file, never append.

## Where the next agent should start

Read [CLAUDE.md](../CLAUDE.md) first for current working rules. The
[2026-10-21 method decision](decisions/2026-10-21-integration-and-budget-guidance.md)
balances small hypothesis tests with integration and bounded scale experiments.
The [research review](decisions/2026-10-21-guidance-research-review.md) shortens
those rules and clarifies the target and evaluation. Neither changes any
experimental result.

The programme changed direction on 2026-10-04/05: **the self-model now chooses
the organism's actions.** For that frontier, read:

1. `docs/decisions/2026-10-04-self-in-the-action-loop-direction.md` — why
   twenty probes bought a few survival points each, the mechanism family, the
   development pilots, and the ladder (Probes 80–84).
2. `docs/decisions/2026-10-05-active-self-discovery-result.md` — **Probe80:
   all nine locked gates pass**, independently reconstructed.
3. "Choosing the next tranche" below: candidate questions and their actual
   dependencies, rather than a mandatory sequence by probe number.

`md/archive/DIRECTION_2026-07-26.md` remains the destination (the terminal
claim and the five properties). Everything about the request-word line
(Probes 57–79) below is unchanged record.

## Current capability and next decision

**Next capability target: acquired, productive communication used by the same
learner that predicts and regulates its body. Not yet demonstrated.** This is
an intermediate milestone from the original vision, not completion of all five
self-model properties and not a choice of architecture.

The intended demonstration is one learner retaining experience of its body,
learning caregiver words through situated interaction, and producing a useful
new combination of known words when its needs require it. Its bodily knowledge
must influence both action and communication. A scored structure/confidence
readout or a supplied request lookup does not by itself meet this target.

Before confirmation, specify a versioned scenario and numeric endpoints for
learning, withheld combinations, bodily outcomes and the predicted effects of
self-information and communication interventions. Preserve comparison with the
development baseline. This evaluation is **not built**; the commands below test
only the existing nonlinguistic component. No new thresholds are implied here.

The runnable Probe80 path is `active_self.run_arm` connecting `SelfPosterior`
to `LivedBody`:
action, prediction, retained bodily evidence and a scored structure/confidence
readout operate together. It has no listener, generated language or navigation.
The spatial request-word organism is a separate baseline; its capabilities do
not automatically transfer into this path. No new integration was built by the
2026-10-21 documentation update.

Reconstruct the existing result without retraining:

```bash
python3 scripts/summarize_probe80.py --input runs/organism/probe80_active_self/evidence.json
```

For a small runnable diagnostic (exploratory, not confirmation):

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.active_self --mode diagnostic --block 0 --k 2
```

### Built 2026-11-20, not yet run to its budget

Record: `docs/decisions/2026-11-20-learned-self-budget-experiment.md`.

**First missing connection:** no learner in the repository could both find out
its body and take in words; Probe80's posterior is hand-derived for its physics.
Two development pieces now address that. Neither is a capability yet.

1. **A learned self on Probe80's body family** (`learned_self.py` on the exact,
   vectorized physics in `vec_body.py`). One GRU trained across bodies gets the
   same lived information as `SelfPosterior` and uses one recurrent state to
   act and to predict its own sensations. A six-minute sanity run reached
   survival 0.89 / 0.48 / 0.20 at K1-K3 (random 0.45 / 0.14 / 0.02; Probe80's
   posterior 0.98 / 0.86 / 0.68) and was still rising. **The sustained run has
   not been done.** Untested: whether it approaches the posterior, at what
   budget, and every control except the memory-reset lesion.

   ```bash
   PYTHONPATH=src caffeinate -i .venv/bin/python -m homesocial.organism.learned_self train --out-dir runs/organism/learned_self_dev --max-hours 9
   PYTHONPATH=src .venv/bin/python -m homesocial.organism.learned_self eval --out-dir runs/organism/learned_self_dev
   ```

2. **A ceiling check for words** (`caregiver_world.py`, scenario
   `caregiver-v0`). With oracle speakers, the resource word is worth +0.42 to
   +0.90 survival and a wrong word equals silence, but the size word adds only
   +0.02 to +0.07 and does not depend on the body's own rates. **v0 would teach
   a one-slot vocabulary; do not train a learner for productive speech in it.**

**Next action:** run piece 1 to its budget (about nine hours) and read the
curve in `runs/organism/learned_self_dev/curve.jsonl`. If survival nears the
posterior's at K1-K3, give the same learner words and design the confirmatory
comparison on a new band. If it is flat and clearly below, test one named
limitation at a time or keep the posterior as the belief and learn only the
speech. In parallel, find a `caregiver-v1` whose second slot passes the same
ceiling survey before any language training. Change detection, K5 and
navigation are not automatic prerequisites. Keep later candidate questions
below as background.

## The direction in one paragraph

In the request-word line, the self-model primarily affected speech while a
frozen policy chose motor actions. Probe80 instead uses its self-model to choose
actuators and establishes large gains over its tested controls. These different
experiments support investigating the action connection; their effect-size
difference does not isolate that connection as the sole cause of progress.
Nor does beating Probe80's action-value bandit establish superiority over a
trained alternative with history-dependent decisions. The mechanism
(`src/homesocial/organism/self_posterior.py`) is an online posterior over the
organism's own unnamed structure — sampled discrete structure, exact Gaussian
self-parameters, truncation for implied inequalities — that acts by sampling
a self consistent with the evidence, reports its structure with a confidence,
and **checks itself against its own sensations**: when they are impossible
under it (below −30 nats in one tick) it withdraws its confidence and rebuilds
from remembered experience. Words enter later as social actuators over the
variables it discovered in the proposed Probe82 work. This is a promising
starting point, not proof that the posterior is the eventual architecture.

## Probe80: a self-posterior discovers its own body while living in it

**9/9 locked gates pass**; 300 identities × 9 arms × 6 lives; survey band
5,000,000,000; runner and standard-library checker agree on every gate.

| K | random | structure-free | **posterior** | reset | certainty | yoked | true structure | exact structure, life 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.436 | 0.969 | **0.978** | 0.864 | 0.978 | 0.225 | 1.000 | **1.00** |
| 2 | 0.164 | 0.417 | **0.864** | 0.661 | 0.756 | 0.047 | 0.906 | **1.00** |
| 3 | 0.047 | 0.039 | **0.675** | 0.469 | 0.583 | 0.006 | 0.747 | **0.93** |
| 4 | 0.017 | 0.006 | **0.472** | 0.297 | 0.336 | 0.000 | 0.589 | **0.67** |
| 5 | 0.006 | 0.000 | 0.272 | 0.158 | 0.253 | 0.003 | 0.558 | 0.25 |

Survival is the six-life mean. Gates (paired over five blocks, all 5/5):
posterior beats random at K1–K4 (+0.46 to +0.70) and the structure-free learner
at K2–K4 (+0.45 to +0.64; K1 ties, as predicted: one hidden resource needs no
structure); another body's evidence is catastrophic (+0.678; yoked recovers
0.4% of its own structure); retained self-knowledge pays across lives
(+0.202); uncertainty drives discovery (+0.879 exact over the certainty
lesion); self-reports are calibrated at K1–K4 (Brier 0.053; 96.5% correct at
confidence ≥ 0.9 over 1,161 reports; 1.7% at ≤ 0.1); self-criticism is what
calibrates them (uncritical Brier 0.127, +0.075 [+0.044, +0.106]); relabeling
transports 100/100. K5 is the declared stress condition: discovery 0.25,
overconfident, survival still above random.

Not established: self-criticism's survival effect (+0.034, lower bound
−0.0005). Not shown: language, change, space, bodies outside the authored
family.

## Development pilots toward Probes 81 and 82 (not evidence)

Recorded in full in the direction document, Section 3.

- **Change**: after a hidden swap of two large actuators, the self-critical
  posterior that concludes "my body changed" when no rebuild of its past
  survives its own replay recovers the new structure (4/4); the uncritical one
  stays confidently wrong (confidence 1.0); stationary twins concluded change
  0/4 times (Probe79: 17% of stable bodies). A *small*-alias remap is never
  used again and never noticed: detection and staleness are separate problems.
- **Words**: one discovered resource can be restored only by a caregiver's
  word. Labels from a caregiver who sees the body speed acquisition (K2
  first-life survival 0.62 against 0.12 silent; report fidelity 0.97 against
  0.76). The pilot's own scrambled-label control leaked which words matter for
  this body — Probe82 must scramble reference while preserving vocabulary
  statistics, and add a labels-without-self-model arm.

## Where the five properties now stand

From `md/archive/DIRECTION_2026-07-26.md` section 2, updated after Probe80.

1. **Discovered** — *advanced by Probe80*. The count of hidden resources and the
   actuator partition are inferred online from two scalars, without forks, with
   shocks and death, across K1–K4 (exact 1.00, 1.00, 0.93, 0.67 after six
   lives). Within an authored family (bounded, depleting, saturating) and with
   the sensor width given; K5 0.25.
2. **Corrigible** — **yes** for evidence integration (Probe63). Autonomous
   correction after hidden change: Probe78's oracle timing passes, Probe79's
   detector fails; Probe81 will test self-criticism as the detector.
3. **Load-bearing across uses** — *advanced by Probe80 in the actuator
   substrate*: one posterior acts, predicts and reports, and five lesions each
   cost a named endpoint. Still nonspatial; the island organism's motor parent
   does not read any self-model (Probe84).
4. **Productive under novel demand** — *partial*, unchanged since Probe66. Words
   for discovered variables are piloted, not tested.
5. **Reflexive** — *advanced by Probe80*: calibrated confidence about its own
   structure, caused by acting on a fact about its own model (its sensations
   were impossible under it); the lesion loses it.

## Evidence ladder and closed lines

| Probe | Mechanism | Result |
|---|---|---|
| 48–56 | neural mouths, COMA critics, recurrent self-belief, ranked belief | all **fail** at 80,000 ticks; see probe69 |
| 57 | structured learned causal self + social planner | **all local gates pass** |
| 58 | causal-stage replication, 5 seeds | **5/5 pass**, sd <= 0.009 |
| 59 | online adaptation after a body-rule change | recalibration 5/5; survival gates **fail** |
| 60 | outcome-aware realized-consequence planner | planner repair passes; load-bearing gate **fails** |
| 61 | discovered bodily structure from two sensations | **4 of 5 fail**; closed |
| 62 | self-uncertainty ceiling survey | posterior **ties** the point filter; closed in its ecology |
| 63 | individual body + online self-calibration | **7/7 pass**; standing obstacle closed |
| 64 | portion requests under a rationed caregiver | **7/7 pass**; **survival hears none of it** |
| 65 | a caregiver that answers eighteen ticks late | **9/9 pass**; ordering *reverses* with the horizon; **a belief reaches survival** |
| 66 | a caregiver whose size words do not mean one thing | **6/6 pass**; factorization **found rather than given** |
| 67 | acting on the width of the rate belief | **survey, no**; ceiling arm loses to baseline |
| 68 | the grain of the decision surface | **4/5 pass, G5 fails**; the floor is a **formula**, the grain is **pinned by viability** |
| 69 | structured against black-box, swept over budget | **6/7 pass, G6 fails**; structure is **sample efficiency, not a ceiling**; the families **tie** at 64x |
| 70 | oracle-rate lifetime basket allocation | **ceiling, no**; loses to first-come at all six stores, 30/30 paired cells |
| 71 | fixed K5 body + first-use composition law | **construction ceiling, 4/6**; body/law viable, range and axis-gap gates **fail** |
| 72 | compensated quantum, clean rate/state survival contrasts | **3/5**; true rates buy +0.0214 survival, but the granularity interaction is **-0.0229, 2/5 positive** |
| 73 | learned rate against a matched species-rate lesion | **5/5 pass**; learned rates buy **+0.0143 survival**, true rates +0.0386 |
| 74 | retained rate evidence across resets of one fixed individual | **survey, 6/6 pass**; retained rates buy **+0.0300 survival**, 5/5 blocks; wrong-identity memory loses the benefit |
| 75 | unchanged persistent-memory mechanism, disjoint bodies/episodes | **confirmation, 6/6 pass**; retained rates buy **+0.0357 survival**, 5/5; independently reconstructed |
| 76 | scalar saturation interactions identify unnamed restoration groups | **instrument, 6/6 pass**; exact 25/25 worlds, held-out 750/750; matched-fork and shock-free privileges |
| 77 | random sequential predecessor-effect contrasts | **0/5 pass** at one and six episodes; inadequate grouping and pair coverage; closed |
| 78 | oracle-timed reset after hidden body changes | **3/3 clauses pass**; +18 survivors/700, exact stable-sham parity; timing remains privileged |
| 79 | two consecutive observable prediction errors trigger reset | **1/3 pass**; stationary false alarms and sign-scrambled controls fail; closed before survival treatment |
| 80 | self-posterior in the action loop: discovered structure, posterior-sampling action, self-criticism | **9/9 pass**; exact structure 1.00/1.00/0.93/0.67 at K1–K4; calibrated self-report (Brier 0.053); +0.45 to +0.70 survival over random and structure-free |

The following closures retain their original verdicts in the tested settings.
Do not repeat them unchanged or move their gates. A materially different task,
budget or integration may justify a new experiment with an explicit rationale;
these entries are not prohibitions on entire architecture families. See
CLAUDE.md for the development/confirmation distinction.

Closed lines:

- report entropy, head width, vocabulary size, replay, sparse-return loss
  weights, horizons as a tuning knob, counterfactual token-credit variants;
- more guided lexical exposure after comprehension reaches ceiling;
- **black-box recurrent self-belief width/loss/head variants** — *restated, not
  overturned*: probe69 moved **budget**, and its §5 shows head variants were the
  wrong lever, which is why they all failed;
- a larger legacy neural mouth on the same state;
- **assuming compute alone repairs the request-word system's limitations** —
  Probe69 does not establish this. It does establish that the tested neural
  estimator approximately ties the structured one at 5,120,000 ticks. Bounded
  budget experiments are allowed; neither an architectural ceiling nor success
  in action or language follows from that estimator comparison;
- probe61's structure discovery **mechanism** (a sparsity-penalized latent);
- stacking probe64's `caregiver_store` onto probe65's lag;
- acting on the rate posterior through probe60's objective, at any horizon;
- lowering the grant without compensating the period;
- the mixed `individual − state_oracle` contrast as a gate;
- **fixed per-need lifetime basket quotas from metabolic rates**, whether the
  rates are species, oracle or permuted, and moving the store within 22–34 to
  rescue them; probe70's oracle arm loses 30/30 paired cells.
- **retraining the parent or building a composition treatment on Probe71's fixed
  K5 body**; its natural range is 0.02570 against 0.030 and axis-gap movement is
  -0.01707 against 0.020, so the construction is not licensed. Moving a rate,
  clock, portion, birth
  distribution or threshold is a new ecology, not a repair.
- **halving the help quantum and caregiver period at fixed expected nutrition as
  a way to amplify the survival value of individual-rate knowledge** in the
  fixed lag-18 ecology. It amplifies open-loop accuracy value (probe68) and
  raises every arm's survival, but the clean survival rate step is -0.0229
  (probe72).

Probe77's random sequential grouping and Probe79's fixed threshold-and-streak
detector are also closed at their preregistered budgets. Positive sub-endpoints
do not license threshold tuning or a detector survival treatment.

Do not carry self priors between independently redrawn bodies. Probe74 keeps
the physical individual fixed and explicitly tests a wrong-identity prior.
Do not cite its aggregate true/retained tie as saturation or equivalence; its
paired interval spans +/-0.0217 and the evidence budget was not swept.

**New standing requirement from probe69:** any claim that one mechanism family
beats another must state the **budget** it was measured at, and must not be cited
as a ceiling unless the budget was swept.


Inference variants tried and rejected **during Probe80's development** (not
closed lines; recorded so they are not re-tried blindly): a structural change
hazard of 0.02/0.05, island SMC (4 × 250), moment-matched shock attribution,
looser sensor tolerance, rarer resampling, and N = 400 particles at K4.

## Choosing the next tranche

The following are candidate research questions, not a mandatory execution order.
Plan integration into a shared runnable path as work proceeds, rather than
waiting for every item to pass. An isolated probe remains appropriate when its
answer can avoid a costly mistake or unblock that path. Preserve its controls
and claim boundaries. Review the allocation after each bounded tranche.

1. **Probe81 — does the organism notice that its own body changed, and only
   then?** Relevant to changed-body claims; not inherently a prerequisite for
   language in stationary bodies. A candidate mechanism is self-criticism plus
   "conclude change when no rebuild of the past survives its own replay". Requirements
   from the pilot: a change that touches what the organism does (large-actuator
   swap), stationary twins for false alarms with Probe79's 17% as the number to
   beat, the uncritical and keep-best lesions, an oracle-timed restart as
   reference, and a rebuild that does not replay the whole remembered past
   (cost grew to 11 rebuilds per identity). Staleness of untested knowledge is a
   separate, later question.
2. **Probe82 — words for a self it discovered.** One discovered resource only a
   caregiver can restore; words as social actuators; labels from a caregiver who
   sees the body as evidence about the agent's own discovered variables.
   Controls: reference-scrambled labels that preserve vocabulary statistics,
   labels without a self-model, scrambled listener. Endpoints: survival,
   acquisition speed, report fidelity of "what do you need?".
3. **Probe83 — productive and future speech** over discovered variables: a
   compositional caregiver protocol (which × how much × when), held-out
   combinations, and Probe65's counterfactual-branch test for "I will need".
4. **Probe84 — spatial integration**: the self-posterior as the spatial
   organism's motor model, and independently developed parents (the old Probe52
   debt). This is distinct from integrating action, prediction and communication
   earlier within a simpler world. Neither an actuator demonstration nor
   downstream seed replication establishes the full developmental system.
5. **Budget and architecture experiments**: longer training, more varied
   experience, capacity or inference compute may be necessary. Amortizing the
   posterior with a learned network is one hypothesis, not a selected solution.
   Probe69 motivates budget sweeps; it does not validate distillation, shared
   neural control or language. Specify the suspected limitation, comparable
   baselines, held-out development curves, resource cap and checkpoint schedule.

Carry-over items from the request-word line that remain open: score Probe66's
declining against survival; run Probe61's missing controls if F0 or G2 are ever
cited; replicate the Probe52 childhood-to-adult pipeline.

## On `docs/VISION_AND_STATUS.md`

That document (2026-08-10) argued for a clean v2 and named the argmax mouth as
the largest gap. The new direction takes its first advice — build the new
learner cleanly beside the old line, which stays as microscope and baseline —
without its architectural bet: Probe80's learner is a structured posterior, not
a recurrent latent and token decoder, because structure buys sample efficiency
(Probe69) and calibrated self-report is what the reflexive property needs. Its
diagnosis of the mouth still stands and is what Probes 82–83 are for.

## Record of the request-word line (Probes 63–79, unchanged)

### Probe75: independent persistent-memory confirmation

Same exact mechanism and six gates, 5 × 28 identities × 5 test episodes,
700 episodes per cell, with new 3.6B identity and 3.7B episode bands. All six
original gates pass. Reset/retained/true/swapped survivors are **147/172/173/151**.
Retained minus reset is **+0.035714 [+0.003133, +0.068296], 5/5**, adding 25
survivors. True minus reset is +0.037143 [+0.009525, +0.064761], 5/5.
Retained minus swapped is +0.030000 [+0.005393, +0.054607], 5/5, ungated.

On retained histories, birth-rate MAE falls 0.004992 → 0.001065, matched
regret 0.006880 → 0.004799, and 5.81% of need words change. C3 regret value
is +0.002082 [+0.001205, +0.002959]; C4 birth-error value +0.003927
[+0.003360, +0.004494]; C5 own-versus-donor identity value +0.002411
[+0.002059, +0.002763], all 5/5. True survival 0.247143 passes the 0.15 floor.

One shared calibration episode per identity is excluded from survival gates.
Relevant evidence history differs intentionally; per-tick compute is matched.
Retained histories finish with 896.44 evidence ticks and 26.86 readings on
average. True minus retained survival is +0.001429 [-0.011724, +0.014581],
which does not establish equivalence or saturation. Bodies are stationary,
identity assignment is external, and the frozen parent is shared.

`scripts/summarize_probe75.py` independently reconstructs all summaries/gates
from raw episode records, or committed compact block sums when raw output is
absent. The latter cannot re-inspect omitted episode seeds. Resume now pins
source as well as parent/configuration and refuses changed partial runs.
The real completed confirmation resumed without replaying episodes; real
interruption and learner-update truth-denial guards also pass.

### Probe76: causal overlap as a different structure question

**All six locked gates pass.** Ordinary unsaturated mean trajectories collapse
several axes into one scalar effect. Saturating a resource changes whether
another restoration helps it: matched scalar intervention interactions can
therefore reveal causal groups without axis labels, state vectors, a dimension
penalty or a supplied target map. Five seed blocks × K1–K5 × 40 development
and 40 held-out contexts, fixed opaque ten-actuator catalog and mean denominator.

This instrument deliberately permits matched forks and removes shocks. It
retains real portion sizes, six-tick requests and next-tick absorption. Six
locked gates require count/partition recovery, held-out overlap prediction,
permutation invariance, destroyed sensory pairing and physical causal remapping.
An unreachable-axis challenge must leave total bodily dimension unresolved.
Exact reachable counts and partitions are recovered in **25/25 worlds**, with
mean counts **1, 2, 3, 4, 5**. Held-out causal overlap is **750/750 correct**;
ID permutations transport all partitions. Shuffled sensory pairing destroys
the exact active-set/partition in 25/25, collapsing every model to one group.
All 20 physical remappings invalidate the stale grouping and are recovered;
remapped held-out overlap is **700/700 correct**. The twenty unreachable-axis
challenges recover K reachable groups while K+1 bodily variables remain live:
total dimension is explicitly unresolved.

Actual budget: 4,400 contexts, 532,400 unique branches, 6,921,200 physical ticks;
53.46 seconds. Independent reconstruction verifies all raw branch sensations
against physical formulas (maximum error 8.88e-16), all six gates, and 14,450
distinct derived RNG seeds. A pre-run seed collision was corrected before any
diagnostic/survey generation without changing physical distributions or gates.
Real resume regenerates no completed world.

This licenses online exploration without the matched-fork and shock-free
privileges. It does not finish discovered state, produce language or show a
survival gain. Groups describe restorative causal overlap in an authored
bounded-resource family; no state estimator or motor-use mechanism is built.

### Probe74: rate memory belongs to one persistent individual

Five paired blocks x 28 identities x five test episodes = 700 episodes per arm,
after one shared calibration episode per identity. All four metabolic
multipliers, including safety, stay fixed across resets; birth, shocks, reading
times, help and motor randomness are redrawn. All cells compute an episodic
state/uptake calibrator plus own-memory and donor-memory calibrators. Only the
rate vector passed to the fixed request rule changes.

| rates used by request rule | survival | mean episode steps |
|---|---:|---:|
| episode-reset recursive | 0.2271 | 150.2 |
| **retained recursive** | **0.2571** | **165.1** |
| true individual | 0.2571 | 167.0 |
| retained from another body | 0.2257 | 146.8 |

**All six continuation gates pass.** Retained minus reset survival is +0.0300
[+0.0141, +0.0459], 21 additional survivors, positive 5/5. True minus reset is
+0.0300 [+0.0054, +0.0546], 5/5, establishing the physical precondition in this
new identity construction. Retained minus swapped survival is +0.0314
[+0.0213, +0.0415], 5/5, reported as an ungated control contrast.

On retained histories, birth-rate MAE falls 0.004734 -> 0.001010 (**78.7%**),
same-history regret falls 0.005500 -> 0.003654, and **5.33%** of scored need words
change. Regret value is +0.001846 [+0.000222, +0.003471], 5/5. Donor memory has
+0.002788 [+0.001839, +0.003736] more birth-rate MAE than own memory, 5/5.
The lag-24 granularity diagnostic measured margin 0.13836 before the survey.

The budget is explicit: episode reset can use at most 400 ticks; retained rates
can use one calibration plus five test episodes, at most 2,400. On retained
histories the final cumulative mean is 969.1 nonterminal evidence ticks and
28.39 readings; calibration averaged 149.7 physical ticks and 4.35 readings.
Compute is matched, evidence history intentionally differs. Episodes within an
identity are dependent, and the five paired blocks are the interval units.

**The aggregate true/retained tie is not saturation.** True minus retained is
0.0000 [-0.0217, +0.0217], with both signs across blocks; no evidence-budget
sweep was run. This does not establish equivalence or exhaust Probe73's old
headroom. The memory assignment to bodies is maintained externally, metabolism
is stationary, and reset after simulated death is not biological persistence.
The motor policy still does not read the self-model and the mouth still selects
three supplied need words. The result remains a local survey, all blocks still
sharing one parent.

### Probe73: a learned bodily constant reaches survival

Five paired seed blocks x 140 lives per cell, 700 lives per arm, lag 24. Every
cell runs the same recursive calibrator, filtered state, uptake belief, request
ledger, frozen motor parent and planner compute. Only the rate vector exposed to
the request rule is species, learned or true.

| rates used by request rule | survival | mean life steps | matched regret | rate MAE |
|---|---:|---:|---:|---:|
| species | 0.1743 | 126.1 | 0.01097 | 0.00355 |
| **learned** | **0.1886** | **131.3** | **0.00481** | **0.00065** |
| true | 0.2129 | 145.5 | — | — |

**All five gates pass.** Learned minus species survival is **+0.0143
[+0.0003, +0.0283]**, with four positive seed blocks and one exact tie. True
minus species is **+0.0386 [+0.0213, +0.0559] on 5/5**, so the construction has
matched-state physical headroom and the learner captures about 37% of it. On the
learned histories, species-rate minus learned-rate regret is +0.00616
[+0.00421, +0.00811] and species-rate minus learned-rate MAE is +0.00290
[+0.00262, +0.00317], both 5/5. The rate-only counterfactual changes 27.5% of
need words.

This is the first clean survival gate on the learned bodily rate. It advances
load-bearing across uses: one recursive estimate now changes future prediction,
speech/regret and survival together. It does not finish the property because the
frozen motor policy does not read the self-model, and nothing about the body,
planner or vocabulary is discovered. The result adds ten survivors in 700 lives
and its lower interval bound is only +0.0003; do not inflate the magnitude.

The remaining true-minus-learned survival value was +0.0243 [+0.0164, +0.0322].
It motivated Probe74's persistent-identity construction. That survey now passes
its physical and identity-specific memory gates, but its differently sampled
bodies are not a paired estimate of how much of Probe73's gap was recovered.

### Probe72: a word-level amplifier does not amplify survival value

Five paired seed blocks x 140 lives per cell, 700 lives per arm, lag 18. The
baseline and fine ecologies deliver exactly 0.0666667 expected grant per tick;
the fine ecology halves both portion size and caregiver period. Three existing
arms isolate rate knowledge from state knowledge.

| ecology | oracle | state oracle | population | rate value | state value |
|---|---:|---:|---:|---:|---:|
| baseline | 0.3643 | 0.3200 | 0.2557 | +0.0443 | +0.0643 |
| fine | 0.5057 | 0.4843 | 0.3914 | **+0.0214** | +0.0929 |
| fine - baseline | +0.1414 | +0.1643 | +0.1357 | **-0.0229** | +0.0286 |

**G1 fails.** The rate-value interaction is -0.0229 [-0.0869, +0.0412],
with only 2/5 blocks positive against the locked 4/5 requirement. Probe68's
open-loop +0.0855 on 8/8 seeds therefore does not carry to survival. **G2 fails
with it:** state value moves +0.0286 while rate value moves negatively.

The negative is not collapse or lack of signal. Oracle survival is 0.3643 and
0.5057, nutrition per tick is exact, and **G3 passes**: in the fine ecology,
true individual rates over species rates buy **+0.0214 [+0.0089, +0.0340] on
5/5 blocks**, with current state perfect in both arms. Individual-rate knowledge
can reach survival here; making the decision grain finer did not increase its
value. All three arms gain 13.6–16.4 survival points from the finer clock, and
`state_oracle` gains most.

This closes the compensated quantum as a survival-value amplifier in the fixed
lag-18 ecology. It does not close the separate learned-rate question. No
self-model property advances because the positive endpoint is an oracle ceiling,
not the recursive learner.

### Probe71: a viable larger body that misses its construction gate

Five seeds x 100 lives, no learned component. K5 makes the existing safety axis
reportable and adds health, growing the natural message space from six to ten.
The fixed portions, six-tick caregiver, 400-tick life, shocks and individual
spread were not retuned.

| body | oracle survival | health request share | median axis gap | predicted rent | live rent |
|---|---:|---:|---:|---:|---:|
| K3 | 0.950 | — | 0.08325 | 0.01400 | 0.02245 |
| K5 | **0.976** | **0.206996** | **0.06618** | **0.02345** | **0.02680** |

The parameter-free first-use law is correct on the process it names: aggregate
formula versus i.i.d. resample is 0.01400 versus 0.01380 in K3 and 0.02345
versus 0.02318 in K5. Per-life MAE is 0.00697 and 0.00350, both under the locked
0.015. Fixed-count permutation exposes a separate homeostatic-order residual;
it is larger in K3 and is not fitted away. Health carries 20.70% of requests, so
the fifth axis is not decorative. The correctly pooled natural predictor range
is **0.02570** against a gate of 0.03, so C5 fails.

**C6 also fails.** The pooled gap step is **-0.0170660** against a
preregistered absolute movement of 0.020. Direction right, 85.3% of the required
magnitude, still a failure. The continuation rule requires all six clauses, so
no expanded parent is trained and no factorized-versus-enumerative treatment is
built. The fixed K5 construction closes as a bridge. No self-model property
advances.

### Probe70: constitution is not realized demand

Five seeds x 40 lives, same oracle requester in every arm, stores 22–34 plus the
unlimited inertness control. The caregiver divided its lifetime basket by either
this body's oracle rates, species rates, or the oracle rate values assigned to
the wrong need labels.

| store | first-come | oracle rates | species rates | permuted rates | oracle − first |
|---:|---:|---:|---:|---:|---:|
| unlimited | 0.715 | 0.715 | 0.715 | 0.715 | **0.000 exact** |
| 34 | 0.695 | 0.575 | 0.385 | 0.235 | **−0.120** |
| 30 | 0.655 | 0.465 | 0.300 | 0.185 | **−0.190** |
| 28 | 0.630 | 0.425 | 0.250 | 0.160 | **−0.205** |
| 26 | 0.525 | 0.410 | 0.235 | 0.145 | **−0.115** |
| 24 | 0.460 | 0.335 | 0.205 | 0.120 | **−0.125** |
| 22 | 0.375 | 0.270 | 0.155 | 0.110 | **−0.105** |

All **30/30** paired finite-store contrasts are negative; every 95% paired
interval excludes zero in the harmful direction. The rates are informative —
oracle allocation beats both lesions everywhere — and still insufficient. At
store 30 it gains +0.010 useful uptake per basket unit while losing 0.190
survival, refusing 2.26 grants per life against 0.44 first-come and leaving 1.573
more store mass unspent. Water deaths rise 0.015 → 0.210.

The refusal effect is not swallowed by probe68's grain. Mean counterfactual
capacity is 0.498 per refusal at store 30, max 0.60, against a rerun lag-0 median
margin of **0.08485** (formula MAE 0.000610). The world hears the allocation and
hears it negatively.

> **A constitution is a prior over demand, not realized demand.** The live request
> stream already integrates action, shock timing, current state, uptake, clipping
> and earlier grants. A fixed rate partition discards that feedback and strands
> resources in accounts that cannot lend them.

The fixed lifetime-share constitution mechanism is closed before language is
built. A dynamically rebalanced allocator is a new mechanism and must first name
what it can do that first-come does not already do. No self-model property
advances.

#### The bias probe69 was built to correct

Probe68 found that since probe59 every probe had held the ecology fixed and
varied the mechanism. Probe69 found the other half of the same blind spot:

> **Every comparison in this repository between a structured mechanism and a
> black-box one was run at exactly one compute budget — 80,000 ticks — and that
> budget is the point of maximum separation between the two families.**

Measured on this machine, 80,000 ticks is **90 seconds**.

### Probe69: structure is sample efficiency, not a ceiling

Five seeds, matched instrument (`_balanced_accuracy` over `world.lowest_need()`,
the same function object both audits import), 200 audit lives, 24.4 hours over
51 cells.

| budget | structured | black-box w64 | black-box w256 | gap (w256) |
|---:|---:|---:|---:|---:|
| 80,000 | 0.9234 | 0.5906 | 0.7447 | **0.1788** |
| 320,000 | 0.9338 | 0.7818 | 0.8390 | 0.0948 |
| 1,280,000 | 0.9330 | 0.8695 | 0.9028 | 0.0301 |
| 5,120,000 | 0.9351 | — | **0.9349** | **+0.0003** |

**The structured model is saturated at the locked budget** — 16x the compute buys
+0.0095, non-monotonically. **The black-box is not saturated until it has caught
up** — +0.1582 [+0.1419, +0.1744] on 5/5 seeds, with the gap collapsing
monotonically on every seed and reaching a tie at 5,120,000.

> The structured advantage is a **sample-efficiency** advantage and it is large —
> 0.9234 in ninety seconds against roughly **64x** the compute for the same
> number. It is **not a ceiling**. The two families tie at 0.935.

Both halves matter and they point opposite ways. A budget-constrained programme
should build the structured mechanism. **A programme that intends to scale cannot
cite this repository's evidence ladder as a reason not to.**

**G6 failed.** The survey fixed a gap-closure rate of 0.515 per 4x and predicted a
residual gap of 0.025 with no crossing; measured +0.0003 with one seed of three
crossing. Direction right, magnitude wrong — and the error is diagnosable: the
ratio was fitted on one seed whose last two points were already inside the noise
floor (structured per-seed sd at 1.28M is **0.0153**, fifty times the difference
being extrapolated). **A ratio fitted between two points that are both inside the
noise cannot constrain a third.** It is reported as a failed prediction, not
folded into G1–G5, and it is *not* evidence of crossing — at that spread the
honest word is **tie**.

**Probe55's readout bottleneck was a small-budget artifact.** `head − ridge` runs
−0.1349 (w64@80k) → +0.0121 (w64@1.28M) → **+0.0397** (w256@5.12M): the head
catches the linear readout of its own state and passes it. Probe55's diagnosis
was right and its inference was not — the limit was an under-trained head, not an
architecture, which is consistent with every head variant the line was closed on
having failed.

**Controls hold everywhere.** Zero and shuffled lesions <= 0.3368 against chance
0.3333 in all 51 cells, including where accuracy is 0.935. And G7: the instrument
reproduces probe54's published 0.6219 to within 0.031 on a different band, so
these cells are commensurable with the record.

#### What probe69 explicitly does not show

**Belief-side only.** It does not show a better belief reaching a word, and
probe68's margin law says it need not. **A black-box that ties the structured
model on accuracy inherits probe68's problem whole.** No property advances. The
mouth is still `argmax` over three scores.

### Probe68: the grain of the decision surface (unchanged, still binding)

    margin  =  E_grant [ min( grant * uptake, g ) ]

with `g` the gap between the two emptiest projected axes. Reproduced with MAE
0.000000 at lags 24 and 30. Probe67's 0.123–0.166 is its **saturated branch**.
**The binding check is "is my effect bigger than the margin of the ecology I am
proposing"**, and `scripts/diagnose_probe68.py` computes it in minutes.

And the grain is **pinned from below by viability**: oracle survival is 0.000 at
half the grant. The lever that exists is the **quantum** at fixed nutrition —
halving it more than doubles the value of knowing what kind of body you are
(rate value +0.1272 → +0.2775) and leaves the value of knowing where it is flat
(state step +0.0101, interval spanning zero). Rate step **+0.0855 [+0.0753,
+0.0956] on 8/8 seeds, 8.5x its own state-side control.**

**Probe72 bounds that result at its endpoint.** The open-loop amplification is
real and does not amplify survival value: the corresponding closed-loop rate
step is -0.0229 [-0.0869, +0.0412]. The margin law describes the grain of the
word decision, not a sufficient statistic for the live body's hazard.


## Executive handover (probes 57–60, unchanged)

> A parameter-persistent embodied organism learns a public word lexicon, a
> persistent causal belief over its own hidden food/water/energy state, and a
> full-vocabulary model of how its words change caregiver help. It composes those
> models to communicate its inferred need with 94.99% fidelity and 92% survival on
> held-out lives, while matched causal controls fail.

Five-seed replication: balanced accuracy 0.9181 +/- 0.0066, grounded survival
0.9060 +/- 0.0182, fidelity 0.9471 +/- 0.0034, belief fork 1.0000, scrambled
listener 0.0420, zero belief 0.0000, full promotion gate 5/5.

Binding on all later phases: state a **belief-side endpoint** alongside any
behavioural one; **check the oracle ceiling before locking a gate**; **run the
survey on a seed band disjoint from the treatment**; **state the endpoint in the
currency the world is denominated in**; **measure the margin of the ecology you
are proposing and compare your effect with it** (probe68); **do not gate on a
contrast that mixes two belief errors** (probe68); and **state the budget any
family comparison was made at, and do not call it a ceiling unless you swept it**
(probe69).

## Standing findings

- **Self-knowledge is worth an order of magnitude more when it chooses the
  action than when it chooses a word** (direction pilots; Probe80).
- **Structure matters exactly when there is structure** (Probe80): a
  structure-free learner ties the posterior with one hidden resource and
  collapses with two or more.
- **A self-model of another body is worse than none** (Probe80, yoked).
- **Uncertainty about the self is what discovers** (Probe80): acting on the
  single most probable self survives but recovers almost none of its structure.
- **A self-model that checks itself against its own sensations reports
  calibrated confidence; one that does not is confidently wrong** (Probe80;
  development diagnosis: 81/103 confident errors followed an impossible
  sensation, against 2/390 correct reports).

- **Prediction and attribution come apart** (probe63).
- **A self-model can only localize a fact its own parameter set can express**
  (probe63).
- **Accuracy and regret come apart** (probe65).
- **A belief's point and its width live on different scales relative to the
  decision**, and only the first reaches one (probe67).
- **A belief error reaches a word only if it exceeds the grain, and the grain is
  pinned from below by viability** (probe68).
- **The homeostatic bound is an active part of the decision surface, not a
  backdrop** — probe62, probe63/67, probe68.
- **Structure buys sample efficiency, not a ceiling** (probe69).
- **A ratio fitted between two points inside the noise floor cannot constrain a
  third** (probe69, G6).
- **A constitution is a prior over demand, not realized demand**: fixed rate
  accounts discard the live request stream and can improve basket efficiency
  while reducing survival (probe70).
- **First-use compositional rent has a closed-form exchangeable ceiling, and
  homeostatic order is a measurable residual around it** (probe71). The
  instrument passes; the fixed body that was meant to license its treatment does
  not.
- **Amplifying a self fact's value at the word-decision surface does not imply
  amplifying its survival value** (probe72). Finer help lifts every arm, while
  the clean rate interaction reverses; true rates still have +0.0214 physical
  value in the fine ecology.
- **A learned individual bodily rate can be physically load-bearing** (probe73):
  the rate-only lesion worsens matched regret on 5/5 blocks and costs +0.0143
  survival, while preserving state, uptake, evidence, planner and compute.
- **Retained evidence pays rent when the physical individual persists**
  (probe74 survey, independently confirmed by probe75): resetting only the rate
  evidence costs 0.0300 survival locally and 0.0357 in confirmation;
  wrong-identity memory loses the benefit. Current-state belief still resets,
  and extra evidence history is the intended budget difference.
- **Saturation makes restorative causal groups identifiable from scalar
  intervention evidence** (probe76). This identifies reachable components,
  not arbitrary total hidden dimension; matched forks and absent shocks are
  still instrument privileges.

## Artifacts and records

- `runs/organism/probe80_active_self/` — `survey_block*_k*.json`, `survey_graded.json`, logs, `parallel_tail/` splice record
- `docs/decisions/2026-10-04-self-in-the-action-loop-direction.md`
- `docs/decisions/2026-10-04-active-self-discovery-preregistration.md`, `docs/decisions/2026-10-05-active-self-discovery-result.md`

- `runs/organism/probe75_cross_life_replication/{diagnostic,replication,evidence}.json`
- `docs/decisions/2026-09-30-cross-life-replication-{preregistration,result}.md`
- `runs/organism/probe76_saturation_structure/{diagnostic,survey}.json`
- `docs/decisions/2026-09-30-saturation-structure-{preregistration,result}.md`
- `runs/organism/probe74_cross_life_self/{granularity_diagnostic,survey}.json`
- `runs/organism/probe73_learned_rate_survival/{granularity_diagnostic,treatment}.json`
- `runs/organism/probe72_granularity_survival/treatment.json`
- `runs/organism/probe71_expanded_body/construction_survey.json`
- `runs/organism/probe70_constitution_budget/ceiling_survey.json`
- `runs/organism/probe69_belief_scaling/{treatment,extrapolation}/scaling_survey.json`
- `runs/organism/probe68_granularity/` — `treatment.json`, `ceiling_survey.json`,
  `quantum_survey.json`, `quantum_margin.json`, `diagnostic.json`
- `runs/organism/probe67_uncertain_horizon/`, `probe65_future_request/`,
  `probe66_discovered_convention/`, `probe64_portion_request/`,
  `probe63_individual_self/`, `probe62_uncertain_self/`,
  `probe61_discovered_self/full/`, `probe57_structured_causal_self/`,
  `probe58_causal_stage_replication/`, `probe59_online_adaptation/`,
  `probe60_outcome_aware_self_planner/`
- `docs/decisions/2026-09-30-cross-life-self-{preregistration,result}.md`
- `docs/decisions/2026-08-24-learned-rate-survival-{preregistration,result}.md`
- `docs/decisions/2026-08-22-granularity-survival-{preregistration,result}.md`
- `docs/decisions/2026-08-22-expanded-body-composition-law-{preregistration,result}.md`
- `docs/decisions/2026-08-20-constitution-budget-ceiling-{preregistration,survey}.md`
- `docs/decisions/2026-08-18-architecture-audit-and-route.md`
- `docs/decisions/2026-08-18-belief-scaling-result.md`
- `docs/decisions/2026-08-17-belief-scaling-{ceiling-survey,preregistration}.md`
- `docs/decisions/2026-08-17-decision-granularity-{preregistration,ceiling-survey,result}.md`
- `docs/decisions/2026-08-16-uncertain-horizon-ceiling-survey.md`
- `docs/decisions/2026-08-16-{future-request,discovered-convention}-result.md`
- `docs/decisions/2026-08-14-portion-request-result.md`
- `docs/decisions/2026-08-03-individual-self-calibration-result.md`
- `docs/decisions/2026-08-02-discovered-self-structure-result.md`
- `docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`
- `docs/decisions/2026-07-26-causal-self-report-independent-verification.md`

Latest local verification: **614 tests and 13 subtests passed** on both MLX
0.32.0 (72.66 seconds) and 0.32.3 (71.36 seconds), on 2026-10-02. The latter
reproduces CI runtime/test dependency versions. The preceding GitHub run failed
29 legacy training tests because discrete gather indices lacked explicit
gradient boundaries; the maintenance repair preserves the straight-through
soft gradient and pins CI dependencies. See
`docs/decisions/2026-10-02-mlx-ci-compatibility.md`. All new behavior is
default off. `tests/test_expanded_body.py` asserts exact legacy-schema constants,
five unique axes/words/consequence surfaces, paired random streams and a real
health viability path. `tests/test_composition_law.py` distinguishes fixed-count
permutation from i.i.d. resampling, checks the equation against Monte Carlo and
guards the survey seed bands. `tests/test_constitution_budget.py` asserts exact
first-come reproduction, unlimited-store inertness, burden-label permutation
and discrete quota rounding. `tests/test_belief_scaling.py` asserts that both families are
scored by the *same* `_balanced_accuracy` object — a divergent copy would make
every probe69 table a comparison between two definitions of one word — that
balanced accuracy scores a majority-class predictor at chance rather than at the
class prior, and that the
seed-isolation guard refuses a budget larger than its own stride, which matters
because probe69's budgets are 64x anything developed on before. Simulator
event/kind metadata remains audit-only and never enters causal belief, listener
learning, utterance planning, policy, or replay.

`tests/test_granularity_survival.py` guards the locked difference-in-differences,
the state-value lesion, viability and exact-nutrition controls, seed boundaries,
chunk reconstruction and resumable progress. The runner adds no training path
and its two ecology kwargs remain default off in the inherited closed loop.

`tests/test_learned_rate_survival.py` guards the one-factor rate lesion, all five
locked gates, tick-weighted belief/regret reconstruction, seed isolation and
resumable progress. Every treatment cell instantiates the same recursive tier;
the runner changes only which already-computed rate vector reaches the fixed
request rule.

`tests/test_cross_life_self.py` guards four-axis persistent identity, independent
episode streams, reset state with retained RLS statistics, no truth access at
learner reset, all six locked continuation clauses, correct cyclic donor
assignment, seed boundaries, and interruption-safe resume. Resume also pins
parent weights/configuration contents. A real diagnostic checks exact parity
with the inherited reset estimator, and the completed survey was resumed
without replaying episodes. No global model/configuration changes were added.

`tests/test_cross_life_replication.py` guards exact mechanism/grade reuse,
locked/disjoint seed bands, source/parent fingerprints and safe legacy progress.
`tests/test_probe75_summary.py` independently guards raw and compact artifact
reconstruction, tick weighting, five-block intervals and malformed records.
`tests/test_saturation_structure.py` guards scalar-only inference, real grant
timing, unequal alias group sizes, held-out separation, relabeling, remapping,
unreachable inertia, source-pinned resume and independently separated seed domains.

## Research basis

- Tang et al. (ICML 2023), self-predictive representation learning:
  https://proceedings.mlr.press/v202/tang23d.html
- Jaques et al. (ICML 2019), counterfactual social influence:
  https://proceedings.mlr.press/v97/jaques19a.html
- Foerster et al. (AAAI 2018), counterfactual multi-agent policy gradients:
  https://ojs.aaai.org/index.php/AAAI/article/view/11794
- Mesnard et al. (ICML 2021), counterfactual credit assignment:
  https://proceedings.mlr.press/v139/mesnard21a.html
- Lambrechts et al. (ICML 2025), asymmetric actor-critic under partial
  observability: https://proceedings.mlr.press/v267/lambrechts25a.html

These papers motivate mechanisms and controls. Only repository experiments are
evidence about this organism.
