# Natural language acquisition POC

Research project: can an embodied learner acquire a self model and make that model useful for both regulation
and communication (such that even deflationalists will concede that is uses language meaningfully ;D)? 

**Status:** In development.
- Online individual-body calibration and causally useful three-word need reports in a small custom
simulation. 
- A matched rate-only lesion now shows that using the learned individual rate
  improves survival at a long consequence horizon.
- Retaining evidence about the same body's rates across episodes improves
  survival over relearning each episode; a wrong-identity memory control loses
  that benefit. This result now replicates on disjoint bodies and episode streams.
- A rationed caregiver that can be asked for a portion size, where the right
request depends on a fact about the self that perfect state knowledge cannot supply.
- ToDo: fuller natural language acquisition

## Current results

Probe77 tests a sequential replacement for Probe76's matched forks, with
ordinary shocks and death-limited experience. **All five continuation gates
fail** at both one-episode and six-episode budgets. Across 500 bodies, exact
grouping after six episodes is **36%, 6%, 1%, 0%, 0%** at K1–K5. The five-resource
bodies collect only about eight usable grant observations in their first life.
This closes uniform random exploration plus predecessor-effect contrasts at
those budgets; it does not rule out directed exploration or a different
estimator. See the [result](docs/decisions/2026-09-30-sequential-structure-result.md).

Probe78's changing-body ceiling passes all three continuation clauses. An
oracle change cue that discards obsolete rate evidence increases survival from
**149/700 to 167/700**, **+0.0257 [+0.0026,+0.0488]**, while stationary sham
episodes are exactly unchanged. The cue supplies no new constants or state
reading. This establishes the value of correctly timed forgetting, not that
the organism detects change. See the [result](docs/decisions/2026-10-02-changing-body-ceiling-result.md).
Probe79's observable-error detector **fails two of three screen gates**. It
reduces matched-history rate error by 48.7% and request regret by 34.2%, but
resets 17.1% of unchanged bodies (10% allowed) and does not reliably beat
sign-scrambled detection. This fixed detector is closed before a survival
treatment. See the [result](docs/decisions/2026-10-02-evidence-reset-result.md)
and the [proposed next roadmap](docs/decisions/2026-10-02-next-roadmap-proposal.md).

The newest identification instrument, Probe76, infers which opaque restorative
actions target the same hidden resource from scalar saturation interactions.
It recovers reachable counts **1–5** and exact action partitions in **25/25
worlds**, predicts held-out overlap **750/750** correctly, and passes all six
locked gates. Sensory shuffling destroys grouping; physical remapping
invalidates the old grouping and is recovered. This uses matched forks and no
shocks, and leaves unreachable bodily dimensions unresolved. It licenses
testing sequential exploration, not calling an online organism's structure
discovered. See the [result](docs/decisions/2026-09-30-saturation-structure-result.md).

Probe75 independently replicates persistent individual rate memory on fresh
bodies and episode streams, keeping Probe74's mechanism, sample and six gates
unchanged. Five paired blocks × 28 identities × five test episodes give:

| rates used by request planning | survivors / 700 | survival |
|---|---:|---:|
| relearn each episode | 147 | 0.2100 |
| **retain individual rate evidence** | **172** | **0.2457** |
| true individual rates | 173 | 0.2471 |
| memory initialized from another body | 151 | 0.2157 |

Retained minus reset survival is **+0.0357 [+0.0031, +0.0683]**, positive on
**5/5 blocks**, adding 25 survivors. **All six original gates pass.** An
independent standard-library checker reconstructs the result from episode
records and verifies the saved summaries. This replicates the stationary-body
memory mechanism under one frozen motor parent; it does not discover axes,
recognize identity, establish saturation or acquire generative language. See
the [result](docs/decisions/2026-09-30-cross-life-replication-result.md) and
[preregistration](docs/decisions/2026-09-30-cross-life-replication-preregistration.md).

The preceding survey, Probe74, tests memory about a body that actually persists
across episodic resets. Current-state belief resets, while a separate recursive
rate estimator retains its evidence. Five paired blocks x 28 identities x five
test episodes give 700 episodes per arm, after one calibration episode per
identity:

| rates used by request planning | survivors / 700 | survival |
|---|---:|---:|
| relearn each episode | 159 | 0.2271 |
| **retain individual rate evidence** | **180** | **0.2571** |
| true individual rates | 180 | 0.2571 |
| memory initialized from another body | 158 | 0.2257 |

Retained minus reset survival is **+0.0300 [+0.0141, +0.0459]**, positive on
**5/5 blocks**. Birth-rate error falls 78.7%; matched regret improves on all
five blocks. All six preregistered continuation gates pass. The evidence
budget is intentionally larger for retained memory, with matched estimator
compute. This local survey licenses independent replication; the aggregate
retained/true-rate tie does not establish saturation. See the
[result](docs/decisions/2026-09-30-cross-life-self-result.md),
[preregistration](docs/decisions/2026-09-30-cross-life-self-preregistration.md),
and [current state](docs/STATE.md).

The preceding result, Probe73, asks whether the learned bodily rate itself changes
whether the organism lives. At lag 24, every cell runs the same online recursive
calibrator, filtered state, uptake belief, request ledger, frozen motor parent
and planner compute. Only the rate vector exposed to request planning changes.
Five paired seed blocks x 140 lives per cell give:

| rates used by request planning | survival | mean life steps |
|---|---:|---:|
| species rates | 0.1743 | 126.1 |
| **learned individual rates** | **0.1886** | **131.3** |
| true individual rates | 0.2129 | 145.5 |

Learned minus species survival is **+0.0143 [+0.0003, +0.0283]**, with four
positive blocks and one tie. The matched true-rate ceiling is +0.0386
[+0.0213, +0.0559] on 5/5 blocks. Learned rates remove 81.7% of the species
rate error, reduce matched word regret on every block, and change 27.5% of need
words. All five preregistered gates pass, although the primary lower confidence
bound is narrow. See the [result](docs/decisions/2026-08-24-learned-rate-survival-result.md),
[preregistration](docs/decisions/2026-08-24-learned-rate-survival-preregistration.md),
and [current state](docs/STATE.md).

In Probe63, each simulated organism is born with unknown individual metabolic
constants and receives sparse interoceptive readings. A 21-scalar recursive
least-squares (RLS) self-calibrator is evaluated over 5 seed blocks x 40 lives,
with gates fixed before the treatment implementation.

| condition | body-state error | correct lowest-need report |
|---|---:|---:|
| species-level filter | 0.0785 | 66.5% |
| same readings, no persistent self-model (`snap`) | 0.0454 | 81.5% |
| online individual self-model (RLS) | **0.0133** | **94.6%** |

Relative to the identical-evidence `snap` control, RLS reduces body-state error
by 70.7% and improves report accuracy by 13.2 percentage points. The recursive
arm passes all seven locked gates, including moved-ground-truth localization,
shuffled readings, a null world, and parameter recovery. In a separate
closed-loop evaluation, report fidelity improves from 0.780 to 0.926; the
survival difference (0.620 to 0.640) is directionally positive but unresolved
at five seeds and was deliberately not a gate.

The mechanism is deliberately modest: standard RLS over a hand-designed causal
body model. The evidence supports **online identification of individual bodily
parameters and faithful reporting**, not discovered architecture or developed
language. See the full [result record](docs/decisions/2026-08-03-individual-self-calibration-result.md),
[preregistration](docs/decisions/2026-08-03-individual-self-calibration-preregistration.md),
and [current state](docs/STATE.md).

In Probe64 the caregiver gains a second thing it can be asked for -- a large or a
small portion -- and a finite basket, so asking for more than you will burn costs
something. The right size is `rate x 18 ticks`, a fact about how fast *this* body
burns rather than about where it currently is. Same 5 seed blocks x 40 lives,
gates locked before the run.

| speaker | knows | correct portion request |
|---|---|---:|
| species-level filter | species rates, filtered state | 0.7235 |
| same readings, no self-model (`snap`) | species rates, corrected state | 0.7839 |
| **reads the true body every tick**, species rates | perfect state, no self-knowledge | 0.9178 |
| online individual self-model (RLS) | learned rates | **0.9528** |
| born knowing its own rates, never reads its body | true rates, no state readings | 0.9930 |

The third row is the control this probe exists for. An organism that knows *what
it is* and is unsure *where it is* chooses better than one that reads its own
body perfectly and believes it is typical: +0.0350 [+0.0053, +0.0648] for the
learned self-model over perfect state knowledge, +0.0753 at the ceiling, paired
across seeds. Species and truth disagree on 27.7% of ticks; perfect state
knowledge recovers 19.4 of those points, knowing your own rates while blind to
your state recovers a further 7.5, and the last 0.7 need both.

Asked for a size word combination it has **never once uttered**, a listener model
factored into need and size answers correctly 0.9850 of the time; an unfactored
table with identical evidence scores exactly chance, 0.5000 on every seed.

The honest other half: **the world cannot hear the difference.** Under the
ration, a speaker saying a *random* size word survives 0.640 against the informed
speaker's 0.605. The informed speaker really is better at the physical thing --
0.708 useful uptake per unit of basket against 0.656, 23% less overflow -- but a
need re-served every 18 ticks recovers before a mis-sized portion can kill it.
All seven locked gates pass; survival was preregistered as not gated for exactly
this reason. [Result](docs/decisions/2026-08-14-portion-request-result.md),
[preregistration](docs/decisions/2026-08-14-portion-request-preregistration.md),
[ceiling survey](docs/decisions/2026-08-14-portion-request-ceiling-survey.md).

## Reproduce the public result

The tested environment is Python 3.13 on Apple Silicon with MLX 0.32. A small
publication bundle is committed; the remaining 138 MB local experiment archive
stays ignored.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/summarize_probe63.py
.venv/bin/python scripts/summarize_probe64.py
.venv/bin/python scripts/summarize_probe65.py
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Re-run the resumable Probe74 survey from the same committed parent checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.cross_life_self
```

Verify Probe75's committed compact evidence with no MLX import, or rerun its
fixed confirmation on the same parent:

```bash
.venv/bin/python scripts/summarize_probe75.py
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.cross_life_replication \
  --out /tmp/probe75-replication.json
```

Completed progress pins its source code. Use a fresh output path when rerunning
after a source change; incomplete progress refuses a different mechanism.

Reconstruct Probe77's aggregate gates from its compact public evidence:

```bash
.venv/bin/python scripts/summarize_probe77.py
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.sequential_structure \
  --out /tmp/probe77-survey.json
```

Verify the changing-body reset ceiling from its compact block evidence:

```bash
.venv/bin/python scripts/summarize_probe78.py
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.changing_body \
  --out /tmp/probe78-survey.json
```

Verify the observable-error detector screen from its compact block evidence:

```bash
.venv/bin/python scripts/summarize_probe79.py \
  --input runs/organism/probe79_evidence_reset/evidence.json
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.evidence_reset \
  --out /tmp/probe79-survey.json
```

Re-run Probe76's fixed structure instrument:

```bash
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.saturation_structure \
  --out /tmp/probe76-survey.json
```

Re-run the preceding Probe73 treatment:

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.learned_rate_survival
```

Re-run a one-life smoke treatment from the committed parent checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.self_calibration \
  --lives 1 --seeds 1 --out /tmp/probe63-smoke.json
```

The complete treatment defaults to 5 x 40 lives and can be run by omitting the
three overrides. It is much slower than the artifact-only summary.

## Research direction

The larger aim is a developmental, socially situated learner in which language
is acquired because it improves prediction, coordination, and bodily
regulation. That vision is explained—and separated from what exists today—in
[Vision and research status](docs/VISION_AND_STATUS.md). An honest workshop
paper route is sketched in [paper/OUTLINE.md](paper/OUTLINE.md).

For public-use boundaries, see [Ethics and responsible claims](ETHICS.md) and
[Project contributions and AI assistance](CONTRIBUTIONS.md). The code is MIT
licensed.

## Repository map

| Path | Purpose |
|---|---|
| `src/homesocial/` | gridworld, learners, language probes, organism lifecycle, and controlled self-model experiments through Probe79 |
| `tests/` | 610 unit and causal guard tests, plus 13 subtests |
| `docs/decisions/` | append-only preregistrations, results, and negative findings |
| `docs/STATE.md` | current scientific state and exact claim boundary |
| `runs/` | allow-listed public artifacts; all other run output remains ignored |
| `scripts/summarize_probe63.py` | dependency-light recalculation of the Probe63 headline |
| `scripts/summarize_probe64.py` | the same for Probe64, including its behavioural negative |
| `scripts/summarize_probe65.py` | the same for Probe65: the horizon crossover and its divergence control |
| `scripts/summarize_probe75.py` | independent reconstruction of all six confirmation gates, with committed compact block evidence |
| `scripts/summarize_probe77.py` | reconstruction of sequential-discovery aggregate gates from compact per-body scores |
| `scripts/summarize_probe78.py` | independent reconstruction of changing-body survival and matched-score gates |
| `scripts/summarize_probe79.py` | independent reconstruction of the observable-error detector's matched-history gates |

---

## Detailed experiment notebook

The remainder of this README is the historical command notebook. It is useful
for finding individual probes, but the documents linked above are the canonical
public entry points.

This is the first prototype for a minimal embodied/social learning environment.

The goal is not to build a full agent yet. The goal is to define the smallest
closed loop where language can become useful because it helps an agent regulate
its own viability.

## Core Idea

The environment contains:

- a partially observed grid world,
- resources such as water, food, shelter, danger, trees, and rocks,
- internal homeostatic variables: food, water, energy, and safety,
- actions such as moving, pointing, asking, consuming, and resting,
- a deterministic teacher that speaks only in response to situated agent action.

This gives us the first testbed for comparing:

1. no social language,
2. grounded in-loop teacher language,
3. later: decoupled/objective language injected outside the loop.

## Run

```bash
python3 -m homesocial.demo
```

If running from a fresh checkout without installing the package:

```bash
PYTHONPATH=src python3 -m homesocial.demo
```

Run the first teacher/no-teacher Q-learning comparison:

```bash
PYTHONPATH=src python3 -m homesocial.experiment --episodes 500 --eval-episodes 50
```

Run the recurrent actor-critic baseline:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 500 --batch-size 16 --ppo-epochs 3 --eval-episodes 20 --hidden-size 128 --log-every 50
```

Run a cheaper fixed-world diagnostic before spending time on randomized hidden
object-kind sweeps:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 120 --batch-size 8 --ppo-epochs 2 --eval-episodes 10 --hidden-size 64 --max-steps 80 --fixed-world --conditions grounded_teacher silent_teacher --log-every 40
```

Run the stricter language-necessity sanity check before larger PPO sweeps. In
this mode object positions do not identify object kind, visible objects share
the same learner-facing name, and unsafe guessing is costly:

```bash
PYTHONPATH=src python3 -m homesocial.experiment --episodes 0 --eval-episodes 20 --fixed-world --diagnostic-mode language_necessary --scripted-probe --conditions grounded_teacher silent_teacher
```

Run a tiny recurrent smoke test against the same diagnostic mode:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 16 --batch-size 4 --ppo-epochs 1 --eval-episodes 2 --hidden-size 32 --max-steps 40 --diagnostic-mode language_necessary --conditions grounded_teacher silent_teacher --log-every 8
```

Run a behavior-cloning warmstart from closed-loop teacher-following episodes:

```bash
PYTHONPATH=src python3 -m homesocial.imitation --expert-episodes 200 --epochs 5 --batch-size 32 --eval-episodes 20 --hidden-size 128 --fixed-world --diagnostic-mode language_necessary --checkpoint runs/bc_homegrid.weights.npz
```

Run the first self-battery consequence probe on a trained checkpoint:

```bash
PYTHONPATH=src python3 -m homesocial.self_battery --checkpoint runs/bc_homegrid.weights.npz --rollout-policy model --teacher-modes grounded masked shuffled wrong
```

Train counterfactual branch consequences for the self-battery:

```bash
PYTHONPATH=src python3 -m homesocial.counterfactual --checkpoint runs/bc_homegrid.weights.npz --output-checkpoint runs/cf_homegrid.weights.npz --episodes 400 --max-decisions 1200 --epochs 6 --batch-size 128 --rollout-policy teacher --branch-selection probe --rank-weight 1.0
```

Train and evaluate the structured self-report head:

```bash
PYTHONPATH=src python3 -m homesocial.report_head --checkpoint runs/cf_homegrid.weights.npz --eval-teacher-modes grounded masked shuffled wrong
```

The default report input is restricted to recurrent state plus the model's own
action-conditioned need/reward predictions. Run the anti-shortcut calibration
suite before interpreting report accuracy:

```bash
PYTHONPATH=src python3 -m homesocial.report_calibration --checkpoint runs/cf_homegrid.weights.npz
```

Compare against an untrained recurrent/consequence representation with:

```bash
PYTHONPATH=src python3 -m homesocial.report_head --checkpoint runs/cf_homegrid.weights.npz --random-model-control
```

Test whether truthful need reports are causally useful to a teacher that cannot
inspect the agent's needs:

```bash
PYTHONPATH=src python3 -m homesocial.report_mediation --checkpoint runs/cf_homegrid.weights.npz
```

This triage gate validates causal communication, not learned-model necessity.
Use `--random-model-control` to expose the current direct-interoception
shortcut.

Train with exact need values removed from the observation while retaining
next-need supervision:

```bash
PYTHONPATH=src python3 -m homesocial.imitation --expert-episodes 500 --epochs 10 --hidden-size 96 --next-needs-weight 5 --diagnostic-mode language_necessary --interoception-mode masked --checkpoint runs/bc_hidden_interoception.weights.npz
```

Evaluate temporal self-state inference and report-mediated action:

```bash
PYTHONPATH=src python3 -m homesocial.interoception --checkpoint runs/bc_hidden_interoception.weights.npz --random-model-control
PYTHONPATH=src python3 -m homesocial.hidden_mediation --checkpoint runs/bc_hidden_interoception.weights.npz --random-model-control
```

Train sequence-dependent hidden body dynamics:

```bash
PYTHONPATH=src python3 -m homesocial.imitation --expert-episodes 800 --epochs 12 --hidden-size 128 --next-needs-weight 6 --diagnostic-mode language_necessary --interoception-mode masked --body-dynamics-mode stochastic --checkpoint runs/bc_stochastic_body.weights.npz
```

Evaluate ordered-history inference and compositional reports:

```bash
PYTHONPATH=src python3 -m homesocial.interoception --checkpoint runs/bc_stochastic_body.weights.npz --history-modes full latest shuffled reversed --random-model-control
PYTHONPATH=src python3 -m homesocial.compositional_report --checkpoint runs/bc_stochastic_body.weights.npz --random-model-control
```

Train a two-slot discrete message protocol only through receiver decisions:

```bash
PYTHONPATH=src python3 -m homesocial.emergent_language --checkpoint runs/bc_stochastic_body.weights.npz --feature-mode self_estimate --random-model-control
```

Test whether the learned convention transfers to new receivers or independent
protocol initializations:

```bash
PYTHONPATH=src python3 -m homesocial.convention_transfer --checkpoint runs/bc_stochastic_body.weights.npz --feature-mode self_estimate
```

Train a population of senders against a shared receiver to test whether symbols
become interoperable rather than private:

```bash
PYTHONPATH=src python3 -m homesocial.population_language --checkpoint runs/bc_stochastic_body.weights.npz --agreement-weights 0 0.05 0.2
```

Train a harder self-request protocol where the receiver must choose the aid
category from the message alone:

```bash
PYTHONPATH=src python3 -m homesocial.self_request_language --checkpoint runs/bc_stochastic_body.weights.npz --agreement-weights 0 0.05 0.2
```

Train a richer self-state protocol where the receiver reconstructs continuous
needs, low-need flags, dominant need, severity, and trend from the message
alone:

```bash
PYTHONPATH=src python3 -m homesocial.self_state_language --checkpoint runs/bc_stochastic_body.weights.npz --agreement-weights 0 0.05 --history-modes full latest --random-model-control
```

For a stricter temporal-reflection audit, balance the dataset across worsening,
steady, and improving states and expose only learned self-estimate deltas:

```bash
PYTHONPATH=src python3 -m homesocial.self_state_language --checkpoint runs/bc_stochastic_body.weights.npz --feature-mode self_estimate_delta --balance-target trend --agreement-weights 0.05 --history-modes full latest --random-model-control
```

Intervene on the learned self-estimate delta channel while holding current
self-state fixed:

```bash
PYTHONPATH=src python3 -m homesocial.temporal_counterfactual --checkpoint runs/bc_stochastic_body.weights.npz --random-model-control
```

Train and test trend communication on counterfactual action branches from the
same current state:

```bash
PYTHONPATH=src python3 -m homesocial.counterfactual_trend_language --checkpoint runs/bc_stochastic_body.weights.npz --random-model-control
```

Run the stricter action-balanced branch audit, then a short option-level
counterfactual audit where branches execute seek/rest/wait policies for several
steps:

```bash
PYTHONPATH=src python3 -m homesocial.counterfactual_trend_language --checkpoint runs/bc_stochastic_body.weights.npz --balance-target action_trend --random-model-control
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/bc_stochastic_body.weights.npz --horizon 6 --random-model-control
```

Train the transition head directly on multi-step option branches, then test
latent option-state communication from the updated world model:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world.weights.npz --horizon 6
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world.weights.npz --horizon 6 --rollout-mode latent --random-model-control
```

Use broader exploratory state collection and an explicit current/future/delta
self-estimate feature when testing trend language:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world_random.weights.npz --horizon 6 --state-policy random
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world_random.weights.npz --horizon 6 --state-policy random --rollout-mode latent_current --trend-weight 2.0 --random-model-control
```

Run the same probe with renewable resources to test whether repeated food/water
encounters improve temporal self-report coverage:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world_renewable.weights.npz --horizon 6 --state-policy random --renewable-resources
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world_renewable.weights.npz --horizon 6 --state-policy random --rollout-mode latent_current --trend-weight 2.0 --random-model-control
```

Use richer resource ecology when probing less rest-dominated option reports:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_model --checkpoint runs/bc_stochastic_body.weights.npz --output-checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich
PYTHONPATH=src python3 -m homesocial.option_counterfactual_language --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --rollout-mode latent_current --trend-weight 2.0 --random-model-control
```

Audit whether the option self-trend message depends on current/future/delta
self-estimate features rather than only option/time regularities:

```bash
PYTHONPATH=src python3 -m homesocial.option_feature_intervention --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --trend-weight 2.0 --balance-target option_trend --random-model-control
```

Replicate the rich-ecology option-world plus feature-intervention pipeline
across seeds, including random-model and option-majority controls:

```bash
PYTHONPATH=src python3 -m homesocial.option_seed_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9901 9902 9903 --horizon 6 --state-policy cycle --resource-ecology rich --trend-weight 2.0 --balance-target option_trend --random-model-control
```

Test whether compact option self-change messages can mediate a useful option
choice, rather than only reconstruct labels. The command reuses the same
grouped option states for trained and random controls:

```bash
PYTHONPATH=src python3 -m homesocial.option_mediation --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --random-model-control
```

Force the mediation receiver to see only predicted future-current self-change
instead of current and future absolute self-estimates:

```bash
PYTHONPATH=src python3 -m homesocial.option_mediation --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --random-model-control
```

Add a direct self-model rank control that chooses by predicted future self-state
without training a message receiver:

```bash
PYTHONPATH=src python3 -m homesocial.option_mediation --checkpoint runs/option_world_rich.weights.npz --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --random-model-control --self-model-rank-control
```

Stress option identity shortcuts by adding branch action noise while keeping the
actual sampled action sequence as both the true outcome and the model input:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9921 9922 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.3 --random-model-control
```

Option-world and mediation source noise can be split for curriculum probes:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9961 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --world-option-action-noise 0.3 --mediation-option-action-noise 0.15 --random-model-control
```

Fine-tune the self/world model directly on grouped branch ranking before
training the message receiver:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9981 9982 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --rank-finetune-epochs 4 --self-model-rank-control --random-model-control
```

Add dynamics replay during rank fine-tuning to preserve ordinary option-world
prediction quality:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9991 9992 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Train the message receiver to communicate the model's own predicted best
self-future instead of the oracle branch label:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10001 10002 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Vary the discrete message capacity for compression probes:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10011 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --message-slots 3 --message-vocabulary 4 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Train with soft message probabilities but still evaluate hard discrete messages:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10021 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --soft-message-training --message-temperature 0.8 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Add an auxiliary hard-message loss that reconstructs predicted self-model
option scores from the emitted symbols:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10031 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-reconstruction-weight 0.1 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Pretrain the hard discrete message code to reconstruct the self-model's option
scores before normal choice training:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Run paired no-pretrain controls on the same seeds before interpreting a
score-pretraining gain:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10041 10042 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Add pairwise self-score rank pressure to the hard-message score head:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --score-rank-weight 0.05 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Distill the receiver's hard-message choice logits toward the self-model score
distribution:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --score-distillation-weight 0.2 --score-distillation-temperature 0.7 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Warm up the receiver on a fixed hard message code after score pretraining:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --frozen-receiver-epochs 5 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Add light sender commitment pressure so each slot becomes more confident in its
current hard symbol while balance pressure keeps code usage spread out:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --message-commitment-weight 0.005 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Apply commitment only during score pretraining, then fine-tune choice without
extra commitment:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 10043 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --score-pretrain-commitment-weight 0.005 --message-commitment-weight 0.0 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

Mediation CSVs include `message_codes_used`, `message_code_entropy`,
`dominant_message_code_fraction`, `target_code_mutual_information`,
`choice_code_mutual_information`, `message_patterns_used`,
`reused_message_pattern_fraction`, and row-pattern mutual information fields to
distinguish code collapse, weak sender/receiver semantics, and one-off pattern
memorization.

`--message-replay-weight` snapshots score-pretrained hard message tokens and
penalizes later sender drift during choice training.
`--heldout-receiver-epochs` trains a fresh receiver against a frozen sender to
test whether the compact self-code transfers beyond the co-trained receiver.
`--receiver-copies` trains multiple co-receivers against the same sender as a
convention-pressure probe.
`--score-rank-code-weight` pushes one message slot to encode self-model rank
buckets as a reusable ordinal-symbol probe.

Replicate the strict self-model-targeted commitment setting across independent
base body checkpoints:

```bash
PYTHONPATH=src python3 -m homesocial.option_body_mediation_replication --checkpoints runs/bc_m8_stochastic_persistent_seed12.weights.npz runs/bc_m8_stochastic_persistent_seed13.weights.npz --seeds 10112 10113 --horizon 6 --state-policy cycle --resource-ecology rich --feature-mode delta --option-action-noise 0.15 --mediation-target-mode self_model --score-pretrain-epochs 15 --message-commitment-weight 0.005 --rank-finetune-epochs 4 --rank-finetune-dynamics-weight 0.25 --self-model-rank-control --random-model-control
```

## Unified organism self-report

The current frontier is the report island: interoception is visible once at
birth and then masked, and a caregiver who cannot inspect the body grants help
only for generated full-vocabulary reports. Start with the exact-dynamics
feasibility seal:

```bash
PYTHONPATH=src python3 -m homesocial.island.report_calibrate --lives 400 --rhythm-lives 100 --rhythm-max-length 4 --unified-uptake
```

Reproduce probe52's need-independent guided lexical childhood and stop at its
frozen comprehension gate:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --report-unified-uptake --report-lexical-childhood-steps 60000 --report-lexical-guided-labels --report-lexical-gate-only --train-steps 200000 --segment-length 64 --hidden-size 256 --report-slots 2 --report-entropy-weight 0.02 --report-audit-lives 200 --seed 1 --run-dir runs/organism/probe52_guided_report_lexicon/gate --log-every-lives 100
```

Continue the exact gate-passed model into the scaffold-free adult report task:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --report-unified-uptake --report-adult-from-lexical-checkpoint runs/organism/probe52_guided_report_lexicon/gate/organism_report_lexical_child_seed1.npz --train-steps 200000 --segment-length 64 --hidden-size 256 --report-slots 2 --report-entropy-weight 0.02 --report-audit-lives 200 --seed 1 --run-dir runs/organism/probe52_guided_report_lexicon/adult --log-every-lives 100
```

The child reaches 100% paired causal word comprehension, and the adult retains
it, but adult self-report still fails at 31.46% fidelity and 3% survival. Run
the read-only state diagnostics without changing the organism:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-audit-lives 200 --report-self-state-diagnostic-only --run-dir runs/organism/probe52_guided_report_lexicon/diagnostic_self_state
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-audit-lives 200 --report-supervised-state-upper-bound-only --run-dir runs/organism/probe52_guided_report_lexicon/diagnostic_supervised_upper_bound
```

The supervised decoder is an audit-only upper bound trained on frozen states,
not an organism capability. Verify that the permitted observation/action
history is epistemically sufficient before learning a causal self-model:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-audit-lives 200 --report-observable-history-filter-only --run-dir runs/organism/probe53_observable_history_filter/feasibility
```

Train the successful structured causal body and full-vocabulary listener
models. Random developmental tokens are independent of body, and every
pre-existing organism parameter remains frozen:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --report-structured-causal-development-steps 80000 --report-audit-lives 200 --seed 1 --run-dir runs/organism/probe57_structured_causal_self/treatment --log-every-lives 200
```

After the self-model and lexical gates re-pass, run the sealed learned
social-consequence planner battery:

```bash
PYTHONPATH=src python3 -m homesocial.organism.harness --report-task --load-checkpoint runs/organism/probe57_structured_causal_self/treatment/organism_causal_self_seed1.npz --report-causal-social-planner-battery --report-audit-lives 200 --report-lexical-gate-lives 300 --seed 1 --run-dir runs/organism/probe57_structured_causal_self/planner_battery
```

Probe57 reaches 94.99% report fidelity and 92% survival versus 3.5% with a
scrambled listener. Belief lesions, mute/fixed-word controls, held-out births
and portions, and perceptible body forks all support the causal interpretation.
See `docs/STATE.md` for the exact boundary: this is learned causal self-report
in the minimal ecology, not a consciousness claim.

Replicate the causal stage across independent developmental seeds and degrade
the one privileged birth reading. Note that `--seed` alone cannot vary this
stage, because the causal parameters initialize deterministically; the
developmental world stream is what must move:

```bash
PYTHONPATH=src python3 -m homesocial.organism.causal_self_replication --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --run-dir runs/organism/probe58_causal_stage_replication --seeds 5 --lives 100
```

All five seeds pass every promotion gate: balanced accuracy 0.9181 +/- 0.0066,
survival 0.9060 +/- 0.0182, fidelity 0.9471 +/- 0.0034, belief fork following
1.00, zero-belief survival 0.00. Replacing the birth reading with the
population mean, which gives zero per-life bodily information, still yields
0.81 survival and 0.906 fidelity, so the belief is a contracting observer
rather than a dead-reckoner. See
`docs/decisions/2026-07-26-causal-self-report-independent-verification.md` for
the full re-audit and for what this result still does not show.

Reproduce the post-shift outcome-aware planner experiment. It preserves the
listener outcome distribution until after applying homeostatic utility, and
factorially separates the adapted belief, planning model, objective, and real
listener:

```bash
PYTHONPATH=src python3 -m homesocial.organism.outcome_aware_self --parent runs/organism/probe57_structured_causal_self/treatment/organism_causal_self_seed1.npz --run-dir runs/organism/probe60_outcome_aware_self_planner --seeds 5 --lives 100 --adaptation-ticks 40000 --lexical-lives 90
```

The planner repair raises survival from 0.500 to 0.904 and both scrambled
listener and zero-belief lesions score 0.000. The full gate still fails:
frozen and stale beliefs survive 0.890 and 0.904, so continual recalibration is
more accurate but not behaviourally load-bearing in this three-choice ecology.
See
`docs/decisions/2026-07-30-outcome-aware-self-planner-result.md`.

Ask whether the organism can discover the variables of its own body instead of
being handed them. The learner gets an overcomplete eight-dimensional latent and
only two scalar sensations per transition -- the mean of its bodily variables,
which is the world's own reward signal, and their minimum, which is what kills
it -- and no axis count, no axis order, no axis name, and no per-life birth
reading. The true number of bodily variables is varied in the world by freezing
axes, so the recovered dimension is scored against a ground truth that moves:

```bash
PYTHONPATH=src python3 -m homesocial.organism.discovered_self --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --run-dir runs/organism/probe61_discovered_self --seeds 5 --lives 100 --worlds K1,K2,K3,K4 --learning-rate 0.03 --sparsity 1e-5 --controls
```

`K4` is the unmodified frozen report ecology, whose true bodily dimension is
four: `safety` depletes at 0.002 per tick and enters both viability signals, and
it is the variable probe57's hand-written three-axis template cannot represent.
Pass `--sparsity` the value recorded in
`runs/organism/probe61_discovered_self/selection.json`, which is chosen by
held-out sensory error alone. Preregistration and locked gates:
`docs/decisions/2026-08-02-discovered-self-structure-preregistration.md`.

The twenty treatment fits may be split by world, because `run_world_seed` is
pure in `(world, seed)`. Run one process per world into
`runs/organism/probe61_discovered_self/full/<world>`, then merge and evaluate the
locked gates over the pooled records; the merge refuses a partial or ragged
sweep:

```bash
PYTHONPATH=src python3 scripts/probe61_merge_worlds.py --run-dir runs/organism/probe61_discovered_self/full
```

Ask whether self-uncertainty is worth anything before building a mechanism to
communicate it. `silent_shock_probability` withholds a shock's perceptible
marker while leaving the body change and every random stream identical, so it
moves what the organism can know about itself and nothing about what happens to
it. Four bodies -- the true one, probe53's exact visible-history filter, that
filter debiased, and a particle cloud conditioned on being alive -- are scored on
one shared history:

```bash
PYTHONPATH=src python3 -m homesocial.organism.uncertain_self --parent runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz --run-dir runs/organism/probe62_uncertain_self --lives 40 --particles 48 --skip-closed-loop
```

Silence opens a 26.6-point gap to the oracle on naming the truly lowest need,
and none of it is recoverable: the Bayes-optimal rule ties the biased point
filter at every silence rate. The cloud is genuinely calibrated -- coverage
0.92--0.94 against a nominal 0.90 -- but at matched inspection budget that
calibration is worth only +0.0 to +2.7 points over rate-matched random, against
up to +22.1 for inspecting at all. The body is bounded in [0,1] and the filter
saturates on ~9.7% of ticks, so its bias equilibrates at +0.051 instead of
accumulating. The ecology carries the self-model. Full record:
`docs/decisions/2026-08-03-self-uncertainty-ceiling-survey.md`.

### Probe63: an organism that finds out what its own body is

Every organism in this ecology used to burn fuel at exactly the species rate, so
its "self-model" was a model of bodies in general -- which is why the hand-written
probe53 filter was exact and why nothing learned could ever beat it. Three
default-inert levers change that: `metabolic_spread` gives each life its own burn
rates, `uptake_spread` its own absorption, and `interoception_probability` the
only channel through which either could be found out (the body is otherwise
visible exactly once, at birth). Measure the headroom before building anything:

```bash
PYTHONPATH=src python3 -m homesocial.organism.individual_self --lives 40 --closed-loop-lives 40
```

At spread 0.60 the species filter carries body error 0.0833 and names the truly
lowest need on 0.660 of ticks; the same filter on this life's constants is
*exact* and names it on 1.000. Unlike probe62's gap, all 34 points are
recoverable. Then run the preregistered treatment, five seeds, two learning
rules against one set of locked gates:

```bash
PYTHONPATH=src python3 -m homesocial.organism.self_calibration --lives 40 --seeds 5
```

The organism keeps one copy of its body filter per constant it could be wrong
about and reads off `d(predicted body)/d(log parameter)` -- a fact about its own
model rather than about its body -- then attributes each interoceptive residual
across its own parameters. Preregistration:
`docs/decisions/2026-08-03-individual-self-calibration-preregistration.md`.
Feasibility: `docs/decisions/2026-08-03-individual-self-ceiling-survey.md`.

### Probe64: making the self-model pay rent in speech

Probe63's organism knew something no listener could hear -- how fast it burns.
Two default-inert levers give the caregiver something to hear it with:
`portion_requests` lets it be asked for a large or a small portion, and
`caregiver_store` gives it a basket rather than a spring, so a large portion
empties it three times as fast. The right size is then `rate * 18 ticks`, which
is a fact about this body's metabolism rather than about where it currently is.
Survey the ecology before locking anything:

```bash
PYTHONPATH=src python3 -m homesocial.organism.portion_request --lives 40 --closed-loop-lives 40 --seed-base 950000000
```

The control this probe turns on is `state_oracle`: a speaker that **reads the
true body every tick** and believes the species rates. Then the preregistered
five-seed treatment against locked gates:

```bash
PYTHONPATH=src python3 -m homesocial.organism.portion_request --treatment --lives 40 --seeds 5
```

Preregistration: `docs/decisions/2026-08-14-portion-request-preregistration.md`.
Ceiling survey: `docs/decisions/2026-08-14-portion-request-ceiling-survey.md`.

### Probe65: a request about a body the organism is not in yet

Probe64's self-model changed what the organism said and bought no survival,
because the consequence horizon was shorter than the correction interval. One
default-inert lever changes that without touching the caregiver's clock, its
supply or its portions: `help_delay` makes the grant at each boundary answer the
request heard `help_delay` ticks earlier. Every word is then a prediction, and
where the body will be is `level - rate * horizon` -- in which an error about
*where you are* enters once and an error about *what you are* enters multiplied
by the horizon. Sweep the horizon before locking anything:

```bash
PYTHONPATH=src python3 -m homesocial.organism.future_request --lives 40 --closed-loop-lives 40 --seed-base 1000000000
```

The controls this probe turns on are `myopic` -- the organism this repository
already had, whose planner looked one grant ahead, dropped into a world whose
caregiver became slow without telling it -- and probe64's `state_oracle`. Then
the preregistered five-seed treatment against locked gates:

```bash
PYTHONPATH=src python3 -m homesocial.organism.future_request --treatment --lives 40 --seeds 5
```

Preregistration: `docs/decisions/2026-08-16-future-request-preregistration.md`.
Ceiling survey: `docs/decisions/2026-08-16-future-request-ceiling-survey.md`.

### Probe66: finding out that a word means the same thing whatever you ask for

Probe64's listener model pooled one belief per size word across needs, and that
pooling is what let it utter a combination it had never uttered -- but the
designer chose to pool. `tangled_size_words` (default off) lets the caregiver
draw its word-to-size orientation *per need*, so "more" may be large for food and
small for water, and asks whether the organism can tell. Before generalizing, it
asks whether the needs it has already used the word for agree with each other --
two exact Beta-Binomial marginal likelihoods, no free parameters -- and where they
do not, it **declines** rather than guessing.

```bash
PYTHONPATH=src python3 -m homesocial.organism.discovered_convention --lives 40 --seeds 5 --seed-base 1040000000
PYTHONPATH=src python3 -m homesocial.organism.discovered_convention --treatment --lives 40 --seeds 5
```

Preregistration: `docs/decisions/2026-08-16-discovered-convention-preregistration.md`.
Ceiling survey: `docs/decisions/2026-08-16-discovered-factorization-ceiling-survey.md`.

### Probe67 -- a self-model that knows how unsure it is, and cannot use it

**A negative, and the most reusable thing in the recent record.** Probe63's RLS has
carried a covariance over its own rate constants since August and no probe had
ever read it. Probe67 reads it: probe60's `E[min(next body)]` takes its
expectation over that posterior as well as over the caregiver's portion draw. No
free parameters, and bit-identical to probe65 at zero width.

The belief is good -- better calibrated than probe62's state posterior on
probe62's own instrument. Acting on it is worthless: the ceiling arm, handed the
magnitude of its own error, *loses* to the point rule at every horizon, and an
eleven-point change in what the organism says produces zero change in survival
over 400 lives an arm.

```bash
PYTHONPATH=src python3 -m homesocial.organism.uncertain_horizon --lives 30 --closed-loop-lives 80 --seeds 5 --delays 0,18,24,30
PYTHONPATH=src python3 scripts/diagnose_probe67.py --lives 10 --delays 0,18,24,30
```

The second command measures whether a mechanism's effect on the decision variable
can reach a decision this ecology treats as worth making. **Probe68 supersedes its
headline number with a formula** -- see below -- so run `diagnose_probe68.py`
instead when planning new work.

Survey: `docs/decisions/2026-08-16-uncertain-horizon-ceiling-survey.md`.

### Probe68: the grain of the decision surface

The margin between the best word and the next best is
`E_grant[min(grant * uptake, gap between the two emptiest axes)]`, reproduced with
MAE 0.000000 at two of four lags. Probe67's 0.12 is its saturated branch. Halving
the *quantum* of help while holding the *rate* of help exactly fixed more than
doubles the value of knowing what kind of body you are, and leaves the value of
knowing where it is unmoved.

Run the granularity check before building anything, in place of probe67's constant:

```bash
PYTHONPATH=src python3 scripts/diagnose_probe68.py --lives 8 --delays 0,18,24,30
PYTHONPATH=src python3 scripts/diagnose_probe68.py --quantum --lives 28 --delays 18
```

Then the two viability surveys, which found that this ecology has no headroom
beneath its grant -- oracle survival 0.000 at half the portion:

```bash
PYTHONPATH=src python3 -m homesocial.organism.decision_granularity --ceiling --lives 30
PYTHONPATH=src python3 -m homesocial.organism.decision_granularity --quantum --lives 25
```

And the preregistered treatment, roughly ninety minutes on an M-series Mac:

```bash
PYTHONPATH=src python3 -m homesocial.organism.decision_granularity --treatment --lives 30 --seeds 8 --out runs/organism/probe68_granularity/treatment.json
```

Four of five locked gates pass; G5 fails and is not rewritten. Records:
`docs/decisions/2026-08-17-decision-granularity-{preregistration,ceiling-survey,result}.md`.

### Probe69: structure against scale

Every comparison in this repository between a structured mechanism and a
black-box one was run at exactly one development budget -- 80,000 ticks, which is
**90 seconds** on this machine -- and that budget is the point of **maximum
separation** between the two families. Swept over budget, the structured model is
already saturated at 80,000 (16x the compute buys +0.0095) while the black-box
climbs +0.1582 and catches it: the two tie at 0.935 by 5,120,000 ticks.

The advantage structure buys is **sample efficiency, not a ceiling**.

The survey -- one seed, about two hours, and enough on its own to show the shape:

```bash
PYTHONPATH=src python3 -m homesocial.organism.belief_scaling --budgets 80000,320000,1280000 --widths 64,256 --seeds 1 --lives 200 --structured
```

The preregistered treatment. Five seeds across three budgets and three families,
then the extrapolation arm at 5,120,000 -- together roughly 24 hours on an
M-series Mac, so run them concurrently on a machine with cores to spare:

```bash
PYTHONPATH=src python3 -u -m homesocial.organism.belief_scaling --budgets 80000,320000,1280000 --widths 64,256 --seeds 5 --lives 200 --structured --treatment-band --run-dir runs/organism/probe69_belief_scaling/treatment
PYTHONPATH=src python3 -u -m homesocial.organism.belief_scaling --budgets 5120000 --widths 256 --seeds 3 --lives 200 --structured --treatment-band --run-dir runs/organism/probe69_belief_scaling/extrapolation
```

Do not pipe either through `grep`: it block-buffers into a file and a healthy run
looks frozen. Both write `scaling_survey.json` after every cell, so read that.

Grade against the locked gates:

```bash
PYTHONPATH=src python3 scripts/grade_probe69.py
```

Six of seven locked gates pass; G6 fails and is not rewritten -- the predicted
residual gap of 0.025 was measured at +0.0003, direction right and magnitude
wrong. Records:
`docs/decisions/2026-08-17-belief-scaling-{ceiling-survey,preregistration}.md` and
`docs/decisions/2026-08-18-belief-scaling-result.md`.

### Probe70: constitution-aware basket ceiling

Before adding a word for “I burn water fast”, this survey asks whether a
caregiver handed the oracle individual rates can use them at all. It partitions
the private lifetime basket by true per-need burden and compares that with the
existing first-come caregiver, plus species-rate and label-permuted controls.

```bash
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.constitution_budget
```

Five seeds x 40 lives across six finite stores plus the unlimited control take
about 35 minutes on this machine. Progress is written after every cell to
`runs/organism/probe70_constitution_budget/ceiling_survey.json`.

The mechanism is **not licensed**. Oracle-rate allocation loses to first-come at
every finite store, by 0.105–0.205 survival, and all 30 paired seed contrasts are
negative. The unlimited control is exact. Records:
`docs/decisions/2026-08-20-constitution-budget-ceiling-{preregistration,survey}.md`.

### Probe71: expanded-body composition-law construction ceiling

Before retraining the frozen parent for more bodily axes, this survey checks a
fixed K5 body and a closed-form first-use occupancy law with no learned
component:

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.composition_law \
  --lives 100 --seeds 5 \
  --out runs/organism/probe71_expanded_body/construction_survey.json
```

Four of six continuation clauses pass. The K5 body survives at 0.976, health
carries 20.70% of natural requests, and the formula matches its i.i.d. resample
control. Natural range is 0.02570 against 0.030, and the median axis gap moves
-0.01707 against the locked absolute 0.020, so C5 and C6 fail: no expanded
parent or composition treatment is built. Records:
`docs/decisions/2026-08-22-expanded-body-composition-law-{preregistration,result}.md`.

### Probe72: decision granularity at the survival endpoint

Probe68's finer help quantum raises the open-loop accuracy value of individual
rates. Probe72 asks whether that amplification reaches survival, using the same
clean `oracle - state_oracle` rate contrast and `state_oracle - population`
lesion over five paired seed blocks x 140 lives per cell:

```bash
PYTHONPATH=src .venv/bin/python -m homesocial.organism.granularity_survival \
  --out runs/organism/probe72_granularity_survival/treatment.json
```

The runner checkpoints every 20 lives and resumes completed chunks. Three of
five locked gates pass. True individual rates buy +0.0214 [+0.0089, +0.0340]
survival in the fine ecology on 5/5 blocks, but the granularity interaction is
-0.0229 [-0.0869, +0.0412] with only 2/5 positive. Smaller, more frequent help
improves every arm's survival; it does not selectively amplify rate knowledge.
Records:
`docs/decisions/2026-08-22-granularity-survival-{preregistration,result}.md`.

### Probe73: learned-rate survival

Probe73 returns to the unchanged help clock at lag 24 and changes one input to
the request rule. Every cell runs the same recursive calibrator; request planning
reads species, learned, or true rates:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_probe68.py \
  --lives 8 --delays 24 --seed-base 2900000000 \
  --out runs/organism/probe73_learned_rate_survival/granularity_diagnostic.json
PYTHONPATH=src .venv/bin/python -m homesocial.organism.learned_rate_survival
```

The runner checkpoints every 20 lives and resumes completed chunks. All five
locked gates pass. Learned rates buy +0.0143 [+0.0003, +0.0283] survival over
the matched species-rate lesion; the true-rate ceiling is +0.0386 [+0.0213,
+0.0559]. Records:
`docs/decisions/2026-08-24-learned-rate-survival-{preregistration,result}.md`.

### Probe74: cross-episode self-knowledge in a persistent body

One individual's metabolic constants now persist across episodic resets while
birth state, shocks and motor randomness are redrawn. Current-state belief still
resets every episode. A separate RLS rate memory retains only its parameters and
evidence statistics. The survey compares reset rates, retained rates, true rates,
and memory initialized from another identity, with matched estimator compute:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_probe68.py \
  --lives 8 --delays 24 --seed-base 3100000020 \
  --out runs/organism/probe74_cross_life_self/granularity_diagnostic.json
PYTHONPATH=src .venv/bin/python -u -m homesocial.organism.cross_life_self
```

Five blocks x 28 identities x five test episodes give 700 test episodes per
arm, after one shared calibration episode per identity. Progress is saved after
each calibration episode and each complete identity/cell sequence; rerunning
the command resumes completed work and rejects a changed parent checkpoint or
configuration. Episode correlation is retained in the five-block intervals.
Records: `docs/decisions/2026-09-30-cross-life-self-{preregistration,result}.md`.

All six continuation gates pass: retained rates buy **+0.0300 [+0.0141,
+0.0459]** survival, positive on 5/5 blocks; wrong-identity memory loses the
benefit. This survey licenses a disjoint-band replication.

Train a fresh option-world model per seed, then test whether option mediation
still works on those independently trained world-model heads:

```bash
PYTHONPATH=src python3 -m homesocial.option_world_mediation_replication --checkpoint runs/bc_stochastic_body.weights.npz --seeds 9901 9902 --horizon 6 --state-policy cycle --resource-ecology rich --random-model-control
```

Run the same option-world plus mediation pipeline across multiple independently
trained base body checkpoints:

```bash
PYTHONPATH=src python3 -m homesocial.option_body_mediation_replication --checkpoints runs/bc_m8_stochastic_persistent_seed12.weights.npz runs/bc_m8_stochastic_persistent_seed13.weights.npz --seeds 9912 9913 --horizon 6 --state-policy cycle --resource-ecology rich --random-model-control
```

Run a larger local sweep once the fixed-world diagnostic is healthy:

```bash
PYTHONPATH=src python3 -m homesocial.recurrent_ac --episodes 2000 --batch-size 32 --ppo-epochs 3 --eval-episodes 50 --hidden-size 256 --max-steps 160 --conditions grounded_teacher silent_teacher --log-every 100
```

## Test

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```
