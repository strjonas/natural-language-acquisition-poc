# Final Implementation Plan

## Principle

Build toward **measurable self-like organization**, not claimed consciousness. The deliverable is an embodied recurrent agent that:

- regulates hidden interoceptive needs,
- uses grounded language only when it helps action,
- distinguishes self-caused from world/teacher-caused change,
- forms causally important latent state for control,
- reports its own state only if reports track hidden variables and causal history.

Language is never directly rewarded.

---

## M0: Fix Experimental Validity First

**Files:** `src/homesocial/recurrent_ac.py`, `src/homesocial/experiment.py`, `src/homesocial/env.py`, `src/homesocial/observations.py`, `tests/test_recurrent_ac.py`, new `tests/test_language_necessity.py`, new `preregistration.md`.

1. Decouple teacher behavior from model input shape.

```python
teacher_mode: Literal["grounded", "silent", "masked", "shuffled", "wrong"]
include_language_channel: bool
objective_mode: Literal["homeostatic", "external_task"]
```

Silent, grounded, shuffled, wrong, and masked agents must have identical observation/input dimensions. Silent/masked fills the language channel with a null/OOV token.

2. Add teacher factory.

```python
def build_teacher(mode: TeacherMode, seed: int) -> TeacherLike:
    ...
```

Modes:

- `grounded`: current situated teacher.
- `silent`: no useful utterance.
- `masked`: teacher exists, encoder masks language.
- `shuffled`: deterministic seed-based utterance permutation.
- `wrong`: plausible but harmful/wrong advice.

3. Add reduced diagnostic audit.

Test the existing `language_necessary` setup with bounded enumeration:

- hidden object kind,
- candidate object slot,
- fixed diagnostic layout,
- ask-then-act oracle,
- blind one/two-step silent micro-policies.

This does **not** prove full POMDP optimality. It only proves that, in the diagnostic state family, teacher language has causal value.

4. Add `preregistration.md`.

Before each milestone run, write:

- exact seeds,
- metrics,
- CI method,
- gate margins,
- allowed deviations.

Post-run changes must be logged as deviations.

**M0 gate:**

- input shape invariant across teacher modes,
- hidden kind does not leak through masked observations,
- grounded oracle beats best blind diagnostic controls by predeclared margin,
- shuffled/wrong collapse toward silent.

---

## M1: Behavior Cloning Warmstart

**Files:** new `src/homesocial/imitation.py`; edit `src/homesocial/recurrent_ac.py`, `src/homesocial/experiment.py`; new `tests/test_imitation.py`.

Interfaces:

```python
def collect_expert_episodes(config, n: int, seed: int, noise: float = 0.05) -> ExpertDataset: ...
def train_bc(model_config, dataset: ExpertDataset, checkpoint_path: str) -> BCResult: ...
def evaluate_checkpoint(path: str, teacher_mode: TeacherMode, seeds: list[int]) -> list[EpisodeStats]: ...
```

Use existing `TeacherFollowingAgent`, `collect_episode`, and `pad_trajectories`.

Train:

- action cross-entropy,
- existing next-observation head,
- next-interoception head,
- reward-delta head,
- teacher-utterance head.

Start with 1k expert episodes. Increase to 5k only if curves are still improving.

**Gate:**

After `ASK`, the policy follows teacher advice above blind baseline, improves viability on held-out seeds, and fails when language is masked/shuffled.

---

## M2: PPO Curriculum

**Files:** new `src/homesocial/curriculum.py`; edit `src/homesocial/recurrent_ac.py`; new `tests/test_curriculum_monotone.py`.

Add CLI/config fields:

```bash
--init-from
--save-checkpoint
--teacher-mode
--eval-teacher-mode
--objective-mode
--include-language-channel
```

Curriculum:

1. visible-kind resource control,
2. fixed-position `language_necessary`,
3. randomized-position `language_necessary`,
4. hidden-kind reshuffled per seed.

Advance only when the prior stage passes.

**Gate:**

Grounded BC+PPO beats silent and shuffled/wrong on paired held-out seeds for:

- viability,
- resource use,
- danger avoidance.

PPO does not need to beat BC. It must preserve and generalize grounded language use.

---

## M3: Language Advantage Suite

**Files:** new `src/homesocial/language_eval.py`, new `tests/test_teacher_signal_adversarial.py`.

Run current-valid controls:

- eval-time language masking,
- shuffled utterance mapping,
- wrong advice,
- held-out hidden-kind assignments,
- frequency-matched nonsense utterance if needed.

Defer synonym and spatial-relocation tests until the encoder/teacher supports lexical variants and spatial advice. With the current one-hot encoder, unseen synonyms collapse to OOV and the test is invalid.

**Gate:**

The trained grounded agent should degrade under masked/wrong/shuffled advice and outperform silent on held-out hidden-kind cases.

---

## M4: Minimal Self-Battery

**Files:** new `src/homesocial/self_battery.py`, `src/homesocial/probes.py`, `src/homesocial/interventions.py`, new `tests/test_self_battery_controls.py`.

Use frozen recurrent latents and existing consequence heads. Do not add a new world model yet.

Run every probe under four conditions:

1. grounded,
2. silent,
3. shuffled/wrong,
4. strict no-stakes.

`no-stakes` means reward and termination do not depend on interoceptive needs. Run a strict version with next-need auxiliary loss disabled; optionally run a prediction-only variant separately.

Tests:

- self-action vs exogenous-change attribution,
- k-step interoceptive forecasting,
- self vs teacher/world event distinction,
- threat generalization to new danger surface with same interoceptive signature,
- latent causal centrality by ablating compact latent slices versus matched random slices.

Body/tool-boundary is marked N/A until held-object/tool mechanics exist.

**Gate:**

Report full metric vector with paired CIs. Thesis-supporting pattern:

- grounded > silent ≈ shuffled,
- grounded > no-stakes on causation, interoception, and latent-centrality tests.

If grounded beats silent but not no-stakes, the result is “language helps, homeostatic stakes not shown.”

---

## M5: Multi-Step World Model Only If Needed

**Files:** conditional new `src/homesocial/world_model.py`.

The one-step heads already exist. Add multi-step imagination only if:

- k=1 forecasting works but k>1 fails, or
- reflective counterfactual reports need imagined rollouts.

Interface:

```python
def transition(latent, action) -> next_latent: ...
def imagine(latent, action_seq) -> ViabilityPrediction: ...
```

Do not add Dreamer/Crafter/Mamba/LLM dialogue before the grid proves a real grounded-language effect.

2026-06-27 update: the new option-level counterfactual language probe supports
this gate. Recursive unrolling of the current one-step predictor gives a better
trained-over-random signal than primitive one-step branches, but the result is
underpowered and not yet a stable imagination substrate. The next architecture
step should therefore be direct multi-step option-branch world-model training,
not larger language heads.

---

## M6: Reflective Report With Parrot Guards

**Files:** new `src/homesocial/report_head.py`, new `tests/test_report_calibration.py`.

Train a small decoder on frozen recurrent latents. Reports are structured first:

- current need,
- cause of event,
- predicted consequence.

Example output classes:

```python
NeedReport = Literal["need_water", "need_food", "need_rest", "safe"]
CauseReport = Literal["self_caused", "teacher_caused", "world_caused"]
ConsequenceReport = Literal["water_up", "food_up", "safety_down", "no_change"]
```

Anti-parrot tests:

- hold out the self-relevant latent slice identified in M4,
- shuffled latent labels,
- text-only/base-rate decoder,
- decoupled-text pretraining applied to the report head, not the perception encoder.

Comparison matrix:

1. no language,
2. grounded,
3. report head pretrained on decoupled text,
4. grounded then decoupled text,
5. decoupled text then re-grounded.

**Gate:**

Reports must track hidden interoceptive state and causal history above predeclared baseline-relative margins, and accuracy must drop when the self-relevant latent slice is removed.

---

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest tests.test_env tests.test_qlearning

python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest tests.test_language_necessity
```

MLX-dependent tests should skip cleanly if MLX is unavailable.

---

## Scaling Rule

Stay local through M4 unless M2 and M3 pass. Rent GPUs only after the grid shows a real grounded-language effect.

Expected local scope: GRU policy under roughly 2M parameters on M4 32GB.

Conditional scale-up: 1x A100/4090 for larger recurrent/attention agents or Craftax-like environments, estimated roughly `$500-$2000` per serious sweep. Ask before spending.

---

## First Week

1. Fix `include_language_channel` vs `teacher_mode`.
2. Add teacher ablation factory.
3. Add shape-invariance tests.
4. Write `preregistration.md` for M0.
5. Add reduced `language_necessary` audit.
6. Run M0.
7. If M0 fails, fix the environment before touching learners.

---

## Council Value

The strongest ideas that survived critique are: matched input channels before any comparison, reduced diagnostic audits without overclaiming optimality, strict no-stakes controls, behavior-cloning warmstart before PPO, self-battery before self-report, and decoupled-text tests aimed at the report pathway rather than perception.
