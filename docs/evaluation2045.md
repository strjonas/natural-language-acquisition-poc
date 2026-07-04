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

Later 2026-06-27 update: direct option-branch world-model training now works as
a substrate test: horizon-6 final need MSE improved from `0.111690` to
`0.005782`. The follow-on communication audit improved full future-state
reconstruction but only modestly improved trend language, so the next gate is
better option diversity and balanced future-change coverage before scaling the
language head.

The broader random-state collector plus current/future/delta option features
improved trend communication to `0.6213` versus `0.4513` random on 375 balanced
option samples. This is progress, but still too rest-heavy to treat as solved;
future probes need more balanced causes of food, water, rest, wait, and danger
changes.

Renewable resources were tested as a minimal ecology change. They improved
water branch coverage and trend margin slightly (`0.6237` versus `0.4368`
random), but food remained rare and full-state reconstruction worsened. The
next probe should add richer resource ecology rather than simply keeping one
resource object alive.

Rich resource ecology improved the actual coverage issue: rich+cycle eval has
`seek_food:14/14/14`, `seek_water:28/28/28`, and `rest:58/58/58` versus
standard random `seek_food:2/2/2`, `seek_water:6/6/6`, and `rest:85/85/85`.
Trend communication reached `0.6606` versus `0.5079` random. The next gate is
replication and anti-shortcut controls, because the random control also improves
in richer option structure.

First anti-shortcut control passed directionally: on the rich+cycle checkpoint,
trend accuracy fell from `0.6606` to `0.5060` when the future-current delta
block was shuffled and to `0.3527` when its sign was negated. Zeroing or
shuffling current/future blocks also hurt, so the message is not a clean
delta-only code; it appears to depend on the full current/future self-state
feature. The next gate remains seed replication plus harder environments where
option identity and time cannot carry much trend information.

Three-seed replication passed the main direction: trained original trend
communication averaged `0.6585`, random-model control averaged `0.5185`, and
option-majority control stayed at `0.3333`. Negating the future-current delta
collapsed trained trend to `0.3431` on average. The option world model itself
replicated, improving final-need MSE from `0.1246` to `0.0141` and trend
prediction from `0.2279` to `0.8314`. This is now a replicated local
self-change communication result, but not yet a final self-language result:
the random feature baseline remains above chance, so the next gate should make
feature shortcuts harder with more varied trajectories and independent body
model seeds.

Option mediation is now a stronger use test. A receiver that only sees compact
messages for each possible option chose the best future-body option at `0.7827`
accuracy versus `0.5522` for random latent features and `0.2000` for the
balanced target-majority baseline. Negating the self-change delta collapsed
trained choice accuracy to `0.1916` and increased regret from `0.0251` to
`0.1525`. This moves the result from "communicates a label" toward "a learned
self-change message can guide a body-relevant decision." It is still one seed
and the grouped collector is slow, so the next gate is to optimize collection
and replicate mediation across body-model seeds.

Source reuse now makes mediation replication practical and tighter: the grouped
option states are collected once, then featurized through trained and random
models. Across three mediation seeds on the same option-world checkpoint,
trained choice averaged `0.7936`, random averaged `0.5812`, and target-majority
was `0.2000`. Negating the self-change delta dropped trained choice to
`0.2120` with regret `0.1501`. This replicates message-mediated self-change
use inside one checkpoint; independent body/world checkpoints are still
required before claiming robustness.

Independent option-world mediation now passes a bounded two-seed gate. Each
seed loaded the same base body checkpoint, trained a fresh rich-ecology
horizon-6 option-world model, then ran matched-source mediation. The world
model replicated (`final_need_mse 0.1257 -> 0.0141`, trend `0.2266 -> 0.8440`).
Mediation also replicated: trained choice averaged `0.7853`, random `0.5597`,
target-majority `0.2000`, and negated-delta trained choice `0.1864` with
regret `0.1535`. The remaining robustness gap is now independent base body
checkpoints and richer/noisier option ecologies.

Independent base-body mediation also passes on the older m8 stochastic
persistent checkpoints. Across body seeds 12 and 13, trained choice averaged
`0.7472`, random averaged `0.5858`, target-majority was `0.2000`, and
negated-delta trained choice fell to `0.1590`. Fresh option-world training on
those bodies improved final-need MSE from about `0.1377` to `0.0192` and trend
from `0.1659` to `0.7686`. This is a significant robustness win. The random
latent baseline is still high, so the next gate should target richer/noisier
option ecologies and controls that reduce generic route/feature regularity.

Delta-only mediation is the first anti-shortcut refinement against that random
baseline. Restricting the receiver to the predicted future-current self-change
vector kept trained choice high (`0.7790`) while reducing random choice from
the prior full-feature `0.5812` to `0.5161`. Shuffling delta collapsed trained
choice to `0.2027`; negating delta collapsed it to `0.0738`. This supports the
claim that the useful message is carried by self-change, but the random
delta-only control is still above chance on some seeds. The next environment
change should make option outcomes less stereotyped.

Noisy-branch mediation is the first version of that environment change. With
`--option-action-noise 0.3`, each option branch sometimes takes another valid
body action, and the receiver sees only predicted self-change for the actual
sampled action sequence. Across two fresh noisy option-world seeds, trained
choice averaged `0.5020`, random averaged `0.2974`, and target-majority stayed
at `0.2000`. Shuffling trained delta fell to `0.2049`; negating it fell to
`0.0908`. This sharply reduces the random shortcut, but trained performance
also drops from the deterministic-option result. The likely next requirement is
stronger/noisier world-model training, not a larger receiver.

Calibration runs narrowed that diagnosis. A larger best-option value gap
(`0.02`) raised `0.3`-noise trained choice only to `0.5200`, and doubling noisy
world-model samples/epochs reached only `0.5261`. Lowering branch noise to
`0.15` raised trained choice to `0.5670` over two seeds while keeping
negated-delta collapse (`0.1117`), but random also rose to `0.3883`.
Training the world model at `0.3` noise and mediating at `0.15` was not a clear
win because negated trained delta stayed above majority (`0.2250`). The next
change should improve action-outcome diversity or train the self model with a
direct intervention-aware objective, not merely tune noise probability.

The new self-model rank control separates self-model quality from receiver
quality. At `0.15` branch noise, the trained message receiver chose correctly
at `0.5536`, while direct predicted-future self-state ranking reached `0.5268`;
the random receiver was `0.3482`, but random predicted-future ranking was only
`0.2321`. At `0.3` noise, trained direct ranking (`0.4940`) slightly exceeded
the message receiver (`0.4701`), while random direct ranking fell below
majority (`0.1726`) even though the random receiver stayed high (`0.3521`).
This means the current bottleneck is mixed: trained self-model rankings are
only moderate under noisy branches, and the learned receiver can still exploit
regularities not captured by the direct self-state rank control.

Direct rank fine-tuning is the first improvement that changes this picture. A
four-epoch grouped branch-ranking update at `0.15` noise raised direct trained
self-model choice to `0.6436` over two seeds while random self-model rank stayed
near majority (`0.2062`). The message receiver also improved modestly to
`0.5871`, and shuffling/negating trained delta still collapsed choice
(`0.2005` / `0.0907`). The caveat is that the old world-trend metric dropped to
`0.5130`, so this objective improves branch choice ranking more than general
world-model calibration. The next objective should combine rank pressure with
world-dynamics preservation, then improve the message bottleneck so compact
communication can match the stronger self-model rank signal.

Rank fine-tuning with dynamics replay partially fixes that caveat. Adding a
`0.25` replay-weighted option-world loss during rank fine-tune kept direct
trained self-model choice high (`0.6207`) and improved world-trend accuracy
from `0.5130` to `0.6175`, with lower final-need MSE (`0.0279`). Message choice
held at `0.5867`, and negated-delta trained choice collapsed further (`0.0660`).
The tradeoff is that random message choice rose to `0.3709`, so dynamics replay
helps preserve self/world calibration but does not solve receiver leakage.

Self-model-targeted communication addresses that leakage directly. Instead of
training the message receiver on the oracle best branch, `--mediation-target-mode
self_model` trains it to communicate the model's own predicted best future
self-state. With rank fine-tuning plus dynamics replay, trained message choice
stayed useful (`0.5631`), direct trained self-model rank stayed high (`0.6288`),
and random message choice fell from the oracle-target `0.3709` to `0.1775`.
This is the strongest anti-leakage communication result so far: the message is
now tied to the agent's self-model forecast rather than a supervised answer key.
The remaining gap is expressivity/compression: trained messages still trail the
direct self-model rank by about `0.066`.

The first message-capacity sweep did not close that gap. Configurable
`--message-slots` and `--message-vocabulary` now exist, but one-seed probes with
3 slots, 8-symbol vocabulary, and doubled mediation epochs all stayed below the
2-slot/4-symbol self-model-target baseline (`0.5631`). Random remained mostly
controlled, so the failure is not shortcut leakage; it is likely discrete
sender/receiver optimization. The next communication step should change the
training procedure, for example soft-message distillation or an auxiliary
self-score reconstruction loss, rather than simply adding symbols.

Soft-message training was the first such procedure tested. It trains through
probability messages and still evaluates hard messages. At temperature `0.8`,
trained hard-message choice fell to `0.4709`; at `0.35`, it reached only
`0.5096`, both below the hard straight-through baseline. Random stayed low, so
the issue is a soft-to-hard mismatch rather than leakage. The next attempt
should preserve hard-message training while adding an auxiliary semantic loss
that forces messages to carry self-score structure.

The first auxiliary semantic loss is now implemented but did not beat the
baseline. A receiver-side score head reconstructs the sender's predicted option
scores from hard messages. Weights `0.1` and `0.03` kept random controlled
(`0.1164` and `0.1968`) but trained message choice stayed around `0.541`, below
the `0.5631` self-model-target baseline. This suggests the issue is not just
missing score supervision at the receiver hidden state. The next communication
attempt should be staged: pretrain or distill a hard-message code against
self-model scores, then fine-tune the choice receiver.

Staged hard-message score pretraining is the first promising version of that
idea. Pretraining discrete messages to reconstruct self-model option scores
before choice training reached `0.5851` and `0.5813` trained-message choice on
two probes, above the prior `0.5631` hard-message baseline, while negated-delta
choice still collapsed. A second 40-epoch seed landed at `0.5368`, so this is a
real lead but not yet a stable solution. Paired no-pretrain controls now put
the selected pretrain average at `0.5677` versus `0.5547` without pretraining.
On seed `10043`, 15 pretrain epochs beat 0, 5, and 30 epochs (`0.5813` versus
`0.5469`, `0.5547`, and `0.5703`). This supports staged shaping of the message
code, but the effect is too small and seed-variable to treat as the final
communication mechanism.

Pairwise score-rank shaping was tested next with `--score-rank-weight`. It did
not solve the bottleneck: rank weight `0.2` hurt seed `10043` (`0.5516`), and
rank weight `0.05` reached `0.5750` on seed `10043` and `0.5868` on seed
`10041`, roughly matching but not beating score pretraining in a robust way.
Receiver score-distribution distillation was then tested with
`--score-distillation-weight`; it also stayed below scorepre15 on seed `10043`
(`0.5781` at weight `0.2`, `0.5719` at weight `0.05`, versus `0.5813`). The
current evidence points away from simple simultaneous auxiliary losses and
toward staged or discrete-code-specific optimization.

The first staged structural alternative, `--frozen-receiver-epochs`, also failed
on seed `10043`: five frozen-code receiver epochs reached `0.5625`, and thirty
epochs reached `0.5484`, both below scorepre15. The next useful change should
constrain message-code stability itself rather than only changing the receiver
schedule.

Light hard-message commitment pressure is a better lead. With
`--message-commitment-weight 0.005`, seed `10043` improved from `0.5813` to
`0.5906`, and seed `10041` improved from the prior `0.5851` scorepre result to
`0.5934`. The remaining paired seed `10042` also improved from `0.5368` to
`0.5487`. Across the three paired seeds, selected scorepre averaged `0.5677`
trained choice and `0.1602` random choice; commitment `0.005` averaged `0.5776`
trained choice and `0.1494` random choice. A larger weight `0.02` hurt, so the
useful regime is narrow. This is the closest compact self-message result so
far, though it still trails direct self-rank.

The commitment effect also survives a stricter independent-body check on two
older m8 stochastic body checkpoints. Matched noisy self-model-targeted runs
improved from `0.5382` to `0.5459` trained-message choice and lowered random
choice from `0.2208` to `0.2131`. The gain is small and regret did not improve,
while direct self-rank stayed much higher at `0.6295`. This confirms that
commitment helps directionally, but the cross-body bottleneck is now compact
message expressivity/stability rather than self-model ranking.

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
