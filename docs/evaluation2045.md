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

Adding raw message capacity on top of commitment was tested and failed on the
best m11 seed: 2x4 commitment reached `0.5906`, while 3x4 fell to `0.5781` and
2x8 fell to `0.5375`. The bottleneck is therefore not the number of symbols
alone. It is how the sender learns a stable, usable discrete code.

Commitment timing was also tested. Late-only commitment underperformed on seed
`10043` (`0.5766`), while pretrain-only commitment reached `0.6031` on that
seed, the best single compact-message result so far. Across the three paired
m11 seeds, pretrain-only and constant commitment were essentially tied
(`0.5779` versus `0.5776`), with pretrain-only slightly better on regret but
worse on the negated-delta collapse. This suggests commitment is most useful
while shaping the self-score code, but retaining it during choice training gives
cleaner causal intervention behavior.

Message-code diagnostics now show that this is not a simple code-collapse
problem. On seed `10043`, constant and pretrain-only commitment both used 14 of
16 possible codes with similar entropy (`2.286` versus `2.267`). The remaining
issue is likely code semantics and receiver alignment rather than insufficient
active symbols. The follow-up semantics check agrees: target-code mutual
information was nearly identical (`0.069` versus `0.070` nats), while
choice-code information was only moderately higher (`0.119` versus `0.114`
nats).

A direct branch-code target auxiliary loss was tested and failed to improve the
best seed. Balanced `--code-target-weight 0.05` reached only `0.5734` accuracy,
and `0.005` reached `0.5672`, both below the `0.6031` pretrain-only baseline and
without improving target-code mutual information. Row-pattern diagnostics also
showed that full message tuples are mostly one-off patterns: the baseline used
573 patterns on 640 eval samples with only `0.181` reuse, while a shuffled-delta
control had high target-pattern MI despite only `0.2203` accuracy. The next
useful direction is stable codebook/replay or transfer-style receiver alignment,
not more direct branch-target BCE shaping.

Simple score-codebook replay was then tested with `--message-replay-weight`.
Weights `0.005` and `0.001` reached `0.5797` and `0.5766` accuracy, below the
`0.6031` pretrain-only baseline. This suggests that the issue is not merely
sender drift after score pretraining; the receiver still needs a better
alignment or transfer objective.

The first held-out receiver test is a partial positive transfer result. A fresh
receiver trained against the frozen best sender reached `0.5734` accuracy after
70 epochs and `0.5703` after 140 epochs, versus `0.6031` for the co-trained
receiver and `0.2109` for a random frozen sender. The code is therefore
learnable by another receiver, but still not as reusable as a stable convention
should be.

Training several receiver heads against one sender with `--receiver-copies` was
also tested. Two and three receiver copies reduced primary accuracy to `0.5625`
and `0.5594`, and held-out transfer stayed near `0.567` rather than improving
over the single-receiver held-out result. This argues against simple
multi-receiver pressure as the next convention solution.

Simplifying the code directly also exposed a tradeoff. Lowering balance pressure
to `0.005` dropped primary/held-out accuracy to `0.5563`/`0.5297`. A single-slot
`1x8` code increased row-pattern reuse from `0.181` to `0.572`, but accuracy was
only `0.5641` primary and `0.5453` held-out. The issue is therefore not just
reuse; reusable symbols must preserve the self-model ranking semantics.

Directly imposing reusable rank symbols was also tested with
`--score-rank-code-weight`. Weights `0.005` and `0.02` reached only
`0.5797`/`0.5625` and `0.5563`/`0.5375` primary/held-out accuracy. This
preserved target-code information better than the `1x8` simplification but did
not close the transfer gap.

Several follow-up convention probes were negative. Reducing receiver capacity
to 32 or 64 hidden units dropped held-out transfer to `0.5437` and `0.5500`.
Training the fresh receiver with score-distribution distillation reached
`0.5703`, roughly matching longer receiver training but not beating the
`0.5734` baseline. New sender-side value-code and score-distillation pressures
also underperformed: `0.5547`/`0.5266` and `0.5828`/`0.5531`
primary/held-out accuracy. These failures are useful: the bottleneck is not raw
receiver capacity or a missing scalar-value auxiliary loss. The next meaningful
step should change the interaction structure so conventions must survive across
independent senders/receivers. A first repeated-partner-turnover implementation
was also negative: resetting receivers every 10 epochs reached only
`0.5547`/`0.4953` primary/held-out accuracy, and one mid-run reset reached
`0.5578`/`0.5344`. Resetting the same receiver pathway is therefore too
destabilizing; the next test should be a true population option game with
several independent senders and a shared receiver, plus held-out sender/receiver
transfer.

That true population option game was implemented next and was also negative in
its naive form. Four independent senders trained against one shared receiver
reached `0.5422-0.5750` primary accuracy and `0.5312-0.5516` held-out receiver
accuracy, below the `0.6031`/`0.5734` single-sender baseline. Sender agreement
pressure worsened accuracy, a two-sender population still underperformed, and
doubling mediation epochs did not recover the baseline. The next attempt should
be staged population training: preserve a competent sender/receiver convention
first, then introduce new senders or receivers gradually.

Staged population training is now implemented and is a partial positive. It
preserves the base sender exactly (`0.6031` primary, `0.5734` held-out), while a
new sender trained against the fixed receiver reaches `0.5781` primary accuracy.
However, that new sender transfers poorly to a fresh receiver (`0.5359`), and
longer new-sender training worsens both primary and held-out transfer. The next
variant should add explicit transfer pressure while introducing new senders,
for example a small receiver ensemble or periodic fresh-receiver distillation.

The first staged transfer-pressure variant improved the specific weak point.
Training the new sender through two auxiliary receivers learned from the base
sender raised new-sender held-out transfer from `0.5359` to `0.5516` while
leaving the base sender at `0.6031`/`0.5734`. Stronger transfer weight and a
larger four-receiver ensemble were worse. This is not solved, but it is the
first sign that explicit transfer pressure can move new senders toward a more
receiver-independent self-message convention.

Base-sender token imitation was tested as a complementary convention-learning
pressure. Imitation alone improved the new sender's fixed-receiver accuracy to
`0.5828` but reduced fresh-receiver transfer to `0.5266`. Combining imitation
with the two-receiver transfer ensemble gave a better balance (`0.5766`
primary, `0.5500` held-out), close to the transfer-only held-out result while
keeping more task accuracy. The convention is still not fully shared; the next
step should improve transfer without merely copying surface tokens.

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

## Current Experimental State

The current narrow target is not full self-report yet. It is the prerequisite:
can a learned message about action-conditioned self-consequences remain usable
when a new sender and a new listener are introduced? Staged population training
now preserves a competent base sender/receiver, then admits a new sender.
Auxiliary transfer receivers helped, and training the original convention with
two receiver copies helped further: new-sender held-out listener accuracy rose
from mean `0.5334` to `0.5464` across seeds `10043-10045`. This is a modest
but real directional improvement, not a solved language system.

Negative results are equally important. Token imitation improved fixed-receiver
accuracy but hurt transfer. Receiver-logit distillation and score-head
reconstruction/ranking also hurt transfer in the tested settings. The evidence
currently favors developmental listener diversity over post-hoc imitation or
distillation.

Added diagnostics for whether option messages carry two grounded
self-consequence fields: positive predicted body-state delta and relative
state-local option value. The best staged receiver-diversity run carries both
signals and loses them under `shuffle_delta`, but naive fixed-slot supervision
for those fields reduced held-out listener transfer. This keeps the next target
focused on emergent compositional pressure, not hand-assigned symbol slots.
Receiver-decoded field pressure was also negative: independent field receivers
trained on the base sender reduced new-sender held-out transfer at weights
`0.1` and `0.02`. The next version should make these fields operationally useful
inside the interaction itself, rather than adding auxiliary field losses.
Added that interaction-level field-use evaluation. A query-conditioned receiver
can use current staged messages to answer relative-value queries well and its
performance degrades under `shuffle_delta`/`negate_delta`, but positive
body-improvement queries remain weak. This is now the next concrete bottleneck:
messages must support useful self-improvement choices, not only relative option
ordering.
Opportunity-aware accounting sharpened this: only about `34%` of states have a
positive-improvement option, and the new sender reaches `0.6636` accuracy on
that subset with positive selected delta, far above shuffled/negated controls
but still well below oracle improvement. Receiver-side opportunity weighting did
not close the gap.
Sender-side field-action receiver pressure also failed as a route forward: it
slightly improved opportunity accuracy but reduced held-out listener transfer
and did not improve selected positive delta. The next intervention should change
the state distribution or main interaction so positive self-improvement reports
matter directly.
That distribution shift is now partially supported: `--min-positive-delta`
filters option-mediation states to those where some option can improve the
current weakest body need. With natural target frequencies on seeds
`10043-10044`, the staged new sender reached mean `0.9123` choice accuracy and
`+0.1338` chosen body delta versus oracle `+0.1517`; a fresh held-out receiver
reached `0.9126` and `+0.1334`. Delta shuffling/negation collapsed chosen delta
below zero. The limitation is target imbalance: target-majority already reaches
`0.8315` and `+0.1075`, while hard target balancing leaves too few opportunity
states. The next step is opportunity-rich but stratified/weighted interaction,
not abandoning the opportunity focus.
That next step is partially implemented with `target_option_resample` and
separate train/eval balance targets. Stratified opportunity training with
natural evaluation reduces target-majority to `0.0267` while preserving useful
new-sender and held-out receiver deltas (`+0.1200` / `+0.1212`, oracle
`+0.1517`). This is a cleaner control substrate than natural imbalance, but it
also lowers raw accuracy and leaves negated-delta less destructive than desired,
so the next gate should strengthen intervention robustness across more diverse
opportunity ecologies.
The first stronger gate is now in place: `reverse_delta_rank` swaps delta
features within each state so high-value options inherit low-value
self-change and vice versa. On the same two seeds it drops new-sender accuracy
to `0.0848` and chosen delta to `-0.1762`; a fresh held-out receiver drops to
`0.1339` and `-0.1570`. This is much cleaner than global negation and supports
the claim that the compact messages carry option-specific self-change rank.
The next research step can move from proving this local causal dependence to
broadening the ecology and interaction loop.
The first broadening attempt is now available through
`--mediation-train-resource-ecologies`. Training on both `standard` and `rich`
opportunity states while evaluating on natural `rich` states improves
intervention robustness: `negate_delta` drops to `-0.1069` chosen delta and
`reverse_delta_rank` to `-0.1973`. It also reduces held-out receiver transfer
from `0.8656` to `0.8330`. This is a useful tradeoff: diversity strengthens
self-change dependence, but the compact option-message channel loses some
within-ecology efficiency. The next move should broaden the interaction format,
not only add more ecology mixes.
The first broader-format attempt is `homesocial.option_dialogue`, a two-turn
proposal/correction protocol. It is useful infrastructure but not yet a win:
on balanced rich opportunity states, final accuracy improved over the first
proposal (`0.4420` versus `0.3367` in the straight-through proposal run), but
chosen body delta remained negative (`-0.0150`). This suggests longer social
interaction needs to be coupled to the stronger rank-finetuned self-model or
made necessary in the environment; adding a second token-level correction turn
alone is not enough.
That coupling now works: optional option-world training and rank finetuning are
wired into `option_dialogue`. With `world_epochs=8` and
`rank_finetune_epochs=8`, two balanced rich-opportunity seeds reach mean final
accuracy `0.6750` and positive chosen delta `+0.0533`; `shuffle_delta`,
`negate_delta`, and `reverse_delta_rank` all collapse below zero chosen delta.
Forced-proposal evaluation now makes that dependency explicit: when the first
proposal is forcibly set to the second-best or worst true option, the original
second-turn channel recovers final accuracy `0.6483` / `0.6310` and positive
chosen delta `+0.0505` / `+0.0430`; shuffled and reverse-rank self-change
channels collapse to negative deltas and near-chance or worse final choices.
This is the strongest current two-turn result: the second message can repair a
bad social proposal by communicating learned action-conditioned self-change.
It remains a compact protocol, not natural language, and has only two-seed
evidence so far.
Repair-proposal training now moves this from evaluation-only correction toward
learned correction pressure. A heavy `repair_weight=1.0` over-weighted repair
on the first seed and hurt the ordinary model-proposal path, but
`repair_weight=0.5` replicated across seeds `10043-10044`: normal-path final
accuracy stayed essentially flat (`0.6750` to `0.6793`), second-best repair
rose `0.6483` to `0.6830`, and worst-proposal repair rose `0.6310` to
`0.6863`; chosen body delta improved most on the worst-proposal case
(`+0.0430` to `+0.0550`). Shuffled and reverse-rank self-change interventions
still produce negative chosen deltas. Next, replace oracle-injected bad
proposals with a learned/social source of incomplete or mistaken proposals.
That replacement is only partially successful so far. `model_runner_up` now
uses the proposal receiver's own second-most likely option as a learned-partner
mistake source. The no-repair dialogue already repairs this on seed `10043`
from `0.1540` proposal accuracy to `0.6587` final accuracy with positive
chosen delta `+0.0446`. Training directly on runner-up repair was negative
(`0.6107` runner-up final accuracy), and a hybrid
`model_runner_up second_best worst` repair setting improved forced final
accuracies but reduced body-delta quality versus the simpler second-best/worst
repair. Keep `model_runner_up` as an audit mode; the next training target
should be a separate partially informed partner whose mistakes are learned,
situated, and homeostatically meaningful.
That separate partner now exists in audit form. `limited_partner` is a frozen
proposal model trained on masked option features; the first diagnostic uses
only the first half of the delta feature vector. Across seeds `10043-10044`,
its proposal accuracy is `0.5130`, and the dialogue second turn repairs this to
`0.6574` final accuracy with positive chosen delta `+0.0482`. The same
limited-partner condition collapses under `shuffle_delta` (`-0.0945`) and
`reverse_delta_rank` (`-0.1499`). Directly training on all limited-partner
proposals was negative on seed `10043`, so keep it as an audit source for now.
Next: repair training should sample only consequential partner mistakes, not
every partial-partner proposal.
That filter now exists (`--repair-only-mistakes`, `--repair-min-regret`) but
did not fix the issue on seed `10043`. Filtered limited-partner repair at
weight `0.5` reduced limited-partner final accuracy to `0.5913`; weight `0.1`
was less damaging (`0.6660`) but still below audit-only (`0.6847`) and also
hurt second-best/worst repair. The next repair-training attempt should freeze
or stage the existing convention, then train only the reply/final correction
path on consequential partner mistakes.
That staged variant is now implemented and tested, but it is negative too.
Freezing first-message/proposal modules and training only reply/final on
consequential limited-partner mistakes reduced limited-partner final accuracy
to `0.5873` in a conservative `3`-epoch setting and to `0.5227` in a stronger
`20`-epoch setting, versus `0.6847` audit-only. The next training direction
should not be another auxiliary repair loss on the same compact protocol; it
should make partial partner proposals part of the actual interaction and score
the final body outcome.
The first main-loop version of that idea also failed: `--train-proposal-mode
limited_partner` made the limited partner's proposal the primary training
context. With proposal loss disabled, normal final accuracy dropped to
`0.4867`; with proposal loss kept at `0.2`, the proposal head recovered
(`0.6580` proposal accuracy) but limited-partner final accuracy was only
`0.5893`, still below audit-only `0.6847`. This strengthens the conclusion
that the next step should leave the compact option-label objective and build an
environment-level partner task scored by downstream body outcome.
That outcome-scored objective now exists in the compact protocol as
`--final-objective outcome`. Main-loop limited-partner training with this
objective did not solve partner repair, but outcome-scored consequential repair
is the first positive partner-training result: over seeds `10043-10044`, normal
dialogue improved from `0.6750` final accuracy and `+0.053317` chosen delta to
`0.6897` and `+0.063918`, with regret falling from `0.060078` to `0.049477`.
Limited-partner final accuracy remained lower than audit-only (`0.6420` versus
`0.6573`), but its chosen delta improved slightly (`+0.051588` versus
`+0.048185`). This says the outcome objective is useful, but the compact
protocol is still not enough for robust social partner repair.
The first environment-level bridge now exists as
`homesocial.situated_partner_dialogue`. It groups actual grid branch rollouts
by state, lets a partial-body partner propose an executable option, and trains
the self-reply/final choice on downstream viability. On two positive-opportunity
seeds (`12001-12002`), the partial-body partner proposal is usually wrong
(`0.0444` accuracy) and harmful (`-0.190652` body delta), while the self-reply
path reaches `0.6482` final accuracy and `+0.127544` chosen delta against a
`+0.149517` oracle. Zeroing outcome features drops chosen delta to `+0.072034`.
This is the strongest current evidence that partner correction can be grounded
in embodied consequence information, but it is still offline over branch
samples; the next step is an online episode loop where partner proposals and
self-replies affect actual trajectories.
That online evaluator now exists and gives a mixed/negative first result. With
`6` committed option steps and all consequential training states, dialogue
reduces branch regret versus the partial partner (`0.035363` versus `0.324147`)
and worsens when outcome features are zeroed (`0.085087` regret), but realized
episode viability is not better: dialogue mean viability is `0.660783` versus
the myopic oracle's `0.661520`, and min viability still reaches `0.0`. The
offline self-reply signal is real, but the next work must train or plan against
online survival, not just branch-choice regret.
That next training variant now exists: trajectory-valued scoring plus a direct
best-option loss. With `trajectory_mean` and `target_weight=0.25`, dialogue
nearly matches the trajectory oracle on branch regret (`0.006252`) and keeps
average minimum viability above zero (`0.004466`), while `zero_outcome` falls
back to `0.121859` regret and zero minimum viability. This is progress, but
termination is still `0.9833`; the bottleneck is now the short-horizon option
set, not only the reply model.
The option/ecology substrate fix produced the first strong online survival
result. With the extended situated option set and renewable rich resources,
dialogue reaches mean viability `0.740067`, average minimum viability
`0.216862`, and truncates `65.00%` of episodes, close to the oracle
(`0.752919`, `0.257745`, `66.67%`). The partial partner remains much worse
(`0.704022`, `0.013009`, `5.00%` truncation), and `zero_outcome` loses most of
the gain. This is now evidence that grounded self-reply can improve actual
online viability when the environment supports reusable regulation.

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
