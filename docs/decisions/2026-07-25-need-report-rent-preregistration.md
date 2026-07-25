# Need-report rent preregistration

Date: 2026-07-25
Status: locked before implementation or evaluation

## 0. Why this experiment and not the preregistered next step

`docs/STATE.md` named a fixed-stream matched update-response audit as the next
experiment. It is not run. Operating rule 2 of `DIRECTION_2026-07-12.md` is
binding:

> Any single approach gets at most 3 negative probes or 2 days, whichever comes
> first. Then the negative result propagates UP one level (question the
> substrate), never down (add a patch).

Probes 45, 46 and 47 are three consecutive negatives on one approach —
rebalancing the world model's continual food/water calibration. The named next
step is a fourth probe one level further down (isolating the sampler from
policy feedback inside the replay update). Operating rule 1, the altitude rule,
independently forbids it: its distance to any goal-level metric is four levels
of indirection.

The blocked gate is also not on the critical path. It is a *magnitude* gate on
predicted need deltas. What a self-report needs from the bodily forecast is
*ordering* — which need is worst — and ordering is already sealed at 100% over
a ten-tick horizon, with lived food/water identity decodable at 100%. The
calibration line is therefore closed here, not paused.

Altitude statement for this experiment, per operating rule 1: it moves
**self-report fidelity** and **language acquisition**, and is the repository's
gate G3.

## 1. Claim under test

> An organism whose own bodily state is hidden from its senses learns, in the
> loop and from survival consequences alone, to infer that state and to emit a
> token utterance naming it, such that a caregiver who cannot see the body
> gives the right help.

Falsifiable content: the utterance tracks a hidden variable, is generated over
the full closed vocabulary rather than selected from a supervised map,
collapses under interventions on the internal channel, generalizes to unseen
bodily situations, and pays rent in survival.

No claim about consciousness, phenomenology, or understanding is made or
implied by any outcome of this experiment.

## 2. Substrate: the report island

A new island mode, default off. All existing behavior is untouched.

- **Masked interoception.** The four need values are zeroed in the learner's
  observation vector on every tick except tick 0 of a life. Tick 0 is the birth
  reading: the true needs are visible exactly once. The hidden state is
  therefore an exact function of observable history — birth reading, elapsed
  ticks, and the organism's own consumption events — and inferable in
  principle, which the recovery-line post-mortem requires of any substrate.
- **Randomized birth.** Food, water and energy are drawn independently and
  uniformly from a discrete grid at each life. Safety starts at 1.0 and is not
  reportable.
- **Scenery only.** The world contains no consumable objects. Help from the
  caregiver is the only source of restoration, so language is necessary by
  construction.
- **Scarce periodic help.** Every `help_period` ticks the caregiver grants
  exactly one help, determined by the last utterance it heard from the agent:
  a need word selects the resource, anything else grants nothing. Supply is
  calibrated to slightly exceed total metabolic demand, so a correct
  allocation survives and a misallocation is paid for in body.
- **Perceptible portions.** Food and water help arrives as a small or large
  portion, distinguishable only by the object's surface identity, never by an
  interoceptive channel. This makes a blind fixed-ratio schedule insufficient
  and forces integration of one's own history.
- **Embodied uptake.** Help appears as an object at the agent's own cell; the
  agent must still act (consume, or rest at shelter) to receive it. Free
  unsheltered rest is disabled in this mode, so energy also depends on help.
- **The caregiver never sees the body.** Its only evidence about the agent's
  state is what the agent says.

## 3. Mechanism: a generated utterance

The organism gains one head: per-slot logits over the **entire closed
vocabulary** (52 tokens), for `report_slots` slots, read from the same shared
recurrent state as policy, value and every world-model head. At every tick the
organism samples an utterance from that head in parallel with its motor action.
Saying nothing is available as padding.

The joint policy is factored: `log pi(action) + sum_i log pi(token_i)`, trained
by the same advantage as the motor policy, with its own entropy coefficient.

Binding constraint, checked in code and in tests: **no loss term ever receives
the true bodily state, the correct word, the caregiver's response, or any
simulator fact as a target for this head.** The only path from being right to
the parameters is: right word -> right help -> body restored -> survival ->
advantage. Behavior cloning is not used for speech in any condition.

## 4. Feasibility gates (locked, run before any training)

Exact-dynamics policies on the same environment, 400 lives each, fixed seeds.
An oracle reads the true needs and names the lowest; the blind policies do not
read the body at all.

| # | Requirement |
|---|---|
| F1 | Oracle survival >= 95% |
| F2 | Mute survival <= 5% |
| F3 | Uniform random need word: survival at least 20 points below oracle |
| F4 | Best fixed blind rhythm over a locked family: survival at least 20 points below oracle |
| F5 | Oracle mean viability exceeds every blind policy's by >= 0.05 |

F4 is the decisive one. If any schedule that ignores the body matches the
oracle, the substrate does not require self-knowledge and must be redesigned
before training — the same failure that invalidated the recovery line.
Environment constants (metabolism, help period, portion sizes, life length)
may be calibrated against these feasibility policies *before* the gates are
declared met, exactly as gate B1 calibrated the island's difficulty. They are
frozen once training starts.

## 5. Training gates (locked)

One online lifelong run per condition, fixed budget, no grid search. Fidelity
is measured only on ticks at least 25 after birth, so the birth reading cannot
be echoed.

| # | Requirement |
|---|---|
| T1 | Held-out report fidelity (utterance names the lowest need) >= 60%, against 33.3% chance |
| T2 | Held-out survival >= 80% |
| T3 | Fidelity at ticks >= 200 after birth is within 10 points of overall fidelity |

If T1 fails at the locked budget, the negative result propagates up to the
substrate. It does not license tuning the entropy coefficient, the head width,
the help period, or the model size.

## 6. Audit battery (locked)

The battery is the project's methodological signature and is applied
unchanged. All audits are read-only, on held-out seeds, with no parameter
updates.

1. **Fidelity above chance and above perception.** Report fidelity versus the
   1/3 chance rate, and versus a decoder trained on the *masked observation
   vector alone* to predict the lowest need. The latter must be at chance:
   the answer is not in the senses.
2. **Rent.** Survival and mean viability under: grounded reports; scrambled
   reports (the caregiver receives a uniformly random need word instead of what
   was said, matched in traffic); mute (the caregiver hears nothing); and a
   fixed word. Grounded must exceed every control by >= 15 points of survival.
3. **Internal-channel intervention.** Zero, shuffle across time, and freeze the
   recurrent core state feeding the report head. Fidelity must collapse toward
   chance while the observation stream is unchanged.
4. **Counterfactual body.** Replay identical observable histories in which only
   the portion actually received differs, and measure whether the report
   follows the resulting true bodily ordering. This separates an integrator
   from a clock.
5. **Generalization.** Held-out birth-state combinations excluded from
   training, and a portion schedule not seen during training.
6. **Persistence.** Fidelity as a function of ticks since the birth reading,
   reported as a curve, to show that the report is maintained rather than
   echoed.

## 7. Stop rule

The experiment stops at the locked training budget. If T1-T3 pass, the audit
battery runs once and is reported in full, including any audit that fails. If
T1 fails, no rescue is attempted in this line; the result is recorded and the
substrate is questioned.

No larger compute, external data, or generated-data request is made by this
preregistration. Everything specified here runs locally on Apple Metal.

## 8. Claim boundary

A full pass would establish, at miniature scale, a persistent self-model that
is causally grounded, continually maintained, and truthfully communicated in a
learned token language under survival pressure. It would not establish
consciousness, sentience, phenomenal experience, or open-ended language.
