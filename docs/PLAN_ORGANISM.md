> **FROZEN HISTORICAL RECORD.** Superseded by `docs/DIRECTION_2026-07-26.md`.
> Kept as evidence of what was tried. Do not append to it and do not treat any
> plan, status, or number in it as current. Current state: `docs/STATE.md`.

# Organism Implementation Plan (C0-C3)

Date: 2026-07-13
Parent: `DIRECTION_2026-07-12.md` (phases/gates). This file is the concrete
engineering plan. Update by rewriting sections, not appending.

## 0. Principles (binding)

1. One agent, one environment, one training loop. New code goes in the three
   packages below; the ~30 probe scripts are frozen.
2. Language is never rewarded directly. It pays rent only through
   consequences (help arrives, warnings avert harm, answers guide action).
3. The LLM (OpenRouter, `deepseek/deepseek-v4-flash`, reasoning disabled)
   generates offline world content only: the caregiver utterance bank. No live
   LLM calls in the training loop. The LLM never plays the agent.
4. Every headline claim runs through one harness with fixed seeds and the
   audit battery (grounded/silent/shuffled, channel interventions,
   random-model control).
5. Secrets: `OPENROUTER_API_KEY` lives in `.env` (gitignored). Never commit
   keys; never print them to logs.

## 1. Repository layout (new code)

```text
src/homesocial/creole/        # closed language: vocab, situations, bank
  vocab.py                    #   closed vocabulary, tokenizer, OOV validation
  situations.py               #   speech-act schema: (act, slots) -> situation key
  bank.py                     #   UtteranceBank: load, validate, seeded sampling
  generate_bank.py            #   offline OpenRouter generation with cache
src/homesocial/island/        # the creole island environment
  world.py                    #   grid, ecology, recalibrated difficulty
  caregiver.py                #   joint attention, label/warn/answer/ask, help
  api.py                      #   reset/step -> ObsPacket(vision, intero, tokens)
src/homesocial/organism/      # the unified agent
  model.py                    #   encoder + recurrent core + heads (MLX)
  losses.py                   #   world-model + actor-critic + intrinsic losses
  memory.py                   #   sequential replay + persistent episodic buffer
  train.py                    #   THE loop: act -> store -> learn, lifelong
  harness.py                  #   the one benchmark/audit command
tests/creole/ island/ organism/
runs/organism/                # all new artifacts (checkpoints, CSVs, banks)
```

## 2. Workstream A: creole language + utterance bank (do first)

Independent of everything else, cheap, and unblocks B/C design.

- **A1 vocabulary.** ~100 words, utterances 1-5 tokens, SVO-lite creole.
  Categories: objects (water, food, berry, shelter, danger, tree, rock, ...),
  places/relations (near, at, north, south, east, west, here, far),
  needs/feelings (hungry, thirsty, tired, hurt, safe, good, bad),
  verbs (go, take, eat, drink, rest, look, avoid, want, give, feel, have),
  social/logic (you, me, yes, no, what, where, this, not). Fixed token ids;
  `<pad> <eos> <unk>` reserved.
- **A2 situation schema.** Speech acts the caregiver can perform:
  `LABEL(obj)`, `WARN(hazard, place?)`, `ANSWER_WHERE(obj, place)`,
  `ASK_STATE()`, `OFFER(obj)`, `CONFIRM()/DENY()`, `PRAISE()/CORRECT(act)`.
  A situation key is the act plus grounded slot values; keys are enumerable
  from environment state, so the bank has full coverage by construction.
- **A3 bank generation.** For each situation key, the LLM produces N (~8-16)
  surface variants under a hard constraint: only closed-vocab words. A strict
  validator tokenizes and rejects any variant with OOV tokens or length > 5;
  rejected variants are regenerated once, then dropped (template fallback
  guarantees >= 3 variants per key). Output: `runs/organism/utterance_bank.jsonl`
  keyed by situation; runtime sampling is seed-deterministic. Requests are
  cached on disk so regeneration is incremental.
- **A4 tests.** No OOV in bank, coverage (every key >= 3 variants), length
  bounds, deterministic sampling, validator unit tests.

Exit: bank generated and committed (it is small text; commit for
reproducibility), tests green. Est. cost < $1.

## 3. Workstream B: island environment

- **B1 difficulty recalibration.** Port the extended-renewable rich ecology
  into `island/world.py`. Tune metabolism/regrowth so the scripted oracle
  survives >= 95% of 1000-step lives while a random policy dies fast (< 10%
  survival). This removes the "oracle dies 35% of the time" pathology that
  invalidated the recovery line. Gate B1 is a measured table, not a feeling.
- **B2 token observation channel.** `ObsPacket.tokens`: the caregiver's last
  utterance as a padded token-id sequence (len 5). Matched-shape controls
  carry over: silent = all-pad, shuffled = seeded permutation of the bank.
- **B3 caregiver.** Deterministic policy over situations wired to the bank:
  joint attention (labels what the agent points at / approaches), warnings
  before hazards, `ANSWER_WHERE` for asked-about resources (including
  out-of-sight ones), periodic `ASK_STATE`. In C3, help becomes conditional
  on the agent's own utterances being informative and truthful.
- **B4 hidden information.** Object kinds hidden (as in `language_necessary`),
  out-of-sight resource locations only available through language, delayed
  effects. Language must carry information perception cannot.

## 4. Workstream C: organism agent (gate G1)

- **C1 model core (MLX, ~5M params to start).** Encoder: visible-object slots
  + interoception MLP + token-embedding GRU. Core: GRU latent 256-512,
  hidden state carried across episodes (lifelong). Heads: policy, value,
  next-obs, next-interoception, reward, next-caregiver-utterance.
- **C2 online loop.** Sequential replay of fixed-length segments; every k env
  steps run one world-model update and one actor-critic (A2C/PPO) update with
  the predictive heads as auxiliary losses. Dreamer-lite latent imagination is
  a staged upgrade behind a flag, adopted only if model-free + aux losses
  stalls (timeboxed comparison, 3 days).
- **C3 intrinsic motivation.** Learning-progress bonus on the interoception
  prediction head (information gain about own body dynamics), annealed.
- **C4 BC bootstrap.** Allowed as curriculum only; every result states
  with/without; the harness has a `--bc-warmstart` flag so the ablation is one
  command.
- **C5 harness.** One command: lifelong run -> survival metrics + grounded vs
  silent vs shuffled + audit battery + CSV. Fixed seed sets. This is the only
  source of headline numbers.

Gate G1 (from DIRECTION): unified agent matches the patchwork's best online
survival profile on the recalibrated island and shows grounded > silent >
shuffled on held-out seeds. Kill: 2 weeks without beating the
partial-partner-equivalent baseline -> stop, rethink architecture, no
hyperparameter archaeology.

## 5. Workstream D: speaking + rent (gate G3)

Utterance action head over the same vocabulary (sequence of <= 5 tokens,
`<eos>`-terminated). Rent mechanisms: caregiver grants help only when the
request/report is informative and true (verified against hidden state);
partner tasks where the partner cannot see agent internals. Audits: reports
track masked interoception above controls, collapse under internal-channel
interventions, generalize to novel states, and are causally load-bearing.

## 6. Order of work and timeboxes

1. A1-A4 creole + bank (0.5-1 day) — DONE marker goes here when true.
2. B1-B2 island + recalibration + token channel (1-2 days).
3. C1-C2 skeleton + smoke test: loss goes down, agent survives longer than
   random (1-2 days).
4. C training to G1 (timeboxed 2 weeks of sessions).
5. B3-B4 full caregiver -> gate G2 (fast-mapping probes).
6. D -> gate G3.
7. Only then: GPU ask with concrete spec.

Timebox rule from DIRECTION applies inside every step: 3 negative probes or
2 days per approach, then the negative result propagates up a level.

## 7. Risks and mitigations

- **From-scratch RL fails again** (it did in the probe era): mitigations are
  the aux predictive losses, intrinsic learning-progress bonus, easier
  recalibrated ecology, curriculum, and BC bootstrap with ablation. If all
  fail: G1 kill criterion, top-level rethink (that is a finding, not a
  failure).
- **LLM output violates the closed vocabulary**: hard validator + regenerate
  once + template fallback; the bank can never contain OOV.
- **Reasoning-token burn**: `reasoning: {"enabled": false}` verified working.
- **MLX throughput too low**: profile at C2-smoke; target >= 500 env
  steps/sec collected + train updates on M4/32GB; shrink model before
  shrinking the world.
- **Token channel introduces shortcut confounds**: matched-shape silent and
  shuffled controls are wired into the harness from day one (M0 discipline).
