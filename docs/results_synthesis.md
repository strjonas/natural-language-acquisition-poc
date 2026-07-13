# Probe-Era Results Synthesis (through v0-probe-era, 2026-07-13)

One row per established or refuted claim. "Audit" names the strongest control
that supports (or kills) it. Sources: `preregistration.md`,
`grand_architecture_roadmap.md`, `evaluation2045.md` (all frozen).

## Established

| # | Claim | Strongest audit | Replication |
|---|-------|-----------------|-------------|
| 1 | Grounded teacher language is causally actionable in the grid (M0) | Oracle beats blind controls; shuffled/wrong collapse toward silent; matched input shapes | Preregistered gates passed |
| 2 | A recurrent agent infers its own hidden body state from action/outcome history (M7/M8) | Masked interoception; full-history beats latest-only, shuffled, reversed, random-model (next-need MSE 0.0021, lowest-need acc 0.92) | 2 body seeds |
| 3 | The learned self-estimate supports aid requests decodable by another agent (M11) | Receiver picks food/water/rest/avoid from message alone, 0.84 acc; intent-symbol alignment 1.0 across 4 senders | 1 body model, 4 senders |
| 4 | Multi-aspect current self-state is communicable from learned features (M12) | Need MSE 0.0057, dominant-need 0.83 vs random-model control | 1 seed; trend field failed its majority gate |
| 5 | Learned self-change (delta) carries temporal trend information (M13/M14) | Trend 0.84 balanced; zero/shuffle/negate the delta channel degrades to 0.67/0.48 | 1 seed; random control high (0.77) — caveat |
| 6 | Direct multi-step option-branch world-model training predicts own future state (h=6) | Final-need MSE 0.13 -> 0.014; trend 0.23 -> 0.83+ | 3 seeds + fresh heads + 2 independent bodies |
| 7 | Compact self-change messages causally mediate another agent's option choice | Choice 0.78-0.79 vs 0.56-0.58 random, 0.20 majority; negate-delta collapses to ~0.19; delta-only receiver: negate -> 0.07 | 3 mediation seeds, 2 fresh world heads, 2 independent bodies |
| 8 | Reverse-delta-rank is a clean row-wise causal audit | Reversing option-specific self-change rank drives receivers to harmful choices (acc 0.08, delta -0.18) | 2 seeds |
| 9 | Two-turn dialogue can repair bad proposals using learned self-change | Forced second-best/worst proposals repaired to 0.65-0.69 final acc, positive body delta; interventions collapse below zero | 2 seeds |
| 10 | Grounded self-reply improves actual online survival (peak probe-era result) | `situated_partner_dialogue`, extended renewable ecology: viability 0.740 / min 0.217 / 65% truncation vs partner 0.704 / 0.013 / 5%; near oracle 0.753; zero-outcome ablation destroys gain | 2 online seeds, 1 substrate |
| 11 | One-step in-loop rehearsal on exact branch labels helps online survival | Adaptive dialogue 0.739/0.283 vs frozen 0.688/0.127 on same audit | 1 audit; exact simulator labels (not learned) |

## Refuted / failed (do not retry as-is)

| # | Claim | Evidence |
|---|-------|----------|
| F1 | From-scratch RL (tabular, A2C, PPO) learns ask-then-act here | Never succeeded in randomized hidden-kind setting; all competence entered via BC warmstart |
| F2 | Message conventions transfer/align across senders, receivers, populations | Zero-shot swaps fail (M10); population, turnover, staged variants all below single-pair baseline; best staged gain marginal (0.533 -> 0.546) |
| F3 | Message capacity, soft training, auxiliary semantic losses, distillation close the message-vs-direct-rank gap | ~15 negative probes; messages ~0.58 vs direct self-rank ~0.63 |
| F4 | Online adaptation over *learned* values is stable | Persistent updates collapse survival; only value-calibrated local (discarded) rehearsal is mildly positive (+0.005 viability) |
| F5 | Learner-side recovery/risk machinery can create a safety floor | Entire line (kNN gates, critics, rollout labels, recovery policies, constrained selectors, floor calibrators) produced third-decimal deltas and failed cross-checkpoint; root cause: the oracle itself dies in ~35% of episodes — an environment problem |

## Standing design lessons

1. Interventions on the learned self-channel (negate/shuffle/reverse-rank)
   plus matched random-model controls are the project's signature audit —
   carry into everything.
2. Random-feature receivers exploit environmental regularity; always report
   the random-model control, and prefer delta-only/noisy designs that
   suppress it.
3. Negative results propagate UP (fix the substrate), not down (add a patch).
4. Language must pay rent through consequences; direct language rewards and
   fixed-slot code supervision consistently hurt transfer.
