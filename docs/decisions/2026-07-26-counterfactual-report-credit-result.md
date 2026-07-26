# Counterfactual report-credit result

Date: 2026-07-26
Status: complete negative result; report-optimizer line closed
Preregistration: `2026-07-26-counterfactual-report-credit-preregistration.md`
Artifacts: `runs/organism/probe50_counterfactual_report_credit/`

## Result

The default-off COMA-style report critic was trained from ordinary lived
returns.  It saw the shared recurrent state, the other sampled token, and slot
identity; it never saw true needs, correct words, grant identity, listener
parse, simulator events, or counterfactual outcomes.  All 52 alternatives were
scored in one forward pass and the actor used a stopped-gradient per-slot
counterfactual baseline.

| Metric | Probe49 legacy credit | Probe50 counterfactual credit | Gate |
|---|---:|---:|---:|
| Held-out report fidelity | 0.3435 | **0.3292** | >=0.60 |
| Held-out survival | 0.0500 | **0.0150** | >=0.80 |
| Mean life steps | 131.39 | **92.84** | — |
| Scrambled survival | 0.0200 | **0.0400** | grounded >= control +0.15 |
| Perceptible-fork report change | 0.0345 | **0.0000** | causal direction required |
| Perceptible-fork body following | 0.0044 | **0.0000** | causal direction required |

The treatment makes heard need words almost perfectly uniform (food 0.3363,
water 0.3300, energy 0.3337), but independent of the actual hidden state.
Zero, shuffle, and freeze interventions remain around chance rather than
selectively destroying an above-chance report.  Scrambling improves survival.

The matched oracle-listener motor diagnostic still reaches 94% survival and
>=99.45% uptake for every need.  The task and motor pathway remain viable.

## Representation diagnostic

| Input | Balanced accuracy |
|---|---:|
| Probe50 recurrent state | 0.4934 |
| Probe50 masked observation | 0.3552 |
| Probe49 recurrent state | **0.6404** |
| Probe49 masked observation | 0.3285 |

The counterfactual critic neither reads out the self-state nor preserves its
strongest form.  The remaining above-observation signal does not reach speech.

## Decision

Per the locked stop rule, the report-output optimizer line is closed.  Do not
try entropy, critic-width, target-horizon, replay, vocabulary, loss-weight, or
another token-credit variant.

The next architecture must move up a level:

1. grounded lexical development, where word meanings are acquired from
   external resource/help situations without revealing the organism's current
   body; or
2. a learned social-consequence planner that uses the already demonstrated
   self-state to imagine how utterances change caregiver action and future
   viability.

No larger compute or generated data is justified.  The local bottleneck is
architectural, not scale.

