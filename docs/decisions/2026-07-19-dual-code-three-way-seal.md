# Dual-code three-way seed-1 seal

Date: 2026-07-19
Status: sealed before any interpreted delayed-task training run

## Immutable source point

Implementation and preregistration commit: `e3e0d9e`
(`add delayed dual-code semantic childhood`).

The two literal run commands, budgets, endpoints, gates, and stop rules are in
`docs/decisions/2026-07-19-dual-code-three-way-preregistration.md` at that
commit. This sealing record changes documentation only; the locked runs use
the same source code as `e3e0d9e`.

## Preflight evidence available before sealing

- All 223 tests passed on the Mac Metal backend.
- `git diff --check` passed.
- Write-enabled and write-disabled hidden-64/binding-16 models had identical
  41-tensor initialization and 105,930 parameters; recurrent-only remained
  85,304 parameters.
- An isolated bodily consequence loss had nonzero gradients through token
  embedding, token GRU, and binding-value projection only when writes were
  enabled.
- Whole-sequence, stepwise, and copied-carry execution agreed within numerical
  tolerance.
- The fixed delayed protocol was verified as four outgoing primitive ticks plus
  a six-tick, center/NORTH, final-WAIT padding return with actor/entropy weight
  zero for the forced transition.
- The task's terminal consume option was verified to use a one-step planner
  score; inspect's imagined second action was restricted to the forced return.
- Post-label erasure/reassignment was verified to occur before the first fresh
  bank read, followed by replay of the identical padding return observation.
- Calibration over 5,000 held-out worlds left three nonlinguistic object rules
  at chance. A learner-visible, quota-aware lexical oracle solved 1,000/1,000
  held-out trials without timeout or death.

## Non-result engineering artifacts

The earlier seed-91, hidden-16, binding-4, 83-tick smoke automatically wrote
six-trial behavioral/audit rows under
`runs/organism/probe17_threeway_memory_smoke`. It differs from the locked
architecture, task timing, body initialization, seed, budget, and evaluation
size. It was used only to exercise wiring and was not used to choose promotion
thresholds or interpret the delayed-task hypothesis.

No seed-1 delayed-task checkpoint, life CSV, or 600-trial evaluation existed at
the moment this seal was written.
