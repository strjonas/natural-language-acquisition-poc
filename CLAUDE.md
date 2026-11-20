# Working in this repository

Build an embodied learner whose persistent self-knowledge supports prediction,
action and acquired, productive communication. Establish these abilities by
intervention and generalization, not fluent output or terminology alone.

Read the current target and capability status at the top of [STATE](docs/STATE.md),
then only the code and decision records relevant to the task. The original
[vision](docs/VISION_AND_STATUS.md) supplies context. Do not reread the entire
archive or start another roadmap review on every turn.

This file governs working method; STATE records the current frontier. Dated
roadmaps are not mandatory execution orders. Historical protocols and results
stay unchanged. Rationale: [initial method decision](docs/decisions/2026-10-21-integration-and-budget-guidance.md)
and [research review](docs/decisions/2026-10-21-guidance-research-review.md).

## Environment

- Use `.venv/bin/python`; the machine's bare Python may lack test dependencies.
- Use `PYTHONPATH=src` for module invocations. MLX runs on Apple Silicon; some
  newer experiments use NumPy. Measure runtime for the actual workload.
- Full regression: `.venv/bin/python -m pytest -q`. Use checks appropriate to
  the change; documentation edits do not need a training run or full regression.

## Research integrity

1. **Development and confirmation are different.** Pilots and integration
   prototypes may iterate on development data. Before a confirmatory run, lock
   the mechanism, endpoints, budget, sample and failure rule in a preregistration;
   disclose prior development and use separate evaluation data. Preserve failed
   gates. Failure to establish a claim is not proof of impossibility. Revisiting
   a closed setting needs a reason and a new protocol, not renamed thresholds.
2. **Match evidence to the claim.** Causal claims need interventions with
   predicted effects and controls for alternative explanations. An ablation need
   not destroy all competence. Lesions test dependence within an implementation;
   they do not prove an architecture necessary. Broader comparisons need credible
   alternatives with comparable observations, history, tuning and stated budgets.
   Random, oracle and diagnostic arms alone do not establish general superiority.
3. **Quantify uncertainty.** Use independent seeds/blocks and paired comparisons
   where appropriate; report effect sizes and intervals. Five is the default
   confirmatory replication count, not a guarantee of adequate precision. Plan
   sample size for the decision and variability; keep existing locked samples.
   A pilot may use fewer runs. Distinguish developmental-parent replication from
   downstream seeds; never treat many episodes from one parent as independent
   training replications.
4. **Keep learning sources explicit.** The main language claim uses consequence
   rather than direct linguistic-correctness rewards. Predicting observed tokens
   is an auxiliary learning objective already used here, not that policy reward.
   Record supervision and privileged information; a diagnostic oracle or supplied
   answer does not become acquired competence. Preserve existing protocol-specific
   data restrictions and disclose any change of training assumptions.
5. **Keep results reproducible.** Decision records are append-only. Most `runs/`
   output is ignored: retain compact machine-readable evidence plus configuration,
   seed and source/checkpoint identifiers for headline results, alongside their
   interpretation and limits. Keep reproduction commands runnable.

## Choosing and completing work

- **Keep a concrete target.** STATE names the next observable agent capability,
  its current status and the runnable baseline. Keep a versioned evaluation
  scenario across iterations, including generalization beyond development cases.
  If the task or metric changes, explain why and retain the old comparison;
  gains in a different world are not automatically cumulative progress.
- **Choose the useful next step, then execute it.** Briefly record the current
  blocker, the experiment or build that can resolve it, its budget and the next
  decision. Small tests can prevent expensive mistakes; an integration prototype
  can reveal interactions isolated tests miss. No separate planning document is
  needed for routine work. Carry successful components into the agent path or
  explicitly record why they are only diagnostic instruments.
- **Test the connection.** If better knowledge should improve behavior, verify
  that the policy or listener consumes it and can respond differently. An oracle
  can reveal opportunity, but a hand-written reference is not necessarily an
  upper bound. Before replacing a learner, check relevant established methods
  and reusable implementations; do not conduct a new literature review per tweak.
- **Review at a meaningful checkpoint.** If refinement does not advance the
  target or resolve its blocker, compare integration, a budget change and a
  different mechanism. Unrelated robustness properties need not be prerequisites.
  Keep structured, neural and hybrid alternatives open. Neither a few failures
  nor the newest successful probe settles the architecture.
- **Finish with an actionable handover.** Update STATE with what the agent can
  now do, what the evidence establishes, the artifact/command, and the next
  concrete action. Explicitly mark unfinished and untested parts. A probe count,
  passing gate count or rewritten plan is not a capability gain. Keep guidance
  brief; do not expand it instead of doing the research.

## Budget and scale

A small environment need not imply a short training run or a small learner.
Do not require low-budget success before testing a scale-dependent hypothesis.
Specify the suspected limitation and what changes: training duration, capacity,
experience, task richness or inference compute. Check learning-signal and
implementation sanity, measure throughput, set a resource cap and evaluation
schedule, and save resumable checkpoints. Small proxies are useful only to the
extent that their conclusions transfer; test that assumption when it matters.

Track training and held-out development curves, including continuous measures
as well as pass/fail gates. Reserve a sustained arm when delayed generalization
is a plausible hypothesis; do not stop all runs at the first plateau. An early
null neither proves a ceiling nor guarantees eventual grokking. A confirmatory
budget extension is a new experiment. Final confirmation data stays untouched
during selection. Resource limits bound a conclusion, not the possible learner.

## Engineering and known traps

Keep new mechanisms modular and preserve reproducible baselines. Avoid adding
experimental knobs to the legacy `OrganismConfig` or enlarging `train.py`; wire
new modules into a shared runnable loop where relevant.

- Causal parameters initialize deterministically. Vary `seed_base`, not just
  `mx.random.seed`; see `test_structured_causal_development_actually_varies_with_its_seed`.
- A 1,000,000 seed stride leaked developmental worlds into planner evaluation.
  Use `causal_replication_seed_base`; retain `_check_seed_isolation` guards.
- Homeostatic clipping can re-anchor a broken belief. Intervene on the state
  and compare evidence correction against a matched clipping-only baseline.
- A plausible explanation is not a causal result: probe33's 6x gradient
  inflation was real but did not explain the failure under control.
- Check numerical documentation against code and artifacts; do not copy stale
  vocabulary sizes or model/configuration counts.

Before waiting on a long run, give an ETA and uncertainty based on measured unit
cost and contention. An overrun calls for a progress check, not an automatic
failure verdict. Persist progress per unit and avoid output filters that hide
updates. Prefer completing a replicate early while respecting the fixed design.
