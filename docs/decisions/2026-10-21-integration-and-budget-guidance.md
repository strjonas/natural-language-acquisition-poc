# Balancing experiments, integration and training budget

Date: 2026-10-21. Status: **working-method decision, not an experimental result.**

The user requested durable guidance against repeated narrow refinement while
preserving careful hypothesis tests, and explicitly raised integration and scale
as possible requirements. The operating rules now live in
[CLAUDE.md](../../CLAUDE.md), also reached through [AGENTS.md](../../AGENTS.md).

## Evidence and limits

- The [August vision review](../VISION_AND_STATUS.md) identified the gap between
  specialized probes and a developmental language learner. The
  [October direction](2026-10-04-self-in-the-action-loop-direction.md) recorded
  the limited behavioral effects of the request-word line. This supports
  reviewing experiment selection; it does not show that all those probes
  were unnecessary.
- [Probe69](2026-08-18-belief-scaling-result.md) found that a neural estimator
  approximately matched the structured estimator at 64 times the original
  training ticks. This limits earlier small-budget conclusions. It is a
  supervised estimator comparison, not evidence that scaling alone produces
  grounded language or a general self-model.
- [Probe80](2026-10-05-active-self-discovery-result.md) establishes large benefits
  from its self-posterior acting in an actuator environment, under its locked
  comparisons. It has no language or spatial integration. Its gains and the
  older request-word gains come from different experiments; their difference
  is not a controlled estimate of changing just one architectural connection.
- [Power et al., Grokking](https://arxiv.org/abs/2201.02177) demonstrate delayed
  generalization in small algorithmic tasks. This motivates leaving room for
  sustained training in a relevant budget experiment. It neither predicts
  grokking here nor licenses indefinite training or arbitrary model growth.

## Decision

Use small experiments when their outcomes change a consequential decision. Use
an integration prototype when the unresolved question is about interacting
components. Use a budget experiment when training duration, capacity or evidence
may be limiting the result. Choose the least costly design that can actually
answer the question; the smallest possible run may be uninformative.

Maintain a runnable agent path alongside diagnostic instruments. At planned
review points, distinguish gains in scientific knowledge from gains in agent
capability, and reconsider local refinement when neither advances the decision.
Integration may itself be exploratory and incomplete. It need not await every
robustness property, nor does it justify skipping controls or building the
whole proposed system on untested assumptions.

Preserve locked confirmatory tests and their failures. Scope a negative result
to its task, mechanism, information, budget and endpoint. A materially different
test needs an explicit rationale and fresh protocol; a renamed repeat does not
reopen a closed line. Development pilots may iterate on separate data and must
be disclosed as development when confirming a later claim.

## What changes, and what stays open

This decision supersedes blanket bans on larger compute until earlier phases
pass, mandatory execution by probe number, and automatic promotion of a failed
configuration into a family-wide prohibition. It leaves historical protocols,
results, data-source constraints and the language-reward rule intact.

The proposed Probe81–84 questions remain useful candidates. Change detection
is not automatically a prerequisite for stationary language experiments, and
spatial integration is distinct from connecting action, prediction and
communication within a simpler shared loop. Dependencies should be justified
by the intended experiment.

No architecture, scale threshold, fixed allocation of effort, or new experiment
is selected by this decision. Which learner will scale, whether the posterior
can be amortized successfully, and how much integration is needed for productive
language remain empirical questions. Revise the working guidance if its review
cost or sequencing rules themselves begin to obstruct informative work.
