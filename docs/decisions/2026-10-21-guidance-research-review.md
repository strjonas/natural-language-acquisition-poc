# Research review of the working guidance

Date: 2026-10-21. Status: **method refinement, not an agent result.**
Follows [the initial decision](2026-10-21-integration-and-budget-guidance.md).
The user asked whether that revision itself overfit the latest diagnosis and
requested research. Current operating instructions remain in
[CLAUDE.md](../../CLAUDE.md); this rationale is reference material, not a new
mandatory checklist.

## What the review changes

Keep the balance between diagnostics, integration and budget experiments. Shorten
the standing rules and make the positive capability target explicit in STATE.
The initial revision gave agents discretion without a sufficiently concrete
next deliverable and retained some categorical claims about failed gates and
destructive controls. Neither more instructions nor a particular architecture
has been shown to make this research programme succeed.

## External evidence and its scope

- [Gloaguen et al., Evaluating AGENTS.md](https://arxiv.org/html/2602.11988v1):
  generated context files marginally reduced average issue-resolution performance
  and increased cost in their benchmarks; developer-written files produced small
  gains with extra cost. This supports removing redundant instructions, not
  deleting essential scientific constraints. These were coding tasks, not
  longitudinal research agents; our shorter guidance is an informed design
  choice, not a validated intervention in this repository.
- [Patterson et al., Empirical Design in RL](https://arxiv.org/html/2304.01315):
  diagnostic environments, fair baselines, experiment duration and uncertainty
  each serve distinct purposes. A small illustrative task can be valuable, while
  changing tasks or comparing differently tuned systems limits conclusions.
  Applied here: keep a stable target comparison, scope architecture claims, and
  use isolated tests to understand rather than replace integrated evaluation.
- [Agarwal et al., Statistical Precipice](https://arxiv.org/abs/2108.13264):
  few-run RL comparisons can change interpretation when uncertainty is reported.
  Retain effect sizes and intervals; a customary seed count is not sufficient
  assurance of precision. This does not retrospectively change any locked sample.
- [Yang et al., Tensor Programs V](https://arxiv.org/abs/2203.03466):
  hyperparameters can transfer from smaller models under a particular
  parameterization, demonstrated on Transformers and ResNets. Small experiments
  and scaling can therefore complement each other, but proxy transfer requires
  justification. This is not a prescription to adopt that parameterization here.
- [Power et al., Grokking](https://arxiv.org/abs/2201.02177) demonstrate delayed
  generalization in algorithmic tasks. [Schaeffer et al., Emergent Abilities](https://papers.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html)
  show that metric choice can create apparently abrupt improvements in their
  studied settings. Track continuous learning curves alongside success gates
  and allow justified sustained training. Neither paper establishes the presence
  or absence of delayed generalization in this organism.
- [Lin et al., Dynalang](https://proceedings.mlr.press/v235/lin24g.html) combine
  multimodal future prediction with acting from imagined rollouts, and demonstrate
  grounded language generation. This makes shared language/world-model learning
  a concrete reference to examine rather than inventing every component anew.
  It does not establish this project's bodily self-discovery or acquisition
  claim, and its training assumptions must be checked before reuse.

## Repository-specific corrections

Probe80's bandit chooses from aggregate action values rather than conditioning
decisions on a learned hidden-state representation. Its failure is evidence
against that baseline, not against every alternative without an explicit
self-posterior. A causal lesion can show that this implementation uses its
self-model without establishing that no other mechanism could solve the task.
Keep the original nine-gate verdict and its specified comparisons intact.

The no-direct-language-reward rule does not prohibit predicting observed
caregiver tokens: the existing training loop already includes that auxiliary
objective. State the learning sources instead of silently removing useful
learning signals or adding privileged supervision.

The next capability target comes from the existing vision: the same learner
acts on bodily knowledge and acquires communication that generalizes to new
combinations. It does not select posterior, recurrent or hybrid implementation,
fix a token count, or require all uncertainty/change/spatial properties first.
Confirmatory thresholds and budgets still need a concrete experiment design.

Judge this guidance through subsequent work: does a handover resume a runnable
path, does evidence resolve the stated blocker, and does capability improve on
a comparable task? If not, investigate the remaining technical or resource
constraint rather than assuming another documentation revision is the solution.
