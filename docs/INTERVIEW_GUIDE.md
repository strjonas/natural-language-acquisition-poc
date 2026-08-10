# Interview guide

## 30-second version

> I built an AI-assisted research environment for asking whether an embodied
> agent can learn facts about its own individual body and use them in
> communication. In the strongest experiment, each simulated organism had
> unknown metabolic parameters and only occasional internal readings. A small
> online recursive estimator reduced body-state error by about 71% versus a
> control with the identical readings but no persistent self-model, and improved
> correct need reports by 13 percentage points over five seed blocks. I designed
> and directed the research and used coding agents substantially for
> implementation. It is a constrained simulation result, not natural language
> or consciousness, but it taught me a lot about partial observability, online
> system identification, causal controls, and honest experimental design.

## 90-second version

> The larger project asks how language could become grounded in an embodied
> control loop rather than merely learned from text. I started with a small
> homeostatic gridworld: an agent has food, water, energy, and safety variables,
> acts in a partially observed world, and can interact with a teacher. The
> project went through many failed probes, which revealed that our supposed
> “self-model” was only a species model: every body had identical constants, so
> there was literally nothing individual to learn.
>
> I changed the experimental question in a preregistered way. Each life now has
> hidden individual metabolic or absorption rates and sparse interoceptive
> readings. The learner uses a 21-parameter recursive least-squares estimator to
> attribute prediction residuals to its own body parameters and carry that
> evidence between readings. Against a `snap` control receiving exactly the same
> readings, it reduced body error from 0.045 to 0.013 and raised lowest-need
> report accuracy from 81.5% to 94.6%. Shuffled readings harm it, the null world
> produces zero false correction, and moved-ground-truth worlds test whether it
> identifies metabolism versus absorption rather than always giving one answer.
>
> The important limitation is that this is standard RLS in a hand-designed
> causal template and the language is only a three-word report policy. My next
> research step would be a generated compositional mini-language sharing one
> recurrent state with prediction and control, compared against matched
> decoupled language.

## Your role, stated accurately

Say this early and without apology:

> I provided the conception, philosophical and technical direction,
> experimental questions, orchestration, and interpretation. I used AI coding
> agents extensively for implementation, testing, and documentation. I treated
> that as a reason to demand stronger executable tests and causal controls, not
> as a reason to claim I manually wrote every line.

Do not say “I built every component myself.” Also do not say “the AI built it”
as though research direction, falsification criteria, interpretation, and
responsibility were irrelevant.

## Likely technical questions

### Why RLS?

The body parameters are constant within a life, observations arrive online,
and the local prediction sensitivities are available. RLS accumulates the
second-order structure needed to distinguish correlated causes. The simpler
normalized LMS learner predicted state well but confused metabolism with
absorption; that contrast is itself a useful result.

### What is the strongest control?

`Snap`. It receives the same interoceptive readings and is corrected to truth
at each reading, but it does not learn persistent individual parameters. The
gap after those moments isolates what carrying a self-model adds. Shuffled
readings, a null world, and moved causal ground truth address different
shortcuts.

### Why not call this a robot result?

It is robotics-relevant simulation and system-identification work, not a
physical-robot demonstration. Sensor noise, actuator uncertainty, contact,
real-time constraints, and hardware safety are still untested. That is exactly
why a HiWi role would be valuable: it offers the bridge from controlled causal
experiments to embodied systems.

### Is it novel?

The ingredients are not individually novel. The defensible contribution is the
specific test: hidden individual body dynamics, identical-evidence controls,
and one model carrying evidence into constrained self-report. The larger
grounded-language program is a research direction, not a completed novelty
claim.

### What failed?

Several recurrent learned-self approaches failed identity or load-bearing
gates. A structure-discovery probe recovered useful latent factors but failed
four of five locked gates. A self-uncertainty representation was calibrated but
not behaviorally useful in its original ecology. These failures drove the
individual-body setup rather than being erased from the record.

### What would you do next on a real robot?

Choose one slowly varying hidden parameter—wheel radius, payload-dependent
motor response, battery discharge, or gripper bias—provide intermittent trusted
calibration events, and compare RLS with a reset/snapshot baseline. Keep safety
control outside the learner. Only after identification works would I add a
human-readable report whose correctness changes assistance or maintenance.

## Good final sentence

> I am not presenting this as a finished theory of self or language. I am
> presenting it as evidence that I can turn a vague embodied-AI question into a
> falsifiable simulator, learn from negative experiments, implement causal
> controls, and state exactly what a result does and does not show.
