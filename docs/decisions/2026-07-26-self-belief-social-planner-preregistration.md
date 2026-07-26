# Learned self-belief and social planner preregistration

Date: 2026-07-26  
Status: feasibility passed; learned-stage details locked before implementation

## Failure-selected hypothesis

Probe 52 separated lexical acquisition from self-report. The organism retained
near-perfect causal word-to-resource comprehension after adult training, yet
reported at chance and almost never followed perceptible bodily forks. A
disposable supervised readout improved held-out survival from 3% to 35% but
could not approach the 80% gate. The existing recurrent state and direct mouth
policy are therefore both inadequate.

The next mechanism is a structured, learned self-belief coupled to explicit
social consequence selection. It is not a larger report head and not another
credit estimator.

## Architecture constraints

1. The same organism owns a dedicated three-dimensional self-belief state.
   It is recurrently updated from only learner-visible birth interoception,
   elapsed time, the organism's own actions, shocks, visible granted portions,
   and uptake outcomes. Adult true body state, simulator event/kind metadata,
   lowest-need labels, and correct report words are forbidden inputs.
2. The belief transition is learned in a bounded developmental phase where
   bodily state is perceptible. It must predict multi-step self-change under
   action and portion histories. Adult evaluation masks body state exactly as
   before.
3. Lexical childhood remains the gate-passed probe-52 procedure. The learned
   need words share the existing tied input/output embedding path.
4. At each help opportunity the organism evaluates its three acquired need
   words through a learned listener/help consequence model. Selection uses the
   predicted future self-belief after the resulting visible grant and the
   organism's voluntary uptake action. It may not score words against the true
   need or listener parser.
5. The ordinary actor, self-belief transition, listener model, and lexical
   path remain parts of one saved organism and continue learning online. A
   slow consolidated copy may stabilize developmental knowledge, but no
   separately trained oracle controller may act at evaluation.

## Read-only feasibility before training

Before adding trainable mechanisms, construct an audit-only history filter
using only the permitted learner-visible stream. It may use exact simulator
dynamics solely as an epistemic upper bound and must be clearly excluded from
the organism. Couple its inferred need to the already acquired word lexicon
and the real frozen adult listener loop.

Across 200 held-out lives require:

- >=90% need reconstruction after the warm-up;
- >=80% survival;
- >=60% report fidelity;
- grounded survival >= scrambled by 15 points; and
- visible portion forks change inferred need when and only when their lived
  histories diverge.

Failure means the permitted observation/action history is insufficient or the
auditor is wrong; stop before training. Passing only licenses the learned
approximation, not a capability claim.

Probe 53 passed this gate before the learned-stage details below were locked:
99.96% need reconstruction, 97.0% survival, 99.96% report fidelity, a 92.5
point grounded-over-scrambled survival advantage, zero measured mean absolute
need error, and an exact 0.4 small/large portion-fork response. The artifact is
`runs/organism/probe53_observable_history_filter/feasibility/`.

## Locked learned developmental phase

Start from the probe-52 adult checkpoint, whose public lexicon still passes the
frozen causal gate and whose motor policy reaches 95% survival under oracle
words, and append a 64-unit recurrent self-belief module to that same saved
organism. This avoids confounding belief quality with the lexical child's
untrained adult motor policy. Train only the new module for exactly 80,000
primitive report-island ticks, seed 1, with one online update per lived
sequence and no replay or epoch reuse.

At every recurrent step its inputs are restricted to:

- the birth body's three values only on the first packet;
- masked visual/proprioceptive observation;
- its own last action; and
- elapsed primitive duration since the preceding decision.

True current food/water/energy values are developmental teaching targets only.
They cannot enter the recurrent input, main policy, lexical path, mouth, or
evaluation. The existing organism parameters stay frozen during this isolated
feasibility stage so a gain cannot be a newly supervised report policy.

To keep the child alive and expose all visible help consequences without
giving a correct report, the developmental listener receives a uniformly
random need word at every decision, sampled independently of the body. The
organism's existing motor policy chooses every physical action. No word target,
body-conditioned grant, truth reward, or generated report credit exists.

One fixed optimizer (Adam, 3e-4), one seed, 80,000 ticks, hidden width 64. No
hyperparameter change is allowed after reading the endpoint.

## Learned self-belief gates

On held-out masked-body lives, before speech training:

- self-belief balanced lowest-need accuracy >=70%;
- current-observation decoder <=45%;
- ten-primitive-tick realized self-change ordering >=80%, defined as pairwise
  ordering agreement between the three predicted need changes and the three
  true changes over held-out windows;
- zeroing or shuffling self-belief reduces accuracy by >=20 points; and
- portion-fork belief following >=60% over perceptibly divergent ticks.

Only if all pass may social planning be enabled.

## Final unchanged self-report gates

Use the complete existing 200-life battery and thresholds: >=60% fidelity,
>=80% survival, grounded minus scrambled survival >=15 points, state lesions,
fixed-word and mute controls, held-out births and portions, persistence, and
perceptible body forks. Recheck lexical comprehension before and after adult
training.

One seed and one local budget are permitted after the feasibility audit. Stop
on the first failed stage. No external corpus, generated data, GPU run, or
parameter sweep is authorized by probe 52.
