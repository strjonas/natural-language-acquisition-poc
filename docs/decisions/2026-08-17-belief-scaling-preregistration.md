# Structure against scale: preregistration

Date: 2026-08-17. Status: **locked before the treatment runs.** Survey:
`docs/decisions/2026-08-17-belief-scaling-ceiling-survey.md`. Module:
`src/homesocial/organism/belief_scaling.py`.

Bands: treatment development 1,900,000,000, treatment audit 2,100,000,000 —
disjoint from the survey's 1,500,000,000 / 1,700,000,000 and from every band
previously claimed. `_check_seed_isolation` refuses a sweep whose development
stream could reach an evaluation base, and refuses a budget larger than the
20,000,000 seed stride.

## 1. What is being tested, and what is not

**Tested.** Whether the pattern that organizes this repository's evidence
ladder — five structured passes against nine black-box failures — is confounded
with the 80,000-tick development budget those comparisons were all measured at.

**Not tested, and explicitly not claimed.** Probe54's gate. It is failed, it
stays failed, and no result here promotes it. The harness lock at
`harness.py:1220` is untouched. This probe cannot "pass" probe54 because it does
not run probe54's contrast; it runs a curve whose endpoint is a shape.

This touches a line `docs/STATE.md` closes. Section 0 of the survey states the
challenge and the answer in full; the short form is that the contrary evidence
is inside probe55's own result document, and that a flat curve here strengthens
the closure rather than reopening it.

## 2. Mechanism

No mechanism is added. Both arms already exist and are unmodified:

- `train_explicit_self_belief` — a GRU of width `W` appended to the frozen
  probe52 adult, supervised on developmental body values, `lr` 3e-4.
- `train_structured_causal_self_model` — probe57's constrained causal template,
  `lr` 3e-3.

The only thing that moves is `steps`, and — for the black-box only — `W`. Both
trainers already take `steps` as a free argument; the 80,000 lock lives in the
harness, which this module does not import from and does not touch.

Scoring is `_balanced_accuracy` over `world.lowest_need()`, the same function
object both audits already share, at 200 audit lives.

## 3. Design

**Curve arm** — 5 seeds x {80,000, 320,000, 1,280,000} x {black-box w64,
black-box w256, structured}.

**Extrapolation arm** — 3 seeds x 5,120,000 x black-box w256, plus 1 seed
x 5,120,000 structured as a saturation check.

Width 64 is retained because it is probes 54 and 55's width, and comparability
with the published record is itself a gate (G7).

Both `mx.random.seed` and `seed_base` move with the seed. For the structured
arm the parameters initialize deterministically, so only the world stream can
vary it — `CLAUDE.md`'s first trap — which is why `seed_base` and not
`mx.random.seed` alone carries the seed for both arms.

## 4. Locked gates

Survey values are shown for orientation. They were measured on a disjoint band
at one seed and are **not** thresholds read off the treatment's own lives.

| | gate | survey |
|---|---|---:|
| **G1** | structured@1.28M − structured@80k **< +0.05**, mean over 5 seeds | +0.0158 |
| **G2** | blackbox_w256@1.28M − blackbox_w256@80k **> +0.10**, 95% CI excludes zero, **>= 4/5** seeds individually positive | +0.1526 |
| **G3** | gap(1.28M) **< 0.5 x** gap(80k) at width 256, **>= 4/5** seeds | 0.264x |
| **G4** | zero- and shuffled-belief balanced accuracy **<= 0.40** in **every** cell of **every** seed | <= 0.3338 |
| **G5** | gap(80k) > gap(320k) > gap(1.28M) at width 256, **>= 4/5** seeds | holds |
| **G6** | gap at 5.12M within **[0.005, 0.045]**, mean over 3 seeds, **and** blackbox_w256@5.12M **<** structured@5.12M on **3/3** seeds | predicted 0.025 |
| **G7** | blackbox_w64@80k within **+/-0.05** of probe54's published 0.6219, mean over 5 seeds | 0.5988 |

`gap` is always `structured − black-box` at the same budget and seed.

### G6 is the sharp one

The survey fixed the gap-closure rate at **0.515 per 4x of compute** (measured
0.480 and 0.550) *before* this file was written, and the prediction it yields is
neither "scale wins" nor "structure wins" but **convergence without crossing**:
the black-box approaches the structured level and does not overtake it inside
the range. G6 gates both halves of that. Probe65's practice — fix the crossover
from already-published numbers before the run — is the precedent.

## 5. Failure conditions

- **If G2 fails**, compute is not the explanation for the black-box failures.
  The `STATE.md` closure is then **strengthened**, and that is what gets
  reported. It will not be rescued by width, learning rate, optimizer,
  schedule, loss, or head variants — all of which are already closed and none
  of which this probe is licensed to try.
- **If G1 fails**, both families are compute-limited, the "saturated against
  climbing" framing is wrong, and the survey's central claim about the locked
  budget does not hold. Report it as such.
- **If G6's non-crossing half fails** — the black-box overtakes the structured
  model — the survey's extrapolation was wrong in the direction that most
  favours the v2 architecture, and that must be reported as a *failed
  prediction* rather than folded into a success.
- **If G7 fails**, the instrument does not reproduce the record and no other
  gate may be cited until that is resolved.

Gates are not adjusted after seeing results. Budgets, widths, seeds and the
audit size are fixed by this document.

## 6. Claim boundaries, stated in advance

- **Belief-side only.** Balanced lowest-need accuracy. Nothing here shows a
  better belief reaching a word, and probe68's margin law says it need not:
  probes 59, 60, 62, 64 and 67 each improved a belief and bought no behaviour. A
  black-box that matches the structured model's accuracy inherits that problem
  whole.
- **Nothing about survival, language, the mouth, or the five properties.**
- **One architecture family** (GRU, two widths), **one parent** (the single
  frozen probe52 adult every causal-stage result here shares), **one loss**.
- **Sample efficiency is not a ceiling.** Whatever the treatment returns, the
  structured model reaching 0.926 in 90 seconds is not diminished by a black-box
  reaching it in 16x that. The claim at stake is about the *ceiling*, and the
  two are different quantities.
- **The top of the curve remains unmeasured** above 5,120,000.
