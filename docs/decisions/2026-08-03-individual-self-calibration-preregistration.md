# Probe63: individual self-calibration -- preregistration

Date: 2026-08-03. Status: **preregistered, not yet implemented.** Gates below are
locked. They are not adjusted after seeing results; a failed gate closes what it
gates, per `CLAUDE.md`.

Feasibility record this rests on:
`docs/decisions/2026-08-03-individual-self-ceiling-survey.md`.
Survey artifacts: `runs/organism/probe63_individual_self/ceiling_survey.json`.

## 1. The problem this attacks

`docs/STATE.md` records a standing obstacle, item 2 of "what this result is still
missing":

> The **hand-coded probe53 filter is still better** (99.96%, zero error). This is
> a lossy approximation of a closed form the designer already had. Nothing yet
> shows the learned model doing what the analytic filter cannot.

Four consecutive probes have failed to move it. Probe62 finally diagnosed why,
and the diagnosis is not about any of the mechanisms tried. Every organism in
this ecology burns fuel at exactly the species rate: `food_metabolism` is 0.008
for all of them, in every life. So the "self-model" is a model of *bodies in
general*. Its content is a physics the designer knows in closed form -- which is
exactly why probe53's filter is exact and unbeatable, and why nothing learned can
beat it. **There has been nothing individual to learn.**

The organism also gets exactly one interoceptive reading, at birth, and then
flies blind for 400 ticks. So even if there were a fact about itself, it would be
unknowable in principle rather than merely unknown.

Probe63 changes both, with three default-inert levers already built and guarded
(`tests/test_individual_self.py`, 10 guards):

| lever | what it makes individual |
|---|---|
| `metabolic_spread` | how fast this body spends each need |
| `uptake_spread` | how much good the same granted help does it |
| `interoception_probability` | whether it can ever find out |

The first two are phase B1's "systematically biased body the current model cannot
represent"; the third is B1's "return interoception intermittently and
unpredictably".

## 2. What the survey established, before any gate was locked

Following probe60's binding instruction. All numbers at 40 lives, one shared
history per condition, seed base 930,000,000.

1. **Headroom is large and, unlike probe62's, recoverable.** At
   `metabolic_spread` 0.60 the species filter carries body error 0.0833 and names
   the truly-lowest need on 0.660 of ticks; the *same filter* on this life's true
   constants carries error **0.0000** and names it on **1.000**. So 34.0 points
   are available and every one of them is reachable -- the entire remaining error
   of this repository's self-model in that ecology is individuality.
2. **Evidence alone does not collect it.** At reading rate 0.03, snapping the
   belief to truth whenever a reading arrives reaches 0.0478 error and 0.817
   accuracy; carrying a *model of one's own rates* between readings reaches
   0.0254 and 0.908. B1 warns its first gate is "satisfiable by clipping"; this
   is the measured size of that warning, +9.2 points.
3. **Survival is finally sensitive.** Closed loop at spread 0.60, rate 0.10:
   species filter 0.525, self-modelling 0.675, oracle 0.700. Probes 59 and 60
   moved survival by +1.4 and +0.0 points against locked +5-point gates; this
   moves it +15.0 against a ceiling of +17.5.
4. **Localization is identifiable, but only in a model that can express the
   truth.** Pooled least squares on the organism's own parameter sensitivities
   recovers metabolic share 0.909 in a metabolism world and uptake share 0.622 in
   an absorption world, with exactly zero correction in a world where nothing is
   individual. With a self-model whose absorption parameter was *shared* across
   needs, the absorption world instead misattributed 0.703 of its mass to
   metabolism -- mis-specification presenting as misattribution.

## 3. The mechanism

New module `src/homesocial/organism/self_calibration.py`, default off, no knob
added to `OrganismConfig`, every existing parameter path left bit-identical --
the `causal_self.py` pattern required by `CLAUDE.md` §5.

**The reflexive quantity.** The organism carries one copy of its own body filter
per parameter it could be wrong about, each with exactly that one constant
perturbed. The column-wise difference from the unperturbed filter is

    J[need, parameter] = d(predicted body) / d(log parameter)

This is not a fact about where the body is. It is a fact about how the
organism's own *model* would respond if a particular belief about itself were
wrong. It is what has to be computed to answer "what about me is miscalibrated",
and it is built only from the organism's own model and its own history.

**The parameters.** Seven constants -- `food_metabolism`, `water_metabolism`,
`energy_metabolism`, `move_energy_metabolism`, `portion_small`, `portion_large`,
`shock_size` -- carried **per need**, so that "how well I absorb food" and "how
well I absorb water" can be different facts about this body. The survey showed
that a shared parameterization cannot represent a per-need truth and misattributes
it; that finding is what fixes this choice.

**The update.** Correction `m` starts at exactly zero, which is the species body.
At each reading, with residual `r = reading - predicted`, normalized LMS:

    m[i] <- m[i] + eta * J[i] * r[i] / (||J[i]||^2 + epsilon)

`eta = 0.5`, `epsilon = 1e-8`, both fixed a priori. NLMS is chosen because it is
a standard online rule that is stable for any `eta` in (0, 2) and therefore needs
no per-ecology scale tuning; `eta` will not be swept. Belief then snaps to the
reading and the filter bank restarts, and between readings the belief propagates
with parameters `theta * exp(m)`.

This is online, per-life, initialized from the species prior, and never told
which parameter is at fault.

### 3b. Amendment: a second arm, added before the treatment run

Recorded in full because it happened after the mechanism was first written.

A six-life smoke test of the NLMS rule showed it localizing well in the
metabolism world (metabolic share 0.90) and **not at all** in the absorption
world (uptake share 0.075 against a measured ceiling of 0.622). The reason is
structural rather than incidental: NLMS moves along the *current* sensitivity
direction, so it splits a residual in proportion to how sensitive each parameter
is at that moment and has no way to separate two constants whose effects arrive
together. Help arrives on a clock here, so absorption and metabolism both look
like a deficit proportional to elapsed time.

That exposes a defect in this preregistration as first written: **G3's thresholds
were derived from a pooled least-squares ceiling, while the mechanism gated
against them was NLMS** -- a different estimator class. G3 as written therefore
asks whether a greedy rule reaches 73% of a least-squares ceiling, which is not
the question the probe is about.

The correction is to add a second arm, **recursive least squares**, which is the
*online form of the very estimator the ceiling was measured with*: same ridge,
no forgetting, no step size. It is added here, before the five-seed run, and:

- **no locked threshold is changed**, in either direction;
- **both arms are scored against the identical gates**, and both results are
  reported whatever they are;
- NLMS remains the arm named in §3 and its failures are reported as failures,
  not withdrawn.

A six-life smoke test of the recursive arm was also seen before this amendment
was written, and is disclosed for the same reason: it localized at 0.833 in the
metabolism world and 0.380 in the absorption world -- above and below the locked
0.70 and 0.45 respectively. The thresholds were left where they were.

One further honesty note about a label. The `individual` tier receives **no**
readings, so it is "what knowing yourself is worth with no evidence", not a
global ceiling; an arm with both a self-model and readings can and does beat it.
It is reported as context, never as a bound.

## 4. Conditions

Treatment ecology: `metabolic_spread` 0.60, `uptake_spread` 0.0,
`interoception_probability` 0.03, 40 lives per seed, **5 seeds** at disjoint
bases (`930,000,000 + 2,000,000 * k`, none overlapping probe62's 880,000,000
band or any developmental band).

Comparison tiers, all sharing one history so no contrast is a policy difference:

| tier | belief |
|---|---|
| `population` | species filter. Today's repository. |
| `snap` | species filter, corrected to truth at every reading. **The control B1 names.** |
| `learned` | the mechanism above. |
| `individual` | species filter on this life's true constants. Ceiling. |

Additional worlds: **absorption** (`metabolic_spread` 0.0, `uptake_spread` 0.60),
**both** (0.60 / 0.60), and **null** (0.0 / 0.0).

## 5. Locked gates

Each gate is scored per seed; "passes" means the stated fraction of the 5 seeds.

- **G1 -- corrigibility.** `learned` body error <= **0.75 x** `snap` body error,
  on **>= 4/5** seeds. The analytic ceiling is 0.53x, so this asks the learner to
  capture about half of what modelling itself can buy over being corrected.
- **G2 -- the report.** `learned` named-need accuracy >= `snap` + **4.0 points**,
  on **>= 4/5** seeds. Stated against `snap` rather than `population` on purpose:
  snapping alone already buys +15.7 over the species filter, so a gate against
  `population` would pass on evidence and say nothing about a self-model.
- **G3 -- localization.** In the metabolism world, metabolic share of the learned
  correction >= **0.70**; in the absorption world, uptake share >= **0.45**; each
  on **>= 4/5** seeds. Ceilings are 0.909 and 0.622.
- **G4 -- no false discovery.** In the **null** world, total |m| <= **10%** of the
  treatment world's total, and `learned` body error within **0.005** of
  `population`, on **>= 4/5** seeds. An organism whose body is typical must find
  nothing and lose nothing.
- **G5 -- shuffled readings.** Readings replaced by another life's body, matched
  in count and timing. `learned` body error must **not** beat `population`, on
  **>= 4/5** seeds. If a self-model improves on nonsense about itself, it was
  never reading the evidence.
- **G6 -- rate recovery.** Mean |recovered scale - true scale| < the species
  prior error (spread/2 = **0.30**), on **>= 4/5** seeds.

## 6. Failure rule

- **G1 or G2 fails -> the mechanism is closed.** Do not rescue it by sweeping
  `eta`, changing the update rule, raising the reading rate, adding parameters,
  or re-tuning the spread. Each is forbidden here by name.
- **G5 fails -> the result is void**, whatever else passed.
- **G3 fails while G1, G2, G4, G5, G6 pass** -> the corrigibility claim stands
  and the *discovery* claim is closed. These are reported separately and the
  distinction is stated in the result doc.
- **G4 or G6 fails** -> reported as a failed gate against the corresponding
  claim; it does not by itself void G1/G2.

## 7. What this cannot show, stated in advance

- It is **parameter** self-discovery, not **structure** discovery. The organism
  finds the values of its own bodily constants; it is still told that it has
  three needs and what kinds of thing act on them. Probe61 attempted structure
  discovery and is closed; this does not reopen it and must not be described as
  having done so.
- Interoception is granted free and unprompted. The organism does not choose to
  look at itself, so nothing here is about the *value* of self-inspection.
- The ecology is changed relative to v1, so no number here is comparable to
  probe57's. The null world is included precisely so the change can be shown to
  be inert when its levers are off.
- Survival is reported as context and is **not gated**. Probes 59 and 60
  established that this help loop flattens belief-side differences; the survey
  shows survival separating `population` from the rest but *not* `snap` from
  `learned`, so it cannot carry the claim this probe is about.
