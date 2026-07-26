# Unified-help uptake preregistration

Date: 2026-07-26
Status: locked before implementation, feasibility evaluation, or training

## 0. Altitude and evidence

This experiment moves the goal-level metrics **self-report fidelity** and
**survival** directly.  It is a substrate change after the locked need-report
run failed T1 and T2, not a learner-side hyperparameter rescue.

The failed 200,000-step checkpoint reports the lowest hidden need with 0.45%
fidelity and survives 0% of 200 held-out lives.  A post-hoc read-only audit
then replaced its reports with the true lowest-need word while leaving its
frozen motor policy in control.  Food and water uptake were 218/218 and
271/271, but energy uptake was only 190/1,189 (15.98%); oracle-listener
survival was still 0%.

The substrate therefore required two initially unrewarded conventions to be
discovered together for energy: say `tired` to make shelter appear, then use
`REST` rather than the already learned `CONSUME` response.  Neither behavior
receives useful consequence credit until the other happens.  Food and water
do not have this circular dependency because they share `CONSUME`.

## 1. Claim under test

> Removing the resource-specific motor convention, while preserving embodied
> uptake and every self-report information constraint, is sufficient for
> consequence-only learning to discover truthful reports of all three hidden
> needs.

The treatment does not reveal the body, supervise a word, shape speech reward,
reduce the vocabulary, alter the recurrent model, or give help for free.

## 2. Sole substrate manipulation

Add a default-off `unified_uptake` report-island condition.  In that condition,
food, water, and energy help all remain perceptible objects at the organism's
cell and all require the same voluntary `CONSUME` action.  Consuming the energy
object restores energy and removes it.  Its small/large surface identities,
portion sizes, metabolic dynamics, help clock, shocks, masked interoception,
birth reading, listener rule, vocabulary, report head, optimizer, loss, and
training budget are unchanged.

The legacy condition remains byte-for-byte behaviorally unchanged when the
flag is false.  A test must show that the forced treatment changes energy
uptake from `REST` to `CONSUME` without changing food or water uptake.

## 3. Feasibility seal before learning

Run the existing exact-dynamics F1-F5 harness on 400 headline lives per main
policy and the locked body-blind rhythm family, with `unified_uptake=true` and
all other ecology constants unchanged.

| Gate | Requirement |
|---|---|
| F1 | truthful oracle survival >= 95% |
| F2 | mute survival <= 5% |
| F3 | random-word survival >=20 points below oracle |
| F4 | best body-blind rhythm survival >=20 points below oracle |
| F5 | oracle mean viability >=0.05 above every blind policy |

If any gate fails, stop.  Do not train and do not tune metabolism, portions,
help period, or life length in this experiment.

## 4. Locked learning comparison

If F1-F5 pass, run exactly one seed-1 treatment trajectory:

- 200,000 primitive steps;
- hidden size 256, token embedding 32;
- segment length 64;
- two report slots over the full 52-token vocabulary;
- learning rate 3e-4;
- motor and report entropy weights 0.02;
- bodily-prediction and caregiver-token-prediction weights zero;
- no behavior cloning, curriculum, replay, privileged critic, or direct
  report target.

The legacy checkpoint is the already completed control.  The treatment is not
a seed expansion, and no interim curve may change the locked endpoint.

## 5. Gates and audits

The original T1-T3 gates remain unchanged:

| Gate | Requirement |
|---|---|
| T1 | held-out report fidelity >=60% |
| T2 | held-out survival >=80% |
| T3 | fidelity at ticks >=200 is within 10 points of overall fidelity |

If T1-T3 pass, interpret the already locked rent, internal-channel lesion,
counterfactual perceptible-portion, held-out-birth, held-out-portion, masked-
observation decoder, and persistence audits.  The post-hoc oracle-listener
uptake diagnostic is reported separately and is never a capability gate.

## 6. Interpretation and stop rule

- If feasibility fails, unified uptake makes the ecology insufficiently
  dependent on self-knowledge and the treatment is rejected before learning.
- If feasibility passes but T1 fails, the distinct motor convention was not
  the main language-discovery barrier.  Stop this line; do not tune entropy,
  width, vocabulary, or reward weights.
- If T1 passes but T2 fails, communication was learned but the remaining motor
  or viability policy is insufficient; report that separation without calling
  the system a successful self-reporting organism.
- Only a full T1-T3 and causal-audit pass supports the miniature claim of a
  persistent, causally grounded communicated self-model.

No larger compute, external data, or generated data is justified by this
experiment.
