# Counterfactual report-credit preregistration

Date: 2026-07-26
Status: locked before implementation or training

## 0. Altitude and diagnosis

This experiment directly targets **self-report fidelity**.  It changes the
communication credit estimator, not the ecology, vocabulary, self-state
inputs, training data, or report target.

The unified-uptake treatment passed all five feasibility gates and removed the
motor coordination barrier: under oracle words its frozen motor policy takes
99.94% of food, 99.69% of water, and 99.83% of energy grants and survives 95.5%
of held-out lives.  Its learned reports nevertheless reach only 34.35%
fidelity and 5% survival.

A post-hoc decoder split by whole held-out life isolates the failure.  A fixed
linear readout of the shared recurrent state recovers the hidden lowest need
at 64.04% balanced accuracy, while the current masked observation reaches
32.85% (chance 33.33%).  The organism therefore contains a usable persistent
self-state representation; the consequence-only report actor fails to read it
out under the current high-variance joint-action gradient.

## 1. Claim under test

> Counterfactual token-level credit learned from lived returns is sufficient
> to turn the organism's already present hidden self-state into truthful,
> causally useful generated reports without bodily or lexical supervision.

The method is adapted from counterfactual multi-agent policy gradients: the
two report slots are treated as components of a joint social action.  Each
component receives an advantage relative to its alternative tokens while the
other sampled token is held fixed.

## 2. Treatment

Add a default-off report critic used only during learning.  For each causally
heard utterance and each slot, it receives:

- the same recurrent state that feeds the mouth;
- the sampled token in the other slot; and
- the identity of the slot being evaluated.

It outputs a return estimate for every token in the unchanged 52-token
vocabulary.  The chosen-token estimate is fitted to the ordinary lived return.
The report actor uses the stopped-gradient counterfactual advantage

`Q(chosen | state, other token) - sum_a pi(a|state) Q(a | state, other token)`.

No critic input or target may include true needs, lowest-need identity, correct
word, grant identity, listener parse, simulator event metadata, a forked
outcome, or future observation.  It sees only the agent's state, its own joint
utterance, and the same scalar return already used by actor-critic learning.
The critic is absent at execution.

The critic MSE uses the existing value-loss coefficient (0.5); no new numeric
weight is introduced.  The legacy actor gradient remains the default when the
flag is false, and a test must show exact default construction plus zero
dependence on true-body tensors.

Research basis:

- Foerster et al., *Counterfactual Multi-Agent Policy Gradients* (AAAI 2018),
  for an action-conditioned centralized critic and per-component
  counterfactual baseline;
- Mesnard et al., *Counterfactual Credit Assignment in Model-Free
  Reinforcement Learning* (ICML 2021), for lower-variance causal credit without
  changing the task reward.

These papers motivate the estimator; neither is evidence it works here.

## 3. Locked run

The probe49 feasibility seal is reused because the ecology is byte-identical.
Run exactly one seed-1 trajectory with:

- unified uptake enabled;
- 200,000 primitive steps;
- hidden size 256, token embedding 32, segment length 64;
- two full-vocabulary report slots;
- learning rate 3e-4;
- motor and report entropy weights 0.02;
- bodily and caregiver-token prediction weights zero;
- no behavior cloning, curriculum, replay, body label, word target, privileged
  critic input, or generated data.

The sole manipulation relative to probe49 is the counterfactual report critic.
No interim result may alter the endpoint or coefficients.

## 4. Gates and stop rule

The locked T1-T3 gates and full read-only causal battery are unchanged:

- T1 report fidelity >=60%;
- T2 survival >=80%;
- T3 tick-200+ fidelity within 10 points of overall fidelity;
- grounded survival at least 15 points above scrambled, mute, and every fixed
  word;
- internal-state lesions collapse fidelity toward chance;
- perceptible portion forks change reports in the direction of the changed
  bodily ordering;
- held-out birth and portion schedules preserve the capability.

The hidden-state decoder and oracle-listener uptake rows remain diagnostics,
never capability gates.

If T1 fails, the language-output optimization line is closed after this probe.
Do not try another entropy, critic width, target horizon, replay, vocabulary,
or loss-weight variant.  Propagate upward to grounded lexical development or
a world-model/planning architecture.  If T1 passes but causal audits fail, do
not call the output self-report.

No larger compute or data request is justified before a local pass.
