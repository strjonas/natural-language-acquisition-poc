# Homeostatic terminal-utility correction preregistration

Date: 2026-07-25
Status: locked before implementation and before any corrected planner output is
read

## What this changes and why it is not tuning

Every planner in the repository scores an imagined action by the predicted
*minimum* of the four bodily needs afterwards. The read-only diagnosis in
`2026-07-25-persistent-self-query-planner-result.md` shows that after the
ten-tick label detour this quantity is fixed by energy, which no consumption
choice affects, and that it is actively biased against knowledge because an
uncertain object's smeared predicted delta raises a minimum more than a correct
concentrated one does.

The correction replaces that utility with the predicted level of the need the
organism currently observes as its lowest. This is the classical drive-reduction
rule of homeostatic control, restricted to the deficit that is currently most
urgent. The deployed rule is the degenerate `n -> infinity` member of the same
family.

This is a defect in the question the planner asks its own self-model, not a
hyperparameter. No capacity, optimizer, reward, entropy, training budget,
language traffic, inspection subsidy, or hidden target changes. The gates below
are copied verbatim from
`2026-07-25-persistent-self-query-planner-preregistration.md` and are not
weakened.

## Mechanism

Define the terminal bodily score of an imagined action as

```text
score = predicted_need[argmin(current_observed_needs)] after the action
```

where `current_observed_needs` are the four interoceptive values already present
in the organism's own observation vector at the state being scored, and the
predicted need is the existing learned consequence head clipped to [0, 1].

Binding constraints:

1. The index is taken from the organism's own observation. Hidden kind, the
   simulator's demanded resource, correctness, reward, and audit labels never
   enter it.
2. The index is recomputed at each state that is scored, including inside a
   hypothetical branch, from that state's own predicted interoception.
3. The rule replaces the minimum in all four scoring sites: one-step action
   consequences, the two-step rollout's second step, the terminal-consume
   shortcut, and the terminal consume scorer used by observation branching and
   by the persistent self-query backup.
4. The previous minimum rule remains selectable and stays the default for the
   existing open-island planner and for every already-sealed artifact, so past
   results remain reproducible. The corrected rule is enabled explicitly.
5. No other planner semantics change. Reward weight in the delayed
   observation-branching planner stays zero.

## Locked read-only test, part one: feasibility

Rerun the mechanism of `2026-07-25-persistent-self-query-planner-preregistration.md`
unchanged except for the terminal utility, on the same checkpoint

`runs/organism/probe24_persistent_childhood_corrected/organism_grounded_choice30000x3h40r6n0.55q8_options_inspect_bind16_replay256x1_seed1.npz`

SHA-256 `02bb27563d6af08e1255e0d8d48e4a46e048d4ae48be54b2fd0620827cac728e`,

on the same 300 fixed held-out contexts from seed `1_700_000`, with reuse count
7, comparing intact hypothetical writes, acute write suppression, and collapsed
all-padding label branches.

Feasibility passes only if all hold:

1. intact mean best-inspect advantage over immediate consumption is positive;
2. at least 75% of contexts have positive intact inspect advantage;
3. write suppression reduces mean inspect advantage by at least 0.05;
4. collapsed labels reduce mean inspect advantage by at least 0.05;
5. at least 60% of inspect options support label-contingent terminal choices;
6. a matching-resource branch selects the inspected target at least 60%; and
7. a danger branch avoids the inspected target at least 90%.

## Locked read-only test, part two: behavior

Only if part one passes, activate the corrected planner read-only on the same
checkpoint for 300 fixed eight-round held-out lives at the repository's existing
locked planning scale 6.0. Report grounded, acute silent, acute independently
shuffled, acute write-suppressed, and planner-removed behavior on identical
seeds.

Behavioral promotion requires:

1. at least 60% aggregate grounded correctness;
2. at least 65% correctness in each of rounds 3-8;
3. grounded exceeds planner-removed correctness by at least 15 points;
4. acute write suppression reduces correctness by at least 15 points; and
5. both acute silence and acute shuffle reduce correctness by at least 15
   points.

## If both parts pass

One fresh matched write-enabled / write-disabled training pair may be run at the
identical 30,000-tick seed-1 configuration of
`2026-07-25-persistent-childhood-learning-preregistration.md`, with the
corrected planner activated after 15,000 ticks, and evaluated against that
preregistration's nine promotion gates unchanged.

Seed 2, freshly trained silent and shuffled controls, open-island transfer,
generated speech, and any compute or data request remain blocked until that
fresh pair passes.

## Stop rule

If part one fails, the negative result propagates up to the substrate again and
no further planner variant is implemented on this task. Do not search over
utility rules: the corrected rule is fixed here, in advance, as the currently
lowest observed need, and the diagnostic table in the result document is not to
be used to select a different member of the family afterwards.

## Methodological basis

- Hull (1943), *Principles of Behavior*: drive reduction on the currently
  dominant deficit.
- Keramati and Gutkin (2011/2014), homeostatic reinforcement learning: utility
  as reduction of a deficit norm, of which the deployed minimum is the limiting
  case.
- Silver and Veness (2010); Ross et al. (2011); Guez et al. (2012): sensing is
  valued only through the downstream utility it changes, which is precisely
  what a degenerate utility destroys.
