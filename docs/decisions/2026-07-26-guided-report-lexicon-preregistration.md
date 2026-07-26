# Caregiver-guided report-lexicon preregistration

Date: 2026-07-26  
Status: locked after probe 51 diagnosis and before implementation or data

## Failure-selected mechanism

Probe 51 separated tokens but failed to ground their bodily consequences.
Voluntary inspection fell from 1.287 labels per life in the first 20k ticks to
0.074 in the last 20k, while choices stayed at chance. The smallest causal
change is developmental pedagogy: give the organism reliable joint-attention
episodes without telling it what it currently needs or which later action is
correct.

## Single change

At the beginning of every childhood choice round, the caregiver uniformly
selects one of the three external objects, independently of the low bodily
need. A fixed, learner-visible inspect macro brings the organism to that object;
the ordinary caregiver emits its true external label (`this hungry`, `this
thirsty`, or `this tired`); and the existing six-tick padding-only return brings
the organism back to the canonical pose.

The guided inspect decision has actor and entropy weight zero. Its observations,
word, bodily costs, and world-model targets remain ordinary online experience.
The caregiver does not select the needed resource, does not demonstrate a
consume action, does not reward the label, does not score a report, and never
observes or names the future hidden adult body. After the return, all choices
are the organism's own. Guidance disappears completely in adulthood.

Everything else is identical to the exact-budget probe 51 condition: seed 1,
60,000 primitive ticks, hidden 256, embedding 32, binding 16, three randomly
remapped food/water/energy objects, eight rounds, low need 0.55, horizon 40,
four-tick options, six-tick return, segment 64, tied full-vocabulary mouth,
childhood body prediction 1.0, token prediction 0.5, and no report credit in
childhood.

## Mechanics gates before learning

Across 300 fixed simulator lives, require:

1. exactly one guided label in every completed round;
2. guided target kind matches the currently low need between 28% and 38%
   (consistent with independence and inconsistent with a need hint);
3. each of hungry, thirsty, and tired comprises 28%-38% of guided labels;
4. every guided inspect and return respects the four- and six-tick protocol;
5. zero guided decisions receive actor credit; and
6. exact-budget regression and the full repository suite pass.

No learned endpoint may be read before these mechanics pass.

## Frozen comprehension and adult gates

Use the exact same 300-life paired comprehension evaluator and unchanged five
thresholds from probe 51: intact >=80%, each need >=70%, cyclic <=45%, intact
minus cyclic >=30 points, paired action changes >=60%.

If any gate fails, stop before adulthood. If all pass, continue the same model
with fresh optimizer moments for the unchanged 200,000-tick unified-uptake
adult condition and full original report battery. Childhood performance never
counts as self-report evidence.

## Stop and scale rule

One new mechanism, one seed, no sweep. Do not change exposure rate, target
selection, weights, duration, gate, vocabulary, report credit, or budget. If
guided experience still fails comprehension, close the object-local binding
path for report words and move to a predictive social-consequence architecture.
Neither outcome authorizes larger compute or data generation.
