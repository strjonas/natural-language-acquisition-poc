# Grounded report-lexicon development preregistration

Date: 2026-07-26  
Status: locked before implementation, training, or evaluation

Pre-data mechanics amendment: the initial draft proposed labeling all three
objects in one held-out life. Three four-tick inspections plus three six-tick
returns consume 30 metabolic ticks; in the energy condition that protocol can
itself terminate the organism or change which need is lowest. Before any code
or result was produced, the gate below was corrected to one target inspection
per life. This is the only amendment.

## Question selected by the preceding failures

The unified-uptake treatment established two facts at once: an oracle speaker
lets the learned motor organism survive, and a frozen decoder reads the hidden
lowest need from recurrent state substantially better than from the current
masked observation. The report head nevertheless stays near an unconditioned
three-word mixture. A per-token counterfactual critic made both representation
and behavior worse. The remaining local question is therefore not whether to
add more report credit, but whether the same organism can first acquire the
public meanings of the three report words and then use that grounded lexical
geometry for production.

This experiment restores lexical comprehension (C2) before retrying generated
self-report (C3). It is not a consciousness test and childhood performance is
not self-report.

## Mechanism locked before code

Add a default-off report-lexicon childhood to the existing delayed semantic
choice ecology.

- Each childhood life randomly remaps six visible surfaces onto two food, two
  water, and two energy-restoring consumables.
- A trial presents one object of each functional kind. One of food, water, or
  energy is publicly low in the ordinary interoceptive observation.
- A voluntary kind-blind inspection elicits exactly one public object label:
  `this hungry`, `this thirsty`, or `this tired`, according to the inspected
  object's bodily consequence. The label says what the external object does;
  it never names the organism's hidden adult state and never states which
  action or report is correct.
- The existing fixed six-tick return removes deixis before choice. The same
  surface-to-kind mapping and object-local lexical bank persist for eight
  alternating/cycling demand rounds.
- Consuming the matching object changes the corresponding bodily variable.
  There is no label reward, inspection reward, report reward, correctness
  target, simulator counterfactual, or hidden-kind input.
- During this childhood only, the already existing next-body and caregiver
  token prediction objectives may learn from fully observable public
  experience. Both weights are exactly zero after the switch to the masked
  report island.
- The mouth uses a default-off tied lexical decoder: each slot projects the
  organism state into the same embedding space used to encode heard tokens,
  then scores the full closed vocabulary by dot product with that shared token
  table. There is no need-word-only output mask.

The developmental stage lasts exactly 60,000 primitive ticks with hidden size
256, token embedding 32, binding width 16, three visible slots, eight rounds,
low need 0.55, horizon 40, six-tick return, consume and inspect object options,
segment length 64, seed 1, the default optimizer, `next_needs_weight = 1.0`
and `token_prediction_weight = 0.5` in childhood. The report head receives no
actor loss in childhood because no listener acts on its utterances.

## Frozen comprehension gate

Before any adult training update, freeze the organism and evaluate 300 held-out
seeded lives. In each life a protocol composed entirely of learner-visible
inspect, label, return, and observation packets labels the object whose external
kind matches the currently low need. At the common returned pose, that labeled
object remains among two unlabeled alternative resources and the organism's
learned terminal consequence scores select one of the three. Selecting the
target is not trivial: the surface-to-kind mapping is newly randomized and the
two alternatives remain equally visible.

Run two paired conditions on identical worlds and recurrent histories:

1. intact caregiver need words;
2. a fixed cyclic intervention `hungry -> thirsty -> tired -> hungry` applied
   only to the three heard content tokens.

The gate passes only if all conditions hold:

1. intact correct resource selection is at least 80%;
2. intact correctness is at least 70% separately for food, water, and energy;
3. cyclic-word correctness is at most 45%;
4. intact minus cyclic correctness is at least 30 percentage points; and
5. the selected resource changes in at least 60% of paired lives.

The intervention is essential: a body-only or surface-only selector can score
well only if it ignores the words, in which case paired actions will not move.
The evaluator may read simulator kind only to score the frozen action after it
is selected; no such metadata enters the organism.

If any comprehension gate fails, stop before adult training. Diagnose
exposure, token-to-binding geometry, and consequence-to-action geometry; do not
change thresholds or run a report-head sweep.

## Adult continuation locked conditionally

Only if the frozen comprehension gate passes, continue the *same model* for
exactly 200,000 adult primitive ticks. Resetting optimizer moments at the
developmental boundary is permitted and must be reported; model parameters are
not reset. Adult conditions are the sealed unified-uptake report ecology from
probe 49: full 52-token vocabulary, two report slots, hidden body after the
birth reading, help every six ticks, shocks unchanged, unified `CONSUME`
uptake, report entropy 0.02, ordinary return credit (not the failed
counterfactual critic), and no true-body or caregiver-token prediction loss.
Object-option action heads remain present so the architecture is continuous.

Run the full existing 200-life report battery, oracle-listener uptake audit,
and frozen recurrent-state versus observation decoder. Promotion still uses
the original report gates without relaxation: at least 60% grounded fidelity,
at least 80% survival, late-life persistence, grounded advantage over
scrambled/mute, held-out births and portions, representation lesions, and the
corrected lived-consequence counterfactual. Lexical childhood does not count
toward any adult endpoint.

## Scale and stop rule

One seed and one fixed local two-stage run. Do not sweep childhood length,
losses, tying scale, binding width, report entropy, adult budget, or seeds. A
failed comprehension gate stops before adulthood. A passed comprehension gate
followed by failed report gates closes the lexical-bridge hypothesis and
selects a social-consequence planner as the next mechanism-level question.
Neither outcome authorizes larger compute or generated data.
