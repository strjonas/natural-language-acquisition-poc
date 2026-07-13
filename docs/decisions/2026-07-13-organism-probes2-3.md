# Organism probes 2-3: entropy, then infancy curriculum

Date: 2026-07-13
Goal-level metric: survival learning (prerequisite of gate G1).

## Probe 2: entropy 0.06, 400k steps — NEGATIVE

7000 lives, life length flat at ~57-58 steps, eval survival 0%. Together
with probe 1 (entropy 0.02, 200k) this says the failure is not exploration
temperature: undirected action noise does not chain approach -> face ->
consume within the ~54-step thirst clock.

## Substrate response: infancy metabolism curriculum (probe 3)

Per the timebox rule, two matching negatives propagate UP: the substrate
assumes an infant can survive adult metabolic pressure while incompetent,
which no developing organism actually faces. Added
`metabolism_curriculum_start/steps` to `OrganismConfig`: food/water
metabolism scales from `start` (e.g. 0.25 — thirst clock ~215 steps) at
global step 0 to adult 1.0 at `steps`. Applied per life reset; no kind
leakage; no behavior cloning; the adult-world evaluation is unchanged.

Probe 3 setup: start 0.25, anneal over 300k, total 500k steps, entropy
0.03, seed 1. Result recorded below after the run.

## Probe 3 result

(pending)
