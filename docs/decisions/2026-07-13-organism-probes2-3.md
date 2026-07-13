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

## Probe 3 result — NEGATIVE, diagnostically useful

6600+ lives over 500k steps. Infant lives (0.25x metabolism, ~214-step
thirst clock) lasted ~166 steps — above the ~92-step passive energy clock,
so the agent genuinely learned to rest — but life length tracked the
annealing metabolism clock straight back down to ~58 at adult level. Eval
survival 0%. One late life reached truncation (recent survival 0.01), so
the ceiling is reachable but not learned.

Diagnosis: longer lives alone do not produce consume-interaction discovery.
The approach -> face -> consume chain on the right object is too sparse for
undirected exploration regardless of clock length. This replicates
probe-era failure F1 (from-scratch RL never discovered ask-then-act) in the
new substrate.

## Decision

Three negative probes: stop varying exploration pressure. Next mitigation
in the plan's sanctioned order is the **BC-bootstrap childhood**: clone a
small number of oracle lives (approach/consume/rest competence only, no
language behavior to imitate since the oracle ignores tokens), then
continue online lifelong RL, and always report with/without the bootstrap
(`--bc-warmstart` ablation). After that works, the G1 grounded vs silent vs
shuffled comparison becomes meaningful: kind knowledge (which berry is
food this world) is NOT in the oracle demonstrations' observable state — it
must come from caregiver labels, so the language effect stays clean.
An alternative if BC contaminates: caregiver-guided shaping via OFFER
situations (caregiver hands food to a starving infant), which is
developmentally natural and keeps all competence in-loop.
