# Caregiver-guided report-lexicon result

Date: 2026-07-26  
Decision: lexical gate passed; adult self-report gate failed

## Integrity and provenance

Before learning, the locked 300-life simulator audit completed 2,400 rounds
and passed every mechanics gate:

- exactly one guided label per round;
- label target matched the currently low need in 33.46% of rounds;
- food, water, and energy comprised 33.79%, 33.04%, and 33.17% of labels;
- every inspect and return used exactly four and six primitive ticks; and
- guided decisions had zero actor and entropy weight.

The childhood artifact is
`runs/organism/probe52_guided_report_lexicon/gate/`. It contains one exact
60,000-tick seed-1 run and the frozen 300-life comprehension audit. The adult
artifact is `runs/organism/probe52_guided_report_lexicon/adult/`. It continues
the same model parameters for exactly 200,000 adult ticks with fresh optimizer
moments and no guided labels, body target, token target, or other childhood
scaffold. The adult run completed 2,610 lives plus 37 ticks of its final open
life. Completed-life CSV summaries therefore cover 199,963 of the exact
200,000 trainer ticks.

## Childhood establishes a causal public lexicon

| Frozen endpoint | Locked gate | Result |
|---|---:|---:|
| Intact correctness | >= 80% | **100%** |
| Food correctness | >= 70% | **100%** |
| Water correctness | >= 70% | **100%** |
| Energy correctness | >= 70% | **100%** |
| Cyclic-word correctness | <= 45% | **0%** |
| Intact minus cyclic | >= 30 points | **100 points** |
| Paired action change | >= 60% | **100%** |

The caregiver selected external resources uniformly and independently of the
child's body, never named the child's need, never demonstrated consumption,
and supplied no report reward. Changing only the heard content word changed
the selected resource on every paired held-out trial. This is causal lexical
comprehension, not correlation with body state and not self-report.

## Adult continuation retains the lexicon but fails self-report

The original sealed 200-life report battery failed decisively:

| Adult endpoint | Result | Gate or control |
|---|---:|---:|
| Grounded survival | **3.0%** | >= 80% required |
| Grounded report fidelity | **31.46%** | >= 60% required |
| Grounded mean life | 105.19 ticks | 400-tick maximum |
| Scrambled-listener survival | **4.0%** | grounded should be >=15 points higher |
| Mute survival | 0% | language remains necessary |
| Held-out birth survival | 5.5% | transfer failure |
| Held-out portion survival | 12.0% | transfer failure |
| Body-fork report change | 0.69% | 725 divergent ticks |
| Body-fork following | **0.21%** | causal grounding failure |
| Oracle-listener survival | **95.0%** | motor/listener substrate remains viable |

The post-adult comprehension recheck still scored 100% intact correctness,
98.33% paired action change, and 1.67% cyclic correctness. Adult learning did
not erase the public lexicon. The failure is the organism's inability to map
its changing hidden body to the appropriate retained word.

## Read-only localization

A whole-life held-out ridge audit on the frozen adult checkpoint found:

| Decoder input | Raw accuracy | Balanced accuracy |
|---|---:|---:|
| Shared recurrent state | 58.13% | **44.36%** |
| Current masked observation | 46.64% | 33.64% |

Across 16,116 training and 5,474 whole-life held-out samples, the recurrent
state retains a clear above-observation body signal, but it is weaker than
probe 49's 64.04% balanced result and far below the declared learned-belief
gate. Guided lexical childhood did not yield a protected, sufficiently
accurate adult self-state.

A second audit fitted a disposable supervised ridge readout on 51,483 frozen
states from 140 separate oracle-speaking lives, then used its predicted need
word to control the real listener in 60 held-out lives. It never changed an
organism parameter. Survival rose to **35.0%**, mean life to 228.77 ticks, raw
report fidelity to 45.06%, and balanced fidelity to 49.53% over 12,250
held-out ticks. This is an upper bound, not an agent capability. It shows that
a better state-to-word mapping helps materially, but the available adult state
is itself too inaccurate to pass the control gate.

## Decision and claim boundary

Probe 52 establishes a new, narrow positive result: a single parameter-
persistent organism can acquire a causally grounded public report lexicon from
need-independent embodied pedagogy and retain it after the scaffold is
removed. It does not establish truthful self-report, reflection,
consciousness, sentience, or a real self-description.

The failure localizes to two connected missing mechanisms:

1. a protected learned belief over the organism's own latent bodily state,
   updated from its lived history; and
2. an utterance policy that evaluates how candidate words change listener
   action and future homeostasis.

More report-head optimization, exposure, width, sparse-return credit, seeds,
or scale is not licensed. The next experiment must implement those mechanisms
explicitly while keeping true adult body state and correct report labels out
of every learner input and loss.
