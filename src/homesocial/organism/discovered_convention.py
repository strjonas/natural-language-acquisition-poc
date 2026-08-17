"""Probe66: finding out whether a word means the same thing whatever you ask for.

Probe64 moved property 4 -- *productive under novel demand* -- from **no** to
**partial**, and named the one reason it is not **yes**:

    That the listener-model factorization is discovered. It is given.

Its `factored` listener holds one belief per size word pooled over needs -- *this
word means large, whatever I am asking for* -- and that pooling is what lets it
utter a combination it has never uttered. But the designer chose to pool. The
world happened to reward pooling, and an organism that always pools would look
identical whether or not it did.

`tangled_size_words` moves the ground truth, which probe61's preregistration made
binding for any discovery claim: the caregiver draws its word-to-size orientation
**per need**, so "more" may be large for food and small for water. The ceiling
survey then measured the thing this probe turns on, and it came out the opposite
way from the reasoning it was run to check. A pooling model does *not* shrug in a
tangled world:

    factoring world     factored 0.988 / 1.000 / 1.000
    tangled world       factored 0.350 / 0.375 / 0.500

It commits, and commits wrongly -- at or below the 0.500 a model with no evidence
scores. So there is something to discover, and being handed the answer is a bet
the world can settle against you.

The mechanism has one idea and no free parameters. Before generalizing what a
word does to a need it has never used that word for, the organism asks whether
the needs it *has* used it for **agree with each other**:

    H_factors    every need shares one meaning for this word
    H_tangled    every need has its own

Both are scored by their exact Beta-Binomial marginal likelihood under a uniform
prior, and the posterior odds are the organism's answer. The held-out need
contributes nothing to either -- it has no evidence, so its factor is one on both
sides -- which is precisely what makes this a judgement about *transfer* rather
than about the cell itself.

And the organism is allowed to **decline**: to say the need word with no size
word, which is exactly what every organism before probe64 did, and which probe64
measured as survivable (`no_size` survives 0.630 against `always_large`'s 0.005).
Declining is a real move in this ecology and not a scoring dodge, which is why
this probe scores two numbers and never one: how often it speaks, and how often
it is right when it speaks. A model that declines everywhere passes neither.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import lgamma
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import (
    NEED_TO_REPORT_WORD,
    PORTION_SIZES,
    REPORT_NEEDS,
    SIZE_TO_WORD,
    SIZE_WORDS,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.portion_request import (
    GrantWatcher,
    ListenerModel,
    MoveFraction,
    SelfModelTier,
)
from homesocial.organism.report_audit import make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]

MODELS = ("factored", "tabular", "discovered")

# Disjoint from every band in use. Probe65's treatment reaches 1,028,000,120 and
# its construction band 1,011,000,040 was spent on the ceiling survey.
SURVEY_SEED_BASE = 1_040_000_000
SEED_BASE = 1_060_000_000
SEED_STRIDE = 2_000_000
SEEDS = 5

INTEROCEPTION = 0.03
OPERATING_STORE = 30.0
WORLD = {"metabolic_spread": 0.60, "uptake_spread": 0.00}
# The two worlds whose true structure differs. A learner that always answers
# "it factors" scores well in the first and must be told apart from one that
# finds out; that is the whole point of running both.
WORLDS = {"factoring": False, "tangled": True}
HELD_OUT_NEED = "energy"

# The organism speaks when its own posterior says the word transfers. Fixed a
# priori at even odds -- the point at which the evidence favours factoring at all
# -- and never swept. It is a decision rule, not a tuned threshold.
SPEAK_THRESHOLD = 0.5


def _log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def _log_evidence(large: float, total: float) -> float:
    """Exact log marginal likelihood of one cell's counts under a uniform prior.

    Beta-Binomial with Beta(1,1), binomial coefficient dropped because it is
    common to both hypotheses and cancels in the odds.
    """

    return _log_beta(1.0 + large, 1.0 + total - large) - _log_beta(1.0, 1.0)


class DiscoveredConvention:
    """A listener model that works out whether the word's meaning transfers.

    It keeps exactly what a tabular model keeps -- how often each (need, word)
    pair produced a large portion -- and adds one question asked of those counts:
    do the needs agree? Nothing tells it the answer, nothing sets a pooling
    strength by hand, and the held-out cell contributes to neither hypothesis.
    """

    def __init__(self, *, speak_threshold: float = SPEAK_THRESHOLD) -> None:
        self._large: dict[tuple[str, str], float] = {}
        self._count: dict[tuple[str, str], float] = {}
        self._threshold = float(speak_threshold)

    # -- evidence ------------------------------------------------------------

    def observe(self, need: str, word: str, granted_large: bool) -> None:
        key = (need, word)
        self._large[key] = self._large.get(key, 0.0) + float(granted_large)
        self._count[key] = self._count.get(key, 0.0) + 1.0

    def evidence(self, need: str, word: str) -> float:
        return self._count.get((need, word), 0.0)

    # -- the structure question ----------------------------------------------

    def probability_factors(self, word: str) -> float:
        """Posterior that this word means one thing across every need.

        Two exact marginal likelihoods under a uniform prior and equal prior
        odds. A need with no evidence contributes the same factor to both, so it
        cannot move the answer -- which is what makes this a judgement about
        whether meaning *transfers* rather than about any one cell.
        """

        cells = [
            (self._large.get((need, word), 0.0), self._count.get((need, word), 0.0))
            for need in REPORT_NEEDS
        ]
        informative = [(k, n) for k, n in cells if n > 0.0]
        if len(informative) < 2:
            # One need cannot agree or disagree with anything. With nothing to
            # compare, the prior stands and the organism has discovered nothing.
            return 0.5
        pooled_large = sum(k for k, _ in informative)
        pooled_total = sum(n for _, n in informative)
        log_factored = _log_evidence(pooled_large, pooled_total)
        log_tangled = sum(_log_evidence(k, n) for k, n in informative)
        # Numerically stable logistic of the log odds. With hundreds of grants a
        # life the log Bayes factor routinely exceeds what exp can carry.
        odds = log_factored - log_tangled
        if odds >= 0.0:
            return float(1.0 / (1.0 + np.exp(-odds)))
        scaled = np.exp(odds)
        return float(scaled / (1.0 + scaled))

    def transfers(self, word: str) -> bool:
        return self.probability_factors(word) >= self._threshold

    # -- speaking -------------------------------------------------------------

    def pooled_probability_large(self, word: str) -> float:
        large = sum(self._large.get((need, word), 0.0) for need in REPORT_NEEDS)
        total = sum(self._count.get((need, word), 0.0) for need in REPORT_NEEDS)
        if total <= 0.0:
            return 0.5
        return large / total

    def word_for(self, need: str, size: str) -> str | None:
        """The word that gets ``size`` of ``need``, or ``None`` to say nothing.

        If the cell has been used before, its own evidence settles it and no
        transfer is needed. Otherwise the organism generalizes only if its own
        posterior says the word's meaning transfers, and declines if it does not
        -- which leaves the caregiver to draw a portion, exactly as it did for
        every organism before probe64.
        """

        words = tuple(SIZE_WORDS)
        if any(self.evidence(need, word) > 0.0 for word in words):
            scores = {
                word: (
                    self._large[(need, word)] / self._count[(need, word)]
                    if self.evidence(need, word) > 0.0
                    else 0.5
                )
                for word in words
            }
        elif all(self.transfers(word) for word in words):
            scores = {word: self.pooled_probability_large(word) for word in words}
        else:
            return None
        if size == "large":
            return max(words, key=lambda word: scores[word])
        return min(words, key=lambda word: scores[word])


# -- the holdout ---------------------------------------------------------------


def _world_kwargs(tangled: bool, *, store: float, interoception: float) -> dict:
    return dict(
        WORLD,
        interoception_probability=interoception,
        portion_requests=True,
        caregiver_store=store,
        tangled_size_words=tangled,
    )


def composition_holdout(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    tangled: bool,
    lives: int,
    seed_base: int,
    store: float = OPERATING_STORE,
    interoception: float = INTEROCEPTION,
    held_out_need: str = HELD_OUT_NEED,
) -> dict[str, object]:
    """Probe64's holdout, run in both worlds, with a third model watching.

    Through the whole life the speaker never says a size word while asking about
    ``held_out_need``, so the caregiver never once demonstrates what a size word
    does to an energy grant. All three models watch the identical transcript --
    the speaker is driven by ``factored``, exactly as in probe64, so the
    discovered model can never be starved or fed by a policy difference of its
    own. At the end each is asked for **both** sizes of the held-out need, which
    is what stops a two-word vocabulary being solved by elimination.

    Two numbers per model and never one. ``spoke`` is how often it was willing to
    answer at all; ``accuracy_when_spoken`` is how often it was right when it
    was. A model that declines everywhere scores perfectly on the second and
    zero on the first, and passes nothing.
    """

    kwargs = _world_kwargs(tangled, store=store, interoception=interoception)
    report = replace(organism.report, **kwargs)

    asked = 0
    spoke = {name: 0 for name in MODELS}
    correct = {name: 0 for name in MODELS}
    factors_posterior = 0.0
    scored_lives = 0
    sizeless = 0

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tier = SelfModelTier("recursive", organism, report)
        tier.reset(packet, world)
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        models: dict[str, object] = {
            "factored": ListenerModel("factored"),
            "tabular": ListenerModel("tabular"),
            "discovered": DiscoveredConvention(),
        }
        watcher = GrantWatcher(report.help_period)
        hidden = None
        rng = Random(seed + 59_000_003)

        while True:
            need = REPORT_NEEDS[int(np.argmin(tier.point()))]
            fraction = moves.fraction
            size = tier.size_for(need, fraction)
            word = (
                None
                if need == held_out_need
                else models["factored"].word_for(need, size)  # type: ignore[union-attr]
            )
            sizeless += int(word is None)
            tokens = (
                TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]],
                PAD_ID if word is None else TOKEN_TO_ID[word],
            )
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear(tokens)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            granted = watcher.poll(packet)
            if granted is not None and word is not None:
                for observer in models.values():
                    observer.observe(granted[0], word, granted[1])  # type: ignore[union-attr]
            if terminated or truncated:
                break
            moves.observe(before, action, packet)
            tier.update(before, action, packet, world, info.get("interoception"))

        # What the caregiver actually does with each word for the held-out need,
        # this life. In the factoring world it is the species convention; in the
        # tangled world it is this life's own draw.
        convention = world.size_convention[held_out_need]
        truth = {size: word for word, size in convention.items()}
        for name, observer in models.items():
            for size in PORTION_SIZES:
                said = observer.word_for(held_out_need, size)  # type: ignore[union-attr]
                spoke[name] += int(said is not None)
                correct[name] += int(said is not None and said == truth[size])
        factors_posterior += float(
            np.mean(
                [
                    models["discovered"].probability_factors(word)  # type: ignore[union-attr]
                    for word in SIZE_WORDS
                ]
            )
        )
        asked += len(PORTION_SIZES)
        scored_lives += 1

    return {
        "world": "tangled" if tangled else "factoring",
        "tangled_size_words": tangled,
        "lives": scored_lives,
        "held_out_need": held_out_need,
        "sizeless_requests_per_life": sizeless / max(1, scored_lives),
        "questions": asked,
        # How often each model was willing to answer at all.
        "spoke": {name: spoke[name] / max(1, asked) for name in MODELS},
        # How often it was right when it did. Read only beside ``spoke``.
        "accuracy_when_spoken": {
            name: correct[name] / max(1, spoke[name]) for name in MODELS
        },
        # The headline: how often it committed to an answer and was wrong. This
        # is the quantity a caregiver pays for, and the one a model that declines
        # honestly is allowed to lower.
        "confidently_wrong": {
            name: (spoke[name] - correct[name]) / max(1, asked) for name in MODELS
        },
        # Probe64's single number, kept so the two documents can be compared.
        "first_use_accuracy": {name: correct[name] / max(1, asked) for name in MODELS},
        # What the discovered model concluded about the world it was in.
        "probability_factors": factors_posterior / max(1, scored_lives),
    }


# -- the locked gates ----------------------------------------------------------


def _verdict(flags: list[bool], needed: int = 4) -> dict[str, object]:
    return {
        "per_seed": flags,
        "passing": sum(flags),
        "required": needed,
        "pass": sum(flags) >= needed,
    }


def score_gates(rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Exactly the gates in the preregistration, scored per seed."""

    def column(world: str, key: str, name: str) -> list[float]:
        return [float(row[key][name]) for row in rows[world]]  # type: ignore[index]

    def posterior(world: str) -> list[float]:
        return [float(row["probability_factors"]) for row in rows[world]]

    # Every threshold below is written in
    # `docs/decisions/2026-08-16-discovered-convention-preregistration.md` and was
    # fixed against a survey on a disjoint seed band before any treatment seed
    # existed. Nothing here reads a threshold from the data.
    g1 = [
        speaks >= 0.95 and found >= given - 0.02
        for speaks, found, given in zip(
            column("factoring", "spoke", "discovered"),
            column("factoring", "accuracy_when_spoken", "discovered"),
            column("factoring", "accuracy_when_spoken", "factored"),
        )
    ]
    g2 = [speaks <= 0.75 for speaks in column("tangled", "spoke", "discovered")]
    g3 = [
        found - given <= -0.10
        for found, given in zip(
            column("tangled", "confidently_wrong", "discovered"),
            column("tangled", "confidently_wrong", "factored"),
        )
    ]
    g4 = [
        factoring - tangled >= 0.12
        for factoring, tangled in zip(posterior("factoring"), posterior("tangled"))
    ]
    g5 = [
        given <= 0.60
        for given in column("tangled", "first_use_accuracy", "factored")
    ]
    g6 = [
        abs(factoring - 0.5) <= 1e-9 and abs(tangled - 0.5) <= 1e-9
        for factoring, tangled in zip(
            column("factoring", "first_use_accuracy", "tabular"),
            column("tangled", "first_use_accuracy", "tabular"),
        )
    ]

    gates = {
        "G1_generalizes_where_it_is_licensed": _verdict(g1),
        "G2_declines_where_it_is_not": _verdict(g2),
        "G3_less_often_confidently_wrong": _verdict(g3),
        "G4_the_recovered_structure_tracks_the_world": _verdict(g4),
        "G5_the_given_factorization_fails": _verdict(g5),
        "G6_the_tabular_control_is_chance": _verdict(g6),
    }
    gates["all_pass"] = all(gate["pass"] for gate in gates.values())  # type: ignore[index]
    return gates


def run(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    seeds: int,
    seed_base: int,
) -> dict[str, object]:
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in WORLDS}
    for index in range(seeds):
        base = seed_base + SEED_STRIDE * index
        for name, tangled in WORLDS.items():
            row = composition_holdout(
                model, organism, tangled=tangled, lives=lives, seed_base=base
            )
            rows[name].append(row)
            print(
                f"seed {index} {name:>9}: "
                + " ".join(
                    f"{key[:4]} spoke {row['spoke'][key]:.3f}"
                    f"/right {row['accuracy_when_spoken'][key]:.3f}"
                    for key in MODELS
                )
                + f" | P(factors) {row['probability_factors']:.3f}",
                flush=True,
            )
    return {"rows": rows, "gates": score_gates(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--lives", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--seed-base", type=int, default=SURVEY_SEED_BASE)
    parser.add_argument(
        "--treatment",
        action="store_true",
        help="Run the preregistered five-seed treatment instead of the survey.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(args.checkpoint)
    seed_base = SEED_BASE if args.treatment else args.seed_base
    result = run(
        model, organism, lives=args.lives, seeds=args.seeds, seed_base=seed_base
    )
    default = "treatment.json" if args.treatment else "ceiling_survey.json"
    out = Path(args.out or f"runs/organism/probe66_discovered_convention/{default}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))

    print("\n== locked gates ==")
    for name, verdict in result["gates"].items():
        if name == "all_pass":
            continue
        mark = "PASS" if verdict["pass"] else "FAIL"
        print(
            f"{mark}  {name}: {verdict['passing']}/"
            f"{len(verdict['per_seed'])} {verdict['per_seed']}"
        )
    print(f"all_pass = {result['gates']['all_pass']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
