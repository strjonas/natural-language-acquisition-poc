"""Read-only audits for the organism's own self-report (gate G3).

Every function here drives a trained organism through held-out lives without
updating a parameter. The battery is the one preregistered in
`docs/decisions/2026-07-25-need-report-rent-preregistration.md`:

1. fidelity, against chance and against a decoder given only the senses;
2. rent, against scrambled, mute and fixed-word listeners;
3. interventions on the internal channel that feeds the mouth;
4. a counterfactual body: same history, different portion actually received;
5. generalization to birth states never trained on;
6. persistence: fidelity as a function of ticks since the birth reading.

The interventions in (3) touch only the state handed to the report head. The
organism keeps acting on its true state, so anything that survives the
intervention was not being read out of the self-model in the first place.
"""

from __future__ import annotations

from dataclasses import replace
from random import Random

import mlx.core as mx
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.env import Action
from homesocial.island.report import (
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportConfig,
    ReportWorld,
    heard_need,
)
from homesocial.island.world import IslandConfig
from homesocial.organism.model import OrganismModel
from homesocial.organism.train import (
    ACTIONS,
    OrganismConfig,
    available_action_mask,
    execute_agent_action,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
REPORT_WORD_IDS = tuple(TOKEN_TO_ID[NEED_TO_REPORT_WORD[n]] for n in REPORT_NEEDS)
# Ticks after birth before a report counts. The body is visible exactly once,
# at tick 0; nothing measured here can be an echo of that reading.
FIDELITY_WARMUP = 25
INTERVENTIONS = ("none", "zero", "shuffle", "freeze")


def make_report_world(config: OrganismConfig, *, seed: int, **overrides: object):
    report = replace(config.report, **overrides)  # type: ignore[arg-type]
    world = ReportWorld(
        config.island_config(semantic_choice_trial=False),
        report=report,
        seed=seed,
    )
    world.reset(seed)
    return world


class _ReportDriver:
    """Runs one organism through one life, optionally with a lesioned mouth."""

    def __init__(
        self,
        model: OrganismModel,
        *,
        intervention: str = "none",
        fixed_word: str | None = None,
        rng: Random,
    ) -> None:
        if intervention not in INTERVENTIONS:
            raise ValueError(f"Unknown intervention: {intervention}.")
        self.model = model
        self.intervention = intervention
        self.fixed_word = fixed_word
        self.rng = rng
        self.hidden: mx.array | None = None
        self._history: list[mx.array] = []
        self._frozen: mx.array | None = None

    def reset(self) -> None:
        self.hidden = None
        self._history = []
        self._frozen = None

    def _report_states(self, states: mx.array) -> mx.array:
        if self.intervention == "none":
            return states
        if self.intervention == "zero":
            return mx.zeros_like(states)
        if self.intervention == "freeze":
            if self._frozen is None:
                self._frozen = states
            return self._frozen
        # Shuffle: read the mouth off some other moment of this same life. The
        # senses are untouched; only the self-state is out of time.
        if not self._history:
            return states
        return self._history[self.rng.randrange(len(self._history))]

    def step(self, world: ReportWorld, packet) -> tuple[int, tuple[int, ...]]:
        vector = mx.array(packet.vector()[None, None, :])
        tokens = mx.array(np.asarray(packet.tokens, dtype=np.int32)[None, None, :])
        states, self.hidden = self.model.core_states(vector, tokens, self.hidden)
        logits = self.model.policy_logits(states, vector)
        mask = available_action_mask(
            packet,
            self.model.action_size,
            visible_slots=self.model.visible_slots or None,
        )
        logits = mx.where(mx.array(mask)[None, None, :], logits, -1e9)
        mx.eval(logits, states)
        probabilities = np.asarray(mx.softmax(logits[0, 0], axis=-1), dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        action_index = int(
            self.rng.choices(range(self.model.action_size), weights=probabilities)[0]
        )

        if self.fixed_word is not None:
            said = (TOKEN_TO_ID[NEED_TO_REPORT_WORD[self.fixed_word]], PAD_ID)
        elif self.model.can_speak:
            report_states = self._report_states(states)
            report_logits = self.model.report_logits(report_states)[0, 0]
            mx.eval(report_logits)
            report_probabilities = np.asarray(
                mx.softmax(report_logits, axis=-1), dtype=np.float64
            )
            drawn: list[int] = []
            for slot in range(self.model.report_slots):
                weights = report_probabilities[slot]
                weights = weights / weights.sum()
                drawn.append(
                    int(self.rng.choices(range(self.model.vocab_size), weights=weights)[0])
                )
            said = tuple(drawn)
        else:
            said = (PAD_ID,) * max(1, self.model.report_slots)
        self._history.append(states)
        return action_index, said


def evaluate_report(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    listener_mode: str = "grounded",
    intervention: str = "none",
    fixed_word: str | None = None,
    mute_organism: bool = False,
    report_overrides: dict[str, object] | None = None,
    collect_traces: bool = False,
) -> dict[str, object]:
    """Drive a trained organism through held-out lives without learning."""

    overrides = dict(report_overrides or {})
    overrides["listener_mode"] = listener_mode
    survived = 0
    viability_sum = 0.0
    steps_sum = 0
    said_need = 0
    truthful = 0
    grants = 0
    truthful_grants = 0
    word_counts = {need: 0 for need in REPORT_NEEDS}
    bucket_said = [0, 0, 0, 0]
    bucket_truthful = [0, 0, 0, 0]
    traces: list[dict[str, object]] = []

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed, **overrides)
        packet = world.reset(seed)
        driver = _ReportDriver(
            model,
            intervention=intervention,
            fixed_word=fixed_word,
            rng=Random(seed + 61_000_003),
        )
        driver.reset()
        life_viability = 0.0
        steps = 0
        while True:
            action_index, said = driver.step(world, packet)
            lowest_before = world.lowest_need()
            tick = world.grid.step_count
            world.hear(None if mute_organism else said)
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            steps += 1
            life_viability += float(info["mean_viability"])
            word = heard_need(said)
            if word is not None:
                word_counts[word] += 1
                if tick >= FIDELITY_WARMUP:
                    said_need += 1
                    correct = int(word == lowest_before)
                    truthful += correct
                    bucket = min(3, tick // 100)
                    bucket_said[bucket] += 1
                    bucket_truthful[bucket] += correct
            if info.get("granted_need") is not None:
                grants += 1
                truthful_grants += int(info["granted_need"] == info["lowest_need"])
            if collect_traces:
                traces.append(
                    {
                        "life": life,
                        "tick": tick,
                        "said": word,
                        "lowest": lowest_before,
                        "needs": info.get("report_needs"),
                    }
                )
            if terminated or truncated:
                survived += int(not terminated)
                break
        viability_sum += life_viability / config.report.life_steps
        steps_sum += steps

    result: dict[str, object] = {
        "lives": lives,
        "listener_mode": listener_mode,
        "intervention": intervention,
        "survival": survived / lives,
        "mean_viability": viability_sum / lives,
        "mean_life_steps": steps_sum / lives,
        "need_word_ticks": said_need,
        "report_fidelity": truthful / max(1, said_need),
        "speech_rate": said_need / max(1, steps_sum),
        "grants": grants,
        "grant_fidelity": truthful_grants / max(1, grants),
        "word_distribution": {
            need: word_counts[need] / max(1, sum(word_counts.values()))
            for need in REPORT_NEEDS
        },
        "fidelity_by_bucket": [
            bucket_truthful[index] / max(1, bucket_said[index]) for index in range(4)
        ],
        "bucket_counts": bucket_said,
    }
    if collect_traces:
        result["traces"] = traces
    return result


def audit_observation_decoder(
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
) -> dict[str, object]:
    """Can the lowest need be read off the senses alone? It must not be.

    A multinomial logistic regression is fitted on the masked observation
    vectors themselves, with the true lowest need as its label — the very
    supervision the organism never receives. Its held-out accuracy is the
    ceiling any purely perceptual shortcut could reach.
    """

    from homesocial.island.report_calibrate import NEED_UTTERANCE, uptake_action

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    rng = Random(seed_base + 991)
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed)
        packet = world.reset(seed)
        while True:
            need = rng.choice(REPORT_NEEDS)
            if world.grid.step_count >= FIDELITY_WARMUP:
                vectors.append(packet.vector())
                labels.append(REPORT_NEEDS.index(world.lowest_need()))
            world.hear(NEED_UTTERANCE[need])
            packet, _, terminated, truncated, _ = world.step(uptake_action(world))
            if terminated or truncated:
                break

    features = np.asarray(vectors, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int32)
    split = int(0.7 * len(features))
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = targets[:split], targets[split:]

    weights = mx.zeros((features.shape[1], len(REPORT_NEEDS)))
    bias = mx.zeros((len(REPORT_NEEDS),))
    train_x_mx = mx.array(train_x)
    train_y_mx = mx.array(train_y)
    for _ in range(400):
        logits = train_x_mx @ weights + bias
        probabilities = mx.softmax(logits, axis=-1)
        onehot = mx.eye(len(REPORT_NEEDS))[train_y_mx]
        error = probabilities - onehot
        weights = weights - 0.5 * (train_x_mx.T @ error) / train_x_mx.shape[0]
        bias = bias - 0.5 * error.mean(axis=0)
        mx.eval(weights, bias)

    predictions = np.asarray(
        mx.argmax(mx.array(test_x) @ weights + bias, axis=-1)
    )
    majority = float(np.mean(test_y == np.bincount(train_y).argmax()))
    return {
        "samples": int(len(features)),
        "heldout_accuracy": float(np.mean(predictions == test_y)),
        "majority_baseline": majority,
        "chance": 1.0 / len(REPORT_NEEDS),
    }


def audit_counterfactual_body(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
) -> dict[str, object]:
    """Same organism, same history, one different portion actually received.

    Each pair runs identically to a fork tick. There the body is given the
    resource it just took in, once at the small portion and once at the large
    one. Every later observation is the same shape; only the amount inside the
    organism differs. If the report follows the body, the two branches diverge.
    """

    from homesocial.island.report_calibrate import NEED_UTTERANCE, uptake_action

    pairs = 0
    diverged = 0
    followed = 0
    both_named = 0
    for life in range(lives):
        seed = seed_base + life
        branch_words: list[list[str | None]] = []
        branch_lowest: list[list[str]] = []
        for large in (False, True):
            world = make_report_world(config, seed=seed)
            packet = world.reset(seed)
            driver = _ReportDriver(model, rng=Random(seed + 71_000_003))
            driver.reset()
            words: list[str | None] = []
            lowest: list[str] = []
            forked = False
            while True:
                action_index, said = driver.step(world, packet)
                if world.grid.step_count >= FIDELITY_WARMUP:
                    words.append(heard_need(said))
                    lowest.append(world.lowest_need())
                world.hear(said)
                packet, _, terminated, truncated, info = execute_agent_action(
                    world,
                    packet,
                    action_index,
                    consume_options=config.consume_options,
                    inspect_options=config.inspect_options,
                )
                event = str(info.get("event") or "")
                if (
                    not forked
                    and world.grid.step_count >= FIDELITY_WARMUP
                    and event
                    in {"consumed_food", "consumed_water", "rested_shelter"}
                ):
                    forked = True
                    need = {
                        "consumed_food": "food",
                        "consumed_water": "water",
                        "rested_shelter": "energy",
                    }[event]
                    portion = (
                        config.report.portion_large
                        if large
                        else config.report.portion_small
                    )
                    # Undo the portion the world gave and apply the branch's.
                    granted = info.get("granted_large")
                    given = (
                        config.report.portion_large
                        if granted
                        else config.report.portion_small
                    )
                    current = getattr(world.grid.needs, need)
                    world.grid.needs = replace(
                        world.grid.needs,
                        **{need: min(1.0, max(0.0, current - given + portion))},
                    )
                if terminated or truncated:
                    break
            branch_words.append(words)
            branch_lowest.append(lowest)

        span = min(len(branch_words[0]), len(branch_words[1]))
        for index in range(span):
            small_low = branch_lowest[0][index]
            large_low = branch_lowest[1][index]
            if small_low == large_low:
                continue
            pairs += 1
            small_word = branch_words[0][index]
            large_word = branch_words[1][index]
            if small_word != large_word:
                diverged += 1
            if small_word is not None and large_word is not None:
                both_named += 1
                followed += int(
                    small_word == small_low and large_word == large_low
                )
    return {
        "divergent_ticks": pairs,
        "report_changed_rate": diverged / max(1, pairs),
        "both_named": both_named,
        "report_followed_body_rate": followed / max(1, both_named),
    }


def report_battery(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    heldout_birth_levels: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.90),
    heldout_portions: tuple[float, float] = (0.25, 0.55),
) -> list[dict[str, object]]:
    """The whole preregistered battery, in one pass."""

    rows: list[dict[str, object]] = []

    grounded = evaluate_report(
        model, config, lives=lives, seed_base=seed_base
    )
    rows.append({"condition": "grounded", **grounded})

    for listener in ("scrambled", "mute"):
        rows.append(
            {
                "condition": f"listener_{listener}",
                **evaluate_report(
                    model,
                    config,
                    lives=lives,
                    seed_base=seed_base,
                    listener_mode=listener,
                ),
            }
        )
    rows.append(
        {
            "condition": "organism_mute",
            **evaluate_report(
                model,
                config,
                lives=lives,
                seed_base=seed_base,
                mute_organism=True,
            ),
        }
    )
    for need in REPORT_NEEDS:
        rows.append(
            {
                "condition": f"fixed_word_{need}",
                **evaluate_report(
                    model,
                    config,
                    lives=lives,
                    seed_base=seed_base,
                    fixed_word=need,
                ),
            }
        )
    for intervention in ("zero", "shuffle", "freeze"):
        rows.append(
            {
                "condition": f"intervention_{intervention}",
                **evaluate_report(
                    model,
                    config,
                    lives=lives,
                    seed_base=seed_base,
                    intervention=intervention,
                ),
            }
        )
    rows.append(
        {
            "condition": "heldout_birth_levels",
            **evaluate_report(
                model,
                config,
                lives=lives,
                seed_base=seed_base + 500_000,
                report_overrides={"birth_levels": heldout_birth_levels},
            ),
        }
    )
    rows.append(
        {
            "condition": "heldout_portions",
            **evaluate_report(
                model,
                config,
                lives=lives,
                seed_base=seed_base + 600_000,
                report_overrides={
                    "portion_small": heldout_portions[0],
                    "portion_large": heldout_portions[1],
                },
            ),
        }
    )
    rows.append(
        {
            "condition": "observation_decoder",
            **audit_observation_decoder(
                config, lives=max(8, lives // 4), seed_base=seed_base + 700_000
            ),
        }
    )
    rows.append(
        {
            "condition": "counterfactual_body",
            **audit_counterfactual_body(
                model, config, lives=max(8, lives // 4), seed_base=seed_base + 800_000
            ),
        }
    )
    return rows


def most_frequent_utterances(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    top: int = 8,
) -> list[tuple[str, str, int]]:
    """What the organism actually says, per true bodily state.

    Returned as (lowest need, utterance, count). This is a description, not a
    gate: it is how a reader sees the learned words rather than a number.
    """

    counts: dict[tuple[str, str], int] = {}
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed)
        packet = world.reset(seed)
        driver = _ReportDriver(model, rng=Random(seed + 81_000_003))
        driver.reset()
        while True:
            action_index, said = driver.step(world, packet)
            lowest = world.lowest_need()
            if world.grid.step_count >= FIDELITY_WARMUP:
                text = " ".join(
                    VOCAB[token] for token in said if token != PAD_ID
                ) or "<silence>"
                key = (lowest, text)
                counts[key] = counts.get(key, 0) + 1
            world.hear(said)
            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            if terminated or truncated:
                break
    ordered = sorted(counts.items(), key=lambda item: -item[1])[:top]
    return [(need, text, count) for (need, text), count in ordered]
