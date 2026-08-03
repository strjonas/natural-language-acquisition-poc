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
from homesocial.env import Action, DELTAS, DIRECTION_ORDER
from homesocial.island.report import (
    HELP_SURFACES,
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    SHOCK_SURFACE_NEEDS,
    ReportConfig,
    ReportWorld,
    heard_need,
)
from homesocial.island.world import IslandConfig, ObsPacket, SURFACES
from homesocial.organism.model import OrganismModel
from homesocial.organism.train import (
    ACTIONS,
    OrganismConfig,
    available_action_mask,
    decode_object_option,
    execute_agent_action,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
REPORT_WORD_IDS = tuple(TOKEN_TO_ID[NEED_TO_REPORT_WORD[n]] for n in REPORT_NEEDS)
# Ticks after birth before a report counts. The body is visible exactly once,
# at tick 0; nothing measured here can be an echo of that reading.
FIDELITY_WARMUP = 25
INTERVENTIONS = ("none", "zero", "shuffle", "freeze")
HELP_SURFACE_PORTIONS = {
    surface: (need, large)
    for (need, large), surface in HELP_SURFACES.items()
}


class _ObservableBodyFilter:
    """Audit-only exact-dynamics filter over the learner-visible history.

    This deliberately is not learned and never enters ``OrganismModel``. It
    establishes whether the report ecology is epistemically solvable before a
    learned self-belief is attempted. After the birth packet it reads only
    public packet fields and the organism's own selected action. In particular
    it never reads ``packet.needs``, report truth, simulator event/kind
    metadata, or the listener's hidden parse.
    """

    def __init__(self, config: OrganismConfig) -> None:
        self.report = config.report
        self.consume_options = config.consume_options
        self.inspect_options = config.inspect_options
        self.visible_slots = config.island.max_visible_slots
        self._belief: np.ndarray | None = None

    def reset(self, packet: ObsPacket) -> None:
        if packet.step_count != 0 or packet.mask_needs:
            raise ValueError("The observable filter requires one birth reading.")
        # Use the actual learner-facing vector so the audit cannot accidentally
        # depend on the bookkeeping truth retained in a masked packet.
        self._belief = np.asarray(packet.vector()[:3], dtype=np.float64).copy()

    @property
    def belief(self) -> np.ndarray:
        if self._belief is None:
            raise RuntimeError("The observable filter has not been reset.")
        return self._belief.copy()

    def lowest_need(self) -> str:
        return REPORT_NEEDS[int(np.argmin(self.belief))]

    @staticmethod
    def _surface_at(packet: ObsPacket, dx: int, dy: int) -> str | None:
        for visible_dx, visible_dy, surface_index in packet.visible:
            if visible_dx == dx and visible_dy == dy:
                return SURFACES[surface_index]
        return None

    def _terminal_surface(
        self, packet: ObsPacket, action_index: int
    ) -> tuple[str | None, str | None, int]:
        """Return public option kind, target surface, and routed move count."""

        if action_index < len(ACTIONS):
            action = ACTIONS[action_index]
            if action not in {Action.CONSUME, Action.REST}:
                return None, None, int(action == Action.MOVE_FORWARD)
            surface = self._surface_at(packet, 0, 0)
            if surface is None:
                direction = DIRECTION_ORDER[packet.direction_index]
                dx, dy = DELTAS[direction]
                surface = self._surface_at(packet, dx, dy)
            return action.value, surface, 0

        decoded = decode_object_option(
            action_index,
            consume_options=self.consume_options,
            inspect_options=self.inspect_options,
            visible_slots=self.visible_slots,
        )
        if decoded is None:
            return None, None, 0
        option_kind, slot = decoded
        if slot >= len(packet.visible):
            return option_kind, None, 0
        dx, dy, surface_index = packet.visible[slot]
        distance = abs(dx) + abs(dy)
        # The public kind-blind servo stops adjacent to its target. When the
        # target begins underfoot it first takes one public move away.
        routed_moves = 1 if distance == 0 else max(0, distance - 1)
        return option_kind, SURFACES[surface_index], routed_moves

    def update(
        self,
        before: ObsPacket,
        action_index: int,
        after: ObsPacket,
    ) -> None:
        if self._belief is None:
            raise RuntimeError("The observable filter has not been reset.")
        duration = after.step_count - before.step_count
        if duration <= 0:
            raise ValueError("A lived action must advance primitive time.")

        option_kind, surface, routed_moves = self._terminal_surface(
            before, action_index
        )
        primitive_move = (
            action_index < len(ACTIONS)
            and ACTIONS[action_index] == Action.MOVE_FORWARD
        )
        pre_ticks = duration - 1
        pre_moves = min(pre_ticks, routed_moves)

        # Indexed on the trailing axis so the identical dynamics can also be
        # carried by a particle cloud of shape (particles, 3). The (3,) case is
        # bit-identical to plain positional indexing.
        belief = self._belief
        belief[..., 0] -= self.report.food_metabolism * pre_ticks
        belief[..., 1] -= self.report.water_metabolism * pre_ticks
        belief[..., 2] -= (
            self.report.move_energy_metabolism * pre_moves
            + self.report.energy_metabolism * (pre_ticks - pre_moves)
        )
        belief[...] = np.clip(belief, 0.0, 1.0)

        # An unused object is cleared at an intermediate grant boundary. A
        # boundary on the terminal tick occurs after uptake, so it does not
        # invalidate the action's consequence.
        spoiled_before_terminal = any(
            (before.step_count + offset) % self.report.help_period == 0
            for offset in range(1, duration)
        )
        portion_spec = HELP_SURFACE_PORTIONS.get(surface or "")
        uptake = False
        if portion_spec is not None and not spoiled_before_terminal:
            need, large = portion_spec
            uptake = option_kind == "consume" or (
                option_kind == Action.REST.value and need == "energy"
            )
            if uptake:
                portion = (
                    self.report.portion_large
                    if large
                    else self.report.portion_small
                )
                belief[..., REPORT_NEEDS.index(need)] += portion

        final_energy_cost = (
            self.report.move_energy_metabolism
            if primitive_move
            else self.report.energy_metabolism
        )
        belief[..., 0] -= self.report.food_metabolism
        belief[..., 1] -= self.report.water_metabolism
        belief[..., 2] -= final_energy_cost
        belief[...] = np.clip(belief, 0.0, 1.0)

        # A final-tick shock is visible in the next packet. Shocks occurring
        # inside a temporally abstract option are not visible and intentionally
        # remain epistemic error for this audit rather than being read from
        # privileged ``info``.
        visible_surfaces = {
            SURFACES[surface_index] for _, _, surface_index in after.visible
        }
        shock_needs = {
            SHOCK_SURFACE_NEEDS[surface]
            for surface in visible_surfaces
            if surface in SHOCK_SURFACE_NEEDS
        }
        if len(shock_needs) > 1:
            raise RuntimeError("A report tick exposed more than one shock marker.")
        if shock_needs:
            shock_need = next(iter(shock_needs))
            belief[..., REPORT_NEEDS.index(shock_need)] -= self.report.shock_size
            belief[...] = np.clip(belief, 0.0, 1.0)


def _evaluate_observable_history_filter(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    listener_mode: str,
) -> dict[str, float]:
    """Deploy the exact-dynamics visible-history upper bound in the real loop."""

    survived = 0
    correct = 0
    samples = 0
    steps = 0
    absolute_error = 0.0
    error_samples = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(
            config, seed=seed, listener_mode=listener_mode
        )
        packet = world.reset(seed)
        body_filter = _ObservableBodyFilter(config)
        body_filter.reset(packet)
        driver = _ReportDriver(model, rng=Random(seed + 171_000_003))
        driver.reset()
        while True:
            action_index, _ = driver.step(world, packet)
            predicted_need = body_filter.lowest_need()
            true_need = world.lowest_need()
            if packet.step_count >= FIDELITY_WARMUP:
                samples += 1
                correct += int(predicted_need == true_need)
            world.hear(
                (TOKEN_TO_ID[NEED_TO_REPORT_WORD[predicted_need]], PAD_ID)
            )
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            body_filter.update(before, action_index, packet)
            truth = np.asarray(info["report_needs"], dtype=np.float64)
            absolute_error += float(np.abs(body_filter.belief - truth).mean())
            error_samples += 1
            steps += 1
            if terminated or truncated:
                survived += int(not terminated)
                break
    return {
        "lives": float(lives),
        "survival": survived / lives,
        "mean_life_steps": steps / lives,
        "need_reconstruction": correct / max(1, samples),
        "report_fidelity": correct / max(1, samples),
        "samples": float(samples),
        "mean_absolute_need_error": absolute_error / max(1, error_samples),
    }


def audit_observable_history_portion_fork(
    config: OrganismConfig,
    *,
    seed: int = 4_700_001,
) -> dict[str, float]:
    """Paired visible small/large grant must alter belief only after uptake."""

    worlds: list[ReportWorld] = []
    packets: list[ObsPacket] = []
    filters: list[_ObservableBodyFilter] = []
    for large in (False, True):
        world = make_report_world(
            config,
            seed=seed,
            listener_mode="grounded",
            birth_levels=(0.35,),
            shock_probability=0.0,
            unified_uptake=True,
        )
        packet = world.reset(seed)
        world.force_next_help_portion(large=large)
        body_filter = _ObservableBodyFilter(config)
        # The filter uses the overridden ecology constants below, while its
        # action/option configuration remains the organism's public interface.
        body_filter.report = world.report
        body_filter.reset(packet)
        worlds.append(world)
        packets.append(packet)
        filters.append(body_filter)

    word = (TOKEN_TO_ID[NEED_TO_REPORT_WORD["food"]], PAD_ID)
    wait_index = ACTIONS.index(Action.WAIT)
    for _ in range(config.report.help_period):
        for branch in range(2):
            world = worlds[branch]
            before = packets[branch]
            world.hear(word)
            after, _, terminated, truncated, _ = execute_agent_action(
                world,
                before,
                wait_index,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            assert not terminated and not truncated
            filters[branch].update(before, wait_index, after)
            packets[branch] = after

    surface_sets = [
        {SURFACES[index] for _, _, index in packet.visible}
        for packet in packets
    ]
    before_uptake_l1 = float(
        np.abs(filters[0].belief - filters[1].belief).sum()
    )
    consume_index = ACTIONS.index(Action.CONSUME)
    for branch in range(2):
        world = worlds[branch]
        before = packets[branch]
        world.hear(word)
        after, _, terminated, truncated, _ = execute_agent_action(
            world,
            before,
            consume_index,
            consume_options=config.consume_options,
            inspect_options=config.inspect_options,
        )
        assert not terminated and not truncated
        filters[branch].update(before, consume_index, after)
        packets[branch] = after

    belief_delta = filters[1].belief - filters[0].belief
    truth_delta = np.asarray(
        [
            getattr(worlds[1].grid.needs, need)
            - getattr(worlds[0].grid.needs, need)
            for need in REPORT_NEEDS
        ],
        dtype=np.float64,
    )
    max_tracking_error = float(np.max(np.abs(belief_delta - truth_delta)))
    expected_delta = config.report.portion_large - config.report.portion_small
    surface_diverged = surface_sets[0] != surface_sets[1]
    passed = (
        surface_diverged
        and before_uptake_l1 <= 1e-9
        and abs(float(belief_delta[0]) - expected_delta) <= 1e-9
        and abs(float(belief_delta[1])) <= 1e-9
        and abs(float(belief_delta[2])) <= 1e-9
        and max_tracking_error <= 1e-9
    )
    return {
        "surface_diverged": float(surface_diverged),
        "preuptake_belief_l1": before_uptake_l1,
        "postuptake_food_belief_delta": float(belief_delta[0]),
        "postuptake_water_belief_delta": float(belief_delta[1]),
        "postuptake_energy_belief_delta": float(belief_delta[2]),
        "max_truth_tracking_error": max_tracking_error,
        "fork_passed": float(passed),
    }


def audit_observable_history_filter_feasibility(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int = 200,
    seed_base: int = 4_500_000,
) -> dict[str, float]:
    """Epistemic upper bound before learning an explicit self-belief."""

    grounded = _evaluate_observable_history_filter(
        model,
        config,
        lives=lives,
        seed_base=seed_base,
        listener_mode="grounded",
    )
    scrambled = _evaluate_observable_history_filter(
        model,
        config,
        lives=lives,
        seed_base=seed_base + 100_000,
        listener_mode="scrambled",
    )
    fork = audit_observable_history_portion_fork(
        config, seed=seed_base + 200_000
    )
    result = {
        **{f"grounded_{key}": value for key, value in grounded.items()},
        **{f"scrambled_{key}": value for key, value in scrambled.items()},
        "grounded_minus_scrambled_survival": (
            grounded["survival"] - scrambled["survival"]
        ),
        **fork,
    }
    result["gate_passed"] = float(
        grounded["need_reconstruction"] >= 0.90
        and grounded["survival"] >= 0.80
        and grounded["report_fidelity"] >= 0.60
        and result["grounded_minus_scrambled_survival"] >= 0.15
        and fork["fork_passed"] == 1.0
    )
    return result


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
        self.last_states: mx.array | None = None
        self._history: list[mx.array] = []
        self._frozen: mx.array | None = None

    def reset(self) -> None:
        self.hidden = None
        self.last_states = None
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
        self.last_states = states
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


def audit_oracle_listener_uptake(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
) -> dict[str, object]:
    """Can the frozen motor policy take each kind of correctly requested help?

    This is a post-hoc substrate diagnostic, not a report-capability gate.  A
    privileged driver supplies the correct need word while the organism's
    report is ignored.  If one granted resource is systematically not taken
    up, joint discovery of that word and its distinct motor response creates a
    circular credit-assignment barrier before truthful reporting can emerge.
    """

    grants = {need: 0 for need in REPORT_NEEDS}
    uptakes = {need: 0 for need in REPORT_NEEDS}
    survived = 0
    steps = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed)
        packet = world.reset(seed)
        driver = _ReportDriver(model, rng=Random(seed + 91_000_003))
        driver.reset()
        while True:
            action_index, _ = driver.step(world, packet)
            world.hear(
                (
                    TOKEN_TO_ID[NEED_TO_REPORT_WORD[world.lowest_need()]],
                    PAD_ID,
                )
            )
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            steps += 1
            granted = info.get("granted_need")
            if granted in grants:
                grants[str(granted)] += 1
            event_need = {
                "consumed_food": "food",
                "consumed_water": "water",
                "consumed_shelter": "energy",
                "rested_shelter": "energy",
            }.get(str(info.get("event") or ""))
            if event_need is not None:
                uptakes[event_need] += 1
            if terminated or truncated:
                survived += int(not terminated)
                break

    return {
        "lives": lives,
        "survival": survived / lives,
        "mean_life_steps": steps / lives,
        "grants_by_need": grants,
        "uptakes_by_need": uptakes,
        "uptake_per_grant": {
            need: uptakes[need] / max(1, grants[need]) for need in REPORT_NEEDS
        },
    }


def _fit_ridge_decoder(
    train_x: np.ndarray,
    train_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the fixed disposable ridge readout used only by audits."""

    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train = (train_x - mean) / scale
    train = np.concatenate(
        [train, np.ones((len(train), 1), dtype=np.float32)], axis=1
    )
    targets = np.eye(len(REPORT_NEEDS), dtype=np.float32)[train_y]
    penalty = np.eye(train.shape[1], dtype=np.float32) * 1e-3
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ targets)
    return mean, scale, weights


def _ridge_decoder_predict(
    features: np.ndarray,
    decoder: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, scale, weights = decoder
    normalized = (features - mean) / scale
    normalized = np.concatenate(
        [normalized, np.ones((len(normalized), 1), dtype=np.float32)], axis=1
    )
    return np.argmax(normalized @ weights, axis=1)


def _ridge_decoder_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    """Fixed ridge classifier used only to read a frozen representation."""

    predicted = _ridge_decoder_predict(
        test_x, _fit_ridge_decoder(train_x, train_y)
    )
    recalls = [
        float(np.mean(predicted[test_y == label] == label))
        for label in range(len(REPORT_NEEDS))
        if np.any(test_y == label)
    ]
    return {
        "accuracy": float(np.mean(predicted == test_y)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def audit_supervised_state_reporter_upper_bound(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    train_lives: int = 140,
    test_lives: int = 60,
    seed_base: int = 3_000_000,
) -> dict[str, float]:
    """Can any fixed state readout close the report-control loop?

    True bodily labels fit only a disposable audit decoder on separate lives;
    no organism weight is updated. Training lives use oracle speech so the
    decoder sees long regulated histories. Held-out deployment gets only the
    organism state and uses its predicted need word to control the real
    listener. This is an upper bound, never a learned-agent claim.
    """

    if train_lives <= 0 or test_lives <= 0:
        raise ValueError("State-reporter upper bound requires train/test lives.")
    train_states: list[np.ndarray] = []
    train_labels: list[int] = []
    for life in range(train_lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed)
        packet = world.reset(seed)
        driver = _ReportDriver(
            model,
            fixed_word="food",
            rng=Random(seed + 131_000_003),
        )
        driver.reset()
        while True:
            action_index, _ = driver.step(world, packet)
            label = REPORT_NEEDS.index(world.lowest_need())
            if world.grid.step_count >= FIDELITY_WARMUP:
                assert driver.last_states is not None
                train_states.append(
                    np.asarray(driver.last_states[0, 0], dtype=np.float32)
                )
                train_labels.append(label)
            said = (
                TOKEN_TO_ID[NEED_TO_REPORT_WORD[REPORT_NEEDS[label]]],
                PAD_ID,
            )
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

    decoder = _fit_ridge_decoder(
        np.stack(train_states), np.asarray(train_labels, dtype=np.int64)
    )
    survived = 0
    steps_total = 0
    predictions: list[int] = []
    labels: list[int] = []
    for life in range(test_lives):
        seed = seed_base + train_lives + life
        world = make_report_world(config, seed=seed)
        packet = world.reset(seed)
        driver = _ReportDriver(
            model,
            fixed_word="food",
            rng=Random(seed + 131_000_003),
        )
        driver.reset()
        while True:
            action_index, _ = driver.step(world, packet)
            assert driver.last_states is not None
            state = np.asarray(driver.last_states[0, 0], dtype=np.float32)[None, :]
            predicted = int(_ridge_decoder_predict(state, decoder)[0])
            true_label = REPORT_NEEDS.index(world.lowest_need())
            if world.grid.step_count >= FIDELITY_WARMUP:
                predictions.append(predicted)
                labels.append(true_label)
            said = (
                TOKEN_TO_ID[NEED_TO_REPORT_WORD[REPORT_NEEDS[predicted]]],
                PAD_ID,
            )
            world.hear(said)
            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            steps_total += 1
            if terminated or truncated:
                survived += int(not terminated)
                break

    predicted_array = np.asarray(predictions, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    recalls = [
        float(np.mean(predicted_array[label_array == label] == label))
        for label in range(len(REPORT_NEEDS))
        if np.any(label_array == label)
    ]
    return {
        "train_lives": float(train_lives),
        "test_lives": float(test_lives),
        "train_samples": float(len(train_states)),
        "test_samples": float(len(labels)),
        "survival": survived / test_lives,
        "mean_life_steps": steps_total / test_lives,
        "report_fidelity": float(np.mean(predicted_array == label_array)),
        "balanced_report_fidelity": float(np.mean(recalls)),
    }


def audit_hidden_self_state_decoder(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
) -> dict[str, object]:
    """Can a held-out linear readout recover the body from the shared state?

    The frozen organism drives its ordinary grounded loop.  Samples are split
    by whole life, never by adjacent ticks.  A fixed ridge classifier receives
    either the recurrent state that actually feeds the mouth or the current
    masked observation.  True need labels train only these disposable audit
    decoders and never touch the organism.
    """

    split_life = max(1, int(0.7 * lives))
    train_states: list[np.ndarray] = []
    train_observations: list[np.ndarray] = []
    train_labels: list[int] = []
    test_states: list[np.ndarray] = []
    test_observations: list[np.ndarray] = []
    test_labels: list[int] = []

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed)
        packet = world.reset(seed)
        driver = _ReportDriver(model, rng=Random(seed + 101_000_003))
        driver.reset()
        while True:
            action_index, said = driver.step(world, packet)
            if world.grid.step_count >= FIDELITY_WARMUP:
                assert driver.last_states is not None
                state = np.asarray(driver.last_states[0, 0], dtype=np.float32)
                observation = np.asarray(packet.vector(), dtype=np.float32)
                label = REPORT_NEEDS.index(world.lowest_need())
                if life < split_life:
                    train_states.append(state)
                    train_observations.append(observation)
                    train_labels.append(label)
                else:
                    test_states.append(state)
                    test_observations.append(observation)
                    test_labels.append(label)
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

    train_y = np.asarray(train_labels, dtype=np.int64)
    test_y = np.asarray(test_labels, dtype=np.int64)
    if not train_labels or not test_labels:
        raise RuntimeError("Self-state decoder audit collected no held-out samples.")
    state_metrics = _ridge_decoder_metrics(
        np.stack(train_states), train_y, np.stack(test_states), test_y
    )
    observation_metrics = _ridge_decoder_metrics(
        np.stack(train_observations),
        train_y,
        np.stack(test_observations),
        test_y,
    )
    majority = float(
        np.mean(test_y == np.bincount(train_y, minlength=len(REPORT_NEEDS)).argmax())
    )
    return {
        "lives": lives,
        "train_samples": len(train_labels),
        "heldout_samples": len(test_labels),
        "state_accuracy": state_metrics["accuracy"],
        "state_balanced_accuracy": state_metrics["balanced_accuracy"],
        "observation_accuracy": observation_metrics["accuracy"],
        "observation_balanced_accuracy": observation_metrics["balanced_accuracy"],
        "majority_baseline": majority,
        "chance": 1.0 / len(REPORT_NEEDS),
    }


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
    """Same pre-fork history, one perceptible portion and consequence changed.

    A reference oracle first fixes the requests and motor actions for the
    entire life.  Both branches replay that exogenous stream, including the
    same shocks and ordinary portion draws.  At the first grant after warmup,
    exactly one variable is changed: the small branch sees and receives the
    small portion, while the large branch sees and receives the large portion.
    The surface cue is essential.  Silently editing the hidden body would ask
    the organism to report an intervention it had no epistemic access to.
    """

    from homesocial.island.report_calibrate import NEED_UTTERANCE, uptake_action

    pairs = 0
    diverged = 0
    followed = 0
    both_named = 0
    forked_lives = 0
    fork_grant_tick = (
        (FIDELITY_WARMUP // config.report.help_period) + 1
    ) * config.report.help_period
    for life in range(lives):
        seed = seed_base + life
        # Lock the social and motor stream before either causal branch exists.
        # The truthful reference is used only as an environment driver; none
        # of its hidden-state access reaches the organism or its report head.
        reference = make_report_world(config, seed=seed)
        reference_packet = reference.reset(seed)
        requests: list[str] = []
        actions: list[Action] = []
        while True:
            request = reference.lowest_need()
            action = uptake_action(reference)
            requests.append(request)
            actions.append(action)
            reference.hear(NEED_UTTERANCE[request])
            reference_packet, _, terminated, truncated, _ = reference.step(action)
            if terminated or truncated:
                break

        branch_words: list[list[str | None]] = []
        branch_lowest: list[list[str]] = []
        branch_forked: list[bool] = []
        for large in (False, True):
            world = make_report_world(config, seed=seed)
            packet = world.reset(seed)
            driver = _ReportDriver(model, rng=Random(seed + 71_000_003))
            driver.reset()
            words: list[str | None] = []
            lowest: list[str] = []
            forked = False
            uptake_after_fork = False
            for request, action in zip(requests, actions, strict=True):
                _, said = driver.step(world, packet)
                tick = world.grid.step_count
                if tick + 1 == fork_grant_tick:
                    world.force_next_help_portion(large=large)
                if uptake_after_fork:
                    words.append(heard_need(said))
                    lowest.append(world.lowest_need())
                # Replay the locked request and motor action, rather than
                # allowing a changed report to create downstream perceptual
                # differences that would confound the portion intervention.
                world.hear(NEED_UTTERANCE[request])
                packet, _, terminated, truncated, info = world.step(action)
                if info.get("granted_need") is not None and tick + 1 == fork_grant_tick:
                    forked = True
                if (
                    forked
                    and tick >= fork_grant_tick
                    and str(info.get("event") or "")
                    in {
                        "consumed_food",
                        "consumed_water",
                        "consumed_shelter",
                        "rested_shelter",
                    }
                ):
                    uptake_after_fork = True
                if terminated or truncated:
                    break
            branch_words.append(words)
            branch_lowest.append(lowest)
            branch_forked.append(uptake_after_fork)

        if not all(branch_forked):
            continue
        forked_lives += 1

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
        "forked_lives": forked_lives,
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
