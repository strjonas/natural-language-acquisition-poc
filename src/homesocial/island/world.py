"""The creole island world.

Differences from the probe-era grid:

- **Surface/kind decoupling.** Objects have stable surface names (what they
  look like: berry, spring, thorn, ...) but their bodily kind (food, water,
  danger, ...) is assigned per world at reset. Perception exposes only the
  surface; the caregiver's label is the only source of kind information, so
  language is necessary by construction and fast-mapping is testable.
- **Recalibrated viability.** Resources are renewable and the horizon is long
  (1000 steps). A competent policy should survive indefinitely; death is a
  learnable consequence of neglect or ignorance, not ecology noise.
- **Token language.** The caregiver speaks creole token sequences sampled from
  the LLM-generated utterance bank, with matched-shape silent and shuffled
  controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from random import Random

import numpy as np

from homesocial.creole.bank import UtteranceBank, load_default_bank
from homesocial.creole.situations import Situation
from homesocial.creole.vocab import TOKENS_PER_UTTERANCE, encode_utterance
from homesocial.env import (
    Action,
    DELTAS,
    DIRECTION_ORDER,
    Direction,
    HomeostaticSocialGrid,
    Needs,
    Observation,
    SilentTeacher,
    WorldObject,
)

SURFACES = (
    "water",
    "spring",
    "berry",
    "roots",
    "mushroom",
    "hut",
    "thorn",
    "tree",
    "rock",
)
SURFACE_INDEX = {surface: index for index, surface in enumerate(SURFACES)}

# Environment kinds. "poison" is a consumable hazard (harm on consume, safe to
# walk past); "danger" is a contact hazard (harm on walk-in). The creole maps
# both to the word "danger".
ENV_KIND_TO_CREOLE = {
    "water": "water",
    "food": "food",
    "shelter": "shelter",
    "danger": "danger",
    "poison": "danger",
    "tree": "tree",
    "rock": "rock",
}

# (surface, count): the island's fixed population of object instances.
ISLAND_POPULATION = (
    ("water", 2),
    ("spring", 1),
    ("berry", 2),
    ("roots", 1),
    ("mushroom", 1),
    ("hut", 2),
    ("thorn", 2),
    ("tree", 1),
    ("rock", 1),
)

# Consumable surface identities are deliberately exchangeable across lives.
# Each reset assigns the quota below by a seeded uniform shuffle, so no surface
# has a privileged out-of-language prior over bodily kind.
CONSUMABLE_SURFACES = ("water", "spring", "berry", "roots", "mushroom")
CONSUMABLE_KIND_QUOTA = ("food", "food", "water", "water", "poison")
VARIABLE_SURFACE_KINDS = {
    surface: ("food", "water", "poison") for surface in CONSUMABLE_SURFACES
}
STABLE_SURFACE_KINDS = {
    "hut": "shelter",
    "thorn": "danger",
    "tree": "tree",
    "rock": "rock",
}

LANGUAGE_MODES = ("grounded", "silent", "shuffled")
CAREGIVER_SEED_SALT = 1_000_003
SEMANTIC_CHOICE_SHUFFLED_KINDS = {
    2: ("food", "water", "danger", "danger"),
    3: ("food", "water", "danger"),
}


def _caregiver_seed(seed: int | None) -> int:
    """Derive a deterministic language RNG stream independent of the grid.

    SplitMix64-style avalanche mixing avoids the measurable cross-life
    association produced when two Mersenne Twister instances received merely
    offset versions of the same small integer seed.
    """

    value = ((0 if seed is None else seed) + CAREGIVER_SEED_SALT) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


@dataclass(frozen=True)
class IslandConfig:
    width: int = 9
    height: int = 9
    max_steps: int = 1000
    semantic_choice_trial: bool = False
    semantic_choice_horizon: int = 20
    semantic_choice_objects: int = 2
    semantic_choice_low_need: float = 0.35
    # A multi-round childhood keeps one surface-kind mapping alive across
    # recurring bodily demands so a label can pay rent more than once.
    semantic_choice_rounds: int = 1
    # Zero preserves the original free-action choice task. Positive values
    # enable the delayed-choice protocol: after every object label, control is
    # returned to the agent only after a fixed-duration, padding-only return
    # to the canonical center/NORTH pose.
    semantic_choice_return_duration: int = 0
    language_mode: str = "grounded"
    ask_state_period: int = 25
    visible_radius: int = 2
    max_visible_slots: int = 8
    low_need_praise_threshold: float = 0.6
    low_need_ask_threshold: float = 0.5
    caregiver_offer_threshold: float = 0.0
    caregiver_offer_distance: int = 0

    def __post_init__(self) -> None:
        if self.language_mode not in LANGUAGE_MODES:
            raise ValueError(f"Unknown language mode: {self.language_mode}.")
        if self.semantic_choice_horizon <= 0:
            raise ValueError("semantic_choice_horizon must be positive.")
        if self.semantic_choice_objects not in {2, 3}:
            raise ValueError("semantic_choice_objects must be 2 or 3.")
        if self.semantic_choice_rounds <= 0:
            raise ValueError("semantic_choice_rounds must be positive.")
        if not 0.0 < self.semantic_choice_low_need < 0.75:
            raise ValueError("semantic_choice_low_need must be in (0, 0.75).")
        if self.semantic_choice_return_duration < 0:
            raise ValueError("semantic_choice_return_duration must be nonnegative.")
        if 0 < self.semantic_choice_return_duration < 6:
            raise ValueError(
                "semantic_choice_return_duration must be zero or at least 6."
            )
        if (
            self.semantic_choice_trial
            and self.max_visible_slots < self.semantic_choice_objects
        ):
            minimum = "two" if self.semantic_choice_objects == 2 else "three"
            raise ValueError(
                f"semantic_choice_trial requires at least {minimum} visible slots."
            )
        if not 0.0 <= self.caregiver_offer_threshold <= 1.0:
            raise ValueError("caregiver_offer_threshold must be in [0, 1].")
        if self.caregiver_offer_distance < 0:
            raise ValueError("caregiver_offer_distance must be nonnegative.")


class IslandGrid(HomeostaticSocialGrid):
    """Grid with per-world surface->kind assignment and renewable ecology."""

    def __init__(
        self,
        *,
        width: int = 9,
        height: int = 9,
        max_steps: int = 1000,
        semantic_choice_trial: bool = False,
        semantic_choice_objects: int = 2,
        semantic_choice_low_need: float = 0.35,
        semantic_choice_rounds: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            seed=seed,
            teacher=SilentTeacher(),
            randomize_world=True,
            renewable_resources=True,
        )
        self.kind_by_surface: dict[str, str] = {}
        self.semantic_choice_trial = semantic_choice_trial
        self.semantic_choice_objects = semantic_choice_objects
        self.semantic_choice_low_need = semantic_choice_low_need
        self.semantic_choice_rounds = semantic_choice_rounds
        self.choice_need: str | None = None
        self.choice_surfaces: tuple[str, ...] = ()
        self.choice_round_index = 0
        self.choice_round_start_step = 0
        self.choice_object_template: tuple[WorldObject, ...] = ()
        self.offered_kind: str | None = None
        self.offered_surface: str | None = None
        self.offered_pos: tuple[int, int] | None = None

    def _assign_kinds(self) -> dict[str, str]:
        kinds = dict(STABLE_SURFACE_KINDS)
        assigned = list(CONSUMABLE_KIND_QUOTA)
        self.rng.shuffle(assigned)
        kinds.update(zip(CONSUMABLE_SURFACES, assigned))
        return kinds

    def _object_for(self, surface: str, kind: str, pos: tuple[int, int]) -> WorldObject:
        if kind == "water":
            return WorldObject(surface, kind, pos, water_delta=0.40, consumable=True)
        if kind == "food":
            return WorldObject(surface, kind, pos, food_delta=0.40, consumable=True)
        if kind == "poison":
            return WorldObject(
                surface, kind, pos,
                safety_delta=-0.30, energy_delta=-0.05, consumable=True,
            )
        if kind == "shelter":
            return WorldObject(surface, kind, pos)
        if kind == "danger":
            return WorldObject(surface, kind, pos, safety_delta=-0.25, energy_delta=-0.05)
        return WorldObject(surface, kind, pos, blocks=True)

    def _make_default_world(self) -> list[WorldObject]:
        self.offered_kind = None
        self.offered_surface = None
        self.offered_pos = None
        self.kind_by_surface = self._assign_kinds()
        self.choice_need = None
        self.choice_surfaces = ()
        self.choice_round_index = 0
        self.choice_round_start_step = 0
        self.choice_object_template = ()
        if self.semantic_choice_trial:
            return self._make_semantic_choice_world()
        surfaces = [
            surface for surface, count in ISLAND_POPULATION for _ in range(count)
        ]
        free_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.agent_pos
        ]
        positions = self.rng.sample(free_cells, len(surfaces))
        return [
            self._object_for(surface, self.kind_by_surface[surface], pos)
            for surface, pos in zip(surfaces, positions)
        ]

    def start_next_semantic_choice_round(self) -> Observation:
        """Advance one persistent-mapping round after consequence observation."""

        if not self.semantic_choice_trial:
            raise ValueError("No semantic-choice round is active.")
        if self.choice_round_index + 1 >= self.semantic_choice_rounds:
            raise ValueError("The final semantic-choice round has completed.")
        if not self.choice_object_template or self.choice_need is None:
            raise RuntimeError("Semantic-choice round template is unavailable.")

        # The forced transition is one learner decision for replay/credit
        # accounting but represents a new exogenous bodily demand, not another
        # metabolism tick. Consumption consequences were exposed in the packet
        # immediately before this transition.
        self.step_count += 1
        self.choice_round_index += 1
        self.choice_round_start_step = self.step_count
        self.choice_need = "water" if self.choice_need == "food" else "food"
        self.needs = replace(
            self.needs,
            food=(
                self.semantic_choice_low_need
                if self.choice_need == "food"
                else 0.75
            ),
            water=(
                self.semantic_choice_low_need
                if self.choice_need == "water"
                else 0.75
            ),
            energy=0.75,
            safety=0.75,
        )
        self.agent_pos = (self.width // 2, self.height // 2)
        self.direction = Direction.NORTH
        self.objects = list(self.choice_object_template)
        self.offered_kind = None
        self.offered_surface = None
        self.offered_pos = None
        return self._observe(None, None)

    def _make_semantic_choice_world(self) -> list[WorldObject]:
        """Make a remapped two- or three-way semantic choice.

        The full five-surface hidden assignment still exists in
        ``kind_by_surface``. Two-way trials preserve the original need-matched
        resource versus poison probe. Three-way trials cross body context with
        one food, one water, and one poison so the functional word cannot be
        recovered from the currently low need.
        """

        self.choice_need = self.rng.choice(("food", "water"))
        self.needs = replace(
            self.needs, **{self.choice_need: self.semantic_choice_low_need}
        )

        matching_surfaces = [
            surface
            for surface in CONSUMABLE_SURFACES
            if self.kind_by_surface[surface] == self.choice_need
        ]
        poison_surface = next(
            surface
            for surface in CONSUMABLE_SURFACES
            if self.kind_by_surface[surface] == "poison"
        )
        resource_surface = self.rng.choice(matching_surfaces)
        if self.semantic_choice_objects == 2:
            surfaces = [resource_surface, poison_surface]
        else:
            other_need = "water" if self.choice_need == "food" else "food"
            wrong_surface = self.rng.choice(
                [
                    surface
                    for surface in CONSUMABLE_SURFACES
                    if self.kind_by_surface[surface] == other_need
                ]
            )
            surfaces = [resource_surface, wrong_surface, poison_surface]
        self.rng.shuffle(surfaces)
        self.choice_surfaces = tuple(surfaces)

        # Put the body at the center and the pair on opposite sides of a
        # randomly selected axis.  Its initial heading is perpendicular to the
        # pair, making either option require the same turn/move/consume motion.
        center = (self.width // 2, self.height // 2)
        self.agent_pos = center
        if self.semantic_choice_objects == 2 and self.rng.choice((False, True)):
            positions = [
                (center[0] - 2, center[1]),
                (center[0] + 2, center[1]),
            ]
            self.direction = self.rng.choice((Direction.NORTH, Direction.SOUTH))
        elif self.semantic_choice_objects == 2:
            positions = [
                (center[0], center[1] - 2),
                (center[0], center[1] + 2),
            ]
            self.direction = self.rng.choice((Direction.EAST, Direction.WEST))
        else:
            cardinal_positions = [
                (center[0] - 2, center[1]),
                (center[0] + 2, center[1]),
                (center[0], center[1] - 2),
                (center[0], center[1] + 2),
            ]
            positions = self.rng.sample(cardinal_positions, 3)
            self.direction = self.rng.choice(tuple(DIRECTION_ORDER))
        objects = [
            self._object_for(surface, self.kind_by_surface[surface], pos)
            for surface, pos in zip(surfaces, positions)
        ]
        self.choice_object_template = tuple(objects)
        return objects

    def offer(self, kind: str, *, distance: int = 0) -> None:
        """Put a renewable resource in hand or at a visible nearby cell."""

        if kind not in {"food", "water"}:
            raise ValueError(f"Cannot offer non-consumable kind: {kind}.")
        candidates = [
            surface
            for surface, assigned_kind in self.kind_by_surface.items()
            if assigned_kind == kind
        ]
        self.offered_kind = kind
        self.offered_surface = self.rng.choice(candidates)
        if distance <= 0:
            self.offered_pos = self.agent_pos
            return
        ax, ay = self.agent_pos
        positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if abs(x - ax) + abs(y - ay) == distance
            and not (
                (obj := self.object_at((x, y))) is not None and obj.blocks
            )
        ]
        self.offered_pos = self.rng.choice(positions) if positions else self.agent_pos

    def _consume_ahead(self) -> str | None:
        can_consume_offer = (
            self.offered_kind is not None
            and self.offered_pos in {self.agent_pos, self._ahead_pos()}
        )
        if not can_consume_offer:
            return super()._consume_ahead()
        assert self.offered_kind is not None
        kind = self.offered_kind
        if kind == "food":
            self.needs = replace(self.needs, food=self.needs.food + 0.40)
        else:
            self.needs = replace(self.needs, water=self.needs.water + 0.40)
        self.offered_kind = None
        self.offered_surface = None
        self.offered_pos = None
        return f"consumed_{kind}"


@dataclass(frozen=True)
class ObsPacket:
    """What the learner senses: surfaces, body, tokens. Kinds are never here."""

    step_count: int
    position: tuple[int, int]
    direction_index: int
    needs: tuple[float, float, float, float]
    last_action_index: int  # -1 before the first action
    visible: tuple[tuple[int, int, int], ...]  # (dx, dy, surface_index)
    tokens: tuple[int, ...]
    grid_size: tuple[int, int] = (9, 9)
    max_visible_slots: int = 8
    visible_radius: int = 2

    def vector(
        self,
        *,
        max_visible_slots: int | None = None,
        visible_radius: int | None = None,
    ) -> np.ndarray:
        if max_visible_slots is None:
            max_visible_slots = self.max_visible_slots
        if visible_radius is None:
            visible_radius = self.visible_radius
        parts: list[float] = list(self.needs)
        direction_onehot = [0.0] * len(DIRECTION_ORDER)
        direction_onehot[self.direction_index] = 1.0
        parts.extend(direction_onehot)
        width, height = self.grid_size
        parts.append(self.position[0] / max(1, width - 1))
        parts.append(self.position[1] / max(1, height - 1))
        action_onehot = [0.0] * len(Action)
        if self.last_action_index >= 0:
            action_onehot[self.last_action_index] = 1.0
        parts.extend(action_onehot)
        for slot in range(max_visible_slots):
            if slot < len(self.visible):
                dx, dy, surface_index = self.visible[slot]
                surface_onehot = [0.0] * len(SURFACES)
                surface_onehot[surface_index] = 1.0
                parts.extend(
                    [1.0, dx / visible_radius, dy / visible_radius, *surface_onehot]
                )
            else:
                parts.extend([0.0] * (3 + len(SURFACES)))
        return np.asarray(parts, dtype=np.float32)

    @staticmethod
    def vector_size(*, max_visible_slots: int = 8) -> int:
        return (
            4
            + len(DIRECTION_ORDER)
            + 2
            + len(Action)
            + max_visible_slots * (3 + len(SURFACES))
        )


NEED_TO_WORD = {
    "food": "hungry",
    "water": "thirsty",
    "energy": "tired",
    "safety": "hurt",
}
NEED_TO_RESOURCE_KIND = {"food": "food", "water": "water", "energy": "shelter"}


class Caregiver:
    """Deterministic joint-attention speaker over the utterance bank."""

    def __init__(
        self,
        bank: UtteranceBank,
        *,
        language_mode: str = "grounded",
        ask_state_period: int = 25,
        low_need_praise_threshold: float = 0.6,
        low_need_ask_threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        if language_mode not in LANGUAGE_MODES:
            raise ValueError(f"Unknown language mode: {language_mode}.")
        self.language_mode = language_mode
        self.bank = bank
        self.ask_state_period = ask_state_period
        self.low_need_praise_threshold = low_need_praise_threshold
        self.low_need_ask_threshold = low_need_ask_threshold
        self._rng = Random(seed)

    def reset(self, seed: int) -> None:
        self._rng = Random(seed)

    def situation_for(
        self,
        grid: IslandGrid,
        action: Action,
        event: str | None,
        needs_before: Needs,
    ) -> Situation | None:
        obj_ahead = grid.object_ahead()
        if event == "consumed_poison":
            verb = "drink" if obj_ahead and obj_ahead.name in ("water", "spring") else "eat"
            return Situation("correct", (("verb", verb),))
        if event == "hit_danger":
            return Situation("warn")
        if action in (Action.POINT, Action.ASK) and obj_ahead is not None:
            creole_kind = ENV_KIND_TO_CREOLE[obj_ahead.kind]
            return Situation(
                "label", (("surface", obj_ahead.name), ("kind", creole_kind))
            )
        if action == Action.ASK and obj_ahead is None:
            return self._answer_where(grid)
        if obj_ahead is not None and obj_ahead.kind == "danger":
            return Situation("warn", (("place", "there"),))
        if self._praiseworthy(event, needs_before):
            return Situation("praise")
        if self.ask_state_period > 0 and grid.step_count % self.ask_state_period == 0:
            return self._ask_state(grid.needs)
        return None

    def _praiseworthy(self, event: str | None, needs_before: Needs) -> bool:
        threshold = self.low_need_praise_threshold
        if event == "consumed_water":
            return needs_before.water < threshold
        if event == "consumed_food":
            return needs_before.food < threshold
        if event == "rested_shelter":
            return needs_before.energy < threshold
        return False

    def _answer_where(self, grid: IslandGrid) -> Situation | None:
        needs = grid.needs
        regulable = {
            name: getattr(needs, name) for name in ("food", "water", "energy")
        }
        weakest = min(regulable, key=regulable.get)
        target_kind = NEED_TO_RESOURCE_KIND[weakest]
        target = self._nearest_object(grid, target_kind)
        if target is None:
            return None
        place = self._place_word(grid, target.pos)
        return Situation("answer_where", (("kind", target_kind), ("place", place)))

    @staticmethod
    def _nearest_object(grid: IslandGrid, kind: str) -> WorldObject | None:
        ax, ay = grid.agent_pos
        candidates = [obj for obj in grid.objects if obj.kind == kind]
        if not candidates:
            return None
        return min(candidates, key=lambda obj: abs(obj.pos[0] - ax) + abs(obj.pos[1] - ay))

    @staticmethod
    def _place_word(grid: IslandGrid, pos: tuple[int, int]) -> str:
        ax, ay = grid.agent_pos
        dx, dy = pos[0] - ax, pos[1] - ay
        dx_ahead, dy_ahead = DELTAS[grid.direction]
        if (ax + dx_ahead, ay + dy_ahead) == pos:
            return "here"
        if abs(dx) <= 2 and abs(dy) <= 2:
            return "there"
        if abs(dx) >= abs(dy):
            return "east" if dx > 0 else "west"
        return "south" if dy > 0 else "north"

    def _ask_state(self, needs: Needs) -> Situation:
        values = {name: getattr(needs, name) for name in NEED_TO_WORD}
        weakest = min(values, key=values.get)
        if values[weakest] < self.low_need_ask_threshold:
            return Situation("ask_state", (("need", NEED_TO_WORD[weakest]),))
        return Situation("ask_state")

    def utter(
        self,
        grid: IslandGrid,
        action: Action,
        event: str | None,
        needs_before: Needs,
    ) -> tuple[Situation | None, tuple[str, ...] | None]:
        situation = self.situation_for(grid, action, event, needs_before)
        return situation, self.utter_situation(situation)

    def utter_situation(
        self, situation: Situation | None
    ) -> tuple[str, ...] | None:
        if situation is None or self.language_mode == "silent":
            return None
        if self.language_mode == "shuffled":
            # Draw anew for every event.  In particular, a label's surface form
            # is independent of its true surface/kind rather than a stable
            # substitution cipher learned during a run.  Keeping the draw
            # within the same act preserves label traffic and other pragmatic
            # speech-act identities without leaking their grounded slots.
            return self.bank.sample_from_act(situation.act, self._rng)
        return self.bank.sample(situation, self._rng)

    def utter_semantic_choice_label(
        self, situation: Situation, *, choice_objects: int = 2
    ) -> tuple[str, ...] | None:
        """Emit a length-matched minimal label for the paired-choice trial."""

        if situation.act != "label":
            raise ValueError("Semantic-choice speech must be an object label.")
        if self.language_mode == "silent":
            return None
        if self.language_mode == "shuffled":
            kind = self._rng.choice(
                SEMANTIC_CHOICE_SHUFFLED_KINDS[choice_objects]
            )
        else:
            kind = situation.slot("kind")
            assert kind is not None
        return ("this", kind)


class IslandWorld:
    """Environment facade: recalibrated grid + caregiver + token observations."""

    def __init__(
        self,
        config: IslandConfig | None = None,
        *,
        bank: UtteranceBank | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or IslandConfig()
        grid_max_steps = (
            (
                self.config.semantic_choice_horizon
                * self.config.semantic_choice_rounds
                + self.config.semantic_choice_rounds
                - 1
            )
            if self.config.semantic_choice_trial
            else self.config.max_steps
        )
        self.grid = IslandGrid(
            width=self.config.width,
            height=self.config.height,
            max_steps=grid_max_steps,
            semantic_choice_trial=self.config.semantic_choice_trial,
            semantic_choice_objects=self.config.semantic_choice_objects,
            semantic_choice_low_need=self.config.semantic_choice_low_need,
            semantic_choice_rounds=self.config.semantic_choice_rounds,
            seed=seed,
        )
        self.caregiver = Caregiver(
            bank if bank is not None else load_default_bank(),
            language_mode=self.config.language_mode,
            ask_state_period=self.config.ask_state_period,
            low_need_praise_threshold=self.config.low_need_praise_threshold,
            low_need_ask_threshold=self.config.low_need_ask_threshold,
            seed=_caregiver_seed(seed),
        )
        self.caregiver_offer_threshold = self.config.caregiver_offer_threshold
        self.caregiver_offer_distance = self.config.caregiver_offer_distance
        self._last_action_index = -1
        self._inspected_surfaces: set[str] = set()
        self._semantic_choice_return_pending = False
        self._semantic_choice_round_pending = False

    @property
    def tokens_per_utterance(self) -> int:
        return TOKENS_PER_UTTERANCE

    def reset(self, seed: int | None = None) -> ObsPacket:
        observation = self.grid.reset(seed)
        if seed is not None:
            self.caregiver.reset(_caregiver_seed(seed))
        self._last_action_index = -1
        self._inspected_surfaces.clear()
        self._semantic_choice_return_pending = False
        self._semantic_choice_round_pending = False
        return self._packet(observation, None)

    @property
    def semantic_choice_return_pending(self) -> bool:
        """Whether the next transition is the fixed delayed-choice return."""

        return self._semantic_choice_return_pending

    @property
    def semantic_choice_round_pending(self) -> bool:
        """Whether a consequence packet awaits the next bodily demand."""

        return self._semantic_choice_round_pending

    @property
    def semantic_choice_center(self) -> tuple[int, int]:
        return (self.config.width // 2, self.config.height // 2)

    def complete_semantic_choice_return(self) -> None:
        self._semantic_choice_return_pending = False

    def start_next_semantic_choice_round(
        self,
    ) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
        """Emit the forced non-agent transition into the next choice round."""

        if not self._semantic_choice_round_pending:
            raise ValueError("No semantic-choice round transition is pending.")
        observation = self.grid.start_next_semantic_choice_round()
        self._semantic_choice_round_pending = False
        self._last_action_index = list(Action).index(Action.WAIT)
        packet = self._packet(observation, None)
        viability = min(packet.needs)
        return (
            packet,
            0.0,
            False,
            False,
            {
                "event": "semantic_choice_round_transition",
                "viability": viability,
                "mean_viability": sum(packet.needs) / len(packet.needs),
                "semantic_choice_trial": True,
                "semantic_choice_round_transition": True,
                "semantic_choice_round_index": self.grid.choice_round_index,
                "choice_need": self.grid.choice_need,
            },
        )

    def step(
        self, action: Action | str
    ) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
        action = Action(action)
        needs_before = self.grid.needs
        choice_need_before = self.grid.choice_need
        choice_round_index_before = self.grid.choice_round_index
        object_ahead_before = self.grid.object_ahead()
        inspected_before = set(self._inspected_surfaces)
        offered_before = self.grid.offered_kind
        offered_consumable_before = (
            action == Action.CONSUME
            and offered_before is not None
            and self.grid.offered_pos
            in {self.grid.agent_pos, self.grid._ahead_pos()}
        )
        observation, reward, terminated, truncated, info = self.grid.step(action)
        info["offered_consumed"] = (
            offered_consumable_before
            and info.get("event") == f"consumed_{offered_before}"
        )
        self._maybe_offer()
        if self.grid.offered_kind is not None:
            situation = Situation(
                "offer", (("kind", self.grid.offered_kind),)
            )
            words = self.caregiver.utter_situation(situation)
        elif self.config.semantic_choice_trial and not (
            action in (Action.ASK, Action.POINT)
            and object_ahead_before is not None
        ):
            # Paired childhood isolates one linguistic pathway: jointly attend
            # to an object and ask/point for its label. Empty-space ASK must not
            # leak the ordinary island's need-matched ``answer_where`` hint,
            # and terminal praise/correction cannot become an extra target.
            situation, words = None, None
        elif self.config.semantic_choice_trial:
            situation = self.caregiver.situation_for(
                self.grid, action, info.get("event"), needs_before
            )
            assert situation is not None and situation.act == "label"
            words = self.caregiver.utter_semantic_choice_label(
                situation,
                choice_objects=self.config.semantic_choice_objects,
            )
        else:
            situation, words = self.caregiver.utter(
                self.grid, action, info.get("event"), needs_before
            )
        if (
            action in (Action.ASK, Action.POINT)
            and object_ahead_before is not None
            and situation is not None
            and situation.act == "label"
        ):
            self._inspected_surfaces.add(object_ahead_before.name)
            if (
                self.config.semantic_choice_trial
                and self.config.semantic_choice_return_duration > 0
            ):
                self._semantic_choice_return_pending = True

        consumed_kind: str | None = None
        consumed_surface: str | None = None
        event = info.get("event")
        if (
            self.config.semantic_choice_trial
            and object_ahead_before is not None
            and object_ahead_before.consumable
            and event in {"consumed_food", "consumed_water", "consumed_poison"}
        ):
            consumed_kind = object_ahead_before.kind
            consumed_surface = object_ahead_before.name
            final_round = (
                choice_round_index_before + 1
                >= self.config.semantic_choice_rounds
            )
            if not terminated and not final_round:
                truncated = False
                self._semantic_choice_round_pending = True
            else:
                truncated = True

        timeout = (
            self.config.semantic_choice_trial
            and consumed_surface is None
            and (
                self.grid.step_count - self.grid.choice_round_start_step
                >= self.config.semantic_choice_horizon
            )
        )
        if timeout:
            truncated = True
        info.update(
            {
                "semantic_choice_trial": self.config.semantic_choice_trial,
                "choice_need": choice_need_before,
                "semantic_choice_round_index": choice_round_index_before,
                "semantic_choice_rounds": self.config.semantic_choice_rounds,
                "semantic_choice_round_complete": consumed_surface is not None,
                "semantic_choice_round_pending": (
                    self._semantic_choice_round_pending
                ),
                "chosen_kind": consumed_kind,
                "chosen_surface": consumed_surface,
                "chosen_surface_inspected": (
                    consumed_surface in inspected_before
                    if consumed_surface is not None
                    else False
                ),
                "correct": (
                    consumed_kind is not None
                    and consumed_kind == choice_need_before
                ),
                "poison": consumed_kind == "poison",
                "wrong_resource": (
                    consumed_kind in {"food", "water"}
                    and consumed_kind != choice_need_before
                ),
                "timeout": timeout,
            }
        )
        self._last_action_index = list(Action).index(action)
        packet = self._packet(observation, words)
        info["situation"] = situation.key() if situation is not None else None
        info["utterance"] = " ".join(words) if words else None
        return packet, reward, terminated, truncated, info

    def _maybe_offer(self) -> None:
        if self.config.semantic_choice_trial or self.grid.offered_kind is not None:
            return
        threshold = self.caregiver_offer_threshold
        needs = self.grid.needs
        candidates = {
            kind: getattr(needs, kind)
            for kind in ("food", "water")
            if getattr(needs, kind) < threshold
        }
        if candidates:
            self.grid.offer(
                min(candidates, key=candidates.get),
                distance=self.caregiver_offer_distance,
            )

    def _packet(
        self, observation: Observation, words: tuple[str, ...] | None
    ) -> ObsPacket:
        ax, ay = observation.position
        visible_items = [
                (obj.pos[0] - ax, obj.pos[1] - ay, SURFACE_INDEX[obj.name])
                for obj in observation.visible
        ]
        if (
            self.grid.offered_surface is not None
            and self.grid.offered_pos is not None
        ):
            visible_items.append(
                (
                    self.grid.offered_pos[0] - ax,
                    self.grid.offered_pos[1] - ay,
                    SURFACE_INDEX[self.grid.offered_surface],
                )
            )
        visible = sorted(
            visible_items,
            key=lambda item: (abs(item[0]) + abs(item[1]), item[2]),
        )[: self.config.max_visible_slots]
        return ObsPacket(
            step_count=observation.step_count,
            position=observation.position,
            direction_index=DIRECTION_ORDER.index(observation.direction),
            needs=(
                observation.needs.food,
                observation.needs.water,
                observation.needs.energy,
                observation.needs.safety,
            ),
            last_action_index=self._last_action_index,
            visible=tuple(visible),
            tokens=encode_utterance(words),
            grid_size=(self.config.width, self.config.height),
            max_visible_slots=self.config.max_visible_slots,
            visible_radius=self.config.visible_radius,
        )
