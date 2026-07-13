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

from dataclasses import dataclass, field
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

# Surfaces whose kind varies per world, and their candidate kinds.
VARIABLE_SURFACE_KINDS = {
    "spring": ("water", "poison"),
    "berry": ("food", "poison"),
    "roots": ("food", "poison"),
    "mushroom": ("food", "poison"),
}
STABLE_SURFACE_KINDS = {
    "water": "water",
    "hut": "shelter",
    "thorn": "danger",
    "tree": "tree",
    "rock": "rock",
}
FOOD_CANDIDATE_SURFACES = ("berry", "roots", "mushroom")

LANGUAGE_MODES = ("grounded", "silent", "shuffled")


@dataclass(frozen=True)
class IslandConfig:
    width: int = 9
    height: int = 9
    max_steps: int = 1000
    language_mode: str = "grounded"
    ask_state_period: int = 25
    visible_radius: int = 2
    max_visible_slots: int = 8
    low_need_praise_threshold: float = 0.6
    low_need_ask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.language_mode not in LANGUAGE_MODES:
            raise ValueError(f"Unknown language mode: {self.language_mode}.")


class IslandGrid(HomeostaticSocialGrid):
    """Grid with per-world surface->kind assignment and renewable ecology."""

    def __init__(
        self,
        *,
        width: int = 9,
        height: int = 9,
        max_steps: int = 1000,
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

    def _assign_kinds(self) -> dict[str, str]:
        kinds = dict(STABLE_SURFACE_KINDS)
        # At most one of the food-candidate surfaces is poisonous, so the
        # island always offers at least two genuine food sources.
        poison_pick = self.rng.choice((None,) + FOOD_CANDIDATE_SURFACES)
        for surface in FOOD_CANDIDATE_SURFACES:
            kinds[surface] = "poison" if surface == poison_pick else "food"
        kinds["spring"] = "water" if self.rng.random() < 0.5 else "poison"
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
        self.kind_by_surface = self._assign_kinds()
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

    def vector(self, *, max_visible_slots: int = 8, visible_radius: int = 2) -> np.ndarray:
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
        self.bank = bank.shuffled(seed) if language_mode == "shuffled" else bank
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
        if situation is None or self.language_mode == "silent":
            return situation, None
        return situation, self.bank.sample(situation, self._rng)


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
        self.grid = IslandGrid(
            width=self.config.width,
            height=self.config.height,
            max_steps=self.config.max_steps,
            seed=seed,
        )
        self.caregiver = Caregiver(
            bank if bank is not None else load_default_bank(),
            language_mode=self.config.language_mode,
            ask_state_period=self.config.ask_state_period,
            low_need_praise_threshold=self.config.low_need_praise_threshold,
            low_need_ask_threshold=self.config.low_need_ask_threshold,
            seed=seed if seed is not None else 0,
        )
        self._last_action_index = -1

    @property
    def tokens_per_utterance(self) -> int:
        return TOKENS_PER_UTTERANCE

    def reset(self, seed: int | None = None) -> ObsPacket:
        observation = self.grid.reset(seed)
        if seed is not None:
            self.caregiver.reset(seed)
        self._last_action_index = -1
        return self._packet(observation, None)

    def step(
        self, action: Action | str
    ) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
        action = Action(action)
        needs_before = self.grid.needs
        observation, reward, terminated, truncated, info = self.grid.step(action)
        situation, words = self.caregiver.utter(
            self.grid, action, info.get("event"), needs_before
        )
        self._last_action_index = list(Action).index(action)
        packet = self._packet(observation, words)
        info["situation"] = situation.key() if situation is not None else None
        info["utterance"] = " ".join(words) if words else None
        return packet, reward, terminated, truncated, info

    def _packet(
        self, observation: Observation, words: tuple[str, ...] | None
    ) -> ObsPacket:
        ax, ay = observation.position
        visible = sorted(
            (
                (obj.pos[0] - ax, obj.pos[1] - ay, SURFACE_INDEX[obj.name])
                for obj in observation.visible
            ),
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
        )
