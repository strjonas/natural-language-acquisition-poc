from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from random import Random
from typing import Iterable


class Action(StrEnum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_FORWARD = "move_forward"
    POINT = "point"
    ASK = "ask"
    CONSUME = "consume"
    REST = "rest"
    WAIT = "wait"


class Direction(StrEnum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"


DIRECTION_ORDER = [
    Direction.NORTH,
    Direction.EAST,
    Direction.SOUTH,
    Direction.WEST,
]

DELTAS = {
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}

STANDARD_MODE = "standard"
LANGUAGE_NECESSARY_MODE = "language_necessary"
DIAGNOSTIC_MODES = (STANDARD_MODE, LANGUAGE_NECESSARY_MODE)
DETERMINISTIC_BODY = "deterministic"
STOCHASTIC_BODY = "stochastic"
BODY_DYNAMICS_MODES = (DETERMINISTIC_BODY, STOCHASTIC_BODY)
STANDARD_RESOURCE_ECOLOGY = "standard"
RICH_RESOURCE_ECOLOGY = "rich"
RESOURCE_ECOLOGIES = (STANDARD_RESOURCE_ECOLOGY, RICH_RESOURCE_ECOLOGY)


@dataclass(frozen=True)
class WorldObject:
    name: str
    kind: str
    pos: tuple[int, int]
    food_delta: float = 0.0
    water_delta: float = 0.0
    energy_delta: float = 0.0
    safety_delta: float = 0.0
    blocks: bool = False
    consumable: bool = False


@dataclass(frozen=True)
class Needs:
    food: float = 0.75
    water: float = 0.75
    energy: float = 0.75
    safety: float = 0.85

    def clipped(self) -> Needs:
        return Needs(
            food=_clip01(self.food),
            water=_clip01(self.water),
            energy=_clip01(self.energy),
            safety=_clip01(self.safety),
        )

    def viability(self) -> float:
        return min(self.food, self.water, self.energy, self.safety)

    def mean_viability(self) -> float:
        return (self.food + self.water + self.energy + self.safety) / 4.0


@dataclass(frozen=True)
class Observation:
    step_count: int
    position: tuple[int, int]
    direction: Direction
    needs: Needs
    visible: tuple[WorldObject, ...]
    object_ahead: WorldObject | None
    teacher_utterance: str | None
    last_event: str | None


class SituatedTeacher:
    """Small deterministic teacher whose speech is contingent on agent action."""

    def respond(
        self,
        action: Action,
        obj_ahead: WorldObject | None,
        needs: Needs,
        event: str | None,
    ) -> str | None:
        if action == Action.POINT and obj_ahead is not None:
            return f"that is {obj_ahead.name}"

        if action == Action.ASK and obj_ahead is not None:
            if obj_ahead.kind == "water":
                return "drink water"
            if obj_ahead.kind == "food":
                return "eat food"
            if obj_ahead.kind == "shelter":
                return "rest at shelter"
            if obj_ahead.kind == "danger":
                return "avoid danger"
            return f"that is {obj_ahead.name}"

        if event == "consumed_water" and needs.water < 0.8:
            return "water helps thirst"
        if event == "consumed_food" and needs.food < 0.8:
            return "food helps hunger"
        if event == "rested_shelter" and needs.energy < 0.8:
            return "shelter helps rest"
        if event in {"hit_danger", "consumed_danger", "rested_danger"}:
            return "danger hurts you"

        return None


class SilentTeacher:
    """No-language control condition with the same world dynamics."""

    def respond(
        self,
        action: Action,
        obj_ahead: WorldObject | None,
        needs: Needs,
        event: str | None,
    ) -> None:
        return None


class HomeostaticSocialGrid:
    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        max_steps: int = 120,
        seed: int | None = None,
        teacher: SituatedTeacher | None = None,
        randomize_world: bool = False,
        diagnostic_mode: str = STANDARD_MODE,
        body_dynamics_mode: str = DETERMINISTIC_BODY,
        renewable_resources: bool = False,
        resource_ecology: str = STANDARD_RESOURCE_ECOLOGY,
    ) -> None:
        if width < 5 or height < 5:
            raise ValueError("Grid must be at least 5x5.")
        if diagnostic_mode not in DIAGNOSTIC_MODES:
            raise ValueError(f"Unknown diagnostic mode: {diagnostic_mode}.")
        if body_dynamics_mode not in BODY_DYNAMICS_MODES:
            raise ValueError(f"Unknown body dynamics mode: {body_dynamics_mode}.")
        if resource_ecology not in RESOURCE_ECOLOGIES:
            raise ValueError(f"Unknown resource ecology: {resource_ecology}.")

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.rng = Random(seed)
        self.teacher = teacher or SituatedTeacher()
        self.randomize_world = randomize_world
        self.diagnostic_mode = diagnostic_mode
        self.body_dynamics_mode = body_dynamics_mode
        self.renewable_resources = renewable_resources
        self.resource_ecology = resource_ecology

        self.step_count = 0
        self.agent_pos = (1, 1)
        self.direction = Direction.EAST
        self.needs = Needs()
        self.objects: list[WorldObject] = []
        self.body_strain = 0.0
        self.food_pressure = 0.0
        self.water_pressure = 0.0
        self.food_metabolism = 0.01
        self.water_metabolism = 0.014
        self.energy_metabolism = 1.0

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng.seed(seed)

        self.step_count = 0
        self.agent_pos = (1, 1)
        self.direction = Direction.EAST
        self.needs = Needs()
        self.body_strain = 0.0
        self.food_pressure = 0.0
        self.water_pressure = 0.0
        if self.body_dynamics_mode == STOCHASTIC_BODY:
            self.food_metabolism = self.rng.uniform(0.007, 0.015)
            self.water_metabolism = self.rng.uniform(0.010, 0.020)
            self.energy_metabolism = self.rng.uniform(0.75, 1.35)
        else:
            self.food_metabolism = 0.01
            self.water_metabolism = 0.014
            self.energy_metabolism = 1.0
        self.objects = self._make_default_world()
        return self._observe(None, None)

    def step(
        self, action: Action | str
    ) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        action = Action(action)
        self.step_count += 1

        before = self.needs.mean_viability()
        event = self._apply_action(action)
        self._apply_metabolism(action)
        body_event = self._apply_stochastic_body_event()
        self.needs = self.needs.clipped()

        obj_ahead = self.object_ahead()
        utterance = self.teacher.respond(action, obj_ahead, self.needs, event)
        obs = self._observe(utterance, body_event or event)

        after = self.needs.mean_viability()
        reward = after - before

        terminated = self.needs.viability() <= 0.0
        truncated = self.step_count >= self.max_steps
        info = {
            "event": event,
            "body_event": body_event,
            "viability": self.needs.viability(),
            "mean_viability": self.needs.mean_viability(),
        }
        return obs, reward, terminated, truncated, info

    def object_ahead(self) -> WorldObject | None:
        return self.object_at(self._ahead_pos())

    def object_at(self, pos: tuple[int, int]) -> WorldObject | None:
        for obj in self.objects:
            if obj.pos == pos:
                return obj
        return None

    def render_ascii(self) -> str:
        chars = {
            "food": "F",
            "water": "W",
            "shelter": "S",
            "danger": "!",
            "tree": "T",
            "rock": "R",
        }
        facing = {
            Direction.NORTH: "^",
            Direction.EAST: ">",
            Direction.SOUTH: "v",
            Direction.WEST: "<",
        }
        rows: list[str] = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.agent_pos:
                    row.append(facing[self.direction])
                    continue
                obj = self.object_at(pos)
                row.append(chars.get(obj.kind, "?") if obj else ".")
            rows.append("".join(row))
        return "\n".join(rows)

    def _apply_action(self, action: Action) -> str | None:
        if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            self.direction = _turn(self.direction, -1 if action == Action.TURN_LEFT else 1)
            return "turned"

        if action == Action.MOVE_FORWARD:
            return self._move_forward()

        if action == Action.CONSUME:
            return self._consume_ahead()

        if action == Action.REST:
            return self._rest()

        if action == Action.POINT:
            return "pointed" if self.object_ahead() else "pointed_empty"

        if action == Action.ASK:
            return "asked" if self.object_ahead() else "asked_empty"

        return "waited"

    def _move_forward(self) -> str:
        target = self._ahead_pos()
        if not self._in_bounds(target):
            return "bumped_wall"

        obj = self.object_at(target)
        if obj and obj.blocks:
            return "blocked"

        self.agent_pos = target
        if obj and obj.kind == "danger":
            self.needs = replace(
                self.needs,
                safety=self.needs.safety + obj.safety_delta,
                energy=self.needs.energy + obj.energy_delta,
            )
            return "hit_danger"

        return "moved"

    def _consume_ahead(self) -> str | None:
        obj = self.object_ahead()
        if obj is None:
            return "consumed_empty"
        if self.diagnostic_mode == LANGUAGE_NECESSARY_MODE and obj.kind == "danger":
            self.needs = replace(
                self.needs,
                energy=self.needs.energy - 0.08,
                safety=self.needs.safety - 0.35,
            )
            return "consumed_danger"
        if not obj.consumable:
            return "not_consumable"

        self.needs = replace(
            self.needs,
            food=self.needs.food + obj.food_delta,
            water=self.needs.water + obj.water_delta,
            energy=self.needs.energy + obj.energy_delta,
            safety=self.needs.safety + obj.safety_delta,
        )
        if not self.renewable_resources:
            self.objects = [candidate for candidate in self.objects if candidate is not obj]

        if obj.kind == "water":
            return "consumed_water"
        if obj.kind == "food":
            return "consumed_food"
        return f"consumed_{obj.kind}"

    def _rest(self) -> str:
        obj_here = self.object_at(self.agent_pos)
        obj_ahead = self.object_ahead()
        near_danger = (obj_here and obj_here.kind == "danger") or (
            obj_ahead and obj_ahead.kind == "danger"
        )
        if self.diagnostic_mode == LANGUAGE_NECESSARY_MODE and near_danger:
            self.needs = replace(
                self.needs,
                energy=self.needs.energy - 0.08,
                safety=self.needs.safety - 0.35,
            )
            return "rested_danger"

        near_shelter = (obj_here and obj_here.kind == "shelter") or (
            obj_ahead and obj_ahead.kind == "shelter"
        )
        if near_shelter:
            self.needs = replace(
                self.needs,
                energy=self.needs.energy + 0.25,
                safety=self.needs.safety + 0.08,
            )
            return "rested_shelter"

        self.needs = replace(self.needs, energy=self.needs.energy + 0.05)
        return "rested_unsheltered"

    def _apply_metabolism(self, action: Action) -> None:
        energy_cost = 0.015
        if action == Action.MOVE_FORWARD:
            energy_cost = 0.035
        elif action in (Action.POINT, Action.ASK):
            energy_cost = 0.02
        elif action == Action.REST:
            energy_cost = -0.02

        if self.body_dynamics_mode == STOCHASTIC_BODY:
            if action == Action.MOVE_FORWARD:
                self.body_strain = min(1.0, 0.72 * self.body_strain + 0.22)
            elif action == Action.REST:
                self.body_strain = max(0.0, 0.30 * self.body_strain - 0.08)
            else:
                self.body_strain *= 0.90
            energy_cost = (
                energy_cost * self.energy_metabolism
                + 0.045 * self.body_strain
                + 0.018 * self.body_strain**2
            )

        food_cost = self.food_metabolism
        water_cost = self.water_metabolism
        if self.body_dynamics_mode == STOCHASTIC_BODY:
            food_cost *= 1.0 + 1.4 * self.food_pressure
            water_cost *= 1.0 + 1.4 * self.water_pressure
            self.food_pressure *= 0.82
            self.water_pressure *= 0.82

        self.needs = replace(
            self.needs,
            food=self.needs.food - food_cost,
            water=self.needs.water - water_cost,
            energy=self.needs.energy - energy_cost,
            safety=self.needs.safety - 0.002,
        )

    def _apply_stochastic_body_event(self) -> str | None:
        if self.body_dynamics_mode != STOCHASTIC_BODY:
            return None
        if self.rng.random() >= 0.18:
            return None

        draw = self.rng.random()
        if draw < 0.28:
            self.food_pressure = min(1.0, self.food_pressure + 0.65)
            self.needs = replace(self.needs, food=self.needs.food - 0.03)
            return "body_hunger"
        if draw < 0.56:
            self.water_pressure = min(1.0, self.water_pressure + 0.65)
            self.needs = replace(self.needs, water=self.needs.water - 0.04)
            return "body_thirst"
        if draw < 0.84:
            self.body_strain = min(1.0, self.body_strain + 0.25)
            self.needs = replace(self.needs, energy=self.needs.energy - 0.10)
            return "body_fatigue"

        self.body_strain = max(0.0, self.body_strain - 0.35)
        self.food_pressure *= 0.45
        self.water_pressure *= 0.45
        self.needs = replace(
            self.needs,
            energy=self.needs.energy + 0.06,
            safety=self.needs.safety + 0.03,
        )
        return "body_recovery"

    def _observe(self, utterance: str | None, event: str | None) -> Observation:
        return Observation(
            step_count=self.step_count,
            position=self.agent_pos,
            direction=self.direction,
            needs=self.needs,
            visible=tuple(self._visible_objects(radius=2)),
            object_ahead=self.object_ahead(),
            teacher_utterance=utterance,
            last_event=event,
        )

    def _visible_objects(self, radius: int) -> Iterable[WorldObject]:
        ax, ay = self.agent_pos
        for obj in self.objects:
            ox, oy = obj.pos
            if abs(ox - ax) <= radius and abs(oy - ay) <= radius:
                yield obj

    def _ahead_pos(self) -> tuple[int, int]:
        dx, dy = DELTAS[self.direction]
        x, y = self.agent_pos
        return (x + dx, y + dy)

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _make_default_world(self) -> list[WorldObject]:
        if self.diagnostic_mode == LANGUAGE_NECESSARY_MODE:
            return self._make_language_necessary_world()

        if self.resource_ecology == RICH_RESOURCE_ECOLOGY:
            return self._make_world_from_specs(self._rich_world_specs())

        return self._make_world_from_specs(self._standard_world_specs())

    def _make_world_from_specs(
        self,
        specs: list[tuple[str, str, float, float, float, float, bool, bool]],
    ) -> list[WorldObject]:
        if self.randomize_world:
            positions = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) != self.agent_pos
            ]
            object_positions = self.rng.sample(positions, len(specs))
        else:
            object_positions = self._fixed_positions(len(specs))
        return [
            WorldObject(
                name,
                kind,
                pos,
                food_delta=food_delta,
                water_delta=water_delta,
                energy_delta=energy_delta,
                safety_delta=safety_delta,
                blocks=blocks,
                consumable=consumable,
            )
            for pos, (
                name,
                kind,
                food_delta,
                water_delta,
                energy_delta,
                safety_delta,
                blocks,
                consumable,
            ) in zip(object_positions, specs)
        ]

    def _make_language_necessary_world(self) -> list[WorldObject]:
        specs = (
            self._rich_language_necessary_specs()
            if self.resource_ecology == RICH_RESOURCE_ECOLOGY
            else self._standard_language_necessary_specs()
        )
        if self.randomize_world:
            positions = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) != self.agent_pos
            ]
            object_positions = self.rng.sample(positions, len(specs))
        else:
            object_positions = self._fixed_positions(len(specs))

        shuffled_specs = self.rng.sample(specs, len(specs))
        return [
            WorldObject(
                "object",
                kind,
                pos,
                food_delta=food_delta,
                water_delta=water_delta,
                energy_delta=energy_delta,
                safety_delta=safety_delta,
                consumable=consumable,
            )
            for pos, (
                kind,
                food_delta,
                water_delta,
                energy_delta,
                safety_delta,
                consumable,
            ) in zip(object_positions, shuffled_specs)
        ]

    def _fixed_positions(self, count: int) -> list[tuple[int, int]]:
        positions = [
            (2, 1),
            (5, 1),
            (1, 5),
            (4, 3),
            (3, 4),
            (5, 5),
            (1, 3),
            (3, 1),
            (5, 3),
            (2, 5),
            (4, 5),
            (6, 2),
            (2, 3),
            (6, 5),
        ]
        if count > len(positions):
            raise ValueError("Not enough fixed positions for resource ecology.")
        return positions[:count]

    def _standard_world_specs(
        self,
    ) -> list[tuple[str, str, float, float, float, float, bool, bool]]:
        return [
            ("water", "water", 0.0, 0.45, 0.0, 0.0, False, True),
            ("berries", "food", 0.4, 0.0, 0.0, 0.0, False, True),
            ("hut", "shelter", 0.0, 0.0, 0.0, 0.0, False, False),
            ("thorn", "danger", 0.0, 0.0, -0.08, -0.35, False, False),
            ("tree", "tree", 0.0, 0.0, 0.0, 0.0, True, False),
            ("rock", "rock", 0.0, 0.0, 0.0, 0.0, True, False),
        ]

    def _rich_world_specs(
        self,
    ) -> list[tuple[str, str, float, float, float, float, bool, bool]]:
        return [
            ("water", "water", 0.0, 0.45, 0.0, 0.0, False, True),
            ("berries", "food", 0.4, 0.0, 0.0, 0.0, False, True),
            ("hut", "shelter", 0.0, 0.0, 0.0, 0.0, False, False),
            ("thorn", "danger", 0.0, 0.0, -0.08, -0.35, False, False),
            ("water", "water", 0.0, 0.38, 0.0, 0.0, False, True),
            ("roots", "food", 0.34, 0.0, 0.0, 0.0, False, True),
            ("spring", "water", 0.0, 0.32, 0.0, 0.0, False, True),
            ("mushroom", "food", 0.28, 0.0, 0.0, 0.0, False, True),
            ("hut", "shelter", 0.0, 0.0, 0.0, 0.0, False, False),
            ("thorn", "danger", 0.0, 0.0, -0.08, -0.35, False, False),
            ("tree", "tree", 0.0, 0.0, 0.0, 0.0, True, False),
            ("rock", "rock", 0.0, 0.0, 0.0, 0.0, True, False),
        ]

    def _standard_language_necessary_specs(
        self,
    ) -> list[tuple[str, float, float, float, float, bool]]:
        return [
            ("water", 0.0, 0.45, 0.0, 0.0, True),
            ("food", 0.4, 0.0, 0.0, 0.0, True),
            ("shelter", 0.0, 0.0, 0.0, 0.0, False),
            ("danger", 0.0, 0.0, -0.08, -0.35, False),
            ("danger", 0.0, 0.0, -0.08, -0.35, False),
            ("danger", 0.0, 0.0, -0.08, -0.35, False),
        ]

    def _rich_language_necessary_specs(
        self,
    ) -> list[tuple[str, float, float, float, float, bool]]:
        return [
            ("water", 0.0, 0.45, 0.0, 0.0, True),
            ("food", 0.4, 0.0, 0.0, 0.0, True),
            ("shelter", 0.0, 0.0, 0.0, 0.0, False),
            ("danger", 0.0, 0.0, -0.08, -0.35, False),
            ("water", 0.0, 0.38, 0.0, 0.0, True),
            ("food", 0.34, 0.0, 0.0, 0.0, True),
            ("water", 0.0, 0.32, 0.0, 0.0, True),
            ("food", 0.28, 0.0, 0.0, 0.0, True),
            ("shelter", 0.0, 0.0, 0.0, 0.0, False),
            ("danger", 0.0, 0.0, -0.08, -0.35, False),
            ("danger", 0.0, 0.0, -0.08, -0.35, False),
            ("danger", 0.0, 0.0, -0.08, -0.35, False),
        ]


def _turn(direction: Direction, offset: int) -> Direction:
    idx = DIRECTION_ORDER.index(direction)
    return DIRECTION_ORDER[(idx + offset) % len(DIRECTION_ORDER)]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
