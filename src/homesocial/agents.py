from __future__ import annotations

from random import Random

from .env import Action, Observation, WorldObject


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = Random(seed)
        self.actions = tuple(Action)

    def act(self, observation: Observation) -> Action:
        return self.rng.choice(self.actions)


class NeedSeekingAgent:
    """A tiny hand-coded baseline for checking whether the loop is coherent."""

    def act(self, observation: Observation) -> Action:
        if observation.last_event in {"bumped_wall", "blocked"}:
            return Action.TURN_RIGHT

        ahead = observation.object_ahead
        if ahead is not None:
            if self._is_needed_resource(ahead, observation):
                return Action.CONSUME
            if ahead.kind in {"water", "food", "shelter", "danger"}:
                return Action.ASK
            return Action.TURN_RIGHT

        target = self._nearest_useful_visible_object(observation)
        if target is None:
            return Action.MOVE_FORWARD

        desired_direction = self._desired_direction(observation.position, target.pos)
        if desired_direction != observation.direction.value:
            return Action.TURN_RIGHT
        return Action.MOVE_FORWARD

    def _is_needed_resource(self, obj: WorldObject, observation: Observation) -> bool:
        needs = observation.needs
        if obj.kind == "water":
            return needs.water < 0.9
        if obj.kind == "food":
            return needs.food < 0.9
        if obj.kind == "shelter":
            return needs.energy < 0.85 or needs.safety < 0.85
        return False

    def _nearest_useful_visible_object(self, observation: Observation) -> WorldObject | None:
        useful = [obj for obj in observation.visible if obj.kind in {"water", "food", "shelter"}]
        if not useful:
            return None
        ax, ay = observation.position
        return min(useful, key=lambda obj: abs(obj.pos[0] - ax) + abs(obj.pos[1] - ay))

    def _desired_direction(
        self, current: tuple[int, int], target: tuple[int, int]
    ) -> str:
        ax, ay = current
        tx, ty = target
        if abs(tx - ax) >= abs(ty - ay):
            return "east" if tx > ax else "west"
        return "south" if ty > ay else "north"


class ResourceCyclingAgent:
    """Explores resource contexts without immediately consuming every resource."""

    def __init__(
        self,
        *,
        offset: int = 0,
        phase_length: int = 12,
        linger_steps: int = 3,
    ) -> None:
        self.offset = offset
        self.phase_length = max(1, phase_length)
        self.linger_steps = max(0, linger_steps)
        self.last_target: str | None = None
        self.linger_remaining = self.linger_steps

    def act(self, observation: Observation) -> Action:
        target_kind = self._target_kind(observation.step_count)
        if target_kind != self.last_target:
            self.last_target = target_kind
            self.linger_remaining = self.linger_steps

        if observation.last_event in {"bumped_wall", "blocked"}:
            return Action.TURN_RIGHT

        ahead = observation.object_ahead
        if ahead is not None:
            if ahead.kind == target_kind:
                if self.linger_remaining > 0:
                    self.linger_remaining -= 1
                    return Action.WAIT
                self.linger_remaining = self.linger_steps
                if target_kind in {"food", "water"}:
                    return Action.CONSUME
                if target_kind == "shelter":
                    return Action.REST
                return Action.TURN_RIGHT
            if ahead.kind == "danger":
                return Action.TURN_RIGHT

        target = self._nearest_visible_kind(target_kind, observation)
        if target is None:
            return Action.MOVE_FORWARD

        desired_direction = self._desired_direction(observation.position, target.pos)
        if desired_direction != observation.direction.value:
            return Action.TURN_RIGHT
        return Action.MOVE_FORWARD

    def _target_kind(self, step_count: int) -> str:
        order = ("food", "water", "shelter", "danger")
        phase = step_count // self.phase_length
        return order[(phase + self.offset) % len(order)]

    def _nearest_visible_kind(
        self,
        kind: str,
        observation: Observation,
    ) -> WorldObject | None:
        candidates = [obj for obj in observation.visible if obj.kind == kind]
        if not candidates:
            return None
        ax, ay = observation.position
        return min(candidates, key=lambda obj: abs(obj.pos[0] - ax) + abs(obj.pos[1] - ay))

    def _desired_direction(
        self,
        current: tuple[int, int],
        target: tuple[int, int],
    ) -> str:
        ax, ay = current
        tx, ty = target
        if abs(tx - ax) >= abs(ty - ay):
            return "east" if tx > ax else "west"
        return "south" if ty > ay else "north"


class TeacherFollowingAgent:
    """Sanity-check agent that treats teacher utterances as grounded advice."""

    def __init__(self) -> None:
        self.last_advice: str | None = None

    def act(self, observation: Observation) -> Action:
        if observation.teacher_utterance:
            self.last_advice = observation.teacher_utterance

        if self.last_advice:
            advice = self.last_advice
            self.last_advice = None
            if advice in {"drink water", "eat food", "water helps thirst", "food helps hunger"}:
                return Action.CONSUME
            if advice == "rest at shelter":
                return Action.REST
            if advice in {"avoid danger", "danger hurts you"}:
                return Action.TURN_RIGHT

        if observation.object_ahead is not None:
            return Action.ASK

        if observation.last_event in {"bumped_wall", "blocked"}:
            return Action.TURN_RIGHT

        return Action.MOVE_FORWARD
