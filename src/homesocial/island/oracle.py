"""Scripted policies for island calibration.

The oracle is an environment-validation instrument, not a learner: it reads
true object kinds directly from the grid. Gate B1 requires the oracle to
survive >= 95% of full-length lives while a random policy dies fast; only then
is the island a world where death is a learnable consequence of neglect.
"""

from __future__ import annotations

from collections import deque
from random import Random

from homesocial.env import Action, DELTAS, DIRECTION_ORDER, Direction
from homesocial.island.world import IslandGrid

LOW_NEED_THRESHOLD = 0.55
SAFETY_THRESHOLD = 0.60
TOPUP_ENERGY = 0.90
EMERGENCY_ENERGY = 0.20


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self._rng = Random(seed)
        self._actions = list(Action)

    def act(self, grid: IslandGrid) -> Action:
        return self._rng.choice(self._actions)


class OraclePolicy:
    """Greedy need regulation with full kind knowledge and safe pathing."""

    def act(self, grid: IslandGrid) -> Action:
        needs = grid.needs
        if needs.energy < EMERGENCY_ENERGY:
            # Unsheltered rest is net-positive energy; never die mid-trek.
            return Action.REST
        deficits = {
            "food": needs.food,
            "water": needs.water,
            "energy": needs.energy,
        }
        weakest = min(deficits, key=deficits.get)
        if deficits[weakest] < LOW_NEED_THRESHOLD:
            return self._address(grid, weakest)
        if needs.safety < SAFETY_THRESHOLD:
            return self._address(grid, "energy")  # shelter rest restores safety
        if needs.energy < TOPUP_ENERGY:
            return Action.REST
        return Action.WAIT

    def _address(self, grid: IslandGrid, need: str) -> Action:
        kind = {"food": "food", "water": "water", "energy": "shelter"}[need]
        act_on_target = Action.REST if kind == "shelter" else Action.CONSUME
        target_positions = {
            obj.pos for obj in grid.objects if obj.kind == kind
        }
        if not target_positions:
            return Action.REST
        facing = self._facing_target(grid, target_positions)
        if facing is not None:
            return facing if isinstance(facing, Action) else act_on_target
        step_direction = self._bfs_direction(grid, target_positions)
        if step_direction is None:
            return Action.REST
        return self._turn_or_move(grid.direction, step_direction)

    def _facing_target(
        self, grid: IslandGrid, targets: set[tuple[int, int]]
    ) -> Action | bool | None:
        """None if not adjacent; an Action to turn; True if already facing."""

        ax, ay = grid.agent_pos
        for direction in DIRECTION_ORDER:
            dx, dy = DELTAS[direction]
            if (ax + dx, ay + dy) in targets:
                if grid.direction == direction:
                    return True
                return self._turn_toward(grid.direction, direction)
        return None

    def _bfs_direction(
        self, grid: IslandGrid, targets: set[tuple[int, int]]
    ) -> Direction | None:
        """First step direction of a shortest safe path to any target-adjacent cell."""

        blocked = {
            obj.pos
            for obj in grid.objects
            if obj.blocks or obj.kind == "danger" or obj.pos in targets
        }
        goal_cells = set()
        for tx, ty in targets:
            for dx, dy in DELTAS.values():
                cell = (tx + dx, ty + dy)
                if self._walkable(grid, cell, blocked):
                    goal_cells.add(cell)
        start = grid.agent_pos
        if start in goal_cells:
            return None  # adjacent already; handled by _facing_target
        queue = deque([start])
        first_step: dict[tuple[int, int], Direction] = {}
        seen = {start}
        while queue:
            cell = queue.popleft()
            for direction in DIRECTION_ORDER:
                dx, dy = DELTAS[direction]
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in seen or not self._walkable(grid, nxt, blocked):
                    continue
                seen.add(nxt)
                first_step[nxt] = first_step.get(cell, direction)
                if nxt in goal_cells:
                    return first_step[nxt]
                queue.append(nxt)
        return None

    @staticmethod
    def _walkable(
        grid: IslandGrid, cell: tuple[int, int], blocked: set[tuple[int, int]]
    ) -> bool:
        x, y = cell
        return 0 <= x < grid.width and 0 <= y < grid.height and cell not in blocked

    @staticmethod
    def _turn_toward(current: Direction, desired: Direction) -> Action:
        order = DIRECTION_ORDER
        delta = (order.index(desired) - order.index(current)) % len(order)
        return Action.TURN_RIGHT if delta in (1, 2) else Action.TURN_LEFT

    def _turn_or_move(self, current: Direction, desired: Direction) -> Action:
        if current == desired:
            return Action.MOVE_FORWARD
        return self._turn_toward(current, desired)
