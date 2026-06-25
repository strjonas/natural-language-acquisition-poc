from __future__ import annotations

import numpy as np

from .env import (
    BODY_DYNAMICS_MODES,
    DETERMINISTIC_BODY,
    STOCHASTIC_BODY,
    Direction,
    Observation,
)


OBJECT_KINDS = ("water", "food", "shelter", "danger", "tree", "rock")
LAST_EVENTS = (
    "turned",
    "moved",
    "bumped_wall",
    "blocked",
    "consumed_empty",
    "not_consumable",
    "consumed_water",
    "consumed_food",
    "consumed_danger",
    "rested_shelter",
    "rested_unsheltered",
    "rested_danger",
    "pointed",
    "pointed_empty",
    "asked",
    "asked_empty",
    "waited",
    "hit_danger",
)
STOCHASTIC_BODY_EVENTS = (
    "body_hunger",
    "body_thirst",
    "body_fatigue",
    "body_recovery",
)
TEACHER_UTTERANCES = (
    "that is water",
    "that is berries",
    "that is hut",
    "that is thorn",
    "that is tree",
    "that is rock",
    "drink water",
    "eat food",
    "rest at shelter",
    "avoid danger",
    "water helps thirst",
    "food helps hunger",
    "shelter helps rest",
    "danger hurts you",
)
EXACT_INTEROCEPTION = "exact"
MASKED_INTEROCEPTION = "masked"
INTEROCEPTION_MODES = (EXACT_INTEROCEPTION, MASKED_INTEROCEPTION)


def observation_vector(
    observation: Observation,
    *,
    width: int,
    height: int,
    include_language: bool = True,
    mask_language: bool = False,
    include_object_kinds: bool = False,
    interoception_mode: str = EXACT_INTEROCEPTION,
    body_dynamics_mode: str = DETERMINISTIC_BODY,
    max_visible_slots: int = 6,
) -> np.ndarray:
    if interoception_mode not in INTEROCEPTION_MODES:
        raise ValueError(f"Unknown interoception mode: {interoception_mode}.")
    if body_dynamics_mode not in BODY_DYNAMICS_MODES:
        raise ValueError(f"Unknown body dynamics mode: {body_dynamics_mode}.")
    features: list[float] = []

    x, y = observation.position
    features.extend([x / max(1, width - 1), y / max(1, height - 1)])

    features.extend(_one_hot(_direction_index(observation.direction), 4))

    needs = observation.needs
    if interoception_mode == MASKED_INTEROCEPTION:
        features.extend([0.0, 0.0, 0.0, 0.0])
    else:
        features.extend([needs.food, needs.water, needs.energy, needs.safety])

    ahead = observation.object_ahead
    features.append(1.0 if ahead is not None else 0.0)
    if include_object_kinds:
        features.extend(
            _one_hot(
                _object_index(ahead.kind if ahead else None),
                len(OBJECT_KINDS) + 1,
            )
        )

    event_vocabulary = _event_vocabulary(body_dynamics_mode)
    features.extend(
        _one_hot(
            _event_index(observation.last_event, event_vocabulary),
            len(event_vocabulary) + 1,
        )
    )

    visible = tuple(
        sorted(
            observation.visible,
            key=lambda obj: (
                abs(obj.pos[0] - x) + abs(obj.pos[1] - y),
                obj.pos[1],
                obj.pos[0],
            ),
        )
    )
    features.append(min(len(visible), 6) / 6.0)
    slot_size = 4 + ((len(OBJECT_KINDS) + 1) if include_object_kinds else 0)
    for slot in range(max_visible_slots):
        if slot >= len(visible):
            features.extend([0.0 for _ in range(slot_size)])
            continue

        obj = visible[slot]
        dx = (obj.pos[0] - x) / max(1, width - 1)
        dy = (obj.pos[1] - y) / max(1, height - 1)
        is_ahead = 1.0 if ahead is not None and obj.pos == ahead.pos else 0.0
        features.extend([1.0, dx, dy, is_ahead])
        if include_object_kinds:
            features.extend(_one_hot(_object_index(obj.kind), len(OBJECT_KINDS) + 1))

    if include_language:
        utterance = None if mask_language else observation.teacher_utterance
        features.extend(
            _one_hot(
                _utterance_index(utterance),
                len(TEACHER_UTTERANCES) + 1,
            )
        )

    return np.asarray(features, dtype=np.float32)


def observation_vector_size(
    *,
    include_language: bool = True,
    include_object_kinds: bool = False,
    body_dynamics_mode: str = DETERMINISTIC_BODY,
    max_visible_slots: int = 6,
) -> int:
    if body_dynamics_mode not in BODY_DYNAMICS_MODES:
        raise ValueError(f"Unknown body dynamics mode: {body_dynamics_mode}.")
    event_vocabulary = _event_vocabulary(body_dynamics_mode)
    size = 2 + 4 + 4 + 1 + (len(event_vocabulary) + 1) + 1
    if include_object_kinds:
        size += len(OBJECT_KINDS) + 1
    size += max_visible_slots * (
        4 + ((len(OBJECT_KINDS) + 1) if include_object_kinds else 0)
    )
    if include_language:
        size += len(TEACHER_UTTERANCES) + 1
    return size


def teacher_utterance_index(utterance: str | None) -> int:
    return _utterance_index(utterance)


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0 for _ in range(size)]
    values[index] = 1.0
    return values


def _direction_index(direction: Direction) -> int:
    return {
        Direction.NORTH: 0,
        Direction.EAST: 1,
        Direction.SOUTH: 2,
        Direction.WEST: 3,
    }[direction]


def _object_index(kind: str | None) -> int:
    if kind is None or kind not in OBJECT_KINDS:
        return len(OBJECT_KINDS)
    return OBJECT_KINDS.index(kind)


def _utterance_index(utterance: str | None) -> int:
    if utterance is None or utterance not in TEACHER_UTTERANCES:
        return len(TEACHER_UTTERANCES)
    return TEACHER_UTTERANCES.index(utterance)


def _event_index(
    event: str | None,
    vocabulary: tuple[str, ...] = LAST_EVENTS,
) -> int:
    if event is None or event not in vocabulary:
        return len(vocabulary)
    return vocabulary.index(event)


def _event_vocabulary(body_dynamics_mode: str) -> tuple[str, ...]:
    if body_dynamics_mode == STOCHASTIC_BODY:
        return LAST_EVENTS + STOCHASTIC_BODY_EVENTS
    return LAST_EVENTS
