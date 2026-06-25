from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal, Protocol

from .env import Action, Needs, SilentTeacher, SituatedTeacher, WorldObject
from .observations import TEACHER_UTTERANCES


TeacherMode = Literal["grounded", "silent", "masked", "shuffled", "wrong"]

TEACHER_MODES: tuple[TeacherMode, ...] = (
    "grounded",
    "silent",
    "masked",
    "shuffled",
    "wrong",
)


class TeacherLike(Protocol):
    def respond(
        self,
        action: Action,
        obj_ahead: WorldObject | None,
        needs: Needs,
        event: str | None,
    ) -> str | None:
        ...


@dataclass
class MappingTeacher:
    base: TeacherLike
    utterance_map: dict[str, str]

    def respond(
        self,
        action: Action,
        obj_ahead: WorldObject | None,
        needs: Needs,
        event: str | None,
    ) -> str | None:
        utterance = self.base.respond(action, obj_ahead, needs, event)
        if utterance is None:
            return None
        return self.utterance_map.get(utterance, utterance)


def build_teacher(mode: TeacherMode | str, seed: int = 0) -> TeacherLike:
    mode = normalize_teacher_mode(mode)
    if mode == "silent":
        return SilentTeacher()
    if mode in {"grounded", "masked"}:
        return SituatedTeacher()
    if mode == "shuffled":
        return MappingTeacher(SituatedTeacher(), _shuffled_mapping(seed))
    if mode == "wrong":
        return MappingTeacher(SituatedTeacher(), _wrong_mapping())
    raise ValueError(f"Unknown teacher mode: {mode}")


def normalize_teacher_mode(mode: TeacherMode | str) -> TeacherMode:
    aliases = {
        "grounded_teacher": "grounded",
        "silent_teacher": "silent",
        "masked_teacher": "masked",
        "shuffled_teacher": "shuffled",
        "wrong_teacher": "wrong",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in TEACHER_MODES:
        raise ValueError(f"Unknown teacher mode: {mode}")
    return normalized  # type: ignore[return-value]


def masks_language(mode: TeacherMode | str) -> bool:
    return normalize_teacher_mode(mode) == "masked"


def _shuffled_mapping(seed: int) -> dict[str, str]:
    rng = Random(seed)
    shuffled = list(TEACHER_UTTERANCES)
    rng.shuffle(shuffled)
    if all(source == target for source, target in zip(TEACHER_UTTERANCES, shuffled)):
        shuffled = shuffled[1:] + shuffled[:1]
    return dict(zip(TEACHER_UTTERANCES, shuffled))


def _wrong_mapping() -> dict[str, str]:
    return {
        "that is water": "that is thorn",
        "that is berries": "that is thorn",
        "that is hut": "that is thorn",
        "that is thorn": "that is water",
        "that is tree": "that is water",
        "that is rock": "that is berries",
        "drink water": "avoid danger",
        "eat food": "avoid danger",
        "rest at shelter": "avoid danger",
        "avoid danger": "drink water",
        "water helps thirst": "danger hurts you",
        "food helps hunger": "danger hurts you",
        "shelter helps rest": "danger hurts you",
        "danger hurts you": "drink water",
    }
