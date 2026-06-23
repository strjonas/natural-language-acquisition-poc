from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random

from .env import Action, HomeostaticSocialGrid, Observation


def encode_observation(
    observation: Observation,
    *,
    include_language: bool = True,
    include_object_kinds: bool = True,
) -> tuple[object, ...]:
    if observation.object_ahead is None:
        ahead = "none"
    elif include_object_kinds:
        ahead = observation.object_ahead.kind
    else:
        ahead = "object"

    visible = tuple(
        sorted(
            (
                obj.kind if include_object_kinds else "object",
                obj.pos[0] - observation.position[0],
                obj.pos[1] - observation.position[1],
            )
            for obj in observation.visible
        )
    )
    needs = observation.needs
    utterance = observation.teacher_utterance if include_language else None
    return (
        observation.position,
        observation.direction.value,
        ahead,
        visible,
        _bin_need(needs.food),
        _bin_need(needs.water),
        _bin_need(needs.energy),
        _bin_need(needs.safety),
        utterance,
    )


@dataclass(frozen=True)
class EpisodeStats:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    mean_viability: float
    min_viability: float
    resource_uses: int
    danger_hits: int
    teacher_utterances: int


class QLearningAgent:
    def __init__(
        self,
        *,
        learning_rate: float = 0.25,
        discount: float = 0.95,
        epsilon: float = 0.2,
        seed: int | None = None,
        include_language: bool = True,
        include_object_kinds: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.include_language = include_language
        self.include_object_kinds = include_object_kinds
        self.rng = Random(seed)
        self.actions = tuple(Action)
        self.q: defaultdict[tuple[tuple[object, ...], Action], float] = defaultdict(float)

    def act(self, observation: Observation, *, explore: bool = True) -> Action:
        state = encode_observation(
            observation,
            include_language=self.include_language,
            include_object_kinds=self.include_object_kinds,
        )
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)

        values = [(self.q[(state, action)], action) for action in self.actions]
        max_value = max(value for value, _action in values)
        best = [action for value, action in values if value == max_value]
        return self.rng.choice(best)

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        state = encode_observation(
            observation,
            include_language=self.include_language,
            include_object_kinds=self.include_object_kinds,
        )
        next_state = encode_observation(
            next_observation,
            include_language=self.include_language,
            include_object_kinds=self.include_object_kinds,
        )
        current = self.q[(state, action)]
        if done:
            target = reward
        else:
            target = reward + self.discount * max(
                self.q[(next_state, next_action)] for next_action in self.actions
            )
        self.q[(state, action)] = current + self.learning_rate * (target - current)


def run_episode(
    env: HomeostaticSocialGrid,
    agent: QLearningAgent,
    *,
    seed: int,
    train: bool,
) -> EpisodeStats:
    observation = env.reset(seed=seed)
    total_reward = 0.0
    viability_sum = observation.needs.mean_viability()
    min_viability = observation.needs.viability()
    resource_uses = 0
    danger_hits = 0
    teacher_utterances = 0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(observation, explore=train)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if train:
            agent.update(observation, action, reward, next_observation, done)

        total_reward += reward
        steps += 1
        viability_sum += next_observation.needs.mean_viability()
        min_viability = min(min_viability, next_observation.needs.viability())

        event = info["event"]
        if event in {"consumed_water", "consumed_food", "rested_shelter"}:
            resource_uses += 1
        if event == "hit_danger":
            danger_hits += 1
        if next_observation.teacher_utterance:
            teacher_utterances += 1

        observation = next_observation

    return EpisodeStats(
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        mean_viability=viability_sum / (steps + 1),
        min_viability=min_viability,
        resource_uses=resource_uses,
        danger_hits=danger_hits,
        teacher_utterances=teacher_utterances,
    )


def _bin_need(value: float) -> str:
    if value < 0.25:
        return "critical"
    if value < 0.5:
        return "low"
    if value < 0.75:
        return "ok"
    return "high"
