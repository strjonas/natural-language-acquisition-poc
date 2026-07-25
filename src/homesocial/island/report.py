"""The report island: a world where the body is hidden and help must be asked for.

This is the substrate for gate G3. Three things are true here at once, and it
is their conjunction that makes a self-report testable rather than decorative:

1. **The body is hidden from the senses.** After a single birth reading the
   interoceptive channel is severed. The organism's own need levels are an
   exact function of what it has observed — birth reading, elapsed ticks, and
   the portions it has itself taken in — but nothing in the current
   observation states them. Knowing them requires keeping track.

2. **Help is scarce and periodic.** Every ``help_period`` ticks the caregiver
   grants exactly one help. Supply slightly exceeds total metabolic demand, so
   an organism that allocates each grant to its worst need lives, and one that
   allocates otherwise does not. Nothing else restores the body: the world
   holds no consumables and unsheltered rest gives nothing.

3. **The caregiver cannot see the body.** Its only evidence is what the agent
   says. It parses the last utterance it heard for a need word and grants the
   matching resource. A wrong word is answered helpfully and truthfully with
   the wrong help.

Nothing in this module rewards speech, scores an utterance, or reveals which
word would have been correct. The only path from saying the right thing to
staying alive runs through the body.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.env import Action, Needs, Observation, WorldObject
from homesocial.island.world import (
    SURFACE_INDEX,
    IslandConfig,
    IslandGrid,
    IslandWorld,
    ObsPacket,
)

# The needs an organism can ask about, in the order used by every audit.
REPORT_NEEDS = ("food", "water", "energy")
NEED_TO_REPORT_WORD = {"food": "hungry", "water": "thirsty", "energy": "tired"}
REPORT_WORD_TO_NEED = {word: need for need, word in NEED_TO_REPORT_WORD.items()}
REPORT_WORD_IDS = {
    TOKEN_TO_ID[word]: need for need, word in NEED_TO_REPORT_WORD.items()
}

# What a granted help looks like. Portion size is legible only as the object's
# surface identity, so tracking how much one has taken in is a perceptual fact
# about the world, never an interoceptive one.
HELP_SURFACES: dict[tuple[str, bool], str] = {
    ("food", False): "berry",
    ("food", True): "roots",
    ("water", False): "water",
    ("water", True): "spring",
    ("energy", False): "hut",
    ("energy", True): "mushroom",
}
HELP_SURFACE_NEEDS = {
    surface: need for (need, _), surface in HELP_SURFACES.items()
}

# A shock is the world acting on the body without being asked. It is
# perceptible — a marker with a need-specific surface appears beside the agent
# for exactly one tick — but it is not interoceptive: nothing says how much is
# left, only that something happened and to which need. A schedule fixed in
# advance cannot answer a shock; an organism that keeps track of itself can.
SHOCK_SURFACES = {"food": "thorn", "water": "tree", "energy": "rock"}
SHOCK_SURFACE_NEEDS = {surface: need for need, surface in SHOCK_SURFACES.items()}


@dataclass(frozen=True)
class ReportConfig:
    """Ecology constants. Frozen once feasibility gates are declared met."""

    # Frozen by the feasibility run recorded in
    # runs/organism/report_feasibility/: oracle 95.5% survival, mute 0%,
    # random word 2%, best body-blind rhythm 63.3%. Do not retune.
    help_period: int = 6
    life_steps: int = 400
    portion_small: float = 0.20
    portion_large: float = 0.60
    large_portion_probability: float = 0.5
    # Birth levels are drawn independently per reportable need. The held-out
    # grid is reserved for the generalization audit and never trained on.
    birth_levels: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
    food_metabolism: float = 0.008
    water_metabolism: float = 0.012
    energy_metabolism: float = 0.016
    move_energy_metabolism: float = 0.035
    safety_metabolism: float = 0.002
    shock_probability: float = 0.025
    shock_size: float = 0.25
    reveal_birth_needs: bool = True
    # Audit conditions. ``scrambled`` replaces what the caregiver heard with a
    # uniformly random need word; ``mute`` makes it hear nothing. Both keep the
    # help clock and every other dynamic identical.
    listener_mode: str = "grounded"

    def __post_init__(self) -> None:
        if self.help_period <= 0:
            raise ValueError("help_period must be positive.")
        if self.life_steps <= 0:
            raise ValueError("life_steps must be positive.")
        if not 0.0 < self.portion_small <= self.portion_large:
            raise ValueError("Portions must satisfy 0 < small <= large.")
        if not 0.0 <= self.large_portion_probability <= 1.0:
            raise ValueError("large_portion_probability must be in [0, 1].")
        if not 0.0 <= self.shock_probability <= 1.0:
            raise ValueError("shock_probability must be in [0, 1].")
        if self.shock_size < 0.0:
            raise ValueError("shock_size must be nonnegative.")
        if not self.birth_levels:
            raise ValueError("birth_levels must be non-empty.")
        if any(not 0.0 < level <= 1.0 for level in self.birth_levels):
            raise ValueError("birth_levels must lie in (0, 1].")
        if self.listener_mode not in ("grounded", "scrambled", "mute"):
            raise ValueError(f"Unknown listener mode: {self.listener_mode}.")


class ReportGrid(IslandGrid):
    """An empty island with a randomized birth body and no free restoration."""

    def __init__(self, *, report: ReportConfig, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.report = report
        self.birth_needs = Needs()

    def _make_default_world(self) -> list[WorldObject]:
        levels = self.report.birth_levels
        self.needs = Needs(
            food=self.rng.choice(levels),
            water=self.rng.choice(levels),
            energy=self.rng.choice(levels),
            safety=1.0,
        )
        self.birth_needs = self.needs
        self.agent_pos = (self.width // 2, self.height // 2)
        self.kind_by_surface = {}
        self.choice_need = None
        self.choice_surfaces = ()
        self.choice_round_index = 0
        self.choice_round_start_step = 0
        self.choice_object_template = ()
        self.offered_kind = None
        self.offered_surface = None
        self.offered_pos = None
        return []

    def _consume_ahead(self) -> str | None:
        """Consume granted help, which arrives in hand rather than in the world."""

        obj = self.object_at(self.agent_pos) or self.object_ahead()
        if obj is None:
            return "consumed_empty"
        if not obj.consumable:
            return "not_consumable"
        self.needs = replace(
            self.needs,
            food=self.needs.food + obj.food_delta,
            water=self.needs.water + obj.water_delta,
            energy=self.needs.energy + obj.energy_delta,
            safety=self.needs.safety + obj.safety_delta,
        )
        self.objects = [item for item in self.objects if item is not obj]
        return f"consumed_{obj.kind}"

    def _rest(self) -> str:
        """Rest restores only what a granted shelter provides."""

        obj = self.object_at(self.agent_pos) or self.object_ahead()
        if obj is not None and obj.kind == "shelter":
            self.needs = replace(
                self.needs, energy=self.needs.energy + obj.energy_delta
            )
            self.objects = [item for item in self.objects if item is not obj]
            return "rested_shelter"
        return "rested_unsheltered"

    def _apply_metabolism(self, action: Action) -> None:
        report = self.report
        energy_cost = (
            report.move_energy_metabolism
            if action == Action.MOVE_FORWARD
            else report.energy_metabolism
        )
        self.needs = replace(
            self.needs,
            food=self.needs.food - report.food_metabolism,
            water=self.needs.water - report.water_metabolism,
            energy=self.needs.energy - energy_cost,
            safety=self.needs.safety - report.safety_metabolism,
        )


def heard_need(tokens: tuple[int, ...] | None) -> str | None:
    """The need word a listener hears in an utterance, or None.

    The listener has one fixed rule, decided before the organism existed: the
    first need word in what it hears names the need. Everything else in the
    closed vocabulary is heard and ignored.
    """

    if tokens is None:
        return None
    for token in tokens:
        need = REPORT_WORD_IDS.get(int(token))
        if need is not None:
            return need
    return None


class ReportWorld(IslandWorld):
    """Island facade for the report task."""

    def __init__(
        self,
        config: IslandConfig | None = None,
        *,
        report: ReportConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.report = report or ReportConfig()
        island = config or IslandConfig()
        super().__init__(
            replace(island, max_steps=self.report.life_steps),
            bank=_NO_BANK,
            seed=seed,
        )
        self._pad_id = TOKEN_TO_ID[PAD_TOKEN]
        self._heard_need: str | None = None
        self._heard_tick = -1
        self._last_utterance: tuple[int, ...] | None = None
        self._listener_rng = Random(0)
        self._help_rng = Random(0)
        self._shock_rng = Random(0)
        self._shock_marker: WorldObject | None = None
        self._grants = 0
        self._grants_by_need = {need: 0 for need in REPORT_NEEDS}
        self._silent_grants = 0

    def _build_grid(self, max_steps: int, seed: int | None) -> IslandGrid:
        return ReportGrid(
            report=self.report,
            width=self.config.width,
            height=self.config.height,
            max_steps=max_steps,
            seed=seed,
        )

    # -- life cycle ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> ObsPacket:
        packet = super().reset(seed)
        base = 0 if seed is None else seed
        self._listener_rng = Random(base + 7_700_017)
        self._help_rng = Random(base + 3_300_013)
        self._shock_rng = Random(base + 5_500_011)
        self._shock_marker = None
        self._heard_need = None
        self._heard_tick = -1
        self._last_utterance = None
        self._grants = 0
        self._grants_by_need = {need: 0 for need in REPORT_NEEDS}
        self._silent_grants = 0
        return packet

    def hear(self, tokens: tuple[int, ...] | None) -> None:
        """Deliver one agent utterance to the listener.

        Called once per tick, before the agent's motor action is executed. The
        listener remembers only the most recent need word it has heard since
        its last grant, so an organism must speak within every help window.
        """

        self._last_utterance = tuple(tokens) if tokens is not None else None
        if self.report.listener_mode == "mute":
            return
        if self.report.listener_mode == "scrambled":
            # Matched traffic: whenever the organism says anything at all the
            # listener hears a uniformly random need word instead.
            if tokens is not None and any(
                int(token) != self._pad_id for token in tokens
            ):
                self._heard_need = self._listener_rng.choice(REPORT_NEEDS)
                self._heard_tick = self.grid.step_count
            return
        need = heard_need(self._last_utterance)
        if need is not None:
            self._heard_need = need
            self._heard_tick = self.grid.step_count

    def step(
        self, action: Action | str
    ) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
        action = Action(action)
        needs_before = self.grid.needs
        self._clear_shock_marker()
        observation, reward, terminated, truncated, info = self.grid.step(action)

        shock_need = None if terminated else self._apply_shock()
        if shock_need is not None:
            terminated = self.grid.needs.viability() <= 0.0

        granted_need: str | None = None
        granted_large: bool | None = None
        grant_source_tick: int | None = None
        if not terminated and self.grid.step_count % self.report.help_period == 0:
            granted_need, granted_large, grant_source_tick = self._grant_help()

        self._last_action_index = list(Action).index(action)
        packet = self._packet(self.grid._observe(None, info.get("event")), None)
        info.update(
            {
                "viability": self.grid.needs.viability(),
                "mean_viability": self.grid.needs.mean_viability(),
                "report_needs": tuple(
                    getattr(self.grid.needs, need) for need in REPORT_NEEDS
                ),
                "lowest_need": self.lowest_need(),
                "lowest_need_before": self.lowest_need(needs_before),
                "heard_need": self._heard_need,
                "granted_need": granted_need,
                "granted_large": granted_large,
                "grant_source_tick": grant_source_tick,
                "shock_need": shock_need,
                "help_pending": self.pending_help_need(),
                "utterance_tokens": self._last_utterance,
                "spoke": self._last_utterance is not None
                and any(int(token) != self._pad_id for token in self._last_utterance),
            }
        )
        truncated = truncated or self.grid.step_count >= self.report.life_steps
        return packet, reward, terminated, truncated, info

    # -- the world acting on the body ---------------------------------------

    def _clear_shock_marker(self) -> None:
        if self._shock_marker is None:
            return
        self.grid.objects = [
            obj for obj in self.grid.objects if obj is not self._shock_marker
        ]
        self._shock_marker = None

    def _apply_shock(self) -> str | None:
        report = self.report
        if report.shock_probability <= 0.0:
            return None
        if self._shock_rng.random() >= report.shock_probability:
            return None
        need = self._shock_rng.choice(REPORT_NEEDS)
        self.grid.needs = replace(
            self.grid.needs,
            **{need: getattr(self.grid.needs, need) - report.shock_size},
        ).clipped()
        marker_pos = self._marker_position()
        if marker_pos is not None:
            self._shock_marker = WorldObject(
                SHOCK_SURFACES[need], "tree", marker_pos
            )
            self.grid.objects = [*self.grid.objects, self._shock_marker]
        return need

    def _marker_position(self) -> tuple[int, int] | None:
        ax, ay = self.grid.agent_pos
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            pos = (ax + dx, ay + dy)
            if self.grid._in_bounds(pos) and self.grid.object_at(pos) is None:
                return pos
        return None

    # -- listener -----------------------------------------------------------

    def _grant_help(self) -> tuple[str | None, bool | None, int]:
        """Deliver one help for the last thing heard, spoiling anything unused.

        Also reports which tick's utterance the listener actually acted on. Only
        that utterance had any effect on the world; the rest were said into the
        air. Learning that carries the utterance's causal footprint is what the
        organism is entitled to, and no more.
        """

        # Unused help spoils, so a grant cannot be hoarded and consumed later:
        # the word must be right when it is said. A shock marker is perception,
        # not help, and survives its tick.
        self.grid.objects = [
            obj for obj in self.grid.objects if obj is self._shock_marker
        ]
        need = self._heard_need
        # An unanswered grant still has a moment that decided it: the last thing
        # the organism said before help came, which happened not to be a
        # request. That moment is what gets the credit for the silence.
        source_tick = (
            self._heard_tick if need is not None else self.grid.step_count - 1
        )
        self._heard_need = None
        self._heard_tick = -1
        self._grants += 1
        if need is None:
            self._silent_grants += 1
            return None, None, source_tick
        large = self._help_rng.random() < self.report.large_portion_probability
        portion = (
            self.report.portion_large if large else self.report.portion_small
        )
        surface = HELP_SURFACES[(need, large)]
        pos = self.grid.agent_pos
        if need == "food":
            obj = WorldObject(surface, "food", pos, food_delta=portion, consumable=True)
        elif need == "water":
            obj = WorldObject(surface, "water", pos, water_delta=portion, consumable=True)
        else:
            obj = WorldObject(surface, "shelter", pos, energy_delta=portion)
        self.grid.objects = [*self.grid.objects, obj]
        self._grants_by_need[need] += 1
        return need, large, source_tick

    # -- read-only accessors used by audits ---------------------------------

    def lowest_need(self, needs: Needs | None = None) -> str:
        current = self.grid.needs if needs is None else needs
        return min(REPORT_NEEDS, key=lambda need: getattr(current, need))

    def pending_help_need(self) -> str | None:
        for obj in self.grid.objects:
            need = HELP_SURFACE_NEEDS.get(obj.name)
            if need is not None:
                return need
        return None

    @property
    def grant_counts(self) -> dict[str, int]:
        counts = dict(self._grants_by_need)
        counts["none"] = self._silent_grants
        counts["total"] = self._grants
        return counts

    # -- perception ---------------------------------------------------------

    def _packet(
        self, observation: Observation, words: tuple[str, ...] | None
    ) -> ObsPacket:
        packet = super()._packet(observation, words)
        return replace(
            packet,
            mask_needs=(
                observation.step_count > 0 if self.report.reveal_birth_needs else True
            ),
        )


class _NullBank:
    """The report island has no caregiver commentary; only the agent speaks."""

    def sample(self, situation: object, rng: object) -> tuple[str, ...] | None:
        return None

    def sample_from_act(self, act: str, rng: object) -> tuple[str, ...] | None:
        return None


_NO_BANK = _NullBank()


def report_word_ids() -> tuple[int, ...]:
    """Token ids of the three need words, in ``REPORT_NEEDS`` order."""

    return tuple(TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]] for need in REPORT_NEEDS)


def vocabulary_size() -> int:
    return len(VOCAB)
