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
# Every bodily axis this body actually has. ``safety`` is not reportable and no
# help restores it, but it depletes and it enters both viability signals, so it
# is a real variable of the body whether or not any model has a slot for it.
BODY_NEEDS = ("food", "water", "energy", "safety")
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
HELP_SURFACE_LARGE = {
    surface: large for (_, large), surface in HELP_SURFACES.items()
}

# Two words the caregiver already knows and has never had a use for. Which one
# means "large" is a fact about the listener, not about the language: nothing
# tells the organism, and the probe64 listener model has to find it out from
# which surface appears. They are drawn from the existing closed vocabulary
# because the point is to make more of it carry meaning, never to add symbols.
SIZE_WORDS = {"more": "large", "not": "small"}
SIZE_TO_WORD = {size: word for word, size in SIZE_WORDS.items()}
SIZE_WORD_IDS = {TOKEN_TO_ID[word]: size for word, size in SIZE_WORDS.items()}
PORTION_SIZES = ("small", "large")

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
    # Default-off observability lever for the probe62 ceiling survey. A silent
    # shock changes the body exactly as a loud one does and is drawn from the
    # same stream; only its perceptible marker is withheld. So this constant
    # moves what the organism can *know* about itself while leaving what happens
    # to it bit-identical. The silence draw comes from its own generator, so
    # every other stream stays untouched at any value, and the draws are nested:
    # the silent set at 0.25 is a subset of the silent set at 0.50.
    silent_shock_probability: float = 0.0
    # Default-off levers for probe63. They are a pair and neither is meaningful
    # alone.
    #
    # ``metabolic_spread`` makes the body *individual*: each life scales every
    # axis's per-tick metabolism by its own draw from Uniform(1-s, 1+s). Until
    # now every organism in this ecology burned fuel at exactly the species
    # rate, so a self-model built from species constants was already correct
    # about this body and nothing in it was about *this* organism. A spread
    # makes those constants systematically wrong per individual, in a
    # need-specific direction that persists for the whole life -- which is the
    # "systematically biased body the current model cannot represent" that
    # phase B1 asks for.
    #
    # ``interoception_probability`` is the only channel through which that fact
    # could ever be found out: the body is otherwise visible exactly once, at
    # birth. A reading is delivered in ``info`` and never in the packet, so the
    # motor path and everything the policy sees stay bit-identical at any rate.
    #
    # Both draw from dedicated generators, so at their defaults every other
    # stream in the ecology is untouched.
    metabolic_spread: float = 0.0
    # The second way a body can be individual: how much good the same help
    # does it. The granted surface is public and still names the portion class,
    # so what varies is absorption, which is private. Paired with
    # ``metabolic_spread`` this lets the ground truth *move* -- one world where
    # only the burn rate is individual, one where only absorption is, one where
    # both are -- so a self-model that merely detects "something is off" can be
    # told apart from one that finds out *what* about itself is off.
    uptake_spread: float = 0.0
    interoception_probability: float = 0.0
    # Default-off pair for probe64. Together they give the listener a second
    # thing it can be asked for, and make asking for too much of it cost
    # something.
    #
    # ``portion_requests`` lets the caregiver hear a size word beside the need
    # word and grant that portion class instead of drawing one. The draw is
    # still consumed, so a world where nothing is ever asked for is
    # bit-identical to the old one. The granted surface still names the class,
    # so the portion stays learner-visible exactly as before.
    #
    # ``caregiver_store`` is the total portion mass the caregiver has for this
    # whole life; each grant spends its own size, and a grant it cannot afford
    # is not made. Zero means the old unlimited caregiver. Without it "large"
    # dominates and the size word is decoration; with it the organism has to
    # ask for as much as it will actually burn and no more -- which is a fact
    # about how fast *this* body burns, not about where it currently is.
    portion_requests: bool = False
    caregiver_store: float = 0.0
    # Default-off lever for probe65. The caregiver still grants once per
    # ``help_period``, but what it grants is the last request it heard at or
    # before ``help_delay`` ticks ago -- a listener that takes time to act
    # rather than one that reads the mind of the tick it acts on.
    #
    # This is the one change that makes a request have to be about the *future*.
    # A need word said now is answered when the body is somewhere else, and
    # where it will be depends on how fast this body burns. Two organisms in
    # identical states burning at different rates should name different needs,
    # which is exactly the divergence `md/archive/DIRECTION_2026-07-26.md` phase
    # C2 asks a future-tense report to show.
    #
    # It is deliberately *not* a change to ``help_period``: that constant is
    # frozen by the feasibility run and `docs/STATE.md` forbids retuning it. The
    # supply rate, the clock, and every portion stay exactly as they were; only
    # the lag between asking and being answered moves. At zero the caregiver
    # takes the same path it always has, byte for byte.
    help_delay: int = 0
    # Default-off lever for probe66. It changes nothing about what the caregiver
    # grants and everything about whether the convention *factors*.
    #
    # Off, a size word means the same thing whatever is being asked for: "more"
    # is large for food, for water and for energy. That is what makes probe64's
    # factored listener model able to utter a combination it has never uttered.
    # It is also *given* -- the designer chose to pool, and an organism that
    # always pools would look identical whether or not the world rewarded it.
    #
    # On, the caregiver draws its own word-to-size mapping per need at birth, so
    # "more" may be large for food and small for water. The convention is still
    # deterministic and still learnable within a life; what changes is that it no
    # longer factors, and an organism that pools regardless is now confidently
    # wrong. That is the moved ground truth probe61's preregistration made
    # binding for any discovery claim: a learner that always answers "it
    # factors" must be told apart from one that finds out.
    #
    # The draw comes from its own generator, so at the default every other stream
    # is bit-identical and the mapping is exactly ``SIZE_WORDS``.
    tangled_size_words: bool = False
    reveal_birth_needs: bool = True
    # Default-off treatment for the preregistered substrate test: every help
    # object uses the same voluntary CONSUME response.  The legacy ecology
    # keeps energy-specific REST uptake exactly unchanged.
    unified_uptake: bool = False
    # Bodily axes switched off, for the probe61 ground-truth dimension sweep.
    # A frozen axis is born full, never depletes, and is never shocked, so it
    # contributes no variance to anything the organism can feel.  Its random
    # draws are still consumed, so every unfrozen stream stays bit-identical to
    # the default ecology.  ``safety`` is nameable here even though it is not a
    # reportable need: it is a real depleting variable of this body.
    frozen_needs: tuple[str, ...] = ()
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
        if not 0.0 <= self.silent_shock_probability <= 1.0:
            raise ValueError("silent_shock_probability must be in [0, 1].")
        if not 0.0 <= self.metabolic_spread < 1.0:
            raise ValueError("metabolic_spread must be in [0, 1).")
        if not 0.0 <= self.uptake_spread < 1.0:
            raise ValueError("uptake_spread must be in [0, 1).")
        if not 0.0 <= self.interoception_probability <= 1.0:
            raise ValueError("interoception_probability must be in [0, 1].")
        if self.shock_size < 0.0:
            raise ValueError("shock_size must be nonnegative.")
        if self.caregiver_store < 0.0:
            raise ValueError("caregiver_store must be nonnegative.")
        if self.help_delay < 0:
            raise ValueError("help_delay must be nonnegative.")
        if not self.birth_levels:
            raise ValueError("birth_levels must be non-empty.")
        if any(not 0.0 < level <= 1.0 for level in self.birth_levels):
            raise ValueError("birth_levels must lie in (0, 1].")
        if self.listener_mode not in ("grounded", "scrambled", "mute"):
            raise ValueError(f"Unknown listener mode: {self.listener_mode}.")
        for need in self.frozen_needs:
            if need not in BODY_NEEDS:
                raise ValueError(f"Unknown bodily axis to freeze: {need}.")
        if len(set(self.frozen_needs)) != len(self.frozen_needs):
            raise ValueError("frozen_needs must not repeat an axis.")
        if set(self.frozen_needs) >= set(BODY_NEEDS):
            raise ValueError("At least one bodily axis must stay live.")

    def metabolism_of(self, need: str) -> float:
        if need in self.frozen_needs:
            return 0.0
        return float(getattr(self, f"{need}_metabolism"))

    def live_needs(self) -> tuple[str, ...]:
        """The bodily axes that actually vary in this ecology.

        This is the ground truth the probe61 dimension sweep is scored against.
        An axis born full whose metabolism is zero never moves, so nothing the
        organism can feel carries any information about it.
        """

        return tuple(
            need for need in BODY_NEEDS if self.metabolism_of(need) > 0.0
        )


class ReportGrid(IslandGrid):
    """An empty island with a randomized birth body and no free restoration."""

    def __init__(self, *, report: ReportConfig, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.report = report
        self.birth_needs = Needs()
        # This life's own metabolic rates, as multipliers on the species
        # constants. Set by ``ReportWorld.reset`` from a dedicated generator;
        # all ones is the species body and is what every organism had before
        # probe63.
        self.metabolic_scale: dict[str, float] = {need: 1.0 for need in BODY_NEEDS}
        # What the last uptake actually did: how much arrived and how much of it
        # fell off the top of a body that only holds 1.0. Audit-only, like
        # ``death_need``. Measured at the moment of the add, so the same tick's
        # metabolism is not netted off it; the resulting overestimate is under
        # one tick of burn and is identical for every tier.
        self.last_uptake: dict[str, float] | None = None

    def _make_default_world(self) -> list[WorldObject]:
        levels = self.report.birth_levels
        # The draws happen unconditionally so that freezing one axis leaves
        # every other axis's stream bit-identical to the default ecology.
        drawn = {
            need: self.rng.choice(levels) for need in REPORT_NEEDS
        }
        frozen = set(self.report.frozen_needs)
        self.needs = Needs(
            food=1.0 if "food" in frozen else drawn["food"],
            water=1.0 if "water" in frozen else drawn["water"],
            energy=1.0 if "energy" in frozen else drawn["energy"],
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
        self._record_uptake(
            {
                "food": obj.food_delta,
                "water": obj.water_delta,
                "energy": obj.energy_delta,
            }
        )
        self.needs = replace(
            self.needs,
            food=self.needs.food + obj.food_delta,
            water=self.needs.water + obj.water_delta,
            energy=self.needs.energy + obj.energy_delta,
            safety=self.needs.safety + obj.safety_delta,
        )
        self.objects = [item for item in self.objects if item is not obj]
        return f"consumed_{obj.kind}"

    def _record_uptake(self, deltas: dict[str, float]) -> None:
        delivered = 0.0
        overflow = 0.0
        for need, delta in deltas.items():
            if delta <= 0.0:
                continue
            delivered += delta
            overflow += max(0.0, getattr(self.needs, need) + delta - 1.0)
        if delivered <= 0.0:
            return
        self.last_uptake = {
            "delivered": delivered,
            "overflow": overflow,
            "useful": delivered - overflow,
        }

    def _rest(self) -> str:
        """Rest restores only what a granted shelter provides."""

        obj = self.object_at(self.agent_pos) or self.object_ahead()
        if obj is not None and obj.kind == "shelter":
            self._record_uptake({"energy": obj.energy_delta})
            self.needs = replace(
                self.needs, energy=self.needs.energy + obj.energy_delta
            )
            self.objects = [item for item in self.objects if item is not obj]
            return "rested_shelter"
        return "rested_unsheltered"

    def _apply_metabolism(self, action: Action) -> None:
        report = self.report
        frozen = set(report.frozen_needs)
        scale = self.metabolic_scale
        energy_cost = (
            report.move_energy_metabolism
            if action == Action.MOVE_FORWARD
            else report.energy_metabolism
        )
        if "energy" in frozen:
            energy_cost = 0.0
        # One multiplier per axis, so an individual's resting and moving energy
        # costs scale together. A body that burns energy fast burns it fast in
        # both regimes; the alternative would be two independent facts about the
        # self where the biology has one.
        self.needs = replace(
            self.needs,
            food=self.needs.food - report.metabolism_of("food") * scale["food"],
            water=self.needs.water - report.metabolism_of("water") * scale["water"],
            energy=self.needs.energy - energy_cost * scale["energy"],
            safety=self.needs.safety
            - report.metabolism_of("safety") * scale["safety"],
        )


def heard_size(tokens: tuple[int, ...] | None) -> str | None:
    """The portion size a listener hears in an utterance, or None.

    Same fixed rule as ``heard_need`` and decided at the same time: the first
    size word names the size. An utterance that names no size leaves the
    caregiver to choose for itself, exactly as it always has.
    """

    if tokens is None:
        return None
    for token in tokens:
        size = SIZE_WORD_IDS.get(int(token))
        if size is not None:
            return size
    return None


def heard_word(tokens: tuple[int, ...] | None) -> str | None:
    """The size *word* a listener hears, before any convention is applied.

    ``heard_size`` resolves a word to a size through the species convention and
    is kept for callers that want that. The caregiver carries the word instead,
    because probe66's convention can depend on which need is being asked about,
    and a size resolved at hearing time would have thrown that away.
    """

    if tokens is None:
        return None
    for token in tokens:
        if int(token) in SIZE_WORD_IDS:
            return VOCAB[int(token)]
    return None


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
        self._heard_size: str | None = None
        self._heard_tick = -1
        # Requests the caregiver has heard but not yet been slow enough to act
        # on. Only read when ``help_delay`` is positive; at zero the listener
        # takes the single-slot path it always took.
        self._heard_log: list[tuple[int, str, str | None]] = []
        self._last_utterance: tuple[int, ...] | None = None
        self._listener_rng = Random(0)
        self._help_rng = Random(0)
        self._shock_rng = Random(0)
        # Silence is a property of perception, not of dynamics, so it draws from
        # its own generator and never perturbs the shock stream itself.
        self._silence_rng = Random(0)
        # This life's own body, and the intermittent channel through which it
        # could be found out. Separate generators for the same reason as above:
        # at their defaults every other stream is bit-identical.
        self._body_rng = Random(0)
        self._intero_rng = Random(0)
        # This life's word-to-size convention, per need. At the default it is
        # exactly ``SIZE_WORDS`` for every need, so the caregiver behaves as it
        # always has.
        self._convention_rng = Random(0)
        self._size_convention: dict[str, dict[str, str]] = {}
        self._uptake_scale = {need: 1.0 for need in REPORT_NEEDS}
        self._shock_marker: WorldObject | None = None
        self._last_shock_silent = False
        # Read-only counterfactual audits may force exactly one forthcoming
        # portion size.  Training and ordinary evaluation never set this, so
        # their RNG path and ecology are unchanged.
        self._forced_next_help_large: bool | None = None
        self._forced_next_help_need: str | None = None
        self._grants = 0
        self._grants_by_need = {need: 0 for need in REPORT_NEEDS}
        self._silent_grants = 0
        # Probe64 bookkeeping. All audit-only, like ``death_need``: reported in
        # ``info`` and never in the packet, never in a belief, never in a
        # planner input.
        self._store_remaining = 0.0
        self._store_spent = 0.0
        self._refused_grants = 0
        self._granted_sizes = {size: 0 for size in PORTION_SIZES}

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
        self._silence_rng = Random(base + 8_800_019)
        self._body_rng = Random(base + 9_900_023)
        self._intero_rng = Random(base + 1_100_027)
        self._convention_rng = Random(base + 2_200_029)
        self._size_convention = self._draw_size_convention()
        self.grid.metabolic_scale = self._draw_metabolic_scale()
        self._uptake_scale = self._draw_uptake_scale()
        self._shock_marker = None
        self._forced_next_help_large = None
        self._forced_next_help_need = None
        self._heard_need = None
        self._heard_size = None
        self._heard_tick = -1
        self._heard_log = []
        self._last_utterance = None
        self._grants = 0
        self._grants_by_need = {need: 0 for need in REPORT_NEEDS}
        self._silent_grants = 0
        self._store_remaining = self.report.caregiver_store
        self._store_spent = 0.0
        self._refused_grants = 0
        self._granted_sizes = {size: 0 for size in PORTION_SIZES}
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
                self._heard_size = None
                self._heard_tick = self.grid.step_count
                self._log_request()
            return
        need = heard_need(self._last_utterance)
        if need is not None:
            self._heard_need = need
            # The size is bound to the request that carried it. A size word
            # said in an utterance that names no need is not a request for
            # anything, and a stale size never survives into the next request.
            self._heard_size = heard_word(self._last_utterance)
            self._heard_tick = self.grid.step_count
            self._log_request()

    def _log_request(self) -> None:
        """Remember a request a slow caregiver has not reached yet.

        Only consulted when ``help_delay`` is positive; at zero the list is
        cleared at every grant and never read, so the listener's behaviour is
        bit-identical to the caregiver of every earlier probe.
        """

        assert self._heard_need is not None
        self._heard_log.append(
            (self._heard_tick, self._heard_need, self._heard_size)
        )

    def step(
        self, action: Action | str
    ) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
        action = Action(action)
        needs_before = self.grid.needs
        self._clear_shock_marker()
        self.grid.last_uptake = None
        observation, reward, terminated, truncated, info = self.grid.step(action)

        self._last_shock_silent = False
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
        # A body reading, if one arrived this tick. Delivered here and never in
        # the packet, so what the policy sees is bit-identical at any rate: this
        # lever moves what a *self-model* can be corrected by, and nothing else.
        # Drawn unconditionally so the stream does not depend on the rate.
        interoception = (
            tuple(getattr(self.grid.needs, need) for need in REPORT_NEEDS)
            if self._intero_rng.random() < self.report.interoception_probability
            else None
        )
        info.update(
            {
                "interoception": interoception,
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
                # Audit-only, like death_need. The organism perceives a grant
                # through the surface that appears beside it, never through
                # these.
                "uptake": self.grid.last_uptake,
                "store_remaining": self._store_remaining,
                "store_spent": self._store_spent,
                "shock_need": shock_need,
                # Audit-only, like death_need: whether that shock left a marker.
                # Never reaches a belief, a listener, a planner, or a policy.
                "shock_silent": shock_need is not None and self._last_shock_silent,
                # Audit-only: which bodily axis actually ran out. Never reaches
                # a model, a belief, a listener, or a planner.
                "death_need": self.death_need() if terminated else None,
                "help_pending": self.pending_help_need(),
                "utterance_tokens": self._last_utterance,
                "spoke": self._last_utterance is not None
                and any(int(token) != self._pad_id for token in self._last_utterance),
            }
        )
        truncated = truncated or self.grid.step_count >= self.report.life_steps
        return packet, reward, terminated, truncated, info

    # -- this life's own body ------------------------------------------------

    def _draw_size_convention(self) -> dict[str, dict[str, str]]:
        """Which word means "large", per need, for this life.

        At the default every need shares the species convention and this is
        exactly ``SIZE_WORDS``. Tangled, each need draws its own orientation, so
        the convention is still one deterministic fact per life and simply does
        not factor. The draws happen unconditionally so that the generator's
        stream does not depend on the setting.
        """

        convention = {}
        for need in REPORT_NEEDS:
            flipped = self._convention_rng.random() < 0.5
            if self.report.tangled_size_words and flipped:
                convention[need] = {
                    word: ("small" if size == "large" else "large")
                    for word, size in SIZE_WORDS.items()
                }
            else:
                convention[need] = dict(SIZE_WORDS)
        return convention

    @property
    def size_convention(self) -> dict[str, dict[str, str]]:
        """What each word does, per need. Audit-only, like ``metabolic_scale``.

        Reading this is knowing the answer. It is what a ceiling listener is
        allowed and what any learner must instead infer from which surface
        appears beside it.
        """

        return {need: dict(words) for need, words in self._size_convention.items()}

    def _draw_metabolic_scale(self) -> dict[str, float]:
        """One multiplier per bodily axis, fixed for this life.

        At spread 0 this is exactly all ones -- the species body every organism
        in this ecology had before probe63 -- and the draws are still consumed,
        from a generator nothing else reads.
        """

        spread = self.report.metabolic_spread
        return {
            need: self._body_rng.uniform(1.0 - spread, 1.0 + spread)
            for need in BODY_NEEDS
        }

    def _draw_uptake_scale(self) -> dict[str, float]:
        """How much of a granted portion this body actually absorbs.

        Drawn after the metabolic scale and from the same dedicated generator,
        so a world that varies only metabolism keeps bit-identical rates whether
        or not absorption also varies.
        """

        spread = self.report.uptake_spread
        return {
            need: self._body_rng.uniform(1.0 - spread, 1.0 + spread)
            for need in REPORT_NEEDS
        }

    @property
    def metabolic_scale(self) -> dict[str, float]:
        """This life's true rates as multipliers. Audit-only, like death_need.

        Reading this is knowing the answer. It is what the ceiling tier is
        allowed and what any learner must instead infer.
        """

        return dict(self.grid.metabolic_scale)

    @property
    def uptake_scale(self) -> dict[str, float]:
        """This life's absorption, as multipliers. Audit-only, like death_need."""

        return dict(self._uptake_scale)

    def individual_report(self) -> ReportConfig:
        """This life's ecology constants with its own metabolism substituted.

        Lets an exact-dynamics filter be built for *this* body rather than for
        the species, which is the only difference between the ``population`` and
        ``individual`` tiers of the probe63 survey.
        """

        scale = self.grid.metabolic_scale
        return replace(
            self.report,
            food_metabolism=self.report.food_metabolism * scale["food"],
            water_metabolism=self.report.water_metabolism * scale["water"],
            energy_metabolism=self.report.energy_metabolism * scale["energy"],
            move_energy_metabolism=(
                self.report.move_energy_metabolism * scale["energy"]
            ),
            safety_metabolism=self.report.safety_metabolism * scale["safety"],
        )

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
        # The draw is consumed either way, so freezing one axis leaves the
        # shock stream of the others bit-identical.
        if need in report.frozen_needs:
            return None
        self.grid.needs = replace(
            self.grid.needs,
            **{need: getattr(self.grid.needs, need) - report.shock_size},
        ).clipped()
        # Drawn for every landed shock regardless of the rate, so the silent set
        # at a lower rate is a subset of the silent set at a higher one and the
        # sweep compares nested observability over one fixed shock history.
        silent = self._silence_rng.random() < report.silent_shock_probability
        self._last_shock_silent = silent
        marker_pos = None if silent else self._marker_position()
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

    def force_next_help_portion(self, *, large: bool) -> None:
        """Force one perceptible help size for a paired causal audit.

        The intervention is consumed by the next non-silent grant.  It changes
        both the lived bodily consequence and the learner-visible surface that
        identifies the portion, so the resulting state remains inferable from
        experience.  No training path calls this method.
        """

        self._forced_next_help_large = bool(large)

    def force_next_help_need(self, need: str | None) -> None:
        """Force which resource the next grant delivers, for a paired audit.

        The intervention replaces what the listener heard, so the grant and its
        perceptible surface stay consistent with each other. No training path
        calls this method.
        """

        if need is not None and need not in REPORT_NEEDS:
            raise ValueError(f"Unknown need to force: {need}.")
        self._forced_next_help_need = need

    def _take_request(self) -> tuple[str | None, str | None, int]:
        """The request this grant answers, and the tick it was made on.

        With no delay the caregiver answers the last thing it heard and forgets
        everything else -- the single slot every earlier probe used, reproduced
        here exactly. With a delay it answers the most recent request that is
        already at least ``help_delay`` ticks old, and keeps anything newer for
        the grants still to come. Requests older than the one it acts on are
        dropped, so nothing is ever answered twice and a request cannot be
        hoarded past its turn.
        """

        if self.report.help_delay <= 0:
            need, size, tick = self._heard_need, self._heard_size, self._heard_tick
            self._heard_need = None
            self._heard_size = None
            self._heard_tick = -1
            self._heard_log.clear()
            return need, size, tick

        deadline = self.grid.step_count - self.report.help_delay
        position = -1
        for index, entry in enumerate(self._heard_log):
            if entry[0] > deadline:
                break
            position = index
        if position < 0:
            # Nothing said long enough ago to have been acted on yet. Early in
            # a life this is simply the lag, and it costs every organism the
            # same grants.
            return None, None, -1
        tick, need, size = self._heard_log[position]
        del self._heard_log[: position + 1]
        return need, size, tick

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
        need, size, heard_tick = self._take_request()
        if self._forced_next_help_need is not None:
            need = self._forced_next_help_need
            self._forced_next_help_need = None
        # An unanswered grant still has a moment that decided it: the last thing
        # the organism said before help came, which happened not to be a
        # request. That moment is what gets the credit for the silence.
        source_tick = (
            heard_tick if need is not None else self.grid.step_count - 1
        )
        self._grants += 1
        if need is None:
            self._silent_grants += 1
            return None, None, source_tick
        # Drawn unconditionally, so a world in which the organism never asks
        # for a size keeps a bit-identical help stream at any setting of
        # ``portion_requests``.
        drawn_large = self._help_rng.random() < self.report.large_portion_probability
        large = drawn_large
        if self.report.portion_requests and size is not None:
            # ``size`` is the word that was said; what it does is this life's
            # convention for *this* need, which at the default is the species
            # one for every need.
            large = self._size_convention[need][size] == "large"
        if self._forced_next_help_large is not None:
            large = self._forced_next_help_large
            self._forced_next_help_large = None

        # A caregiver with a basket rather than a spring. It cannot give what it
        # no longer has, and a large portion empties the basket three times as
        # fast as a small one -- which is the whole reason a size word is worth
        # saying rather than merely sayable.
        cost = self.report.portion_large if large else self.report.portion_small
        if self.report.caregiver_store > 0.0:
            if cost > self._store_remaining + 1e-12:
                self._refused_grants += 1
                return None, None, source_tick
            self._store_remaining -= cost
        # Counted whether or not there is a basket to count it against, so that
        # what a grant costs is a comparable number in both ecologies.
        self._store_spent += cost
        # The surface still names the portion class the caregiver granted; how
        # much of it this body takes up is private to this body.
        portion = (
            self.report.portion_large if large else self.report.portion_small
        ) * self._uptake_scale[need]
        surface = HELP_SURFACES[(need, large)]
        pos = self.grid.agent_pos
        if need == "food":
            obj = WorldObject(surface, "food", pos, food_delta=portion, consumable=True)
        elif need == "water":
            obj = WorldObject(surface, "water", pos, water_delta=portion, consumable=True)
        else:
            obj = WorldObject(
                surface,
                "shelter",
                pos,
                energy_delta=portion,
                consumable=self.report.unified_uptake,
            )
        self.grid.objects = [*self.grid.objects, obj]
        self._grants_by_need[need] += 1
        self._granted_sizes["large" if large else "small"] += 1
        return need, large, source_tick

    # -- read-only accessors used by audits ---------------------------------

    def death_need(self, needs: Needs | None = None) -> str | None:
        """Which bodily axis ran out, for lesion-specificity audits only."""

        current = self.grid.needs if needs is None else needs
        exhausted = [
            need for need in BODY_NEEDS if getattr(current, need) <= 0.0
        ]
        if not exhausted:
            return None
        return min(exhausted, key=lambda need: getattr(current, need))

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
        counts["refused"] = self._refused_grants
        for size in PORTION_SIZES:
            counts[f"granted_{size}"] = self._granted_sizes[size]
        return counts

    @property
    def store_remaining(self) -> float:
        """What the caregiver has left. Audit-only; nothing perceives it."""

        return self._store_remaining

    @property
    def store_spent(self) -> float:
        return self._store_spent

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
