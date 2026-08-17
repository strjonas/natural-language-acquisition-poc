"""Probe64: something about itself that no amount of looking at itself would tell it.

Preregistered in `docs/decisions/2026-08-14-portion-request-preregistration.md`.
Gates are locked there and are not adjusted here.

Probe63 gave the organism an individual body and a way to find out what it is.
It closed the repository's oldest obstacle and left the newest one wide open:
the organism now knows something no listener can hear. Its whole public
protocol is three words naming which need is lowest, because three things are
all the caregiver can grant. `docs/DIRECTION_2026-07-26.md` calls the missing
property *productive under novel demand* and says plainly what unlocks it --
make the world require a distinction the current protocol cannot express.

So the caregiver learns to do one more thing. It can be asked for a **large or
a small** portion, and it carries a **basket rather than a spring**: a finite
store for the whole life, from which a large portion takes three times what a
small one does. That single change turns portion size from something that
happens to the organism into something it has to decide, and the decision has a
property nothing else in this repository has had:

    The right size is set by how fast *this body* burns, and knowing exactly
    where the body is right now does not tell you.

The reason is arithmetic. One grant arrives per help period and three needs
compete for it, so a portion has to carry its need for about ``help_period * 3``
ticks before that need is served again. What must be covered is
``rate * interval`` -- a fact about the individual's *rate*, invisible in any
single reading of its state. Two organisms whose bodies are in identical states,
one burning fast and one burning slow, should ask for different portions.

That gives this probe the control the claim needs, and it is one the repository
has never been able to run before:

    state_oracle   reads the true body every tick and believes the species
                   rates. Perfect knowledge of where it is; no knowledge of
                   what it is.

If a self-model does not beat *that*, then everything a self-model buys here
was state estimation wearing a self-model's clothes, and the mechanism closes.

The second half of the probe is about the word rather than the body. Nothing
tells the organism which of the two size words the caregiver treats as "large".
It has to find that out from what appears beside it, and it has to be able to
say a combination it has never said before -- which a listener model factored
into *need* and *size* can do and a table of six cells cannot.

Nothing here is trained by reward. The self-model is probe63's, the motor policy
is probe52's frozen parent, and the size rule is one line of arithmetic shared
bit-identically by every tier. What differs between tiers is only what each of
them believes it is.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import (
    HELP_SURFACE_LARGE,
    HELP_SURFACE_NEEDS,
    NEED_TO_REPORT_WORD,
    PORTION_SIZES,
    REPORT_NEEDS,
    SIZE_TO_WORD,
    SIZE_WORDS,
)
from homesocial.island.world import SURFACES
from homesocial.organism.individual_self import (
    FilterTier,
    OracleTier,
    _true_body,
    metabolic_ticks,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import (
    FIDELITY_WARMUP,
    _ObservableBodyFilter,
    make_report_world,
)
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.self_calibration import (
    OracleCalibration,
    RecursiveSelfCalibration,
)
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

# Ordered worst-informed to best-informed about *itself*. ``state_oracle`` sits
# deliberately in the middle: it is the best-informed tier about its state and
# among the worst about its nature.
TIERS = ("population", "snap", "state_oracle", "recursive", "individual", "oracle")
# Tiers whose beliefs about their own constants are given rather than found.
ORACLE_TIERS = ("individual", "oracle")
# Speaker-side controls on the size channel. Each keeps the need word, the
# history, and the caregiver identical and removes only the information in the
# size word.
SPEAKER_CONTROLS = ("always_large", "always_small", "random_size", "no_size")

# Disjoint from every band already in use, including probe63's 930,000,000 with
# a 2,000,000 stride over five seeds. The survey band is disjoint from the
# treatment band as well: probe63 made it binding to measure the ceiling at the
# operating point a gate will be scored at, and a threshold read off a life the
# gate then scores is a threshold fitted to its own answer.
SURVEY_SEED_BASE = 950_000_000
# 960,000,000 was spent during construction on exploratory runs whose numbers
# are recorded in the ceiling survey document. Those lives are therefore burnt
# for gate purposes, and the treatment band starts above them.
CONSTRUCTION_SEED_BASE = 960_000_000
SEED_BASE = 970_000_000
SEED_STRIDE = 2_000_000
SEEDS = 5

INTEROCEPTION = 0.03
TREATMENT_WORLD = "metabolic"
WORLDS = {
    "metabolic": {"metabolic_spread": 0.60, "uptake_spread": 0.00},
    "both": {"metabolic_spread": 0.60, "uptake_spread": 0.60},
    "null": {"metabolic_spread": 0.00, "uptake_spread": 0.00},
}

# Numerical guards, not tuning. ``m`` is already clipped inside the calibrator;
# these keep a diverged self-model from handing the planner a nonsense input
# under the shuffled-readings control instead of producing a NaN.
MIN_BELIEVED_RATE = 1e-6
UPTAKE_BOUNDS = (0.05, 5.0)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


# -- the rule every tier speaks by --------------------------------------------


def service_interval(report) -> int:
    """Ticks between one need being served and its being served again.

    One grant arrives per ``help_period`` and three needs compete for it, so a
    portion has to carry its need roughly three help periods. Fixed a priori
    from the ecology's own constants, identical for every tier, never swept: it
    is a property of the caregiver's clock, not a parameter of any organism.
    """

    return int(report.help_period) * len(REPORT_NEEDS)


def choose_size(
    *,
    believed_level: float,
    believed_rate: float,
    believed_uptake: float,
    report,
    interval: int,
) -> str:
    """The smallest portion that covers what this body burns before it is next served.

    Two quantities decide it and they come from different places. ``required``
    is ``rate * interval``: what will be spent before this need is served
    again, a fact about the individual's metabolism that is invisible in any
    single reading of its state. ``headroom`` is ``1 - level``: how much of a
    portion a body this full can actually hold, a fact about the state that says
    nothing about the rate. A portion is enough when what lands covers what is
    burnt.

    Asking large when small would do is not free -- it empties the caregiver's
    basket three times as fast -- so the rule takes the cheapest size that
    suffices. When nothing suffices it takes the one that actually delivers
    most, and a body with no room left is delivered exactly the same amount by
    either, so it asks for the cheap one: paying triple for a portion that falls
    off the top is the one move a rationed organism can never afford.
    """

    required = max(0.0, believed_rate) * interval
    headroom = max(0.0, 1.0 - believed_level)
    delivered = {
        size: min(
            (report.portion_large if size == "large" else report.portion_small)
            * believed_uptake,
            headroom,
        )
        for size in PORTION_SIZES
    }
    for size in PORTION_SIZES:
        if delivered[size] >= required:
            return size
    best = max(delivered.values())
    for size in PORTION_SIZES:
        if delivered[size] >= best - 1e-12:
            return size
    return PORTION_SIZES[-1]


def species_rate(report, need: str, move_fraction: float) -> float:
    """Per-tick metabolism the species constants predict for one need."""

    if need == "energy":
        return float(
            report.move_energy_metabolism * move_fraction
            + report.energy_metabolism * (1.0 - move_fraction)
        )
    return float(getattr(report, f"{need}_metabolism"))


class MoveFraction:
    """How much of this organism's life is spent moving, from its own history.

    Energy is the one axis whose per-tick cost depends on what the body is
    doing, so a rate belief for it needs a mixing weight. Everything this reads
    -- the organism's own selected action and the routed distance of its own
    option -- is already learner-visible, and every tier shares one instance
    over one shared history, so it can never be the thing that separates two
    tiers.
    """

    def __init__(self, organism: OrganismConfig, report) -> None:
        self._reference = _ObservableBodyFilter(replace(organism, report=report))
        # One help period of the species mixture, so the first few actions do
        # not swing the estimate. Its value at zero evidence is the species
        # expectation and is identical for every tier.
        self._moves = 0.5 * report.help_period
        self._ticks = 1.0 * report.help_period

    def reset(self, packet) -> None:
        self._reference.reset(packet)

    def observe(self, before, action_index: int, after) -> None:
        duration, moves, _ = metabolic_ticks(
            self._reference, before, action_index, after
        )
        self._moves += moves
        self._ticks += duration
        self._reference.update(before, action_index, after)

    @property
    def fraction(self) -> float:
        return float(self._moves / self._ticks)


# -- what each tier believes it is --------------------------------------------


class SelfModelTier:
    """A body belief plus a belief about the constants that produced it.

    Probe63's tiers all answered one question -- *where am I* -- and that is all
    its endpoints needed. A size request needs a second answer, *what am I*, and
    the two come apart: ``state_oracle`` has the first exactly right and the
    second wrong, and it is the control this probe turns on.
    """

    def __init__(self, name: str, organism: OrganismConfig, report) -> None:
        if name not in TIERS:
            raise ValueError(f"Unknown tier: {name}.")
        self.name = name
        self._report = report
        if name == "population":
            self._body = FilterTier(organism, report, use_individual=False, snap=False)
        elif name == "snap":
            self._body = FilterTier(organism, report, use_individual=False, snap=True)
        elif name in ("state_oracle", "oracle"):
            self._body = OracleTier()
        elif name == "recursive":
            self._body = RecursiveSelfCalibration(organism, report)
        else:
            self._body = OracleCalibration(organism, report)
        self._scale = {need: 1.0 for need in REPORT_NEEDS}
        self._uptake = {need: 1.0 for need in REPORT_NEEDS}

    def reset(self, packet, world) -> None:
        self._body.reset(packet, world)
        if self.name in ORACLE_TIERS:
            # Given, not found. These are the ceiling and they are allowed the
            # answer for exactly the reason ``oracle`` is allowed the body.
            self._scale = {n: world.metabolic_scale[n] for n in REPORT_NEEDS}
            self._uptake = {n: world.uptake_scale[n] for n in REPORT_NEEDS}
        else:
            self._scale = {need: 1.0 for need in REPORT_NEEDS}
            self._uptake = {need: 1.0 for need in REPORT_NEEDS}

    def update(self, before, action_index, after, world, reading) -> None:
        self._body.update(before, action_index, after, world, reading)
        if self.name == "recursive":
            self._scale = self._body.recovered_scale()
            self._uptake = {
                need: float(np.clip(value, *UPTAKE_BOUNDS))
                for need, value in self._body.recovered_uptake().items()
            }

    def point(self) -> np.ndarray:
        return self._body.point()

    def believed_scale(self) -> dict[str, float]:
        return dict(self._scale)

    def believed_uptake(self) -> dict[str, float]:
        """How much good this tier thinks a granted portion does it.

        Read-only. Probe64 uses it inside ``size_for``; probe65 needs the same
        belief to work out what its own outstanding requests will be worth when
        they land.
        """

        return dict(self._uptake)

    def believed_rate(self, need: str, move_fraction: float) -> float:
        return max(
            MIN_BELIEVED_RATE,
            species_rate(self._report, need, move_fraction) * self._scale[need],
        )

    def size_for(self, need: str, move_fraction: float) -> str:
        index = REPORT_NEEDS.index(need)
        return choose_size(
            believed_level=float(self.point()[index]),
            believed_rate=self.believed_rate(need, move_fraction),
            believed_uptake=self._uptake[need],
            report=self._report,
            interval=service_interval(self._report),
        )


def true_size(world, need: str, move_fraction: float, report) -> str:
    """What the size rule returns when every one of its inputs is the truth.

    This is the target every tier is scored against, and it is exactly what the
    ``oracle`` tier computes -- so ``oracle`` scores 1.000 by construction and is
    carried as the anchor that says so.
    """

    index = REPORT_NEEDS.index(need)
    return choose_size(
        believed_level=float(_true_body(world)[index]),
        believed_rate=species_rate(report, need, move_fraction)
        * world.metabolic_scale[need],
        believed_uptake=world.uptake_scale[need],
        report=report,
        interval=service_interval(report),
    )


# -- what the words do ---------------------------------------------------------


def visible_help(packet) -> tuple[str, bool] | None:
    """The granted portion the organism can currently see, from surface identity.

    Portion class has always been legible in this ecology as which surface
    appears -- ``roots`` rather than ``berry``. This reads the same public
    packet field the observable body filter reads and nothing else. It does not
    touch ``granted_need``, ``granted_large``, or any other simulator metadata,
    which `docs/STATE.md` holds to be barred from listener learning.
    """

    for _, _, surface_index in packet.visible:
        surface = SURFACES[surface_index]
        need = HELP_SURFACE_NEEDS.get(surface)
        if need is not None:
            return need, HELP_SURFACE_LARGE[surface]
    return None


class ListenerModel:
    """Which of the two size words makes the caregiver grant the large class.

    Nothing tells the organism. What it has is what anyone watching would have:
    it knows what it said, and it can see which surface appeared beside it. From
    those two it learns the convention.

    ``factored`` holds one belief per size word, pooled over needs: *this word
    means large, whatever I am asking for*. ``tabular`` holds one belief per
    (need word, size word) cell from exactly the same observations. They differ
    on precisely one thing -- whether a combination never uttered can be uttered
    correctly the first time it is needed -- which is the whole of property 4 in
    this ecology.
    """

    def __init__(self, mode: str = "factored") -> None:
        if mode not in ("factored", "tabular"):
            raise ValueError(f"Unknown listener model: {mode}.")
        self.mode = mode
        self._large: dict[tuple[str, ...], float] = {}
        self._count: dict[tuple[str, ...], float] = {}

    def _key(self, need: str, word: str) -> tuple[str, ...]:
        return (word,) if self.mode == "factored" else (need, word)

    def observe(self, need: str, word: str, granted_large: bool) -> None:
        key = self._key(need, word)
        self._large[key] = self._large.get(key, 0.0) + float(granted_large)
        self._count[key] = self._count.get(key, 0.0) + 1.0

    def evidence(self, need: str, word: str) -> float:
        return self._count.get(self._key(need, word), 0.0)

    def probability_large(self, need: str, word: str) -> float:
        count = self.evidence(need, word)
        if count <= 0.0:
            # No evidence is not weak evidence. An unseen combination is a coin
            # flip, and a speaker that has to guess gets it right half the time.
            return 0.5
        return self._large[self._key(need, word)] / count

    def knows(self, need: str) -> bool:
        return any(self.evidence(need, word) > 0.0 for word in SIZE_WORDS)

    def word_for(self, need: str, size: str) -> str:
        """The word this model expects to produce ``size`` of ``need``.

        Untried words are tried first: with a deterministic caregiver one grant
        settles what a word does, so finding out costs two grants and the
        organism pays it once. The tie-break when nothing is known is vocabulary
        order, a fixed convention rather than a coin two arms could flip
        differently.
        """

        words = tuple(SIZE_WORDS)
        unseen = [word for word in words if self.evidence(need, word) <= 0.0]
        if unseen:
            return unseen[0]
        scores = {word: self.probability_large(need, word) for word in words}
        if size == "large":
            return max(words, key=lambda word: scores[word])
        return min(words, key=lambda word: scores[word])


class GrantWatcher:
    """Notices that a grant arrived, without reading that it did.

    A help object is placed at the agent's feet at each grant boundary and
    spoils at the next one, so "a help surface is visible and the help clock has
    moved on since the last one I saw" identifies a new grant from perception
    and the caregiver's public period alone.
    """

    def __init__(self, help_period: int) -> None:
        self._period = int(help_period)
        self._last_window = -1

    def reset(self) -> None:
        self._last_window = -1

    def poll(self, packet) -> tuple[str, bool] | None:
        seen = visible_help(packet)
        if seen is None:
            return None
        window = int(packet.step_count) // self._period
        if window == self._last_window:
            return None
        self._last_window = window
        return seen


# -- the speaker ---------------------------------------------------------------


class Speaker:
    """One organism's whole public protocol: a need word and a size word.

    The need word is probe63's, unchanged: the argmin of whatever this tier
    believes its body to be. The size word is new, and it is where the self
    model has to pay rent -- because the caregiver's basket punishes an organism
    that asks for more than it will burn, and its clock punishes one that asks
    for less.
    """

    def __init__(
        self,
        tier: SelfModelTier,
        *,
        listener_mode: str = "factored",
        control: str | None = None,
        rng: Random | None = None,
    ) -> None:
        if control is not None and control not in SPEAKER_CONTROLS:
            raise ValueError(f"Unknown speaker control: {control}.")
        self.tier = tier
        self.control = control
        self.listener = ListenerModel(listener_mode)
        self._rng = rng or Random(0)

    def intended_size(self, need: str, move_fraction: float) -> str:
        if self.control == "always_large":
            return "large"
        if self.control == "always_small":
            return "small"
        if self.control == "random_size":
            return self._rng.choice(PORTION_SIZES)
        return self.tier.size_for(need, move_fraction)

    def utterance(self, need: str, move_fraction: float) -> tuple[int, ...]:
        need_id = TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]]
        if self.control == "no_size":
            # Says only what the old organism could say, so the caregiver draws
            # a portion for itself exactly as it always did.
            return (need_id, PAD_ID)
        size = self.intended_size(need, move_fraction)
        return (need_id, TOKEN_TO_ID[self.listener.word_for(need, size)])


# -- one condition -------------------------------------------------------------


def _world_kwargs(world_name: str, *, store: float, interoception: float) -> dict:
    return dict(
        WORLDS[world_name],
        interoception_probability=interoception,
        portion_requests=True,
        caregiver_store=store,
    )


def _donor_bodies(organism: OrganismConfig, model, *, seed: int, kwargs):
    """One other life's body trajectory, for the shuffled-readings control."""

    world = make_report_world(organism, seed=seed, **kwargs)
    packet = world.reset(seed)
    hidden = None
    rng = Random(seed + 59_000_003)
    bodies = []
    while True:
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        world.hear(
            (
                TOKEN_TO_ID[NEED_TO_REPORT_WORD["food"]],
                TOKEN_TO_ID[SIZE_TO_WORD["small"]],
            )
        )
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            action,
            consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        bodies.append(_true_body(world))
        if terminated or truncated:
            break
    return bodies


def run_open_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    world_name: str,
    store: float,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
    shuffle_readings: bool = False,
    driver: str = "population",
) -> dict[str, object]:
    """Every tier scored on one shared history and one shared choice of need.

    The driver names the need, every tier is asked what size it would request
    *for that need*, and all of them are scored against what the rule returns
    under the truth. The need channel is therefore bit-identical across tiers
    and the only thing that varies is the belief about the self that sets the
    size. A difference here cannot be a difference of trajectory, of policy, or
    of which need was asked about.
    """

    kwargs = _world_kwargs(world_name, store=store, interoception=interoception)
    report = replace(organism.report, **kwargs)

    hits = {tier: 0 for tier in TIERS}
    large_calls = {tier: 0 for tier in TIERS}
    body_error = {tier: 0.0 for tier in TIERS}
    ticks = 0
    discriminating = 0
    readings = 0
    scale_error = 0.0
    lives_scored = 0

    for life in range(lives):
        seed = seed_base + life
        donor = (
            _donor_bodies(organism, model, seed=seed + 777_777, kwargs=kwargs)
            if shuffle_readings
            else None
        )
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tiers = {name: SelfModelTier(name, organism, report) for name in TIERS}
        for tier in tiers.values():
            tier.reset(packet, world)
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        speaker = Speaker(tiers[driver])
        watcher = GrantWatcher(report.help_period)
        hidden = None
        rng = Random(seed + 59_000_003)
        reading_index = 0

        while True:
            need = REPORT_NEEDS[int(np.argmin(tiers[driver].point()))]
            fraction = moves.fraction
            if packet.step_count >= FIDELITY_WARMUP:
                ticks += 1
                target = true_size(world, need, fraction, report)
                discriminating += int(
                    tiers["population"].size_for(need, fraction) != target
                )
                truth = _true_body(world)
                for name, tier in tiers.items():
                    chosen = tier.size_for(need, fraction)
                    hits[name] += int(chosen == target)
                    large_calls[name] += int(chosen == "large")
                    body_error[name] += float(np.abs(tier.point() - truth).mean())

            tokens = speaker.utterance(need, fraction)
            word = _size_word_of(tokens)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear(tokens)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            granted = watcher.poll(packet)
            if granted is not None and word is not None:
                speaker.listener.observe(granted[0], word, granted[1])
            if terminated or truncated:
                break
            reading = info.get("interoception")
            if reading is not None:
                readings += 1
                if donor is not None:
                    # Matched in count and timing; only the body is another
                    # organism's.
                    reading = tuple(donor[reading_index % len(donor)])
                reading_index += 1
            moves.observe(before, action, packet)
            for tier in tiers.values():
                tier.update(before, action, packet, world, reading)

        recovered = tiers["recursive"].believed_scale()
        actual = world.metabolic_scale
        scale_error += float(
            np.mean([abs(recovered[n] - actual[n]) for n in REPORT_NEEDS])
        )
        lives_scored += 1

    denominator = max(1, ticks)
    return {
        "world": world_name,
        "caregiver_store": store,
        "shuffled_readings": shuffle_readings,
        "interoception_probability": interoception,
        "lives": lives,
        "seed_base": seed_base,
        "scored_ticks": ticks,
        "readings_per_life": readings / max(1, lives),
        # How often the question is worth asking at all: the share of scored
        # ticks where believing you are a typical member of your species gives a
        # different answer from the truth. No tier can beat ``population`` by
        # more than this, so it is the headroom every gate is measured inside.
        "discriminating_share": discriminating / denominator,
        "size_accuracy": {tier: hits[tier] / denominator for tier in TIERS},
        "large_share": {tier: large_calls[tier] / denominator for tier in TIERS},
        "body_error": {tier: body_error[tier] / denominator for tier in TIERS},
        "rate_error": scale_error / max(1, lives_scored),
        "prior_rate_error": WORLDS[world_name]["metabolic_spread"] / 2.0,
    }


def _size_word_of(tokens: tuple[int, ...]) -> str | None:
    for token in tokens:
        word = _ID_TO_SIZE_WORD.get(int(token))
        if word is not None:
            return word
    return None


_ID_TO_SIZE_WORD = {TOKEN_TO_ID[word]: word for word in SIZE_WORDS}


def run_closed_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    tier_name: str,
    world_name: str,
    store: float,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
    control: str | None = None,
    listener_mode: str = "factored",
    need_driver: str | None = None,
) -> dict[str, object]:
    """One tier speaking for itself, so it lives the life its own words bought.

    This is where the caregiver's basket actually bites. The endpoints are
    physical rather than rule-relative -- how much of the store was spent, how
    much of what arrived fell off the top of a full body, how often the
    caregiver had nothing left to give, and whether the organism was alive at
    the end -- so none of them presupposes that the size rule is the right one.

    ``need_driver`` names a *second* tier that chooses the need word while the
    tier under test still chooses the size. It is an instrument, not an
    organism: probes 59 and 60 established that survival here is dominated by
    which need is asked for, so leaving both channels free measures the need
    channel and calls it the size channel. Pinning the need at one tier for
    every arm is the only way a behavioural number can be about the size word.
    """

    kwargs = _world_kwargs(world_name, store=store, interoception=interoception)
    report = replace(organism.report, **kwargs)

    survived = 0
    truthful = 0
    said = 0
    spent = 0.0
    delivered = 0.0
    overflow = 0.0
    refused = 0
    grants = 0
    large_grants = 0
    steps = 0

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tier = SelfModelTier(tier_name, organism, report)
        tier.reset(packet, world)
        namer = tier
        if need_driver is not None and need_driver != tier_name:
            namer = SelfModelTier(need_driver, organism, report)
            namer.reset(packet, world)
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        speaker = Speaker(
            tier,
            listener_mode=listener_mode,
            control=control,
            rng=Random(seed + 31_000_007),
        )
        watcher = GrantWatcher(report.help_period)
        hidden = None
        rng = Random(seed + 59_000_003)

        while True:
            need = REPORT_NEEDS[int(np.argmin(namer.point()))]
            fraction = moves.fraction
            if packet.step_count >= FIDELITY_WARMUP:
                said += 1
                truthful += int(need == world.lowest_need())
            tokens = speaker.utterance(need, fraction)
            word = _size_word_of(tokens)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear(tokens)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            steps += int(info["duration"])
            granted = watcher.poll(packet)
            if granted is not None and word is not None:
                speaker.listener.observe(granted[0], word, granted[1])
            uptake = info.get("uptake")
            if uptake is not None:
                delivered += float(uptake["delivered"])
                overflow += float(uptake["overflow"])
            if terminated or truncated:
                survived += int(not terminated)
                break
            moves.observe(before, action, packet)
            reading = info.get("interoception")
            tier.update(before, action, packet, world, reading)
            if namer is not tier:
                namer.update(before, action, packet, world, reading)

        counts = world.grant_counts
        spent += world.store_spent
        refused += int(counts["refused"])
        grants += int(counts["total"]) - int(counts["none"]) - int(counts["refused"])
        large_grants += int(counts["granted_large"])

    return {
        "tier": tier_name,
        "control": control,
        "listener_mode": listener_mode,
        "need_driver": need_driver,
        "world": world_name,
        "caregiver_store": store,
        "lives": lives,
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
        "mean_life_steps": steps / lives,
        "store_spent": spent / lives,
        "delivered": delivered / lives,
        "overflow": overflow / lives,
        # What the caregiver's basket actually bought: the share of everything
        # it handed over that a body with room for it took up, and the same
        # quantity per unit of basket. The second is the one a rationed
        # caregiver would care about, and unlike survival it is not flattened by
        # homeostasis -- but unlike survival it also rewards simple frugality,
        # so neither is read alone.
        "uptake_efficiency": (delivered - overflow) / max(1e-9, delivered),
        "useful_per_store": (delivered - overflow) / max(1e-9, spent),
        "refused_grants": refused / lives,
        "grants": grants / lives,
        "large_share_granted": large_grants / max(1, grants),
    }


# -- saying what has never been said -------------------------------------------


def composition_holdout(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    world_name: str,
    store: float,
    lives: int,
    seed_base: int,
    held_out_need: str = "energy",
    interoception: float = INTEROCEPTION,
) -> dict[str, object]:
    """Can it ask for a portion it has never once asked for?

    Through the whole life the speaker never says a size word while asking about
    ``held_out_need``. It asks for energy the way the organism of every earlier
    probe asked for it -- by naming it and taking whatever comes -- so the
    caregiver never once demonstrates what a size word does to an energy grant.
    Everything else runs unchanged, and both listener models watch exactly the
    same transcript.

    At the end each model is asked the two questions it was never shown the
    answer to: which word gets a large portion of energy, and which gets a
    small one. The factored model has both answers from the needs it did ask
    about. The table has two empty cells and nothing to fill them from, so it
    can only guess -- and because it is asked about both sizes, guessing scores
    exactly one half whatever its tie-break happens to be. That symmetry is the
    reason both sizes are scored: with two words in play, a model that always
    tries the word it has not tried would otherwise identify the answer by
    elimination and look like it had generalized.
    """

    kwargs = _world_kwargs(world_name, store=store, interoception=interoception)
    report = replace(organism.report, **kwargs)

    scores = {mode: 0 for mode in ("factored", "tabular")}
    informed = {mode: 0 for mode in ("factored", "tabular")}
    wanted = 0
    scored_lives = 0
    questions = 0

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **kwargs)
        packet = world.reset(seed)
        tier = SelfModelTier("recursive", organism, report)
        tier.reset(packet, world)
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        models = {mode: ListenerModel(mode) for mode in scores}
        watcher = GrantWatcher(report.help_period)
        hidden = None
        rng = Random(seed + 59_000_003)
        wanted_here = 0

        while True:
            need = REPORT_NEEDS[int(np.argmin(tier.point()))]
            fraction = moves.fraction
            size = tier.size_for(need, fraction)
            # One speaker, so both models see one identical stream of evidence.
            # The word is chosen by the factored model; the table is a passive
            # observer of the same transcript, which is the strongest form of
            # this control -- it cannot be starved by a policy difference.
            word = None if need == held_out_need else models["factored"].word_for(
                need, size
            )
            if word is None:
                wanted_here += 1
            tokens = (
                TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]],
                PAD_ID if word is None else TOKEN_TO_ID[word],
            )
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear(tokens)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            granted = watcher.poll(packet)
            if granted is not None and word is not None:
                for observer in models.values():
                    observer.observe(granted[0], word, granted[1])
            if terminated or truncated:
                break
            moves.observe(before, action, packet)
            tier.update(before, action, packet, world, info.get("interoception"))

        wanted += wanted_here
        scored_lives += 1
        for mode, observer in models.items():
            for size in PORTION_SIZES:
                scores[mode] += int(
                    observer.word_for(held_out_need, size) == SIZE_TO_WORD[size]
                )
            informed[mode] += int(observer.knows(held_out_need))
        questions += len(PORTION_SIZES)

    return {
        "world": world_name,
        "caregiver_store": store,
        "held_out_need": held_out_need,
        "lives": scored_lives,
        "sizeless_requests_per_life": wanted / max(1, scored_lives),
        # Both sizes are asked, so a model with no evidence scores exactly one
        # half whatever its tie-break is, and cannot identify the answer by
        # elimination.
        "first_use_accuracy": {
            mode: scores[mode] / max(1, questions) for mode in scores
        },
        "had_any_evidence": {
            mode: informed[mode] / max(1, scored_lives) for mode in informed
        },
    }


# -- the survey ----------------------------------------------------------------


STORE_SWEEP = (0.0, 34.0, 30.0, 26.0, 22.0)
OPERATING_STORE = 30.0


def survey(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    closed_loop_lives: int,
) -> dict[str, object]:
    """Is there a basket size at which the size word is worth saying?

    Two things have to be true at once and neither is guaranteed. The size
    question has to *discriminate* -- the truth has to disagree with the species
    rates often enough for a self-model to have anything to contribute -- and
    the ration has to *bite*, so that asking for too much costs something a
    survival or efficiency number can see. A store so large that always-large
    wins makes the word decoration; one so small that nothing survives makes the
    endpoint a floor. This finds the band, before any gate is locked, at the
    operating point the gates will be scored at.
    """

    discrimination = [
        run_open_loop(
            model,
            organism,
            world_name=name,
            store=OPERATING_STORE,
            lives=lives,
            seed_base=seed_base,
        )
        for name in (TREATMENT_WORLD, "null")
    ]

    ration = []
    tiers_at_operating_point = []
    if closed_loop_lives:
        for store in STORE_SWEEP:
            for control in (None, *SPEAKER_CONTROLS):
                ration.append(
                    run_closed_loop(
                        model,
                        organism,
                        tier_name="oracle",
                        world_name=TREATMENT_WORLD,
                        store=store,
                        lives=closed_loop_lives,
                        seed_base=seed_base + 500_000,
                        control=control,
                        need_driver="oracle",
                    )
                )
        for tier in TIERS:
            tiers_at_operating_point.append(
                run_closed_loop(
                    model,
                    organism,
                    tier_name=tier,
                    world_name=TREATMENT_WORLD,
                    store=OPERATING_STORE,
                    lives=closed_loop_lives,
                    seed_base=seed_base + 500_000,
                    need_driver="oracle",
                )
            )

    return {
        "discrimination": discrimination,
        "ration": ration,
        "tiers_at_operating_point": tiers_at_operating_point,
        "tiers": list(TIERS),
        "store_sweep": list(STORE_SWEEP),
        "operating_store": OPERATING_STORE,
        "seed_base": seed_base,
    }


# -- the locked gates ----------------------------------------------------------


def _verdict(flags: list[bool], needed: int = 4) -> dict[str, object]:
    return {
        "per_seed": flags,
        "passing": sum(flags),
        "required": needed,
        "pass": sum(flags) >= needed,
    }


def score_gates(rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Exactly the gates in the preregistration, scored per seed.

    Nothing here reads a threshold from the data. Every number below is written
    in `docs/decisions/2026-08-14-portion-request-preregistration.md` and was
    fixed against a ceiling survey run on a disjoint seed band, at this
    operating point, before any treatment seed existed.
    """

    def accuracy(world: str, tier: str) -> list[float]:
        return [float(row["size_accuracy"][tier]) for row in rows[world]]  # type: ignore[index]

    treatment = TREATMENT_WORLD
    g1 = [
        learned - snap >= 0.05
        for learned, snap in zip(accuracy(treatment, "recursive"), accuracy(treatment, "snap"))
    ]
    g2 = [
        learned - state >= 0.01
        for learned, state in zip(
            accuracy(treatment, "recursive"), accuracy(treatment, "state_oracle")
        )
    ]
    g3 = [
        individual - state >= 0.03
        for individual, state in zip(
            accuracy(treatment, "individual"), accuracy(treatment, "state_oracle")
        )
    ]
    g4 = [
        abs(learned - species) <= 0.005
        for learned, species in zip(accuracy("null", "recursive"), accuracy("null", "population"))
    ]
    g5 = [
        learned - species <= 0.01
        for learned, species in zip(
            accuracy("shuffled", "recursive"), accuracy("shuffled", "population")
        )
    ]
    g6 = [
        float(row["first_use_accuracy"]["factored"]) >= 0.90  # type: ignore[index]
        and float(row["first_use_accuracy"]["tabular"]) <= 0.60  # type: ignore[index]
        for row in rows["holdout"]
    ]
    g7 = []
    for row in rows["behaviour"]:
        speakers = {entry["control"]: entry for entry in row["speakers"]}  # type: ignore[index]
        sized = float(speakers[None]["survival"])
        g7.append(
            sized >= float(speakers["always_large"]["survival"]) + 0.10
            and sized >= float(speakers["always_small"]["survival"]) + 0.10
        )

    gates = {
        "G1_self_model_over_evidence": _verdict(g1),
        "G2_rate_over_perfect_state": _verdict(g2),
        "G3_rate_headroom_exists": _verdict(g3),
        "G4_no_false_discovery": _verdict(g4),
        "G5_shuffled_readings": _verdict(g5),
        "G6_first_use_composition": _verdict(g6),
        "G7_the_size_word_pays_rent": _verdict(g7),
    }
    gates["all_pass"] = all(gate["pass"] for gate in gates.values())  # type: ignore[index]
    return gates


def treatment(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    lives: int,
    seeds: int,
    closed_loop_lives: int,
) -> dict[str, object]:
    """Five seeds of everything the preregistration scores."""

    rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in (*WORLDS, "shuffled", "holdout", "behaviour")
    }
    for index in range(seeds):
        seed_base = SEED_BASE + SEED_STRIDE * index
        for name in WORLDS:
            row = run_open_loop(
                model,
                organism,
                world_name=name,
                store=OPERATING_STORE,
                lives=lives,
                seed_base=seed_base,
            )
            rows[name].append(row)
            accuracy = row["size_accuracy"]
            print(
                f"seed {index} {name:>10}: "
                + " ".join(f"{tier[:4]} {accuracy[tier]:.3f}" for tier in TIERS)
                + f" | disc {row['discriminating_share']:.3f}",
                flush=True,
            )
        shuffled = run_open_loop(
            model,
            organism,
            world_name=TREATMENT_WORLD,
            store=OPERATING_STORE,
            lives=lives,
            seed_base=seed_base,
            shuffle_readings=True,
        )
        rows["shuffled"].append(shuffled)
        print(
            f"seed {index} {'shuffled':>10}: "
            f"pop {shuffled['size_accuracy']['population']:.3f} "
            f"rls {shuffled['size_accuracy']['recursive']:.3f}",
            flush=True,
        )

        holdout = composition_holdout(
            model,
            organism,
            world_name=TREATMENT_WORLD,
            store=OPERATING_STORE,
            lives=lives,
            seed_base=seed_base,
        )
        rows["holdout"].append(holdout)
        print(
            f"seed {index} {'holdout':>10}: "
            f"factored {holdout['first_use_accuracy']['factored']:.3f} "
            f"tabular {holdout['first_use_accuracy']['tabular']:.3f}",
            flush=True,
        )

        speakers = []
        for control in (None, *SPEAKER_CONTROLS):
            speakers.append(
                run_closed_loop(
                    model,
                    organism,
                    tier_name="recursive",
                    world_name=TREATMENT_WORLD,
                    store=OPERATING_STORE,
                    lives=closed_loop_lives,
                    seed_base=seed_base,
                    control=control,
                    need_driver="oracle",
                )
            )
        rows["behaviour"].append({"seed": index, "speakers": speakers})
        print(
            f"seed {index} {'behaviour':>10}: "
            + " ".join(
                f"{entry['control'] or 'sized'} {entry['survival']:.3f}"
                for entry in speakers
            ),
            flush=True,
        )

    return {"rows": rows, "gates": score_gates(rows)}


def _format_row(row: dict[str, object], key: str) -> str:
    values = row[key]
    assert isinstance(values, dict)
    return " | ".join(f"{values[tier]:.4f}" for tier in TIERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--lives", type=int, default=40)
    parser.add_argument("--closed-loop-lives", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--seed-base", type=int, default=SURVEY_SEED_BASE)
    parser.add_argument(
        "--treatment",
        action="store_true",
        help="Run the preregistered five-seed treatment instead of the survey.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(args.checkpoint)

    if args.treatment:
        result = treatment(
            model,
            organism,
            lives=args.lives,
            seeds=args.seeds,
            closed_loop_lives=args.closed_loop_lives,
        )
        out = Path(
            args.out or "runs/organism/probe64_portion_request/treatment.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        gates = result["gates"]
        print("\n== locked gates ==")
        for name, verdict in gates.items():
            if name == "all_pass":
                continue
            mark = "PASS" if verdict["pass"] else "FAIL"
            print(
                f"{mark}  {name}: {verdict['passing']}/"
                f"{len(verdict['per_seed'])} {verdict['per_seed']}"
            )
        print(f"all_pass = {gates['all_pass']}")
        print(f"\nWrote {out}")
        return

    result = survey(
        model,
        organism,
        lives=args.lives,
        seed_base=args.seed_base,
        closed_loop_lives=args.closed_loop_lives,
    )

    out = Path(
        args.out or "runs/organism/probe64_portion_request/ceiling_survey.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))

    print("== size accuracy against the truth ==")
    print("| world | " + " | ".join(TIERS) + " | discriminating |")
    print("|---|" + "---:|" * (len(TIERS) + 1))
    for row in result["discrimination"]:
        print(
            f"| {row['world']} | {_format_row(row, 'size_accuracy')} "
            f"| {row['discriminating_share']:.4f} |"
        )

    if result["ration"]:
        print("\n== the ration, need pinned at oracle ==")
        print("| store | speaker | survival | overflow | efficiency | refused |")
        print("|---:|---|---:|---:|---:|---:|")
        for row in result["ration"]:
            print(
                f"| {row['caregiver_store']:.0f} | {row['control'] or 'sized'} "
                f"| {row['survival']:.3f} | {row['overflow']:.2f} "
                f"| {row['uptake_efficiency']:.3f} | {row['refused_grants']:.1f} |"
            )
    if result["tiers_at_operating_point"]:
        print(f"\n== every tier at store {OPERATING_STORE:.0f} ==")
        print("| tier | survival | overflow | efficiency | large share |")
        print("|---|---:|---:|---:|---:|")
        for row in result["tiers_at_operating_point"]:
            print(
                f"| {row['tier']} | {row['survival']:.3f} | {row['overflow']:.2f} "
                f"| {row['uptake_efficiency']:.3f} "
                f"| {row['large_share_granted']:.2f} |"
            )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
