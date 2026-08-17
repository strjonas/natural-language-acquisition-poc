"""Guards for probe64's levers and mechanism.

Two kinds of test live here. The first kind protects the rest of the
repository: every probe64 lever must be exactly inert at its default, over full
lives, or every number in `docs/decisions/` before today silently means
something else. The second kind protects the claim: the size rule has to
actually depend on the rate at a fixed state, or the whole probe is measuring
state estimation again under a new name.
"""

from __future__ import annotations

from dataclasses import replace
from random import Random

import numpy as np
import pytest

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import (
    HELP_SURFACES,
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportConfig,
    ReportWorld,
    SIZE_TO_WORD,
    heard_size,
)
from homesocial.island.world import IslandConfig
from homesocial.organism.portion_request import (
    ListenerModel,
    choose_size,
    service_interval,
    species_rate,
    visible_help,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
LIFE = 240


def _world(**overrides) -> ReportWorld:
    report = replace(ReportConfig(life_steps=LIFE), **overrides)
    world = ReportWorld(IslandConfig(), report=report, seed=11)
    world.reset(11)
    return world


def _live(world: ReportWorld, *, need_word: str, size_word: str | None) -> dict:
    """Drive one whole life saying the same thing every tick."""

    rng = Random(5)
    actions = ["move_forward", "consume", "rest", "wait", "turn_left"]
    tokens = (
        TOKEN_TO_ID[need_word],
        PAD_ID if size_word is None else TOKEN_TO_ID[size_word],
    )
    trace = []
    while True:
        world.hear(tokens)
        _, _, terminated, truncated, info = world.step(rng.choice(actions))
        trace.append(
            (
                info["granted_need"],
                info["granted_large"],
                round(float(info["viability"]), 12),
            )
        )
        if terminated or truncated:
            break
    return {"trace": trace, "counts": dict(world.grant_counts)}


# -- inertness -----------------------------------------------------------------


def test_a_size_word_does_nothing_until_the_caregiver_is_listening_for_one():
    """The default caregiver hears "more" and grants exactly what it always did."""

    silent = _live(_world(), need_word="hungry", size_word=None)
    spoken = _live(_world(), need_word="hungry", size_word=SIZE_TO_WORD["large"])
    assert spoken["trace"] == silent["trace"]


def test_the_store_is_inert_at_zero():
    """A caregiver with no basket is the caregiver every earlier probe faced."""

    unlimited = _live(_world(caregiver_store=0.0), need_word="thirsty", size_word=None)
    requested = _live(
        _world(portion_requests=True, caregiver_store=0.0),
        need_word="thirsty",
        size_word=None,
    )
    assert requested["trace"] == unlimited["trace"]
    assert unlimited["counts"]["refused"] == 0


def test_asking_for_nothing_leaves_the_caregivers_own_draw_untouched():
    """Turning the lever on without ever asking must not perturb the help stream."""

    off = _live(_world(), need_word="tired", size_word=None)
    on = _live(_world(portion_requests=True), need_word="tired", size_word=None)
    assert on["trace"] == off["trace"]


# -- the lever actually levers -------------------------------------------------


@pytest.mark.parametrize("size", ["small", "large"])
def test_a_request_gets_the_portion_it_asked_for(size):
    world = _world(portion_requests=True)
    result = _live(world, need_word="hungry", size_word=SIZE_TO_WORD[size])
    granted = [row[1] for row in result["trace"] if row[0] == "food"]
    assert granted, "the caregiver never granted anything"
    assert set(granted) == {size == "large"}


def test_the_surface_still_names_the_portion_it_granted():
    """Portion class stays legible as perception, which is all the listener model reads."""

    world = _world(portion_requests=True)
    world.hear((TOKEN_TO_ID["thirsty"], TOKEN_TO_ID[SIZE_TO_WORD["large"]]))
    for _ in range(world.report.help_period):
        packet, _, _, _, _ = world.step("wait")
    seen = visible_help(packet)
    assert seen == ("water", True)
    assert HELP_SURFACES[("water", True)] == "spring"


def test_a_size_word_without_a_need_word_is_not_a_request():
    world = _world(portion_requests=True)
    world.hear((TOKEN_TO_ID[SIZE_TO_WORD["large"]], PAD_ID))
    assert heard_size((TOKEN_TO_ID[SIZE_TO_WORD["large"]],)) == "large"
    for _ in range(world.report.help_period):
        _, _, _, _, info = world.step("wait")
    assert info["granted_need"] is None


def test_a_stale_size_never_survives_into_the_next_request():
    """Each request carries its own size, or none."""

    world = _world(portion_requests=True)
    world.hear((TOKEN_TO_ID["hungry"], TOKEN_TO_ID[SIZE_TO_WORD["large"]]))
    world.hear((TOKEN_TO_ID["thirsty"], PAD_ID))
    granted = None
    for _ in range(world.report.help_period):
        _, _, _, _, info = world.step("wait")
        if info["granted_need"] is not None:
            granted = info
    assert granted is not None
    assert granted["granted_need"] == "water"
    # No size was asked for, so the caregiver drew for itself as it always has.
    assert granted["granted_large"] in (True, False)


def test_the_basket_empties_and_then_the_caregiver_has_nothing_to_give():
    # One large portion and change: the second request is one the caregiver
    # cannot afford, which is the whole point of a basket.
    store = 1.0
    world = _world(portion_requests=True, caregiver_store=store)
    result = _live(world, need_word="hungry", size_word=SIZE_TO_WORD["large"])
    counts = result["counts"]
    assert counts["refused"] > 0
    assert world.store_spent == pytest.approx(store - world.store_remaining)
    assert world.store_spent <= store + 1e-12
    assert counts["granted_large"] * ReportConfig().portion_large == pytest.approx(
        world.store_spent
    )


def test_a_small_asker_outlasts_a_large_one_on_the_same_basket():
    """Three times the portion is three times the basket. That is the whole cost."""

    small = _live(
        _world(portion_requests=True, caregiver_store=1.0),
        need_word="hungry",
        size_word=SIZE_TO_WORD["small"],
    )
    large = _live(
        _world(portion_requests=True, caregiver_store=1.0),
        need_word="hungry",
        size_word=SIZE_TO_WORD["large"],
    )
    assert small["counts"]["granted_small"] > large["counts"]["granted_large"]
    assert small["counts"]["refused"] < large["counts"]["refused"]


def test_overflow_is_measured_and_a_full_body_wastes_a_large_portion():
    world = _world(portion_requests=True)
    world.grid.needs = replace(world.grid.needs, food=0.99)
    world.hear((TOKEN_TO_ID["hungry"], TOKEN_TO_ID[SIZE_TO_WORD["large"]]))
    uptake = None
    for _ in range(world.report.help_period + 2):
        _, _, _, _, info = world.step("consume")
        if info.get("uptake") is not None:
            uptake = info["uptake"]
    assert uptake is not None
    assert uptake["delivered"] == pytest.approx(ReportConfig().portion_large)
    assert uptake["overflow"] > 0.5
    assert uptake["useful"] == pytest.approx(
        uptake["delivered"] - uptake["overflow"]
    )


# -- the mechanism -------------------------------------------------------------


def test_the_size_rule_moves_with_the_rate_at_a_fixed_state():
    """The whole probe in one assertion.

    Same believed body, same headroom, same portions, same caregiver: only how
    fast this body burns differs, and the request differs. If this fails there
    is nothing here that perfect knowledge of the state could not supply.
    """

    report = ReportConfig()
    interval = service_interval(report)
    common = dict(
        believed_level=0.5, believed_uptake=1.0, report=report, interval=interval
    )
    slow = choose_size(believed_rate=0.2 / interval * 0.5, **common)
    fast = choose_size(believed_rate=0.2 / interval * 2.0, **common)
    assert slow == "small"
    assert fast == "large"


def test_the_size_rule_also_respects_a_body_with_no_room_left():
    """A rate fact does not license paying triple for what falls off the top."""

    report = ReportConfig()
    interval = service_interval(report)
    full = choose_size(
        believed_level=0.98,
        believed_rate=0.05,
        believed_uptake=1.0,
        report=report,
        interval=interval,
    )
    assert full == "small"


def test_the_species_rate_is_the_same_number_the_filter_burns():
    report = ReportConfig()
    assert species_rate(report, "food", 0.5) == report.food_metabolism
    assert species_rate(report, "energy", 0.0) == report.energy_metabolism
    assert species_rate(report, "energy", 1.0) == report.move_energy_metabolism


def test_service_interval_is_the_caregivers_clock_not_a_tuned_constant():
    report = ReportConfig()
    assert service_interval(report) == report.help_period * len(REPORT_NEEDS)


# -- the listener model --------------------------------------------------------


def test_the_listener_model_learns_which_word_means_large():
    model = ListenerModel("factored")
    assert model.probability_large("food", SIZE_TO_WORD["large"]) == 0.5
    model.observe("food", SIZE_TO_WORD["large"], True)
    model.observe("food", SIZE_TO_WORD["small"], False)
    assert model.word_for("food", "large") == SIZE_TO_WORD["large"]
    assert model.word_for("food", "small") == SIZE_TO_WORD["small"]


def test_the_factored_model_says_a_combination_it_has_never_said():
    """Property 4, in the smallest form this ecology admits."""

    factored = ListenerModel("factored")
    tabular = ListenerModel("tabular")
    # Everything it has ever been shown is about food and water.
    for observer in (factored, tabular):
        observer.observe("food", SIZE_TO_WORD["large"], True)
        observer.observe("food", SIZE_TO_WORD["small"], False)
        observer.observe("water", SIZE_TO_WORD["large"], True)
        observer.observe("water", SIZE_TO_WORD["small"], False)

    assert factored.word_for("energy", "large") == SIZE_TO_WORD["large"]
    # The table has an empty cell and nothing to fill it from, so it does what
    # an organism that has only memorized its own transcript can do: it tries a
    # word to find out.
    assert not tabular.knows("energy")
    assert factored.knows("energy")


def test_an_untried_word_is_tried_rather_than_assumed():
    model = ListenerModel("factored")
    model.observe("food", SIZE_TO_WORD["large"], True)
    # It has never said the other word, so it does not yet know that word means
    # small -- it knows only that this one means large.
    assert model.word_for("food", "small") == SIZE_TO_WORD["small"]
    assert model.evidence("food", SIZE_TO_WORD["small"]) == 0.0


def test_the_listener_model_reads_perception_and_nothing_else():
    """``visible_help`` must not be reachable from simulator metadata."""

    world = _world(portion_requests=True)
    world.hear((TOKEN_TO_ID["hungry"], TOKEN_TO_ID[SIZE_TO_WORD["small"]]))
    packet = None
    for _ in range(world.report.help_period):
        packet, _, _, _, _ = world.step("wait")
    assert visible_help(packet) == ("food", False)
    # The same packet with its surfaces removed says nothing, whatever the
    # simulator knows about the grant it just made.
    stripped = replace(packet, visible=())
    assert visible_help(stripped) is None


# -- the convention the listener has to find out -------------------------------


def test_tangled_size_words_are_inert_at_their_default():
    """Off, every need shares the species convention, tick for tick.

    Probe66's lever draws a word-to-size mapping per need. At its default the
    draw still happens -- so no other stream moves -- and every need gets exactly
    ``SIZE_WORDS``, which is what every earlier probe's caregiver did.
    """

    plain = _live(
        _world(portion_requests=True), need_word="hungry",
        size_word=SIZE_TO_WORD["large"],
    )
    explicit = _live(
        _world(portion_requests=True, tangled_size_words=False),
        need_word="hungry", size_word=SIZE_TO_WORD["large"],
    )
    assert explicit["trace"] == plain["trace"]
    for need in REPORT_NEEDS:
        assert _world().size_convention[need] == {"more": "large", "not": "small"}


def test_a_tangled_caregiver_does_not_mean_one_thing_by_one_word():
    """On, at least one need reads the size words the other way round.

    That is the moved ground truth a discovery claim needs: a listener model that
    always pools would be right in one world and confidently wrong in the other,
    and only a learner that finds out which world it is in can be right in both.
    """

    report = replace(ReportConfig(life_steps=LIFE), tangled_size_words=True)
    seen = set()
    for seed in range(12):
        world = ReportWorld(IslandConfig(), report=report, seed=seed)
        world.reset(seed)
        for need in REPORT_NEEDS:
            seen.add(world.size_convention[need]["more"])
    assert seen == {"large", "small"}


def test_a_tangled_caregiver_grants_by_the_need_it_heard():
    """The word alone no longer settles the portion; the pair does."""

    report = replace(
        ReportConfig(life_steps=LIFE), portion_requests=True, tangled_size_words=True
    )
    world = ReportWorld(IslandConfig(), report=report, seed=3)
    world.reset(3)
    convention = world.size_convention
    granted = {}
    rng = Random(2)
    while True:
        need = REPORT_NEEDS[world.grid.step_count % len(REPORT_NEEDS)]
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], TOKEN_TO_ID["more"]))
        _, _, terminated, truncated, info = world.step(
            rng.choice(["move_forward", "wait", "turn_left"])
        )
        if info["granted_need"] is not None:
            granted.setdefault(info["granted_need"], set()).add(info["granted_large"])
        if terminated or truncated:
            break
    for need, sizes in granted.items():
        expected = convention[need]["more"] == "large"
        assert sizes == {expected}, need
