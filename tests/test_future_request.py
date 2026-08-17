"""Guards for probe65's lever and mechanism.

Two kinds of test, as in `test_portion_request.py`. The first kind protects the
rest of the repository: ``help_delay`` must be exactly inert at its default over
full lives, or every number in `docs/decisions/` before today silently means
something else. The second kind protects the claim: the named need has to
actually depend on the *rate* at a fixed state, and the divergence scorer has to
score a present-only reporter at exactly zero on a pair it cannot answer -- or
the probe is measuring state estimation again under a new name.
"""

from __future__ import annotations

from dataclasses import replace
from random import Random

import numpy as np
import pytest

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import (
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportConfig,
    ReportWorld,
)
from homesocial.island.world import IslandConfig
from homesocial.organism.future_request import (
    ARM_NAMES,
    RequestLedger,
    choose_need,
    expected_portion,
    answer_horizon,
    score_divergence,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
LIFE = 240
WORD = {need: TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]] for need in REPORT_NEEDS}


def _world(**overrides) -> ReportWorld:
    report = replace(ReportConfig(life_steps=LIFE), **overrides)
    world = ReportWorld(IslandConfig(), report=report, seed=11)
    world.reset(11)
    return world


def _live(world: ReportWorld, *, need_word: str) -> dict:
    """Drive one whole life saying the same thing every tick."""

    rng = Random(5)
    actions = ["move_forward", "consume", "rest", "wait", "turn_left"]
    tokens = (TOKEN_TO_ID[need_word], PAD_ID)
    trace = []
    while True:
        world.hear(tokens)
        _, _, terminated, truncated, info = world.step(rng.choice(actions))
        trace.append(
            (
                info["granted_need"],
                info["granted_large"],
                info["grant_source_tick"],
                round(float(info["viability"]), 12),
            )
        )
        if terminated or truncated:
            break
    return {"trace": trace, "counts": dict(world.grant_counts)}


def _script(world: ReportWorld, script: dict[int, str], steps: int) -> list:
    """Say a need word only on the listed ticks, and record what was granted."""

    granted = []
    for _ in range(steps):
        word = script.get(world.grid.step_count)
        world.hear(None if word is None else (WORD[word], PAD_ID))
        _, _, terminated, truncated, info = world.step("wait")
        if info["granted_need"] is not None or world.grid.step_count % 6 == 0:
            granted.append((world.grid.step_count, info["granted_need"]))
        if terminated or truncated:
            break
    return granted


# -- inertness -----------------------------------------------------------------


def test_help_delay_is_inert_at_its_default():
    """A caregiver with no lag is the caregiver every earlier probe faced.

    Passing the lever explicitly at zero must reproduce a life that never
    mentions it, tick for tick, including which utterance each grant was
    credited to.
    """

    absent = _live(_world(), need_word="hungry")
    explicit = _live(_world(help_delay=0), need_word="hungry")
    assert explicit["trace"] == absent["trace"]
    assert explicit["counts"] == absent["counts"]


def test_a_delayed_caregiver_changes_the_life():
    """The lever is not merely accepted; at a positive value it does something."""

    prompt = _live(_world(), need_word="thirsty")
    slow = _live(_world(help_delay=12), need_word="thirsty")
    assert slow["trace"] != prompt["trace"]


def test_help_delay_rejects_a_negative_lag():
    with pytest.raises(ValueError):
        ReportConfig(help_delay=-1)


# -- what the lag actually does -------------------------------------------------


def test_the_first_grants_of_a_delayed_life_have_nothing_to_answer():
    """Nothing said long enough ago yet, so the early grants are silent.

    With a lag of twelve and a help period of six, the boundary at tick 6 can
    only answer requests from tick -6 or earlier. The boundary at tick 12 is the
    first that can hear the word said at birth.
    """

    world = _world(help_delay=12)
    granted = _script(world, {0: "food"}, steps=20)
    assert (6, None) in granted
    assert (12, "food") in granted


def test_a_delayed_request_is_answered_once_and_in_order():
    """Two words, six ticks apart, come back in the order they were said.

    The caregiver answers the most recent request that is already old enough and
    forgets everything before it, so no request is ever served twice and none
    jumps the queue. The gap at tick 12 is the point: at that boundary the only
    thing said early enough has already been used.
    """

    world = _world(help_delay=6)
    granted = _script(world, {0: "food", 7: "water"}, steps=26)
    assert (6, "food") in granted
    assert (12, None) in granted
    assert (18, "water") in granted
    assert world.grant_counts["food"] == 1
    assert world.grant_counts["water"] == 1


def test_a_delayed_caregiver_still_grants_on_its_own_clock():
    """The lag moves what is granted, never how often."""

    prompt = _live(_world(), need_word="hungry")
    slow = _live(_world(help_delay=12), need_word="hungry")
    assert slow["counts"]["total"] == prompt["counts"]["total"]


# -- the rule --------------------------------------------------------------------


def test_the_horizon_is_one_tick_when_the_caregiver_answers_at_once():
    """The old organism's horizon, which is why ``myopic`` is not a straw arm."""

    assert answer_horizon(0) == 1


def test_the_horizon_is_the_lag_itself():
    """A word acted on at all is acted on exactly ``delay`` ticks later."""

    assert [answer_horizon(delay) for delay in (3, 6, 12, 18)] == [3, 6, 12, 18]


def test_the_horizon_matches_what_the_caregiver_actually_does():
    """Read off the world rather than asserted: how far the body moves between
    a word being said and the grant that answers it."""

    for delay in (6, 12, 18):
        world = _world(help_delay=delay)
        granted = _script(world, {0: "food"}, steps=delay + 8)
        answered = [tick for tick, need in granted if need == "food"]
        assert answered == [answer_horizon(delay)]


def test_the_named_need_depends_on_the_rate_at_a_fixed_state():
    """Same body, same everything, different burn rates: different word.

    This is the whole probe in one assertion. If it fails, nothing downstream is
    about a self-model, because the decision would be a function of the state
    alone and reading the state would be enough.
    """

    report = ReportConfig()
    levels = np.asarray([0.50, 0.44, 0.60])
    nothing = np.zeros(3)
    uptake = {need: 1.0 for need in REPORT_NEEDS}
    slow = choose_need(
        believed_levels=levels,
        believed_rates=np.asarray([0.004, 0.006, 0.008]),
        believed_uptake=uptake,
        arriving=nothing,
        horizon=24,
        report=report,
    )
    fast = choose_need(
        believed_levels=levels,
        believed_rates=np.asarray([0.020, 0.006, 0.008]),
        believed_uptake=uptake,
        arriving=nothing,
        horizon=24,
        report=report,
    )
    assert slow == "water"
    assert fast == "food"


def test_at_no_horizon_the_rate_cannot_change_the_word():
    """The old organism's decision is recovered exactly as the special case."""

    report = ReportConfig()
    levels = np.asarray([0.50, 0.44, 0.60])
    nothing = np.zeros(3)
    uptake = {need: 1.0 for need in REPORT_NEEDS}
    words = {
        choose_need(
            believed_levels=levels,
            believed_rates=np.asarray(rates),
            believed_uptake=uptake,
            arriving=nothing,
            horizon=0,
            report=report,
        )
        for rates in ([0.004, 0.006, 0.008], [0.020, 0.006, 0.008], [0.0, 0.0, 0.0])
    }
    assert words == {"water"}


def test_help_already_on_its_way_moves_the_word():
    """A need with two portions in flight is not the one to ask about."""

    report = ReportConfig()
    levels = np.asarray([0.30, 0.55, 0.60])
    uptake = {need: 1.0 for need in REPORT_NEEDS}
    rates = np.asarray([0.006, 0.006, 0.006])
    unaware = choose_need(
        believed_levels=levels,
        believed_rates=rates,
        believed_uptake=uptake,
        arriving=np.zeros(3),
        horizon=12,
        report=report,
    )
    aware = choose_need(
        believed_levels=levels,
        believed_rates=rates,
        believed_uptake=uptake,
        arriving=np.asarray([0.8, 0.0, 0.0]),
        horizon=12,
        report=report,
    )
    assert unaware == "food"
    assert aware == "water"


# -- the ledger ------------------------------------------------------------------


def test_the_ledger_counts_only_what_lands_before_the_answer_comes():
    """Its own words, the caregiver's public clock, and nothing else.

    With a lag of twelve and a period of six, the word said at tick 6 is
    answered at tick 18. One grant lands before then, at tick 12, and it answers
    what was said at tick 0. So exactly one portion -- the food asked for at
    birth -- is on its way.
    """

    report = ReportConfig()
    ledger = RequestLedger(
        help_period=6, delay=12, expected_portion=expected_portion(report)
    )
    ledger.record(0, "food")
    ledger.record(6, "water")
    arriving = ledger.arriving(6, {need: 1.0 for need in REPORT_NEEDS})
    assert arriving[REPORT_NEEDS.index("food")] == pytest.approx(
        expected_portion(report)
    )
    assert arriving[REPORT_NEEDS.index("water")] == 0.0


def test_the_word_being_chosen_is_never_already_on_its_way():
    """The regression guard for the one bug that inverted the whole probe.

    The grant at ``now + delay`` is the grant this very word will be answered
    by. Counting it as in-flight makes an organism believe help is already
    coming for the need it is about to name, so it names something else -- and
    the survey caught it as a *perfect* speaker surviving 0.225 against 0.475
    for one that ignored the lag altogether.
    """

    ledger = RequestLedger(help_period=6, delay=6, expected_portion=0.4)
    ledger.record(0, "food")
    assert list(ledger.arriving(0, {need: 1.0 for need in REPORT_NEEDS})) == [
        0.0,
        0.0,
        0.0,
    ]


def test_the_ledger_is_empty_for_a_caregiver_that_answers_at_once():
    ledger = RequestLedger(help_period=6, delay=0, expected_portion=0.4)
    ledger.record(0, "food")
    assert list(ledger.arriving(3, {need: 1.0 for need in REPORT_NEEDS})) == [
        0.0,
        0.0,
        0.0,
    ]


def test_the_ledger_values_a_portion_by_what_this_body_absorbs():
    """Absorption is a belief about the self, so two bodies count it differently."""

    ledger = RequestLedger(help_period=6, delay=12, expected_portion=0.4)
    ledger.record(0, "food")
    ledger.record(6, "water")
    index = REPORT_NEEDS.index("food")
    poor = ledger.arriving(6, {need: 0.5 for need in REPORT_NEEDS})
    rich = ledger.arriving(6, {need: 1.5 for need in REPORT_NEEDS})
    assert poor[index] == pytest.approx(0.2)
    assert rich[index] == pytest.approx(0.6)


# -- identical present, divergent future -----------------------------------------


def _record(
    life: int, cell, target: str, present: str, named: dict, inflight=(0, 0, 0)
) -> dict:
    return {
        "cell": cell,
        "inflight": inflight,
        "life": life,
        "target": target,
        "present": present,
        "named": named,
    }


def test_a_present_only_reporter_cannot_diverge_on_a_matched_pair():
    """The control that makes the divergence number mean something.

    Two lives whose bodies fall in one cell and whose present-lowest need agrees
    are indistinguishable to anything reading only the present, so a reporter
    that reads only the present says the same word twice -- by construction, not
    by weakness. A reporter that knows what each body *is* can say different
    words, and the scorer has to separate the two.
    """

    cell = (30, 44, 60)
    left = _record(
        0,
        cell,
        "food",
        "water",
        {**{arm: "water" for arm in ARM_NAMES}, "recursive": "food"},
    )
    right = _record(
        1,
        cell,
        "energy",
        "water",
        {**{arm: "water" for arm in ARM_NAMES}, "recursive": "energy"},
    )
    scored = score_divergence([left, right])
    assert scored["pairs"] == 1
    # ``myopic`` is the arm G7 is scored against, and the one whose zero is
    # structural: with no lookahead its inputs are the present body and the
    # species constants, both matched here.
    assert scored["divergence"]["myopic"] == 0.0
    assert scored["correct_divergence"]["myopic"] == 0.0
    assert scored["divergence"]["state_oracle"] == 0.0
    assert scored["divergence"]["recursive"] == 1.0
    assert scored["correct_divergence"]["recursive"] == 1.0


def test_two_ticks_of_one_life_are_never_a_pair():
    """One life has one body and one set of rates, so nothing about the self
    separates two of its ticks. Such a pair would let the scorer credit ordinary
    drift as self-knowledge."""

    cell = (30, 44, 60)
    named = {arm: "food" for arm in ARM_NAMES}
    records = [
        _record(0, cell, "food", "water", named),
        _record(0, cell, "energy", "water", named),
    ]
    assert score_divergence(records)["pairs"] == 0


def test_pairs_need_an_agreeing_present_and_a_disagreeing_future():
    cell = (30, 44, 60)
    named = {arm: "food" for arm in ARM_NAMES}
    same_future = [
        _record(0, cell, "food", "water", named),
        _record(1, cell, "food", "water", named),
    ]
    different_present = [
        _record(0, cell, "food", "water", named),
        _record(1, cell, "energy", "food", named),
    ]
    assert score_divergence(same_future)["pairs"] == 0
    assert score_divergence(different_present)["pairs"] == 0


# -- the arms are what the documents say they are --------------------------------


def test_myopic_and_state_oracle_share_one_self_model():
    """One integer is the whole difference between them.

    Every document about this probe says so, and the survival gate G8 is a
    contrast between exactly these two arms. If they were ever built from
    separate self-models, that gate would be measuring two estimators instead of
    one belief about the caregiver.
    """

    from dataclasses import replace

    from homesocial.organism.future_request import build_arms
    from homesocial.organism.train import OrganismConfig

    organism = OrganismConfig()
    report = replace(organism.report, help_delay=18)
    _, arms, _ = build_arms(organism, report, delay=18)
    assert arms["myopic"].tier is arms["state_oracle"].tier
    assert arms["myopic"].believed_delay == 0
    assert arms["state_oracle"].believed_delay == 18
    # And every other arm believes the truth about the caregiver.
    for name in ("population", "snap", "recursive", "individual", "oracle"):
        assert arms[name].believed_delay == 18
