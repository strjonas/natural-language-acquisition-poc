from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.env import Action
from homesocial.island.report import (
    HELP_SURFACES,
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportConfig,
    ReportWorld,
)
from homesocial.island.world import IslandConfig, IslandWorld, SURFACES
from homesocial.organism.report_audit import (
    audit_counterfactual_body,
    audit_hidden_self_state_decoder,
    audit_observable_history_filter_feasibility,
    audit_observable_history_portion_fork,
    audit_observation_decoder,
    audit_oracle_listener_uptake,
    audit_supervised_state_reporter_upper_bound,
    evaluate_report,
    most_frequent_utterances,
    report_battery,
)
from homesocial.organism.causal_self import (
    audit_structured_causal_self_model,
    causal_social_token,
    train_structured_causal_self_model,
)
from homesocial.organism.causal_self_replication import (
    BASE_SEED_BASE,
    SEED_STRIDE,
    _check_seed_isolation,
)
from homesocial.organism.outcome_aware_self import (
    CausalTokenPlanner,
    legacy_expected_state_scores,
    outcome_aware_scores,
    tied_deficit_intervention,
)
from homesocial.organism.self_belief import train_explicit_self_belief
from homesocial.organism.train import (
    OrganismConfig,
    OrganismTrainer,
    audit_report_lexical_comprehension,
    audit_report_lexical_guidance_mechanics,
    decode_object_option,
    execute_agent_action,
    load_organism_checkpoint,
    object_option_action_index,
    save_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


def report_config(**overrides: object) -> OrganismConfig:
    base = dict(
        report_task=True,
        report_slots=2,
        total_steps=600,
        segment_length=32,
        hidden_size=32,
        token_prediction_weight=0.0,
        next_needs_weight=0.0,
        max_steps=ReportConfig().life_steps,
        report=ReportConfig(life_steps=60),
        island=IslandConfig(max_steps=60, max_visible_slots=2),
        seed=3,
    )
    base.update(overrides)
    return OrganismConfig(**base)  # type: ignore[arg-type]


def lexical_report_config(**overrides: object) -> OrganismConfig:
    base = dict(
        report_task=True,
        report_slots=2,
        report_lexical_childhood_steps=40,
        tied_report_lexicon=True,
        total_steps=40,
        segment_length=16,
        hidden_size=32,
        token_prediction_weight=0.5,
        next_needs_weight=1.0,
        consume_options=True,
        inspect_options=True,
        episodic_binding_size=8,
        report=ReportConfig(life_steps=60, unified_uptake=True),
        island=IslandConfig(
            max_steps=60,
            semantic_choice_trial=True,
            semantic_choice_horizon=40,
            semantic_choice_objects=3,
            semantic_choice_low_need=0.55,
            semantic_choice_rounds=8,
            semantic_choice_return_duration=6,
            report_lexicon_choice=True,
            max_visible_slots=3,
        ),
        seed=3,
    )
    base.update(overrides)
    return OrganismConfig(**base)  # type: ignore[arg-type]


def test_report_head_requires_the_report_task():
    with pytest.raises(ValueError):
        OrganismConfig(report_slots=2)
    with pytest.raises(ValueError):
        OrganismConfig(
            report_task=True,
            report_slots=0,
            counterfactual_report_credit=True,
        )


def test_explicit_self_belief_masks_current_body_after_birth():
    config = report_config(explicit_self_belief=True)
    trainer = OrganismTrainer(config)
    vector_a = np.zeros((1, 1, trainer.model.vector_size), dtype=np.float32)
    vector_b = vector_a.copy()
    vector_a[..., :3] = (0.1, 0.2, 0.3)
    vector_b[..., :3] = (0.9, 0.8, 0.7)
    duration = mx.array([[1.0]], dtype=mx.float32)
    belief_a, _ = trainer.model.self_belief_states(
        mx.array(vector_a), duration, None
    )
    belief_b, _ = trainer.model.self_belief_states(
        mx.array(vector_b), duration, None
    )
    mx.eval(belief_a, belief_b)
    assert np.array_equal(np.asarray(belief_a), np.asarray(belief_b))


def test_explicit_self_belief_development_is_exact_and_persistent(tmp_path):
    config = report_config(
        explicit_self_belief=True,
        self_belief_relational_urgency=True,
        consume_options=True,
        inspect_options=True,
        report=ReportConfig(life_steps=30, unified_uptake=True),
    )
    trainer = OrganismTrainer(config)
    checkpoint = tmp_path / "belief.npz"
    stats = tmp_path / "belief.csv"
    output_config, result = train_explicit_self_belief(
        trainer.model,
        config,
        steps=20,
        urgency_rank_weight=1.0,
        seed_base=137,
        checkpoint=str(checkpoint),
        stats_csv=str(stats),
        log_every_lives=0,
    )
    assert result["development_ticks"] == 20
    assert result["updates"] >= 1
    assert result["base_parameters_unchanged"] == 1.0
    assert output_config.explicit_self_belief
    assert output_config.self_belief_development_steps == 20
    assert output_config.self_belief_relational_urgency
    loaded, loaded_config = load_organism_checkpoint(str(checkpoint))
    assert loaded.has_explicit_self_belief
    assert loaded_config.self_belief_development_steps == 20
    assert loaded.has_self_belief_relational_urgency
    assert loaded_config.self_belief_relational_urgency
    assert stats.exists()


def test_structured_causal_self_development_is_exact_and_persistent(tmp_path):
    config = report_config(
        structured_causal_self_model=True,
        consume_options=True,
        inspect_options=True,
        report=ReportConfig(life_steps=30, unified_uptake=True),
    )
    trainer = OrganismTrainer(config)
    checkpoint = tmp_path / "causal.npz"
    stats = tmp_path / "causal.csv"
    output_config, result = train_structured_causal_self_model(
        trainer.model,
        config,
        steps=20,
        seed_base=157,
        checkpoint=str(checkpoint),
        stats_csv=str(stats),
        log_every_lives=0,
    )
    assert result["development_ticks"] == 20
    assert result["base_parameters_unchanged"] == 1.0
    assert output_config.structured_causal_self_model
    assert output_config.structured_causal_development_steps == 20
    loaded, loaded_config = load_organism_checkpoint(str(checkpoint))
    assert loaded.has_structured_causal_self_model
    assert loaded_config.structured_causal_development_steps == 20
    gate = audit_structured_causal_self_model(
        loaded, loaded_config, lives=2, seed_base=177
    )
    assert 0.0 <= gate["balanced_accuracy"] <= 1.0
    assert 0.0 <= gate["listener_no_help_accuracy"] <= 1.0
    assert stats.exists()


def test_structured_causal_development_actually_varies_with_its_seed():
    """Guard the replication path.

    The causal parameters initialize deterministically, so an mx random seed
    cannot perturb this stage; only ``seed_base`` can. A harness that holds
    ``seed_base`` fixed would retrain a bit-identical model for every --seed
    and silently make multi-seed replication meaningless.
    """

    def develop(seed_base: int) -> dict[str, np.ndarray]:
        config = report_config(
            structured_causal_self_model=True,
            consume_options=True,
            inspect_options=True,
            report=ReportConfig(life_steps=30, unified_uptake=True),
        )
        trainer = OrganismTrainer(config)
        train_structured_causal_self_model(
            trainer.model, config, steps=60, seed_base=seed_base, log_every_lives=0
        )
        return {
            name: np.asarray(value).copy()
            for name, value in tree_flatten(trainer.model.parameters())
            if name.startswith("causal_")
        }

    first = develop(157)
    second = develop(9_157)
    assert set(first) == set(second)
    assert any(
        not np.array_equal(first[name], second[name]) for name in first
    ), "distinct seed_base values must produce distinct causal parameters"


def test_maxmin_planner_abandons_the_deficit_when_two_needs_are_close():
    """Pin the A1b defect so a repair has something to move.

    ``causal_social_token`` maximizes the *minimum* predicted axis. A need word
    concentrates listener mass on one surface, so it delivers a large gain to a
    single axis -- but once that axis passes the second-lowest, ``min`` stops
    improving and the rest of the gain is wasted. A word whose listener mass is
    spread delivers a small gain to every axis, which raises ``min`` directly.
    When the two lowest needs are close together the spreading word wins, and
    the organism stops asking for what it actually lacks.

    Measured on the real checkpoint: with a clear single deficit both the
    developed and the adapted planner emit a need word 100% of the time; with
    the two lowest needs within 0.05 that falls to 49.6% and 42.2%.

    When the planner objective is replaced, update this test to assert the new
    behaviour rather than deleting it.
    """

    config = report_config(
        structured_causal_self_model=True,
        consume_options=True,
        inspect_options=True,
        report=ReportConfig(life_steps=30, unified_uptake=True),
    )
    model = OrganismTrainer(config).model
    no_help = len(SURFACES)
    food_token = TOKEN_TO_ID[NEED_TO_REPORT_WORD["food"]]
    spread_token = TOKEN_TO_ID["more"]
    need_surfaces = [SURFACES.index(name) for name in ("roots", "spring", "mushroom")]

    uptake = np.full((len(SURFACES), 3), -20.0, dtype=np.float32)
    for axis, surface in enumerate(need_surfaces):
        uptake[surface, axis] = float(np.log(np.expm1(0.90)))
    model.causal_uptake_raw = mx.array(uptake)

    listener = np.full((model.vocab_size, no_help + 1), -20.0, dtype=np.float32)
    listener[:, no_help] = 20.0
    # The food word buys food, and only food.
    listener[food_token, need_surfaces[0]] = 20.0
    listener[food_token, no_help] = -20.0
    # The spreading word buys a third of each, helping every axis a little.
    for surface in need_surfaces:
        listener[spread_token, surface] = 20.0
    listener[spread_token, no_help] = -20.0
    model.causal_listener_logits = mx.array(listener)
    mx.eval(model.parameters())

    # One clear deficit: the food word raises the minimum most, and wins.
    clear = causal_social_token(
        model, np.array([0.20, 0.90, 0.95]), step_count=0, help_period=6
    )
    # Two nearly tied deficits: fixing food leaves min at water's 0.32, while
    # spreading lifts both to ~0.6. The planner abandons the real deficit.
    tied = causal_social_token(
        model, np.array([0.30, 0.32, 0.90]), step_count=0, help_period=6
    )

    assert clear == food_token
    assert tied == spread_token


def test_outcome_aware_planner_repairs_the_locked_tied_deficit_intervention():
    result = tied_deficit_intervention()

    assert result == {
        "clear_legacy_targeted": 1.0,
        "clear_outcome_aware_targeted": 1.0,
        "tied_legacy_diffuse": 1.0,
        "tied_outcome_aware_targeted": 1.0,
        "gate_passed": 1.0,
    }


def test_outcome_aware_planner_keeps_utility_inside_listener_expectation():
    """The listener can grant one resource, never a fraction of all three."""

    uptake = np.eye(3, dtype=np.float64) * 0.9
    listener = np.zeros((2, 4), dtype=np.float64)
    listener[0, 0] = 1.0
    listener[1, :3] = 1.0 / 3.0
    belief = np.array([0.30, 0.32, 0.90], dtype=np.float64)
    drift = np.zeros(3, dtype=np.float64)

    legacy = legacy_expected_state_scores(
        listener, drift, uptake, belief, ticks_to_help=1
    )
    outcome_aware = outcome_aware_scores(
        listener, drift, uptake, belief, ticks_to_help=1
    )

    assert legacy[1] > legacy[0]
    assert outcome_aware[0] > outcome_aware[1]
    assert outcome_aware[0] == pytest.approx(0.32)
    assert outcome_aware[1] == pytest.approx((0.32 + 0.30 + 0.30) / 3.0)


def test_outcome_aware_planner_can_ground_an_arbitrary_full_vocabulary_token():
    """Selection follows learned consequences, not the public need-word table."""

    config = report_config(
        structured_causal_self_model=True,
        consume_options=True,
        inspect_options=True,
        report=ReportConfig(life_steps=30, unified_uptake=True),
    )
    model = OrganismTrainer(config).model
    no_help = len(SURFACES)
    arbitrary_token = TOKEN_TO_ID["more"]
    food_surface = SURFACES.index("roots")
    uptake = np.full((len(SURFACES), 3), -20.0, dtype=np.float32)
    uptake[food_surface, 0] = float(np.log(np.expm1(0.90)))
    listener = np.full(
        (model.vocab_size, no_help + 1), -20.0, dtype=np.float32
    )
    listener[:, no_help] = 20.0
    listener[arbitrary_token, :] = -20.0
    listener[arbitrary_token, food_surface] = 20.0
    model.causal_uptake_raw = mx.array(uptake)
    model.causal_listener_logits = mx.array(listener)
    mx.eval(model.parameters())

    planner = CausalTokenPlanner.from_model(model)
    selected = planner.token(
        np.array([0.20, 0.90, 0.95]), step_count=0, help_period=6
    )

    assert selected == arbitrary_token


def test_replication_seed_streams_never_touch_the_evaluation_bands():
    """No developmental stream may overlap the worlds it is scored against.

    Development draws worlds from ``seed_base + life``. The promotion gate and
    the planner battery draw from fixed evaluation bases, so a badly chosen
    stride would let one replication seed develop on its own held-out lives and
    silently inflate that seed's result.
    """

    for index in range(8):
        seed_base = BASE_SEED_BASE + index * SEED_STRIDE
        _check_seed_isolation(seed_base)

    for colliding in (7_100_000, 6_500_000, 7_500_000):
        with pytest.raises(ValueError, match="overlaps the evaluation band"):
            _check_seed_isolation(colliding)


def test_causal_social_planner_selects_by_learned_listener_consequence():
    config = report_config(
        structured_causal_self_model=True,
        consume_options=True,
        inspect_options=True,
        report=ReportConfig(life_steps=30, unified_uptake=True),
    )
    model = OrganismTrainer(config).model
    uptake_raw = np.full((len(SURFACES), 3), -10.0, dtype=np.float32)
    listener_logits = np.full(
        (len(VOCAB), len(SURFACES) + 1), -8.0, dtype=np.float32
    )
    listener_logits[:, -1] = 8.0
    positive_raw = np.log(np.expm1(0.4))
    for need_index, need in enumerate(REPORT_NEEDS):
        token = TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]]
        surface = next(
            surface
            for (mapped_need, large), surface in HELP_SURFACES.items()
            if mapped_need == need and not large
        )
        surface_index = SURFACES.index(surface)
        uptake_raw[surface_index, need_index] = positive_raw
        listener_logits[token, :] = -8.0
        listener_logits[token, surface_index] = 8.0
    model.causal_uptake_raw = mx.array(uptake_raw)
    model.causal_listener_logits = mx.array(listener_logits)
    for need_index, need in enumerate(REPORT_NEEDS):
        belief = np.full((3,), 0.8, dtype=np.float64)
        belief[need_index] = 0.2
        token = causal_social_token(
            model, belief, step_count=1, help_period=6
        )
        assert token == TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]]


def test_trainer_builds_a_report_world_and_a_mouth():
    trainer = OrganismTrainer(report_config())
    assert isinstance(trainer.world, ReportWorld)
    assert trainer.model.can_speak
    assert trainer.model.report_slots == 2


def test_report_lexical_childhood_is_public_semantic_choice_not_report_world():
    trainer = OrganismTrainer(lexical_report_config())
    assert isinstance(trainer.world, IslandWorld)
    assert not isinstance(trainer.world, ReportWorld)
    assert trainer.world.config.report_lexicon_choice
    assert trainer.model.can_speak
    assert trainer.model.has_tied_report_lexicon


def test_report_lexical_childhood_labels_external_resource_with_need_word():
    trainer = OrganismTrainer(lexical_report_config())
    world = trainer.world
    packet = trainer.packet
    slot = 0
    surface = packet.visible[slot][2]
    kind = world.grid.kind_by_surface[SURFACES[surface]]
    action = object_option_action_index(
        "inspect",
        slot,
        consume_options=True,
        inspect_options=True,
        visible_slots=3,
    )
    labeled, _, terminated, truncated, info = execute_agent_action(
        world,
        packet,
        action,
        consume_options=True,
        inspect_options=True,
    )
    assert not terminated and not truncated
    assert str(info["situation"]).startswith("label|")
    expected = {"food": "hungry", "water": "thirsty", "energy": "tired"}[kind]
    assert VOCAB[labeled.tokens[1]] == expected


def test_every_report_lexicon_target_fits_the_fixed_four_tick_option():
    config = lexical_report_config()
    for seed in range(12):
        for round_index in range(8):
            for slot in range(3):
                world = IslandWorld(config.island, seed=seed)
                packet = world.reset(seed)
                for _ in range(round_index):
                    observation = world.grid.start_next_semantic_choice_round()
                    packet = world._packet(observation, None)
                action = object_option_action_index(
                    "inspect",
                    slot,
                    consume_options=True,
                    inspect_options=True,
                    visible_slots=3,
                )
                _, _, terminated, truncated, info = execute_agent_action(
                    world,
                    packet,
                    action,
                    consume_options=True,
                    inspect_options=True,
                )
                assert not terminated and not truncated
                assert info["duration"] == 4

        # Exercise the exact within-round path used by learning: after one
        # label the fixed return must restore the same reachable canonical
        # heading before another object option is sampled.
        for second_slot in range(3):
            world = IslandWorld(config.island, seed=seed)
            packet = world.reset(seed)
            first_action = object_option_action_index(
                "inspect",
                0,
                consume_options=True,
                inspect_options=True,
                visible_slots=3,
            )
            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                first_action,
                consume_options=True,
                inspect_options=True,
            )
            assert not terminated and not truncated
            packet, _, terminated, truncated, return_info = execute_agent_action(
                world,
                packet,
                list(Action).index(Action.WAIT),
                consume_options=True,
                inspect_options=True,
            )
            assert not terminated and not truncated
            assert return_info["duration"] == 6
            second_action = object_option_action_index(
                "inspect",
                second_slot,
                consume_options=True,
                inspect_options=True,
                visible_slots=3,
            )
            _, _, terminated, truncated, second_info = execute_agent_action(
                world,
                packet,
                second_action,
                consume_options=True,
                inspect_options=True,
            )
            assert not terminated and not truncated
            assert second_info["duration"] == 4


def test_lexical_childhood_uses_exact_budget_then_reaches_adult_world():
    # 47 ends exactly on a non-final resource consume for seed 3. The
    # semantic-round transition must not clear the global truncation flag.
    trainer = OrganismTrainer(
        lexical_report_config(
            total_steps=47,
            report_lexical_childhood_steps=47,
        )
    )
    stats = trainer.train()
    assert stats
    assert trainer.global_steps == 47
    assert sum(life.steps for life in stats) == 47
    assert isinstance(trainer.world, ReportWorld)


def test_energy_consumption_completes_a_report_lexicon_round():
    config = lexical_report_config()
    world = IslandWorld(config.island, seed=29)
    packet = world.reset(29)
    energy_slot = next(
        slot
        for slot, (_, _, surface) in enumerate(packet.visible)
        if world.grid.kind_by_surface[SURFACES[surface]] == "energy"
    )
    action = object_option_action_index(
        "consume",
        energy_slot,
        consume_options=True,
        inspect_options=True,
        visible_slots=3,
    )
    _, _, terminated, truncated, info = execute_agent_action(
        world,
        packet,
        action,
        consume_options=True,
        inspect_options=True,
    )
    assert not terminated and not truncated
    assert info["event"] == "consumed_energy"
    assert info["semantic_choice_round_complete"]
    assert info["semantic_choice_round_pending"]


def test_guided_joint_attention_is_fixed_and_receives_no_actor_credit():
    base = lexical_report_config()
    trainer = OrganismTrainer(
        replace(
            base,
            island=replace(
                base.island,
                report_lexicon_guided_labels=True,
            ),
        )
    )
    guidance = trainer.world.semantic_choice_guidance_surface
    assert guidance is not None
    action, _, _, _, decision_weight, _ = trainer._act()
    decoded = decode_object_option(
        action,
        consume_options=True,
        inspect_options=True,
        visible_slots=3,
    )
    assert decoded is not None and decoded[0] == "inspect"
    selected_surface = SURFACES[trainer.packet.visible[decoded[1]][2]]
    assert selected_surface == guidance
    assert decision_weight == 0.0


def test_guided_lexical_mechanics_are_balanced_and_need_independent():
    result = audit_report_lexical_guidance_mechanics(lives=300)
    assert result["mechanics_passed"] == 1.0
    assert result["guided_labels_per_round"] == 1.0
    assert result["four_tick_inspect_rate"] == 1.0
    assert result["six_tick_return_rate"] == 1.0


def test_report_logits_cover_the_whole_closed_vocabulary():
    trainer = OrganismTrainer(report_config())
    vector = mx.array(trainer.packet.vector()[None, None, :])
    tokens = mx.array(
        np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
    )
    states, _ = trainer.model.core_states(vector, tokens, None)
    logits = trainer.model.report_logits(states)
    assert logits.shape == (1, 1, 2, len(VOCAB))


def test_tied_report_logits_use_the_shared_lexical_table():
    trainer = OrganismTrainer(
        report_config(tied_report_lexicon=True)
    )
    vector = mx.array(trainer.packet.vector()[None, None, :])
    tokens = mx.array(
        np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
    )
    states, _ = trainer.model.core_states(vector, tokens, None)
    logits = trainer.model.report_logits(states)
    assert logits.shape == (1, 1, 2, len(VOCAB))
    assert hasattr(trainer.model, "report_query")
    assert not hasattr(trainer.model, "report_head")


def test_counterfactual_report_critic_scores_every_token_per_slot():
    trainer = OrganismTrainer(
        report_config(counterfactual_report_credit=True)
    )
    vector = mx.array(trainer.packet.vector()[None, None, :])
    tokens = mx.array(
        np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
    )
    states, _ = trainer.model.core_states(vector, tokens, None)
    utterance = mx.array([[[TOKEN_TO_ID["hungry"], PAD_ID]]])
    values = trainer.model.counterfactual_report_q(states, utterance)
    assert values.shape == (1, 1, 2, len(VOCAB))


def test_a_mute_organism_is_unchanged_and_still_trains():
    trainer = OrganismTrainer(report_config(report_slots=0))
    assert not trainer.model.can_speak
    stats = trainer.train()
    assert stats
    assert all(life.report_need_words == 0 for life in stats)


def test_training_runs_and_records_what_was_said():
    trainer = OrganismTrainer(report_config())
    stats = trainer.train()
    assert stats
    assert sum(life.report_need_words for life in stats) > 0
    assert all(
        life.report_truthful_words <= life.report_need_words for life in stats
    )


@pytest.mark.parametrize("counterfactual_report_credit", [False, True])
def test_nothing_in_the_loss_ever_sees_the_true_body_when_unsupervised(
    counterfactual_report_credit: bool,
):
    """The report head's only gradient is the advantage of the life it led to.

    Zeroing the bodily-prediction weight removes the one loss term that is
    given true need values, so no term in the objective can state the hidden
    body. This test pins that wiring: with that weight at zero, replacing the
    stored true needs with garbage must not change the loss at all.
    """

    trainer = OrganismTrainer(
        report_config(
            next_needs_weight=0.0,
            counterfactual_report_credit=counterfactual_report_credit,
        )
    )
    segment, hidden, bootstrap = trainer.collect_segment()
    advantages, returns = trainer._policy_targets(segment, bootstrap)

    def loss_with(needs: list[tuple[float, ...]], next_needs: list[tuple[float, ...]]):
        return float(
            trainer._loss(
                mx.array(np.stack(segment.vectors)[None, ...]),
                mx.array(np.asarray(segment.tokens, dtype=np.int32)[None, ...]),
                hidden,
                mx.array(np.asarray(segment.actions, dtype=np.int32)),
                mx.array(advantages),
                mx.array(returns),
                mx.array(np.asarray(segment.planning_scales, dtype=np.float32)),
                mx.array(np.stack(segment.action_masks)),
                mx.array(np.asarray(segment.decision_weights, dtype=np.float32)),
                segment.semantic_choice_delayed,
                mx.array(np.stack(segment.next_vectors)),
                mx.array(np.asarray(next_needs, dtype=np.float32)),
                mx.array(np.asarray(segment.env_rewards, dtype=np.float32)),
                mx.array(np.asarray(segment.next_tokens, dtype=np.int32)),
                mx.array(np.asarray(needs, dtype=np.float32)),
                mx.array(np.asarray(segment.report_tokens, dtype=np.int32)),
                mx.array(np.asarray(segment.report_weights, dtype=np.float32)),
            )
        )

    honest = loss_with(segment.needs, segment.next_needs)
    garbage = [(0.0, 0.0, 0.0, 0.0)] * len(segment.needs)
    lied_to = loss_with(garbage, garbage)
    assert honest == pytest.approx(lied_to, abs=1e-6)


def test_the_report_head_does_receive_gradient_from_consequences():
    trainer = OrganismTrainer(report_config())
    segment, hidden, bootstrap = trainer.collect_segment()
    before = np.asarray(trainer.model.report_head.weight)
    trainer.update(segment, hidden, bootstrap)
    after = np.asarray(trainer.model.report_head.weight)
    assert not np.allclose(before, after)


def test_tied_mouth_sends_report_credit_into_the_heard_token_table():
    trainer = OrganismTrainer(report_config(tied_report_lexicon=True))
    segment, hidden, bootstrap = trainer.collect_segment()
    before = np.asarray(trainer.model.token_embedding.weight).copy()
    trainer.update(segment, hidden, bootstrap)
    after = np.asarray(trainer.model.token_embedding.weight)
    assert not np.allclose(before, after)


def test_lexical_comprehension_audit_is_paired_and_read_only():
    trainer = OrganismTrainer(lexical_report_config())
    before = np.asarray(trainer.model.token_embedding.weight).copy()
    result = audit_report_lexical_comprehension(
        trainer.model, lives=6, base_seed=67
    )
    assert result["lives"] == 6
    assert 0.0 <= result["intact_correct_rate"] <= 1.0
    assert 0.0 <= result["cyclic_correct_rate"] <= 1.0
    assert 0.0 <= result["paired_action_change_rate"] <= 1.0
    assert np.array_equal(before, np.asarray(trainer.model.token_embedding.weight))


def test_counterfactual_report_critic_learns_only_on_heard_returns():
    trainer = OrganismTrainer(
        report_config(counterfactual_report_credit=True)
    )
    segment, hidden, bootstrap = trainer.collect_segment()
    assert sum(segment.report_weights) > 0
    before = np.asarray(trainer.model.report_critic_out.weight)
    trainer.update(segment, hidden, bootstrap)
    after = np.asarray(trainer.model.report_critic_out.weight)
    assert not np.allclose(before, after)


def test_utterance_is_sampled_from_the_state_that_acts():
    trainer = OrganismTrainer(report_config())
    vector = mx.array(trainer.packet.vector()[None, None, :])
    tokens = mx.array(
        np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
    )
    states, _ = trainer.model.core_states(vector, tokens, None)
    said = trainer._sample_report(states)
    assert len(said) == 2
    assert all(0 <= token < len(VOCAB) for token in said)


def test_checkpoint_round_trip_preserves_the_report_configuration():
    config = report_config()
    trainer = OrganismTrainer(config)
    path = "runs/organism/_test_report_checkpoint.npz"
    save_checkpoint(trainer.model, config, path)
    model, restored = load_organism_checkpoint(path)
    assert model.can_speak
    assert restored.report_task
    assert restored.report == config.report
    from pathlib import Path

    Path(path).unlink()
    Path(path + ".json").unlink()


def test_lexical_checkpoint_continuation_preserves_model_parameters():
    child = lexical_report_config()
    trainer = OrganismTrainer(child)
    path = "runs/organism/_test_lexical_child_checkpoint.npz"
    save_checkpoint(trainer.model, child, path)
    loaded, restored = load_organism_checkpoint(path)
    assert restored.tied_report_lexicon
    assert restored.report_lexical_childhood_steps == 40
    before = np.asarray(loaded.token_embedding.weight).copy()
    adult = replace(
        restored,
        report_lexical_childhood_steps=0,
        token_prediction_weight=0.0,
        next_needs_weight=0.0,
        island=replace(
            restored.island,
            semantic_choice_trial=False,
            semantic_choice_return_duration=0,
            report_lexicon_choice=False,
            report_lexicon_guided_labels=False,
        ),
    )
    continued = OrganismTrainer(adult, model=loaded)
    assert isinstance(continued.world, ReportWorld)
    assert continued.model.has_tied_report_lexicon
    assert np.array_equal(
        before, np.asarray(continued.model.token_embedding.weight)
    )

    from pathlib import Path

    Path(path).unlink()
    Path(path + ".json").unlink()


def test_audits_run_read_only_on_an_untrained_organism():
    config = report_config()
    trainer = OrganismTrainer(config)
    before = np.asarray(trainer.model.report_head.weight).copy()
    result = evaluate_report(trainer.model, config, lives=2, seed_base=7)
    assert 0.0 <= result["report_fidelity"] <= 1.0
    assert result["lives"] == 2
    for intervention in ("zero", "shuffle", "freeze"):
        lesioned = evaluate_report(
            trainer.model,
            config,
            lives=1,
            seed_base=7,
            intervention=intervention,
        )
        assert 0.0 <= lesioned["report_fidelity"] <= 1.0
    after = np.asarray(trainer.model.report_head.weight)
    assert np.array_equal(before, after)


def test_observation_decoder_cannot_read_a_masked_body():
    """The senses alone must not reveal which need is lowest."""

    config = report_config()
    result = audit_observation_decoder(config, lives=12, seed_base=31)
    assert result["samples"] > 50
    assert result["heldout_accuracy"] < 0.55


def test_counterfactual_and_census_audits_produce_rows():
    config = report_config()
    trainer = OrganismTrainer(config)
    counterfactual = audit_counterfactual_body(
        trainer.model, config, lives=3, seed_base=41
    )
    assert "report_followed_body_rate" in counterfactual
    assert counterfactual["forked_lives"] > 0
    census = most_frequent_utterances(
        trainer.model, config, lives=2, seed_base=41, top=3
    )
    assert len(census) <= 3


def test_oracle_listener_uptake_diagnostic_is_read_only():
    config = report_config()
    trainer = OrganismTrainer(config)
    before = np.asarray(trainer.model.policy.weight).copy()
    result = audit_oracle_listener_uptake(
        trainer.model, config, lives=3, seed_base=47
    )
    assert result["lives"] == 3
    assert set(result["uptake_per_grant"]) == set(REPORT_NEEDS)
    assert np.array_equal(before, np.asarray(trainer.model.policy.weight))


def test_hidden_self_state_decoder_is_heldout_and_read_only():
    config = report_config()
    trainer = OrganismTrainer(config)
    before = np.asarray(trainer.model.input.weight).copy()
    result = audit_hidden_self_state_decoder(
        trainer.model, config, lives=12, seed_base=57
    )
    assert result["train_samples"] > 0
    assert result["heldout_samples"] > 0
    assert 0.0 <= result["state_balanced_accuracy"] <= 1.0
    assert 0.0 <= result["observation_balanced_accuracy"] <= 1.0
    assert np.array_equal(before, np.asarray(trainer.model.input.weight))


def test_supervised_state_reporter_is_a_read_only_closed_loop_upper_bound():
    config = report_config()
    trainer = OrganismTrainer(config)
    before = np.asarray(trainer.model.input.weight).copy()
    result = audit_supervised_state_reporter_upper_bound(
        trainer.model,
        config,
        train_lives=4,
        test_lives=2,
        seed_base=77,
    )
    assert result["train_samples"] > 0
    assert result["test_samples"] > 0
    assert 0.0 <= result["survival"] <= 1.0
    assert 0.0 <= result["balanced_report_fidelity"] <= 1.0
    assert np.array_equal(before, np.asarray(trainer.model.input.weight))


def test_observable_history_portion_fork_tracks_only_lived_uptake():
    config = report_config(
        consume_options=True,
        inspect_options=True,
        report=ReportConfig(life_steps=60, unified_uptake=True),
    )
    result = audit_observable_history_portion_fork(config, seed=107)
    assert result["surface_diverged"] == 1.0
    assert result["preuptake_belief_l1"] == pytest.approx(0.0)
    assert result["postuptake_food_belief_delta"] == pytest.approx(0.4)
    assert result["postuptake_water_belief_delta"] == pytest.approx(0.0)
    assert result["postuptake_energy_belief_delta"] == pytest.approx(0.0)
    assert result["max_truth_tracking_error"] == pytest.approx(0.0)
    assert result["fork_passed"] == 1.0


def test_observable_history_filter_is_a_read_only_epistemic_upper_bound():
    config = report_config(
        consume_options=True,
        inspect_options=True,
        report=ReportConfig(life_steps=60, unified_uptake=True),
    )
    trainer = OrganismTrainer(config)
    before = np.asarray(trainer.model.input.weight).copy()
    result = audit_observable_history_filter_feasibility(
        trainer.model,
        config,
        lives=2,
        seed_base=127,
    )
    assert result["grounded_lives"] == 2
    assert 0.0 <= result["grounded_need_reconstruction"] <= 1.0
    assert 0.0 <= result["grounded_survival"] <= 1.0
    assert 0.0 <= result["scrambled_survival"] <= 1.0
    assert result["fork_passed"] == 1.0
    assert result["gate_passed"] in {0.0, 1.0}
    assert np.array_equal(before, np.asarray(trainer.model.input.weight))


def test_battery_covers_every_preregistered_condition():
    config = report_config()
    trainer = OrganismTrainer(config)
    rows = report_battery(trainer.model, config, lives=2, seed_base=53)
    conditions = {row["condition"] for row in rows}
    assert {
        "grounded",
        "listener_scrambled",
        "listener_mute",
        "organism_mute",
        "intervention_zero",
        "intervention_shuffle",
        "intervention_freeze",
        "heldout_birth_levels",
        "heldout_portions",
        "observation_decoder",
        "counterfactual_body",
    } <= conditions
    assert all(f"fixed_word_{need}" in conditions for need in REPORT_NEEDS)


def test_need_words_are_ordinary_vocabulary_members():
    """The mouth is not given a special three-way choice; it has a language."""

    trainer = OrganismTrainer(report_config())
    logits = trainer.model.report_logits(
        mx.zeros((1, 1, trainer.model.state_size))
    )
    assert logits.shape[-1] == len(VOCAB)
    need_ids = {TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]] for need in REPORT_NEEDS}
    assert len(need_ids) == 3
    assert need_ids < set(range(len(VOCAB)))


def test_only_the_utterance_the_listener_acted_on_gets_credit():
    """An utterance said into the air must not be credited for what followed."""

    trainer = OrganismTrainer(report_config())
    segment, _, _ = trainer.collect_segment()
    weights = segment.report_weights
    assert len(weights) == len(segment)
    credited = sum(weights)
    assert credited > 0
    # At most one utterance per help window can have been acted on.
    period = trainer.config.report.help_period
    assert credited <= len(segment) / period + 1
    assert all(weight in (0.0, 1.0) for weight in weights)


def test_credit_lands_on_the_tick_the_listener_used():
    from homesocial.island.report import heard_need

    trainer = OrganismTrainer(report_config())
    world = trainer.world
    packet = world.reset(5)
    said_ticks: dict[int, str] = {}
    for _ in range(world.report.help_period * 2):
        tick = world.grid.step_count
        tokens = (TOKEN_TO_ID[NEED_TO_REPORT_WORD["water"]], PAD_ID)
        if tick % 2 == 0:
            tokens = (TOKEN_TO_ID["good"], PAD_ID)
        if heard_need(tokens) is not None:
            said_ticks[tick] = "need"
        world.hear(tokens)
        packet, _, terminated, truncated, info = world.step("wait")
        source = info.get("grant_source_tick")
        if source is not None:
            assert info["granted_need"] == "water"
            assert source in said_ticks
            assert source == max(t for t in said_ticks if t < world.grid.step_count)
        if terminated or truncated:
            break
