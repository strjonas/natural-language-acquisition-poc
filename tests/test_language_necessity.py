import unittest

import numpy as np

from homesocial.env import LANGUAGE_NECESSARY_MODE, Action, HomeostaticSocialGrid
from homesocial.language_necessity import (
    run_blind_diagnostic,
    run_teacher_following_diagnostic,
)
from homesocial.observations import (
    TEACHER_UTTERANCES,
    observation_vector,
    observation_vector_size,
    teacher_utterance_index,
)
from homesocial.teachers import TEACHER_MODES, build_teacher, masks_language


class LanguageNecessityTests(unittest.TestCase):
    def test_teacher_modes_keep_input_shape_invariant(self):
        expected_size = observation_vector_size(
            include_language=True,
            include_object_kinds=False,
        )

        for mode in TEACHER_MODES:
            with self.subTest(mode=mode):
                env = HomeostaticSocialGrid(
                    seed=1,
                    teacher=build_teacher(mode, seed=1),
                    randomize_world=False,
                    diagnostic_mode=LANGUAGE_NECESSARY_MODE,
                )
                env.reset(seed=1)
                observation, _reward, _terminated, _truncated, _info = env.step(
                    Action.ASK
                )

                vector = observation_vector(
                    observation,
                    width=env.width,
                    height=env.height,
                    include_language=True,
                    mask_language=masks_language(mode),
                    include_object_kinds=False,
                )

                self.assertEqual(vector.shape[0], expected_size)

    def test_masked_language_channel_keeps_null_token(self):
        env = HomeostaticSocialGrid(seed=1)
        env.reset(seed=1)
        observation, _reward, _terminated, _truncated, _info = env.step(Action.ASK)

        with_text = observation_vector(
            observation,
            width=env.width,
            height=env.height,
            include_language=True,
            mask_language=False,
            include_object_kinds=False,
        )
        masked = observation_vector(
            observation,
            width=env.width,
            height=env.height,
            include_language=True,
            mask_language=True,
            include_object_kinds=False,
        )
        language_width = len(TEACHER_UTTERANCES) + 1
        text_slice = with_text[-language_width:]
        masked_slice = masked[-language_width:]

        self.assertEqual(np.argmax(text_slice), teacher_utterance_index("drink water"))
        self.assertEqual(np.argmax(masked_slice), teacher_utterance_index(None))

    def test_hidden_kind_does_not_leak_through_masked_language_channel(self):
        env = HomeostaticSocialGrid(
            seed=1,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        first = env.reset(seed=1)
        first_kind = first.object_ahead.kind
        first, _reward, _terminated, _truncated, _info = env.step(Action.ASK)
        first_vector = observation_vector(
            first,
            width=env.width,
            height=env.height,
            include_language=True,
            mask_language=True,
            include_object_kinds=False,
        )

        second = env.reset(seed=2)
        second_kind = second.object_ahead.kind
        second, _reward, _terminated, _truncated, _info = env.step(Action.ASK)
        second_vector = observation_vector(
            second,
            width=env.width,
            height=env.height,
            include_language=True,
            mask_language=True,
            include_object_kinds=False,
        )

        self.assertNotEqual(first_kind, second_kind)
        np.testing.assert_array_equal(first_vector, second_vector)

    def test_grounded_oracle_beats_diagnostic_controls(self):
        grounded = run_teacher_following_diagnostic("grounded", eval_episodes=12)
        silent = run_teacher_following_diagnostic("silent", eval_episodes=12)
        masked = run_teacher_following_diagnostic("masked", eval_episodes=12)
        shuffled = run_teacher_following_diagnostic("shuffled", eval_episodes=12)
        wrong = run_teacher_following_diagnostic("wrong", eval_episodes=12)
        blind_controls = [
            run_blind_diagnostic(policy, eval_episodes=12)
            for policy in ("consume", "rest", "avoid", "ask_then_consume")
        ]
        best_blind_viability = max(stat.mean_viability for stat in blind_controls)
        best_control_resource_uses = max(
            [silent.resource_uses, masked.resource_uses]
            + [stat.resource_uses for stat in blind_controls]
        )

        self.assertGreater(grounded.mean_viability, best_blind_viability + 0.01)
        self.assertGreater(grounded.resource_uses, best_control_resource_uses)
        self.assertLess(masked.mean_viability, grounded.mean_viability)
        self.assertLess(shuffled.mean_viability, grounded.mean_viability)
        self.assertLess(wrong.mean_viability, grounded.mean_viability)


if __name__ == "__main__":
    unittest.main()
