import unittest

from homesocial.env import (
    LANGUAGE_NECESSARY_MODE,
    Action,
    Direction,
    HomeostaticSocialGrid,
)


class HomeostaticSocialGridTests(unittest.TestCase):
    def test_pointing_at_object_gets_grounded_label(self):
        env = HomeostaticSocialGrid(seed=1)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(Action.POINT)

        self.assertEqual(obs.object_ahead.name, "water")
        self.assertEqual(obs.teacher_utterance, "that is water")
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, 0.0)

    def test_consuming_water_improves_water_need(self):
        env = HomeostaticSocialGrid(seed=1)
        before = env.reset()

        obs, reward, terminated, truncated, info = env.step(Action.CONSUME)

        self.assertEqual(info["event"], "consumed_water")
        self.assertGreater(obs.needs.water, before.needs.water)
        self.assertGreater(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_danger_hurts_safety(self):
        env = HomeostaticSocialGrid(seed=1)
        env.reset()
        env.agent_pos = (3, 3)
        env.direction = Direction.EAST
        before = env.needs.safety

        obs, reward, terminated, truncated, info = env.step(Action.MOVE_FORWARD)

        self.assertEqual(info["event"], "hit_danger")
        self.assertLess(obs.needs.safety, before)
        self.assertEqual(obs.teacher_utterance, "danger hurts you")

    def test_episode_truncates_at_max_steps(self):
        env = HomeostaticSocialGrid(max_steps=2, seed=1)
        env.reset()

        env.step(Action.WAIT)
        obs, reward, terminated, truncated, info = env.step(Action.WAIT)

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_language_necessary_mode_randomizes_hidden_kind_at_fixed_positions(self):
        env = HomeostaticSocialGrid(
            seed=1,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        env.reset(seed=1)
        first_mapping = {obj.pos: obj.kind for obj in env.objects}
        env.reset(seed=2)
        second_mapping = {obj.pos: obj.kind for obj in env.objects}

        self.assertNotEqual(first_mapping, second_mapping)
        self.assertEqual({obj.name for obj in env.objects}, {"object"})

    def test_language_necessary_mode_punishes_unsafe_guessing(self):
        env = HomeostaticSocialGrid(
            seed=1,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        env.reset(seed=5)
        before = env.needs.safety

        obs, reward, terminated, truncated, info = env.step(Action.CONSUME)

        self.assertEqual(info["event"], "consumed_danger")
        self.assertLess(obs.needs.safety, before)
        self.assertEqual(obs.teacher_utterance, "danger hurts you")


if __name__ == "__main__":
    unittest.main()
