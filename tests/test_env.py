import unittest

from homesocial.env import Action, Direction, HomeostaticSocialGrid


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


if __name__ == "__main__":
    unittest.main()

