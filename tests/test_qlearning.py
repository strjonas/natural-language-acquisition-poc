import unittest

from homesocial.agents import TeacherFollowingAgent
from homesocial.env import (
    LANGUAGE_NECESSARY_MODE,
    Action,
    HomeostaticSocialGrid,
    SilentTeacher,
)
from homesocial.experiment import run_condition, run_scripted_condition
from homesocial.qlearning import QLearningAgent, encode_observation, run_episode


class QLearningTests(unittest.TestCase):
    def test_silent_teacher_emits_no_language(self):
        env = HomeostaticSocialGrid(seed=1, teacher=SilentTeacher())
        env.reset()

        obs, reward, terminated, truncated, info = env.step(Action.POINT)

        self.assertIsNone(obs.teacher_utterance)
        self.assertEqual(info["event"], "pointed")

    def test_observation_encoder_can_drop_language(self):
        env = HomeostaticSocialGrid(seed=1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(Action.POINT)

        with_language = encode_observation(obs, include_language=True)
        without_language = encode_observation(obs, include_language=False)
        without_object_kinds = encode_observation(
            obs, include_language=False, include_object_kinds=False
        )

        self.assertIn("that is water", with_language)
        self.assertNotIn("that is water", without_language)
        self.assertNotIn("water", without_object_kinds)

    def test_q_learning_episode_runs_and_updates_q_values(self):
        env = HomeostaticSocialGrid(seed=1)
        agent = QLearningAgent(seed=1)

        stats = run_episode(env, agent, seed=1, train=True)

        self.assertGreater(stats.steps, 0)
        self.assertGreater(len(agent.q), 0)

    def test_experiment_condition_returns_average_stats(self):
        stats = run_condition(
            condition="grounded_teacher",
            episodes=2,
            eval_episodes=2,
            seed=1,
            randomize_world=True,
            include_object_kinds=False,
        )

        self.assertGreater(stats.steps, 0)
        self.assertGreaterEqual(stats.teacher_utterances, 0)

    def test_teacher_following_agent_uses_grounded_advice(self):
        env = HomeostaticSocialGrid(seed=1)
        agent = TeacherFollowingAgent()
        obs = env.reset()

        self.assertEqual(agent.act(obs), Action.ASK)
        obs, reward, terminated, truncated, info = env.step(Action.ASK)

        self.assertEqual(obs.teacher_utterance, "drink water")
        self.assertEqual(agent.act(obs), Action.CONSUME)

    def test_scripted_probe_separates_grounded_from_silent_diagnostic(self):
        grounded = run_scripted_condition(
            condition="grounded_teacher",
            eval_episodes=5,
            seed=1,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        silent = run_scripted_condition(
            condition="silent_teacher",
            eval_episodes=5,
            seed=1,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )

        self.assertGreater(grounded.resource_uses, silent.resource_uses)
        self.assertGreater(grounded.mean_viability, silent.mean_viability)


if __name__ == "__main__":
    unittest.main()
