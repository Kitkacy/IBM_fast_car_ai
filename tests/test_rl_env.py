import unittest
from unittest.mock import MagicMock
import numpy as np
from torcs_rl.env import TorcsRLEnv


class TestRlEnv(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.S.d = {
            "speedX": 50.0,
            "trackPos": 0.0,
            "angle": 0.0,
            "track": [200.0] * 19,
            "distRaced": 0.0,
            "curLapTime": 0.0,
            "lastLapTime": 0.0,
        }

        self.env = TorcsRLEnv(client=self.mock_client)

    def test_reward_logic_standard(self):
        self.env.prev_dist_raced = 0.0
        reward, terminated, reason = self.env._compute_reward(
            self.mock_client.S.d, steer=0.0
        )
        self.assertFalse(terminated)
        self.assertEqual(reason, "in_progress")

    def test_reward_logic_off_track(self):
        obs = self.mock_client.S.d.copy()
        obs["track"] = [-1.0] * 19
        self.env.prev_dist_raced = 0.0
        reward, terminated, reason = self.env._compute_reward(obs, steer=0.0)
        self.assertTrue(terminated)
        self.assertEqual(reason, "off_track")

    def test_flatten_observation_shape(self):
        obs = self.env._flatten_observation(self.mock_client.S.d)
        self.assertEqual(obs.shape, (22,))

    def test_action_space_is_two_dimensional(self):
        self.assertEqual(self.env.action_space.shape, (2,))

    def test_agent_to_torcs(self):
        action = np.array([0.25, 0.6], dtype=np.float32)
        torcs_action = self.env._agent_to_torcs(action)
        self.assertEqual(torcs_action["steer"], 0.25)
        self.assertAlmostEqual(torcs_action["accel"], 0.6)
        self.assertEqual(torcs_action["brake"], 0.0)

    def test_agent_to_torcs_brake(self):
        action = np.array([0.0, -0.5], dtype=np.float32)
        torcs_action = self.env._agent_to_torcs(action)
        self.assertEqual(torcs_action["accel"], 0.0)
        self.assertAlmostEqual(torcs_action["brake"], 0.5)

if __name__ == "__main__":
    unittest.main()
