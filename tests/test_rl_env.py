import unittest
from unittest.mock import MagicMock
import numpy as np
from torcs_rl.env import TorcsRLEnv

class TestRlEnv(unittest.TestCase):
    def setUp(self):
        # Mock the runtime backend to avoid launching TORCS
        self.mock_backend = MagicMock()
        self.mock_backend.name = "mock"
        self.mock_backend.default_torcs_exe = "torcs"
        self.mock_backend.default_race_config = "config.xml"
        self.mock_backend.default_autostart_script = "autostart.sh"

        # Mock client
        self.mock_client = MagicMock()
        self.mock_client.S.d = {
            "speedX": 50.0,
            "trackPos": 0.0,
            "angle": 0.0,
            "track": [200.0] * 19,
            "damage": 0,
            "img": np.zeros((64, 64, 3))
        }
        self.mock_backend.create_client.return_value = self.mock_client

        self.env = TorcsRLEnv(runtime_backend=self.mock_backend)

    def test_reward_logic_standard(self):
        obs = self.mock_client.S.d
        obs_pre = obs.copy()

        reward, terminated, reason = self.env._compute_reward(obs, obs_pre)

        self.assertGreater(reward, 0)
        self.assertFalse(terminated)
        self.assertEqual(reason, "in_progress")

    def test_reward_logic_damage(self):
        obs = self.mock_client.S.d.copy()
        obs["damage"] = 10
        obs_pre = self.mock_client.S.d.copy()
        obs_pre["damage"] = 0

        reward, terminated, reason = self.env._compute_reward(obs, obs_pre)

        self.assertEqual(reward, -10.0)

    def test_reward_logic_off_track(self):
        obs = self.mock_client.S.d.copy()
        obs["track"] = [-1.0] * 19
        obs_pre = self.mock_client.S.d.copy()

        reward, terminated, reason = self.env._compute_reward(obs, obs_pre)

        self.assertEqual(reward, -10.0)
        self.assertTrue(terminated)
        self.assertEqual(reason, "off_track")

    def test_observation_normalization(self):
        # Reset stats
        self.env._obs_count = 0

        obs1 = np.ones(self.env.obs_dim)
        norm1 = self.env._normalize_observation(obs1, update=True)

        # First observation should just be clipped as count < 2
        np.testing.assert_array_equal(norm1, np.clip(obs1, -self.env.obs_norm_clip, self.env.obs_norm_clip))

        obs2 = np.zeros(self.env.obs_dim)
        norm2 = self.env._normalize_observation(obs2, update=True)

        # Now stats are updated, let's check third one
        obs3 = np.array([0.5] * self.env.obs_dim)
        norm3 = self.env._normalize_observation(obs3, update=False)

        self.assertEqual(norm3.shape, (self.env.obs_dim,))
        self.assertFalse(np.all(norm3 == obs3))

if __name__ == "__main__":
    unittest.main()
