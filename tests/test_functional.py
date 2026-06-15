import unittest
from unittest.mock import MagicMock
import numpy as np
from torcs_rl.env import TorcsRLEnv


def _make_mock_client():
    mock_client = MagicMock()
    mock_client.S.d = {
        "speedX": 50.0,
        "trackPos": 0.0,
        "angle": 0.0,
        "track": [200.0] * 19,
        "distRaced": 0.0,
        "curLapTime": 0.0,
        "lastLapTime": 0.0,
    }
    return mock_client


class TestEnvConstruction(unittest.TestCase):
    def test_observation_space_shape(self):
        mock_client = _make_mock_client()
        env = TorcsRLEnv(client=mock_client)
        self.assertEqual(env.observation_space.shape, (22,))
        self.assertEqual(env.action_space.shape, (2,))

    def test_custom_reward_weights(self):
        mock_client = _make_mock_client()
        env = TorcsRLEnv(
            client=mock_client,
            track_weight=0.5,
            heading_weight=0.2,
            steering_weight=0.05,
            speed_weight=0.1,
            terminal_penalty=200.0,
        )
        self.assertEqual(env.track_weight, 0.5)
        self.assertEqual(env.heading_weight, 0.2)
        self.assertEqual(env.terminal_penalty, 200.0)


if __name__ == "__main__":
    unittest.main()
