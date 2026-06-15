import unittest
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np
from torcs_rl.env import TorcsRLEnv


class TestEnvConstruction(unittest.TestCase):
    def test_observation_space_shape(self):
        mock_backend = MagicMock()
        mock_backend.name = "mock"

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
        mock_backend.create_client.return_value = mock_client

        env = TorcsRLEnv(runtime_backend=mock_backend)
        self.assertEqual(env.observation_space.shape, (22,))
        self.assertEqual(env.action_space.shape, (2,))

    def test_race_config_passing(self):
        mock_backend = MagicMock()
        mock_backend.name = "mock"
        mock_backend.default_torcs_exe = "torcs"
        mock_backend.default_race_config = "default.xml"
        mock_backend.default_autostart_script = "autostart.sh"

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
        mock_backend.create_client.return_value = mock_client

        custom_config = "my_track.xml"
        env = TorcsRLEnv(race_config=custom_config, runtime_backend=mock_backend)

        args, kwargs = mock_backend.create_client.call_args
        self.assertEqual(kwargs["race_config"], str(Path(custom_config).expanduser()))


if __name__ == "__main__":
    unittest.main()
