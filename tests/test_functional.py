import unittest
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np
from torcs_rl.env import TorcsRLEnv

class TestVisionMode(unittest.TestCase):
    def test_vision_observation_space(self):
        # Mock backend
        mock_backend = MagicMock()
        mock_backend.name = "mock"

        # Mock client
        mock_client = MagicMock()
        mock_client.S.d = {
            "speedX": 50.0,
            "trackPos": 0.0,
            "angle": 0.0,
            "track": [200.0] * 19,
            "img": np.zeros((64, 64, 3))
        }
        mock_backend.create_client.return_value = mock_client

        # Non-vision env
        env_no_vision = TorcsRLEnv(vision=False, runtime_backend=mock_backend)
        obs_dim_no_vision = env_no_vision.obs_dim

        # Vision env
        env_vision = TorcsRLEnv(vision=True, runtime_backend=mock_backend)
        obs_dim_vision = env_vision.obs_dim

        # Vision should have more dimensions (64*64*3 = 12288 extra)
        self.assertEqual(obs_dim_vision, obs_dim_no_vision + 64 * 64 * 3)
        self.assertEqual(env_vision.observation_space.shape, (obs_dim_vision,))

    def test_race_config_passing(self):
        # Mock backend
        mock_backend = MagicMock()
        mock_backend.name = "mock"
        mock_backend.default_torcs_exe = "torcs"
        mock_backend.default_race_config = "default.xml"
        mock_backend.default_autostart_script = "autostart.sh"

        # Mock client
        mock_client = MagicMock()
        mock_client.S.d = {
            "speedX": 50.0,
            "trackPos": 0.0,
            "angle": 0.0,
            "track": [200.0] * 19,
        }
        mock_backend.create_client.return_value = mock_client

        # Env with custom race config
        custom_config = "my_track.xml"
        env = TorcsRLEnv(race_config=custom_config, runtime_backend=mock_backend)

        # Verify create_client was called with custom_config
        args, kwargs = mock_backend.create_client.call_args
        self.assertEqual(kwargs["race_config"], str(Path(custom_config).expanduser()))

if __name__ == "__main__":
    unittest.main()
