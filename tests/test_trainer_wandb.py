import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from torcs_rl.trainer import WandbOutputFormat, _coerce_wandb_scalar, _create_tensorboard_session_dir


class TestWandbScalarCoercion(unittest.TestCase):
    def test_accepts_python_and_numpy_scalars(self):
        self.assertEqual(_coerce_wandb_scalar(3), 3)
        self.assertEqual(_coerce_wandb_scalar(1.25), 1.25)
        self.assertEqual(_coerce_wandb_scalar(True), True)
        self.assertEqual(_coerce_wandb_scalar(np.int64(7)), 7)
        self.assertEqual(_coerce_wandb_scalar(np.float32(2.5)), 2.5)
        self.assertEqual(_coerce_wandb_scalar(np.array([4.0], dtype=np.float32)), 4.0)
        self.assertIsNone(_coerce_wandb_scalar(np.array([1.0, 2.0], dtype=np.float32)))
        self.assertIsNone(_coerce_wandb_scalar("not-a-scalar"))


class TestWandbOutputFormat(unittest.TestCase):
    def test_logs_numpy_scalars_and_skips_excluded_keys(self):
        mock_wandb = Mock()
        mock_wandb.run = object()

        key_values = {
            "train/loss": np.float32(1.5),
            "train/updates": np.int64(8),
            "train/flag": np.bool_(True),
            "train/vector": np.array([1.0, 2.0], dtype=np.float32),
            "train/text": "skip-me",
            "train/excluded": 9.0,
        }
        key_excluded = {
            "train/loss": (),
            "train/updates": (),
            "train/flag": (),
            "train/vector": (),
            "train/text": (),
            "train/excluded": ("wandb",),
        }

        with patch("torcs_rl.trainer.wandb", mock_wandb):
            WandbOutputFormat().write(key_values, key_excluded, step=123)

        mock_wandb.log.assert_called_once_with(
            {
                "train/loss": 1.5,
                "train/updates": 8,
                "train/flag": True,
            },
            step=123,
        )


class TestTensorboardSessionDir(unittest.TestCase):
    def test_creates_unique_session_subdirectories(self):
        with TemporaryDirectory() as tmpdir:
            tb_root = Path(tmpdir) / "tensorboard"
            tb_root.mkdir()

            first = _create_tensorboard_session_dir(tb_root, "torcs-sac")
            second = _create_tensorboard_session_dir(tb_root, "torcs-sac")

            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, tb_root)
            self.assertEqual(second.parent, tb_root)


if __name__ == "__main__":
    unittest.main()
