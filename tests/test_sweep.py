"""Unit tests for torcs_rl.sweep — hyperparameter sweep config & runner."""

import argparse
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from torcs_rl.sweep import build_sweep_config, make_sweep_runner


def _make_args(**overrides):
    defaults = dict(
        algo="sac",
        timesteps=1000,
        run_name="test-run",
        log_dir=Path("runs"),
        seed=42,
        wandb=False,
        wandb_project="test-project",
        wandb_entity=None,
        sweep_count=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildSweepConfig(unittest.TestCase):
    """Tests for build_sweep_config()."""

    def test_ppo_config_structure(self):
        config = build_sweep_config(_make_args(algo="sac"))
        self.assertEqual(config["method"], "bayes")
        self.assertEqual(config["metric"]["name"], "eval/progress_distance")
        self.assertEqual(config["metric"]["goal"], "maximize")
        self.assertIn("parameters", config)
        self.assertIn("early_terminate", config)
        self.assertEqual(config["early_terminate"]["type"], "hyperband")

    def test_sac_config_has_expected_params(self):
        config = build_sweep_config(_make_args(algo="sac"))
        params = config["parameters"]
        self.assertIn("track_weight", params)
        self.assertIn("heading_weight", params)
        self.assertIn("steering_weight", params)
        self.assertIn("terminal_penalty", params)

    def test_non_sac_raises(self):
        with self.assertRaises(ValueError):
            build_sweep_config(_make_args(algo="td3"))

    def test_sweep_params_have_values(self):
        config = build_sweep_config(_make_args(algo="sac"))
        for key in ("track_weight", "heading_weight", "steering_weight", "terminal_penalty"):
            param = config["parameters"][key]
            self.assertIn("values", param)
            self.assertGreater(len(param["values"]), 0)


class TestMakeSweepRunner(unittest.TestCase):
    """Tests for make_sweep_runner()."""

    def test_returns_callable(self):
        runner = make_sweep_runner(_make_args())
        self.assertTrue(callable(runner))

    @patch("torcs_rl.trainer.train_and_evaluate")
    def test_runner_calls_train_with_sweep_mode(self, mock_train):
        base_args = _make_args()
        runner = make_sweep_runner(base_args)
        runner()
        mock_train.assert_called_once()
        call_args = mock_train.call_args[0][0]
        self.assertTrue(call_args.sweep_mode)
        self.assertTrue(call_args.wandb)
        # Original args should not be mutated
        self.assertFalse(base_args.wandb)
        self.assertFalse(getattr(base_args, "sweep_mode", False))

    @patch("torcs_rl.trainer.train_and_evaluate")
    def test_runner_copies_args(self, mock_train):
        base_args = _make_args(algo="sac")
        runner = make_sweep_runner(base_args)
        runner()
        call_args = mock_train.call_args[0][0]
        self.assertEqual(call_args.algo, "sac")
        # Mutating runner's args should not affect base
        call_args.algo = "td3"
        self.assertEqual(base_args.algo, "sac")


class TestLaunchSweep(unittest.TestCase):
    """Tests for launch_sweep() with mocked wandb."""

    @patch("wandb.agent")
    @patch("wandb.sweep", return_value="test-sweep-id")
    def test_launch_creates_sweep_and_agent(self, mock_sweep, mock_agent):
        from torcs_rl.sweep import launch_sweep

        args = _make_args(sweep_count=2)
        launch_sweep(args)
        mock_sweep.assert_called_once()
        mock_agent.assert_called_once()
        self.assertEqual(mock_agent.call_args[0][0], "test-sweep-id")
        self.assertEqual(mock_agent.call_args[1]["count"], 2)

    @patch("wandb.agent")
    @patch("wandb.sweep", return_value="id")
    def test_launch_writes_sweep_config_json(self, mock_sweep, mock_agent):
        from torcs_rl.sweep import launch_sweep

        args = _make_args(log_dir=Path("test_runs"))
        launch_sweep(args)
        config_path = Path("test_runs/sweeps/sac_sweep.json")
        self.assertTrue(config_path.exists())
        data = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(data["method"], "bayes")
        # Cleanup
        import shutil
        shutil.rmtree(Path("test_runs"), ignore_errors=True)

    @patch("wandb.agent")
    @patch("wandb.sweep", return_value="id")
    def test_launch_sets_generated_sweep_config_on_args(self, mock_sweep, mock_agent):
        from torcs_rl.sweep import launch_sweep

        args = _make_args()
        launch_sweep(args)
        self.assertTrue(hasattr(args, "generated_sweep_config"))
        self.assertEqual(args.generated_sweep_config["method"], "bayes")


if __name__ == "__main__":
    unittest.main()
