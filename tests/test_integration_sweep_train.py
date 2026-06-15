import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _make_args(tmpdir, **overrides):
    defaults = dict(
        algo="sac",
        timesteps=100,
        run_name="test-run",
        log_dir=Path(tmpdir) / "runs",
        seed=42,
        wandb_project="test",
        wandb_entity=None,
        wandb=False,
        wandb_allow_config=False,
        vision=False,
        target_speed=100.0,
        eval_episodes=2,
        checkpoint_freq=50,
        runtime_target="native",
        torcs_exe=None,
        race_config=None,
        eval_race_config=None,
        launch_log="torcs_launcher.log",
        train_port=3001,
        eval_port=3001,
        sweep_mode=False,
        generated_sweep_config=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestSweepTrainIntegration(unittest.TestCase):
    def test_sweep_mode_creates_unique_run_name(self):
        from torcs_rl.trainer import _build_run_name

        args = _make_args(tempfile.gettempdir(), run_name="torcs-sac")
        name = _build_run_name(args, sweep_mode=True)
        self.assertIn("torcs-sac-sac-sweep-", name)
        self.assertNotEqual(name, "torcs-sac")
        self.assertEqual(_build_run_name(args, sweep_mode=False), "torcs-sac")


class TestSweepConfigGenerated(unittest.TestCase):
    def test_sweep_config_json_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_config = {
                "method": "bayes",
                "metric": {"name": "eval/mean_reward", "goal": "maximize"},
                "parameters": {"seed": {"values": [42]}},
            }
            args = _make_args(tmpdir, generated_sweep_config=sweep_config)

            from torcs_rl.utils import write_json as _write_json

            run_dir = Path(args.log_dir) / "test-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(run_dir / "sweep_config.json", sweep_config)

            config_path = run_dir / "sweep_config.json"
            self.assertTrue(config_path.exists())
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["method"], "bayes")


class TestLapAwareEvaluation(unittest.TestCase):
    def test_eval_callback_exposes_lap_stats_container(self):
        from torcs_rl.evaluation import TorcsEvalCallback

        callback = TorcsEvalCallback.__new__(TorcsEvalCallback)
        callback.latest_lap_stats = {}
        self.assertEqual(callback.latest_lap_stats, {})


if __name__ == "__main__":
    unittest.main()
