"""Unit tests for torcs_rl.algorithms."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from torcs_rl.algorithms import (
    apply_wandb_overrides,
    build_ppo_hyperparameters,
    build_sac_hyperparameters,
    build_td3_hyperparameters,
    build_wandb_run_config,
    get_algorithm_registry,
    to_json_serializable,
)
from torcs_rl.config import load_config


def _make_config(**overrides):
    config = replace(load_config(), run_name="test", wandb_allow_config=False)
    return replace(config, **overrides)


class TestHyperparameters(unittest.TestCase):
    def test_ppo_returns_dict(self):
        params = build_ppo_hyperparameters(_make_config(seed=42))
        self.assertIsInstance(params, dict)
        self.assertIn("n_steps", params)
        self.assertIn("learning_rate", params)
        self.assertEqual(params["seed"], 42)

    def test_sac_returns_dict(self):
        params = build_sac_hyperparameters(_make_config(seed=42))
        self.assertIn("buffer_size", params)
        self.assertIn("batch_size", params)
        self.assertEqual(params["seed"], 42)

    def test_td3_returns_dict(self):
        params = build_td3_hyperparameters(_make_config(seed=42))
        self.assertIn("policy_delay", params)
        self.assertEqual(params["seed"], 42)


class TestAlgorithmRegistry(unittest.TestCase):
    def test_all_algorithms_present(self):
        registry = get_algorithm_registry()
        for algo in ("ppo", "sac", "td3"):
            self.assertIn(algo, registry)
            _, builder = registry[algo]
            self.assertTrue(callable(builder))


class TestToJsonSerializable(unittest.TestCase):
    def test_primitives(self):
        self.assertEqual(to_json_serializable(42), 42)
        self.assertEqual(to_json_serializable(3.14), 3.14)
        self.assertEqual(to_json_serializable("hi"), "hi")
        self.assertTrue(to_json_serializable(True))
        self.assertIsNone(to_json_serializable(None))

    def test_path(self):
        self.assertEqual(to_json_serializable(Path("a/b")), "a\\b")

    def test_callable(self):
        result = to_json_serializable(lambda: None)
        self.assertIsInstance(result, str)

    def test_dict(self):
        result = to_json_serializable({"a": 1, "b": Path("x")})
        self.assertEqual(result["a"], 1)
        self.assertIsInstance(result["b"], str)


class TestBuildWandbRunConfig(unittest.TestCase):
    def test_contains_base_keys(self):
        config = _make_config()
        algo_hp = build_ppo_hyperparameters(config)
        payload = build_wandb_run_config(config, algo_hp)
        self.assertEqual(payload["algo"], config.algo)
        self.assertEqual(payload["timesteps"], config.timesteps)
        self.assertEqual(payload["seed"], config.seed)

    def test_contains_reward_weights(self):
        config = _make_config(algo="ppo")
        payload = build_wandb_run_config(config, build_ppo_hyperparameters(config))
        self.assertEqual(payload["reward_weights"]["track_weight"], config.track_weight)
        self.assertEqual(payload["reward_weights"]["speed_weight"], config.speed_weight)

    def test_sac_exposes_configurable_sac_keys(self):
        config = _make_config(algo="sac")
        payload = build_wandb_run_config(config, build_sac_hyperparameters(config))
        self.assertIn("sac_learning_rate", payload)
        self.assertIn("sac_batch_size", payload)


class TestApplyWandbOverrides(unittest.TestCase):
    def test_noop_when_allow_config_false(self):
        config = _make_config(wandb_allow_config=False)
        mock_run = MagicMock()
        mock_run.config = {"seed": 999}
        new_config = apply_wandb_overrides(config, mock_run)
        self.assertEqual(new_config.seed, config.seed)

    def test_noop_when_wandb_run_none(self):
        config = _make_config(wandb_allow_config=True)
        new_config = apply_wandb_overrides(config, None)
        self.assertEqual(new_config.seed, config.seed)

    def test_override_shared_param(self):
        config = _make_config(wandb_allow_config=True)
        mock_run = MagicMock()
        mock_run.config = {"seed": 999}
        new_config = apply_wandb_overrides(config, mock_run)
        self.assertEqual(new_config.seed, 999)

    def test_override_sac_param(self):
        config = _make_config(algo="sac", wandb_allow_config=True)
        mock_run = MagicMock()
        mock_run.config = {"sac_learning_rate": 0.001}
        new_config = apply_wandb_overrides(config, mock_run)
        self.assertEqual(new_config.sac_learning_rate, 0.001)

    def test_override_algo_switches_algorithm(self):
        config = _make_config(algo="ppo", wandb_allow_config=True)
        mock_run = MagicMock()
        mock_run.config = {"algo": "sac", "sac_batch_size": 512}
        new_config = apply_wandb_overrides(config, mock_run)
        self.assertEqual(new_config.algo, "sac")
        self.assertEqual(new_config.sac_batch_size, 512)


if __name__ == "__main__":
    unittest.main()
