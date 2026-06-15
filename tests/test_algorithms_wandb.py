"""Unit tests for torcs_rl.algorithms — W&B config & override logic."""

import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from torcs_rl.algorithms import (
    build_ppo_hyperparameters,
    build_sac_hyperparameters,
    build_td3_hyperparameters,
    build_wandb_run_config,
    apply_wandb_overrides,
    get_algorithm_registry,
    to_json_serializable,
)
from torcs_rl.config import (
    DEFAULT_ALGORITHM,
    DEFAULT_SEED,
    DEFAULT_TIMESTEPS,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_TRACK_WEIGHT,
    DEFAULT_HEADING_WEIGHT,
    DEFAULT_STEERING_WEIGHT,
    DEFAULT_TERMINAL_PENALTY,
    DEFAULT_GUI,
)


def _make_args(**overrides):
    defaults = dict(
        algo=DEFAULT_ALGORITHM,
        timesteps=DEFAULT_TIMESTEPS,
        run_name="test",
        gui=DEFAULT_GUI,
        seed=DEFAULT_SEED,
        eval_episodes=DEFAULT_EVAL_EPISODES,
        checkpoint_freq=DEFAULT_CHECKPOINT_FREQ,
        runtime_target="native",
        track_weight=DEFAULT_TRACK_WEIGHT,
        heading_weight=DEFAULT_HEADING_WEIGHT,
        steering_weight=DEFAULT_STEERING_WEIGHT,
        terminal_penalty=DEFAULT_TERMINAL_PENALTY,
        wandb_allow_config=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestHyperparameters(unittest.TestCase):
    def test_ppo_returns_dict(self):
        params = build_ppo_hyperparameters(42)
        self.assertIsInstance(params, dict)
        self.assertIn("n_steps", params)
        self.assertIn("learning_rate", params)
        self.assertEqual(params["seed"], 42)

    def test_sac_returns_dict(self):
        params = build_sac_hyperparameters(42)
        self.assertIn("buffer_size", params)
        self.assertIn("batch_size", params)
        self.assertEqual(params["seed"], 42)

    def test_td3_returns_dict(self):
        params = build_td3_hyperparameters(42)
        self.assertIn("policy_delay", params)
        self.assertEqual(params["seed"], 42)


class TestAlgorithmRegistry(unittest.TestCase):
    def test_all_algorithms_present(self):
        registry = get_algorithm_registry()
        for algo in ("ppo", "sac", "td3"):
            self.assertIn(algo, registry)
            model_cls, builder = registry[algo]
            self.assertTrue(callable(builder))


class TestToJsonSerializable(unittest.TestCase):
    def test_primitives(self):
        self.assertEqual(to_json_serializable(42), 42)
        self.assertEqual(to_json_serializable(3.14), 3.14)
        self.assertEqual(to_json_serializable("hi"), "hi")
        self.assertTrue(to_json_serializable(True))
        self.assertIsNone(to_json_serializable(None))

    def test_path(self):
        from pathlib import Path
        self.assertEqual(to_json_serializable(Path("a/b")), "a\\b")

    def test_callable(self):
        result = to_json_serializable(lambda: None)
        self.assertIsInstance(result, str)

    def test_dict(self):
        from pathlib import Path as P
        result = to_json_serializable({"a": 1, "b": P("x")})
        self.assertEqual(result["a"], 1)
        self.assertIsInstance(result["b"], str)


class TestBuildWandbRunConfig(unittest.TestCase):
    def test_contains_base_keys(self):
        args = _make_args()
        algo_hp = build_ppo_hyperparameters(42)
        config = build_wandb_run_config(args, algo_hp)
        self.assertEqual(config["algo"], args.algo)
        self.assertEqual(config["timesteps"], DEFAULT_TIMESTEPS)
        self.assertEqual(config["seed"], DEFAULT_SEED)

    def test_contains_direct_algo_keys(self):
        args = _make_args(algo="ppo")
        algo_hp = build_ppo_hyperparameters(42)
        config = build_wandb_run_config(args, algo_hp)
        self.assertIn("n_steps", config)
        self.assertIn("learning_rate", config)
        self.assertIn("batch_size", config)
        self.assertEqual(config["reward_weights"]["track_weight"], args.track_weight)

    def test_sac_keys(self):
        args = _make_args(algo="sac")
        algo_hp = build_sac_hyperparameters(42)
        config = build_wandb_run_config(args, algo_hp)
        self.assertIn("buffer_size", config)
        self.assertIn("learning_rate", config)


class TestApplyWandbOverrides(unittest.TestCase):
    def test_noop_when_allow_config_false(self):
        args = _make_args(wandb_allow_config=False)
        algo_hp = build_ppo_hyperparameters(42)
        mock_run = MagicMock()
        mock_run.config = {"learning_rate": 1e-3}
        new_args, new_hp = apply_wandb_overrides(args, algo_hp, mock_run)
        self.assertEqual(new_hp["learning_rate"], algo_hp["learning_rate"])

    def test_noop_when_wandb_run_none(self):
        args = _make_args(wandb_allow_config=True)
        algo_hp = build_ppo_hyperparameters(42)
        new_args, new_hp = apply_wandb_overrides(args, algo_hp, None)
        self.assertEqual(new_hp["learning_rate"], algo_hp["learning_rate"])

    def test_override_learning_rate(self):
        args = _make_args(algo="ppo", wandb_allow_config=True)
        algo_hp = build_ppo_hyperparameters(42)
        mock_run = MagicMock()
        mock_run.config = {"learning_rate": 0.001}
        new_args, new_hp = apply_wandb_overrides(args, algo_hp, mock_run)
        self.assertEqual(new_hp["learning_rate"], 0.001)

    def test_override_shared_param(self):
        args = _make_args(wandb_allow_config=True)
        algo_hp = build_ppo_hyperparameters(42)
        mock_run = MagicMock()
        mock_run.config = {"seed": 999}
        new_args, new_hp = apply_wandb_overrides(args, algo_hp, mock_run)
        self.assertEqual(new_args.seed, 999)

    def test_override_algo_switches_algorithm(self):
        args = _make_args(algo="ppo", wandb_allow_config=True, wandb_project="proj")
        algo_hp = build_ppo_hyperparameters(42)
        mock_run = MagicMock()
        mock_run.config = {"algo": "sac"}
        new_args, new_hp = apply_wandb_overrides(args, algo_hp, mock_run)
        self.assertEqual(new_args.algo, "sac")
        # SAC hyperparams should be returned
        self.assertIn("buffer_size", new_hp)
        self.assertNotIn("n_steps", new_hp)


if __name__ == "__main__":
    unittest.main()
