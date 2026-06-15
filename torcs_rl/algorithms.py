from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO, SAC, TD3

from .config import AppConfig

try:
    from rl_zoo3.utils import linear_schedule
except Exception:
    linear_schedule = None


def build_algorithm(config: AppConfig):
    algo = config.algo
    if algo == "ppo":
        if linear_schedule is not None:
            learning_rate = linear_schedule(3e-4)
            clip_range = linear_schedule(0.2)
        else:
            learning_rate = 3e-4
            clip_range = 0.2
        return PPO, {
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "learning_rate": learning_rate,
            "clip_range": clip_range,
            "policy_kwargs": {"net_arch": [64, 64]},
            "seed": int(config.seed),
            "verbose": 1,
        }
    if algo == "sac":
        return SAC, {
            "learning_rate": float(config.sac_learning_rate),
            "buffer_size": int(config.sac_buffer_size),
            "learning_starts": int(config.sac_learning_starts),
            "batch_size": int(config.sac_batch_size),
            "tau": float(config.sac_tau),
            "gamma": float(config.sac_gamma),
            "train_freq": int(config.sac_train_freq),
            "gradient_steps": int(config.sac_gradient_steps),
            "ent_coef": config.sac_ent_coef if isinstance(config.sac_ent_coef, str) else float(config.sac_ent_coef),
            "target_entropy": config.sac_target_entropy if isinstance(config.sac_target_entropy, str) else float(config.sac_target_entropy),
            "policy_kwargs": {"net_arch": list(config.sac_net_arch)},
            "seed": int(config.seed),
            "verbose": 1,
        }
    if algo == "td3":
        return TD3, {
            "learning_rate": 1e-3,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
            "policy_kwargs": {"net_arch": [256, 256]},
            "seed": int(config.seed),
            "verbose": 1,
        }
    raise ValueError(f"Unsupported algo: {config.algo}")
