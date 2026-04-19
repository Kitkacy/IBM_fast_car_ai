import argparse
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, SAC, TD3

try:
    from rl_zoo3.utils import linear_schedule
except Exception:
    linear_schedule = None


def build_ppo_hyperparameters(seed):
    if linear_schedule is not None:
        learning_rate = linear_schedule(3e-4)
        clip_range = linear_schedule(0.2)
    else:
        learning_rate = 3e-4
        clip_range = 0.2

    return {
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
        "seed": int(seed),
        "verbose": 1,
    }


def build_sac_hyperparameters(seed):
    return {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_kwargs": {"net_arch": [256, 256]},
        "seed": int(seed),
        "verbose": 1,
    }


def build_td3_hyperparameters(seed):
    return {
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
        "seed": int(seed),
        "verbose": 1,
    }


def get_algorithm_registry():
    return {
        "ppo": (PPO, build_ppo_hyperparameters),
        "sac": (SAC, build_sac_hyperparameters),
        "td3": (TD3, build_td3_hyperparameters),
    }


def to_json_serializable(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_serializable(v) for v in value]
    if callable(value):
        return str(value)
    return str(value)


def build_wandb_run_config(args, algo_hyperparameters):
    base = {
        "algo": args.algo,
        "timesteps": int(args.timesteps),
        "run_name": args.run_name,
        "target_speed": float(args.target_speed),
        "vision": bool(args.vision),
        "seed": int(args.seed),
        "eval_episodes": int(args.eval_episodes),
        "checkpoint_freq": int(args.checkpoint_freq),
        "runtime_target": str(args.runtime_target),
        "train_port": int(args.train_port),
        "eval_port": int(args.eval_port),
        "algorithm_hyperparameters": to_json_serializable(algo_hyperparameters),
    }
    for key, value in algo_hyperparameters.items():
        base[f"{args.algo}_{key}"] = to_json_serializable(value)
    return base


def apply_wandb_overrides(args, algo_hyperparameters, wandb_run):
    if wandb_run is None or not args.wandb_allow_config:
        return args, algo_hyperparameters

    mutable = vars(args).copy()
    for key, value in dict(wandb_run.config).items():
        if key in mutable and value is not None:
            mutable[key] = value

    algo = str(mutable.get("algo", args.algo)).lower()
    algorithm_registry = get_algorithm_registry()
    if algo not in algorithm_registry:
        raise ValueError(f"Unsupported algo from wandb config: {algo}")
    mutable["algo"] = algo

    _, kwargs_builder = algorithm_registry[algo]
    merged_algo_kwargs = kwargs_builder(seed=mutable.get("seed", args.seed))

    for key, value in dict(wandb_run.config).items():
        if key in merged_algo_kwargs and value is not None:
            merged_algo_kwargs[key] = value
            continue
        prefixed = f"{algo}_"
        if key.startswith(prefixed):
            raw_key = key[len(prefixed):]
            if raw_key in merged_algo_kwargs and value is not None:
                merged_algo_kwargs[raw_key] = value

    return argparse.Namespace(**mutable), merged_algo_kwargs
