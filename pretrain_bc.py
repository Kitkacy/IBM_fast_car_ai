import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC, TD3
from torch.utils.data import DataLoader, TensorDataset

from torcs_rl.algorithms import build_sac_hyperparameters, build_td3_hyperparameters


ALGORITHMS = {
    "sac": (SAC, build_sac_hyperparameters),
    "td3": (TD3, build_td3_hyperparameters),
}


class StaticTorcsSpaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-clone an SB3 actor from teacher TORCS data.")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--dataset", type=Path, default=Path("data/corkscrew_teacher.npz"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.dataset)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)

    if observations.ndim != 2:
        raise ValueError(f"Expected observations with shape [N, obs_dim], got {observations.shape}")
    if actions.shape != (observations.shape[0], 3):
        raise ValueError(f"Expected actions with shape [{observations.shape[0]}, 3], got {actions.shape}")

    algorithm_name = str(args.algorithm).lower()
    model_cls, hyperparam_builder = ALGORITHMS[algorithm_name]
    run_name = args.run_name or f"{algorithm_name}-full-bc-corkscrew"

    run_dir = args.log_dir / run_name
    model_dir = run_dir / "models"
    tb_dir = run_dir / "tensorboard"
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    env = StaticTorcsSpaceEnv(obs_dim=observations.shape[1])
    hyperparams = hyperparam_builder(seed=args.seed)
    hyperparams["learning_rate"] = args.learning_rate

    model = model_cls("MlpPolicy", env, tensorboard_log=str(tb_dir), device=args.device, **hyperparams)
    device = model.device

    scaled_actions = model.policy.scale_action(actions).astype(np.float32)
    dataset = TensorDataset(
        th.as_tensor(observations, dtype=th.float32),
        th.as_tensor(scaled_actions, dtype=th.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = model.policy.actor.optimizer
    model.policy.actor.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_count = 0
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            if algorithm_name == "sac":
                pred_actions = model.policy.actor(batch_obs, deterministic=True)
            else:
                pred_actions = model.policy.actor(batch_obs)
            loss = F.mse_loss(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.actor.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = int(batch_obs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        mean_loss = total_loss / max(1, total_count)
        print(f"epoch={epoch:03d} loss={mean_loss:.8f}")

    latest_path = model_dir / f"{algorithm_name}_torcs_latest.zip"
    final_path = model_dir / f"{algorithm_name}_torcs_bc.zip"
    model.save(str(latest_path))
    model.save(str(final_path))

    summary = {
        "dataset": str(args.dataset),
        "samples": int(observations.shape[0]),
        "obs_dim": int(observations.shape[1]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "algorithm": algorithm_name,
        "latest_model_path": str(latest_path),
        "final_model_path": str(final_path),
    }
    (run_dir / "bc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
