import copy
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import snakeoil3_gym_windows as snakeoil3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from torcs_process_manager import kill_torcs_process, launch_torcs_process

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except Exception:
    wandb = None
    WandbCallback = None

try:
    from rl_zoo3.utils import linear_schedule
except Exception:
    linear_schedule = None


class TorcsEnvWindowsRL(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        vision=False,
        target_speed=60.0,
        port=3001,
        obs_norm=True,
        obs_norm_clip=5.0,
        obs_norm_eps=1e-6,
        torcs_exe=r"C:\Users\szymo\source\repos\torcs\torcs\wtorcs.exe",
        race_config=r"C:\Users\szymo\source\repos\torcs\torcs\config\raceman\practice.xml",
        launch_log="torcs_launcher.log",
    ):
        self.vision = bool(vision)
        self.target_speed = float(target_speed)
        self.default_speed = 50.0
        self.terminal_judge_start = 500
        self.termination_limit_progress = 5.0

        self.obs_norm = bool(obs_norm)
        self.obs_norm_clip = float(obs_norm_clip)
        self.obs_norm_eps = float(obs_norm_eps)

        self.port = int(port)
        self.torcs_exe = str(Path(torcs_exe).expanduser())
        self.race_config = str(Path(race_config).expanduser()) if race_config else ""
        self.launch_log = Path(__file__).resolve().parent / launch_log
        self.autostart_script = Path(__file__).resolve().parent / "autostart_windows.ps1"
        self.torcs_pid = None

        self.initial_reset = True
        self.time_step = 0

        self.client = snakeoil3.Client(
            p=self.port,
            vision=self.vision,
            torcs_exe=self.torcs_exe,
            launch_log=str(self.launch_log),
            race_config=self.race_config,
        )
        self.torcs_pid = getattr(self.client, "torcs_pid", None)
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()

        initial_obs_raw = self._flatten_observation(self.client.S.d)
        self.obs_dim = int(initial_obs_raw.shape[0])

        self._obs_count = 0
        self._obs_mean = np.zeros(self.obs_dim, dtype=np.float64)
        self._obs_m2 = np.zeros(self.obs_dim, dtype=np.float64)

        initial_obs = self._normalize_observation(initial_obs_raw, update=True)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self._last_obs = initial_obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0

        if not self.initial_reset:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

        self.client = snakeoil3.Client(
            p=self.port,
            vision=self.vision,
            torcs_exe=self.torcs_exe,
            launch_log=str(self.launch_log),
            race_config=self.race_config,
        )
        self.torcs_pid = getattr(self.client, "torcs_pid", None)
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()

        obs_raw = self._flatten_observation(self.client.S.d)
        obs = self._normalize_observation(obs_raw, update=True)
        self._last_obs = obs
        self.initial_reset = False
        return obs, {}

    def step(self, action=None):
        client = self.client
        obs_pre = copy.deepcopy(client.S.d)

        if action is None:
            steer_action = self.action_space.sample()
        else:
            steer_action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)

        torcs_action = self._agent_to_torcs(steer_action)
        self._apply_action(client, torcs_action)

        client.respond_to_server()
        client.get_servers_input()

        raw_obs = client.S.d
        obs_raw = self._flatten_observation(raw_obs)
        obs = self._normalize_observation(obs_raw, update=True)
        reward, terminated, termination_reason = self._compute_reward(raw_obs, obs_pre)
        truncated = False

        if client.R.d["meta"] is True:
            client.respond_to_server()

        self._last_obs = obs
        self.time_step += 1

        info = {
            "raw_obs": raw_obs,
            "torcs_action": torcs_action,
            "termination_reason": termination_reason,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.end()

    def end(self):
        self._kill_torcs()

    def reset_torcs(self):
        self._launch_torcs()
        time.sleep(0.5)

    def _kill_torcs(self):
        if self.torcs_pid is None:
            return

        kill_torcs_process(self.torcs_pid, log_file=str(self.launch_log), port=self.port)
        self.torcs_pid = None

    def _launch_torcs(self):
        self.torcs_pid = launch_torcs_process(
            torcs_exe=self.torcs_exe,
            port=self.port,
            vision=self.vision,
            race_config=self.race_config,
            log_file=str(self.launch_log),
            autostart_script=str(self.autostart_script),
            existing_pid=self.torcs_pid,
        )

    def _agent_to_torcs(self, action):
        steer = float(np.clip(np.asarray(action, dtype=np.float32)[0], -1.0, 1.0))
        return {"steer": steer}

    def _hardcoded_gear(self, speed_x):
        if speed_x < 30.0:
            return 1
        if speed_x < 55.0:
            return 2
        if speed_x < 85.0:
            return 3
        if speed_x < 115.0:
            return 4
        if speed_x < 145.0:
            return 5
        return 6

    def _apply_action(self, client, torcs_action):
        action_torcs = client.R.d
        action_torcs["steer"] = torcs_action["steer"]

        speed_x = float(client.S.d["speedX"])
        if speed_x < self.target_speed - 2.0:
            action_torcs["accel"] = 0.3
        elif speed_x > self.target_speed + 2.0:
            action_torcs["accel"] = 0.0
        else:
            action_torcs["accel"] = 0.1

        action_torcs["gear"] = self._hardcoded_gear(speed_x)

    def _compute_reward(self, obs, obs_pre):
        reward = float(obs["speedX"] * np.cos(obs["angle"]))
        terminated = False
        termination_reason = "in_progress"

        if obs["damage"] - obs_pre["damage"] > 0:
            reward = -10.0

        track = np.asarray(obs["track"], dtype=np.float32)
        progress = float(obs["speedX"] * np.cos(obs["angle"]))

        if track.min() < 0:
            reward = -10.0
            terminated = True
            termination_reason = "off_track"
            self.client.R.d["meta"] = True

        if self.terminal_judge_start < self.time_step and progress < self.termination_limit_progress:
            terminated = True
            termination_reason = "low_progress"
            self.client.R.d["meta"] = True

        if np.cos(obs["angle"]) < 0:
            terminated = True
            termination_reason = "wrong_direction"
            self.client.R.d["meta"] = True

        return reward, terminated, termination_reason

    def _flatten_observation(self, raw_obs):
        base = [
            np.asarray([raw_obs["speedX"]], dtype=np.float32) / self.default_speed,
            np.asarray([raw_obs["trackPos"]], dtype=np.float32),
            np.asarray([raw_obs["angle"]], dtype=np.float32) / np.pi,
            np.asarray(raw_obs["track"], dtype=np.float32) / 200.0,
        ]
        if self.vision:
            base.append(np.asarray(raw_obs["img"], dtype=np.float32) / 255.0)
        return np.concatenate(base, axis=0).astype(np.float32)

    def _update_obs_stats(self, obs_tensor):
        self._obs_count += 1
        if self._obs_count == 1:
            self._obs_mean[...] = obs_tensor
            self._obs_m2.fill(0.0)
            return

        delta = obs_tensor - self._obs_mean
        self._obs_mean += delta / float(self._obs_count)
        delta2 = obs_tensor - self._obs_mean
        self._obs_m2 += delta * delta2

    def _normalize_observation(self, obs, update=True):
        obs_tensor = np.asarray(obs, dtype=np.float32)
        if not self.obs_norm:
            return obs_tensor.astype(np.float32)

        if update:
            self._update_obs_stats(obs_tensor)

        if self._obs_count < 2:
            clipped = np.clip(obs_tensor, -self.obs_norm_clip, self.obs_norm_clip)
            return clipped.astype(np.float32)

        var = self._obs_m2 / float(self._obs_count - 1)
        std = np.sqrt(np.maximum(var, self.obs_norm_eps))
        normalized = (obs_tensor - self._obs_mean) / std
        clipped = np.clip(normalized, -self.obs_norm_clip, self.obs_norm_clip)
        return clipped.astype(np.float32)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate TORCS with SB3 + TensorBoard/WandB")
    parser.add_argument("--timesteps", type=int, default=6_000, help="Total PPO training steps")
    parser.add_argument("--run-name", type=str, default="torcs-ppo", help="Run name for logs and WandB")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"), help="Base output directory")
    parser.add_argument("--target-speed", type=float, default=100.0, help="Target speed controller setpoint")
    parser.add_argument("--vision", action="store_true", help="Enable TORCS vision observations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Episodes for final evaluation")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gym-torcs", help="WandB project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity/team")
    parser.add_argument("--checkpoint-freq", type=int, default=2_000, help="Checkpoint frequency in env steps")
    return parser.parse_args()


def _build_ppo_kwargs(seed):
    # RL Zoo style defaults for PPO on continuous control tasks.
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


def train_and_evaluate(args):
    run_dir = args.log_dir / args.run_name
    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    model_dir = run_dir / "models"
    for directory in (tb_dir, ckpt_dir, eval_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    #race_config = Path(r"C:\Users\szymo\source\repos\torcs\torcs\config\raceman\practice.xml")
    race_config = None

    train_env = VecMonitor(
        DummyVecEnv(
            [
                lambda: Monitor(
                    TorcsEnvWindowsRL(
                        vision=args.vision,
                        target_speed=args.target_speed,
                        port=3001,
                        race_config=str(race_config) if race_config else None,
                    )
                )
            ]
        ),
        filename=str(run_dir / "train_monitor.csv"),
    )
    # Using a separate env for periodic EvalCallback would relaunch TORCS and
    # kill the train process (start script taskkills existing wtorcs). Reuse
    # train_env for callback-time eval and keep separate env only for final eval.

    eval_race_config = Path(r"C:\Users\szymo\source\repos\torcs\torcs\config\raceman\practice_2.xml")
    eval_race_config = None
    eval_env = VecMonitor(
        DummyVecEnv(
            [
                lambda: Monitor(
                    TorcsEnvWindowsRL(
                        vision=args.vision,
                        target_speed=args.target_speed,
                        port=3001,
                        race_config=str(eval_race_config) if eval_race_config else None,
                    )
                )
            ]
        ),
        filename=str(run_dir / "eval_monitor.csv"),
    )
    
    eval_env = train_env
    
    ppo_kwargs = _build_ppo_kwargs(seed=args.seed)
    latest_model_path = model_dir / "ppo_torcs_latest.zip"
    best_model_path = model_dir / "best_model.zip"
    resumed_from = None
    if latest_model_path.exists():
        model = PPO.load(str(latest_model_path), env=train_env)
        resumed_from = str(latest_model_path)
        print(f"[Model] Resumed from latest model: {latest_model_path}")
    else:
        model = PPO("MlpPolicy", train_env, tensorboard_log=str(tb_dir), **ppo_kwargs)
        print("[Model] No existing checkpoint found, starting fresh")

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq),
            save_path=str(ckpt_dir),
            name_prefix="ppo_torcs",
            save_vecnormalize=True,


        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(eval_dir),
            eval_freq=max(1, args.checkpoint_freq),
            deterministic=True,
            render=False,
        ),
    ]

    wandb_run = None
    if args.wandb:
        if wandb is None or WandbCallback is None:
            raise RuntimeError("WandB integration requested but wandb is not installed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            config={"timesteps": args.timesteps, "target_speed": args.target_speed, "vision": args.vision, **ppo_kwargs},
        )
        callbacks.append(WandbCallback(gradient_save_freq=0, model_save_path=str(model_dir), verbose=1))

    try:
        interrupted = False
        try:
            model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), progress_bar=True)
        except KeyboardInterrupt:
            interrupted = True
            print("[Training Interrupted] KeyboardInterrupt received, saving latest checkpoint...")

        model.save(str(latest_model_path))

        if interrupted:
            summary = {
                "interrupted": True,
                "timesteps": int(args.timesteps),
                "latest_model_path": str(latest_model_path),
                "resumed_from": resumed_from,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("[Training Interrupted]", json.dumps(summary))
            return

        model_path = model_dir / "ppo_torcs_final"
        model.save(str(model_path))

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)
        summary = {
            "interrupted": False,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "timesteps": int(args.timesteps),
            "model_path": str(model_path),
            "latest_model_path": str(latest_model_path),
            "resumed_from": resumed_from,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[Training Complete]", json.dumps(summary))
    finally:
        train_env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    train_and_evaluate(_parse_args())
