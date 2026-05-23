import copy
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import DEFAULT_RUNTIME_TARGET, DEFAULT_TRAIN_PORT
from .runtime import create_runtime_backend


class TorcsRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        vision=False,
        target_speed=60.0,
        torcs_port=DEFAULT_TRAIN_PORT,
        obs_norm=True,
        obs_norm_clip=5.0,
        obs_norm_eps=1e-6,
        runtime_backend=None,
        runtime_target=DEFAULT_RUNTIME_TARGET,
        torcs_exe=None,
        race_config="__default__",
        launch_log="torcs_launcher.log",
    ):
        self.runtime_backend = runtime_backend or create_runtime_backend(runtime_target)
        self.runtime_target = self.runtime_backend.name

        self.vision = bool(vision)
        self.target_speed = float(target_speed)
        self.default_speed = 50.0
        self.terminal_judge_start = 500
        self.termination_limit_progress = 5.0

        self.obs_norm = bool(obs_norm)
        self.obs_norm_clip = float(obs_norm_clip)
        self.obs_norm_eps = float(obs_norm_eps)

        self.torcs_port = int(torcs_port)
        selected_torcs_exe = torcs_exe or self.runtime_backend.default_torcs_exe
        self.torcs_exe = str(Path(selected_torcs_exe).expanduser()) if selected_torcs_exe else ""

        if race_config == "__default__":
            selected_race_config = self.runtime_backend.default_race_config
        else:
            selected_race_config = race_config
        self.race_config = str(Path(selected_race_config).expanduser()) if selected_race_config else ""

        self.launch_log = Path(__file__).resolve().parent.parent / launch_log
        self.autostart_script = str(self.runtime_backend.default_autostart_script)
        self.torcs_pid = None

        self.initial_reset = True
        self.time_step = 0

        self.client = self.runtime_backend.create_client(
            port=self.torcs_port,
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

        self.client = self.runtime_backend.create_client(
            port=self.torcs_port,
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
        self._stop_torcs_process()

    def reset_torcs(self):
        self._start_torcs_process()
        time.sleep(0.5)

    def _stop_torcs_process(self):
        if self.torcs_pid is None:
            return

        self.runtime_backend.kill_process(self.torcs_pid, log_file=str(self.launch_log), port=self.torcs_port)
        self.torcs_pid = None

    def _start_torcs_process(self):
        self.torcs_pid = self.runtime_backend.launch_process(
            torcs_exe=self.torcs_exe,
            port=self.torcs_port,
            vision=self.vision,
            race_config=self.race_config,
            log_file=str(self.launch_log),
            autostart_script=self.autostart_script,
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
            base.append((np.asarray(raw_obs["img"], dtype=np.float32) / 255.0).flatten())
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
