import gymnasium as gym
import numpy as np
from gymnasium import spaces

MONITOR_INFO_KEYS = (
    "laps_completed",
    "last_lap_time",
    "progress_distance",
    "distance_raced",
    "mean_throttle",
    "mean_brake",
    "mean_speed",
    "max_speed",
    "termination_reason",
)


class TorcsRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        client,
        track_weight=0.3,
        heading_weight=0.15,
        steering_weight=0.01,
        speed_weight=0.05,
        terminal_penalty=100.0,
        max_episode_steps=4000,
        low_progress_steps=150,
        low_progress_threshold=10.0,
    ):
        self.client = client

        self.track_weight = float(track_weight)
        self.heading_weight = float(heading_weight)
        self.steering_weight = float(steering_weight)
        self.speed_weight = float(speed_weight)
        self.terminal_penalty = float(terminal_penalty)

        self.max_episode_steps = int(max_episode_steps)
        self.low_progress_steps = int(low_progress_steps)
        self.low_progress_threshold = float(low_progress_threshold)

        self.speed_scale = 300.0
        self.track_scale = 200.0
        self.rolling_speed = 15.0
        self.min_speed_threshold = 30.0
        self.min_speed_grace = 200

        self.client.get_servers_input()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        initial_obs = self._flatten_observation(self.client.S.d)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_obs.shape,
            dtype=np.float32,
        )

        self.initial_reset = True
        self.time_step = 0
        self._reset_episode_metrics()
        self._initialize_progress(self.client.S.d)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self._reset_episode_metrics()

        if not self.initial_reset:
            self.client.reset_episode()

        self.client.did_crash = False
        self._initialize_progress(self.client.S.d)
        self.initial_reset = False
        return self._flatten_observation(self.client.S.d), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        torcs_action = self._agent_to_torcs(action)
        self._apply_action(torcs_action)

        self.client.respond_to_server()
        self.client.get_servers_input()

        if self.client.did_crash:
            reward = -self.terminal_penalty
            obs = self._flatten_observation(self.client.S.d)
            info = {
                "raw_obs": self.client.S.d,
                "torcs_action": torcs_action,
                "lap_completed": False,
                "completed_lap_time": float("nan"),
                "current_lap_time": 0.0,
                "laps_completed": int(self.laps_completed),
                "last_lap_time": float(self.last_lap_time) if self.last_lap_time is not None else float("nan"),
                "progress_distance": float(self.progress_distance),
                "distance_raced": 0.0,
                "mean_throttle": 0.0,
                "mean_brake": 0.0,
                "mean_speed": 0.0,
                "max_speed": float(self.max_speed),
                "termination_reason": "torcs_crash",
            }
            return obs, reward, True, False, info

        raw_obs = self.client.S.d
        self.time_step += 1

        speed_x = float(raw_obs["speedX"])
        self.throttle_sum += torcs_action["accel"]
        self.brake_sum += torcs_action["brake"]
        self.speed_sum += speed_x
        self.max_speed = max(self.max_speed, speed_x)

        lap_completed, completed_lap_time = self._update_laps(raw_obs)
        reward, terminated, termination_reason = self._compute_reward(raw_obs, torcs_action["steer"])
        truncated = self.time_step >= self.max_episode_steps
        if truncated and not terminated:
            termination_reason = "max_steps"

        info = {
            "raw_obs": raw_obs,
            "torcs_action": torcs_action,
            "lap_completed": lap_completed,
            "completed_lap_time": completed_lap_time,
            "current_lap_time": float(raw_obs["curLapTime"]),
            "laps_completed": int(self.laps_completed),
            "last_lap_time": float(self.last_lap_time) if self.last_lap_time is not None else float("nan"),
            "progress_distance": float(self.progress_distance),
            "distance_raced": float(raw_obs["distRaced"]),
            "mean_throttle": self.throttle_sum / max(self.time_step, 1),
            "mean_brake": self.brake_sum / max(self.time_step, 1),
            "mean_speed": self.speed_sum / max(self.time_step, 1),
            "max_speed": float(self.max_speed),
            "termination_reason": termination_reason,
        }
        return self._flatten_observation(raw_obs), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass  # client lifecycle managed by caller

    # -- action helpers ----------------------------------------------------

    def _agent_to_torcs(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        pedal = float(np.clip(action[1], -1.0, 1.0))
        accel = max(pedal, 0.0)
        brake = max(-pedal, 0.0)
        return {"steer": steer, "accel": accel, "brake": brake}

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

    def _apply_action(self, torcs_action):
        action_torcs = self.client.R.d
        speed_x = float(self.client.S.d["speedX"])
        action_torcs["steer"] = torcs_action["steer"]
        action_torcs["accel"] = torcs_action["accel"]
        action_torcs["brake"] = 0.0 if speed_x < self.rolling_speed else torcs_action["brake"]
        action_torcs["gear"] = self._hardcoded_gear(speed_x)

    # -- reward ------------------------------------------------------------

    def _compute_reward(self, raw_obs, steer):
        speed_x = float(raw_obs.get("speedX", 0.0))
        track_pos = abs(float(raw_obs["trackPos"]))
        angle = abs(float(raw_obs["angle"]))
        progress_now = float(raw_obs["distRaced"])
        progress_delta = float(np.clip(progress_now - self.prev_dist_raced, -5.0, 5.0))
        self.prev_dist_raced = progress_now
        self.progress_distance = max(progress_now - self.start_dist_raced, 0.0)

        if speed_x >= self.min_speed_threshold:
            self.speed_went_above_threshold = True
            self.steps_below_min_speed = 0
        elif self.speed_went_above_threshold:
            self.steps_below_min_speed += 1

        cos_angle = np.cos(float(raw_obs["angle"]))
        forward_speed = speed_x * max(cos_angle, 0.0)
        speed_reward = self.speed_weight * (forward_speed / self.speed_scale)

        reward = progress_delta
        reward += speed_reward
        reward -= self.track_weight * track_pos
        reward -= self.heading_weight * angle
        reward -= self.steering_weight * abs(steer - self.prev_steer)

        terminated = False
        termination_reason = "in_progress"
        track = np.asarray(raw_obs["track"], dtype=np.float32)
        track_min = float(track.min())
        if track_pos > 1.0:
            terminated = True
            termination_reason = "off_track"
        elif track_min <= 0.0:
            terminated = True
            termination_reason = "collision"
        elif np.cos(float(raw_obs["angle"])) < 0.0:
            terminated = True
            termination_reason = "wrong_direction"
        elif self.time_step >= self.low_progress_steps:
            window_progress = progress_now - self.progress_history[-self.low_progress_steps]
            if window_progress < self.low_progress_threshold:
                terminated = True
                termination_reason = "low_progress"
        elif self.speed_went_above_threshold and self.steps_below_min_speed >= self.min_speed_grace:
            terminated = True
            termination_reason = "low_speed"

        if terminated:
            reward -= self.terminal_penalty

        self.prev_steer = steer
        self.progress_history.append(progress_now)
        return reward, terminated, termination_reason

    # -- episode bookkeeping -----------------------------------------------

    def _reset_episode_metrics(self):
        self.laps_completed = 0
        self.last_lap_time = None
        self.progress_distance = 0.0
        self.prev_dist_raced = 0.0
        self.start_dist_raced = 0.0
        self.last_lap_marker = 0.0
        self.progress_history = []
        self.throttle_sum = 0.0
        self.brake_sum = 0.0
        self.speed_sum = 0.0
        self.max_speed = 0.0
        self.prev_steer = 0.0
        self.speed_went_above_threshold = False
        self.steps_below_min_speed = 0

    def _initialize_progress(self, raw_obs):
        total_dist = float(raw_obs.get("distRaced", 0.0))
        self.prev_dist_raced = total_dist
        self.start_dist_raced = total_dist
        self.last_lap_marker = float(raw_obs.get("lastLapTime", 0.0))
        self.progress_history = [total_dist]

    def _update_laps(self, raw_obs):
        last_lap_time = float(raw_obs["lastLapTime"])
        if last_lap_time > 0.0 and not np.isclose(last_lap_time, self.last_lap_marker, atol=1e-3, rtol=1e-4):
            self.last_lap_marker = last_lap_time
            self.laps_completed += 1
            self.last_lap_time = last_lap_time
            return True, last_lap_time
        if last_lap_time > 0.0:
            self.last_lap_marker = last_lap_time
        return False, float("nan")

    # -- observation -------------------------------------------------------

    def _flatten_observation(self, raw_obs):
        return np.concatenate(
            (
                np.asarray([float(raw_obs["speedX"]) / self.speed_scale], dtype=np.float32),
                np.asarray([float(raw_obs["trackPos"])], dtype=np.float32),
                np.asarray([float(raw_obs["angle"]) / np.pi], dtype=np.float32),
                np.asarray(raw_obs["track"], dtype=np.float32) / self.track_scale,
            ),
            axis=0,
        ).astype(np.float32)
