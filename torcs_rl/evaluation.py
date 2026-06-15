import os
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

from .utils import safe_float, write_json, append_csv_row

try:
    import wandb
except Exception:
    wandb = None


def _learning_status(model):
    learning_starts = getattr(model, "learning_starts", None)
    if learning_starts is None:
        return None, True
    learning_starts = int(learning_starts)
    return learning_starts, int(getattr(model, "num_timesteps", 0)) > learning_starts


def evaluate_with_lap_stats(
    model,
    env,
    *,
    n_eval_episodes,
    deterministic,
    render,
    warn,
    step_callback=None,
    phase="eval",
    lap_events_path=None,
):
    episode_laps = []
    timed_lap_times = []
    episode_metrics = []
    episode_index = 0

    def _callback(locals_, _globals):
        nonlocal episode_index
        info = locals_["info"]

        if step_callback is not None:
            step_callback(locals_, _globals)

        completed_lap_time = safe_float(info.get("completed_lap_time"))
        if info.get("lap_completed") and completed_lap_time is not None:
            warmup = len(episode_laps) == 0
            episode_laps.append(completed_lap_time)
            if not warmup:
                timed_lap_times.append(completed_lap_time)
            if lap_events_path is not None:
                append_csv_row(
                    lap_events_path,
                    (
                        "phase",
                        "timesteps",
                        "episode",
                        "lap_time",
                        "warmup",
                        "laps_completed",
                        "progress_distance",
                        "distance_raced",
                    ),
                    {
                        "phase": phase,
                        "timesteps": int(getattr(model, "num_timesteps", 0)),
                        "episode": episode_index,
                        "lap_time": completed_lap_time,
                        "warmup": int(warmup),
                        "laps_completed": int(info.get("laps_completed", 0)),
                        "progress_distance": safe_float(info.get("progress_distance")),
                        "distance_raced": safe_float(info.get("distance_raced")),
                    },
                )

        if locals_["done"] and "episode" in info:
            timed_episode_laps = episode_laps[1:]
            episode_metrics.append(
                {
                    "reward": float(info["episode"]["r"]),
                    "length": int(info["episode"]["l"]),
                    "laps_completed": int(info["episode"].get("laps_completed", 0) or 0),
                    "timed_laps": int(len(timed_episode_laps)),
                    "last_lap_time": safe_float(info["episode"].get("last_lap_time")),
                    "median_lap_time": (
                        float(np.median(timed_episode_laps)) if timed_episode_laps else None
                    ),
                    "progress_distance": safe_float(info["episode"].get("progress_distance")),
                    "distance_raced": safe_float(info["episode"].get("distance_raced")),
                    "mean_throttle": safe_float(info["episode"].get("mean_throttle")),
                    "mean_brake": safe_float(info["episode"].get("mean_brake")),
                    "mean_speed": safe_float(info["episode"].get("mean_speed")),
                    "max_speed": safe_float(info["episode"].get("max_speed")),
                    "termination_reason": info["episode"].get("termination_reason", "unknown"),
                }
            )
            episode_index += 1
            episode_laps.clear()

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        warn=warn,
        callback=_callback,
        return_episode_rewards=True,
    )

    eligible_episodes = sum(1 for item in episode_metrics if item["timed_laps"] > 0)
    learning_starts, learning_started = _learning_status(model)
    mean_progress = float(np.mean([item["progress_distance"] or 0.0 for item in episode_metrics])) if episode_metrics else 0.0
    mean_distance = float(np.mean([item["distance_raced"] or 0.0 for item in episode_metrics])) if episode_metrics else 0.0
    max_speed = float(np.max([item["max_speed"] or 0.0 for item in episode_metrics])) if episode_metrics else 0.0
    low_movement = mean_distance < 10.0 or max_speed < 5.0
    off_track_rate = (
        float(sum(1 for item in episode_metrics if item["termination_reason"] == "off_track")) / float(len(episode_metrics))
        if episode_metrics
        else 0.0
    )
    mean_completed_laps = (
        float(sum(item["laps_completed"] for item in episode_metrics)) / float(len(episode_metrics))
        if episode_metrics
        else 0.0
    )
    lap_stats = {
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_mean_ep_length": float(np.mean(episode_lengths)),
        "eval_completed_laps": int(sum(item["laps_completed"] for item in episode_metrics)),
        "eval_lap_completion_rate": (
            float(sum(1 for item in episode_metrics if item["laps_completed"] > 0)) / float(len(episode_metrics))
            if episode_metrics
            else 0.0
        ),
        "eval_median_lap_time": float(np.median(timed_lap_times)) if timed_lap_times else None,
        "eval_best_lap_time": float(np.min(timed_lap_times)) if timed_lap_times else None,
        "eval_lap_time_iqr": (
            float(np.percentile(timed_lap_times, 75) - np.percentile(timed_lap_times, 25))
            if len(timed_lap_times) >= 2
            else None
        ),
        "eval_timed_laps": int(len(timed_lap_times)),
        "eval_timed_lap_completion_rate": (
            float(eligible_episodes) / float(len(episode_metrics)) if episode_metrics else 0.0
        ),
        "eval_off_track_rate": off_track_rate,
        "eval_mean_progress": mean_progress,
        "eval_progress_score": mean_progress * (1.0 - 0.5 * off_track_rate) + 10000.0 * mean_completed_laps,
        "eval_mean_distance_raced": mean_distance,
        "eval_mean_throttle": float(np.mean([item["mean_throttle"] or 0.0 for item in episode_metrics])) if episode_metrics else 0.0,
        "eval_mean_brake": float(np.mean([item["mean_brake"] or 0.0 for item in episode_metrics])) if episode_metrics else 0.0,
        "eval_mean_speed": float(np.mean([item["mean_speed"] or 0.0 for item in episode_metrics])) if episode_metrics else 0.0,
        "eval_max_speed": max_speed,
        "eval_low_movement_warning": bool(low_movement),
        "learning_starts": learning_starts,
        "learning_started": bool(learning_started),
    }
    return episode_rewards, episode_lengths, lap_stats, episode_metrics


class LapLoggerCallback(BaseCallback):
    def __init__(self, *, prefix, lap_events_path=None, verbose=0):
        super().__init__(verbose=verbose)
        self.prefix = prefix
        self.lap_events_path = Path(lap_events_path) if lap_events_path else None

    def _on_step(self):
        for info in self.locals.get("infos", []):
            completed_lap_time = safe_float(info.get("completed_lap_time"))
            if info.get("lap_completed") and completed_lap_time is not None:
                self.logger.record(f"{self.prefix}/lap_time", completed_lap_time)
                self.logger.record(f"{self.prefix}/laps_completed", float(info.get("laps_completed", 0)))
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log(
                        {
                            f"{self.prefix}/lap_time": completed_lap_time,
                            f"{self.prefix}/laps_completed": float(info.get("laps_completed", 0)),
                        },
                        step=int(self.num_timesteps),
                    )
                if self.lap_events_path is not None:
                    append_csv_row(
                        self.lap_events_path,
                        (
                            "phase",
                            "timesteps",
                            "episode",
                            "lap_time",
                            "warmup",
                            "laps_completed",
                            "progress_distance",
                            "distance_raced",
                        ),
                        {
                            "phase": self.prefix,
                            "timesteps": int(self.num_timesteps),
                            "episode": "",
                            "lap_time": completed_lap_time,
                            "warmup": 0,
                            "laps_completed": int(info.get("laps_completed", 0)),
                            "progress_distance": safe_float(info.get("progress_distance")),
                            "distance_raced": safe_float(info.get("distance_raced")),
                        },
                    )

            if "episode" not in info:
                continue

            self.logger.record(f"{self.prefix}/episode_reward", float(info["episode"]["r"]))
            self.logger.record(f"{self.prefix}/episode_length", float(info["episode"]["l"]))
            wandb_payload = {
                f"{self.prefix}/episode_reward": float(info["episode"]["r"]),
                f"{self.prefix}/episode_length": float(info["episode"]["l"]),
            }
            for key in (
                "laps_completed",
                "last_lap_time",
                "progress_distance",
                "distance_raced",
                "mean_throttle",
                "mean_brake",
                "mean_speed",
                "max_speed",
            ):
                metric = safe_float(info["episode"].get(key))
                if metric is not None:
                    self.logger.record(f"{self.prefix}/{key}", metric)
                    wandb_payload[f"{self.prefix}/{key}"] = metric
            if wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log(wandb_payload, step=int(self.num_timesteps))
        return True


class TorcsEvalCallback(EvalCallback):
    def __init__(self, *args, lap_events_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_lap_stats = {}
        self.best_lap_median = None
        self.lap_events_path = Path(lap_events_path) if lap_events_path else None
        self.eval_summary_path = None
        if self.log_path is not None:
            self.eval_summary_path = Path(self.log_path).with_name("evaluation_summary.json")

    def _save_evaluation_summary(self, lap_stats, episode_metrics):
        if self.eval_summary_path is None:
            return
        payload = dict(lap_stats)
        payload["episodes"] = episode_metrics
        write_json(self.eval_summary_path, payload)

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return continue_training

        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as exc:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way. "
                    "See the Stable Baselines3 EvalCallback notes."
                ) from exc

        self._is_success_buffer = []
        episode_rewards, episode_lengths, lap_stats, episode_metrics = evaluate_with_lap_stats(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=self.render,
            warn=self.warn,
            step_callback=self._log_success_callback,
            phase="eval",
            lap_events_path=self.lap_events_path,
        )
        self.latest_lap_stats = lap_stats
        self._save_evaluation_summary(lap_stats, episode_metrics)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
            )

        mean_reward = lap_stats["eval_mean_reward"]
        mean_ep_length = lap_stats["eval_mean_ep_length"]
        self.last_mean_reward = mean_reward

        if self.verbose >= 1:
            print(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {lap_stats['eval_std_reward']:.2f}"
            )
            print(f"Episode length: {mean_ep_length:.2f}")
            print(
                f"mean_speed={lap_stats['eval_mean_speed']:.2f}, "
                f"max_speed={lap_stats['eval_max_speed']:.2f}, "
                f"mean_distance={lap_stats['eval_mean_distance_raced']:.2f}"
            )
            if lap_stats["eval_median_lap_time"] is not None:
                print(
                    f"Timed laps={lap_stats['eval_timed_laps']}, "
                    f"median_lap_time={lap_stats['eval_median_lap_time']:.2f}, "
                    f"best_lap_time={lap_stats['eval_best_lap_time']:.2f}"
                )
            if not lap_stats["learning_started"]:
                print(
                    f"Warning: evaluation ran before learning started "
                    f"(timesteps={self.num_timesteps}, learning_starts={lap_stats['learning_starts']})."
                )
            if lap_stats["eval_low_movement_warning"]:
                print("Warning: evaluation shows little movement. The policy may not be getting out of the launch phase.")

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.record("eval/completed_laps", float(lap_stats["eval_completed_laps"]))
        self.logger.record("eval/lap_completion_rate", float(lap_stats["eval_lap_completion_rate"]))
        self.logger.record("eval/timed_laps", float(lap_stats["eval_timed_laps"]))
        self.logger.record(
            "eval/timed_lap_completion_rate",
            float(lap_stats["eval_timed_lap_completion_rate"]),
        )
        self.logger.record("eval/off_track_rate", float(lap_stats["eval_off_track_rate"]))
        self.logger.record("eval/mean_progress", float(lap_stats["eval_mean_progress"]))
        self.logger.record("eval/progress_score", float(lap_stats["eval_progress_score"]))
        self.logger.record("eval/mean_distance_raced", float(lap_stats["eval_mean_distance_raced"]))
        self.logger.record("eval/mean_throttle", float(lap_stats["eval_mean_throttle"]))
        self.logger.record("eval/mean_brake", float(lap_stats["eval_mean_brake"]))
        self.logger.record("eval/mean_speed", float(lap_stats["eval_mean_speed"]))
        self.logger.record("eval/max_speed", float(lap_stats["eval_max_speed"]))
        if lap_stats["eval_median_lap_time"] is not None:
            self.logger.record("eval/median_lap_time", lap_stats["eval_median_lap_time"])
        if lap_stats["eval_best_lap_time"] is not None:
            self.logger.record("eval/best_lap_time", lap_stats["eval_best_lap_time"])
        if lap_stats["eval_lap_time_iqr"] is not None:
            self.logger.record("eval/lap_time_iqr", lap_stats["eval_lap_time_iqr"])

        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb_payload = {
                "eval/mean_reward": mean_reward,
                "eval/mean_ep_length": mean_ep_length,
                "eval/completed_laps": float(lap_stats["eval_completed_laps"]),
                "eval/lap_completion_rate": float(lap_stats["eval_lap_completion_rate"]),
                "eval/timed_laps": float(lap_stats["eval_timed_laps"]),
                "eval/timed_lap_completion_rate": float(lap_stats["eval_timed_lap_completion_rate"]),
                "eval/off_track_rate": float(lap_stats["eval_off_track_rate"]),
                "eval/mean_progress": float(lap_stats["eval_mean_progress"]),
                "eval/progress_score": float(lap_stats["eval_progress_score"]),
                "eval/mean_distance_raced": float(lap_stats["eval_mean_distance_raced"]),
                "eval/mean_throttle": float(lap_stats["eval_mean_throttle"]),
                "eval/mean_brake": float(lap_stats["eval_mean_brake"]),
                "eval/mean_speed": float(lap_stats["eval_mean_speed"]),
                "eval/max_speed": float(lap_stats["eval_max_speed"]),
            }
            if lap_stats["eval_median_lap_time"] is not None:
                wandb_payload["eval/median_lap_time"] = lap_stats["eval_median_lap_time"]
            if lap_stats["eval_best_lap_time"] is not None:
                wandb_payload["eval/best_lap_time"] = lap_stats["eval_best_lap_time"]
            if lap_stats["eval_lap_time_iqr"] is not None:
                wandb_payload["eval/lap_time_iqr"] = lap_stats["eval_lap_time_iqr"]
            wandb.log(wandb_payload, step=int(self.num_timesteps))

        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_reward_model"))
            self.best_mean_reward = mean_reward
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        lap_median = lap_stats["eval_median_lap_time"]
        if lap_median is not None and lap_stats["eval_timed_laps"] >= 3:
            if self.best_lap_median is None or lap_median < self.best_lap_median:
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_lap_model"))
                self.best_lap_median = lap_median

        if self.callback is not None:
            continue_training = continue_training and self._on_event()
        return continue_training
