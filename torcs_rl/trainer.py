from __future__ import annotations

import json
import msvcrt
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .algorithms import build_algorithm
from .config import AppConfig
from .env import MONITOR_INFO_KEYS, TorcsRLEnv
from .evaluation import LapLoggerCallback, TorcsEvalCallback, evaluate_with_lap_stats
from .runtime import create_runtime_backend
from .utils import write_json

try:
    import wandb
except Exception:
    wandb = None


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if callable(value):
        return str(value)
    return value


@contextmanager
def _torcs_port_lock(log_dir: Path):
    lock_path = Path(log_dir) / "torcs-3001.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(
                "Another local TORCS run is already using port 3001. "
                "Stop the other training, evaluation, or sweep process first."
            ) from exc
        try:
            yield
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _log_wandb_artifact(wandb_run, run_name: str, paths: tuple[Path, ...]) -> None:
    if wandb_run is None or wandb is None:
        return
    try:
        artifact = wandb.Artifact(f"{run_name}-artifacts", type="torcs-run")
        for path in paths:
            if path.exists():
                artifact.add_file(str(path))
        wandb_run.log_artifact(artifact)
    except Exception as exc:
        print(f"[W&B] Artifact upload warning: {exc}")


def _env_kwargs(config: AppConfig, runtime_backend, *, max_episode_steps: int) -> dict[str, Any]:
    return {
        "runtime_backend": runtime_backend,
        "torcs_exe": str(config.torcs_exe) if config.torcs_exe else None,
        "launch_log": config.launch_log,
        "gui": config.gui,
        "race_config": str(config.race_config) if config.race_config else None,
        "track_weight": config.track_weight,
        "heading_weight": config.heading_weight,
        "steering_weight": config.steering_weight,
        "speed_weight": config.speed_weight,
        "terminal_penalty": config.terminal_penalty,
        "max_episode_steps": max_episode_steps,
        "low_progress_steps": config.low_progress_steps,
        "low_progress_threshold": config.low_progress_threshold,
    }


def _vec_env(monitor_path: Path, config: AppConfig, runtime_backend, *, max_episode_steps: int):
    kwargs = _env_kwargs(config, runtime_backend, max_episode_steps=max_episode_steps)
    return VecMonitor(
        DummyVecEnv([lambda: TorcsRLEnv(**kwargs)]),
        filename=str(monitor_path),
        info_keywords=MONITOR_INFO_KEYS,
    )


def _set_tensorboard_logger(model, tb_dir: Path) -> None:
    model.set_logger(configure(str(tb_dir), ["stdout", "tensorboard"]))


def _summary(config: AppConfig, algo: str, model, lap_stats: dict[str, Any], **extra) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "algo": algo,
        "timesteps": int(getattr(model, "num_timesteps", 0)),
        "runtime_target": config.runtime_target,
        "gui": bool(config.gui),
        "reward_weights": config.reward_weights,
        **lap_stats,
        **extra,
    }


def _save_final_evaluation(model, env, config: AppConfig, eval_dir: Path, lap_events_path: Path):
    episode_rewards, episode_lengths, lap_stats, episode_metrics = evaluate_with_lap_stats(
        model,
        env,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        render=False,
        warn=True,
        phase="final_eval",
        lap_events_path=lap_events_path,
    )
    payload = dict(lap_stats)
    payload["episodes"] = episode_metrics
    write_json(eval_dir / "evaluation_summary.json", payload)
    return episode_rewards, episode_lengths, lap_stats


def _validate_saved_action_space(model_cls, model_path: Path):
    model = model_cls.load(str(model_path))
    try:
        action_dim = int(np.prod(model.action_space.shape))
    finally:
        del model
    if action_dim != 2:
        raise ValueError(
            f"Unsupported checkpoint action dimension: {action_dim}. "
            "Current TORCS RL only supports steering plus signed pedal."
        )


def _find_best_model_path(config: AppConfig) -> Path:
    best_models_dir = Path("best_models")
    if not best_models_dir.exists():
        raise FileNotFoundError("No best_models directory found for launch mode.")

    candidates = sorted(best_models_dir.glob(f"best_{config.algo}_t*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No best model found for algorithm '{config.algo}' in best_models.")

    def sort_key(path: Path) -> tuple[int, float]:
        stem = path.stem
        marker = "_t"
        if marker in stem:
            try:
                return int(stem.rsplit(marker, 1)[1]), path.stat().st_mtime
            except (ValueError, OSError):
                pass
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        return -1, modified

    return max(candidates, key=sort_key)


def _evaluate(config: AppConfig, model_path: Path | None = None) -> None:
    model_path = model_path or config.evaluate_model
    if model_path is None:
        raise ValueError("Evaluation requires a model path.")

    runtime_backend = create_runtime_backend(config.runtime_target)
    model_cls, _ = build_algorithm(config)
    run_dir = config.log_dir / config.run_name
    eval_dir = run_dir / "eval"
    lap_events_path = run_dir / "lap_events.csv"
    eval_dir.mkdir(parents=True, exist_ok=True)

    _validate_saved_action_space(model_cls, model_path)

    with _torcs_port_lock(config.log_dir):
        eval_env = _vec_env(
            run_dir / "eval_monitor.csv",
            config,
            runtime_backend,
            max_episode_steps=config.eval_max_episode_steps,
        )
        model = model_cls.load(str(model_path), env=eval_env)
        try:
            episode_rewards, episode_lengths, lap_stats = _save_final_evaluation(
                model,
                eval_env,
                config,
                eval_dir,
                lap_events_path,
            )
            summary = _summary(
                config,
                config.algo,
                model,
                lap_stats,
                interrupted=False,
                model_path=str(model_path),
                mean_reward=float(np.mean(episode_rewards)),
                std_reward=float(np.std(episode_rewards)),
                mean_episode_length=float(np.mean(episode_lengths)),
            )
            write_json(run_dir / "summary.json", summary)
            print("[Evaluation Complete]", json.dumps(summary))
        finally:
            eval_env.close()


def train_and_evaluate(config: AppConfig) -> None:
    if config.mode == "evaluate":
        _evaluate(config)
        return
    if config.mode == "launch":
        _evaluate(config, _find_best_model_path(config))
        return

    runtime_backend = create_runtime_backend(config.runtime_target)
    run_dir = config.log_dir / config.run_name
    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    model_dir = run_dir / "models"
    lap_events_path = run_dir / "lap_events.csv"
    for directory in (run_dir, tb_dir, ckpt_dir, eval_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_cls, algorithm_hyperparameters = build_algorithm(config)
    wandb_run = None
    wandb_enabled = config.wandb or config.mode == "launch" or config.mode == "sweep"
    if wandb_enabled:
        if wandb is None:
            raise RuntimeError("W&B integration requested but wandb is not installed.")
        if config.mode == "sweep":
            wandb_run = getattr(wandb, "run", None)
            if wandb_run is None:
                wandb_run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.run_name,
                    job_type="training",
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                )
        else:
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                job_type="training",
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
                config={
                    "mode": config.mode,
                    "algo": config.algo,
                    "timesteps": config.timesteps,
                    "seed": config.seed,
                    "eval_episodes": config.eval_episodes,
                    "checkpoint_freq": config.checkpoint_freq,
                    "runtime_target": config.runtime_target,
                    "run_name": config.run_name,
                    "gui": config.gui,
                    "reward_weights": config.reward_weights,
                    "algorithm_hyperparameters": _jsonable(algorithm_hyperparameters),
                },
            )

    with _torcs_port_lock(config.log_dir):
        train_env = _vec_env(
            run_dir / "train_monitor.csv",
            config,
            runtime_backend,
            max_episode_steps=config.max_episode_steps,
        )
        model_key = f"{config.algo}_torcs"
        latest_model_path = model_dir / f"{model_key}_latest.zip"
        final_model_base = model_dir / f"{model_key}_final"
        final_model_path = final_model_base.with_suffix(".zip")
        resumed_from = None

        if latest_model_path.exists():
            try:
                _validate_saved_action_space(model_cls, latest_model_path)
            except ValueError:
                print(f"[Model] Not resuming incompatible checkpoint: {latest_model_path}")
                model = model_cls("MlpPolicy", train_env, tensorboard_log=str(tb_dir), **algorithm_hyperparameters)
                _set_tensorboard_logger(model, tb_dir)
            else:
                model = model_cls.load(str(latest_model_path), env=train_env)
                _set_tensorboard_logger(model, tb_dir)
                resumed_from = str(latest_model_path)
                print(f"[Model] Resumed from latest model: {latest_model_path}")
        else:
            model = model_cls("MlpPolicy", train_env, tensorboard_log=str(tb_dir), **algorithm_hyperparameters)
            _set_tensorboard_logger(model, tb_dir)

        callbacks = [
            CheckpointCallback(
                save_freq=max(1, config.checkpoint_freq),
                save_path=str(ckpt_dir),
                name_prefix=model_key,
                save_vecnormalize=True,
            ),
            LapLoggerCallback(prefix="train", lap_events_path=lap_events_path),
        ]
        eval_env = _vec_env(
            run_dir / "eval_monitor.csv",
            config,
            runtime_backend,
            max_episode_steps=config.eval_max_episode_steps,
        )
        callbacks.append(
            TorcsEvalCallback(
                eval_env,
                best_model_save_path=str(model_dir),
                log_path=str(eval_dir / "evaluations.npz"),
                eval_freq=max(1, config.eval_freq),
                n_eval_episodes=config.eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
                lap_events_path=lap_events_path,
            )
        )

        summary_path = run_dir / "summary.json"
        try:
            interrupted = False
            try:
                model.learn(
                    total_timesteps=config.timesteps,
                    callback=CallbackList(callbacks),
                    progress_bar=True,
                    tb_log_name="",
                )
            except KeyboardInterrupt:
                interrupted = True
                print("[Training Interrupted] KeyboardInterrupt received, saving latest checkpoint...")

            model.save(str(latest_model_path))

            if interrupted:
                summary = _summary(
                    config,
                    config.algo,
                    model,
                    {},
                    interrupted=True,
                    latest_model_path=str(latest_model_path),
                    resumed_from=resumed_from,
                )
                write_json(summary_path, summary)
                if wandb_run is not None:
                    wandb_run.summary.update(summary)
                print("[Training Interrupted]", json.dumps(summary))
                return

            model.save(str(final_model_base))
            best_models_dir = Path("best_models")
            best_models_dir.mkdir(parents=True, exist_ok=True)
            final_best_path = best_models_dir / f"best_{config.algo}_t{int(model.num_timesteps)}.zip"
            model.save(str(final_best_path))
            print(f"[Best Model] Saved to {final_best_path}")
            episode_rewards, episode_lengths, lap_stats = _save_final_evaluation(
                model,
                train_env,
                config,
                eval_dir,
                lap_events_path,
            )
            summary = _summary(
                config,
                config.algo,
                model,
                lap_stats,
                interrupted=False,
                mean_reward=float(np.mean(episode_rewards)),
                std_reward=float(np.std(episode_rewards)),
                mean_episode_length=float(np.mean(episode_lengths)),
                model_path=str(final_model_path),
                latest_model_path=str(latest_model_path),
                resumed_from=resumed_from,
            )
            write_json(summary_path, summary)
            if wandb_run is not None:
                wandb_run.summary.update(summary)
                _log_wandb_artifact(
                    wandb_run,
                    config.run_name,
                    (
                        summary_path,
                        eval_dir / "evaluation_summary.json",
                        lap_events_path,
                        final_model_path,
                    ),
                )
            print("[Training Complete]", json.dumps(summary))
        finally:
            try:
                train_env.close()
            except Exception:
                pass
            try:
                eval_env.close()
            except Exception:
                pass
            if wandb_run is not None and config.mode != "sweep":
                wandb_run.finish()
