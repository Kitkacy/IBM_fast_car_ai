import json
import msvcrt
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .algorithms import apply_wandb_overrides, build_wandb_run_config, get_algorithm_registry
from .config import REWARD_WEIGHTS
from .env import MONITOR_INFO_KEYS, TorcsRLEnv
from .evaluation import LapLoggerCallback, evaluate_with_lap_stats
from .runtime import create_runtime_backend
from .utils import write_json

try:
    import wandb
except Exception:
    wandb = None


def _build_run_name(args, *, sweep_mode):
    if not sweep_mode:
        return str(args.run_name)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{args.run_name}-{args.algo}-sweep-{timestamp}-{os.getpid()}"


@contextmanager
def _torcs_port_lock(log_dir):
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


def _build_env(**kwargs):
    return TorcsRLEnv(**kwargs)


def _build_vec_env(*, monitor_path, **kwargs):
    return VecMonitor(
        DummyVecEnv([lambda: _build_env(**kwargs)]),
        filename=str(monitor_path),
        info_keywords=MONITOR_INFO_KEYS,
    )


def _patch_wandb_tensorboard(tb_dir):
    if wandb is None:
        return False
    try:
        wandb.tensorboard.patch(root_logdir=str(tb_dir), save=False)
        return True
    except Exception as exc:
        print(f"[W&B] TensorBoard patch warning: {exc}")
        return False


def _log_wandb_artifact(wandb_run, run_name, paths):
    if wandb_run is None or wandb is None:
        return
    try:
        artifact = wandb.Artifact(f"{run_name}-artifacts", type="torcs-run")
        for path in paths:
            file_path = Path(path)
            if file_path.exists():
                artifact.add_file(str(file_path))
        wandb_run.log_artifact(artifact)
    except Exception as exc:
        print(f"[W&B] Artifact upload warning: {exc}")


def _build_summary(args, algorithm_name, model, lap_stats, **extra):
    reward_weights = {
        k: float(getattr(args, k, v))
        for k, v in REWARD_WEIGHTS.items()
        if hasattr(args, k)
    }
    return {
        "algo": algorithm_name,
        "timesteps": int(getattr(model, "num_timesteps", 0)),
        "runtime_target": args.runtime_target,
        "gui": bool(args.gui),
        "reward_weights": reward_weights,
        **lap_stats,
        **extra,
    }


def _save_final_evaluation(model, eval_env, args, eval_dir, lap_events_path):
    episode_rewards, episode_lengths, lap_stats, episode_metrics = evaluate_with_lap_stats(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
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


def _validate_saved_action_space(model_cls, model_path):
    model = model_cls.load(str(model_path))
    try:
        action_dim = int(np.prod(model.action_space.shape))
    finally:
        del model
    if action_dim == 2:
        return
    raise ValueError(
        f"Unsupported checkpoint action dimension: {action_dim}. "
        "Current TORCS RL only supports steering plus signed pedal."
    )


def _evaluate_only(args):
    runtime_backend = create_runtime_backend(args.runtime_target)
    algorithm_name = str(args.algo).lower()
    model_cls, _ = get_algorithm_registry()[algorithm_name]
    run_dir = Path(args.log_dir) / args.run_name
    eval_dir = run_dir / "eval"
    lap_events_path = run_dir / "lap_events.csv"
    eval_dir.mkdir(parents=True, exist_ok=True)

    torcs_exe = str(args.torcs_exe) if args.torcs_exe else None
    _validate_saved_action_space(model_cls, args.evaluate_model)

    env_kwargs = dict(
        runtime_backend=runtime_backend,
        torcs_exe=torcs_exe,
        launch_log=args.launch_log,
        gui=args.gui,
        race_config=str(args.race_config) if args.race_config else None,
        track_weight=args.track_weight,
        heading_weight=args.heading_weight,
        steering_weight=args.steering_weight,
        terminal_penalty=args.terminal_penalty,
        max_episode_steps=args.max_episode_steps,
        low_progress_steps=args.low_progress_steps,
        low_progress_threshold=args.low_progress_threshold,
    )
    with _torcs_port_lock(args.log_dir):
        eval_env = _build_vec_env(monitor_path=run_dir / "eval_monitor.csv", **env_kwargs)
        model = model_cls.load(str(args.evaluate_model), env=eval_env)

        try:
            episode_rewards, episode_lengths, lap_stats = _save_final_evaluation(
                model,
                eval_env,
                args,
                eval_dir,
                lap_events_path,
            )
            summary = _build_summary(
                args,
                algorithm_name,
                model,
                lap_stats,
                interrupted=False,
                mode="evaluation",
                model_path=str(Path(args.evaluate_model)),
                mean_reward=float(np.mean(episode_rewards)),
                std_reward=float(np.std(episode_rewards)),
                mean_episode_length=float(np.mean(episode_lengths)),
            )
            write_json(run_dir / "summary.json", summary)
            if not lap_stats.get("learning_started", True):
                print(
                    f"[Evaluation Warning] Timesteps {model.num_timesteps} did not pass "
                    f"learning_starts={lap_stats.get('learning_starts')}"
                )
            if lap_stats.get("eval_low_movement_warning"):
                print("[Evaluation Warning] The model showed very little movement and may not be getting out of the launch phase.")
            print("[Evaluation Complete]", json.dumps(summary))
        finally:
            eval_env.close()


def train_and_evaluate(args):
    if getattr(args, "evaluate_model", None):
        _evaluate_only(args)
        return

    runtime_backend = create_runtime_backend(args.runtime_target)
    sweep_mode = getattr(args, "sweep_mode", False)
    args.run_name = _build_run_name(args, sweep_mode=sweep_mode)
    run_dir = Path(args.log_dir) / args.run_name
    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    model_dir = run_dir / "models"
    lap_events_path = run_dir / "lap_events.csv"
    for directory in (run_dir, tb_dir, ckpt_dir, eval_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generated_sweep_config = getattr(args, "generated_sweep_config", None)
    if generated_sweep_config is not None:
        write_json(run_dir / "sweep_config.json", generated_sweep_config)

    algorithm_registry = get_algorithm_registry()
    algorithm_name = str(args.algo).lower()
    if algorithm_name not in algorithm_registry:
        raise ValueError(f"Unsupported algo: {args.algo}")
    model_cls, algo_kwargs_builder = algorithm_registry[algorithm_name]
    algorithm_hyperparameters = algo_kwargs_builder(seed=args.seed)

    wandb_run = None
    patched_tensorboard = False
    if args.wandb:
        if wandb is None:
            raise RuntimeError("W&B integration requested but wandb is not installed.")
        patched_tensorboard = _patch_wandb_tensorboard(tb_dir)
        if sweep_mode:
            # In sweep mode the agent owns the run and its config.
            # Reuse the existing run if available; otherwise init WITHOUT
            # a config dict so we don't fight the sweep's values.
            wandb_run = getattr(wandb, "run", None)
            if wandb_run is None:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.run_name,
                    job_type="training",
                )
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                job_type="training",
                sync_tensorboard=False,
                monitor_gym=True,
                save_code=True,
                config=build_wandb_run_config(args, algorithm_hyperparameters),
            )
        if sweep_mode:
            args.wandb_allow_config = True
        args, algorithm_hyperparameters = apply_wandb_overrides(args, algorithm_hyperparameters, wandb_run)
        algorithm_name = str(args.algo).lower()
        model_cls, _ = algorithm_registry[algorithm_name]

    torcs_exe = str(args.torcs_exe) if args.torcs_exe else None
    race_config = str(args.race_config) if getattr(args, "race_config", None) else None

    env_kwargs = dict(
        runtime_backend=runtime_backend,
        torcs_exe=torcs_exe,
        launch_log=args.launch_log,
        gui=args.gui,
        race_config=race_config,
        track_weight=args.track_weight,
        heading_weight=args.heading_weight,
        steering_weight=args.steering_weight,
        terminal_penalty=args.terminal_penalty,
        max_episode_steps=args.max_episode_steps,
        low_progress_steps=args.low_progress_steps,
        low_progress_threshold=args.low_progress_threshold,
    )
    with _torcs_port_lock(args.log_dir):
        train_env = _build_vec_env(monitor_path=run_dir / "train_monitor.csv", **env_kwargs)
        model_key = f"{algorithm_name}_torcs"
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
            else:
                model = model_cls.load(str(latest_model_path), env=train_env)
                resumed_from = str(latest_model_path)
                print(f"[Model] Resumed from latest model: {latest_model_path}")
        else:
            model = model_cls("MlpPolicy", train_env, tensorboard_log=str(tb_dir), **algorithm_hyperparameters)

        callbacks = [
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq),
                save_path=str(ckpt_dir),
                name_prefix=model_key,
                save_vecnormalize=True,
            ),
            LapLoggerCallback(prefix="train", lap_events_path=lap_events_path),
        ]

        summary_path = run_dir / "summary.json"
        try:
            interrupted = False
            try:
                model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), progress_bar=True)
            except KeyboardInterrupt:
                interrupted = True
                print("[Training Interrupted] KeyboardInterrupt received, saving latest checkpoint...")

            model.save(str(latest_model_path))

            if interrupted:
                summary = _build_summary(
                    args,
                    algorithm_name,
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
            final_best_path = best_models_dir / f"best_{algorithm_name}_t{int(model.num_timesteps)}.zip"
            model.save(str(final_best_path))
            print(f"[Best Model] Saved to {final_best_path}")
            episode_rewards, episode_lengths, lap_stats = _save_final_evaluation(
                model,
                train_env,
                args,
                eval_dir,
                lap_events_path,
            )
            summary = _build_summary(
                args,
                algorithm_name,
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
            if not lap_stats.get("learning_started", True):
                print(
                    f"[Training Warning] Timesteps {model.num_timesteps} did not pass "
                    f"learning_starts={lap_stats.get('learning_starts')}"
                )
            if lap_stats.get("eval_low_movement_warning"):
                print("[Training Warning] Final evaluation showed very little movement and may not be getting out of the launch phase.")
            if wandb_run is not None:
                wandb_run.summary.update(summary)
                _log_wandb_artifact(
                    wandb_run,
                    args.run_name,
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
            if wandb_run is not None and not sweep_mode:
                wandb_run.finish()
            if patched_tensorboard and wandb is not None:
                try:
                    wandb.tensorboard.unpatch()
                except Exception:
                    pass
