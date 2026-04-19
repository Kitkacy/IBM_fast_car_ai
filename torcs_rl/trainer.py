import json

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .algorithms import apply_wandb_overrides, build_wandb_run_config, get_algorithm_registry
from .env import TorcsRLEnv
from .runtime import create_runtime_backend

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except Exception:
    wandb = None
    WandbCallback = None


class WandbStopCallback(BaseCallback):
    """Stop training when wandb.run.should_stop() returns True (sweep early-stop)."""

    def _on_step(self) -> bool:
        if wandb is not None and wandb.run is not None and wandb.run.should_stop():
            print("[WandbStopCallback] W&B requested early stop.")
            return False
        return True


def _build_env(
    *,
    runtime_backend,
    runtime_target,
    vision,
    target_speed,
    torcs_exe,
    launch_log,
    port,
    race_config,
):
    return Monitor(
        TorcsRLEnv(
            runtime_backend=runtime_backend,
            runtime_target=runtime_target,
            vision=vision,
            target_speed=target_speed,
            torcs_exe=torcs_exe,
            launch_log=launch_log,
            torcs_port=port,
            race_config=race_config,
        )
    )


def train_and_evaluate(args):
    runtime_backend = create_runtime_backend(args.runtime_target)

    algorithm_registry = get_algorithm_registry()
    algorithm_name = str(args.algo).lower()
    if algorithm_name not in algorithm_registry:
        raise ValueError(f"Unsupported algo: {args.algo}")

    model_cls, algo_kwargs_builder = algorithm_registry[algorithm_name]
    algorithm_hyperparameters = algo_kwargs_builder(seed=args.seed)

    wandb_run = None
    sweep_mode = getattr(args, "sweep_mode", False)
    if args.wandb:
        if wandb is None or WandbCallback is None:
            raise RuntimeError("WandB integration requested but wandb is not installed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=None if sweep_mode else args.run_name,
            job_type="training",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            config=build_wandb_run_config(args, algorithm_hyperparameters),
        )
        # In sweep mode always apply config overrides regardless of --wandb-allow-config.
        if sweep_mode:
            args.wandb_allow_config = True
        args, algorithm_hyperparameters = apply_wandb_overrides(args, algorithm_hyperparameters, wandb_run)
        # Use W&B's generated run name so each sweep trial has a unique directory.
        if sweep_mode and wandb_run is not None:
            args.run_name = wandb_run.name

    run_dir = args.log_dir / args.run_name
    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    model_dir = run_dir / "models"
    for directory in (tb_dir, ckpt_dir, eval_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    race_config = str(args.race_config) if args.race_config else None
    eval_race_config = str(args.eval_race_config) if args.eval_race_config else None
    torcs_exe = str(args.torcs_exe) if args.torcs_exe else None
    train_port = int(args.train_port)
    eval_port = int(args.eval_port)

    train_env = VecMonitor(
        DummyVecEnv(
            [
                lambda: _build_env(
                    runtime_backend=runtime_backend,
                    runtime_target=args.runtime_target,
                    vision=args.vision,
                    target_speed=args.target_speed,
                    torcs_exe=torcs_exe,
                    launch_log=args.launch_log,
                    port=train_port,
                    race_config=race_config,
                )
            ]
        ),
        filename=str(run_dir / "train_monitor.csv"),
    )

    if eval_port == train_port:
        eval_env = train_env
    else:
        eval_env = VecMonitor(
            DummyVecEnv(
                [
                    lambda: _build_env(
                        runtime_backend=runtime_backend,
                        runtime_target=args.runtime_target,
                        vision=args.vision,
                        target_speed=args.target_speed,
                        torcs_exe=torcs_exe,
                        launch_log=args.launch_log,
                        port=eval_port,
                        race_config=eval_race_config,
                    )
                ]
            ),
            filename=str(run_dir / "eval_monitor.csv"),
        )

    model_key = f"{args.algo}_torcs"
    latest_model_path = model_dir / f"{model_key}_latest.zip"
    resumed_from = None

    if latest_model_path.exists():
        try:
            model = model_cls.load(str(latest_model_path), env=train_env)
            resumed_from = str(latest_model_path)
            print(f"[Model] Resumed from latest model: {latest_model_path}")
        except ValueError as exc:
            print(f"[Model] Could not resume due to env/shape mismatch: {exc}")
            model = model_cls("MlpPolicy", train_env, tensorboard_log=str(tb_dir), **algorithm_hyperparameters)
            print("[Model] Starting fresh with current configuration")
    else:
        model = model_cls("MlpPolicy", train_env, tensorboard_log=str(tb_dir), **algorithm_hyperparameters)
        print("[Model] No existing checkpoint found, starting fresh")

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq),
            save_path=str(ckpt_dir),
            name_prefix=model_key,
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

    if args.wandb:
        callbacks.append(WandbCallback(gradient_save_freq=0, model_save_path=str(model_dir), verbose=1))
        callbacks.append(WandbStopCallback())

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

        model_path = model_dir / f"{model_key}_final"
        model.save(str(model_path))

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)
        summary = {
            "interrupted": False,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "timesteps": int(args.timesteps),
            "algo": args.algo,
            "runtime_target": args.runtime_target,
            "train_port": train_port,
            "eval_port": eval_port,
            "model_path": str(model_path),
            "latest_model_path": str(latest_model_path),
            "resumed_from": resumed_from,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[Training Complete]", json.dumps(summary))
    finally:
        train_env.close()
        if eval_env is not train_env:
            eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()
