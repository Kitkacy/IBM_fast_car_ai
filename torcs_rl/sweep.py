"""W&B sweep support driven directly from config.yaml."""

from __future__ import annotations

import json
from dataclasses import replace

from .config import AppConfig


def launch_sweep(config: AppConfig):
    try:
        import wandb
    except ImportError:
        raise RuntimeError("wandb must be installed to run sweeps: pip install wandb")

    if config.sweep is None:
        raise ValueError("Sweep config is missing.")

    sweep_config = {
        "method": config.sweep["method"],
        "metric": config.sweep["metric"],
        "parameters": config.sweep["parameters"],
    }
    if "early_terminate" in config.sweep:
        sweep_config["early_terminate"] = config.sweep["early_terminate"]

    print("[Sweep] Generated sweep config:")
    print(json.dumps(sweep_config, indent=2))

    sweep_dir = config.log_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = sweep_dir / f"{config.algo}_sweep.json"
    sweep_path.write_text(json.dumps(sweep_config, indent=2), encoding="utf-8")

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb_project,
        entity=config.wandb_entity,
    )
    print(f"[Sweep] Created sweep ID: {sweep_id}")
    print(f"[Sweep] Config saved to: {sweep_path}")
    print(f"[Sweep] To run more agents: wandb agent {sweep_id}")

    from .trainer import train_and_evaluate

    def run_trial():
        run = getattr(wandb, "run", None)
        created_run = run is None or getattr(getattr(run, "settings", None), "mode", None) == "disabled"
        if created_run:
            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                job_type="training",
                mode="online",
                monitor_gym=True,
                save_code=True,
            )
        try:
            trial = dict(run.config)
            overrides = {}
            for key in (
                "algo",
                "timesteps",
                "seed",
                "checkpoint_freq",
                "eval_episodes",
                "run_name",
                "track_weight",
                "heading_weight",
                "steering_weight",
                "speed_weight",
                "terminal_penalty",
                "sac_learning_rate",
                "sac_buffer_size",
                "sac_learning_starts",
                "sac_batch_size",
                "sac_train_freq",
                "sac_gradient_steps",
                "sac_gamma",
                "sac_tau",
                "sac_ent_coef",
                "sac_target_entropy",
                "sac_net_arch",
            ):
                if key in trial and trial[key] is not None:
                    value = trial[key]
                    if key == "algo":
                        value = str(value).lower()
                    elif key == "sac_net_arch":
                        value = tuple(int(item) for item in value)
                    overrides[key] = value
            train_and_evaluate(replace(config, **overrides))
        finally:
            if created_run and run is not None:
                run.finish()

    wandb.agent(
        sweep_id,
        function=run_trial,
        count=config.sweep.get("count"),
    )
