"""W&B sweep support for TORCS reward tuning."""

import copy
import json
from pathlib import Path


def build_sweep_config(args):
    algo = str(args.algo).lower()
    if algo != "sac":
        raise ValueError("Sweep generation is only supported for SAC.")

    return {
        "method": "bayes",
        "metric": {"name": "eval/progress_distance", "goal": "maximize"},
        "parameters": {
            # Reward weights
            "track_weight": {"values": [0.1, 0.3, 0.6, 1.0]},
            "heading_weight": {"values": [0.05, 0.15, 0.3, 0.6]},
            "steering_weight": {"values": [0.0, 0.01, 0.03, 0.06]},
            "terminal_penalty": {"values": [50, 100, 200, 400]},
            # SAC hyperparameters
            "sac_learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3,
            },
            "sac_batch_size": {"values": [64, 128, 256, 512]},
            "sac_buffer_size": {"values": [50_000, 100_000, 200_000]},
            "sac_learning_starts": {"values": [500, 1000, 2000]},
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
        },
    }

def make_sweep_runner(base_args):
    """Return a zero-argument callable for wandb.agent().

    Each call represents one sweep trial. W&B populates wandb.config before
    calling this function; train_and_evaluate reads those values via
    apply_wandb_overrides (forced on by sweep_mode=True).
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        raise RuntimeError("wandb must be installed to run sweeps: pip install wandb")

    from .trainer import train_and_evaluate

    def _run():
        trial_args = copy.copy(base_args)
        trial_args.wandb = True
        trial_args.sweep_mode = True
        train_and_evaluate(trial_args)

    return _run


def launch_sweep(args):
    """Create a W&B sweep and start a local agent for *sweep_count* trials."""
    try:
        import wandb
    except ImportError:
        raise RuntimeError("wandb must be installed to run sweeps: pip install wandb")

    sweep_config = build_sweep_config(args)
    args.generated_sweep_config = sweep_config

    print("[Sweep] Generated sweep config:")
    print(json.dumps(sweep_config, indent=2))

    sweep_dir = Path(args.log_dir) / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = sweep_dir / f"{args.algo}_sweep.json"
    sweep_path.write_text(json.dumps(sweep_config, indent=2), encoding="utf-8")

    sweep_id = wandb.sweep(
        sweep_config,
        project=args.wandb_project,
        entity=getattr(args, "wandb_entity", None),
    )
    print(f"[Sweep] Created sweep ID: {sweep_id}")
    print(f"[Sweep] Config saved to: {sweep_path}")
    print(f"[Sweep] To run more agents: wandb agent {sweep_id}")

    count = getattr(args, "sweep_count", None)
    wandb.agent(sweep_id, function=make_sweep_runner(args), count=count)
