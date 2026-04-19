"""W&B sweep support for TORCS RL training.

Usage
-----
Create a sweep and run one agent with 10 trials::

    python main.py --sweep --wandb-project gym-torcs --sweep-count 10

The sweep YAML is printed on stdout so you can inspect or edit it before
the agent starts. To run additional agents in parallel (e.g. on another
machine) copy the printed sweep ID and run::

    wandb agent <sweep-id>
"""

from .config import ALGORITHM_NAMES

# ---------------------------------------------------------------------------
# Default sweep config — edit to fit your search space.
# ---------------------------------------------------------------------------
DEFAULT_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "eval/mean_reward", "goal": "maximize"},
    "parameters": {
        "algo": {"values": list(ALGORITHM_NAMES)},
        "target_speed": {"min": 40.0, "max": 150.0},
        "seed": {"values": [42, 123, 456]},
        # PPO-specific
        "ppo_learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "ppo_n_steps": {"values": [512, 1024, 2048]},
        "ppo_batch_size": {"values": [32, 64, 128]},
        "ppo_gamma": {"min": 0.9, "max": 0.9999},
        # SAC/TD3-specific
        "sac_learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "td3_learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
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
        import copy
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

    sweep_id = wandb.sweep(
        DEFAULT_SWEEP_CONFIG,
        project=args.wandb_project,
        entity=getattr(args, "wandb_entity", None),
    )
    print(f"[Sweep] Created sweep ID: {sweep_id}")
    print(f"[Sweep] To run more agents: wandb agent {sweep_id}")

    count = getattr(args, "sweep_count", None)
    wandb.agent(sweep_id, function=make_sweep_runner(args), count=count)
