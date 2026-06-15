"""Centralised configuration loader.

Reads ``config.yaml`` from the project root at import time.
All downstream code reads its defaults from the ``DEFAULT_*`` constants
defined here; there is no separate CLI/YAML merging layer.
"""

from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None

# ── Constants ────────────────────────────────────────────────────────────

RUNTIME_TARGETS = ("native", "linux", "docker")
ALGORITHM_NAMES = ("ppo", "sac", "td3")

# ── YAML loading ─────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _load_yaml(path: Path) -> dict:
    """Return the parsed YAML dict, or {} on any failure."""
    if not path.exists():
        return {}
    if yaml is None:
        print(f"[config] PyYAML not installed; ignoring {path}")
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


_cfg = _load_yaml(_CONFIG_PATH)

# ── Convenience section accessors ────────────────────────────────────────

_runtime   = _cfg.get("runtime", {})
_training  = _cfg.get("training", {})
_env       = _cfg.get("environment", {})
_reward    = _cfg.get("reward", {})
_wandb     = _cfg.get("wandb", {})

# ── Training ─────────────────────────────────────────────────────────────

DEFAULT_ALGORITHM           = _training.get("algorithm", "sac")
DEFAULT_TIMESTEPS           = _training.get("timesteps", 6_000)
DEFAULT_SEED                = _training.get("seed", 42)
DEFAULT_CHECKPOINT_FREQ     = _training.get("checkpoint_freq", 2_000)
DEFAULT_EVAL_EPISODES       = _training.get("eval_episodes", 3)
DEFAULT_LOG_DIR             = Path(_training.get("log_dir", "runs"))
DEFAULT_RUN_NAME            = _training.get("run_name", f"torcs-{DEFAULT_ALGORITHM}")

# ── Runtime paths ────────────────────────────────────────────────────────

DEFAULT_RUNTIME_TARGET      = _runtime.get("target", "native")
DEFAULT_TORCS_EXE           = Path(_runtime.get("torcs_exe", "../../torcs/torcs/wtorcs.exe"))
DEFAULT_RACE_CONFIG         = Path(_runtime.get("race_config", "../../torcs/torcs/config/raceman/practice.xml"))
DEFAULT_GUI                 = _runtime.get("gui", False)
DEFAULT_LAUNCH_LOG_FILE     = _runtime.get("launch_log", "torcs_launcher.log")

# ── Environment ──────────────────────────────────────────────────────────

DEFAULT_MAX_EPISODE_STEPS       = _env.get("max_episode_steps", 4_000)
DEFAULT_LOW_PROGRESS_STEPS      = _env.get("low_progress_steps", 150)
DEFAULT_LOW_PROGRESS_THRESHOLD  = _env.get("low_progress_threshold", 10.0)

# ── Reward weights ───────────────────────────────────────────────────────

REWARD_WEIGHTS = {
    "track_weight": _reward.get("track_weight", 0.3),
    "heading_weight": _reward.get("heading_weight", 0.15),
    "steering_weight": _reward.get("steering_weight", 0.01),
    "terminal_penalty": _reward.get("terminal_penalty", 100.0),
}

# Convenience aliases for direct import (backward compat).
DEFAULT_TRACK_WEIGHT       = REWARD_WEIGHTS["track_weight"]
DEFAULT_HEADING_WEIGHT     = REWARD_WEIGHTS["heading_weight"]
DEFAULT_STEERING_WEIGHT    = REWARD_WEIGHTS["steering_weight"]
DEFAULT_TERMINAL_PENALTY   = REWARD_WEIGHTS["terminal_penalty"]

# ── W&B ──────────────────────────────────────────────────────────────────

DEFAULT_WANDB               = _wandb.get("enabled", False)
DEFAULT_WANDB_PROJECT       = _wandb.get("project", "gym-torcs")
DEFAULT_WANDB_ENTITY        = _wandb.get("entity", None)
DEFAULT_WANDB_ALLOW_CONFIG  = _wandb.get("allow_config", False)


# ── Namespace builder ────────────────────────────────────────────────────

def build_config_namespace(cli_args=None):
    """Return an ``argparse.Namespace`` with every field populated.

    The CLI *cli_args* (from ``parse_args()``) are merged on top of the
    YAML defaults so that per-run flags like ``--timesteps`` take priority.
    All other values come straight from ``config.yaml``.
    """
    import argparse

    yaml_defaults = argparse.Namespace(
        # Runtime
        runtime_target=DEFAULT_RUNTIME_TARGET,
        torcs_exe=DEFAULT_TORCS_EXE,
        race_config=DEFAULT_RACE_CONFIG,
        gui=DEFAULT_GUI,
        launch_log=DEFAULT_LAUNCH_LOG_FILE,
        log_dir=DEFAULT_LOG_DIR,
        # Environment
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
        low_progress_steps=DEFAULT_LOW_PROGRESS_STEPS,
        low_progress_threshold=DEFAULT_LOW_PROGRESS_THRESHOLD,
        # Reward weights
        track_weight=DEFAULT_TRACK_WEIGHT,
        heading_weight=DEFAULT_HEADING_WEIGHT,
        steering_weight=DEFAULT_STEERING_WEIGHT,
        terminal_penalty=DEFAULT_TERMINAL_PENALTY,
        # W&B
        wandb=DEFAULT_WANDB,
        wandb_project=DEFAULT_WANDB_PROJECT,
        wandb_entity=DEFAULT_WANDB_ENTITY,
        wandb_allow_config=DEFAULT_WANDB_ALLOW_CONFIG,
    )

    if cli_args is None:
        return yaml_defaults

    # Merge: CLI fields take priority; fill in any missing YAML fields.
    merged = argparse.Namespace(**vars(yaml_defaults))
    for key, value in vars(cli_args).items():
        if value is not None:
            setattr(merged, key, value)
    return merged
