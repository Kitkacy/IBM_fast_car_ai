import argparse
from pathlib import Path

from .config import (
    ALGORITHM_NAMES,
    DEFAULT_ALGORITHM,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_PORT,
    DEFAULT_LAUNCH_LOG_FILE,
    DEFAULT_LOG_DIR,
    DEFAULT_RUN_NAME,
    DEFAULT_RUNTIME_TARGET,
    DEFAULT_SEED,
    DEFAULT_TARGET_SPEED,
    DEFAULT_TIMESTEPS,
    DEFAULT_TRAIN_PORT,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_WANDB,
    RUNTIME_TARGETS
)


def _parse_runtime_target(raw_value: str) -> str:
    value = str(raw_value).strip().lower()
    if value not in RUNTIME_TARGETS:
        supported = ", ".join(RUNTIME_TARGETS)
        raise argparse.ArgumentTypeError(f"Unsupported runtime '{raw_value}'. Expected one of: {supported}")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate TORCS with SB3 + TensorBoard/WandB")

    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--algorithm",
        dest="algo",
        type=str,
        choices=ALGORITHM_NAMES,
        default=DEFAULT_ALGORITHM,
        help="RL algorithm name",
    )
    training_group.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS, help="Total training timesteps")
    training_group.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME, help="Run name for logs and WandB")
    training_group.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Base output directory")
    training_group.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    training_group.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ, help="Checkpoint frequency in env steps")
    training_group.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES, help="Episodes for final evaluation")

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--runtime",
        dest="runtime_target",
        type=_parse_runtime_target,
        default=DEFAULT_RUNTIME_TARGET,
        help="Runtime backend target",
    )
    runtime_group.add_argument("--torcs-exe", type=Path, default=None, help="Override TORCS executable path")
    runtime_group.add_argument("--race-config", type=Path, default=None, help="Train race config XML path")
    runtime_group.add_argument("--eval-race-config", type=Path, default=None, help="Eval race config XML path")
    runtime_group.add_argument("--launch-log", type=str, default=DEFAULT_LAUNCH_LOG_FILE, help="Launcher log filename")
    runtime_group.add_argument("--vision", action="store_true", help="Enable TORCS vision observations")
    runtime_group.add_argument("--target-speed", type=float, default=DEFAULT_TARGET_SPEED, help="Target speed controller setpoint")
    runtime_group.add_argument("--train-port", type=int, default=DEFAULT_TRAIN_PORT, help="TORCS UDP port for training env")
    runtime_group.add_argument("--eval-port", type=int, default=DEFAULT_EVAL_PORT, help="TORCS UDP port for eval env")

    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--wandb", action="store_true", default=DEFAULT_WANDB, help="Enable Weights & Biases logging")
    wandb_group.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT, help="WandB project")
    wandb_group.add_argument("--wandb-entity", type=str, default=None, help="WandB entity/team")
    wandb_group.add_argument("--wandb-allow-config", action="store_true", help="Allow W&B config to override CLI params")

    return parser.parse_args()
