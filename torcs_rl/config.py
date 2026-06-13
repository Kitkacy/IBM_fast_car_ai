from pathlib import Path

# Runtime targets available through torcs_windows_rl.runtime.factory.
RUNTIME_TARGETS = ("native", "linux", "docker")

# RL algorithms exposed by torcs_windows_rl.algorithms.
ALGORITHM_NAMES = ("ppo", "sac", "td3")

DEFAULT_ALGORITHM = "ppo"
DEFAULT_RUNTIME_TARGET = "native"
DEFAULT_RUN_NAME = "torcs-ppo"
DEFAULT_LOG_DIR = Path("runs")
DEFAULT_TIMESTEPS = 6_000
DEFAULT_TARGET_SPEED = 100.0
DEFAULT_SEED = 42
DEFAULT_EVAL_EPISODES = 3
DEFAULT_CHECKPOINT_FREQ = 2_000
DEFAULT_WANDB = False
DEFAULT_WANDB_PROJECT = "gym-torcs"
DEFAULT_LAUNCH_LOG_FILE = "torcs_launcher.log"
DEFAULT_TORCS_EXE = Path("../../torcs/torcs/wtorcs.exe")
DEFAULT_RACE_CONFIG = Path("../../torcs/config/raceman/practice.xml")

# Explicit TORCS port defaults to keep train/eval wiring easy to read.
DEFAULT_TRAIN_PORT = 3001
DEFAULT_EVAL_PORT = 3001
