"""CLI argument parser.

Only per-run parameters that genuinely vary between invocations are
exposed as flags.  Runtime paths, reward coefficients, environment limits,
and other stable settings are read from ``config.yaml`` (see
:mod:`torcs_rl.config`) and are **not** overridable from the CLI.

Edit ``config.yaml`` to change non-CLI settings.
"""

import argparse
from pathlib import Path

from .config import (
    ALGORITHM_NAMES,
    DEFAULT_ALGORITHM,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_SEED,
    DEFAULT_TIMESTEPS,
    DEFAULT_RUN_NAME,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / evaluate a TORCS RL agent.",
        epilog="All other settings live in config.yaml — edit that file directly.",
    )

    # ── Per-run flags (things that change every invocation) ─────────────
    run = parser.add_argument_group("Run")
    run.add_argument(
        "--algorithm", dest="algo", type=str,
        choices=ALGORITHM_NAMES, default=DEFAULT_ALGORITHM,
    )
    run.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    run.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    run.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)

    run.add_argument("--sweep", action="store_true")
    run.add_argument("--sweep-count", type=int, default=None, metavar="N")
    run.add_argument("--launch", action="store_true")
    run.add_argument(
        "--evaluate-model", type=Path, default=None,
        help="Evaluate a saved checkpoint without training",
    )

    return parser.parse_args()
