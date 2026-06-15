"""Application configuration loaded from ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MODES = ("train", "evaluate", "sweep", "launch")
ALGORITHM_NAMES = ("ppo", "sac", "td3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AppConfig:
    mode: str
    torcs_exe: Path
    race_config: Path
    gui: bool
    launch_log: Path
    algo: str
    timesteps: int
    seed: int
    checkpoint_freq: int
    eval_episodes: int
    log_dir: Path
    run_name: str
    max_episode_steps: int
    low_progress_steps: int
    low_progress_threshold: float
    track_weight: float
    heading_weight: float
    steering_weight: float
    speed_weight: float
    terminal_penalty: float
    wandb: bool
    wandb_project: str
    wandb_entity: str | None
    evaluate_model: Path | None
    eval_freq: int
    eval_max_episode_steps: int
    sweep: dict[str, Any] | None
    sac_learning_rate: float
    sac_buffer_size: int
    sac_learning_starts: int
    sac_batch_size: int
    sac_train_freq: int
    sac_gradient_steps: int
    sac_gamma: float
    sac_tau: float
    sac_ent_coef: str | float
    sac_target_entropy: str | float
    sac_net_arch: tuple[int, ...]

    @property
    def reward_weights(self) -> dict[str, float]:
        return {
            "track_weight": self.track_weight,
            "heading_weight": self.heading_weight,
            "steering_weight": self.steering_weight,
            "speed_weight": self.speed_weight,
            "terminal_penalty": self.terminal_penalty,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a top-level mapping.")
    return data


def _require_section(data: dict[str, Any], name: str, *, required: bool = True) -> dict[str, Any] | None:
    value = data.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' must be a mapping.")
    return value


def _require_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"'{key}' must be a string.")
    return value


def _require_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer.")
    return value


def _require_float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number.")
    return float(value)


def _require_bool(section: dict[str, Any], key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a boolean.")
    return value


def _require_choice(section: dict[str, Any], key: str, choices: tuple[str, ...]) -> str:
    value = _require_str(section, key).lower()
    if value not in choices:
        raise ConfigError(f"'{key}' must be one of: {', '.join(choices)}.")
    return value


def _require_path(section: dict[str, Any], key: str, base_dir: Path) -> Path:
    raw_value = section.get(key)
    if not isinstance(raw_value, str):
        raise ConfigError(f"'{key}' must be a path string.")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _optional_path(section: dict[str, Any] | None, key: str, base_dir: Path) -> Path | None:
    if section is None:
        return None
    raw_value = section.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ConfigError(f"'{key}' must be a path string or null.")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _optional_str(section: dict[str, Any], key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"'{key}' must be a string or null.")
    return value


def _auto_or_number(section: dict[str, Any], key: str) -> str | float:
    value = section.get(key)
    if value == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be 'auto' or a number.")
    return float(value)


def _int_tuple(section: dict[str, Any], key: str) -> tuple[int, ...]:
    value = section.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"'{key}' must be a non-empty list of integers.")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"'{key}' must contain only integers.")
        result.append(item)
    return tuple(result)


def _validate_sweep(sweep: dict[str, Any] | None, mode: str) -> dict[str, Any] | None:
    if sweep is None:
        if mode == "sweep":
            raise ConfigError("Section 'sweep' is required when mode is 'sweep'.")
        return None
    for key in ("method", "metric", "parameters"):
        if key not in sweep:
            raise ConfigError(f"Section 'sweep' must define '{key}'.")
    if not isinstance(sweep["metric"], dict):
        raise ConfigError("'sweep.metric' must be a mapping.")
    if not isinstance(sweep["parameters"], dict):
        raise ConfigError("'sweep.parameters' must be a mapping.")
    count = sweep.get("count")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
        raise ConfigError("'sweep.count' must be an integer or null.")
    return sweep


def load_config(path: Path | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    base_dir = config_path.parent

    mode_value = raw.get("mode")
    if not isinstance(mode_value, str) or mode_value.lower() not in MODES:
        raise ConfigError(f"'mode' must be one of: {', '.join(MODES)}.")
    mode = mode_value.lower()

    runtime = _require_section(raw, "runtime")
    training = _require_section(raw, "training")
    environment = _require_section(raw, "environment")
    reward = _require_section(raw, "reward")
    wandb = _require_section(raw, "wandb")
    sac = _require_section(raw, "sac")
    evaluation = _require_section(raw, "evaluation", required=False)
    sweep = _validate_sweep(_require_section(raw, "sweep", required=False), mode)

    evaluate_model = _optional_path(evaluation, "model_path", base_dir)
    if mode == "evaluate" and evaluate_model is None:
        raise ConfigError("'evaluation.model_path' is required when mode is 'evaluate'.")
    if evaluation is None:
        raise ConfigError("Section 'evaluation' must be a mapping.")

    return AppConfig(
        mode=mode,
        torcs_exe=_require_path(runtime, "torcs_exe", base_dir),
        race_config=_require_path(runtime, "race_config", base_dir),
        gui=_require_bool(runtime, "gui"),
        launch_log=_require_path(runtime, "launch_log", base_dir),
        algo=_require_choice(training, "algorithm", ALGORITHM_NAMES),
        timesteps=_require_int(training, "timesteps"),
        seed=_require_int(training, "seed"),
        checkpoint_freq=_require_int(training, "checkpoint_freq"),
        log_dir=_require_path(training, "log_dir", base_dir),
        run_name=_require_str(training, "run_name"),
        max_episode_steps=_require_int(environment, "max_episode_steps"),
        low_progress_steps=_require_int(environment, "low_progress_steps"),
        low_progress_threshold=_require_float(environment, "low_progress_threshold"),
        track_weight=_require_float(reward, "track_weight"),
        heading_weight=_require_float(reward, "heading_weight"),
        steering_weight=_require_float(reward, "steering_weight"),
        speed_weight=_require_float(reward, "speed_weight"),
        terminal_penalty=_require_float(reward, "terminal_penalty"),
        wandb=_require_bool(wandb, "enabled"),
        wandb_project=_require_str(wandb, "project"),
        wandb_entity=_optional_str(wandb, "entity"),
        evaluate_model=evaluate_model,
        eval_episodes=_require_int(evaluation, "eval_episodes"),
        eval_freq=_require_int(evaluation, "eval_freq"),
        eval_max_episode_steps=_require_int(evaluation, "max_episode_steps"),
        sweep=sweep,
        sac_learning_rate=_require_float(sac, "learning_rate"),
        sac_buffer_size=_require_int(sac, "buffer_size"),
        sac_learning_starts=_require_int(sac, "learning_starts"),
        sac_batch_size=_require_int(sac, "batch_size"),
        sac_train_freq=_require_int(sac, "train_freq"),
        sac_gradient_steps=_require_int(sac, "gradient_steps"),
        sac_gamma=_require_float(sac, "gamma"),
        sac_tau=_require_float(sac, "tau"),
        sac_ent_coef=_auto_or_number(sac, "ent_coef"),
        sac_target_entropy=_auto_or_number(sac, "target_entropy"),
        sac_net_arch=_int_tuple(sac, "net_arch"),
    )
