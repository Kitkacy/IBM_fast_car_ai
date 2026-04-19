# Repository Navigation Guide

This guide maps the active codebase, lists all CLI arguments, and shows where to edit environment logic, client settings, and hyperparameters.

## Entry Points

- `main.py`: Main execution entry. Handles normal train/eval, W&B launch mode, and sweep mode.
- `server.py`: (**NOT IMPLEMENTED**) FastAPI webhook server for automation callbacks and external run triggering.
- `launch-config.yaml`: (**NOT IMPLEMENTED**) W&B Launch defaults and entry point wiring.

## Libraries, Tools, Dependencies, and Algorithms

### Core Libraries

- `gymnasium`: Gym-compatible RL environment API used by `TorcsRLEnv`.
- `stable-baselines3`: RL algorithm implementations and training callbacks.
- `numpy`: Numeric operations for observations, rewards, and action shaping.
- `tensorboard`: Local training metric logging and visualization backend.

### Experiment and Automation Tools

- `wandb`: Experiment tracking, run metadata, sweeps, and launch/automation integration.
- `rl-zoo3`: (**NOT IMPLEMENTED**) Optional schedule helper (`linear_schedule`) used in PPO defaults when available.
- `fastapi`: (**NOT IMPLEMENTED**) HTTP API framework used by `server.py` webhook service.
- `uvicorn`: (**NOT IMPLEMENTED**) ASGI server used to run the FastAPI app.


### Algorithms Used

- PPO (on-policy): baseline algorithm with optional linear schedules for LR/clip.
- SAC (off-policy): entropy-regularized actor-critic for continuous control.
- TD3 (off-policy): deterministic policy gradient variant with delayed policy updates.

## Core Modules

### `torcs_rl/` (RL Layer)

- `torcs_rl/env.py`: `TorcsRLEnv` gym environment, observation flatten/normalization, reward, termination logic.
- `torcs_rl/trainer.py`: Main training/evaluation loop, checkpointing, eval callback, W&B integration.
- `torcs_rl/algorithms.py`: PPO/SAC/TD3 hyperparameter builders and W&B override mapping.
- `torcs_rl/args.py`: Complete CLI argument definitions.
- `torcs_rl/config.py`: Default values for runtime, ports, timesteps, logging, etc.
- `torcs_rl/sweep.py`: Sweep config and agent runner.

### `torcs_rl/runtime/` (Runtime Backend Abstraction)

- `torcs_rl/runtime/base.py`: Backend contract (`TorcsRuntimeBackend`).
- `torcs_rl/runtime/factory.py`: Target-to-backend selection (`native`, `linux`, `docker`).
- `torcs_rl/runtime/native.py` [Windows]: Implemented local backend.
- `torcs_rl/runtime/stubs.py`: Linux/Docker stubs (**NOT IMPLEMENTED**).
- `torcs_rl/runtime/types.py`: Protocol typing for client object.

### `torcs_scr/` (TORCS Client/Process Layer)

- `torcs_scr/client.py`: SCR UDP client and connection loop.
- `torcs_scr/state.py`: Server state parsing and action representation.
- `torcs_scr/process_manager.py` [Windows]: TORCS launch/kill and launcher logs.
- `torcs_scr/constants.py`: SCR protocol constants and help text.
- `torcs_scr/utils.py`: Helper functions used in SCR parsing.
- `torcs_scr/__init__.py`: Public exports for integration.

## Top-Level Scripts and Artifacts

- `autostart_windows.ps1`, `autostart_windows.bat` [Windows]: TORCS menu click-through.
- `start_torcs_windows.ps1`, `start_torcs_windows.bat` [Windows]: **DEPRECATED** manual TORCS startup helpers.
- `runs/`: Output folder for checkpoints, TensorBoard logs, evaluation files, and summaries.
- `vtorcs-RL-color/`: TORCS source/assets tree.
- `archive/`: Legacy scripts retained for reference from earlier repository versions.

## All CLI Arguments and How to Use Them

Defined in `torcs_rl/args.py` and consumed by `main.py` / `torcs_rl/trainer.py`.

### Training

- `--algorithm {ppo,sac,td3}`: Select RL algorithm.
- `--timesteps INT`: Total training timesteps.
- `--run-name TEXT`: Output run folder name under `runs/` (or chosen log-dir).
- `--log-dir PATH`: Base directory for run artifacts.
- `--seed INT`: Random seed.
- `--checkpoint-freq INT`: Steps between model checkpoints and eval callback runs.
- `--eval-episodes INT`: Number of episodes for final evaluation.

Example:

```bash
python main.py --algorithm ppo --timesteps 50000 --run-name ppo-baseline --seed 123
```

### Runtime / TORCS

- `--runtime {native,linux,docker}`: Runtime backend target.
- `--torcs-exe PATH`: Explicit TORCS executable path.
- `--race-config PATH`: Train race XML config.
- `--eval-race-config PATH`: Eval race XML config.
- `--launch-log TEXT`: Launcher log filename.
- `--vision`: Enable vision observations.
- `--target-speed FLOAT`: Speed target used by built-in action logic.
- `--train-port INT`: UDP port for training env.
- `--eval-port INT`: UDP port for evaluation env.

Example:

```bash
python main.py --runtime native --torcs-exe "C:\path\to\wtorcs.exe" --race-config practice.xml --train-port 3001 --eval-port 3002  # [Windows path example]
```

#### Running `wtorcs.exe` Directly with `-r` XML (Fast, No Menu Click-Through) *maybe* [Windows] only, check

If you want TORCS to start straight into a race session (instead of navigating menus), run it with a raceman XML:

```powershell
"C:\path\to\wtorcs.exe" -r "C:\path\to\torcs\config\raceman\practice.xml" -nofuel -nodamage -nolaptime
```

Recommended options:

- `-r <xml>`: loads race manager config directly (for example `config/raceman/practice.xml`).
- `-nofuel -nodamage -nolaptime`: reduces simulation overhead and keeps runs more stable for RL training.
- `-vision` (optional): include when training with vision observations.

Notes:

- This avoids TORCS menu interaction; it does not make TORCS truly headless.
- In this project, the same approach is used through `--race-config` and the process manager, so CLI runs can skip GUI menu automation too.

### Weights and Biases

- `--wandb`: Enable W&B logging integration.
- `--wandb-project TEXT`: W&B project name.
- `--wandb-entity TEXT`: W&B entity/team.
- `--wandb-allow-config`: Allow `wandb.config` to override CLI/default values.

Example:

```bash
python main.py --wandb --wandb-project gym-torcs --wandb-entity my-team --wandb-allow-config
```

### W&B Launch / Sweep (**SWEEP NOT IMPLEMENTED**)

- `--launch`: Run as W&B Launch job; `main.py` forces `--wandb` and `--wandb-allow-config` behavior.
- `--sweep`: Create sweep and start local sweep agent.
- `--sweep-count N`: Optional limit for how many sweep trials to run locally.

Examples:

```bash
python main.py --launch --wandb-project gym-torcs
python main.py --sweep --wandb-project gym-torcs --sweep-count 10
```

## Where to Modify Key Behavior

### Gym Environment

Edit `torcs_rl/env.py`.

- Observation composition and normalization: `_flatten_observation`, `_normalize_observation`.
- Reward and done criteria: `_compute_reward` and related thresholds (`terminal_judge_start`, `termination_limit_progress`, target speed logic).
- Agent action conversion: `_agent_to_torcs` and `_apply_action`.

### Client Settings (TORCS SCR)

Edit `torcs_scr/client.py` and `torcs_scr/process_manager.py`.

- SCR socket host/port/session defaults: `Client.__init__` (`host`, `port`, `sid`, stage/episodes).
- Connection/retry behavior: `setup_connection` and `_relaunch_torcs`.
- TORCS process launch/kill flow: `torcs_scr/process_manager.py` (`launch_torcs_process`, `kill_torcs_process`, stale-port cleanup).
- Runtime defaults for executable/config path selection [Windows]: `torcs_rl/runtime/native.py`.
- Runtime defaults for executable/config path selection [Windows]: `torcs_rl/runtime/native.py`.

### Hyperparameters

Edit `torcs_rl/algorithms.py`.

- PPO defaults: `build_ppo_hyperparameters`.
- SAC defaults: `build_sac_hyperparameters`.
- TD3 defaults: `build_td3_hyperparameters`.
- W&B override mapping and prefixed keys (`ppo_*`, `sac_*`, `td3_*`): `apply_wandb_overrides`.

Also update base defaults in `torcs_rl/config.py` and CLI exposure in `torcs_rl/args.py` when adding/removing knobs.

## Main Execution Flows (Short Guide)

### 1) Local Training (no W&B)

```bash
python main.py --algorithm ppo --timesteps 20000 --run-name local-ppo
```

Flow:

1. `main.py` parses args and calls `train_and_evaluate`.
2. `torcs_rl/trainer.py` creates backend, env(s), model, callbacks.
3. `torcs_rl/env.py` drives steps through `torcs_scr.Client`.
4. Artifacts are written to `runs/<run-name>/`.

### 2) Training with W&B Tracking

```bash
python main.py --wandb --wandb-project gym-torcs --algorithm sac --timesteps 50000
```

Flow:

1. W&B run is initialized in `torcs_rl/trainer.py`.
2. Optional W&B config overrides are applied when `--wandb-allow-config` is enabled.
3. Training logs, checkpoints, and summaries are synced to W&B.

### 3) W&B Sweep (**NOT IMPLEMENTED**)

```bash
python main.py --sweep --wandb-project gym-torcs --sweep-count 20
```

Flow:

1. `main.py` routes to `torcs_rl/sweep.py`.
2. Sweep is created from `DEFAULT_SWEEP_CONFIG`.
3. Agent executes per-trial runs; overrides feed into `apply_wandb_overrides`.

### 4) W&B Launch Job (**NOT IMPLEMENTED**)

```bash
python main.py --launch --wandb-project gym-torcs
```

Flow:

1. `main.py` forces W&B config override mode.
2. Launch-injected parameters are applied before model creation.
3. Run proceeds through the same trainer pipeline.

### 5) Webhook-Driven Automation (**NOT IMPLEMENTED**)

```bash
python server.py --port 8000
```

Flow:

1. External system (for example W&B Automations) calls `POST /webhook` or `POST /launch`.
2. `server.py` validates and dispatches event logic.
3. New training subprocesses are spawned via `main.py --launch`.

## Quick Task Routing

- Change reward/termination or action behavior: `torcs_rl/env.py`.
- Change hyperparameter defaults or algorithm registry: `torcs_rl/algorithms.py`.
- Change CLI or default argument values: `torcs_rl/args.py`, `torcs_rl/config.py`.
- Change TORCS process management behavior: `torcs_scr/process_manager.py`.
- Change SCR client socket/session behavior: `torcs_scr/client.py`.
- Change training callbacks/checkpointing/evaluation: `torcs_rl/trainer.py`.
- Change sweep search space: `torcs_rl/sweep.py`.


