# RL Algorithms and Hyperparameters Guide

This guide documents the algorithms and default hyperparameters currently implemented in `torcs_rl/algorithms.py`, plus how overrides are applied.

## Available Algorithms

The registry is defined in `get_algorithm_registry()`:

- `ppo` -> Stable Baselines3 `PPO`
- `sac` -> Stable Baselines3 `SAC`
- `td3` -> Stable Baselines3 `TD3`

CLI selection:

- `--algorithm {ppo,sac,td3}`

Default algorithm:

- `ppo` (from `torcs_rl/config.py`)

## Shared Training Defaults

From `torcs_rl/config.py` and `torcs_rl/args.py`:

- `timesteps`: `6000`
- `seed`: `42`
- `checkpoint_freq`: `2000`
- `eval_episodes`: `3`
- `target_speed`: `100.0`
- `runtime_target`: `native`
- `train_port`: `3001`
- `eval_port`: `3001`

## PPO Defaults

From `build_ppo_hyperparameters(seed)`:

- `n_steps`: `2048`
- `batch_size`: `64`
- `n_epochs`: `10`
- `gamma`: `0.99`
- `gae_lambda`: `0.95`
- `ent_coef`: `0.0`
- `vf_coef`: `0.5`
- `max_grad_norm`: `0.5`
- `learning_rate`: `linear_schedule(3e-4)` if `rl_zoo3` is available, else `3e-4`
- `clip_range`: `linear_schedule(0.2)` if `rl_zoo3` is available, else `0.2`
- `policy_kwargs`: `{"net_arch": [64, 64]}`
- `seed`: CLI/effective seed
- `verbose`: `1`

## SAC Defaults

From `build_sac_hyperparameters(seed)`:

- `learning_rate`: `3e-4`
- `buffer_size`: `100000`
- `learning_starts`: `1000`
- `batch_size`: `256`
- `tau`: `0.005`
- `gamma`: `0.99`
- `train_freq`: `1`
- `gradient_steps`: `1`
- `policy_kwargs`: `{"net_arch": [256, 256]}`
- `seed`: CLI/effective seed
- `verbose`: `1`

## TD3 Defaults

From `build_td3_hyperparameters(seed)`:

- `learning_rate`: `1e-3`
- `buffer_size`: `100000`
- `learning_starts`: `1000`
- `batch_size`: `100`
- `tau`: `0.005`
- `gamma`: `0.99`
- `train_freq`: `1`
- `gradient_steps`: `1`
- `policy_delay`: `2`
- `policy_kwargs`: `{"net_arch": [256, 256]}`
- `seed`: CLI/effective seed
- `verbose`: `1`

## Where Hyperparameters Are Used

Training path in `torcs_rl/trainer.py`:

1. Resolve algorithm from `--algorithm`.
2. Build default hyperparameters using the corresponding builder function.
3. Optionally apply W&B config overrides.
4. Instantiate model class with `model_cls("MlpPolicy", train_env, tensorboard_log=..., **algorithm_hyperparameters)`.

## W&B Override Rules

Implemented in `apply_wandb_overrides(args, algo_hyperparameters, wandb_run)`.

Overrides are applied only when:

- `wandb_run` exists, and
- `args.wandb_allow_config` is `True`.

Special behavior:

- In sweep mode, `trainer.py` forces `args.wandb_allow_config = True`.
- If W&B config contains `algo`, algorithm can switch per run/trial.
- For algorithm-specific keys, both direct and prefixed forms are supported:
  - direct: `learning_rate`
  - prefixed: `ppo_learning_rate`, `sac_learning_rate`, `td3_learning_rate`

Prefix mapping logic:

- For selected algorithm `X`, keys starting with `X_` are mapped into that algorithm's kwargs.

## W&B Logging Config Surface

`build_wandb_run_config()` logs:

- Core run settings: algo, timesteps, seed, target speed, ports, runtime target.
- Full `algorithm_hyperparameters` object.
- Flattened prefixed keys for current algorithm (for easy sweep editing).

## Practical Examples

### CLI Run With TD3

`python main.py --algorithm td3 --timesteps 50000 --seed 123`

### W&B Sweep Trial Override Example

If selected `algo=ppo`, a sweep config can include:

- `ppo_learning_rate: 0.0001`
- `ppo_n_steps: 1024`
- `ppo_batch_size: 64`

These are mapped to PPO kwargs before model creation.

### Switching Algo from W&B Launch

Set in Launch config:

- `algo: sac`
- `sac_learning_rate: 0.0002`

The run will rebuild kwargs for SAC and apply matching overrides.

## Files to Edit for Future Changes

- Add/remove algorithms: `torcs_rl/algorithms.py`
- Change default training knobs: `torcs_rl/config.py`
- Change CLI exposure: `torcs_rl/args.py`
- Change override behavior: `torcs_rl/algorithms.py` (`apply_wandb_overrides`)
- Change sweep search space: `torcs_rl/sweep.py`
