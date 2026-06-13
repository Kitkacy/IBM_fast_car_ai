# Testing Todo List for IBM Fast Car AI

Steps required to verify the functionality of the TORCS RL project.

## 1. Environment Setup
- [x] Install dependencies from `pyproject.toml` (Gymnasium, Stable Baselines3, etc.).
- [x] Ensure TORCS is installed and the SCR patch is applied (check `vtorcs-RL-color/`).
- [x] Verify that `autostart.sh` (Linux) or `autostart_windows.ps1` (Windows) has correct permissions.
- [x] Test TORCS manual launch: `torcs -nofuel -nodamage -nolaptime`.

## 2. Unit Testing
- [x] Create unit tests for `torcs_scr/state.py` (parsing and encoding).
- [x] Create unit tests for `torcs_scr/utils.py`.
- [x] Create unit tests for observation normalization in `torcs_rl/env.py`.
- [x] Create unit tests for reward calculation logic in `torcs_rl/env.py`.

## 3. Integration Testing (Mocked/Stubbed)
- [x] Test `Client` connection logic using a mock UDP server.
- [x] Test `TorcsRLEnv.step()` and `reset()` with a stubbed client to avoid needing a full TORCS instance.
- [ ] Verify `ProcessManager` can correctly identify and kill processes (on both Linux and Windows).

## 4. Functional Testing (Full System)
- [x] Run a minimal training script for 1000 steps to verify end-to-end flow.
- [x] Verify `runs/` directory is populated with logs and checkpoints.
- [x] Test vision mode: `TorcsRLEnv(vision=True)` and ensure images are correctly integrated into observation space.
- [x] Test custom track selection via `race_config`.

## 5. Performance & Regression Testing
- [x] Load a pre-trained model from `runs/torcs-ppo/models/best_model.zip` and run evaluation.
- [ ] Compare evaluation metrics (lap time, damage, off-track frequency) against baselines.
- [ ] Verify W&B logging if `args.wandb` is enabled.

## 6. Documentation & Cleanup
- [x] Ensure `tea_debug.log` and `torcs_launcher.log` are being written correctly.
- [x] Create comprehensive testing documentation in `TESTING.md`.
- [ ] Update `README.md` if any new testing procedures are discovered.

