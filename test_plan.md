# Test Plan for IBM Fast Car AI

This document outlines 20 test scenarios for validating the TORCS Reinforcement Learning environment and agent.

## Environment & Client Tests

### 1. Environment Initialization
- **Scenario:** Initialize `TorcsRLEnv` with default parameters.
- **Expected Result:** Environment is created, observation and action spaces are correctly defined, and the TORCS process is launched if `launch_on_start` is true.

### 2. Observation Normalization Stats
- **Scenario:** Run several steps and check if `_obs_mean` and `_obs_m2` are updated correctly.
- **Expected Result:** Statistics evolve as new observations are processed, and observations are clipped within `obs_norm_clip`.

### 3. Vision vs. Non-Vision Mode
- **Scenario:** Initialize environment with `vision=True` and `vision=False`.
- **Expected Result:** Observation space shape changes correctly (includes image data when `vision=True`), and the client correctly requests vision data from the server.

### 4. Client Connection Timeout
- **Scenario:** Attempt to connect the `Client` to a non-existent TORCS server.
- **Expected Result:** Client should handle the timeout gracefully, potentially attempting to relaunch TORCS or exiting with a clear error message.

### 5. Multi-Port Support
- **Scenario:** Launch two `TorcsRLEnv` instances on different ports (e.g., 3001 and 3002).
- **Expected Result:** Both instances run independently without socket conflicts.

## Reward & Termination Logic

### 6. Damage Penalty
- **Scenario:** Simulate a collision where `obs["damage"]` increases.
- **Expected Result:** `_compute_reward` returns a penalty (e.g., -10.0).

### 7. Off-Track Termination
- **Scenario:** Move the car such that at least one track sensor value is negative.
- **Expected Result:** `terminated` is `True`, and `termination_reason` is `"off_track"`.

### 8. Wrong Direction Termination
- **Scenario:** Rotate the car such that `cos(angle)` is negative.
- **Expected Result:** `terminated` is `True`, and `termination_reason` is `"wrong_direction"`.

### 9. Low Progress Termination
- **Scenario:** Maintain low speed (`progress < termination_limit_progress`) beyond `terminal_judge_start` steps.
- **Expected Result:** `terminated` is `True`, and `termination_reason` is `"low_progress"`.

### 10. Target Speed Maintenance
- **Scenario:** Check `_apply_action` logic for acceleration.
- **Expected Result:** `accel` is 0.3 when below target speed, 0.0 when above, and 0.1 when near target speed.

## Process Management

### 11. TORCS Process Launch
- **Scenario:** Call `_start_torcs_process` through the environment.
- **Expected Result:** A new TORCS process is started, and its PID is captured.

### 12. TORCS Process Termination
- **Scenario:** Call `close()` on the environment.
- **Expected Result:** The TORCS process associated with the captured PID is terminated.

### 13. Race Configuration Loading
- **Scenario:** Pass a custom `race_config` path to the environment.
- **Expected Result:** TORCS is launched using the specified XML configuration.

## Client-Server Communication

### 14. Message Parsing
- **Scenario:** Provide a raw string from the TORCS server to `ServerState.parse_server_str`.
- **Expected Result:** The `S.d` dictionary is correctly populated with sensor values (speed, track, angle, etc.).

### 15. Action Encoding
- **Scenario:** Set values in `DriverAction` and call `repr()`.
- **Expected Result:** A correctly formatted string (e.g., `(accel 0.3)(steer 0.0)...`) is generated for transmission.

## RL Training & Evaluation

### 16. PPO Policy Initialization
- **Scenario:** Create a PPO model using the `TorcsRLEnv`.
- **Expected Result:** Model is initialized with correct observation and action dimensions.

### 17. Training Step
- **Scenario:** Execute a single training iteration.
- **Expected Result:** Model parameters are updated, and logs are generated (optionally to W&B).

### 18. Checkpoint Saving
- **Scenario:** Run training with `CheckpointCallback`.
- **Expected Result:** Model files are saved to the `runs/` directory at specified intervals.

### 19. Model Evaluation
- **Scenario:** Load a saved model and run evaluation episodes.
- **Expected Result:** The agent performs steering actions based on observations, and metrics are recorded.

### 20. Sweep Parameter Handling
- **Scenario:** Run `main.py --sweep` with a mock sweep configuration.
- **Expected Result:** The `launch_sweep` function is called and correctly parses hyperparameters.
