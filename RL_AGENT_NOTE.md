# RL Agent Note

This project still contains Snakeoil-style control logic in the Python client and environment wrappers.

Current heuristic references:
- [snakeoil3_gym.py](snakeoil3_gym.py)
- [snakeoil3_jm2.py](snakeoil3_jm2.py)
- [jmcncarai.py](jmcncarai.py)
- [gym_torcs.py](gym_torcs.py#L54)

In particular, [gym_torcs.py](gym_torcs.py#L67) contains Snakeoil-like automatic throttle logic, and [gym_torcs.py](gym_torcs.py#L88) contains automatic gear logic. This means the driver behavior is still partly hand-coded.

For an RL agent approach, we should replace the placeholder random agent with a learned policy network.

Files that show the current RL entry points:
- [gym_torcs.py](gym_torcs.py): environment, observations, reward, and action mapping
- [example_experiment.py](example_experiment.py): episode loop and agent-environment interaction
- [sample_agent.py](sample_agent.py): placeholder agent that must be replaced

What we need to implement:
- Replace the random action output in [sample_agent.py](sample_agent.py#L36) with an RL policy, typically an MLP
- Use observations from [gym_torcs.py](gym_torcs.py#L232)
- Train the policy with an RL algorithm such as SAC or TD3
- Move control decisions from hand-written Snakeoil rules toward learned steering, throttle, and optionally gear

Goal: the environment should handle simulation and reward, while the RL agent learns the driving policy instead of relying on lookahead, path following, or other hard-coded control logic.