from gym_torcs import TorcsEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os

os.makedirs("./runs/torcs-ppo/tensorboard/", exist_ok=True)
os.makedirs("./runs/torcs-ppo/checkpoints/", exist_ok=True)
os.makedirs("./runs/torcs-ppo/models/", exist_ok=True)

# 1. Initialize custom TORCS environment with NO heuristics
env = TorcsEnv(vision=False, throttle=True, gear_change=True)

# 2. Build the Soft Actor-Critic (SAC) Model
# Using parameters inspired by the Corkscrew analysis
model = SAC(
    "MlpPolicy", 
    env, 
    learning_rate=0.0001,
    buffer_size=1000000,
    batch_size=512,
    ent_coef=0.3,
    verbose=1,
    tensorboard_log="./runs/torcs-ppo/tensorboard/" # Log your progress
)

# 3. Create a callback to save the model every 5000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=5000, 
    save_path='./runs/torcs-ppo/checkpoints/',
    name_prefix='sac_torcs'
)

# 4. Start Deep Learning!
print("Starting RL Training...")
model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

# Save the final agent
model.save("./runs/torcs-ppo/models/sac_torcs_final")
env.end()