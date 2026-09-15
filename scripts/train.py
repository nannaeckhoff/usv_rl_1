# Trains a baseline PPO policy (via stable-baselines3) on the environment
# defined in environment.py, using hyperparameters from config.py.
# Saves the trained model to config.MODEL_SAVE_PATH when done.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from environment import make_env
import config

# Train without a visualization window -- much faster than "human" mode,
# and we don't need to watch every single one of 50,000 steps.
env = make_env(render_mode=None, for_training=True)
model = PPO(
    config.POLICY_TYPE,
    env,
    verbose=0,          # prints training progress to console
    seed=config.SEED,
)

model.learn(
    total_timesteps=config.TOTAL_TIMESTEPS,
    progress_bar=True,
)

model.save(config.MODEL_SAVE_PATH)
print(f"\nTraining complete. Model saved to: {config.MODEL_SAVE_PATH}")

env.close()