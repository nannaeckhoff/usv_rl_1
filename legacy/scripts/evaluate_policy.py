# Loads the trained PPO model and runs one episode using its learned
# policy (not random actions). Records reward, cost, and the agent's
# x,y position at every step, then saves two plots:
#   - reward_cost_plot.png: reward and cost over the episode
#   - trained_trajectory.png: the agent's path in x,y space

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from environment import make_env
import config

env = make_env(render_mode=None, for_training=False)
model = PPO.load(config.MODEL_SAVE_PATH)

obs, info = env.reset(seed=config.SEED)
terminated, truncated = False, False

rewards = []
costs = []
positions = [env.unwrapped.task.agent.pos[:2].copy()]
goal_positions = [env.unwrapped.task.goal.pos[:2].copy()]

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, cost, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    costs.append(cost)
    positions.append(env.unwrapped.task.agent.pos[:2].copy())
    goal_positions.append(env.unwrapped.task.goal.pos[:2].copy())

env.close()

print(f"Episode length: {len(rewards)} steps")
print(f"Total reward:   {sum(rewards):.2f}")
print(f"Total cost:     {sum(costs):.2f}")

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

# --- Plot 1: reward and cost per step ---
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(rewards)
ax1.set_ylabel("Reward per step")
ax1.set_title("Trained policy: reward and cost over one episode")
ax2.plot(costs, color="red")
ax2.set_ylabel("Cost per step")
ax2.set_xlabel("Step")
plt.tight_layout()
fig1.savefig(os.path.join(results_dir, "reward_cost_plot.png"), dpi=120)

# --- Plot 2: x,y trajectory (goals numbered in order reached) ---
positions = np.array(positions)
goal_positions = np.array(goal_positions)

unique_goals = [goal_positions[0]]
for g in goal_positions[1:]:
    if not np.allclose(g, unique_goals[-1], atol=1e-3):
        unique_goals.append(g)
unique_goals = np.array(unique_goals)

fig2, ax = plt.subplots(figsize=(6, 6))
ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=1, label="Agent trajectory")
ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="Start")
ax.plot(unique_goals[:, 0], unique_goals[:, 1], "r*", markersize=15, label="Goal positions")

# Number each goal in the order it was reached.
for i, (gx, gy) in enumerate(unique_goals, start=1):
    ax.annotate(str(i), (gx, gy), textcoords="offset points",
                xytext=(8, 8), fontsize=11, fontweight="bold", color="darkred")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Trained policy trajectory ({config.ENV_ID})")
ax.legend()
ax.set_aspect("equal")
fig2.savefig(os.path.join(results_dir, "trained_trajectory.png"), dpi=120)