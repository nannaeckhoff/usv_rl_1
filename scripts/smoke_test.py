# Sanity-check script: verifies the environment from environment.py can be
# created, reset, and stepped through without errors. No training happens
# here, and no independent choices are made outside what config.py defines.

import sys
import os

# Allow "import environment" / "import config" when running this script
# from inside the scripts/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import make_env
import config

env = make_env() 
obs, info = env.reset(seed=config.SEED)

terminated, truncated = False, False
ep_ret, ep_cost = 0, 0
step_count = 0

while not (terminated or truncated): #ikke for løkke med 100, gir seg etter en episode 
    act = env.action_space.sample()  # random actions -- no policy yet
    obs, reward, cost, terminated, truncated, info = env.step(act)
    ep_ret += reward
    ep_cost += cost
    step_count += 1
    env.render()

print(f"Episode finished after {step_count} steps.")
print(f"Total reward: {ep_ret:.2f}")
print(f"Total cost:   {ep_cost:.2f}")

env.close()