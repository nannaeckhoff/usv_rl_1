# Exploratory script: prints out the internal structure of the environment
# to find where the agent's x,y position is stored. Not part of the final
# pipeline -- purely for investigation.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import make_env
import config

env = make_env(render_mode=None, for_training=False)
obs, info = env.reset(seed=config.SEED)

# The actual task object usually lives one or two layers inside the
# gymnasium wrappers. env.unwrapped strips away the wrapper layers.
task = env.unwrapped.task

print("=== Attributes of env.unwrapped.task ===")
print([a for a in dir(task) if not a.startswith("_")])

print("\n=== Attributes of env.unwrapped.task.agent (if it exists) ===")
if hasattr(task, "agent"):
    print([a for a in dir(task.agent) if not a.startswith("_")])
else:
    print("No 'agent' attribute found on task.")

env.close()


print("\n=== Attributes of env.unwrapped.task.goal ===")
print([a for a in dir(task.goal) if not a.startswith("_")])

# Try to print the actual size/radius value if it exists
for attr_name in ("size", "radius", "keepout"):
    if hasattr(task.goal, attr_name):
        print(f"task.goal.{attr_name} = {getattr(task.goal, attr_name)}")