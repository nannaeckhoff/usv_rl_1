# Central configuration for the PointToGoal environment and PPO training
# (environment.py / scripts/train.py). No logic here, only values. The
# legacy Safety-Gym/MuJoCo pipeline this replaced now lives under legacy/.

import os

import torch

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Reproducibility ---
SEED = 42

# --- Environment ---
N_ENVS = 4096      # number of parallel environments simulated at once
DT = 0.1           # simulation timestep
HORIZON = 1000     # steps per episode before it resets

DYNAMICS = "unicycle"  # "unicycle" (default) or "double_integrator"

# Motion model constants (K_THRUST, W_MAX only apply to the unicycle).
K_THRUST = 1.0   # k: how strongly thrust increases speed
DRAG_COEF = 0.5  # c: how strongly drag decreases speed
W_MAX = 3.0      # max turn rate, rad/s

WORLD_HALF_EXTENT = 5.0  # goal/hazards are sampled in [-extent, extent]^2; agent position is clamped to it
GOAL_RADIUS = 0.3
GOAL_BONUS = 1.0         # extra reward on top of progress when the goal is reached
RANDOMIZE_START_POS = False  # False: always start at the origin. True: random, clear of hazards

# --- Hazards ---
NUM_HAZARDS = 3
HAZARD_RADIUS = 0.5
PLACEMENT_RESAMPLE_ROUNDS = 10  # how hard to try placing goal/hazards clear of each other

# --- Safety weighting ---
LAMBDA_COST = 1.0  # training signal = reward - LAMBDA_COST * cost

# --- PPO hyperparameters ---
ROLLOUT_STEPS = 200     # env steps collected per round, before each PPO update
TOTAL_UPDATES = 500     # number of rollout+update rounds to train for
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 10             # passes over the rollout data per update
NUM_MINIBATCHES = 8
LR = 3e-4
ANNEAL_LR = True   # linearly decay LR to 0 over TOTAL_UPDATES
ENTROPY_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
OBS_CLIP = 10.0    # clip range applied to normalized observations

# --- Policy/value network ---
HIDDEN_SIZE = 64
LOG_STD_INIT = -0.5   # initial log standard deviation of the Gaussian policy

# --- Logging / saving ---
SAVE_EVERY = 50                             # rounds between checkpoints
MODEL_SAVE_PATH = "results/baseline.pt"

# --- Task 1 baseline-stage presets ---
# Select one by name (e.g. `apply_preset("stage_a")`, or `--preset stage_a`
# on scripts/train.py) instead of editing the constants above by hand.
PRESETS = {
    "stage_a": {  # goal only, no hazards -- the baseline navigation task
        "NUM_HAZARDS": 0,
        "DYNAMICS": "unicycle",
    },
    "stage_b": {  # one static hazard introduced
        "NUM_HAZARDS": 1,
        "DYNAMICS": "unicycle",
    },
}


def apply_preset(name):
    """Overwrite the matching module-level constants with the preset's
    values, and namespace MODEL_SAVE_PATH under results/<name>/ so stage_a,
    stage_b, etc. never collide or save under the generic default by
    accident. Call right after `import config`, before anything else reads
    the constants (e.g. before building the environment)."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Options: {list(PRESETS)}")
    globals().update(PRESETS[name])
    globals()["MODEL_SAVE_PATH"] = os.path.join("results", name, os.path.basename(MODEL_SAVE_PATH))
