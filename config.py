# Central configuration for the PointToGoal environment and PPO training
# (environment.py / scripts/train.py). No logic here, only values. The
# legacy Safety-Gym/MuJoCo pipeline this replaced now lives under legacy/.

import os

# --- Device ---
# No silent CPU fallback: training runs on a shared machine where the CPU
# belongs to someone else, so PointToGoal raises if CUDA is missing instead
# of quietly eating every core. (Checked there, not here, so plot.py /
# test_env.py -- which import this file but run on CPU -- still work anywhere.)
DEVICE = "cuda"

# --- Reproducibility ---
SEED = 42

# --- Environment ---
N_ENVS = 4096      # number of parallel environments simulated at once
DT = 0.1           # simulation timestep
HORIZON = 1000     # steps per episode before it resets

# Unicycle motion model constants. Default is the damped model: drag bounds
# speed to |v| <= K_THRUST/DRAG_COEF on its own, so no explicit clamp is
# needed, and the agent can brake and reverse (THRUST_MODE "bidirectional",
# same [-1, 1] action range as the MuJoCo point robot). DRAG_COEF*DT must
# stay well below 1 for the Euler update to be stable (checked in env).
#
# To get the drag-free model instead, set DRAG_COEF = 0.0 and V_MAX = 1.0
# (the clamp is then the only thing bounding speed).
K_THRUST = 1.0        # k: how strongly thrust increases speed
DRAG_COEF = 0.5       # c: how strongly drag decreases speed (0 = no drag)
W_MAX = 1.0           # max turn rate, rad/s
V_MAX = None          # hard speed clamp: v in [-V_MAX, V_MAX] ([0, V_MAX] if forward_only); None/inf = unclamped
THRUST_MODE = "bidirectional"  # "bidirectional": a1 in [-1,1], negative brakes, then reverses.
                                # "forward_only": a1 mapped to [0,1], never reverses.

WORLD_HALF_EXTENT = 5.0  # goal/hazards are sampled in [-extent, extent]^2; agent position is clamped to it
GOAL_RADIUS = 0.3
GOAL_BONUS = 1.0         # extra reward on top of progress when the goal is reached
RANDOMIZE_START_POS = False  # False: always start at the origin. True: random, clear of hazards

# --- Hazards ---
NUM_HAZARDS = 3
HAZARD_RADIUS = 0.5
PLACEMENT_RESAMPLE_ROUNDS = 10  # how hard to try placing goal/hazards clear of each other

# --- Safety representation ---
# Which cost formula the environment uses (see environment.SAFETY_MODES):
#   "binary"      -- 1 inside a hazard, 0 outside
#   "penetration" -- depth inside a hazard, 0 at the edge -> 1 at the center
#   "proximity"   -- 0 beyond SAFETY_MARGIN outside the edge -> 1 at the center
#   "velocity"    -- proximity * closing speed toward the nearest hazard
# Change it here, or per run without editing this file:
#   python scripts/train.py --set SAFETY_MODE=proximity
SAFETY_MODE = "penetration"
SAFETY_MARGIN = 0.5  # extra distance outside the hazard edge where "proximity"/"velocity" start costing

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
# Only the fallback for scripts run without a run folder -- scripts/train.py
# always saves into its own results/runs/<run>/seed_<N>/ (see run_tracking.py).
MODEL_SAVE_PATH = "results/baseline.pt"

# --- Task 1 baseline-stage presets ---
# Select one by name (e.g. `apply_preset("stage_a")`, or `--preset stage_a`
# on scripts/train.py) instead of editing the constants above by hand.
PRESETS = {
    "stage_a": {  # goal only, no hazards -- the baseline navigation task
        "NUM_HAZARDS": 0,
    },
    "stage_b": {  # one static hazard introduced
        "NUM_HAZARDS": 1,
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
