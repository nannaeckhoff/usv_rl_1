# Central configuration: all environment/training parameters in one place
# (environment ID, seed, episode limits, etc.). No logic here, only values.

# --- Environment selection ---
# Point agent, Goal task, Level0 (no obstacles) -- matches Task 1 requirement
# of "USV + goal, simple kinematic motion, no obstacle".
ENV_ID = "SafetyPointGoal1-v0" #defined in the documentation 
ENV_ID_TRAINING = "SafetyPointGoal1Gymnasium-v0"  # standard 5-value API, for stable-baselines3

# --- Reproducibility ---
SEED = 42

# --- Rendering ---
# only for smoke test, the rest it will not give visualization of simulation
# "human" opens a live visualization window; None runs headless (faster,
# used later during actual training).
RENDER_MODE = "human"

# --- Training (PPO via stable-baselines3) ---
ALGORITHM = "PPO"
POLICY_TYPE = "MlpPolicy"   # standard feed-forward neural network policy
TOTAL_TIMESTEPS = 500_000

# How often (in timesteps) SB3 logs training progress to the console.
LOG_INTERVAL = 1  # SB3 logs every LOG_INTERVAL episodes/rollouts, not raw steps

# Where to save the trained model
# baseline is 50_000
MODEL_SAVE_PATH = "results/ppo_pointgoal1_baseline_500k"








