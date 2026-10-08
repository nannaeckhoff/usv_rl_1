# Plots for a trained checkpoint:
#   trajectory.png       -- one deterministic episode: agent path (with
#                           heading arrows for the unicycle), goal(s)
#                           reached, hazards (if any)
#   training_curves.png  -- reward/success-rate/cost over the whole training
#                           run, read from the metrics.csv logged next to
#                           the checkpoint by scripts/train.py
#
# Run: python scripts/plot.py --run results/runs/<run> [--seeds 1 2]
#        plots every seed of a run (or just --seeds), using the config that
#        run was trained with -- not whatever config.py says now
#      python scripts/plot.py [--preset stage_a] [--checkpoint PATH]
#        a single checkpoint outside a run folder, with the current config.py

import argparse
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import config as cfg
import run_tracking
from environment import PointToGoal
from train import ActorCritic, RunningMeanStd
from test_env import _build_stage_a_env as _build_eval_env, run_policy, unicycle_heuristic_action


def parse_args():
    p = argparse.ArgumentParser(description="Plot a trajectory and training curves for a trained checkpoint.")
    p.add_argument("--run", type=str, default=None, help="a results/runs/<run> folder from scripts/train.py")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="with --run: which seeds (default: all)")
    p.add_argument("--preset", type=str, default=None, choices=list(cfg.PRESETS))
    p.add_argument("--checkpoint", type=str, default=None, help="defaults to config.MODEL_SAVE_PATH")
    p.add_argument("--metrics", type=str, default=None, help="defaults to metrics.csv next to the checkpoint")
    p.add_argument("--out_dir", type=str, default=None, help="defaults to the checkpoint's own directory")
    return p.parse_args()


def compute_reference_baselines():
    """Re-run the step-9 heuristic and a random policy under the *current*
    cfg (whatever preset is active), so the reference lines are always
    correct for what was actually trained -- never hardcoded numbers."""
    N, steps = 300, cfg.HORIZON

    env_h = _build_eval_env(N, seed=10)
    reward_h, _, _ = run_policy(env_h, unicycle_heuristic_action, steps)

    env_r = _build_eval_env(N, seed=10)
    torch.manual_seed(0)
    reward_r, _, _ = run_policy(env_r, lambda obs: torch.rand(N, env_r.act_dim) * 2 - 1, steps)

    return {"heuristic": reward_h.mean().item() / steps, "random": reward_r.mean().item() / steps}


def plot_training_curves(metrics_path, out_path, baseline_rewards=None):
    env_steps, reward, success_rate, ep_len, cost, violation_rate = [], [], [], [], [], []
    with open(metrics_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            env_steps.append(int(row["env_steps"]))
            reward.append(float(row["reward_per_step"]))
            success_rate.append(float(row["success_rate"]))
            ep_len.append(float(row["mean_episode_length"]))
            cost.append(float(row["cost_per_step"]))
            violation_rate.append(float(row["violation_rate"]))

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    axes[0].plot(env_steps, reward, label="trained policy")
    if baseline_rewards:
        colors = {"heuristic": "purple", "random": "gray"}
        for name, value in baseline_rewards.items():
            axes[0].axhline(value, color=colors.get(name, "black"), linestyle="--", label=f"{name} baseline")
    axes[0].set_ylabel("Reward / step")
    axes[0].set_title("Training curves (task performance)")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(env_steps, success_rate, color="green")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(env_steps, ep_len, color="teal")
    axes[2].set_ylabel("Mean episode length")

    axes[3].plot(env_steps, cost, color="red", label="cost / step")
    axes[3].plot(env_steps, violation_rate, color="darkorange", label="violation rate")
    axes[3].set_ylabel("Cost / violation")
    axes[3].set_xlabel("Env steps")
    axes[3].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def plot_ppo_diagnostics(metrics_path, out_path):
    """PPO health signals, as distinct from task performance above: is the
    optimizer itself behaving well, independent of whether the task is
    easy or hard."""
    env_steps, lr, policy_loss, value_loss, entropy, approx_kl, clip_frac = [], [], [], [], [], [], []
    with open(metrics_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            env_steps.append(int(row["env_steps"]))
            lr.append(float(row["lr"]))
            policy_loss.append(float(row["policy_loss"]))
            value_loss.append(float(row["value_loss"]))
            entropy.append(float(row["entropy"]))
            approx_kl.append(float(row["approx_kl"]))
            clip_frac.append(float(row["clip_frac"]))

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes[0, 0].plot(env_steps, policy_loss)
    axes[0, 0].set_title("Policy loss")
    axes[0, 1].plot(env_steps, value_loss, color="firebrick")
    axes[0, 1].set_title("Value loss")
    axes[1, 0].plot(env_steps, entropy, color="darkorange")
    axes[1, 0].set_title("Entropy (exploration)")
    axes[1, 1].plot(env_steps, approx_kl, color="purple")
    axes[1, 1].axhline(0.02, color="gray", linestyle="--", linewidth=1, label="rule-of-thumb target (~0.01-0.02)")
    axes[1, 1].set_title("Approx. KL per update")
    axes[1, 1].legend(fontsize=7)
    axes[2, 0].plot(env_steps, clip_frac, color="teal")
    axes[2, 0].set_title("Clip fraction")
    axes[2, 0].set_ylim(-0.05, 1.05)
    axes[2, 1].plot(env_steps, lr, color="black")
    axes[2, 1].set_title("Learning rate")

    for ax in axes[-1, :]:
        ax.set_xlabel("Env steps")
    fig.suptitle("PPO health diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved PPO diagnostics to {out_path}")


def plot_trajectory(checkpoint_path, out_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Evaluate under the dynamics the checkpoint was trained with, not
    # whatever config.py says now. Checkpoints saved before the "dynamics"
    # key existed fall back to config.py, with a warning.
    dyn = checkpoint.get("dynamics")
    if dyn is None:
        print(f"WARNING: {checkpoint_path} has no saved dynamics -- using the current config.py values, "
              "which may not match what it was trained with")
        dyn = {"dt": cfg.DT, "k": cfg.K_THRUST, "c": cfg.DRAG_COEF,
               "w_max": cfg.W_MAX, "v_max": cfg.V_MAX, "thrust_mode": cfg.THRUST_MODE,
               "reverse_scale": cfg.REVERSE_THRUST_SCALE}
    # Checkpoints from before reverse_scale existed were trained with
    # symmetric thrust.
    dyn.setdefault("reverse_scale", 1.0)
    print(f"Trajectory dynamics: {dyn}")

    env = PointToGoal(
        N=1, dev="cpu", dt=dyn["dt"], horizon=cfg.HORIZON,
        k=dyn["k"], c=dyn["c"], w_max=dyn["w_max"], v_max=dyn["v_max"], thrust_mode=dyn["thrust_mode"],
        reverse_scale=dyn["reverse_scale"],
        world_half_extent=cfg.WORLD_HALF_EXTENT, goal_radius=cfg.GOAL_RADIUS, goal_bonus=cfg.GOAL_BONUS,
        num_hazards=cfg.NUM_HAZARDS, hazard_radius=cfg.HAZARD_RADIUS,
        placement_resample_rounds=cfg.PLACEMENT_RESAMPLE_ROUNDS,
        randomize_start_pos=cfg.RANDOMIZE_START_POS,
        safety_mode=cfg.SAFETY_MODE, safety_margin=cfg.SAFETY_MARGIN,
        seed=cfg.SEED + 1000,  # a scenario not seen during training
    )

    net = ActorCritic(env.obs_dim, env.act_dim, cfg.HIDDEN_SIZE, cfg.LOG_STD_INIT).to(env.dev)
    net.load_state_dict(checkpoint["model"])
    net.eval()

    # Observations were normalized during training -- reuse the exact
    # trained running stats, frozen (no further .update() calls), or the
    # network would be fed inputs on the wrong scale.
    obs_rms = RunningMeanStd(env.obs_dim, env.dev)
    obs_rms.mean = checkpoint["obs_rms_mean"]
    obs_rms.var = checkpoint["obs_rms_var"]
    obs_rms.count = checkpoint["obs_rms_count"]

    raw_obs = env.reset()
    positions = [env.pos[0].numpy().copy()]
    headings = [env.th[0].item()]
    goals = [env.goal[0].numpy().copy()]
    hazards = env.hazards[0].numpy().copy()
    success = False

    with torch.no_grad():
        for _ in range(cfg.HORIZON):
            obs = obs_rms.normalize(raw_obs, cfg.OBS_CLIP)
            action, _, _ = net.act(obs, deterministic=True)
            raw_obs, reward, cost, terminated, truncated, info = env.step(action)
            if terminated[0].item() or truncated[0].item():
                success = bool(terminated[0].item())
                break  # env has already been reset internally -- don't plot that frame
            positions.append(env.pos[0].numpy().copy())
            headings.append(env.th[0].item())
            goals.append(env.goal[0].numpy().copy())

    positions = np.array(positions)
    headings = np.array(headings)
    goals = np.array(goals)

    unique_goals = [goals[0]]
    for g in goals[1:]:
        if not np.allclose(g, unique_goals[-1], atol=1e-6):
            unique_goals.append(g)
    unique_goals = np.array(unique_goals)

    fig, ax = plt.subplots(figsize=(6, 6))
    for hz in hazards:
        ax.add_patch(Circle(hz, cfg.HAZARD_RADIUS, color="red", alpha=0.3))
    ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=1, label="Agent trajectory")
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="Start")
    ax.plot(unique_goals[:, 0], unique_goals[:, 1], "g*", markersize=15, label="Goal(s)")

    step = max(1, len(positions) // 30)
    arrow_len = 0.2
    for i in range(0, len(positions) - 1, step):
        dx, dy = np.cos(headings[i]) * arrow_len, np.sin(headings[i]) * arrow_len
        ax.arrow(positions[i, 0], positions[i, 1], dx, dy, head_width=0.08, color="black", alpha=0.6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Trained policy -- {'reached goal' if success else 'timed out'}")
    ax.set_aspect("equal")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved trajectory plot to {out_path}")


def plot_checkpoint(checkpoint_path, out_dir=None, metrics_path=None, baselines=None):
    out_dir = out_dir or os.path.dirname(checkpoint_path) or "."
    # Named after the checkpoint (matching scripts/train.py's metrics naming
    # and the plot filenames below), so plotting several checkpoints (e.g.
    # one per seed) in the same directory doesn't overwrite each other.
    checkpoint_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    metrics_path = metrics_path or os.path.join(out_dir, f"{checkpoint_base}_metrics.csv")

    plot_trajectory(checkpoint_path, os.path.join(out_dir, f"{checkpoint_base}_trajectory.png"))
    if os.path.exists(metrics_path):
        baselines = baselines or compute_reference_baselines()
        plot_training_curves(metrics_path, os.path.join(out_dir, f"{checkpoint_base}_training_curves.png"), baselines)
        plot_ppo_diagnostics(metrics_path, os.path.join(out_dir, f"{checkpoint_base}_ppo_diagnostics.png"))
    else:
        print(f"No metrics.csv found at {metrics_path}, skipping training curves / diagnostics.")
    return baselines


def main():
    args = parse_args()
    if args.run:
        run_tracking.load_run_config(cfg, args.run)
        seeds = args.seeds or run_tracking.read_run_info(args.run)["seeds"]
        baselines = None  # same config for every seed -> compute once, reuse
        for seed in seeds:
            print(f"\n=== seed {seed} ===")
            cfg.SEED = seed
            checkpoint_path = os.path.join(run_tracking.seed_dir(args.run, seed), "model.pt")
            if not os.path.exists(checkpoint_path):
                print(f"No checkpoint at {checkpoint_path}, skipping (seed unfinished?)")
                continue
            baselines = plot_checkpoint(checkpoint_path, baselines=baselines)
        return

    if args.preset:
        cfg.apply_preset(args.preset)
    plot_checkpoint(args.checkpoint or cfg.MODEL_SAVE_PATH, args.out_dir, args.metrics)


if __name__ == "__main__":
    main()
