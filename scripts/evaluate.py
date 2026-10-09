# Deterministic evaluation of every seed of a training run (run on the GPU
# machine). Each seed's policy -- and the turn-to-goal heuristic, as a
# reference -- plays the same fixed set of scenarios, one episode each, so
# seeds can be compared episode by episode. Per-episode results plus a few
# full trajectories are saved to <run>/eval.npz; scripts/plot_eval.py turns
# that into plots and tables and needs only numpy + matplotlib.
#
# Run: python scripts/evaluate.py --run results/runs/<run> [--episodes 10000] [--device cuda]

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import config as cfg
import run_tracking
from environment import PointToGoal
from train import ActorCritic, RunningMeanStd
from test_env import unicycle_heuristic_action, _path_blocked

EVAL_SEED = 12345  # no training seed -> scenarios never seen during training
N_TRAJ, N_TRAJ_CLEAR = 8, 2  # episodes whose full path is stored for the trajectory plot (of them, clear path)


def make_env(n, device):
    """Same seed every time -> every policy gets exactly the same scenarios."""
    return PointToGoal(
        N=n, dev=device, dt=cfg.DT, horizon=cfg.HORIZON,
        k=cfg.K_THRUST, c=cfg.DRAG_COEF, w_max=cfg.W_MAX, v_max=cfg.V_MAX, thrust_mode=cfg.THRUST_MODE,
        reverse_scale=cfg.REVERSE_THRUST_SCALE,
        world_half_extent=cfg.WORLD_HALF_EXTENT, goal_radius=cfg.GOAL_RADIUS, goal_bonus=cfg.GOAL_BONUS,
        num_hazards=cfg.NUM_HAZARDS, hazard_radius=cfg.HAZARD_RADIUS,
        placement_resample_rounds=cfg.PLACEMENT_RESAMPLE_ROUNDS, hazard_on_path_prob=cfg.HAZARD_ON_PATH_PROB,
        randomize_start_pos=cfg.RANDOMIZE_START_POS,
        safety_mode=cfg.SAFETY_MODE, safety_margin=cfg.SAFETY_MARGIN, seed=EVAL_SEED,
    )


def load_policy(path, env):
    ckpt = torch.load(path, map_location=env.dev)
    net = ActorCritic(env.obs_dim, env.act_dim, cfg.HIDDEN_SIZE, cfg.LOG_STD_INIT).to(env.dev)
    net.load_state_dict(ckpt["model"])
    rms = RunningMeanStd(env.obs_dim, env.dev)
    rms.mean, rms.var = ckpt["obs_rms_mean"], ckpt["obs_rms_var"]  # frozen training stats
    return lambda obs: net.act(rms.normalize(obs, cfg.OBS_CLIP), deterministic=True)[0]


@torch.no_grad()
def play(policy, n, device, traj_idx):
    """One episode per env. The env auto-resets finished envs; everything
    after an env's first episode is masked out with `active`."""
    env = make_env(n, device)
    obs = env.reset()
    active = torch.ones(n, dtype=torch.bool, device=device)
    out = {k: torch.zeros(n, device=device) for k in ("success", "length", "hazard_steps", "cost")}
    out["min_hazard_dist"] = torch.full((n,), float("inf"), device=device)
    out["start_goal_dist"] = (env.goal - env.pos).norm(dim=-1)
    out["blocked"] = _path_blocked(env).float()  # straight start->goal line goes through a hazard
    out["traj_blocked"] = out["blocked"][traj_idx]
    out["traj_goal"] = env.goal[traj_idx].clone()
    out["traj_hazards"] = env.hazards[traj_idx].clone()
    traj = torch.full((len(traj_idx), cfg.HORIZON + 1, 2), float("nan"), device=device)
    traj[:, 0] = env.pos[traj_idx]

    for t in range(cfg.HORIZON):
        obs, _, cost, terminated, truncated, info = env.step(policy(obs))
        a = active.float()
        out["length"] += a
        out["cost"] += a * cost
        out["hazard_steps"] += a * info["in_hazard"].float()
        out["success"] += a * terminated.float()
        out["min_hazard_dist"] = torch.where(active, torch.minimum(out["min_hazard_dist"], info["min_hazard_dist"]),
                                             out["min_hazard_dist"])
        done = terminated | truncated
        keep = (active & ~done)[traj_idx]  # on the final step env.pos is already the next episode's start
        traj[:, t + 1] = torch.where(keep.unsqueeze(-1), env.pos[traj_idx], traj[:, t + 1])
        active &= ~done
        if not active.any():
            break
    out["traj"] = traj
    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a results/runs/<run> folder from scripts/train.py")
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--device", default=cfg.DEVICE)
    args = p.parse_args()

    run_tracking.load_run_config(cfg, args.run)
    probe = make_env(args.episodes, args.device)
    probe.reset()
    blocked = _path_blocked(probe)
    # Mostly hazard-on-path scenarios; filled up with clear ones (all of them in Stage A)
    n_blocked = min(int(blocked.sum()), N_TRAJ - N_TRAJ_CLEAR)
    traj_idx = torch.cat([blocked.nonzero()[:n_blocked, 0], (~blocked).nonzero()[:N_TRAJ - n_blocked, 0]])

    policies = {"heuristic": unicycle_heuristic_action}
    for seed in run_tracking.read_run_info(args.run)["seeds"]:
        path = os.path.join(run_tracking.seed_dir(args.run, seed), "model.pt")
        if os.path.exists(path):
            policies[f"seed_{seed}"] = load_policy(path, probe)

    saved = {"names": np.array(list(policies)), "hazard_radius": cfg.HAZARD_RADIUS, "dt": cfg.DT}
    for name, policy in policies.items():
        res = play(policy, args.episodes, args.device, traj_idx)
        print(f"{name:>10}: success {100 * res['success'].mean():6.2f}%   "
              f"episodes with violation {100 * (res['hazard_steps'] > 0).mean():6.2f}%")
        saved.update({f"{name}__{k}": v for k, v in res.items()})

    out = os.path.join(args.run, "eval.npz")
    np.savez_compressed(out, **saved)
    print(f"Saved {out}  ->  python scripts/plot_eval.py --run {args.run}")


if __name__ == "__main__":
    main()
