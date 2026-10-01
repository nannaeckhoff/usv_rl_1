# PPO training for the GPU PointToGoal environment (environment.py), in the
# structure of cleanRL's ppo_continuous_action: observation normalization,
# LR annealing, advantage normalization, gradient clipping, and correct
# terminated/truncated handling in GAE. Pure PyTorch, no external RL
# library. All hyperparameters come from config.py. Environment, network
# and update all stay on the GPU; the only CPU syncs are the scalars
# printed/logged once per round.
#
# Run: python scripts/train.py [--preset stage_a] [--seeds 1 2 3 4 5]
#   no --preset  -- trains with the constants as they currently stand in
#                   config.py (single run, config.SEED)
#   --preset     -- overwrite those constants with a named preset from
#                   config.PRESETS first (e.g. stage_a, stage_b)
#   --seeds      -- train once per seed instead of a single config.SEED run;
#                   each seed gets its own checkpoint/metrics.csv, and a
#                   mean +/- std summary is printed across seeds at the end

import csv
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm

import config as cfg
from environment import PointToGoal


# --- policy/value network: small Tanh MLP + a learnable log_std ---

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, log_std_init):
        super().__init__()

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, out_dim),
            )

        self.actor_mean = mlp(act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.critic = mlp(1)

    def forward(self, obs):
        mean = self.actor_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def act(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        if deterministic:
            return mean, None, value
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class RunningMeanStd:
    """Tracks a running mean/variance of observations (Welford's parallel
    algorithm, batched), so the network always sees roughly zero-mean,
    unit-variance inputs even though the raw obs scale depends on
    world_half_extent/num_hazards/etc. Updated with every batch of
    observations seen; normalization uses whatever it has seen so far."""

    def __init__(self, shape, device, eps=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x, clip):
        normed = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normed, -clip, clip)


def compute_gae(rewards, costs, values, dones, last_value):
    """rewards/costs/values/dones: (T, N). Combines reward and cost into the
    training signal here -- the only place they're combined; the env itself
    always keeps them separate. dones[t] (terminated | truncated) marks an
    episode boundary and masks out the bootstrap term there. The reward
    itself has already been corrected (see train()) to bootstrap through a
    truncation -- by the time it gets here, "terminated" and "truncated"
    look the same: just don't reuse values[t+1] across the boundary."""
    signal = rewards - cfg.LAMBDA_COST * costs
    T, N = signal.shape
    advantages = torch.zeros_like(signal)
    last_gae = torch.zeros(N, device=signal.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = signal[t] + cfg.GAMMA * next_value * next_nonterminal - values[t]
        last_gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train():
    torch.manual_seed(cfg.SEED)

    env = PointToGoal(
        N=cfg.N_ENVS, dev=cfg.DEVICE, dt=cfg.DT, horizon=cfg.HORIZON,
        dynamics=cfg.DYNAMICS, k=cfg.K_THRUST, c=cfg.DRAG_COEF, w_max=cfg.W_MAX,
        world_half_extent=cfg.WORLD_HALF_EXTENT, goal_radius=cfg.GOAL_RADIUS, goal_bonus=cfg.GOAL_BONUS,
        num_hazards=cfg.NUM_HAZARDS, hazard_radius=cfg.HAZARD_RADIUS,
        placement_resample_rounds=cfg.PLACEMENT_RESAMPLE_ROUNDS,
        randomize_start_pos=cfg.RANDOMIZE_START_POS, seed=cfg.SEED,
    )
    net = ActorCritic(env.obs_dim, env.act_dim, cfg.HIDDEN_SIZE, cfg.LOG_STD_INIT).to(env.dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR)
    obs_rms = RunningMeanStd(env.obs_dim, env.dev)

    T, N = cfg.ROLLOUT_STEPS, env.N
    obs_buf = torch.zeros(T, N, env.obs_dim, device=env.dev)
    act_buf = torch.zeros(T, N, env.act_dim, device=env.dev)
    logp_buf = torch.zeros(T, N, device=env.dev)
    val_buf = torch.zeros(T, N, device=env.dev)
    rew_buf = torch.zeros(T, N, device=env.dev)
    cost_buf = torch.zeros(T, N, device=env.dev)
    done_buf = torch.zeros(T, N, device=env.dev)

    save_dir = os.path.dirname(cfg.MODEL_SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    # Named after the checkpoint, not a fixed "metrics.csv" -- otherwise
    # two runs with different MODEL_SAVE_PATHs (e.g. different seeds) would
    # silently overwrite each other's metrics in the same directory.
    checkpoint_base = os.path.splitext(os.path.basename(cfg.MODEL_SAVE_PATH))[0]
    metrics_path = os.path.join(save_dir or ".", f"{checkpoint_base}_metrics.csv")
    metrics_fields = [
        "update", "env_steps", "lr", "reward_per_step", "cost_per_step",
        "violation_rate", "success_rate", "mean_episode_length",
        "steps_per_sec", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
    ]
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
    metrics_writer.writeheader()

    raw_obs = env.reset()
    obs_rms.update(raw_obs)
    obs = obs_rms.normalize(raw_obs, cfg.OBS_CLIP)

    pbar = tqdm(range(1, cfg.TOTAL_UPDATES + 1), desc="training", unit="update")
    for update in pbar:
        round_start = time.perf_counter()

        if cfg.ANNEAL_LR:
            frac = 1.0 - (update - 1) / cfg.TOTAL_UPDATES
            optimizer.param_groups[0]["lr"] = frac * cfg.LR

        reward_sum = torch.zeros(N, device=env.dev)
        cost_sum = torch.zeros(N, device=env.dev)
        violation_sum = torch.zeros(N, device=env.dev)
        terminated_sum = torch.zeros(N, device=env.dev)
        done_sum = torch.zeros(N, device=env.dev)
        episode_length_sum = torch.zeros(N, device=env.dev)  # only counted where an episode actually ended this step

        # --- rollout phase: fixed number of steps, current policy, no updates ---
        with torch.no_grad():
            for t in range(T):
                action, log_prob, value = net.act(obs)
                raw_next_obs, reward, cost, terminated, truncated, info = env.step(action)

                # Bootstrap through truncation instead of treating it like a
                # real terminal state: add the value of the state the agent
                # would have continued into (pre-reset) straight onto the
                # reward. `truncated.float()` is 0 everywhere else, so this
                # is a no-op for those envs -- no branching needed.
                final_obs = obs_rms.normalize(info["final_obs"], cfg.OBS_CLIP)
                _, _, final_value = net.forward(final_obs)
                reward = reward + cfg.GAMMA * final_value * truncated.float()

                done = terminated | truncated

                obs_buf[t] = obs
                act_buf[t] = action
                logp_buf[t] = log_prob
                val_buf[t] = value
                rew_buf[t] = reward     # reward and cost stored separately --
                cost_buf[t] = cost      # combined only in compute_gae()
                done_buf[t] = done.float()

                reward_sum += reward
                cost_sum += cost
                violation_sum += info["in_hazard"].float()
                terminated_sum += terminated.float()
                done_sum += done.float()
                episode_length_sum += info["episode_length"].float() * done.float()

                obs_rms.update(raw_next_obs)
                obs = obs_rms.normalize(raw_next_obs, cfg.OBS_CLIP)

            _, _, last_value = net.forward(obs)

        # --- PPO update: clipped objective, minibatch epochs ---
        advantages, returns = compute_gae(rew_buf, cost_buf, val_buf, done_buf, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = T * N
        mb_size = batch_size // cfg.NUM_MINIBATCHES
        obs_flat = obs_buf.reshape(batch_size, -1)
        act_flat = act_buf.reshape(batch_size, -1)
        logp_flat = logp_buf.reshape(batch_size)
        adv_flat = advantages.reshape(batch_size)
        ret_flat = returns.reshape(batch_size)

        policy_loss_sum = value_loss_sum = entropy_sum = approx_kl_sum = clip_frac_sum = 0.0
        n_minibatches = 0
        for _ in range(cfg.EPOCHS):
            perm = torch.randperm(batch_size, device=env.dev)
            for start in range(0, batch_size - mb_size + 1, mb_size):
                idx = perm[start:start + mb_size]
                new_log_probs, entropy, values = net.evaluate_actions(obs_flat[idx], act_flat[idx])
                log_ratio = new_log_probs - logp_flat[idx]
                ratio = torch.exp(log_ratio)

                surr1 = ratio * adv_flat[idx]
                surr2 = torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS) * adv_flat[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values - ret_flat[idx]).pow(2).mean()
                entropy_mean = entropy.mean()

                loss = policy_loss + cfg.VF_COEF * value_loss - cfg.ENTROPY_COEF * entropy_mean

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()

                # PPO health diagnostics, not part of the loss: approx_kl
                # (Schulman's low-variance k3 estimator -- always >= 0) is
                # how far the policy actually moved this minibatch; too high
                # and the update is too aggressive/unstable. clip_frac is
                # how often the clipping was actually needed -- near 0 means
                # clip_eps is barely doing anything, near 1 means updates
                # are being clamped almost everywhere.
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.CLIP_EPS).float().mean()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy_mean.item()
                approx_kl_sum += approx_kl.item()
                clip_frac_sum += clip_frac.item()
                n_minibatches += 1

        # --- logging: everything above ran without touching the CPU; this
        # is the one sync per round, after the rollout and the update ---
        elapsed = time.perf_counter() - round_start
        done_total = done_sum.sum().item()
        row = {
            "update": update,
            "env_steps": update * T * N,
            "lr": optimizer.param_groups[0]["lr"],
            "reward_per_step": reward_sum.mean().item() / T,
            "cost_per_step": cost_sum.mean().item() / T,
            "violation_rate": violation_sum.mean().item() / T,
            "success_rate": terminated_sum.sum().item() / done_total if done_total > 0 else float("nan"),
            "mean_episode_length": episode_length_sum.sum().item() / done_total if done_total > 0 else float("nan"),
            "steps_per_sec": (T * N) / elapsed,
            "policy_loss": policy_loss_sum / n_minibatches,
            "value_loss": value_loss_sum / n_minibatches,
            "entropy": entropy_sum / n_minibatches,
            "approx_kl": approx_kl_sum / n_minibatches,
            "clip_frac": clip_frac_sum / n_minibatches,
        }
        metrics_writer.writerow(row)
        metrics_file.flush()

        if update % cfg.SAVE_EVERY == 0 or update == cfg.TOTAL_UPDATES:
            torch.save({
                "model": net.state_dict(),
                "obs_rms_mean": obs_rms.mean,
                "obs_rms_var": obs_rms.var,
                "obs_rms_count": obs_rms.count,
            }, cfg.MODEL_SAVE_PATH)

    metrics_file.close()
    print(f"\nTraining complete. Model saved to: {cfg.MODEL_SAVE_PATH}")
    print(f"Per-round metrics logged to: {metrics_path}")
    return metrics_path


def _final_metrics_row(metrics_path):
    with open(metrics_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def _mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n  # population std -- just describing these n runs
    return mean, var ** 0.5


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default=None, choices=list(cfg.PRESETS),
                         help="apply a named config preset before training (see config.PRESETS)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="train once per seed (overrides config.SEED), each to its own "
                              "checkpoint/metrics file, then report mean +/- std across seeds")
    args = parser.parse_args()
    if args.preset:
        cfg.apply_preset(args.preset)

    if args.seeds is not None:
        base, ext = os.path.splitext(cfg.MODEL_SAVE_PATH)
        per_seed = []
        for seed in args.seeds:
            print(f"\n=== seed {seed} ===")
            cfg.SEED = seed
            cfg.MODEL_SAVE_PATH = f"{base}_seed{seed}{ext}"
            metrics_path = train()
            final = _final_metrics_row(metrics_path)
            per_seed.append({
                "seed": seed,
                "reward_per_step": float(final["reward_per_step"]),
                "success_rate": float(final["success_rate"]),
                "mean_episode_length": float(final["mean_episode_length"]),
            })

        print("\n=== Per-seed final result ===")
        for r in per_seed:
            print(f"  seed {r['seed']:>3}: reward/step={r['reward_per_step']:.3f}  "
                  f"success_rate={r['success_rate']:.2f}  mean_ep_len={r['mean_episode_length']:.1f}")

        print("\n=== Across seeds (mean +/- std) ===")
        for key in ("reward_per_step", "success_rate", "mean_episode_length"):
            mean, std = _mean_std([r[key] for r in per_seed])
            print(f"  {key}: {mean:.3f} +/- {std:.3f}")
    else:
        train()
