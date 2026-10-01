# Aggregate plot across multiple seed runs (scripts/train.py --seeds), so
# you can see whether a result is consistent across seeds or a fluke of one.
# Reads the per-seed metrics.csv files scripts/train.py writes
# (baseline_seedN_metrics.csv) and plots mean +/- std across seeds for
# reward, success rate, cost and violation rate, aligned on env_steps.
#
# Run: python scripts/plot_seeds.py --preset stage_b
#      python scripts/plot_seeds.py --preset stage_b --seeds 1 2 3

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="Plot mean +/- std training curves across several seed runs.")
    p.add_argument("--preset", type=str, default=None, choices=list(cfg.PRESETS))
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="which seeds to include; default: autodetect all baseline_seed*_metrics.csv "
                         "next to the checkpoint")
    p.add_argument("--out", type=str, default=None, help="defaults to <checkpoint_dir>/baseline_seeds.png")
    return p.parse_args()


def find_seed_metrics(base_path, seeds):
    base, ext = os.path.splitext(base_path)
    if seeds is not None:
        paths = [f"{base}_seed{s}_metrics.csv" for s in seeds]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing metrics files: {missing}")
        return seeds, paths

    found = sorted(glob.glob(f"{base}_seed*_metrics.csv"))
    seed_re = re.compile(rf"{re.escape(os.path.basename(base))}_seed(\d+)_metrics\.csv$")
    seeds, paths = [], []
    for path in found:
        m = seed_re.search(os.path.basename(path))
        if m:
            seeds.append(int(m.group(1)))
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No seed metrics files found matching {base}_seed*_metrics.csv")
    return seeds, paths


def load_metrics(path):
    env_steps, reward, success, cost, violation = [], [], [], [], []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            env_steps.append(int(row["env_steps"]))
            reward.append(float(row["reward_per_step"]))
            success.append(float(row["success_rate"]))
            cost.append(float(row["cost_per_step"]))
            violation.append(float(row["violation_rate"]))
    return (np.array(env_steps), np.array(reward), np.array(success),
            np.array(cost), np.array(violation))


def main():
    args = parse_args()
    if args.preset:
        cfg.apply_preset(args.preset)

    seeds, paths = find_seed_metrics(cfg.MODEL_SAVE_PATH, args.seeds)
    print(f"Found {len(paths)} seed run(s): {seeds}")

    runs = [load_metrics(p) for p in paths]
    # Same ROLLOUT_STEPS/TOTAL_UPDATES across seeds -> same env_steps grid in
    # principle, but truncate to the shortest run in case one is still in
    # progress or was stopped early, so np.stack doesn't choke on ragged
    # lengths.
    min_len = min(len(r[0]) for r in runs)
    if min_len == 0:
        raise ValueError("At least one seed run has no logged rows yet.")
    env_steps = runs[0][0][:min_len]
    reward = np.stack([r[1][:min_len] for r in runs])
    success = np.stack([r[2][:min_len] for r in runs])
    cost = np.stack([r[3][:min_len] for r in runs])
    violation = np.stack([r[4][:min_len] for r in runs])

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    series = [
        (axes[0], reward, "Reward / step", "tab:blue"),
        (axes[1], success, "Success rate", "tab:green"),
        (axes[2], cost, "Cost / step", "tab:red"),
        (axes[3], violation, "Violation rate", "tab:orange"),
    ]
    for ax, data, ylabel, color in series:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(env_steps, mean, color=color, label=f"mean (n={len(seeds)})")
        ax.fill_between(env_steps, mean - std, mean + std, color=color, alpha=0.2, label="+/- 1 std")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[-1].set_xlabel("Env steps")
    axes[0].set_title(f"Across-seed training curves ({cfg.DYNAMICS}, {len(seeds)} seeds: {seeds})")

    fig.tight_layout()
    out_path = args.out or os.path.join(os.path.dirname(cfg.MODEL_SAVE_PATH) or ".", "baseline_seeds.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved seed-comparison plot to {out_path}")


if __name__ == "__main__":
    main()
