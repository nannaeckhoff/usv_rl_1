# Everything needed to judge a training run, all seeds together (runs on any
# machine -- numpy + matplotlib only). Reads <run>/seed_*/model_metrics.csv
# (written during training) and <run>/eval.npz (from scripts/evaluate.py).
#
#   <run>/seeds_training.png  training curves + PPO diagnostics, one line per seed
#   <run>/eval.png            success, speed vs. heuristic, hazard distance, violations
#   <run>/eval_traj.png       the same scenarios driven by every seed (+ heuristic)
#
# and prints a per-seed table and a list of checks. Several runs (e.g. a
# hyperparameter sweep) are compared in one table at the end.
#
# Run: python scripts/plot_eval.py --run results/runs/<run> [results/runs/<run2> ...]

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

TRAIN_KEYS = [("reward_per_step", "Reward / step"), ("success_rate", "Success rate"),
              ("mean_episode_length", "Episode length"), ("cost_per_step", "Cost / step"),
              ("violation_rate", "Violation rate (steps)"), ("entropy", "Entropy"),
              ("approx_kl", "Approx. KL"), ("value_loss", "Value loss"), ("clip_frac", "Clip fraction")]
LOG_Y = {"mean_episode_length", "value_loss"}


def color(name):
    return "black" if name == "heuristic" else plt.cm.tab10(int(name.split("_")[1]) % 10)


def load_metrics(run):
    out = {}
    for path in sorted(glob.glob(os.path.join(run, "seed_*", "model_metrics.csv"))):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        out[os.path.basename(os.path.dirname(path))] = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return out


def load_eval(run):
    z = np.load(os.path.join(run, "eval.npz"))
    ev = {n: {k.split("__")[1]: z[k] for k in z.files if k.startswith(n + "__")} for n in z["names"]}
    return ev, float(z["hazard_radius"])


def converged_update(m):
    """First update where reward/step reaches 95% of its final value (mean of last 10)."""
    r = m["reward_per_step"]
    return int(m["update"][np.argmax(r >= 0.95 * r[-10:].mean())])


def summarize(ev, metrics):
    """One row per policy. Ratios to the heuristic use episodes both succeeded in."""
    heur = ev["heuristic"]
    rows = {}
    for name, e in ev.items():
        s, b, viol = e["success"] > 0, e["blocked"] > 0, e["hazard_steps"] > 0
        both = s & (heur["success"] > 0)
        row = {
            "success%": 100 * s.mean(),
            "succ_blk%": 100 * s[b].mean() if b.any() else np.nan,
            "ep_len": e["length"][s].mean(),
            "len/heur": np.median(e["length"][both] / heur["length"][both]),
            "viol_ep%": 100 * viol.mean(),
            "viol_blk%": 100 * viol[b].mean() if b.any() else np.nan,
            "haz_steps": e["hazard_steps"].mean(),
            "cost/ep": e["cost"].mean(),
            "dmin_p5": np.percentile(e["min_hazard_dist"][b], 5) if b.any() else np.nan,
        }
        if name in metrics:
            m = metrics[name]
            row.update({"conv_upd": converged_update(m), "entropy": m["entropy"][-1],
                        "max_kl": m["approx_kl"][10:].max()})
        rows[name] = row
    return rows


def print_table(rows, title):
    cols = list(dict.fromkeys(k for r in rows.values() for k in r))
    print(f"\n{title}")
    print(f"{'':>12}" + "".join(f"{c:>11}" for c in cols))
    for name, r in rows.items():
        print(f"{name:>12}" + "".join(f"{r.get(c, np.nan):>11.3f}" for c in cols))


def seed_stats(rows):
    seeds = [r for n, r in rows.items() if n != "heuristic"]
    keys = [k for k in seeds[0]]
    return {k: (np.mean([r[k] for r in seeds]), np.std([r[k] for r in seeds])) for k in keys}


def print_checks(rows, stats):
    seeds = {n: r for n, r in rows.items() if n != "heuristic"}
    checks = [
        ("every seed succeeds in >= 99% of episodes", all(r["success%"] >= 99 for r in seeds.values())),
        ("... also when a hazard blocks the straight path",  # nan = no hazards (Stage A)
         all(np.isnan(r["succ_blk%"]) or r["succ_blk%"] >= 99 for r in seeds.values())),
        ("faster than the heuristic (median len/heur < 1)", stats["len/heur"][0] < 1.0),
        ("seeds agree: episode-length std < 2% of mean", stats["ep_len"][1] < 0.02 * stats["ep_len"][0]),
        ("stable updates: approx KL < 0.02 after update 10", all(r["max_kl"] < 0.02 for r in seeds.values())),
    ]
    print("\nChecks:")
    for text, ok in checks:
        print(f"  [{'OK' if ok else '!!'}] {text}")


def plot_training(metrics, out):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for ax, (key, label) in zip(axes.flat, TRAIN_KEYS):
        for name, m in metrics.items():
            ax.plot(m["env_steps"], m[key], color=color(name), lw=1, label=name)
        if key in LOG_Y:
            ax.set_yscale("log")
        if key == "approx_kl":
            ax.axhline(0.02, color="gray", ls="--", lw=1)  # rule-of-thumb upper limit
        ax.set_title(label)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Env steps")
    fig.suptitle("Training, all seeds")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_eval(ev, rows, hazard_radius, out):
    names = list(ev)
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.bar(x - 0.2, [rows[n]["success%"] for n in names], 0.4, label="all episodes")
    ax.bar(x + 0.2, [rows[n]["succ_blk%"] for n in names], 0.4, label="hazard on path")
    ax.set_xticks(x, names)
    ax.set_ylim(min(90, min(r["success%"] for r in rows.values()) - 1), 100.5)
    ax.set_title("Success rate (%)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    heur = ev["heuristic"]
    bins = np.linspace(0, 2, 81)
    for n in names[1:]:
        both = (ev[n]["success"] > 0) & (heur["success"] > 0)
        ax.hist(np.clip(ev[n]["length"][both] / heur["length"][both], 0, 2), bins, histtype="step",
                color=color(n), label=n)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_title("Episode length / heuristic's, same scenario (< 1 = faster)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(0, 2 * hazard_radius + 1, 61)
    for n in names:
        b = ev[n]["blocked"] > 0
        ax.hist(ev[n]["min_hazard_dist"][b], bins, histtype="step", color=color(n), label=n,
                ls="--" if n == "heuristic" else "-")
    ax.axvline(hazard_radius, color="red", lw=1, label="hazard edge")
    ax.set_title("Closest distance to hazard centre, hazard-on-path episodes")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(x - 0.2, [rows[n]["viol_ep%"] for n in names], 0.4, label="all episodes")
    ax.bar(x + 0.2, [rows[n]["viol_blk%"] for n in names], 0.4, label="hazard on path")
    ax.set_xticks(x, names)
    ax.set_title("Episodes that enter a hazard (%)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_trajectories(ev, hazard_radius, out):
    first = next(iter(ev.values()))
    k = len(first["traj"])
    cols = 4
    fig, axes = plt.subplots(-(-k // cols), cols, figsize=(4 * cols, 4 * -(-k // cols)), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= k:
            ax.axis("off")
            continue
        for hz in first["traj_hazards"][i]:
            ax.add_patch(Circle(hz, hazard_radius, color="red", alpha=0.3))
        for n, e in ev.items():
            ax.plot(*e["traj"][i].T, color=color(n), lw=1, ls="--" if n == "heuristic" else "-", label=n)
        ax.plot(*first["traj"][i][0], "go")
        ax.plot(*first["traj_goal"][i], "g*", ms=14)
        ax.set_aspect("equal")
        ax.set_title("hazard on path" if first["traj_blocked"][i] else "clear path", fontsize=9)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True, help="one or more results/runs/<run> folders")
    runs = p.parse_args().run

    comparison = {}
    for run in runs:
        print(f"\n===== {run} =====")
        metrics = load_metrics(run)
        plot_training(metrics, os.path.join(run, "seeds_training.png"))
        if not os.path.exists(os.path.join(run, "eval.npz")):
            print("No eval.npz -- run scripts/evaluate.py on the GPU machine first. Training plot saved.")
            continue
        ev, hazard_radius = load_eval(run)
        rows = summarize(ev, metrics)
        stats = seed_stats(rows)
        print_table(rows, "Deterministic evaluation (same scenarios for every policy):")
        print(f"{'seeds mean':>12}" + "".join(f"{m:>11.3f}" for m, _ in stats.values()))
        print(f"{'seeds std':>12}" + "".join(f"{s:>11.3f}" for _, s in stats.values()))
        print_checks(rows, stats)
        plot_eval(ev, rows, hazard_radius, os.path.join(run, "eval.png"))
        plot_trajectories(ev, hazard_radius, os.path.join(run, "eval_traj.png"))
        print(f"Plots saved in {run}")
        comparison[os.path.basename(run.rstrip("/\\"))] = {k: m for k, (m, _) in stats.items()}

    if len(comparison) > 1:
        print_table(comparison, "Run comparison (mean over seeds):")


if __name__ == "__main__":
    main()
