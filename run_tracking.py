# Per-run bookkeeping for scripts/train.py, so any result can be traced
# back to exactly the config and code that produced it. Every training run
# gets its own folder that is never overwritten:
#
#   results/runs/<date>_<time>_<preset>_<safety mode>[_<name>]/
#       config.json       every config.py value, after preset + --set overrides
#       run_info.json     date, command, note, seeds, git commit/branch
#       uncommitted.diff  code changes not yet committed when the run started
#                         (only if there were any) -- the commit alone would
#                         point at the wrong code
#       seed_<N>/         model.pt, model_metrics.csv, plots
#
# and one row per run is appended to results/runs.csv -- the searchable
# index (open it in Excel and filter by safety_mode, lambda_cost, ...).
#
# To get a run's code back:  git checkout <commit>  then, if the run has
# one,  git apply results/runs/<run>/uncommitted.diff
#
# Runs are tracked in git (plots excluded, see .gitignore): train on the GPU
# machine, then  python scripts/sync_results.py  there, and  git pull  here.

import ast
import csv
import datetime
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join("results", "runs")
RUNS_INDEX = os.path.join("results", "runs.csv")

# config.py values copied into each runs.csv row (the full set is always in
# the run's config.json -- this is just what's worth filtering/sorting on).
INDEX_CONFIG_KEYS = [
    "SAFETY_MODE", "SAFETY_MARGIN", "LAMBDA_COST", "NUM_HAZARDS", "HAZARD_RADIUS",
    "LR", "ENTROPY_COEF", "CLIP_EPS", "GAMMA", "TOTAL_UPDATES", "ROLLOUT_STEPS", "N_ENVS",
]
FINAL_METRIC_KEYS = ["reward_per_step", "success_rate", "cost_per_step", "violation_rate", "mean_episode_length"]


def config_snapshot(cfg):
    """All UPPERCASE values from config.py. MODEL_SAVE_PATH is left out:
    it's set per seed by train.py, so the base value would only mislead."""
    return {k: v for k, v in vars(cfg).items() if k.isupper() and k != "MODEL_SAVE_PATH"}


def apply_overrides(cfg, assignments):
    """Apply `KEY=VALUE` strings (train.py --set) to config. VALUE is parsed
    as a Python literal (3e-4, 2, True, None, "x"), falling back to a plain
    string, so `SAFETY_MODE=proximity` works without extra quoting. Only
    keys that already exist in config.py are accepted, so a typo fails
    loudly instead of silently training with the default."""
    overrides = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key.isupper() or not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key!r}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        setattr(cfg, key, value)
        overrides[key] = value
    return overrides


# Config keys added after some runs were already made, with the value those
# older runs effectively used -- their config.json doesn't have the key, and
# today's config.py default would be wrong for them.
PRE_EXISTING_DEFAULTS = {
    "REVERSE_THRUST_SCALE": 1.0,  # thrust used to be symmetric
}


def load_run_config(cfg, run_dir):
    """Overwrite config with the values a run was trained with (for
    plotting/evaluating it later, independent of what config.py says now)."""
    for key, value in PRE_EXISTING_DEFAULTS.items():
        setattr(cfg, key, value)
    with open(os.path.join(run_dir, "config.json")) as f:
        for key, value in json.load(f).items():
            setattr(cfg, key, value)


def read_run_info(run_dir):
    with open(os.path.join(run_dir, "run_info.json")) as f:
        return json.load(f)


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:  # git not installed
        return ""
    return r.stdout


# results/ is tracked too (so runs can be synced between machines), but it
# isn't code -- unsynced runs must not end up in the next run's diff.
CODE_PATHSPEC = ["--", ".", ":(exclude)results"]


def git_state():
    """Current commit plus every uncommitted code change as one diff:
    tracked edits (staged or not) and new, untracked files (respecting
    .gitignore), everything except results/."""
    diff = _git("diff", "HEAD", "--binary", *CODE_PATHSPEC)
    for path in _git("ls-files", "--others", "--exclude-standard", *CODE_PATHSPEC).splitlines():
        diff += _git("diff", "--no-index", "--binary", "--", "/dev/null", path)
    return {
        "commit": _git("rev-parse", "HEAD").strip() or "unknown",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip() or "unknown",
        "dirty": bool(diff),
    }, diff


def create_run(cfg, preset, name, note, seeds, overrides):
    """Make the run folder and record config + code state in it. Call after
    the preset and overrides are applied, before training."""
    now = datetime.datetime.now()
    parts = [now.strftime("%Y-%m-%d_%H%M%S"), preset or "custom", str(cfg.SAFETY_MODE)]
    if name:
        parts.append(name)
    run_dir = os.path.join(RUNS_DIR, "_".join(parts))
    os.makedirs(run_dir, exist_ok=False)  # never reuse (and overwrite) an existing run

    git, diff = git_state()
    info = {
        "run_id": os.path.basename(run_dir),
        "started": now.isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(["python"] + sys.argv),  # quoted, so it can be pasted back in
        "note": note,
        "preset": preset,
        "seeds": seeds,
        "overrides": overrides,
        **git,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_snapshot(cfg), f, indent=2, default=str)
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    if diff:
        with open(os.path.join(run_dir, "uncommitted.diff"), "w", encoding="utf-8", newline="\n") as f:
            f.write(diff)

    print(f"Run folder: {run_dir}")
    if git["dirty"]:
        print("  (uncommitted code changes saved to uncommitted.diff)")
    return run_dir


def seed_dir(run_dir, seed):
    return os.path.join(run_dir, f"seed_{seed}")


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n  # population std -- just describing these n runs
    return mean, var ** 0.5


def append_to_index(run_dir, status, per_seed):
    """Add this run's row to results/runs.csv. per_seed: one dict per
    finished seed with the FINAL_METRIC_KEYS (last logged round)."""
    info = read_run_info(run_dir)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    row = {
        "run_id": info["run_id"],
        "started": info["started"],
        "status": status,
        "note": info["note"],
        "preset": info["preset"] or "",
        "seeds": " ".join(str(s) for s in info["seeds"]),
        "seeds_finished": len(per_seed),
        **{k.lower(): config.get(k) for k in INDEX_CONFIG_KEYS},
        "overrides": " ".join(f"{k}={v}" for k, v in info["overrides"].items()),
        "commit": info["commit"][:10],
        "uncommitted_changes": info["dirty"],
    }
    for key in FINAL_METRIC_KEYS:
        if per_seed:
            mean, std = mean_std([r[key] for r in per_seed])
            row[f"{key}_mean"], row[f"{key}_std"] = round(mean, 4), round(std, 4)
        else:
            row[f"{key}_mean"], row[f"{key}_std"] = "", ""
    fields = list(row)

    # If an older runs.csv has different columns (INDEX_CONFIG_KEYS changed
    # since), keep its header rather than corrupting it; the full config is
    # in config.json regardless.
    exists = os.path.exists(RUNS_INDEX) and os.path.getsize(RUNS_INDEX) > 0
    if exists:
        with open(RUNS_INDEX, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != fields:
            print(f"WARNING: {RUNS_INDEX} has different columns than this version writes -- "
                  "keeping its header; rename the file to start a fresh index")
            fields = existing
    os.makedirs(os.path.dirname(RUNS_INDEX), exist_ok=True)
    try:
        with open(RUNS_INDEX, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Logged run to {RUNS_INDEX} (status: {status})")
    except PermissionError:
        # Typically runs.csv is open in Excel, which locks it on Windows.
        # Don't lose the row: park it in the run folder instead.
        fallback = os.path.join(run_dir, "index_row.csv")
        with open(fallback, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(f"WARNING: could not write {RUNS_INDEX} (open in Excel?) -- row saved to {fallback}; "
              "paste it into runs.csv by hand")
