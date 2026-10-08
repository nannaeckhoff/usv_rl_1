# Commit and push new training runs (results/runs/ + results/runs.csv), so
# they can be pulled onto another machine. Run on the GPU machine after
# training:
#   python scripts/sync_results.py
# then on your own computer:
#   git pull
#
# Only results are committed -- any code changes you have lying around stay
# uncommitted (each run already saved them in its uncommitted.diff). Plots
# aren't synced (see .gitignore); regenerate them with
#   python scripts/plot.py --run results/runs/<run>

import os
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_PATHS = ["results/runs", "results/runs.csv"]


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r


def main():
    existing = [p for p in RESULTS_PATHS if os.path.exists(os.path.join(REPO_ROOT, p))]
    if not existing:
        sys.exit("No results/runs or results/runs.csv yet -- nothing to sync.")
    git("add", "--", *existing)

    staged = git("diff", "--cached", "--name-only", "--", *existing).stdout.splitlines()
    if not staged:
        print("No new results to sync.")
    else:
        # Name the commit after the runs it contains (the run folder names).
        runs = sorted({p.split("/")[2] for p in staged if p.startswith("results/runs/") and p.count("/") >= 3})
        message = "results: " + (", ".join(runs) if runs else "update runs.csv")
        # Commit only the results paths, even if other files happen to be staged.
        git("commit", "-m", message, "--", *existing)
        print(f"Committed {len(staged)} file(s): {message}")

    push = git("push", check=False)
    if push.returncode != 0:
        # Usually: code was pushed from the other machine in the meantime.
        # Results and code don't touch the same files, so a rebase is safe.
        print("Push rejected -- pulling first (rebase), then pushing again ...")
        git("pull", "--rebase", "--autostash")  # autostash: uncommitted code edits would block the rebase
        git("push")
    print("Pushed. On your own computer: git pull")


if __name__ == "__main__":
    main()
