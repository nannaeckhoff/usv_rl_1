# PPO tuning plan (before the safety study)

Goal: a PPO setup that reliably solves the navigation task, so that every
later difference between safety representations (Task 3) and weightings
(Task 4) is caused by the safety term, not by an untuned learner.

## Why tune on Stage A

Hyperparameters are tuned on **Stage A (no hazards)** and then confirmed on
Stage B:

- Stage A is the pure navigation task. Nothing else affects the result, so a
  better number really means a better learner.
- With λ = 0, Stage B is the same task. The hazards are only extra inputs that
  don't affect the reward, so Stage B adds nothing to tune, only noise.
- The safety term changes the objective. Tuning with it on would mix the
  safety question and the tuning question together.

Stage B is then used twice: with λ = 0 to show the tuning carries over and to
measure how unsafe the agent is without a cost, and with a simple cost to show
that the tuned PPO still works once a cost is added.

## Workflow for every run

```
# GPU machine
python scripts/train.py --preset <stage> --seeds <seeds> --set <KEY=VALUE ...> --name <label> --note "<what changed>"
python scripts/evaluate.py --run results/runs/<run>
python scripts/sync_results.py

# own computer
git pull
python scripts/plot_eval.py --run results/runs/<run> [results/runs/<other run> ...]
```

All changes go through `--set`, and `config.py` stays untouched until Phase 4.
Several runs can be passed to `plot_eval.py` at once, and it ends with a table
comparing them.

To run a sweep and evaluate each run straight after it trains (bash on the
GPU machine):

```
for lr in 1e-4 1e-3 3e-3; do
  python scripts/train.py --preset stage_a --seeds 1 2 3 4 5 --set TOTAL_UPDATES=<N> LR=$lr --name lr$lr --note "tuning: LR=$lr"
  python scripts/evaluate.py --run "$(ls -td results/runs/*/ | head -1)"
done
python scripts/sync_results.py
```

## What "better" means

Read in this order from the `plot_eval.py` table:

1. `success%`: must stay ≥ 99% for every seed. A setting that drops it is
   rejected, whatever else it improves.
2. Seed std of `ep_len`: how much the result depends on the seed. Target
   below 2% of the mean, which is what the "seeds agree" check tests. This
   is the main tuning goal (see Phase 1). Later safety effects are only a
   few percent, so a seed spread of the same size would hide them.
3. `ep_len` and `len/heur`: episode length on the same 10,000 scenarios. Lower
   means a better policy, i.e. faster to the goal.
4. `conv_upd`: the update at which reward reaches 95% of its final value.
   Lower means it learns faster.
5. PPO health in `seeds_training.png`: approx KL < 0.02, entropy falls
   gradually rather than dropping in a few updates, and the value loss goes
   down.

**Decision rule:** accept a change only if its mean is better than the
reference by more than the larger of the two seed stds. Otherwise keep the
reference, which is the standard cleanRL value and easier to justify.

## Phase 1: Reference (no training needed)

The existing run `2026-10-08_142709_stage_a_penetration` (Stage A, current
config, 5 seeds, 500 updates) is the reference **A0**. It was made with code
whose Stage A behaviour is unchanged since then.

```
python scripts/evaluate.py --run results/runs/2026-10-08_142709_stage_a_penetration
```

Result (10,000 episodes): success 99.99%, `ep_len` 38.4 ± 0.9, `len/heur`
0.79, `conv_upd` 76 ± 12, and PPO health is fine.

- The only failures are 6 episodes from seed 1. In all of them the goal starts
  just outside the goal radius (0.33–0.48 m away), and the boat circles until
  it times out. This is a rare edge case, not a tuning problem.
- **Seed spread is too large:** `ep_len` runs from 37.3 (seed 3) to 39.9
  (seed 2), about 7% apart, with a std of 2.4% of the mean. The "seeds
  agree" check fails.
- The training curves show why. Each seed settles at its own reward level
  within about 1e8 steps and never catches up. Entropy falls to about −2.9
  and the LR anneals to 0, so every seed freezes in its own way of solving
  the task.
- That points to exploration and step size, which is what Phase 3 targets.

## Phase 2: Training budget (`TOTAL_UPDATES`)

A0 converges by about update 76 of 500, so most of a run teaches nothing. A
shorter budget makes every later sweep about 3× cheaper. The LR is annealed
over `TOTAL_UPDATES`, so the budget changes the LR schedule too. That is why
it is fixed first and then held constant.

**Run** (GPU machine, about 30 min in total):

```
python scripts/train.py --preset stage_a --seeds 1 2 3 4 5 --set LAMBDA_COST=0 TOTAL_UPDATES=150 --name upd150 --note "Phase 2: budget 150"
python scripts/evaluate.py --run "$(ls -td results/runs/*/ | head -1)"
python scripts/train.py --preset stage_a --seeds 1 2 3 4 5 --set LAMBDA_COST=0 TOTAL_UPDATES=250 --name upd250 --note "Phase 2: budget 250"
python scripts/evaluate.py --run "$(ls -td results/runs/*/ | head -1)"
python scripts/sync_results.py
```

**Compare** (own computer), with all three budgets in one table:

```
git pull
python scripts/plot_eval.py --run results/runs/2026-10-08_142709_stage_a_penetration results/runs/<..._upd150> results/runs/<..._upd250>
```

**Pick** the smallest budget where all of these hold:

- `success%` ≥ 99;
- `ep_len` is within A0's seed std (38.4 ± 0.9);
- the seed std of `ep_len` is no larger than A0's.

The last point matters because a shorter budget also anneals the LR to 0
faster, so exploration stops sooner, which can increase the seed spread.

If neither 150 nor 250 passes, keep 500. The chosen value is called **N**
below, and every run after this uses `TOTAL_UPDATES=N`.

## Phase 3: One parameter at a time (Stage A, seeds 1–5, `TOTAL_UPDATES=N`)

The reference for this phase is the Phase 2 winner. Change one parameter per
run and keep everything else at the reference.

The main goal is to cut the seed spread found in Phase 1, without making
`ep_len` or `conv_upd` worse. Use 5 seeds here, not 3: an std computed from 3
seeds is too noisy to tell whether the spread really went down. The steps
most likely to help are 3b and 3e (exploration) and 3a (step size).

| Step | Parameter | Values to try (ref. **bold**) | What to look for |
|---|---|---|---|
| 3a | `LR` | 1e-4, **3e-4**, 1e-3, 3e-3 | KL is only ~0.003, so larger steps are probably safe. Look for lower `conv_upd` and lower seed std, with no loss in `ep_len`. Reject if KL > 0.02. |
| 3b | `ENTROPY_COEF` | **0**, 0.001, 0.005 | Entropy falls to about −3, so the policy becomes nearly deterministic and each seed stays where it first lands. A small bonus keeps exploration alive for longer. Look for lower seed std. This also matters once a cost appears in Phase 6. |
| 3c | `NUM_MINIBATCHES` | **8**, 32 | More gradient steps per batch, since a minibatch is currently ~100k samples. Look for lower `conv_upd`, and watch KL and clip fraction. |
| 3d | `EPOCHS` | 5, **10** | Only if 3c changed something: fewer passes means faster and more stable updates. |
| 3e | `LOG_STD_INIT` | −1.0, **−0.5**, 0.0 | Initial exploration noise. More noise early on can stop seeds from locking in too soon. Look for lower seed std and lower `conv_upd`. Success should not dip early on. |

Not tuned, because the diagnostics show nothing wrong and the cleanRL defaults
are a defensible choice: `CLIP_EPS` 0.2, `GAE_LAMBDA` 0.95, `VF_COEF` 0.5,
`MAX_GRAD_NORM` 0.5, `OBS_CLIP` 10, `ROLLOUT_STEPS` 200, `N_ENVS` 4096,
`HIDDEN_SIZE` 64. `GAMMA` 0.99 also stays fixed: it is part of the problem
definition (how much speed is rewarded) rather than a learning setting, and
changing it changes what "optimal" means.

Each step reuses the winners of the steps before it. About 11 runs × 5 seeds
× 3–5 min ≈ 3–4.5 hours of GPU time in total.

## Phase 4: Confirm the tuned setup (Stage A, seeds 1–5)

```
python scripts/train.py --preset stage_a --seeds 6 7 8 9 10 --set TOTAL_UPDATES=N <winners> --name tuned --note "tuned PPO, Stage A, fresh seeds"
```

These are new seeds. The winners were picked on seeds 1–5, so a low spread
there could partly be luck. It only counts if it also holds on seeds the
tuning never saw.

Compare with A0 using `plot_eval.py --run <A0> <tuned>`. All checks must be
OK. Then write the winners into `config.py` as the new defaults and commit
them ("Tuned PPO defaults"), so every later run uses them without `--set`.

If the seed std is still above 2%, accept it, but use 10 seeds instead of 5
in Phases 5 and 6, and in every later safety comparison. More seeds give a
more precise mean (its uncertainty shrinks with √n), so small safety effects
can still be told apart from seed noise.

## Phase 5: Stage B without cost (λ = 0, seeds 1–5)

```
python scripts/train.py --preset stage_b --seeds 1 2 3 4 5 --set LAMBDA_COST=0 --name nocost --note "tuned PPO, Stage B, no cost"
```

Check two things:

1. **The tuning carries over:** `success%`, `ep_len` and `conv_upd` match
   Phase 4 within the seed std. If they don't, the hazard inputs are hurting
   learning, and that has to be fixed before going on.
2. **The unsafe reference:** `viol_ep%` and `viol_blk%`, and the hazard
   distance histogram in `eval.png`. Expect many violations, because the
   agent drives straight through the hazard when it costs nothing. This is
   the baseline every safety result in Task 3 is compared against.

## Phase 6: Stage B with a simple cost (λ = 1, seeds 1–5)

```
python scripts/train.py --preset stage_b --seeds 1 2 3 4 5 --set SAFETY_MODE=binary LAMBDA_COST=1 --name binary --note "tuned PPO, Stage B, binary cost"
```

`binary` is the simplest representation: 1 inside a hazard, 0 outside. This
run validates the tuning, not the safety design. It passes if:

- success is still ≥ 99%, and every check is OK, so training is still stable
  with a cost;
- `viol_ep%` is far below Phase 5;
- `ep_len` is only a little higher than in Phase 5. The difference is the
  price of safety. For comparison, the old λ = 1 penetration run was about 6%
  longer than Stage A.

If seeds now disagree or success drops, the tuning does not carry over to the
cost case. Rerun step 3b (entropy) and 3a (LR) with Phase 6's settings before
moving on to Task 3.

## Tuning log

Fill in after each `plot_eval.py`, using mean ± std over seeds.

| Phase | Run folder | Change | success% | ep_len | len/heur | conv_upd | viol_ep% | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-08_142709_stage_a_penetration | reference A0 | | | | | – | reference |
| 2 | | TOTAL_UPDATES=150 | | | | | – | |
| 2 | | TOTAL_UPDATES=250 | | | | | – | |
| 3a | | LR=1e-4 | | | | | – | |
| 3a | | LR=1e-3 | | | | | – | |
| 3a | | LR=3e-3 | | | | | – | |
| 3b | | ENTROPY_COEF=0.001 | | | | | – | |
| 3b | | ENTROPY_COEF=0.005 | | | | | – | |
| 3c | | NUM_MINIBATCHES=32 | | | | | – | |
| 3d | | EPOCHS=5 | | | | | – | |
| 3e | | LOG_STD_INIT=-1.0 | | | | | – | |
| 3e | | LOG_STD_INIT=0.0 | | | | | – | |
| 4 | | tuned, seeds 6–10 | | | | | – | |
| 5 | | Stage B, λ=0 | | | | | | |
| 6 | | Stage B, binary, λ=1 | | | | | | |
