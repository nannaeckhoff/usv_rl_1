# Pre-flight check for the shared GPU machine -- run this BEFORE a real
# training run. Plain prints and asserts, like test_env.py. Run:
#   python scripts/preflight_gpu.py
#
# Checks:
#   1. Machine info: CUDA visible, which GPU, free GPU memory, who else is
#      on the GPU, current CPU load (so you can see the other student's job)
#   2. CPU thread limits are actually in effect (torch uses 1 thread)
#   3. Basic GPU math matches the CPU result
#   4. The environment runs on the GPU at full N_ENVS: tensors stay on the
#      GPU, shapes are right, no NaN/inf
#   5. A tiny training run (2 PPO updates, real N_ENVS/ROLLOUT_STEPS) runs
#      end to end, writes metrics and a loadable checkpoint -- and measures
#      how many CPU cores this process actually used while doing it, plus
#      peak GPU memory and a time estimate for the full run
#
# Nothing is written under results/ or runs.csv -- the mini run saves into
# a temporary folder that is deleted afterwards.

import os

# Must be set before torch (and numpy/MKL) are imported to have any effect.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import shutil
import subprocess
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

torch.set_num_threads(1)

import config as cfg
from environment import PointToGoal

# How many CPU cores this process may use (on average) during training
# before we call it "disturbing the other job". The CUDA driver busy-waits
# one core while waiting on the GPU, so ~1 core is expected.
MAX_CPU_CORES = 1.5

failures = []
warnings_ = []


def warn(msg):
    warnings_.append(msg)
    print(f"    WARNING: {msg}")


def check(name, fn):
    print(f"\n{name}")
    try:
        fn()
        print("    OK")
    except Exception as e:  # keep going, report everything at the end
        failures.append(f"{name}: {e}")
        print(f"    FAILED: {e}")


def check_machine_info():
    assert torch.cuda.is_available(), "CUDA is not available to PyTorch (wrong torch build or no driver?)"
    print(f"    python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"    GPU 0: {torch.cuda.get_device_name(0)} (visible GPUs: {torch.cuda.device_count()})")
    free, total = torch.cuda.mem_get_info()
    print(f"    GPU memory free: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    if free < 0.5 * total:
        warn("less than half the GPU memory is free -- someone else may be using this GPU")

    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        print("    other processes on the GPU:", "none" if not out else "")
        for line in out.splitlines():
            print(f"      {line}")

    n_cores = os.cpu_count()
    print(f"    CPU cores: {n_cores}")
    if hasattr(os, "getloadavg"):
        load1, _, _ = os.getloadavg()
        print(f"    CPU load (1 min avg): {load1:.1f} of {n_cores} cores busy")
        if load1 > n_cores - 1:
            warn("the CPU is (almost) fully used already -- even 1 extra core will slow the other job a bit")
    try:
        import psutil
        print(f"    CPU usage right now: {psutil.cpu_percent(interval=1.0):.0f}%")
    except ImportError:
        pass


def check_thread_limits():
    print(f"    torch threads: {torch.get_num_threads()}, interop threads: {torch.get_num_interop_threads()}")
    print(f"    OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    assert torch.get_num_threads() == 1, "torch should be limited to 1 CPU thread"


def check_gpu_math():
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    gpu = (a.cuda() @ b.cuda()).cpu()
    cpu = a @ b
    err = (gpu - cpu).abs().max().item()
    print(f"    max |GPU - CPU| for a 512x512 matmul: {err:.2e}")
    # TF32 on newer GPUs gives ~1e-2 differences on matmuls of this size; that's fine.
    assert err < 0.1, "GPU result differs a lot from CPU result"


def _build_env(seed=cfg.SEED):
    return PointToGoal(
        N=cfg.N_ENVS, dev=cfg.DEVICE, dt=cfg.DT, horizon=cfg.HORIZON,
        k=cfg.K_THRUST, c=cfg.DRAG_COEF, w_max=cfg.W_MAX, v_max=cfg.V_MAX, thrust_mode=cfg.THRUST_MODE,
        reverse_scale=cfg.REVERSE_THRUST_SCALE,
        world_half_extent=cfg.WORLD_HALF_EXTENT, goal_radius=cfg.GOAL_RADIUS, goal_bonus=cfg.GOAL_BONUS,
        num_hazards=cfg.NUM_HAZARDS, hazard_radius=cfg.HAZARD_RADIUS,
        placement_resample_rounds=cfg.PLACEMENT_RESAMPLE_ROUNDS,
        randomize_start_pos=cfg.RANDOMIZE_START_POS,
        safety_mode=cfg.SAFETY_MODE, safety_margin=cfg.SAFETY_MARGIN, seed=seed,
    )


def check_env_on_gpu():
    assert cfg.DEVICE == "cuda", f"config.DEVICE is {cfg.DEVICE!r}, expected 'cuda'"
    env = _build_env()
    obs = env.reset()
    N = env.N
    gen = torch.Generator(device="cuda").manual_seed(0)
    steps = 200

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        action = torch.rand(N, env.act_dim, device="cuda", generator=gen) * 2 - 1
        obs, reward, cost, terminated, truncated, info = env.step(action)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    for name, t in [("obs", obs), ("reward", reward), ("cost", cost), ("terminated", terminated),
                    ("pos", env.pos), ("hazards", env.hazards)]:
        assert t.is_cuda, f"{name} is not on the GPU"
    assert obs.shape == (N, env.obs_dim), f"obs shape {tuple(obs.shape)}, expected {(N, env.obs_dim)}"
    assert reward.shape == cost.shape == (N,), "reward/cost should have shape (N,)"
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(cost).all(), \
        "NaN/inf in obs, reward or cost"
    print(f"    N_ENVS={N}, safety_mode={env.safety_mode}, num_hazards={env.H}")
    print(f"    env only, random actions: {steps * N / elapsed:,.0f} env-steps/s")


def check_mini_training():
    import train as train_mod  # scripts/ is on sys.path when run as a script

    tmp = tempfile.mkdtemp(prefix="usv_preflight_")
    saved = {k: getattr(cfg, k) for k in ("TOTAL_UPDATES", "SAVE_EVERY", "MODEL_SAVE_PATH")}
    try:
        cfg.TOTAL_UPDATES = 2
        cfg.SAVE_EVERY = 1
        cfg.MODEL_SAVE_PATH = os.path.join(tmp, "model.pt")

        torch.cuda.reset_peak_memory_stats()
        wall0, cpu0 = time.perf_counter(), time.process_time()
        metrics_path = train_mod.train()
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0

        # process_time() counts CPU time of every thread in this process, so
        # cpu / wall = average number of cores we kept busy.
        cores_used = cpu / wall
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"    wall time {wall:.1f}s, CPU time {cpu:.1f}s -> about {cores_used:.2f} CPU cores used on average")
        print(f"    peak GPU memory (PyTorch): {peak_gb:.2f} GB")

        row = train_mod._final_metrics_row(metrics_path)
        for key in ("reward_per_step", "cost_per_step", "policy_loss", "value_loss", "approx_kl"):
            assert row[key] not in ("nan", "inf", "-inf"), f"metric {key} is {row[key]}"
        sps = float(row["steps_per_sec"])
        full_hours = saved["TOTAL_UPDATES"] * cfg.ROLLOUT_STEPS * cfg.N_ENVS / sps / 3600
        print(f"    training speed (update 2): {sps:,.0f} env-steps/s")
        print(f"    estimated time for the full run ({saved['TOTAL_UPDATES']} updates, one seed): ~{full_hours:.2f} h")

        ckpt = torch.load(cfg.MODEL_SAVE_PATH, map_location="cuda")
        assert "model" in ckpt and "obs_rms_mean" in ckpt, "checkpoint is missing keys"
        for name, p in ckpt["model"].items():
            assert torch.isfinite(p).all(), f"NaN/inf in network weight {name}"

        assert cores_used <= MAX_CPU_CORES, \
            f"used {cores_used:.2f} CPU cores on average (limit {MAX_CPU_CORES}) -- would slow down the other job"
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    check("[1] machine info", check_machine_info)
    if failures:  # no CUDA -> nothing else can run
        print("\nCUDA not usable, stopping here.")
        sys.exit(1)
    check("[2] CPU thread limits", check_thread_limits)
    check("[3] GPU math matches CPU", check_gpu_math)
    check("[4] environment on GPU", check_env_on_gpu)
    check("[5] mini training run (2 updates)", check_mini_training)

    print("\n" + "=" * 60)
    for w in warnings_:
        print(f"WARNING: {w}")
    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        sys.exit(1)
    print("All pre-flight checks passed -- safe to start the real run.")
