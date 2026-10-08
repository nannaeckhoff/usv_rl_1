# Diagnostic / validation script for environment.py -- not a pytest suite,
# just plain prints and asserts. Run directly:
#   python scripts/test_env.py
#
# Checks:
#   1. Unicycle dynamics: thrust + zero turn -> straight line;
#      thrust + constant turn -> circular path
#   2. Same seed -> identical reset()/step() trajectory
#   3. A heuristic controller (turn to face the goal, then thrust) beats a
#      random policy on Stage A, actually reaches goals, and costs exactly 0
#      (no hazards on Stage A)
#   4. Each safety mode gives the expected cost at known positions

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import config as cfg
from environment import PointToGoal, SAFETY_MODES, compute_cost


def _push_goal_far_away(env):
    """For pure-dynamics checks: make sure the agent can't accidentally
    reach the goal mid-check and trigger a full reset, which would corrupt
    the very state (heading, position) the check is trying to verify."""
    env.goal = torch.full_like(env.goal, 1.0e4)
    env.last_dist = (env.goal - env.pos).norm(dim=-1)


def check_unicycle_straight_line():
    print("[1a] unicycle: thrust, zero turn -> straight line ...", end=" ")
    env = PointToGoal(N=4, dev="cpu", horizon=200, num_hazards=0, seed=0)
    env.reset()
    _push_goal_far_away(env)
    env.th = torch.zeros(4)  # force a known heading (facing +x) for a clean check

    action = torch.tensor([[1.0, 0.0]]).repeat(4, 1)  # full thrust, zero turn
    for _ in range(100):
        env.step(action)

    assert torch.allclose(env.th, torch.zeros(4), atol=1e-6), "heading should not have changed"
    assert env.pos[:, 1].abs().max().item() < 1e-4, "y should stay ~0 with zero turn rate"
    assert env.pos[:, 0].min().item() > 0.0, "agent should have moved forward"
    print("OK")


def check_unicycle_circle():
    print("[1b] unicycle: thrust + constant turn -> circular path ...", end=" ")
    env = PointToGoal(N=4, dev="cpu", horizon=500, num_hazards=0, k=1.0, c=0.5, w_max=1.0,
                       v_max=float("inf"), thrust_mode="forward_only", seed=1)
    env.reset()
    _push_goal_far_away(env)
    env.th = torch.zeros(4)
    env.v = torch.zeros(4)

    a1, a2 = 1.0, 0.5  # constant thrust + constant turn rate
    action = torch.tensor([[a1, a2]]).repeat(4, 1)
    w = env.w_max * a2  # rad/s -- known exactly since the turn action is constant

    # A circular path satisfies center = pos + (v/w) * (-sin(th), cos(th))
    # at every point along it (derived from integrating the unicycle ODE
    # with constant v, w) -- so if this "candidate center" stays put across
    # steps, the path really is a circle.
    centers = []
    for i in range(300):
        env.step(action)
        if i > 150:  # discard the transient while v is still ramping up
            v, th = env.v, env.th
            centers.append(env.pos + (v / w).unsqueeze(-1) * torch.stack([-torch.sin(th), torch.cos(th)], dim=-1))
    centers = torch.stack(centers)  # (steps, N, 2)

    radius = (env.v[0] / w).abs().item()
    center_std = centers.std(dim=0).mean().item()
    print(f"(radius~{radius:.2f}, center std={center_std:.4f}) ", end="")
    assert center_std < 0.05 * radius, "candidate circle centers should stay ~constant along a real circle"
    print("OK")


def check_speed_clamp_and_braking():
    print("[1c] speed: clamp, reversing, coasting, drag equilibrium ...", end=" ")
    N = 4
    full_thrust = torch.tensor([[1.0, 0.0]]).repeat(N, 1)
    full_brake = torch.tensor([[-1.0, 0.0]]).repeat(N, 1)
    zero_thrust = torch.tensor([[0.0, 0.0]]).repeat(N, 1)

    # Drag-free model: c=0, so the symmetric [-v_max, v_max] clamp bounds v.
    env = PointToGoal(N=N, dev="cpu", horizon=200, num_hazards=0, c=0.0, v_max=1.0,
                      thrust_mode="bidirectional", seed=2)
    env.reset()
    _push_goal_far_away(env)

    env.v = torch.zeros(N)
    for _ in range(50):
        env.step(full_thrust)
    assert torch.allclose(env.v, torch.full((N,), env.v_max), atol=1e-6), \
        "full throttle from rest should drive v to v_max and hold it there"

    env.v = torch.full((N,), env.v_max)
    for _ in range(50):
        env.step(full_brake)
    assert torch.allclose(env.v, torch.full((N,), -env.v_max), atol=1e-6), \
        "full reverse from v_max should drive v to -v_max (bidirectional can reverse)"

    env.v = torch.full((N,), 0.4)
    coast_v = env.v.clone()
    for _ in range(20):
        env.step(zero_thrust)
    assert torch.allclose(env.v, coast_v, atol=1e-6), "zero thrust should neither speed up nor slow down (c=0)"

    # Default damped model (c > 0, v_max = inf): speed converges to the drag
    # equilibrium +k/c under full thrust and -k/c under full reverse.
    env_d = PointToGoal(N=N, dev="cpu", horizon=1000, num_hazards=0, seed=3)
    k, c = env_d.k, env_d.c
    env_d.reset()
    _push_goal_far_away(env_d)
    env_d.v = torch.zeros(N)
    for _ in range(300):
        env_d.step(full_thrust)
    assert torch.allclose(env_d.v, torch.full((N,), k / c), atol=1e-3), \
        "full throttle should converge to the +k/c drag equilibrium"
    for _ in range(300):
        env_d.step(full_brake)
    assert torch.allclose(env_d.v, torch.full((N,), -k / c), atol=1e-3), \
        "full reverse should converge to the -k/c drag equilibrium"

    # forward_only: never reverses, even at the lowest thrust.
    env_f = PointToGoal(N=N, dev="cpu", horizon=200, num_hazards=0, thrust_mode="forward_only", seed=4)
    env_f.reset()
    _push_goal_far_away(env_f)
    env_f.v = torch.zeros(N)
    for _ in range(50):
        env_f.step(full_brake)
        assert (env_f.v >= 0.0).all(), "forward_only: v must never go negative"
    print("OK")


def check_seeding_reproducible():
    print("[2] same seed -> identical trajectory ...", end=" ")

    def run():
        env = PointToGoal(N=16, dev="cpu", horizon=50, num_hazards=2, seed=123)
        obs = env.reset()
        torch.manual_seed(123)  # covers the random actions below
        for _ in range(20):
            action = torch.rand(16, 2) * 2 - 1
            obs, reward, cost, terminated, truncated, info = env.step(action)
        return obs

    obs1, obs2 = run(), run()
    assert torch.equal(obs1, obs2), "identical seeds should give identical trajectories"
    print("OK")


def unicycle_heuristic_action(obs):
    """Turn to face the goal, then thrust. Unicycle obs layout (body frame):
    [v, goal_x, goal_y, hazards...]."""
    goal_x, goal_y = obs[:, 1], obs[:, 2]
    angle_to_goal = torch.atan2(goal_y, goal_x)
    turn = torch.clamp(angle_to_goal / (math.pi / 2), -1.0, 1.0)
    thrust = torch.cos(angle_to_goal).clamp(min=0.0)  # ~1 facing the goal, 0 (coast, don't reverse) sideways/behind
    return torch.stack([thrust, turn], dim=-1)


def run_policy(env, action_fn, steps):
    obs = env.reset()
    total_reward = torch.zeros(env.N)
    total_cost = torch.zeros(env.N)
    ever_reached = torch.zeros(env.N, dtype=torch.bool)
    for _ in range(steps):
        action = action_fn(obs)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cost += cost
        ever_reached |= terminated
    return total_reward, total_cost, ever_reached


def _build_stage_a_env(N, seed):
    return PointToGoal(
        N=N, dev="cpu", horizon=cfg.HORIZON,
        k=cfg.K_THRUST, c=cfg.DRAG_COEF, w_max=cfg.W_MAX, v_max=cfg.V_MAX, thrust_mode=cfg.THRUST_MODE,
        world_half_extent=cfg.WORLD_HALF_EXTENT, goal_radius=cfg.GOAL_RADIUS, goal_bonus=cfg.GOAL_BONUS,
        num_hazards=cfg.NUM_HAZARDS, hazard_radius=cfg.HAZARD_RADIUS, seed=seed,
    )


def check_heuristic_beats_random_on_stage_a():
    print("[3] heuristic vs. random on Stage A ...")
    cfg.apply_preset("stage_a")
    N, steps = 300, cfg.HORIZON

    env_h = _build_stage_a_env(N, seed=10)
    reward_h, cost_h, reached_h = run_policy(env_h, unicycle_heuristic_action, steps)

    env_r = _build_stage_a_env(N, seed=10)  # same seed -> same scenarios, fair comparison
    torch.manual_seed(0)
    reward_r, cost_r, reached_r = run_policy(env_r, lambda obs: torch.rand(N, 2) * 2 - 1, steps)

    print(f"    heuristic: mean_reward={reward_h.mean():.2f}  success_rate={reached_h.float().mean():.2f}  mean_cost={cost_h.mean():.4f}")
    print(f"    random:    mean_reward={reward_r.mean():.2f}  success_rate={reached_r.float().mean():.2f}  mean_cost={cost_r.mean():.4f}")

    assert reward_h.mean() > reward_r.mean(), "heuristic should score higher than random"
    assert reached_h.float().mean() > 0.5, "heuristic should reach the goal in most envs"
    assert reached_r.float().mean() < reached_h.float().mean(), "random should reach the goal less often"
    assert cost_h.mean().item() == 0.0 and cost_r.mean().item() == 0.0, "Stage A has no hazards, cost must be exactly 0"
    print("    OK")


def check_safety_modes():
    print("[4] safety modes: cost at known positions ...", end=" ")
    r, margin = 0.5, 0.5
    hazards = torch.zeros(4, 1, 2)  # one hazard at the origin
    # Agent at: the center, halfway into the hazard, inside the margin
    # (0.25 outside the edge), and well clear of hazard + margin.
    pos = torch.tensor([[0.0, 0.0], [0.25, 0.0], [0.75, 0.0], [2.0, 0.0]])
    toward = torch.tensor([[-1.0, 0.0]]).repeat(4, 1)  # moving at speed 1 toward the hazard
    away = -toward
    heading = torch.zeros(4)

    def cost(mode, vel=toward):
        return compute_cost(pos, vel, heading, hazards, r, mode, margin)

    expected = {
        "binary":      [1.0, 1.0, 0.0, 0.0],
        "penetration": [1.0, 0.5, 0.0, 0.0],
        "proximity":   [1.0, 0.75, 0.25, 0.0],  # 1 - d / (r + margin)
        "velocity":    [0.0, 0.75, 0.25, 0.0],  # proximity * closing speed (0 at the center: no direction)
    }
    assert set(expected) == set(SAFETY_MODES), "every safety mode needs an expected value here"
    for mode, values in expected.items():
        got = cost(mode)
        assert torch.allclose(got, torch.tensor(values), atol=1e-5), f"{mode}: expected {values}, got {got.tolist()}"
    assert (cost("velocity", away) == 0).all(), "velocity: moving away from a hazard should cost nothing"

    no_hazards = torch.zeros(4, 0, 2)
    for mode in SAFETY_MODES:
        assert (compute_cost(pos, toward, heading, no_hazards, r, mode, margin) == 0).all(), \
            f"{mode}: no hazards must mean zero cost"
    print("OK")


if __name__ == "__main__":
    check_unicycle_straight_line()
    check_unicycle_circle()
    check_speed_clamp_and_braking()
    check_seeding_reproducible()
    check_heuristic_beats_random_on_stage_a()
    check_safety_modes()
    print("\nAll checks passed.")
