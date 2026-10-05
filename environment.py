# GPU-batched PointToGoal environmen, replacing Safety Gym/MuJoCo 
#
# All state is stored in tensors with a leading batch dimension N, so there
# is no Python loop over environments -- one call to step() advances every
# environment at once, on the GPU if available.
#
# Motion model: unicycle (see unicycle_update below) -- nonholonomic,
# vehicle-like, closer to a real USV: it can only move along its heading
# and must turn before moving sideways.
#
# Hazards are circular zones the agent should avoid. The cost is currently
# one fixed formula (how far the agent has penetrated a hazard) -- reward
# and cost are always returned separately and never combined here, so a
# training loop is free to weight them however it wants.
#
# Interface mirrors Safety Gym but with tensors:
#   reset()   -> obs
#   step(a)   -> (obs, reward, cost, terminated, truncated, info)
# terminated is true when the goal is reached (the episode ends there --
# there's no more mid-episode goal respawn); truncated is true at the fixed
# horizon. Either way the env is auto-reset via masks, so shapes stay fixed.

import math
import warnings

import torch


def unicycle_update(pos, v, th, action, dt, k, c, w_max, v_max, thrust_mode):
    """Nonholonomic unicycle: thrust a1 and turn rate a2, both in [-1, 1].
    pos: (N, 2), v: (N,) signed speed along the heading, th: (N,) heading in
    radians.

        v  <- v + dt * (k * a1 - c * v)
        th <- th + dt * w_max * a2
        pos <- pos + dt * v * [cos th, sin th]

    thrust_mode selects how a1 is interpreted:
      "bidirectional" (default): a1 used as-is in [-1, 1], matching the
        MuJoCo point robot's [-1, 1] action range. Negative a1 brakes and,
        once v reaches 0, reverses -- v can go negative (driving backwards).
        v is clamped to [-v_max, v_max].
      "forward_only": a1 mapped to [0, 1] -- can only speed up or coast,
        never reverse. v is clamped to [0, v_max].

    With drag c > 0 (the default) speed is already bounded by the drag
    equilibrium |v| <= k/c, so v_max = inf (no clamp) is the normal setting;
    the clamp only really matters when c = 0, where nothing else bounds v.
    The explicit Euler step needs c*dt well below 1 to stay stable (checked
    in PointToGoal.__init__).
    """
    a1 = action[:, 0].clamp(-1.0, 1.0)
    a2 = action[:, 1].clamp(-1.0, 1.0)
    if thrust_mode == "forward_only":
        a1 = (a1 + 1.0) * 0.5

    v = v + dt * (k * a1 - c * v)
    v_min = 0.0 if thrust_mode == "forward_only" else -v_max
    v = v.clamp(min=v_min, max=v_max)
    th = th + dt * w_max * a2
    th = (th + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi)

    heading = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
    pos = pos + dt * v.unsqueeze(-1) * heading  # the whole "physics"
    return pos, v, th


def min_hazard_distance(pos, hazards):
    """Distance from the agent to the nearest hazard center. +inf when there
    are no hazards (Stage A) -- `.min()` over an empty hazard dimension has
    no defined value, unlike `.any()`, so that case is handled explicitly
    here, once, instead of in every function that needs this distance."""
    H = hazards.shape[1]
    if H == 0:
        return torch.full((pos.shape[0],), float("inf"), device=pos.device)
    dist = (hazards - pos.unsqueeze(1)).norm(dim=-1)  # (N, H)
    return dist.min(dim=1).values


def compute_cost(pos, vel, heading, hazards, hazard_radius):
    """The safety representation. Currently one fixed formula (penetration
    depth), but takes the full (pos, vel, heading, hazards) signature so
    velocity/direction-based cost terms can be added later without having
    to change this signature everywhere it's called.

    pos: (N, 2). vel: (N, 2), world-frame velocity (v*[cos th, sin th]).
    heading: (N,) radians (unused by this formula). hazards: (N, H, 2).
    """
    min_dist = min_hazard_distance(pos, hazards)
    depth = (hazard_radius - min_dist).clamp(min=0.0)  # +inf hazard distance -> depth 0
    return depth / hazard_radius


class PointToGoal:
    def __init__(self, N, dev="cuda", dt=0.1, horizon=1000,
                 k=1.0, c=0.5, w_max=1.0, v_max=None, thrust_mode="bidirectional",
                 world_half_extent=5.0, goal_radius=0.3, goal_bonus=1.0,
                 num_hazards=3, hazard_radius=0.5, placement_resample_rounds=10,
                 randomize_start_pos=False, seed=0):
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("dev='cuda' requested but CUDA is not available -- "
                               "refusing to silently fall back to CPU")
        if thrust_mode not in ("bidirectional", "forward_only"):
            raise ValueError(f"thrust_mode must be 'bidirectional' or 'forward_only', got {thrust_mode!r}")
        # Euler stability of the drag term: v <- (1 - c*dt) v + ... must keep
        # 1 - c*dt clearly positive, otherwise drag overshoots past zero and
        # flips the sign of v every step instead of decaying it.
        if c < 0:
            raise ValueError(f"drag c must be >= 0, got {c}")
        if c * dt >= 1.0:
            raise ValueError(f"c*dt = {c * dt:.3g} >= 1: Euler drag update is unstable, reduce c or dt")
        if c * dt > 0.1:
            warnings.warn(f"c*dt = {c * dt:.3g} is not well below 1; the Euler drag update may be inaccurate",
                          stacklevel=2)
        if c == 0 and (v_max is None or math.isinf(v_max)):
            warnings.warn("c = 0 and v_max = inf: nothing bounds the speed", stacklevel=2)
        self.N, self.dev, self.dt = N, dev, dt
        self.horizon = horizon
        self.k, self.c, self.w_max = k, c, w_max          # thrust gain, drag, max turn rate
        self.v_max = float("inf") if v_max is None else v_max
        self.thrust_mode = thrust_mode
        self.world_half_extent = world_half_extent
        self.goal_radius = goal_radius
        self.goal_bonus = goal_bonus
        self.H = num_hazards
        self.hazard_radius = hazard_radius
        self.placement_resample_rounds = placement_resample_rounds
        self.randomize_start_pos = randomize_start_pos

        # All randomness goes through this generator, so runs are reproducible.
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)

        # [v] + [goal_x, goal_y] (body frame) + [hazard_x, hazard_y] per hazard (body frame)
        self.obs_dim = 1 + 2 + 2 * self.H
        self.act_dim = 2  # [thrust, turn rate]

        # State tensors are allocated in reset().
        self.pos = None      # (N, 2)
        self.v = None        # (N,)      signed speed along heading (< 0 = reversing)
        self.th = None       # (N,)      heading, radians
        self.goal = None     # (N, 2)
        self.hazards = None  # (N, H, 2)
        self.last_dist = None
        self.t = None

    def _rand_uniform(self, shape, low, high):
        return torch.rand(shape, generator=self.gen, device=self.dev) * (high - low) + low

    def _sample_positions(self, shape):
        e = self.world_half_extent
        return self._rand_uniform(shape, -e, e)

    def _place_hazards(self):
        """Sample hazard centers, resampling away from the start position
        (the origin) for a fixed number of rounds -- not a guarantee, but
        enough in practice with a reasonable world size / hazard count."""
        shape = (self.N, self.H, 2)
        keepout = self.hazard_radius + self.goal_radius
        haz = self._sample_positions(shape)
        for _ in range(self.placement_resample_rounds):
            bad = haz.norm(dim=-1) < keepout  # (N, H), distance from origin
            haz = torch.where(bad.unsqueeze(-1), self._sample_positions(shape), haz)
        return haz

    def _place_clear_point(self, hazards):
        """Sample a single point per env, resampling away from all hazards.
        Used for the goal, and for the start position when
        randomize_start_pos is on -- both are "a point clear of hazards"."""
        shape = (self.N, 2)
        keepout = self.hazard_radius + self.goal_radius
        point = self._sample_positions(shape)
        for _ in range(self.placement_resample_rounds):
            dist_to_hazards = (hazards - point.unsqueeze(1)).norm(dim=-1)  # (N, H)
            bad = (dist_to_hazards < keepout).any(dim=1)
            point = torch.where(bad.unsqueeze(-1), self._sample_positions(shape), point)
        return point

    def reset(self):
        N, dev = self.N, self.dev
        self.hazards = self._place_hazards()
        self.pos = self._place_clear_point(self.hazards) if self.randomize_start_pos else torch.zeros(N, 2, device=dev)
        self.v = torch.zeros(N, device=dev)
        self.th = self._rand_uniform((N,), -math.pi, math.pi)  # random initial heading
        self.goal = self._place_clear_point(self.hazards)
        self.last_dist = (self.goal - self.pos).norm(dim=-1)
        self.t = torch.zeros(N, dtype=torch.long, device=dev)
        return self._obs()

    def _to_body_frame(self, world_vec, cos_th, sin_th):
        """Rotate a world-frame vector (..., 2) into the agent's body frame
        (by -th), so "ahead" is always the body-frame x-axis. cos_th/sin_th
        must already be broadcastable against world_vec's leading dims."""
        x, y = world_vec[..., 0], world_vec[..., 1]
        bx = cos_th * x + sin_th * y
        by = -sin_th * x + cos_th * y
        return torch.stack([bx, by], dim=-1)

    def _obs(self):
        goal_rel = self.goal - self.pos                    # (N, 2)
        hazards_rel = self.hazards - self.pos.unsqueeze(1)  # (N, H, 2)

        # Body-frame observation: the agent's absolute orientation in the
        # world is meaningless to the policy once everything else is
        # relative, so we rotate goal/hazards by -th and only keep speed
        # (not cos/sin of the absolute heading).
        cos_th, sin_th = torch.cos(self.th), torch.sin(self.th)
        goal_obs = self._to_body_frame(goal_rel, cos_th, sin_th)
        hazards_obs = self._to_body_frame(hazards_rel, cos_th.unsqueeze(-1), sin_th.unsqueeze(-1))
        return torch.cat([self.v.unsqueeze(-1), goal_obs, hazards_obs.reshape(self.N, -1)], dim=-1)

    @torch.no_grad()
    def step(self, action):
        self.pos, self.v, self.th = unicycle_update(
            self.pos, self.v, self.th, action, self.dt, self.k, self.c, self.w_max,
            self.v_max, self.thrust_mode,
        )
        heading = self.th
        vel_world = self.v.unsqueeze(-1) * torch.stack([torch.cos(self.th), torch.sin(self.th)], dim=-1)

        # Keep the agent inside the world -- goals/hazards are only ever
        # placed in [-extent, extent], so nothing useful is out there anyway.
        self.pos = torch.clamp(self.pos, -self.world_half_extent, self.world_half_extent)

        # Reward: progress toward the goal, plus a bonus for reaching it.
        # Reaching the goal now ends the episode (terminated) -- no more
        # mid-episode respawn.
        dist = (self.goal - self.pos).norm(dim=-1)
        terminated = dist < self.goal_radius
        reward = (self.last_dist - dist) + terminated.float() * self.goal_bonus

        cost = compute_cost(self.pos, vel_world, heading, self.hazards, self.hazard_radius)
        min_dist_to_hazard = min_hazard_distance(self.pos, self.hazards)
        ref_in_hazard = min_dist_to_hazard < self.hazard_radius

        self.t = self.t + 1
        truncated = self.t >= self.horizon
        episode_length = self.t  # length of the episode that just ended, for envs where terminated | truncated
        done = terminated | truncated

        info = {
            "goal_reached": terminated,
            "in_hazard": ref_in_hazard,
            "min_hazard_dist": min_dist_to_hazard,
            "episode_length": episode_length,
            # The observation the agent would have seen next had the episode
            # *not* been cut off here (before hazards/pos/goal get reset
            # below) -- used to bootstrap correctly through a truncation
            # instead of treating it like a real terminal state. Same idea
            # as Gymnasium's `info["final_observation"]`.
            "final_obs": self._obs(),
        }

        # Reset finished envs (terminated or truncated) with masks, so
        # tensor shapes stay fixed regardless of how many are done.
        done2 = done.unsqueeze(-1)
        new_hazards = self._place_hazards()
        self.hazards = torch.where(done.view(-1, 1, 1), new_hazards, self.hazards)
        new_start_pos = self._place_clear_point(self.hazards) if self.randomize_start_pos else torch.zeros_like(self.pos)
        self.pos = torch.where(done2, new_start_pos, self.pos)
        self.v = torch.where(done, torch.zeros_like(self.v), self.v)
        self.th = torch.where(done, self._rand_uniform((self.N,), -math.pi, math.pi), self.th)
        self.goal = torch.where(done2, self._place_clear_point(self.hazards), self.goal)
        self.last_dist = (self.goal - self.pos).norm(dim=-1)
        self.t = torch.where(done, torch.zeros_like(self.t), self.t)

        return self._obs(), reward, cost, terminated, truncated, info
