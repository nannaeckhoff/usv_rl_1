# Running the Point Environment on the GPU

## 1. What MuJoCo does

MuJoCo is a physics engine. Given the current state of a robot and the actions applied to it, it computes the next state, taking inertia, damping and contacts into account. It can also produce sensor readings. In Safety Gym, MuJoCo is used for exactly this: it moves the Point robot when actions are applied, handles contacts with objects such as vases and pillars, and provides the inertial sensor readings.

Everything else in the environment is ordinary code. Goals and hazards are zones that are checked by distance, and the reward, the cost, the lidar observations and the placement of objects are computed from positions. MuJoCo runs on the CPU, so each environment is simulated separately.

## 2. Why we do not need it

The Point robot is a single body that moves on a plane with two inputs, forward motion and turning. Without contacts, its motion is simple enough to write down in a few lines. In our study there are no contacts, since goals and hazards are only zones and we do not use vases or pillars. What matters for testing safety rewards is the type of motion (for example, that the robot cannot move sideways), not the exact friction or inertia values. In addition, the cost function of the built-in environment is fixed in the library, while we want to exchange the safety term freely.

Contact-rich behavior is not needed at present. If we need a richer environment later, we will use an existing simulation environment.

## 3. What we build instead

The same task in a simplified form: a point robot, a goal, hazards, a reward for approaching the goal, and a cost (safety term) that can be exchanged. The motion is a simple update rule with drag. Two variants are of interest:

**Unicycle** (inputs: thrust a1, turn rate a2)

```
v   <- v + dt * ( k * a1 - c * v )
th  <- th + dt * w_max * a2
pos <- pos + dt * v * [cos th, sin th]
```

**Double integrator** (input: force a)

```
vel <- vel + dt * ( a - c * vel )
pos <- pos + dt * vel
```

The constants (k, c, w_max) are adjustable parameters. Since the motion model is our own code, it is possible to experiment with different kinematics. We do not need to stick to the one that MuJoCo offers for the Point robot. Exchanging the model only means exchanging this update function, which makes it possible to test how a safety reward behaves for different kinds of motion.

## 4. Running it on the GPU: why and how

### 4.1 Why

Reinforcement learning needs a very large number of environment steps. With MuJoCo, the number of parallel environments is limited by the number of CPU cores (typically some tens), and data have to be copied between CPU and GPU at every step. If the environment consists of simple tensor operations, thousands of environments can be simulated in a single operation on the GPU, and the data never leave the GPU. Experiments then run much faster, so that many safety rewards and seeds can be tested.

### 4.2 How

- **Add a batch dimension.** Store the state of all environments in tensors whose first dimension is the number of environments N. The update is then written once and applies to all environments (see Section 5).
- **No loops over environments.** Write everything as tensor operations. A Python loop over environments would remove the benefit.
- **Reset with masks.** Reset finished environments with `torch.where` instead of indexing, so that the tensor shapes stay fixed.
- **Avoid synchronization with the CPU.** Inside the rollout, do not use `.item()`, do not print tensors, and avoid boolean-mask indexing, since these force the GPU to wait for the CPU.
- **Keep everything on the GPU.** The environment, the policy network and the learning update should all work on GPU tensors.
- **Keep the Safety Gym interface.** Provide `reset()` and `step(a)`, but with tensors as data type, so that existing learning implementations can be used without modification.

Once the environment works, use Claude or a similar tool to find out how to optimize the code further to make it fast.

## 5. Example: point mass to a goal

The example shows the same minimal task, a point that has to reach a goal, once with the built-in environment and once with a custom environment. The learning loop is identical in both cases. Only the environment changes.

### 5.1 Learning loop

A short policy-gradient loop, used only to make the example complete.

```python
import torch, torch.nn as nn

# Any environment with: reset() -> obs, step(a) -> (obs, reward), obs_dim, dev
def train(env, iters=200, T=200):
    pi = nn.Sequential(nn.Linear(env.obs_dim, 64), nn.Tanh(),
                       nn.Linear(64, 2)).to(env.dev)
    opt = torch.optim.Adam(pi.parameters(), lr=3e-3)
    for it in range(iters):
        obs, logps, R = env.reset(), [], []
        for t in range(T):
            d = torch.distributions.Normal(pi(obs), 0.3); a = d.sample()
            obs, r = env.step(a)
            logps.append(d.log_prob(a).sum(-1)); R.append(r)
        R = torch.stack(R)                         # (T, N)
        G = R.flip(0).cumsum(0).flip(0)            # reward-to-go
        adv = G - G.mean(1, keepdim=True)
        loss = -(torch.stack(logps) * adv / (adv.std() + 1e-8)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pi
```

### 5.2 With the built-in environment

Safety Gym runs on the CPU with numpy arrays, so a small adapter converts between numpy arrays and tensors at every step. `SafetyPointGoal0-v0` is the variant without hazards.

```python
import safety_gymnasium

class SafetyGymAdapter:  # built-in env: MuJoCo on the CPU
    def __init__(self, N=16):
        self.env = safety_gymnasium.vector.make("SafetyPointGoal0-v0", num_envs=N)
        self.dev = "cpu"
        self.obs_dim = self.reset().shape[-1]

    def reset(self):
        obs, _ = self.env.reset()
        return torch.as_tensor(obs, dtype=torch.float32)

    def step(self, a):  # tensor -> numpy -> MuJoCo
        obs, r, c, term, trunc, info = self.env.step(a.clamp(-1, 1).cpu().numpy())
        t = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return t(obs), t(r)

pi = train(SafetyGymAdapter(N=16), T=1000)
```

### 5.3 With a custom environment on the GPU

The MuJoCo simulation is replaced by the two update lines in `step`. The goal, the reward and the observation are computed with tensors, and all N environments are handled at once. Training is started with `pi = train(PointToGoal(4096))`.

```python
class PointToGoal:  # custom env, all on the GPU
    def __init__(self, N, dev="cuda", dt=0.1):
        self.N, self.dev, self.dt = N, dev, dt
        self.obs_dim = 4

    def _random_goal(self):
        return torch.rand(self.N, 2, device=self.dev) * 3 - 1.5

    def reset(self):
        self.pos = torch.zeros(self.N, 2, device=self.dev)
        self.vel = torch.zeros(self.N, 2, device=self.dev)
        self.goal = self._random_goal()
        self.last = (self.goal - self.pos).norm(dim=-1)
        return self._obs()

    def _obs(self):
        return torch.cat([self.goal - self.pos, self.vel], -1)  # (N, 4)

    @torch.no_grad()
    def step(self, a):  # a: (N, 2), already on the GPU
        self.vel = self.vel + self.dt * (a.clamp(-1, 1) - 0.5 * self.vel)  # force - drag
        self.pos = self.pos + self.dt * self.vel                            # the whole "physics"
        dist = (self.goal - self.pos).norm(dim=-1)
        reached = dist < 0.3
        reward = (self.last - dist) + reached.float()  # progress + bonus at the goal
        self.goal = torch.where(reached[:, None], self._random_goal(), self.goal)
        self.last = (self.goal - self.pos).norm(dim=-1)
        return self._obs(), reward
```

### 5.4 What changed

- **Physics.** MuJoCo is replaced by two lines: a velocity update with drag and a position update. For a unicycle, these two lines are replaced by the unicycle update from Section 3.
- **Task logic.** Goal, reward and observation are tensor operations instead of library code.
- **Parallelism.** 16 CPU processes become 4096 environments in one set of tensors on the GPU.
- **Learning loop.** Unchanged.

A safety term is added later as one more function, for example a cost computed from the position.

## 6. Where to start

1. Look at the Safety Gym Point Goal environment and identify which parts depend on MuJoCo and which are ordinary code.
2. Write the motion model (unicycle or double integrator) as a function on tensors. Section 5 shows a minimal version. Different kinematics can be tried by exchanging this function.
3. Reimplement the goal, the hazards, the reward and the cost with tensors. Keep the cost as a separate function, so that different safety rewards can be swapped in.
4. Make the environment provide `reset()` and `step(a)` with tensors, and test it with an existing learning implementation.
5. Optimize for speed (ask Claude/similar for help, it's really good for this).

To compare different safety rewards fairly, we need to keep track of the settings of each experiment (we will talk more about this once everything is set up).
