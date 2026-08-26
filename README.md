# RL Humanoid Locomotion — Training the Unitree G1 to Walk

A roadmap for training a walking policy for the Unitree G1 in MuJoCo, using the
hand-rolled RL implementations in [`RL/`](RL/).

## Contents

- [`RL/`](RL/) — from-scratch policy gradient / PPO implementations
  - `utils.py` — policy & value networks, trajectory collection, evaluation
  - `simple_policy_gradient.py` — vanilla REINFORCE
  - `simple_policy_gradient_using_advantage.py` — REINFORCE with a learned baseline
  - `simple_ppo.py` — clipped-objective PPO
- [`scripts/mujoco_spawn.py`](scripts/mujoco_spawn.py) — loads the G1 and runs the passive viewer
- `mujoco_menagerie/` — submodule providing the G1 model

---

## Where things stand

The existing PPO is structurally correct but built for a **discrete-action,
single-env, small-scale** problem. G1 is the opposite of all three:

| | Current code | G1 needs |
|---|---|---|
| Action space | `Categorical` over `env.action_space.n` | 29 continuous joint targets |
| Environment | `gym.make(name)` | doesn't exist — has to be written |
| Sample budget | 5k steps x 200 epochs = 1M | 50–200M steps |
| Throughput | 1 env, ~2–4k steps/s | 20k+ steps/s |

Four things to build, and **the order matters**. Do not jump straight to G1 —
that means three untested subsystems failing at once with no way to tell which.

Each stage below ends with a validation gate. Skipping a gate is the main way
this goes wrong.

---

## Stage 0 — Fix the bug in what already exists

**Time: ~30 min**

In `RL/simple_ppo.py:24`:

```python
advantages = trajectory['rewards'] - value_pred.detach()
```

`trajectory['rewards']` is shape `(N,)`, `value_pred` is `(N, 1)`. The
subtraction **broadcasts to `(N, N)`**. Same bug at `RL/simple_ppo.py:44` in the
value loss, and in `simple_policy_gradient_using_advantage.py:21`. It doesn't
crash — it silently trains on garbage and eats N^2 memory.

Fix: make `ValueNetwork.forward` return `self.net(obs).squeeze(-1)`.

Two more design issues to be aware of (fixed properly in Stage 2):

- `collect_trajectory` computes reward-to-go as the value target with **no
  bootstrapping at truncation**. When an episode hits the time limit, the critic
  is being told "value here = 0", which is false. Tolerable for LunarLander
  (episodes mostly terminate); fatal for a humanoid (episodes mostly truncate).
- `old_policy_net` is a redundant copy — it is always identical to `policy_net`
  at collection time. Store `log_probs` during the rollout under
  `torch.no_grad()` instead.

**Gate:** re-run LunarLander and confirm it still learns. That is the regression
test for everything that follows.

---

## Stage 1 — Continuous actions

**Time: ~half a day**

Add a `GaussianPolicy` to `RL/utils.py` alongside the existing
`PolicyNetwork`:

```python
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        self.mu_net = <same MLP, output act_dim>
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)   # state-independent

    def forward(self, obs):
        return torch.distributions.Normal(self.mu_net(obs), self.log_std.exp())

    def get_log_probs(self, obs, act):
        return self.forward(obs).log_prob(act).sum(-1)   # <-- sum over action dims
```

Three things that bite people here:

1. **`.sum(-1)`** — `Normal.log_prob` returns *per-dimension* log-probs. The
   joint log-prob over independent dims is their sum. Forget this and the PPO
   ratio is wrong by a factor of 29.
2. **`log_std` is a learnable `nn.Parameter`, not a network output.**
   State-dependent std is technically better but much harder to stabilize; every
   reference locomotion implementation uses the simple version.
3. The current rollout does `.sample().item()` and stores `dtype=torch.int64`.
   Both must become float vectors.

**Gate:** train on `Pendulum-v1` — continuous, 3-dim obs, 1-dim action, solves in
minutes. If Gaussian PPO doesn't reach approximately -200 average return, stop and
debug here, not later.

---

## Stage 2 — Harden PPO to survive 100M steps

**Time: 1–2 days**

Six changes, roughly in order of impact:

### 1. GAE(lambda) replaces `rewards_to_go`

Per timestep:

```
delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t       = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
returns_t = A_t + V(s_t)
```

Use `returns` as the critic target. `lambda = 0.95`. This is the single biggest
fix — reward-to-go has enormous variance over 1000-step episodes.

Critically: distinguish **terminated** (robot fell — bootstrap 0) from
**truncated** (time limit — bootstrap `V(s_T)`).

### 2. Minibatch epochs

Replace `for _ in range(80)` full-batch steps with 5 epochs x ~32 shuffled
minibatches. 80 full-batch gradient steps on one batch of data with a fixed clip
range is a good way to collapse the policy.

### 3. Observation normalization

Running mean/std over observations, updated during rollout, frozen at eval.
Humanoid observations mix radians (~1), rad/s (~10), and velocities — without
this the first layer is hopeless. **Save these stats with the checkpoint.**

### 4. Gradient clipping and entropy bonus

`clip_grad_norm_(params, 0.5)`, and add `-0.01 * dist.entropy().sum(-1).mean()`
to the policy loss.

### 5. KL early-stop

Compute approximate KL per minibatch (`((ratio - 1) - logratio).mean()`) and
break out of the epoch loop above ~0.02.

### 6. Hyperparameters

`lr = 3e-4` (the current default of `1e-2` is ~30x too high for continuous
control), `gamma = 0.99`, `clip_eps = 0.2`.

**Gate:** `HalfCheetah-v5` should exceed 3000 return in ~1M steps, and
`Walker2d-v5` should exceed 2000. If those don't work, G1 categorically will
not. This gate is non-negotiable — it's the difference between debugging PPO and
debugging a reward function.

---

## Stage 3 — Write the G1 environment

**Time: 2–3 days. This is the real work.**

Create `RL/g1_env.py` with a `gymnasium.Env` subclass.

### Model facts

Verified against `mujoco_menagerie/unitree_g1/scene.xml`:

- `nq = 36`, `nv = 35`, `nu = 29`, timestep **0.002 s**
- `qpos[0:7]` = free joint (xyz + wxyz quat); `qpos[7:36]` = 29 joint angles
- `qvel[0:6]` = base linear/angular velocity; `qvel[6:35]` = joint velocities
- Actuators are already **position servos** (`kp=500, dampratio=1`), so
  **action = target joint angle**, not torque — no PD loop needed
- Keyframe 0 = `"stand"`, base height 0.79
- Joint order: `[0:6]` left leg, `[6:12]` right leg, `[12:15]` waist,
  `[15:29]` arms

### Control rate

Step the sim 10 times per env step → **50 Hz control**, `dt = 0.02`. Never run
the policy at 500 Hz.

### Action space

`Box(-1, 1, shape=(12,))` — **legs only for the first attempt.** Hold waist and
arms at their keyframe values. 12 DoF instead of 29 is a dramatically easier
problem and it's how everyone gets first walking.

```
ctrl[leg_idx] = default_q[leg_idx] + 0.5 * action     # then clip to joint range
```

### Observation (45-dim)

Use only what a real robot could measure:

| Component | Dim |
|---|---|
| base angular velocity, body frame | 3 |
| projected gravity, body frame (rotate `[0,0,-1]` by inverse base quat) | 3 |
| velocity command `[vx, vy, wz]` | 3 |
| joint angles - default | 12 |
| joint velocities | 12 |
| previous action | 12 |

Deliberately **exclude** base xy position and base linear velocity. Projected
gravity is the tilt sensor — its z component is -1 upright, approaching 0 when
horizontal.

### Reward

Where most of the time goes. Start with a command-tracking formulation, all
terms scaled by `dt`:

| Term | Formula | Weight |
|---|---|---|
| lin vel tracking | `exp(-‖v_xy_cmd - v_xy‖² / 0.25)` | +1.0 |
| ang vel tracking | `exp(-(wz_cmd - wz)² / 0.25)` | +0.5 |
| vertical velocity | `-v_z²` | -2.0 |
| orientation | `-‖g_proj_xy‖²` | -1.0 |
| base height | `-(h - 0.78)²` | -10.0 |
| action rate | `-‖a_t - a_{t-1}‖²` | -0.01 |
| joint accel / torque | `-‖q̈‖²` | ~-2.5e-7 |
| **feet air time** | `sum(air_time_f - 0.5)` at each touchdown | +1.0 |
| alive | constant | +0.15 |

The **feet air time** term is what produces actual stepping rather than
shuffling — it rewards a foot for having been airborne ~0.5 s before landing.
It needs contact detection: check `data.contact` or use `mj_contactForce` on the
ankle_roll geoms. Add this term *second*, once the robot at least stays upright.

### Termination

`g_proj_z > -0.5` (tilted past ~60 degrees), or base height `< 0.5`, or any
non-foot geom touching the floor. Truncate at 1000 steps (20 s).

### Reset

`mj_resetDataKeyframe(model, data, 0)`, then add noise: joint angles +/-0.1 rad,
joint velocities +/-0.1 rad/s, base yaw uniform in `[-pi, pi]`, base height
+0.02. Without reset randomization the policy overfits to one initial condition
and is not robust.

### Command sampling

Resample `[vx, vy, wz]` every ~10 s from e.g. `vx in [-0.5, 1.0]`,
`vy in [-0.3, 0.3]`, `wz in [-0.5, 0.5]`. For the *first* run, freeze it at
`[0.5, 0, 0]` — learn one skill, then generalize.

### Registration

Register with `gymnasium.register(id="G1Walk-v0", entry_point=...)`. The
training scripts use `env.spec.id` for checkpoint filenames, and an unregistered
env has `spec = None`.

**Gate:** write a script that runs the env with random actions and renders — the
robot should fall over and terminate in under a second. Then run with
`action = 0` (hold default pose) — it should stand for the full 1000 steps and
accumulate positive reward. If zero-action doesn't stand, the action mapping or
default pose is wrong, and no amount of RL will fix that.

---

## Stage 4 — Throughput

**Time: ~1 day**

At ~3k steps/s single-threaded, 100M steps is **9 hours of pure simulation**
with the GPU idle. With 16 cores, wrap in `gymnasium.vector.AsyncVectorEnv`
(16 workers) → 20–40k steps/s.

This forces a rewrite of `collect_trajectory`: vector envs never finish
together, so collect a **fixed horizon**, not whole episodes.

```
rollout buffers shaped (T, N, ...) with T=256, N=16  ->  4096 transitions/iteration
handle autoreset: gymnasium vector envs reset a sub-env automatically;
  the terminal observation arrives in infos, and it's needed for GAE bootstrapping
compute GAE per-env down the T axis, then flatten to (T*N, ...) for the update
```

Sanity check: log wall-clock steps/s every iteration. Under 15k means the env's
`step()` has a Python bottleneck — usually recomputing quaternion math or
allocating arrays every step.

> **Alternative:** MJX on the GPU — thousands of parallel envs, ~50x faster. But
> it means rewriting the env in JAX and abandoning the PyTorch PPO. Not the
> right move while learning the fundamentals. Revisit if CPU training proves too
> slow.

---

## Stage 5 — Train and diagnose

Run 50M steps to start (~1–2 hours at 30k steps/s).

### Log these every iteration

- **Each reward term separately.** Non-negotiable. When the robot does something
  stupid, one term will be dominating — that's the bug.
- **Episode length.** The real progress metric, more than return. Should climb
  from ~30 steps toward 1000. Plateauing at ~50 means the robot falls
  immediately and the standing reward is too weak relative to the velocity
  reward.
- **Approx KL** — should sit around 0.005–0.02. Spiking means lr is too high.
- **`log_std`** — should decrease over training. Rising means the policy is
  diverging.
- **Explained variance** of the critic, `1 - Var(returns - values)/Var(returns)`.
  Below 0.3 means the critic is useless and the advantages are noise.

### Common failure modes

| Symptom | Cause |
|---|---|
| Stands still forever | Velocity reward too weak, or alive bonus too strong — it found the local optimum of "don't fall" |
| Vibrates violently | Action rate penalty too small, or control running faster than 50 Hz |
| Skates/shuffles without lifting feet | Add or increase the feet-air-time reward |
| Learns, then suddenly collapses | lr too high, or missing KL early-stop |

### Checkpointing and playback

Save every N iterations: policy weights **plus** obs-normalizer stats. Adapt
`scripts/mujoco_spawn.py` into a playback script that loads a checkpoint and
drives `data.ctrl` from the deterministic mean action (`dist.mean`, not
`dist.sample()`).

---

## Setup

The venv is missing dependencies — `mujoco` is present, `torch` and `gymnasium`
are not:

```sh
uv add torch gymnasium
```

Clone the model submodule if not already present:

```sh
git submodule update --init --recursive
```

---

## Expectations

Roughly **2–4 weeks of evenings** to first walking, with most of it spent in
Stage 3 tuning rewards. That is normal and it's what everyone spends.
