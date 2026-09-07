'''
To create your own environment in Gymnasium, refer to:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

_XML = Path(__file__).resolve().parent.parent.parent / "mujoco_menagerie/unitree_g1/scene.xml"

# mjx keyframe 'home'. Legs carry load through a bent knee
DEFAULT_Q = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
ACTION_SCALE = np.array([0.50, 0.50, 0.50, 0.387, 0.50, 0.262] * 2)

OBS_DIM = 45    # 3 angular vel, 3 gravity, 3 target cmd, 12 joint pos, 12 joint vel, 12 prev actions
ACT_DIM = 12    # 6 actuators per leg
SUBSTEPS = 10   # 10 * 0.002 (mujoco default physics) = 0.02s -> 50Hz

_GRAVITY = np.array([0.0, 0.0, -1.0])

class G1Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, task: str = "B", render_mode: str | None = None, xml_path: str | None = None):
        self.task = task
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or _XML))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * SUBSTEPS

        self._base_ctrl = self.model.key_ctrl[0].copy()
        # limit values of first ACT_DIM actuators
        self._ctrl_lo = self.model.actuator_ctrlrange[:ACT_DIM, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:ACT_DIM, 1].copy()
        self._q_lo = self.model.jnt_range[1:ACT_DIM + 1, 0].copy()  # first is floating base
        self._q_hi = self.model.jnt_range[1:ACT_DIM + 1, 1].copy()


        # Foot geoms (to find if robot has fallen or just standing/walking)
        self._is_foot = np.zeros(self.model.ngeom, dtype=bool)
        for g in range(self.model.ngeom):
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g])
            if body and body.endswith("ankle_roll_link") and self.model.geom_contype[g]:
                self._is_foot[g] = True

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

        self.render_mode = render_mode
        self._renderer = None
        self._prev_action = np.zeros(ACT_DIM)
        self._cmd = np.zeros(3)  # [target_{vx}, target_{vy}, target_{yaw_rate}]

    def _projected_gravity(self) -> np.ndarray:
        q_inv, g = np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(q_inv, self.data.qpos[3:7])
        mujoco.mju_rotVecQuat(g, _GRAVITY, q_inv)
        return g

    def _obs(self) -> np.ndarray:
        d = self.data
        return np.concatenate([
            d.qvel[3:6],               # 3 angular velocity, body frame
            self._projected_gravity(), # 3 gravity vector (tilt)
            self._cmd,                 # 3 [vx, vy, wz]
            d.qpos[7:19] - DEFAULT_Q,  # 12 joint poisitions (relative to default pose)
            d.qvel[6:18],              # 12 joint velocities
            self._prev_action,         # 12 previous actions
        ]).astype(np.float32)

    def _sample_command(self) -> np.ndarray:
        if self.task in ("A", "B"):
            return np.zeros(3)
        if self.task == "C":
            return np.array([0.5, 0.0, 0.0])
        return np.array([
            self.np_random.uniform(-0.5, 1.0),
            self.np_random.uniform(-0.3, 0.3),
            self.np_random.uniform(-0.5, 0.5),
        ])

    def _terminated(self, g_proj: np.ndarray) -> bool:
        # torso tilted too far or height too low (fallen)
        if g_proj[2] > -0.5 or self.data.qpos[2] < 0.5:
            return True
        n = self.data.ncon
        if n:
            g1 = self.data.contact.geom1[:n]
            g2 = self.data.contact.geom2[:n]
            # if any contact is not with a foot
            if np.any(~self._is_foot[g1] & ~self._is_foot[g2]):
                return True
        return False

    def _reward_terms(self, action: np.ndarray) -> dict:
        # placeholder: survival only first.
        return {"alive": 1.0}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        q, v = self.data.qpos, self.data.qvel

        # Domain randomization: add small random noise to starting position/velocities
        q[7:19] = np.clip(DEFAULT_Q + self.np_random.uniform(-0.1, 0.1, ACT_DIM),
                          self._q_lo, self._q_hi)
        v[6:18] = self.np_random.uniform(-0.1, 0.1, ACT_DIM)

        # Randomize starting heading angle
        yaw = self.np_random.uniform(-np.pi, np.pi)
        q[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        # Drop from slightly above ground to avoid initial penetration
        q[2] += 0.02

        self._prev_action[:] = 0.0
        self._cmd = self._sample_command()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        ctrl = self._base_ctrl.copy()
        ctrl[:ACT_DIM] = np.clip(DEFAULT_Q + ACTION_SCALE * action,
                                 self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        reward_terms = self._reward_terms(action)
        self._prev_action[:] = action
        obs = self._obs()
        terminated = self._terminated(self._projected_gravity())
        return obs, float(sum(reward_terms.values())), terminated, False, {"reward_terms": reward_terms}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam_distance, self._cam.elevation, self._cam.azimuth = 3.5, -15.0, 120.0
        self._cam.lookat[:] = self.data.qpos[:3]
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
