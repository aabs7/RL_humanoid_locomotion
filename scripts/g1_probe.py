import mujoco
import numpy as np

XML = 'mujoco_menagerie/unitree_g1/scene.xml'
SUBSTEPS = 10   # 10 * 0.002 = 0.02s -> 50Hz

DEFAULT_Q = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
ACTION_SCALE = np.array([0.5, 0.5, 0.5, 0.387, 0.5, 0.262] * 2)

def foot_geoms(m):
    ids = []
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if body and body.endswith('ankle_roll_link') and m.geom_contype[g]:
            ids.append(g)
    return ids

def projected_gravity(d):
    qinv = np.zeros(4)
    g = np.zeros(3)
    mujoco.mju_negQuat(qinv, d.qpos[3:7])  # conjugate of torso orientation
    mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), qinv)  # rotate gravity into torso frame
    return g

def reset(m, d, yaw=0.0):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # quaternion for yaw
    d.qpos[7:19] = DEFAULT_Q  # default joint positions
    mujoco.mj_forward(m, d)

def step(m, d, action, base_ctrl):
    ctrl = base_ctrl.copy()
    ctrl[:12] = DEFAULT_Q + ACTION_SCALE * action
    np.clip(ctrl[:12], m.actuator_ctrlrange[:12, 0], m.actuator_ctrlrange[:12, 1], out=ctrl[:12])
    d.ctrl[:] = ctrl
    for _ in range(SUBSTEPS):
        mujoco.mj_step(m, d)

def run(m, d, base_ctrl, feet, policy, n=1000, label=""):
    reset(m, d)
    peak_tau, first_bad = 0.0, None
    for i in range(n):
        step(m, d, policy(d), base_ctrl)
        peak_tau = max(peak_tau, float(np.abs(d.actuator_force[:12]).max()))
        g = projected_gravity(d)
        bad = any(
            (d.contact.geom1[c] not in feet and d.contact.geom2[c] not in feet)
            for c in range(d.ncon)
        )
        if first_bad is None and (bad or g[2] > -0.5 or d.qpos[2] < 0.5):
            first_bad = i
    g = projected_gravity(d)
    print(f"{label:16s} survived={first_bad if first_bad is not None else n:>4}/{n}"
          f"  h={d.qpos[2]:.3f}  g_z={g[2]:+.3f}  peak|tau|={peak_tau:5.1f}")


m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)


# Print name of Joints
# print("\nJoints:")
# for i in range(m.njnt):
#     j_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
#     print(f"Index {i:2d}: {j_name}")


# Print name of Actuators
# print("\nActuators:")
# for i in range(m.nu):
#     u_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#     print(f"Index {i:2d}: {u_name}")

ids = foot_geoms(m)
base_ctrl = m.key_ctrl[0].copy()
# print(f"\nLength of base_ctrl: {len(base_ctrl)} "
#       "   i.e., 0 - 11 (leg) , 12 (waist) , 13 - 28 (arms & hands)")
feet = foot_geoms(m)
rng = np.random.default_rng(0)

print(f"nq: {m.nq}, nv={m.nv}, nu={m.nu} mass={m.body_subtreemass[0]:.2f} kg")
print(f"foot geoms: {sorted(feet)}")

# 1. zero action must hold the pose for full episode
run(m, d, base_ctrl, feet, lambda d: np.zeros(12), label="zero action")

# 2. random action must fall quickly
run(m, d, base_ctrl, feet, lambda d: rng.uniform(-1.0, 1.0, size=12), label="random action")

# 3. projected gravity in three known orientations
for name, quat in [("upright", [1, 0, 0, 0]),
                   ("pitched 45deg", [np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0]),
                   ("on its side", [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])]:
    reset(m, d)
    d.qpos[3:7] = quat
    mujoco.mj_forward(m, d)
    print(f"    g_proj {name:14s} {np.round(projected_gravity(d), 3)}")
