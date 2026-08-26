import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_g1/scene_with_hands.xml")
data = mujoco.MjData(model)

# List all the joint names
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"Joint {i}: {joint_name}")

mujoco.mj_resetDataKeyframe(model, data, 0)   # load the "stand" keyframe

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
