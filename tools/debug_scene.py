"""Debug scene orientation with visual markers."""
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('scene/dual_arm_sorting_scene.xml')
data = mujoco.MjData(model)

# IMPORTANT: Forward kinematics first!
mujoco.mj_forward(model, data)

# Get crate IDs
source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'crate_source')
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'crate_target')

print("Scene Debug Info:")
print(f"Source crate position: {data.xpos[source_id]}")
print(f"Target crate position: {data.xpos[target_id]}")
print(f"Source crate quat: {data.xquat[source_id]}")
print(f"Target crate quat: {data.xquat[target_id]}")

# Get rotation matrix
def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

source_rot = quat_to_matrix(data.xquat[source_id])
print("\nSource crate axes:")
print(f"  X (red): {source_rot[:, 0]}")
print(f"  Y (green): {source_rot[:, 1]}")
print(f"  Z (blue): {source_rot[:, 2]}")

print("\n箱子开口应该:")
print("  - 朝上 (Z轴正方向)")
print("  - 开口朝向机器人 (-X方向)")

print("\n启动viewer... 用鼠标旋转查看箱子方向")
print("如果箱子方向还不对，告诉我:")
print("  1. 开口朝哪个方向?")
print("  2. 应该朝哪个方向?")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
