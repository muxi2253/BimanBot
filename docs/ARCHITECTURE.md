# Architecture

## Runtime structure

```text
robot/robot_config.yaml
        |
        v
RobotAdapter --------------------------+
        |                               |
        v                               v
MuJoCo MjModel                 camera configuration
        |                               |
        +--------+-----------+----------+
                 |           |
                 v           v
          MinkIKSolver   Gym environments
                 |           |
                 +-----+-----+
                       v
             public demo / recordings
                       |
                       v
            qpos -> LeRobot v2.1 files
```

`RobotAdapter` resolves model assets relative to the selected YAML file and
provides arm joints, TCP sites, grippers, actuator names, base joints, camera
mounts, and rest poses. Applications can select another configuration through
an explicit path or the `ROBOT_CONFIG` environment variable.

`MinkIKSolver` consumes a loaded `mujoco.MjModel` and a `RobotKinematics`
description. It does not require a URDF at runtime and does not contain the
public robot's joint names. A replacement robot still needs a valid MJCF and a
matching configuration because kinematics cannot be solved without joint axes,
link transforms, limits, and a TCP definition.

## Public modules

| Module | Responsibility |
|---|---|
| `backend/robot_adapter.py` | Load and validate robot-facing configuration |
| `backend/ik_backend.py` | Model-driven Mink IK and optional collision limits |
| `backend/public_demo.py` | Randomized, collision-checked kinematic pick-place |
| `backend/gym_env_reach.py` | Dual-arm Reach Gym environment |
| `backend/gym_env.py` | Pick-place Gym environment and RGB observations |
| `backend/camera_config.py` | Inject configured cameras and calculate intrinsics |
| `backend/serialize_to_lerobot.py` | Replay qpos and write LeRobot v2.1 files |

The public pick-place demonstration carries an object kinematically after the
gripper reaches it. It verifies IK, trajectory clearance, camera, and recording
behavior; it is not a contact-force grasping benchmark.

## Assets

```text
robot/
  dual_arm_robot.urdf
  dual_arm_robot.xml
  robot_config.yaml
scene/
  combined_scene.xml
  objects/*.xml
assets/obj/
```

The URDF is an optional model source. Runtime code loads MJCF. The public robot
uses project-authored primitive geometry; scene OBJ licensing is recorded in
`ASSET_LICENSES.md`.

## Verification boundary

`tests/test_public_pipeline.py` exercises the supported runtime path. Run:

```bash
python -m pytest -q
python tools/check_release_assets.py
```

The repository does not currently provide teleoperation, a public episode
collector, policy training, pretrained weights, contact-rich grasping, or
LeRobot v3.0 export.

