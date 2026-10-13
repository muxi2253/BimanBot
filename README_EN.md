# Dual-Arm Mobile Manipulator for MuJoCo

This repository contains a runnable public MuJoCo model, a configuration-driven robot adapter, model-driven Mink IK, Gymnasium environments, GIF recording, and LeRobot v2.1 conversion.

## Implemented capabilities

| Capability | Entry point |
|---|---|
| Public mobile dual-arm MJCF | `robot/dual_arm_robot.xml` |
| Robot configuration adapter | `backend/robot_adapter.py` |
| Position/orientation IK with Mink | `backend/ik_backend.py` |
| Two-object pick-and-place animation | `backend/public_demo.py` |
| Dual-arm reaching Gym environment | `backend/gym_env_reach.py` |
| Pick-place Gym environment with three RGB views | `backend/gym_env.py` |
| Friction and damping randomization | `domain_randomization=True` |
| qpos trajectory to LeRobot v2.1 | `backend/serialize_to_lerobot.py` |

The public pick-and-place demonstration uses IK, smooth interpolation, and kinematic object carrying. It validates the kinematics, camera, and recording pipeline; it is not a contact-rich grasping benchmark. Teleoperation, training scripts, depth, and segmentation recording are not included yet.

## Recordable outputs

- `demo_wide.gif`: full workspace view of source-to-destination sorting.
- `demo_closeup.gif`: TCP-following close-up of approach, grasp, lift, transfer, place, and release.
- `reach_episode.gif`: periodic dual-arm motion in the reaching Gym environment.

On macOS, run offscreen rendering through `mjpython`:

```bash
mjpython tools/record_demo.py
mjpython tools/record_demo.py --items kebike_0 --seed 42 --fps 20
mjpython tools/record_gym_episode.py --steps 100 --fps 25
```

SKUs spawn randomly above the source bin and settle under gravity by default.
Use `--seed` for a repeatable layout or `--no-rand` for the authored XML poses.
Use `python` instead on Linux.

## Install and verify

```bash
conda create -n dual-arm python=3.10 -y
conda activate dual-arm
pip install -r requirements.txt

python examples/basic_demo.py
python backend/gym_env_reach.py
python -m pytest
```

Open the interactive viewer with:

```bash
# macOS
mjpython tools/view_robot.py

# Linux / Windows
python tools/view_robot.py
```

## Does IK require a robot-specific URDF?

IK requires a concrete kinematic model because joint axes, link transforms, limits, and TCP frames must come from somewhere. `MinkIKSolver` does not require a vendor-specific URDF and does not hard-code this robot's joint names or DOF count.

```text
URDF (optional source) ─┐
                       ├─> MuJoCo MjModel + robot_config.yaml
authored MJCF ─────────┘                 │
                                        v
                        RobotKinematics -> MinkIKSolver
```

Mink operates directly on the loaded `mujoco.MjModel`. URDF is only one possible source for the MJCF. To swap robots, provide a loadable MJCF and configure arm joints, TCP sites, grippers, the optional base joint, rest poses, and camera mounts in `robot_config.yaml`.

The solver discovers unrelated scalar joints, free objects, and the floating base from the loaded model and pins them by default. The configured base participates only when `allow_base_motion=True` is explicitly requested.

## Gym and LeRobot

```python
from backend.gym_env import DualArmPickPlaceEnv

env = DualArmPickPlaceEnv(
    reward_type="shaped",
    camera_obs=True,
    domain_randomization=True,
)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The LeRobot converter reads `exports/episode_XXXXXX/trajectory_qpos.npy` and `annotation.json`, replays three RGB cameras, and writes Parquet, MP4, and v2.1 metadata:

```bash
pip install pandas pyarrow "imageio[ffmpeg]"
mjpython backend/serialize_to_lerobot.py --src exports --out data/lerobot --fps 10
```

The trajectory width must match the scene's `model.nq`. A teleoperation collector is not part of this public release.

## Layout

```text
robot/                   public URDF, MJCF, and robot_config.yaml
scene/                   crates, objects, and combined scene
backend/ik_backend.py    model-driven Mink IK
backend/public_demo.py   recordable public pick-place demo
backend/gym_env*.py      reaching and pick-place Gym environments
backend/camera_config.py config-driven three-camera injection
tools/record_*.py        GIF recording tools
tests/                   public pipeline regression tests
```

## License

[MIT](LICENSE)
