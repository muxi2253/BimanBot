# Quick start

Run all commands from the repository root.

## Install

Python 3.10 or newer is recommended.

```bash
conda create -n dual-arm python=3.10 -y
conda activate dual-arm
pip install -r requirements.txt
```

## Verify the public release

```bash
python tools/verify_robot.py
python examples/basic_demo.py
python -m pytest -q
python tools/check_release_assets.py
```

The public scene currently has 29 bodies, 19 actuators, `nq=42`, `nv=39`, and
a 2 ms simulation timestep. The regression suite covers configured model names,
model-driven IK isolation, randomized object settling, kinematic pick-place with
collision checks, and Gym numerical stability.

## Open the robot viewer

On macOS, MuJoCo requires `mjpython` for an interactive viewer:

```bash
mjpython tools/view_robot.py
```

On Linux or Windows:

```bash
python tools/view_robot.py
```

The viewer script applies the rest pose from `robot/robot_config.yaml`. Press
`H` to restore that pose.

## Run the Gym environments

```bash
python examples/gym_demo.py
python backend/gym_env_reach.py
```

Programmatic pick-place usage:

```python
from backend.gym_env import DualArmPickPlaceEnv

env = DualArmPickPlaceEnv(
    reward_type="shaped",
    camera_obs=False,
    domain_randomization=True,
)
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

Set `camera_obs=True` to include the `top`, `left_wrist`, and `right_wrist` RGB
observations. On macOS, camera rendering also needs a desktop graphics session.

## Record the public demonstrations

On macOS:

```bash
mjpython tools/record_demo.py
mjpython tools/record_gym_episode.py
```

The first command writes the workspace and TCP-following pick-place views. The
second records the dual-arm Reach environment. See
[`GIF_RECORDING_GUIDE.md`](GIF_RECORDING_GUIDE.md) for options.

## Export an existing trajectory

The converter accepts `episode_XXXXXX/trajectory_qpos.npy` plus
`annotation.json` and emits LeRobot v2.1 Parquet, video, and metadata files:

```bash
mjpython backend/serialize_to_lerobot.py \
  --src exports \
  --out data/lerobot \
  --fps 10
```

The trajectory width must equal the scene's `model.nq`. This repository does
not yet contain a public episode collector. LeRobot 0.6.1 expects dataset v3.0
and cannot directly load this v2.1 output.

