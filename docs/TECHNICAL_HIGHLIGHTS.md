# Technical highlights

This project demonstrates a complete public MuJoCo simulation path for a
dual-arm mobile manipulator.

## Configuration-driven robot integration

Robot-facing names and rest poses live in `robot/robot_config.yaml`.
`RobotAdapter` resolves the MJCF, scene, joints, TCP sites, grippers, actuators,
base, torso, and camera mounts. The supported algorithms do not embed vendor
names or require a specific URDF path.

## Model-driven inverse kinematics

`MinkIKSolver` works from the loaded MuJoCo model and a small
`RobotKinematics` description. It pins all non-target joints by default,
supports position and optional orientation targets, and can add collision
avoidance limits for supplied geom pairs. Mobile-base motion is opt-in.

## Collision-checked motion demonstration

The public pick-place demo uses Cartesian waypoints, adaptive joint
interpolation, MuJoCo contact checks, randomized object settling, and seeded
reproducibility. Unsafe samples stop the demonstration with phase and collision
details. Object carrying is kinematic and is described as such in user-facing
documentation.

## Gym and visual observations

The repository includes a 14-action dual-arm Reach environment and a 16-action
pick-place environment. The latter supports sparse, dense, and shaped rewards,
three RGB cameras, and friction and joint-damping randomization.

## Reproducible output paths

The recording tools generate workspace, TCP close-up, and Reach GIFs. The
dataset converter replays full qpos trajectories into three camera videos and
LeRobot v2.1 Parquet/metadata files. Native LeRobot v3.0 output remains future
work.

## Verified scope

The public regression suite checks configuration/model consistency, IK joint
isolation, object settling, collision-checked pick-place, and Gym numerical
stability. `tools/check_release_assets.py` additionally checks the asset bundle,
configuration paths, forbidden release artifacts, and scene loading.

Current limitations are contact-force grasping, public episode collection,
training scripts and checkpoints, depth/segmentation dataset export, LeRobot
v3.0, and sim-to-real deployment.

