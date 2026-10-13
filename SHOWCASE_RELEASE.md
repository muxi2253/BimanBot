# Public showcase release

This showcase uses only the public educational robot and scene shipped in
this repository:

- `robot/dual_arm_robot.xml` and `robot/robot_config.yaml`
- `scene/combined_scene.xml`
- the public object assets under `assets/obj/` and `scene/objects/`

No private Gino URDF, MJCF, CAD mesh, calibration file, or production control
code is included.

## What to show

| Artifact | Description |
|---|---|
| `recordings/showcase_public/demo_wide.gif` | Full public pick-place workspace view |
| `recordings/showcase_public/demo_closeup.gif` | TCP-following gripper view |
| `recordings/showcase_public/reach_episode.gif` | Public dual-arm Reach environment |
| `recordings/showcase_public/spatial_fusion.mp4` | Four non-learning depth-prior baselines |
| `recordings/showcase_public/lerobot/` | LeRobot v2.1 episode with three RGB camera videos |
| `recordings/showcase_public/exports/episode_000000/` | qpos trajectory, RGB, metric depth, instance masks and point clouds |

The episode and spatial-fusion initial states are post-settle: each randomized SKU is dropped into
the source crate, simulated under gravity until stable, and only then recorded
as `initial_qpos.npy`. The authored floating XML pose is not used as the data
episode's initial state or as the spatial-fusion scene state.

## Data notes

Depth arrays are NumPy files in metres. Segmentation arrays are integer label
maps. `*_pointcloud.xyz` files are world-frame XYZ points back-projected from
the public top camera. The LeRobot dataset contains the standard RGB videos,
joint/gripper state, TCP state and next-step actions; depth, masks and point
clouds remain in the accompanying raw episode directory.

## Reproduction

```bash
python tools/record_demo.py --seed 50 --out recordings/showcase_public
python tools/record_gym_episode.py --steps 140 --output recordings/showcase_public/reach_episode.gif
python tools/spatial_fusion_experiment.py --samples 20 --seed 7 --out recordings/showcase_public/spatial_fusion
python tools/export_public_data.py --out recordings/showcase_public/exports
python backend/serialize_to_lerobot.py \
  --src recordings/showcase_public/exports \
  --out recordings/showcase_public/lerobot --fps 10 --overwrite
```

Run `python tools/check_release_assets.py` before publishing the source tree.
