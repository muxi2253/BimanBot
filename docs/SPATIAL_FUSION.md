# Spatial fusion experiment

This is an independent experiment module. It reuses the existing MuJoCo scene,
camera injection and renderer, and does not change the robot control pipeline.

The first version compares four non-learning methods when a fixed camera sees a
current scene and an older scene is available as a depth prior:

- `prior_only`
- `observation_only`
- `fixed` (default weight 0.5)
- `rule` (validity plus local observation noise)

Each sample writes metric depth in metres, validity masks, segmentation masks,
camera extrinsics, a clean current depth truth, a degraded observation, and a
preview image. It also writes `truth_pointcloud.ply` and one PLY point cloud per
baseline; these are world-frame points and can be opened in MeshLab,
CloudCompare or Blender. `metrics.csv` contains depth MAE, changed-region MAE, and
same-pixel 3D point error. The changed region is derived from prior/current
segmentation and depth difference; truth is not passed to fusion inference.

MAE means mean absolute error:

```text
MAE = mean(abs(predicted_depth - truth_depth))
```

It is measured in metres here. For example, `0.005` means an average absolute
depth error of 5 mm over valid pixels. `changed_depth_mae_m` computes the same
quantity only in pixels where the prior and current scene differ. `point_mae_m`
is the average Euclidean distance between corresponding backprojected 3D points.

Run from the repository root:

```bash
conda run -n lerobot python tools/spatial_fusion_experiment.py \
  --samples 20 --seed 7 --out outputs/spatial_fusion
```

The current implementation uses the known camera pose and the `top` camera by
default. `--camera left_wrist` and `--camera right_wrist` are also supported.
The two alternating cases are `prior_wrong_observation_accurate` and
`prior_accurate_observation_degraded`. Train/test splitting and the PyTorch
weight predictor are intentionally deferred until the metric/unit checks pass.

On headless Linux, configure the MuJoCo rendering backend before running, for
example `MUJOCO_GL=egl` or the backend supported by the machine. On macOS, the
renderer needs an active CoreGraphics connection.
