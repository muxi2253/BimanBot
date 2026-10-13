# Public asset boundary

**Status updated**: 2026-10-11

The runnable public asset bundle is implemented with a separately authored,
primitive-geometry robot model:

- `robot/dual_arm_robot.urdf`
- `robot/dual_arm_robot.xml`
- `robot/robot_config.yaml`
- `scene/combined_scene.xml`
- `assets/obj/`

The public robot contains no vendor CAD mesh. Its geometry, dimensions, masses,
joint limits, names, cameras, and gripper are the simplified educational model
recorded in `robot_config.yaml`. Private robot URDF/MJCF, calibration data, and
production hardware assets must not be copied into this repository.

## Relocatable asset configuration

`RobotAdapter` treats the selected YAML file as the root of an asset bundle.
Relative entries under `assets` resolve from the YAML file's directory:

```yaml
assets:
  urdf: dual_arm_robot.urdf
  model: dual_arm_robot.xml
  scene: ../scene/combined_scene.xml
```

Applications may select another bundle by passing `config_path` or setting
`ROBOT_CONFIG=/absolute/path/to/robot_config.yaml`. Absolute asset paths,
environment variables, and `~` are also supported. This configuration-based
mechanism replaces the earlier proposed `ROBOT_ASSET_ROOT` variable, so there
is only one source of truth.

The verified public demo, camera loader, pick-place Gym environment, and Reach
Gym environment use this resolver. The former WebSocket simulation server,
legacy experiment pipeline, and their dedicated tests are outside the public
project scope and are not shipped.

## Asset release status

The project owner has confirmed that the OBJ scene asset bundle is approved
for public release. The per-file inventory and license scope are documented in
[`ASSET_LICENSES.md`](ASSET_LICENSES.md). Code and project-authored robot/scene
definitions follow the repository `LICENSE`; third-party or separately granted
asset rights do not become MIT merely because the files are stored here.

## Release hygiene

Run before creating a public archive or Git tag:

```bash
python tools/check_release_assets.py
```

The check rejects OS metadata, backup URDFs, MuJoCo logs, private environment
files, missing license records, broken configured asset paths, and an unloadable
public scene.
