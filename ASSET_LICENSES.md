# Asset inventory and release permissions

**Last reviewed**: 2026-10-11

This file separates source-code licensing from the permission to distribute
binary and visual assets. The root [`LICENSE`](LICENSE) applies to source code
and the project-authored files explicitly listed below. It does not silently
relicense an asset received under separate terms.

## Project-authored assets — MIT

The following simplified educational assets are authored for this public
project and distributed under the root MIT License:

- `robot/dual_arm_robot.urdf`
- `robot/dual_arm_robot.xml`
- `robot/robot_config.yaml`
- `scene/combined_scene.xml`
- `scene/objects/*.xml`
- `assets/obj/mesh_params.json`

The robot URDF and MJCF use primitive geometry and contain no vendor CAD mesh
or real-device calibration data.

## Public scene asset bundle — redistribution approved

The project owner confirmed on 2026-10-11 that the OBJ asset bundle may be
included in the public repository and public release archives. This approval
covers the files needed to render those objects:

- `assets/obj/*.obj`
- `assets/obj/*.mtl`
- `assets/obj/textures/*`

| Field | Recorded value |
|---|---|
| Source | Project-provided public scene asset bundle |
| Release authority | Project owner confirmation on 2026-10-11 |
| Repository redistribution | Permitted |
| Required attribution | None specified |
| Independent reuse/modification | Not declared; do not assume MIT |

Release authorization confirms that the repository may publicly distribute
these files. It does not, by itself, declare them MIT or grant downstream users
permission to extract and relicense them independently. If the rights holder
chooses a standard reusable asset license later, record its SPDX identifier,
source URL, author, and attribution here before changing that scope.

The current public scene uses the crate, `kebike`, and `bluecup` assets. Other
objects remain in the bundle for optional scenes and experiments.

## Generated media

GIF, video, dataset, log, and checkpoint outputs are generated artifacts and
are ignored by default. A selected demo may be published separately when its
size and hosting location are decided; publishing a rendering does not change
the underlying asset terms above.

## Adding an asset

Every new non-code asset must record:

1. file or directory;
2. author or source;
3. license or explicit redistribution permission;
4. required attribution;
5. whether modification and downstream redistribution are allowed.

Do not merge an asset when any of these fields is unknown.
