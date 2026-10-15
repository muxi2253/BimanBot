#!/usr/bin/env python3
"""Visualize the dual-arm pick-place scene."""

import sys
sys.path.insert(0, 'backend')

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


def view_scene():
    """Launch MuJoCo viewer for the scene."""
    scene_path = "scene/dual_arm_sorting_scene.xml"

    print(f"Loading scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Print some info
    print(f"\nScene info:")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  DOF: {model.nv}")

    # Print body positions
    print(f"\nKey body positions:")
    for name in ['crate_source', 'crate_target', 'left_object', 'right_object']:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            pos = data.xpos[body_id]
            print(f"  {name}: {pos}")

    # Print TCP positions
    for name in ['left_tcp', 'right_tcp']:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            pos = data.site_xpos[site_id]
            print(f"  {name}: {pos}")

    print(f"\nLaunching viewer...")
    print(f"Controls:")
    print(f"  Right-drag: rotate camera")
    print(f"  Left-drag: move camera")
    print(f"  Scroll: zoom")
    print(f"  Double-click: select body")
    print(f"  Ctrl+Right-drag: apply force")
    print(f"  Space: pause/resume")

    # Launch passive viewer (you can interact with the scene)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Keep the viewer open
        while viewer.is_running():
            # Step simulation slowly so you can see it
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == '__main__':
    view_scene()
