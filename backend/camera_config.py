"""Camera configuration derived from the public robot adapter.

Camera mount bodies and poses belong to the robot asset, so they live in
``robot/robot_config.yaml``.  This module only converts those mounts into the
format used by the renderer and computes pinhole intrinsics from MuJoCo FOV.
Private deployments can supply their own config and calibrated intrinsics
without changing the IK or task code.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np

try:
    from .robot_adapter import RobotAdapter
except ImportError:  # Direct script/module execution from backend/.
    from robot_adapter import RobotAdapter


REPO = Path(__file__).resolve().parent.parent
ROBOT_CONFIG_PATH = Path(os.environ.get(
    "ROBOT_CONFIG", REPO / "robot" / "robot_config.yaml"))
_ADAPTER = RobotAdapter(str(ROBOT_CONFIG_PATH))

RENDER_W, RENDER_H = 640, 480
CAMERA_CONFIG = _ADAPTER.get_camera_mounts()


def intrinsics_from_fovy(fovy_deg, width, height):
    fovy_rad = math.radians(float(fovy_deg))
    focal = 0.5 * height / math.tan(0.5 * fovy_rad)
    return dict(fx=focal, fy=focal, cx=width / 2, cy=height / 2)


# The public asset has no real-device calibration.  Its rendered camera model
# is fully described by fovy at the render resolution.
CALIB_W, CALIB_H = RENDER_W, RENDER_H
CAMERA_INTRINSICS = {
    name: intrinsics_from_fovy(cfg["fovy"], RENDER_W, RENDER_H)
    for name, cfg in CAMERA_CONFIG.items()
}
CAMERA_DISTORTIONS = {
    name: np.zeros(5, dtype=float) for name in CAMERA_CONFIG
}


def scale_intrinsics(intr, src_w=CALIB_W, src_h=CALIB_H,
                     dst_w=RENDER_W, dst_h=RENDER_H):
    sx, sy = dst_w / src_w, dst_h / src_h
    return dict(fx=intr["fx"] * sx, fy=intr["fy"] * sy,
                cx=intr["cx"] * sx, cy=intr["cy"] * sy)


CAMERA_INTRINSICS_RENDER = {
    name: scale_intrinsics(intr) for name, intr in CAMERA_INTRINSICS.items()
}


def load_scene_with_cameras(camera_config=None, adapter=None):
    """Load the configured scene and inject body-fixed cameras."""
    adapter = adapter or _ADAPTER
    camera_config = camera_config or adapter.get_camera_mounts()
    scene_rel = adapter.get_asset_path("scene")
    if scene_rel is None:
        raise ValueError("robot config does not define assets.scene")
    scene_path = Path(scene_rel)

    cwd = os.getcwd()
    try:
        os.chdir(scene_path.parent)
        spec = mujoco.MjSpec.from_file(scene_path.name)
        for name, cfg in camera_config.items():
            body = spec.body(cfg["body"])
            if body is None:
                raise ValueError(
                    f"camera '{name}' parent body not found: {cfg['body']}")
            camera = body.add_camera()
            camera.name = name
            camera.pos = cfg["pos"]
            camera.quat = cfg["quat"]
            camera.fovy = cfg["fovy"]
        model = spec.compile()
    finally:
        os.chdir(cwd)

    model.vis.global_.offwidth = RENDER_W
    model.vis.global_.offheight = RENDER_H
    return model, mujoco.MjData(model)
