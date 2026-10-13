#!/usr/bin/env python3
"""Open the public robot in its configured folded home pose."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from robot_adapter import RobotAdapter, load_default_adapter  # noqa: E402


def _joint_qpos(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"configured joint not found in model: {name}")
    return int(model.jnt_qposadr[joint_id])


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise ValueError(f"configured actuator not found in model: {name}")
    return int(actuator_id)


def initialize_home(model: mujoco.MjModel, data: mujoco.MjData,
                    adapter: RobotAdapter) -> None:
    """Apply the configured rest pose and matching position controls."""
    mujoco.mj_resetData(model, data)

    for side in ("left", "right"):
        joint_names = adapter.get_arm_joints(side)
        rest = adapter.get_rest_pose(side)
        if len(joint_names) != len(rest):
            raise ValueError(f"{side} rest pose length does not match joint list")
        for name, value in zip(joint_names, rest):
            qpos = _joint_qpos(model, name)
            data.qpos[qpos] = float(value)
            data.ctrl[_actuator_id(
                model, adapter.get_actuator_name(name))] = float(value)

        opened, _ = adapter.get_gripper_range(side)
        for name in adapter.get_gripper_joints(side):
            data.qpos[_joint_qpos(model, name)] = opened
        gripper = adapter.grippers.get(side, {})
        if gripper.get("actuator"):
            data.ctrl[_actuator_id(model, gripper["actuator"])] = opened

    torso = adapter.config.get("torso", {})
    if torso.get("joint"):
        name = torso["joint"]
        value = float(adapter.rest_pose.get("torso", 0.0))
        data.qpos[_joint_qpos(model, name)] = value
        data.ctrl[_actuator_id(
            model, adapter.get_actuator_name(name))] = value

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData,
               adapter: RobotAdapter) -> None:
    """Run the viewer on the main thread and step physics locally."""
    if platform.system() == "Darwin" and not os.environ.get("MJPYTHON_BIN"):
        raise RuntimeError(
            "MuJoCo's macOS viewer must run through mjpython. Activate the "
            "project environment, then run: mjpython tools/view_robot.py"
        )

    reset_home = False

    def on_key(keycode: int) -> None:
        nonlocal reset_home
        if keycode in (ord("h"), ord("H")):
            reset_home = True

    try:
        with mujoco.viewer.launch_passive(
                model, data, key_callback=on_key) as viewer:
            viewer.sync()
            while viewer.is_running():
                frame_start = time.monotonic()
                if reset_home:
                    initialize_home(model, data, adapter)
                    reset_home = False

                mujoco.mj_step(model, data)
                viewer.sync()
                remaining = model.opt.timestep - (time.monotonic() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        return
    except RuntimeError as exc:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "MuJoCo could not create the macOS viewer. Run this command "
                "from an active desktop session: mjpython tools/view_robot.py"
            ) from exc
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="robot config YAML; defaults to ROBOT_CONFIG or public config")
    args = parser.parse_args()

    adapter = RobotAdapter(args.config) if args.config else load_default_adapter()
    model_path = adapter.get_asset_path("model")
    if model_path is None:
        raise ValueError("robot config does not define assets.model")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    initialize_home(model, data, adapter)
    print("Loaded configured folded home pose. Press H to reset it; close the "
          "viewer to exit.")
    run_viewer(model, data, adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
