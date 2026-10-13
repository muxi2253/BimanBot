#!/usr/bin/env python3
"""Fail when the public release tree contains stale or unresolved assets."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {".DS_Store", "MUJOCO_LOG.TXT", ".env", "CLAUDE.md"}
FORBIDDEN_SUFFIXES = {".bak", ".backup"}
FORBIDDEN_DIRS = {"__pycache__", ".pytest_cache", ".claude"}
FORBIDDEN_RELEASE_PATHS = {
    ".gitmodules",
    "assets/obj/textures/jpg_to_png.py",
    "backend/arm_servo.py",
    "backend/export_visible.ipynb",
    "backend/grasp_strategy.py",
    "backend/pick_phases.py",
    "backend/place_strategy.py",
    "backend/proxy_geometry.py",
    "backend/robot_config.py",
    "backend/sim_ctx.py",
    "backend/snapshot.py",
    "backend/sort_demo.py",
    "backend/sort_demo_mink.py",
    "docs/OPEN_SOURCE_CODE_AUDIT.md",
    "docs/PHD_SIMULATION_ROADMAP.md",
    "docs/PROGRESS_DAY1.md",
    "docs/RECORDING_OPTIONS.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/RL_TRAINING_GUIDE.md",
    "external",
    "scene/scene-v2.0.xml",
    "submodules",
    "tools/add_damping_batch.py",
    "tools/render_episode.py",
    "trans.py",
}
REQUIRED_FILES = {
    "LICENSE",
    "ASSET_LICENSES.md",
    "PUBLIC_ASSETS_REQUIRED.md",
    "robot/robot_config.yaml",
}


def collect_errors() -> list[str]:
    errors: list[str] = []
    for relative in sorted(REQUIRED_FILES):
        if not (ROOT / relative).is_file():
            errors.append(f"missing required release file: {relative}")

    for relative in sorted(FORBIDDEN_RELEASE_PATHS):
        if (ROOT / relative).exists():
            errors.append(f"legacy or internal file restored: {relative}")

    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if (path.name in FORBIDDEN_NAMES
                or path.suffix in FORBIDDEN_SUFFIXES
                or any(part in FORBIDDEN_DIRS for part in path.parts)):
            errors.append(f"forbidden release artifact: {path.relative_to(ROOT)}")

    config_path = ROOT / "robot" / "robot_config.yaml"
    config = {}
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for kind, raw_path in config.get("assets", {}).items():
            path = Path(os.path.expandvars(str(raw_path))).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            if not path.resolve().is_file():
                errors.append(
                    f"configured asset does not exist: assets.{kind}={raw_path}")

    try:
        import mujoco

        scene_value = config.get("assets", {}).get("scene")
        if not scene_value:
            raise ValueError("robot config does not define assets.scene")
        scene = Path(os.path.expandvars(str(scene_value))).expanduser()
        if not scene.is_absolute():
            scene = config_path.parent / scene
        mujoco.MjModel.from_xml_path(str(scene.resolve()))
    except Exception as exc:  # Import and model errors both block a release.
        errors.append(f"public scene failed to load: {exc}")

    return errors


def main() -> int:
    errors = collect_errors()
    if errors:
        print("Release asset check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Release asset check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
