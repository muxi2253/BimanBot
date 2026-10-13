#!/usr/bin/env python3
"""Reproducible depth-prior fusion experiment.

This module is independent of the robot control loop.  It reuses the existing
MuJoCo scene/camera injection path and keeps the camera fixed while moving
selected object bodies between a prior and current scene.

第一版只包含数据生成、四个非学习基线、深度/点云评测和可视化；真值只用于
评测，不会进入融合推理。学习模型放在后续阶段。

Example (from repository root)::

    conda run -n lerobot python tools/spatial_fusion_experiment.py \
        --samples 20 --out outputs/spatial_fusion --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from camera_config import (CAMERA_CONFIG, CAMERA_INTRINSICS_RENDER, RENDER_H,
                           RENDER_W, load_scene_with_cameras)


def _body_freejoint(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"body not found: {body_name}")
    for jid in range(model.njnt):
        if model.jnt_bodyid[jid] == bid and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[jid])
    raise ValueError(f"body has no freejoint: {body_name}")


def _post_settle_qpos(model: mujoco.MjModel, data: mujoco.MjData,
                      item_names: tuple[str, ...], seed: int) -> np.ndarray:
    """Create the public demo's randomized, gravity-settled initial scene."""
    rng = np.random.default_rng(seed)
    data.qpos[:] = model.qpos0
    for name in item_names:
        adr = _body_freejoint(model, name)
        if name.startswith("kebike"):
            xy = [rng.uniform(1.240, 1.245), rng.uniform(0.17, 0.18)]
            yaw = rng.uniform(-0.08, 0.08)
        else:
            xy = [rng.uniform(1.120, 1.125), rng.uniform(0.34, 0.36)]
            yaw = rng.uniform(-np.pi, np.pi)
        data.qpos[adr:adr + 3] = [xy[0], xy[1], rng.uniform(0.76, 0.80)]
        data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    for _ in range(900):
        mujoco.mj_step(model, data)
        if _ > 250 and float(np.max(np.abs(data.qvel))) < 0.015:
            break
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return data.qpos.copy()


def _render_depth(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy().astype(np.float32)
    renderer.disable_depth_rendering()
    return rgb, depth


def _render_seg(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str,
                model: mujoco.MjModel, item_names: Iterable[str]) -> np.ndarray:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    raw = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    geom = raw[:, :, 0]
    kind = raw[:, :, 1]
    valid = (geom >= 0) & (kind == int(mujoco.mjtObj.mjOBJ_GEOM))
    geom_safe = np.where(valid, geom, 0)
    body = model.geom_bodyid[geom_safe]
    labels = np.zeros(geom.shape, dtype=np.int32)
    for idx, name in enumerate(item_names, 1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            labels[body == bid] = idx
    return labels


def _corrupt_depth(depth: np.ndarray, rng: np.random.Generator,
                   sigma: float, dropout: float) -> tuple[np.ndarray, np.ndarray]:
    """Add metric Gaussian noise and missing pixels; return depth and validity."""
    valid = np.isfinite(depth) & (depth > 0)
    noisy = depth.astype(np.float32, copy=True)
    noisy[valid] += rng.normal(0.0, sigma, size=int(valid.sum())).astype(np.float32)
    missing = rng.random(depth.shape) < dropout
    valid &= ~missing & (noisy > 0)
    noisy[~valid] = 0.0
    return noisy, valid


def _sanitize_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Convert MuJoCo far-plane/background samples to the invalid value 0.

    MuJoCo's renderer returns a finite far-plane distance for pixels that do
    not hit useful scene geometry.  Treating those pixels as real depth makes
    the preview nearly black and injects noise into the background.
    """
    out = np.asarray(depth, dtype=np.float32).copy()
    out[~np.isfinite(out) | (out <= 0) | (out > max_depth)] = 0.0
    return out


def _local_std(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """3x3 local standard deviation without treating missing depth as zero."""
    p = np.pad(depth, 1, mode="edge")
    v = np.pad(valid.astype(np.float32), 1, mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(p, (3, 3))
    vwin = np.lib.stride_tricks.sliding_window_view(v, (3, 3))
    count = vwin.sum(axis=(-2, -1))
    mean = (windows * vwin).sum(axis=(-2, -1)) / np.maximum(count, 1.0)
    var = ((windows - mean[..., None, None]) ** 2 * vwin).sum(axis=(-2, -1)) / np.maximum(count, 1.0)
    return np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def fuse(prior: np.ndarray, observation: np.ndarray, *, method: str,
         observation_valid: np.ndarray, fixed_weight: float = 0.5) -> np.ndarray:
    """Fuse two metric depth maps; invalid pixels are handled explicitly."""
    pv = np.isfinite(prior) & (prior > 0)
    ov = observation_valid & np.isfinite(observation) & (observation > 0)
    out = np.zeros(prior.shape, dtype=np.float32)
    both = pv & ov
    if method == "prior_only":
        out[pv] = prior[pv]
    elif method == "observation_only":
        out[ov] = observation[ov]
    else:
        if method == "fixed":
            w = np.full(prior.shape, fixed_weight, dtype=np.float32)
        elif method == "rule":
            noise = _local_std(observation, ov)
            w = np.exp(-noise / 0.01).astype(np.float32)
            w[~ov] = 0.0
        else:
            raise ValueError(f"unknown fusion method: {method}")
        out[both] = (1.0 - w[both]) * prior[both] + w[both] * observation[both]
        out[pv & ~ov] = prior[pv & ~ov]
        out[ov & ~pv] = observation[ov & ~pv]
    return out


def backproject(depth: np.ndarray, intr: dict, cam_pos: np.ndarray,
                cam_xmat: np.ndarray) -> np.ndarray:
    """Backproject axial depth into world coordinates using MuJoCo camera axes."""
    cam_xmat = np.asarray(cam_xmat, dtype=np.float32).reshape(3, 3)
    cam_pos = np.asarray(cam_pos, dtype=np.float32).reshape(3)
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    z = depth.astype(np.float32)
    x = (xx - float(intr["cx"])) / float(intr["fx"]) * z
    y = (float(intr["cy"]) - yy) / float(intr["fy"]) * z
    # MuJoCo cameras look along local -Z; local +Y is image-up.
    local = np.stack((x, y, -z), axis=-1)
    world = local @ cam_xmat.T + cam_pos.reshape(1, 1, 3)
    return world


def _metrics(pred: np.ndarray, truth: np.ndarray, changed: np.ndarray,
             intr: dict, cam_pos: np.ndarray, cam_xmat: np.ndarray) -> dict:
    valid = (pred > 0) & (truth > 0) & np.isfinite(pred) & np.isfinite(truth)
    changed_valid = valid & changed
    row = {
        "valid_fraction": float(valid.mean()),
        "depth_mae_m": float(np.mean(np.abs(pred[valid] - truth[valid]))) if valid.any() else float("nan"),
        "changed_fraction": float(changed.mean()),
        "changed_depth_mae_m": float(np.mean(np.abs(pred[changed_valid] - truth[changed_valid]))) if changed_valid.any() else float("nan"),
    }
    p = backproject(pred, intr, cam_pos, cam_xmat)
    t = backproject(truth, intr, cam_pos, cam_xmat)
    row["point_mae_m"] = float(np.mean(np.linalg.norm(p[valid] - t[valid], axis=1))) if valid.any() else float("nan")
    row["changed_point_mae_m"] = float(np.mean(np.linalg.norm(p[changed_valid] - t[changed_valid], axis=1))) if changed_valid.any() else float("nan")
    return row


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write finite world-frame points as a portable ASCII PLY file."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is None:
        colors = np.full((len(points), 3), 220, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[valid]
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def _save_preview(path: Path, truth: np.ndarray, prior: np.ndarray,
                  observation: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    names = ["truth", "prior", "observation"] + list(predictions)
    arrays = [truth, prior, observation] + list(predictions.values())
    lo, hi = float(np.nanpercentile(truth[truth > 0], 2)), float(np.nanpercentile(truth[truth > 0], 98))
    tiles = []
    for name, arr in zip(names, arrays):
        scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        im = Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")
        canvas = Image.new("RGB", (im.width, im.height + 22), "white")
        canvas.paste(im, (0, 22))
        ImageDraw.Draw(canvas).text((4, 4), name, fill="black")
        tiles.append(canvas)
    out = Image.new("RGB", (tiles[0].width * len(tiles), tiles[0].height), "white")
    for i, tile in enumerate(tiles):
        out.paste(tile, (i * tile.width, 0))
    out.save(path)


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    item_names = tuple(args.items.split(","))
    model, data = load_scene_with_cameras(CAMERA_CONFIG)
    settled_qpos = _post_settle_qpos(model, data, item_names, args.seed)
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    camera = args.camera
    if camera not in CAMERA_INTRINSICS_RENDER:
        raise ValueError(f"unknown camera {camera}; choose from {sorted(CAMERA_INTRINSICS_RENDER)}")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"camera not found: {camera}")
    # The renderer is 640x480 while the source calibration is 1920x1536.
    # Always use the scaled intrinsics matching the actual output array.
    intr = CAMERA_INTRINSICS_RENDER[camera]
    rows = []
    methods = ("prior_only", "observation_only", "fixed", "rule")
    for sample in range(args.samples):
        data.qpos[:] = settled_qpos
        for name in item_names:
            adr = _body_freejoint(model, name)
            if sample % 2 == 0 and name == item_names[0]:
                data.qpos[adr:adr + 3] += np.asarray(args.shift, dtype=float)
        mujoco.mj_forward(model, data)
        _, prior_depth = _render_depth(renderer, data, camera)
        prior_seg = _render_seg(renderer, data, camera, model, item_names)
        prior_depth = _sanitize_depth(prior_depth, args.max_depth)

        # Current scene: move one object while camera pose remains unchanged.
        if item_names:
            adr = _body_freejoint(model, item_names[0])
            data.qpos[adr:adr + 3] += np.asarray(args.shift, dtype=float)
        mujoco.mj_forward(model, data)
        rgb, truth = _render_depth(renderer, data, camera)
        truth = _sanitize_depth(truth, args.max_depth)
        current_seg = _render_seg(renderer, data, camera, model, item_names)
        observation, observation_valid = _corrupt_depth(truth, rng, args.noise_sigma, args.dropout)
        if sample % 2 == 0:
            prior_for_case, obs_for_case = prior_depth, truth
            case = "prior_wrong_observation_accurate"
        else:
            prior_for_case, obs_for_case = truth, observation
            case = "prior_accurate_observation_degraded"
        changed = (prior_seg != current_seg) | (np.abs(prior_for_case - truth) > args.change_threshold)
        case_observation_valid = np.isfinite(obs_for_case) & (obs_for_case > 0)
        predictions = {m: fuse(prior_for_case, obs_for_case, method=m,
                               observation_valid=case_observation_valid,
                               fixed_weight=args.fixed_weight) for m in methods}
        sample_dir = out / f"sample_{sample:05d}"
        sample_dir.mkdir(exist_ok=True)
        np.save(sample_dir / "prior_depth.npy", prior_for_case)
        np.save(sample_dir / "current_depth_truth.npy", truth)
        np.save(sample_dir / "observation_depth.npy", obs_for_case)
        np.save(sample_dir / "observation_valid.npy", observation_valid)
        np.save(sample_dir / "changed_mask.npy", changed)
        np.save(sample_dir / "rgb.npy", rgb)
        np.save(sample_dir / "prior_seg.npy", prior_seg)
        np.save(sample_dir / "current_seg.npy", current_seg)
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_xmat = data.cam_xmat[cam_id].copy().reshape(3, 3)
        np.save(sample_dir / "camera_position.npy", cam_pos)
        np.save(sample_dir / "camera_xmat.npy", cam_xmat)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = cam_xmat
        extrinsic[:3, 3] = cam_pos
        np.save(sample_dir / "camera_extrinsic_local_to_world.npy", extrinsic)
        truth_points = backproject(truth, intr, cam_pos, cam_xmat)
        truth_points[truth <= 0] = np.nan
        truth_colors = rgb.reshape(-1, 3)
        _write_ply(sample_dir / "truth_pointcloud.ply", truth_points.reshape(-1, 3), truth_colors)
        for method, pred in predictions.items():
            pred_points = backproject(pred, intr, cam_pos, cam_xmat)
            pred_points[pred <= 0] = np.nan
            _write_ply(sample_dir / f"{method}_pointcloud.ply", pred_points.reshape(-1, 3), truth_colors)
        meta = {"seed": args.seed, "sample": sample, "case": case, "camera": camera,
                "intrinsics": intr, "noise_sigma_m": args.noise_sigma,
                "dropout": args.dropout, "shift_m": list(args.shift),
                "initial_state": "post_settle_public_sku_freefall",
                "camera_position_world": cam_pos.tolist(),
                "camera_xmat_local_to_world": cam_xmat.tolist(),
                "camera_forward_world": (-cam_xmat[:, 2]).tolist(),
                "camera_convention": "local +X right, +Y image-up, camera looks along local -Z"}
        (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _save_preview(sample_dir / "depth_comparison.png", truth, prior_for_case, obs_for_case, predictions)
        for method, pred in predictions.items():
            metric = _metrics(pred, truth, changed, intr, cam_pos, cam_xmat)
            rows.append({"sample": sample, "case": case, "method": method, **metric})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "experiment.json").write_text(json.dumps({"seed": args.seed, "samples": args.samples,
        "camera": camera, "items": item_names, "methods": methods}, indent=2), encoding="utf-8")
    renderer.close()
    print(f"wrote {len(rows)} metric rows to {out / 'metrics.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="outputs/spatial_fusion")
    p.add_argument("--camera", default="top")
    p.add_argument("--items", default="kebike_0,bluecup_0")
    p.add_argument("--shift", nargs=3, type=float, default=(0.08, -0.04, 0.0), metavar=("DX", "DY", "DZ"))
    p.add_argument("--noise-sigma", type=float, default=0.008)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--fixed-weight", type=float, default=0.5)
    p.add_argument("--change-threshold", type=float, default=0.015)
    p.add_argument("--max-depth", type=float, default=10.0,
                   help="far-plane/background cutoff in metres")
    run(p.parse_args())


if __name__ == "__main__":
    main()
