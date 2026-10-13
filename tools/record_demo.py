"""Record the model-driven public pick/place demo as two GIFs.

Run from the repository root. On macOS, MuJoCo offscreen rendering requires
``mjpython``::

    mjpython tools/record_demo.py
    mjpython tools/record_demo.py --items kebike_0 --fps 20
    mjpython tools/record_demo.py --seed 42

Outputs:
    recordings/demo_wide.gif
    recordings/demo_closeup.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "backend"))

from public_demo import PublicPickPlaceDemo


RENDER_W, RENDER_H = 640, 480
GIF_W, GIF_H = 480, 360

WIDE = dict(
    lookat=[1.02, -0.05, 0.68], distance=2.35,
    azimuth=-50.0, elevation=-10.0)
CLOSE_DIST = 0.82
CLOSE_AZ = 215.0
CLOSE_ELEV = -38.0
CLOSE_ALPHA = 0.16


def _make_camera(*, lookat, distance, azimuth, elevation):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def record(items=None, randomize=True, seed=None):
    demo = PublicPickPlaceDemo()
    model, data = demo.model, demo.data
    model.vis.global_.offwidth = RENDER_W
    model.vis.global_.offheight = RENDER_H

    wide_camera = _make_camera(**WIDE)
    close_camera = _make_camera(
        lookat=[1.15, 0.25, 0.78], distance=CLOSE_DIST,
        azimuth=CLOSE_AZ, elevation=CLOSE_ELEV)
    wide_option = mujoco.MjvOption()
    wide_option.frame = mujoco.mjtFrame.mjFRAME_SITE
    close_option = mujoco.MjvOption()
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    wide_frames: list[Image.Image] = []
    close_frames: list[Image.Image] = []
    last_phase = [None]

    def capture(_model, _data, phase):
        # Keep free-fall out of the presented initial state.
        if phase == "settling":
            return
        tcp = _data.site_xpos[demo.tcp_id]
        target = tcp.copy()
        target[2] = max(target[2], 0.68)
        close_camera.lookat[:] = (
            (1.0 - CLOSE_ALPHA) * np.asarray(close_camera.lookat)
            + CLOSE_ALPHA * target)

        renderer.update_scene(
            _data, camera=wide_camera, scene_option=wide_option)
        wide = Image.fromarray(renderer.render().copy()).resize(
            (GIF_W, GIF_H), Image.Resampling.LANCZOS)
        renderer.update_scene(
            _data, camera=close_camera, scene_option=close_option)
        close = Image.fromarray(renderer.render().copy()).resize(
            (GIF_W, GIF_H), Image.Resampling.LANCZOS)

        copies = 3 if phase != last_phase[0] else 1
        wide_frames.extend([wide] * copies)
        close_frames.extend([close] * copies)
        last_phase[0] = phase

    try:
        result = demo.run(
            items=items, randomize=randomize, seed=seed, callback=capture)
        for _ in range(20):
            capture(model, data, "final")
        print(f"recorded items: {list(result.items)}")
    finally:
        renderer.close()

    return wide_frames, close_frames


def _save_gif(frames, path: Path, duration_ms: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=duration_ms,
        loop=0, optimize=True)
    print(f"saved {path} ({len(frames)} frames, {path.stat().st_size // 1024} KB)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="*", default=None,
                        help="record only the named scene objects")
    parser.add_argument("--fps", type=int, default=20,
                        help="GIF playback rate (default: 20)")
    parser.add_argument("--no-rand", action="store_true",
                        help="use authored item poses instead of randomized free fall")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=REPO / "recordings")
    args = parser.parse_args()

    wide, close = record(args.items, not args.no_rand, args.seed)
    if not wide:
        raise RuntimeError("no frames captured")
    duration_ms = max(1, 1000 // args.fps)
    _save_gif(wide, args.out / "demo_wide.gif", duration_ms)
    _save_gif(close, args.out / "demo_closeup.gif", duration_ms)


if __name__ == "__main__":
    main()
