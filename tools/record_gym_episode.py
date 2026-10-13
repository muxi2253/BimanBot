#!/usr/bin/env python3
"""Record the public dual-arm reaching environment as an animated GIF.

On macOS, MuJoCo offscreen rendering must be launched with ``mjpython``::

    mjpython tools/record_gym_episode.py --steps 100
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from gym_env_reach import DualArmReachEnv


def record_episode(output_path: Path, num_steps: int = 100,
                   render_size: tuple[int, int] = (640, 480),
                   fps: int = 25) -> Path:
    """Run a deterministic motion sequence and write an animated GIF."""
    env = DualArmReachEnv(max_steps=num_steps)
    renderer = mujoco.Renderer(
        env.model, height=render_size[1], width=render_size[0])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = 2.0
    camera.azimuth = 140
    camera.elevation = -18

    frames = []
    info = {"total_distance": float("nan")}
    try:
        env.reset(seed=0)
        for step in range(num_steps):
            phase = 2.0 * np.pi * step / max(1, num_steps - 1)
            action = np.zeros(14, dtype=np.float32)
            action[[1, 8]] = 0.012 * np.sin(phase)
            action[[3, 10]] = 0.010 * np.sin(phase + np.pi / 2)
            action[[5, 12]] = 0.008 * np.sin(2 * phase)
            _, _, terminated, truncated, info = env.step(action)
            renderer.update_scene(env.data, camera=camera)
            frames.append(Image.fromarray(renderer.render().copy()))
            if terminated or truncated:
                break

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / fps),
            loop=0,
            optimize=False,
        )
    finally:
        renderer.close()
        env.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"saved {len(frames)} frames to {output_path} ({size_mb:.2f} MB)")
    print(f"final TCP distance: {info['total_distance']:.3f} m")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("recordings/reach_episode.gif"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    record_episode(args.output, args.steps, (args.width, args.height), args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
