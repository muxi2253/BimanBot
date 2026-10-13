#!/usr/bin/env python3
"""
Convert video to GIF for README demo

Usage:
    python tools/convert_to_gif.py input.mov output.gif

Requirements:
    pip install pillow imageio imageio-ffmpeg
"""

import sys
import imageio
from pathlib import Path


def convert_video_to_gif(input_path, output_path, fps=10, scale=0.5):
    """
    Convert video to optimized GIF

    Args:
        input_path: Path to input video (.mov, .mp4, etc.)
        output_path: Path to output GIF
        fps: Frames per second (lower = smaller file)
        scale: Scale factor (0.5 = half size)
    """
    print(f"Converting {input_path} to GIF...")
    print(f"  FPS: {fps}")
    print(f"  Scale: {scale}")

    # Read video
    reader = imageio.get_reader(input_path)

    # Get video info
    meta = reader.get_meta_data()
    original_fps = meta.get('fps', 30)
    frame_skip = int(original_fps / fps)

    print(f"  Original FPS: {original_fps}")
    print(f"  Frame skip: {frame_skip}")

    # Read and resize frames
    frames = []
    for i, frame in enumerate(reader):
        if i % frame_skip == 0:
            # Resize
            if scale != 1.0:
                import numpy as np
                from PIL import Image
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = Image.fromarray(frame)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                frame = np.array(img)

            frames.append(frame)

    print(f"  Total frames: {len(frames)}")

    # Write GIF
    imageio.mimsave(
        output_path,
        frames,
        fps=fps,
        loop=0  # Loop forever
    )

    # Check file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ GIF created: {output_path}")
    print(f"  Size: {size_mb:.2f} MB")

    if size_mb > 10:
        print("\n⚠ Warning: GIF is large (>10MB)")
        print("  GitHub recommends GIFs under 10MB")
        print("  Try: python tools/convert_to_gif.py input.mov output.gif --fps 8 --scale 0.4")


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/convert_to_gif.py input.mov output.gif [--fps 10] [--scale 0.5]")
        print("\nExample:")
        print("  python tools/convert_to_gif.py ~/Desktop/screen_recording.mov docs/images/robot_demo.gif")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Parse optional arguments
    fps = 10
    scale = 0.5

    for i, arg in enumerate(sys.argv[3:], start=3):
        if arg == '--fps' and i + 1 < len(sys.argv):
            fps = int(sys.argv[i + 1])
        elif arg == '--scale' and i + 1 < len(sys.argv):
            scale = float(sys.argv[i + 1])

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert
    convert_video_to_gif(input_path, output_path, fps=fps, scale=scale)

    print("\n✓ Done! Now update README.md:")
    print(f"  Replace 'docs/images/robot_demo.gif' with '{output_path}'")


if __name__ == '__main__':
    main()
