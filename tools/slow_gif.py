"""对 GIF 做帧插值，在每两帧之间插入一帧混合过渡帧，动画速度减半。

用法：
    python tools/slow_gif.py input.gif                  # 输出 input_slow.gif
    python tools/slow_gif.py input.gif -o output.gif    # 指定输出路径
    python tools/slow_gif.py input.gif --passes 2       # 连续插帧两轮（速度变为 1/4）

依赖：Pillow、numpy（均已列入项目依赖）
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


def interpolate_gif(src: Path, dst: Path, passes: int = 1) -> None:
    img = Image.open(src)

    frames = []
    for frame in ImageSequence.Iterator(img):
        frames.append(frame.convert("RGBA"))

    original_duration = frames[0].info.get("duration", 50)
    print(f"输入：{src.name}  {len(frames)} 帧  {img.size}  {original_duration}ms/帧")

    for p in range(passes):
        new_frames = []
        for i, frame in enumerate(frames):
            new_frames.append(frame)
            if i < len(frames) - 1:
                arr0 = np.array(frame, dtype=np.float32)
                arr1 = np.array(frames[i + 1], dtype=np.float32)
                blended = ((arr0 + arr1) / 2).astype(np.uint8)
                new_frames.append(Image.fromarray(blended, "RGBA"))
        frames = new_frames
        print(f"  第 {p + 1} 轮插帧完成：{len(frames)} 帧")

    rgb_frames = [f.convert("RGB") for f in frames]
    rgb_frames[0].save(
        dst,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=original_duration,
        loop=0,
        optimize=False,
    )
    total_s = len(rgb_frames) * original_duration / 1000
    print(f"输出：{dst}  总时长 {total_s:.1f}s（原来的 {2 ** passes} 倍）")


def main() -> None:
    parser = argparse.ArgumentParser(description="GIF 帧插值减速工具")
    parser.add_argument("input", type=Path, help="输入 GIF 路径")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 GIF 路径（默认 <原名>_slow.gif）")
    parser.add_argument("--passes", type=int, default=1, help="插帧轮数，每轮速度减半（默认 1）")
    args = parser.parse_args()

    src: Path = args.input
    dst: Path = args.output or src.with_stem(src.stem + "_slow")

    if not src.exists():
        raise FileNotFoundError(f"找不到文件：{src}")

    interpolate_gif(src, dst, passes=args.passes)


if __name__ == "__main__":
    main()
