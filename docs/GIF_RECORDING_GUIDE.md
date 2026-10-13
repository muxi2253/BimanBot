# GIF recording guide

Run commands from the repository root after installing `requirements.txt`.

## Pick-place views

```bash
# macOS
mjpython tools/record_demo.py

# Linux with a working OpenGL/EGL context
python tools/record_demo.py
```

One run produces:

- `recordings/demo_wide.gif`: full workspace view;
- `recordings/demo_closeup.gif`: TCP-following close-up.

Useful options:

```bash
mjpython tools/record_demo.py --items kebike_0 --seed 42 --fps 20
mjpython tools/record_demo.py --no-rand
mjpython tools/record_demo.py --out recordings/seed42
```

By default, objects spawn above the source crate and settle under gravity. A
seed reproduces the layout; `--no-rand` uses the MJCF positions.

## Reach environment

```bash
# macOS
mjpython tools/record_gym_episode.py --steps 100 --fps 25

# Linux with a working rendering context
python tools/record_gym_episode.py --steps 100 --fps 25
```

The default output is `recordings/reach_episode.gif`.

## Convert a screen recording

```bash
python tools/convert_to_gif.py input.mov output.gif --fps 8 --scale 0.4
```

Reduce frame rate, scale, or duration before publishing large GIF files. The
generated recordings are ignored by default; choose an explicit media hosting
or repository policy before release.
