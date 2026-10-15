#!/usr/bin/env python3
"""Launch MuJoCo passive viewer to visualize the scene."""

import mujoco
import mujoco.viewer
import time


def view_scene(scene_path="scene/dual_arm_sorting_scene.xml"):
    """Launch interactive viewer for the scene."""
    print(f"Loading scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    print("\nScene loaded successfully!")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print("\nLaunching viewer...")
    print("  - Use mouse to rotate view")
    print("  - Press SPACE to pause/play simulation")
    print("  - Press ESC or close window to exit")
    print()

    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run simulation
        while viewer.is_running():
            step_start = time.time()

            # Step physics
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Maintain real-time rate (30 FPS)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == '__main__':
    import sys
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "scene/dual_arm_sorting_scene.xml"
    view_scene(scene_path)
