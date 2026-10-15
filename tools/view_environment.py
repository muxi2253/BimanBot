#!/usr/bin/env python3
"""Visualize the dual-arm pick-place environment with manual control."""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
import time


def view_environment():
    """Launch environment viewer with controllable simulation."""
    print("Creating environment with visualization...")
    env = DualArmPickPlaceEnv(render_mode='human')

    obs, info = env.reset()

    print("\n=== Scene Information ===")
    print(f"Left TCP (initial): {obs[14:17]}")
    print(f"Right TCP (initial): {obs[17:20]}")
    print(f"Left object: {obs[20:23]}")
    print(f"Right object: {obs[23:26]}")
    print(f"Target left: {info['target_left']}")
    print(f"Target right: {info['target_right']}")
    print(f"\nJoint positions (initial): {obs[:14]}")

    print("\n=== Viewer Controls ===")
    print("The MuJoCo viewer window should be open.")
    print("You can:")
    print("  - Right-drag: rotate camera")
    print("  - Scroll: zoom in/out")
    print("  - Double-click: select body")
    print("  - Hold Ctrl + drag: apply forces")
    print("\nThe simulation will run slowly so you can observe.")
    print("Press Ctrl+C in terminal to quit.")

    try:
        step = 0
        while True:
            # Apply zero action (just hold position)
            action = np.zeros(16)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                print(f"  Success: {info.get('episode_success', False)}")
                print(f"  Left placed: {info.get('left_placed', False)}")
                print(f"  Right placed: {info.get('right_placed', False)}")
                print("\nResetting...")
                obs, info = env.reset()
                step = 0

            # Print status every 50 steps
            if step % 50 == 0:
                print(f"\nStep {step}:")
                print(f"  Left TCP: {obs[14:17]}")
                print(f"  Right TCP: {obs[17:20]}")
                print(f"  Left obj: {obs[20:23]}")
                print(f"  Right obj: {obs[23:26]}")

            step += 1
            time.sleep(0.05)  # Slow down for viewing

    except KeyboardInterrupt:
        print("\n\nShutting down viewer...")

    env.close()
    print("Done!")


if __name__ == '__main__':
    view_environment()
