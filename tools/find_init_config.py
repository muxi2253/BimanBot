#!/usr/bin/env python3
"""Find good initial joint configuration for dual-arm robot.

Uses simple search to find joint angles that place TCP near objects.
"""

import sys
sys.path.insert(0, 'backend')

import mujoco
import numpy as np


def test_joint_config(model, data, config, target_z=0.6):
    """Test a joint configuration and return TCP positions."""
    # Set joint positions (skip base joints 0-3)
    data.qpos[4:4+7] = config[:7]  # Left arm
    data.qpos[13:13+7] = config[7:14]  # Right arm

    mujoco.mj_forward(model, data)

    left_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_tcp')
    right_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_tcp')

    left_tcp = data.site_xpos[left_tcp_id].copy()
    right_tcp = data.site_xpos[right_tcp_id].copy()

    # Compute error
    left_error = abs(left_tcp[2] - target_z)
    right_error = abs(right_tcp[2] - target_z)

    return left_tcp, right_tcp, left_error + right_error


def find_good_config():
    """Search for good joint configuration."""
    scene_path = "scene/dual_arm_sorting_scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    print("Searching for good initial joint configuration...")
    print("Target TCP height: 0.6m (near objects at 0.54m)\n")

    # Common joint configurations to try
    test_configs = [
        # Name, [left_arm_7_joints, right_arm_7_joints]
        ("Zeros", np.zeros(14)),
        ("Forward bent", np.array([0, -1.0, 0, -1.5, 0, 0.5, 0] * 2)),
        ("Wide reach", np.array([0.3, -0.8, 0, -1.8, 0, 1.0, 0, -0.3, -0.8, 0, -1.8, 0, 1.0, 0])),
        ("Table reach", np.array([0, -1.2, 0, -2.0, 0, 0.8, 0] * 2)),
        ("Low reach", np.array([0, -1.5, 0, -2.2, 0, 0.7, 0] * 2)),
    ]

    best_config = None
    best_error = float('inf')
    best_info = None

    print(f"{'Configuration':<20} {'Left TCP':>30} {'Right TCP':>30} {'Error':>10}")
    print("=" * 95)

    for name, config in test_configs:
        left_tcp, right_tcp, error = test_joint_config(model, data, config)

        print(f"{name:<20} [{left_tcp[0]:6.3f}, {left_tcp[1]:6.3f}, {left_tcp[2]:6.3f}]  "
              f"[{right_tcp[0]:6.3f}, {right_tcp[1]:6.3f}, {right_tcp[2]:6.3f}]  {error:8.3f}")

        if error < best_error:
            best_error = error
            best_config = config
            best_info = (name, left_tcp, right_tcp)

    print("\n" + "=" * 95)
    print(f"BEST: {best_info[0]}")
    print(f"  Left TCP:  {best_info[1]}")
    print(f"  Right TCP: {best_info[2]}")
    print(f"  Error: {best_error:.3f}m")
    print(f"\nJoint configuration:")
    print(f"  {best_config}")

    # Check reachability
    left_obj = np.array([0.85, 0.08, 0.54])
    right_obj = np.array([0.85, -0.08, 0.54])

    left_dist = np.linalg.norm(best_info[1] - left_obj)
    right_dist = np.linalg.norm(best_info[2] - right_obj)

    print(f"\nDistance to objects:")
    print(f"  Left:  {left_dist:.3f}m")
    print(f"  Right: {right_dist:.3f}m")

    if left_dist < 0.5 and right_dist < 0.5:
        print("\n✅ This configuration should work!")
    else:
        print("\n⚠️  Still too far from objects. Need better configuration.")

    return best_config


if __name__ == '__main__':
    config = find_good_config()
