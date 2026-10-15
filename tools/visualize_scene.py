#!/usr/bin/env python3
"""Simple viewer using matplotlib to visualize the scene from top."""

import sys
sys.path.insert(0, 'backend')

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def visualize_scene_2d():
    """Create 2D top-down view of the scene."""
    scene_path = "scene/dual_arm_sorting_scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Get positions
    left_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_tcp')
    right_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_tcp')
    left_obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_object')
    right_obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_object')
    source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'crate_source')
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'crate_target')

    left_tcp = data.site_xpos[left_tcp_id]
    right_tcp = data.site_xpos[right_tcp_id]
    left_obj = data.xpos[left_obj_id]
    right_obj = data.xpos[right_obj_id]
    source_bin = data.xpos[source_id]
    target_bin = data.xpos[target_id]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Top-down view (X-Y plane)
    ax1.set_title('Top-Down View (X-Y plane)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters) - Forward →', fontsize=12)
    ax1.set_ylabel('Y (meters) - Left →', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Robot base
    ax1.plot(0, 0, 'ks', markersize=15, label='Robot Base')

    # Bins (as rectangles)
    ax1.add_patch(Rectangle((source_bin[0]-0.25, source_bin[1]-0.2), 0.5, 0.4,
                            fill=True, alpha=0.3, color='blue', label='Source Bin'))
    ax1.add_patch(Rectangle((target_bin[0]-0.25, target_bin[1]-0.2), 0.5, 0.4,
                            fill=True, alpha=0.3, color='orange', label='Target Bin'))

    # Objects
    ax1.plot(left_obj[0], left_obj[1], 'ro', markersize=10, label='Left Object')
    ax1.plot(right_obj[0], right_obj[1], 'go', markersize=10, label='Right Object')

    # TCPs
    ax1.plot(left_tcp[0], left_tcp[1], 'r^', markersize=12, label='Left TCP')
    ax1.plot(right_tcp[0], right_tcp[1], 'g^', markersize=12, label='Right TCP')

    # Lines from TCP to objects
    ax1.plot([left_tcp[0], left_obj[0]], [left_tcp[1], left_obj[1]], 'r--', alpha=0.5)
    ax1.plot([right_tcp[0], right_obj[0]], [right_tcp[1], right_obj[1]], 'g--', alpha=0.5)

    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.8, 0.8)

    # Side view (X-Z plane)
    ax2.set_title('Side View (X-Z plane)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (meters) - Forward →', fontsize=12)
    ax2.set_ylabel('Z (meters) - Up ↑', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Robot base
    ax2.plot(0, 0, 'ks', markersize=15, label='Robot Base')

    # Bins
    ax2.add_patch(Rectangle((source_bin[0]-0.25, 0), 0.5, source_bin[2],
                            fill=True, alpha=0.3, color='blue'))
    ax2.add_patch(Rectangle((target_bin[0]-0.25, 0), 0.5, target_bin[2],
                            fill=True, alpha=0.3, color='orange'))

    # Objects
    ax2.plot(left_obj[0], left_obj[2], 'ro', markersize=10)
    ax2.plot(right_obj[0], right_obj[2], 'go', markersize=10)

    # TCPs
    ax2.plot(left_tcp[0], left_tcp[2], 'r^', markersize=12)
    ax2.plot(right_tcp[0], right_tcp[2], 'g^', markersize=12)

    # Lines from TCP to objects
    ax2.plot([left_tcp[0], left_obj[0]], [left_tcp[2], left_obj[2]], 'r--', alpha=0.5)
    ax2.plot([right_tcp[0], right_obj[0]], [right_tcp[2], right_obj[2]], 'g--', alpha=0.5)

    # Add distance annotations
    left_dist = np.linalg.norm(left_tcp - left_obj)
    right_dist = np.linalg.norm(right_tcp - right_obj)
    ax2.text(0.5, 1.5, f'Left TCP to Object: {left_dist:.2f}m\nRight TCP to Object: {right_dist:.2f}m',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.1, 2.0)

    plt.tight_layout()
    plt.savefig('scene_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to: scene_visualization.png")
    print("   Open this file to see the scene layout.")

    plt.show()


if __name__ == '__main__':
    visualize_scene_2d()
