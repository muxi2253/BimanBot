#!/usr/bin/env python3
"""Simple scene inspector that prints information without GUI."""

import sys
sys.path.insert(0, 'backend')

import mujoco
import numpy as np


def inspect_scene():
    """Print detailed scene information."""
    scene_path = "scene/dual_arm_sorting_scene.xml"

    print(f"=== Loading scene: {scene_path} ===\n")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Initialize
    mujoco.mj_forward(model, data)

    print("SCENE DIMENSIONS:")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  DOF: {model.nv}")
    print(f"  Actuators: {model.nu}")

    print("\n" + "="*60)
    print("INITIAL CONFIGURATION:")
    print("="*60)

    # Bodies
    print("\nKey Bodies:")
    for body_name in ['crate_source', 'crate_target', 'left_object', 'right_object']:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            pos = data.xpos[body_id]
            print(f"  {body_name:20s}: pos={pos} (x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f})")

    # Sites (TCP positions)
    print("\nTCP Sites:")
    for site_name in ['left_tcp', 'right_tcp']:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            pos = data.site_xpos[site_id]
            print(f"  {site_name:20s}: pos={pos} (x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f})")

    # Joint positions
    print("\nJoint Positions (qpos):")
    for i in range(min(14, model.nq)):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name:
            print(f"  Joint {i:2d} ({jnt_name:20s}): {data.qpos[i]:7.3f}")

    print("\n" + "="*60)
    print("SCENE GEOMETRY:")
    print("="*60)

    # Compute distances
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

    print("\nDistances:")
    print(f"  Left TCP to left object:   {np.linalg.norm(left_tcp - left_obj):.3f}m")
    print(f"  Right TCP to right object: {np.linalg.norm(right_tcp - right_obj):.3f}m")
    print(f"  Source to target bin:      {np.linalg.norm(source_bin - target_bin):.3f}m")
    print(f"  Left to right TCP:         {np.linalg.norm(left_tcp - right_tcp):.3f}m")

    print("\n" + "="*60)
    print("PROBLEM ANALYSIS:")
    print("="*60)

    tcp_z = (left_tcp[2] + right_tcp[2]) / 2
    obj_z = (left_obj[2] + right_obj[2]) / 2

    print(f"\n  TCP height:    {tcp_z:.3f}m")
    print(f"  Object height: {obj_z:.3f}m")
    print(f"  Height diff:   {abs(tcp_z - obj_z):.3f}m")

    if abs(tcp_z - obj_z) > 1.0:
        print("\n  ⚠️  WARNING: TCP is >1m away from objects!")
        print("      This explains why the robot can't reach the objects.")
        print("      The initial joint configuration needs adjustment.")

    print("\n" + "="*60)
    print("SUGGESTED FIXES:")
    print("="*60)
    print("\n1. The robot arms start at Z=1.8m but objects are at Z=0.15m")
    print("2. Need to find joint configuration where TCP is near Z=0.5m")
    print("3. Options:")
    print("   a) Manually tune joint angles in scene XML")
    print("   b) Use IK to compute good starting pose")
    print("   c) Simplify task (raise table or lower robot)")
    print()


if __name__ == '__main__':
    inspect_scene()
