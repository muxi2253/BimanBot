#!/usr/bin/env python3
"""
Simple demo: Load robot and run basic movements

This demonstrates:
1. Loading the robot model
2. Using the robot adapter
3. Setting joint positions
4. Running simulation
"""

import mujoco
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from robot_adapter import RobotAdapter


def main():
    print("=" * 60)
    print("Dual-Arm Robot - Basic Demo")
    print("=" * 60)

    # Load robot
    print("\n[1/4] Loading robot model...")
    model = mujoco.MjModel.from_xml_path('robot/dual_arm_robot.xml')
    data = mujoco.MjData(model)
    print(f"  ✓ Model loaded: {model.nbody} bodies, {model.nu} actuators")

    # Load adapter
    print("\n[2/4] Loading robot adapter...")
    adapter = RobotAdapter('robot/robot_config.yaml')
    print(f"  ✓ Robot type: {adapter.robot_type}")
    print(f"  ✓ Left arm joints: {adapter.get_arm_joints('left')}")
    print(f"  ✓ Right arm joints: {adapter.get_arm_joints('right')}")

    # Initialize pose
    print("\n[3/4] Setting initial pose...")

    # Set base height
    data.qpos[2] = 0.0  # wheels and chassis rest on the floor

    # Set arm rest poses
    left_rest = adapter.get_rest_pose('left')
    right_rest = adapter.get_rest_pose('right')

    # Map to qpos (need to find joint addresses)
    for i, jname in enumerate(adapter.get_arm_joints('left')):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            addr = model.jnt_qposadr[jid]
            data.qpos[addr] = left_rest[i]

    for i, jname in enumerate(adapter.get_arm_joints('right')):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            addr = model.jnt_qposadr[jid]
            data.qpos[addr] = right_rest[i]

    # Position actuators must start at the same targets as qpos. Leaving their
    # controls at zero creates an artificial impulse on the first step.
    for jname in adapter.get_arm_joints('left') + adapter.get_arm_joints('right'):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, adapter.get_actuator_name(jname))
        data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]

    torso = adapter.config.get('torso', {})
    if torso:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, torso['joint'])
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR,
            adapter.get_actuator_name(torso['joint']))
        torso_q = float(adapter.rest_pose.get('torso', 0.0))
        data.qpos[model.jnt_qposadr[jid]] = torso_q
        data.ctrl[aid] = torso_q

    for side in ('left', 'right'):
        opened, _ = adapter.get_gripper_range(side)
        for jname in adapter.get_gripper_joints(side):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            data.qpos[model.jnt_qposadr[jid]] = opened
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR,
            adapter.grippers[side]['actuator'])
        data.ctrl[aid] = opened

    # Forward kinematics
    mujoco.mj_forward(model, data)
    print("  ✓ Robot initialized to rest pose")

    # Get TCP positions
    left_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_tcp')
    right_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_tcp')

    left_tcp_pos = data.site_xpos[left_tcp_id]
    right_tcp_pos = data.site_xpos[right_tcp_id]

    print(f"  Left TCP:  [{left_tcp_pos[0]:.3f}, {left_tcp_pos[1]:.3f}, {left_tcp_pos[2]:.3f}]")
    print(f"  Right TCP: [{right_tcp_pos[0]:.3f}, {right_tcp_pos[1]:.3f}, {right_tcp_pos[2]:.3f}]")

    # Run simulation
    print("\n[4/4] Running simulation...")
    sim_time = 2.0  # seconds
    steps = int(sim_time / model.opt.timestep)

    print(f"  Simulating {sim_time}s ({steps} steps @ {model.opt.timestep*1000:.1f}ms timestep)...")

    for i in range(steps):
        mujoco.mj_step(model, data)

        # Print progress
        if (i + 1) % 500 == 0:
            print(f"    t = {data.time:.2f}s")

    print(f"  ✓ Simulation complete (final time: {data.time:.2f}s)")

    # Final state
    print("\n" + "=" * 60)
    print("Final State:")
    print("=" * 60)

    left_tcp_pos_final = data.site_xpos[left_tcp_id]
    right_tcp_pos_final = data.site_xpos[right_tcp_id]

    print(f"Left TCP:  [{left_tcp_pos_final[0]:.3f}, {left_tcp_pos_final[1]:.3f}, {left_tcp_pos_final[2]:.3f}]")
    print(f"Right TCP: [{right_tcp_pos_final[0]:.3f}, {right_tcp_pos_final[1]:.3f}, {right_tcp_pos_final[2]:.3f}]")
    print(f"Base height: {data.qpos[2]:.3f}m")

    # Check stability
    qvel_norm = np.linalg.norm(data.qvel)
    print(f"Velocity norm: {qvel_norm:.6f} (should be near zero if stable)")

    if qvel_norm < 0.01:
        print("\n✓ Robot is stable!")
    else:
        print("\n⚠ Robot has residual motion (may need damping tuning)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - View configured home pose: mjpython tools/view_robot.py")
    print("  - Run model-driven IK demo: mjpython tools/record_demo.py")
    print("  - Run RL environment: python backend/gym_env_reach.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
