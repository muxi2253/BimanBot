"""Test scripted policy and visualize robot workspace."""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
from dual_arm_scripted_policy import DualArmScriptedPolicy


def test_workspace():
    """Test robot workspace and initial configuration."""
    env = DualArmPickPlaceEnv(render_mode=None)
    obs, info = env.reset()

    print("=== Scene Geometry ===")
    print(f"Left TCP (initial): {obs[14:17]}")
    print(f"Right TCP (initial): {obs[17:20]}")
    print(f"Left object position: {obs[20:23]}")
    print(f"Right object position: {obs[23:26]}")
    print(f"Target left (from info): {info['target_left']}")
    print(f"Target right (from info): {info['target_right']}")
    print(f"\nJoint positions (initial): {obs[:14]}")

    # Test a few simple joint configurations
    print("\n=== Testing Joint Configurations ===")

    test_configs = [
        ("Rest", obs[:14]),  # Current
        ("Zeros", np.zeros(14)),
        ("Forward reach", np.array([0.0, -1.0, 0.0, -1.5, 0.0, 0.8, 0.0] * 2)),
    ]

    for name, qpos in test_configs:
        # Set joints
        env.data.qpos[:14] = qpos
        import mujoco
        mujoco.mj_forward(env.model, env.data)

        left_tcp = env.data.site_xpos[env.left_tcp_id]
        right_tcp = env.data.site_xpos[env.right_tcp_id]

        print(f"\n{name}:")
        print(f"  Joints: {qpos}")
        print(f"  Left TCP: {left_tcp}")
        print(f"  Right TCP: {right_tcp}")
        print(f"  Dist to left obj: {np.linalg.norm(left_tcp - obs[20:23]):.3f}")
        print(f"  Dist to right obj: {np.linalg.norm(right_tcp - obs[23:26]):.3f}")

    env.close()


if __name__ == '__main__':
    test_workspace()
