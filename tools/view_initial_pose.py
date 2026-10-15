#!/usr/bin/env python3
"""查看机器人初始姿态的可视化工具。"""

import sys
sys.path.insert(0, 'backend')

from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
import numpy as np

print("启动环境以查看初始姿态...")
print("按 Ctrl+C 退出")
print("")

# Create environment with human rendering
env = DualArmPickPlaceEnv(render_mode="human")

# Reset to see initial pose
obs, info = env.reset()

# Print info
left_tcp = obs[14:17]
right_tcp = obs[17:20]
left_obj = obs[20:23]
right_obj = obs[23:26]

print("初始状态:")
print(f"  左臂TCP: [{left_tcp[0]:.2f}, {left_tcp[1]:.2f}, {left_tcp[2]:.2f}]")
print(f"  右臂TCP: [{right_tcp[0]:.2f}, {right_tcp[1]:.2f}, {right_tcp[2]:.2f}]")
print(f"  左物体:  [{left_obj[0]:.2f}, {left_obj[1]:.2f}, {left_obj[2]:.2f}]")
print(f"  右物体:  [{right_obj[0]:.2f}, {right_obj[1]:.2f}, {right_obj[2]:.2f}]")
print("")
print("查看3D窗口中的机器人姿态...")
print("双臂应该在胸前位置，接近桌面高度")
print("")

try:
    # Keep window open
    while True:
        env.step(np.zeros(16))  # No-op action
except KeyboardInterrupt:
    print("\n退出")

env.close()
