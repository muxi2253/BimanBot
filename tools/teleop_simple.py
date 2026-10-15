#!/usr/bin/env python3
"""简化版命令行遥控演示收集。

每步输入命令控制机器人，适合macOS等不支持实时GUI交互的环境。
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
import pickle
from pathlib import Path
from datetime import datetime


class SimpleTeleopCollector:
    """命令行遥控收集器。"""

    def __init__(self):
        print("初始化环境...")
        self.env = DualArmPickPlaceEnv(max_steps=2000, render_mode=None)

        # Recording state
        self.recording = False
        self.current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'success': False
        }
        self.saved_trajectories = []

        # Control state
        self.control_scale = 0.1
        self.left_gripper = 0.0
        self.right_gripper = 0.0

        print(__doc__)
        print("\n控制命令 (可以组合，如 'ws' 表示同时按w和s):")
        print("\n【左臂控制】")
        print("  w/s - 关节0±  a/d - 关节1±  q/e - 关节2±")
        print("  z/x - 夹爪开/关")
        print("\n【右臂控制】")
        print("  i/k - 关节0±  j/l - 关节1±  u/o - 关节2±")
        print("  n/m - 夹爪开/关")
        print("\n【录制控制】")
        print("  r - 开始/停止录制")
        print("  reset - 重置环境")
        print("  quit - 退出并保存")
        print("  空格/回车 - 无操作步进")
        print("\n控制灵敏度:", self.control_scale)

    def parse_command(self, cmd):
        """解析命令字符串生成动作。"""
        action = np.zeros(16)
        cmd = cmd.lower().strip()

        # Left arm
        if 'w' in cmd:
            action[0] = self.control_scale
        if 's' in cmd:
            action[0] = -self.control_scale
        if 'a' in cmd:
            action[1] = self.control_scale
        if 'd' in cmd:
            action[1] = -self.control_scale
        if 'q' in cmd:
            action[2] = self.control_scale
        if 'e' in cmd:
            action[2] = -self.control_scale

        # Right arm
        if 'i' in cmd:
            action[7] = self.control_scale
        if 'k' in cmd:
            action[7] = -self.control_scale
        if 'j' in cmd:
            action[8] = self.control_scale
        if 'l' in cmd:
            action[8] = -self.control_scale
        if 'u' in cmd:
            action[9] = self.control_scale
        if 'o' in cmd:
            action[9] = -self.control_scale

        # Grippers
        if 'z' in cmd:
            self.left_gripper = 0.0
        if 'x' in cmd:
            self.left_gripper = 1.0
        if 'n' in cmd:
            self.right_gripper = 0.0
        if 'm' in cmd:
            self.right_gripper = 1.0

        action[14] = self.left_gripper
        action[15] = self.right_gripper

        return action

    def toggle_recording(self):
        """切换录制状态。"""
        if not self.recording:
            self.recording = True
            self.current_trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'success': False
            }
            print("\n🔴 开始录制...")
        else:
            self.recording = False

            if len(self.current_trajectory['observations']) > 0:
                self.current_trajectory['observations'] = np.array(
                    self.current_trajectory['observations'])
                self.current_trajectory['actions'] = np.array(
                    self.current_trajectory['actions'])
                self.current_trajectory['rewards'] = np.array(
                    self.current_trajectory['rewards'])
                self.current_trajectory['episode_length'] = len(
                    self.current_trajectory['rewards'])

                self.saved_trajectories.append(self.current_trajectory)
                print(f"✅ 录制停止！保存轨迹 (长度: {self.current_trajectory['episode_length']})")
                print(f"   总回报: {self.current_trajectory['rewards'].sum():.1f}")
                print(f"   已保存轨迹数: {len(self.saved_trajectories)}")
            else:
                print("⚠️  录制停止，但轨迹为空，未保存")

    def reset(self):
        """重置环境。"""
        obs, info = self.env.reset()
        self.recording = False
        print("\n♻️  环境已重置")
        print(f"Left object: {obs[20:23]}")
        print(f"Right object: {obs[23:26]}")
        print(f"Left TCP: {obs[14:17]}")
        print(f"Right TCP: {obs[17:20]}")
        return obs

    def run(self):
        """主循环。"""
        obs, info = self.env.reset()
        step = 0

        print(f"\n初始状态:")
        print(f"  Left object: {obs[20:23]}")
        print(f"  Right object: {obs[23:26]}")
        print(f"  Left TCP: {obs[14:17]}")
        print(f"  Right TCP: {obs[17:20]}")

        while True:
            # Show status
            print(f"\n--- Step {step} ---")
            print(f"Recording: {'🔴' if self.recording else '⚪'}")
            print(f"Left gripper: {'关闭' if self.left_gripper > 0.5 else '打开'}")
            print(f"Right gripper: {'关闭' if self.right_gripper > 0.5 else '打开'}")

            # Get command
            try:
                cmd = input("命令 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n退出...")
                self.save_and_exit()
                break

            # Handle special commands
            if cmd == 'quit' or cmd == 'q':
                self.save_and_exit()
                break
            elif cmd == 'reset':
                obs = self.reset()
                step = 0
                continue
            elif cmd == 'r':
                self.toggle_recording()
                continue
            elif cmd == '' or cmd.isspace():
                # No-op step
                action = np.zeros(16)
                action[14] = self.left_gripper
                action[15] = self.right_gripper
            else:
                # Parse command to action
                action = self.parse_command(cmd)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            step += 1

            # Record if recording
            if self.recording:
                self.current_trajectory['observations'].append(obs)
                self.current_trajectory['actions'].append(action)
                self.current_trajectory['rewards'].append(reward)

                if info.get('episode_success', False):
                    self.current_trajectory['success'] = True

            # Display feedback
            left_dist = info.get('left_distance', 0)
            right_dist = info.get('right_distance', 0)
            print(f"  L_dist={left_dist:.3f}, R_dist={right_dist:.3f}, reward={reward:.1f}")

            # Handle episode end
            if terminated or truncated:
                if terminated:
                    print(f"\n✨ Episode成功！")
                else:
                    print(f"\n⏱️  Episode超时")

                obs = self.reset()
                step = 0
            else:
                obs = next_obs

    def save_and_exit(self):
        """保存所有轨迹并退出。"""
        if len(self.saved_trajectories) > 0:
            save_dir = Path("demonstrations/dual_arm_teleop")
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"teleop_demos_{timestamp}.pkl"

            with open(save_path, 'wb') as f:
                pickle.dump(self.saved_trajectories, f)

            print(f"\n💾 保存了 {len(self.saved_trajectories)} 条轨迹")
            print(f"   文件: {save_path}")

            # Print statistics
            lengths = [t['episode_length'] for t in self.saved_trajectories]
            rewards = [t['rewards'].sum() for t in self.saved_trajectories]
            successes = [t['success'] for t in self.saved_trajectories]

            print(f"\n统计:")
            print(f"  平均长度: {np.mean(lengths):.1f} 步")
            print(f"  平均回报: {np.mean(rewards):.1f}")
            print(f"  成功率: {np.mean(successes)*100:.1f}%")
        else:
            print("\n没有保存任何轨迹")

        self.env.close()
        print("再见！")


if __name__ == "__main__":
    collector = SimpleTeleopCollector()
    collector.run()
