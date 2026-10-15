#!/usr/bin/env python3
"""带ASCII可视化的遥控演示收集工具。

在终端显示机器人、物体、目标的相对位置，方便无GUI操作。
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
import pickle
from pathlib import Path
from datetime import datetime


class VisualTeleopCollector:
    """带可视化的遥控收集器。"""

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

        print("\n控制说明见下方 ⬇️")

    def draw_scene(self, obs):
        """绘制场景的ASCII俯视图。"""
        # Parse observation
        left_tcp = obs[14:17]
        right_tcp = obs[17:20]
        left_obj = obs[20:23]
        right_obj = obs[23:26]

        # Get target from env
        target_pos = self.env.target_pos

        # Create grid (X-Y plane, Z is height)
        # Scene roughly spans X:[0.5, 1.5], Y:[-0.5, 0.5]
        width = 50
        height = 20
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        def to_grid(pos):
            """Convert world XY to grid coordinates."""
            x = int((pos[0] - 0.5) / 1.0 * width)  # X: 0.5-1.5 -> 0-50
            y = int((pos[1] + 0.5) / 1.0 * height)  # Y: -0.5-0.5 -> 0-20
            return max(0, min(height-1, y)), max(0, min(width-1, x))

        # Draw target bin
        ty, tx = to_grid(target_pos)
        for dy in range(-2, 3):
            for dx in range(-4, 5):
                ny, nx = ty + dy, tx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    grid[ny][nx] = '░'

        # Draw objects
        loy, lox = to_grid(left_obj)
        roy, rox = to_grid(right_obj)
        if 0 <= loy < height and 0 <= lox < width:
            grid[loy][lox] = '●'  # Left object (red)
        if 0 <= roy < height and 0 <= rox < width:
            grid[roy][rox] = '◆'  # Right object (green)

        # Draw TCPs
        lty, ltx = to_grid(left_tcp)
        rty, rtx = to_grid(right_tcp)
        if 0 <= lty < height and 0 <= ltx < width:
            if self.left_gripper > 0.5:
                grid[lty][ltx] = '✖'  # Closed gripper
            else:
                grid[lty][ltx] = '+'  # Open gripper
        if 0 <= rty < height and 0 <= rtx < width:
            if self.right_gripper > 0.5:
                grid[rty][rtx] = '✖'  # Closed gripper
            else:
                grid[rty][rtx] = '+'  # Open gripper

        # Print grid
        print("\n" + "="*52)
        print("   俯视图 (X-Y平面)  近 ↑ 远")
        print("="*52)
        for row in grid:
            print("│" + ''.join(row) + "│")
        print("="*52)

        # Legend
        print("图例: + 左臂(开) ✖ 左臂(闭) ● 左物体 ◆ 右物体 ░ 目标箱")

        # Height info (Z axis)
        print(f"\n高度(Z轴):")
        print(f"  左臂: {left_tcp[2]:.2f}m  左物体: {left_obj[2]:.2f}m")
        print(f"  右臂: {right_tcp[2]:.2f}m  右物体: {right_obj[2]:.2f}m")
        print(f"  目标箱: {target_pos[2]:.2f}m")

        # Distances
        left_dist = np.linalg.norm(left_tcp - left_obj)
        right_dist = np.linalg.norm(right_tcp - right_obj)
        print(f"\n距离:")
        print(f"  左臂→左物体: {left_dist:.3f}m {'✓ 可抓取' if left_dist < 0.08 else ''}")
        print(f"  右臂→右物体: {right_dist:.3f}m {'✓ 可抓取' if right_dist < 0.08 else ''}")

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
            print("  💡 左夹爪: 打开")
        if 'x' in cmd:
            self.left_gripper = 1.0
            print("  💡 左夹爪: 关闭")
        if 'n' in cmd:
            self.right_gripper = 0.0
            print("  💡 右夹爪: 打开")
        if 'm' in cmd:
            self.right_gripper = 1.0
            print("  💡 右夹爪: 关闭")

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
        return obs

    def run(self):
        """主循环。"""
        obs, info = self.env.reset()
        step = 0

        print("\n" + "="*52)
        print("  双臂机器人遥控演示收集 (带可视化)")
        print("="*52)
        print("\n控制命令 (可组合，如 'ws'):")
        print("  左臂: w/s(前后) a/d(左右) q/e(上下) z/x(夹爪开/关)")
        print("  右臂: i/k(前后) j/l(左右) u/o(上下) n/m(夹爪开/关)")
        print("  录制: r(开始/停止) reset(重置) quit(退出)")
        print("  空格/回车: 无操作步进")

        self.draw_scene(obs)

        while True:
            # Show status
            print(f"\n{'━'*52}")
            print(f"Step {step}  录制:{'🔴' if self.recording else '⚪'}  "
                  f"轨迹数:{len(self.saved_trajectories)}")
            print(f"{'━'*52}")

            # Get command
            try:
                cmd = input("命令 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n退出...")
                self.save_and_exit()
                break

            # Handle special commands
            if cmd in ['quit', 'q', 'exit']:
                self.save_and_exit()
                break
            elif cmd == 'reset':
                obs = self.reset()
                step = 0
                self.draw_scene(obs)
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
            print(f"\n  📊 Reward: {reward:.1f}")

            # Update visualization
            self.draw_scene(next_obs)

            # Handle episode end
            if terminated or truncated:
                if terminated:
                    print(f"\n✨ Episode成功！自动保存录制")
                    if self.recording:
                        self.toggle_recording()  # Auto-stop on success
                else:
                    print(f"\n⏱️  Episode超时")

                input("\n按回车继续...")
                obs = self.reset()
                step = 0
                self.draw_scene(obs)
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
    collector = VisualTeleopCollector()
    collector.run()
