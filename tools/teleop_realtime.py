#!/usr/bin/env python3
"""实时键盘遥控演示收集工具。

使用说明：
- 启动后会打开MuJoCo窗口
- 按住键盘控制机器人运动
- R键开始/停止录制
- ESC退出

键位映射：
  左臂:
    W/S: 关节0 (base旋转)
    A/D: 关节1 (shoulder)
    Q/E: 关节2 (elbow)
    1/2: 关节3
    3/4: 关节4
    5/6: 关节5
    7/8: 关节6
    Z/X: 夹爪开/关

  右臂:
    I/K: 关节0 (base旋转)
    J/L: 关节1 (shoulder)
    U/O: 关节2 (elbow)
    9/0: 关节3
    -/=: 关节4
    [/]: 关节5
    ;/': 关节6
    N/M: 夹爪开/关

  其他:
    R: 开始/停止录制
    SPACE: 重置环境
    ESC: 退出并保存
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
import pickle
from pathlib import Path
from datetime import datetime
import mujoco
import mujoco.viewer
import time


class RealtimeTeleopCollector:
    """实时键盘遥控收集器。"""

    def __init__(self):
        # Create environment
        self.env = DualArmPickPlaceEnv(max_steps=2000)

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
        self.control_scale = 0.08
        self.keys_pressed = set()
        self.left_gripper = 0.0
        self.right_gripper = 0.0
        self.should_exit = False

        print(__doc__)
        print(f"\n控制灵敏度: {self.control_scale}")
        print("准备就绪！开始控制机器人...")

    def get_action_from_keys(self):
        """根据当前按下的键生成动作。"""
        action = np.zeros(16)

        # Left arm (joints 0-6)
        if 'W' in self.keys_pressed:
            action[0] = self.control_scale
        if 'S' in self.keys_pressed:
            action[0] = -self.control_scale
        if 'A' in self.keys_pressed:
            action[1] = self.control_scale
        if 'D' in self.keys_pressed:
            action[1] = -self.control_scale
        if 'Q' in self.keys_pressed:
            action[2] = self.control_scale
        if 'E' in self.keys_pressed:
            action[2] = -self.control_scale
        if '1' in self.keys_pressed:
            action[3] = self.control_scale
        if '2' in self.keys_pressed:
            action[3] = -self.control_scale
        if '3' in self.keys_pressed:
            action[4] = self.control_scale
        if '4' in self.keys_pressed:
            action[4] = -self.control_scale
        if '5' in self.keys_pressed:
            action[5] = self.control_scale
        if '6' in self.keys_pressed:
            action[5] = -self.control_scale
        if '7' in self.keys_pressed:
            action[6] = self.control_scale
        if '8' in self.keys_pressed:
            action[6] = -self.control_scale

        # Right arm (joints 7-13)
        if 'I' in self.keys_pressed:
            action[7] = self.control_scale
        if 'K' in self.keys_pressed:
            action[7] = -self.control_scale
        if 'J' in self.keys_pressed:
            action[8] = self.control_scale
        if 'L' in self.keys_pressed:
            action[8] = -self.control_scale
        if 'U' in self.keys_pressed:
            action[9] = self.control_scale
        if 'O' in self.keys_pressed:
            action[9] = -self.control_scale
        if '9' in self.keys_pressed:
            action[10] = self.control_scale
        if '0' in self.keys_pressed:
            action[10] = -self.control_scale
        if '-' in self.keys_pressed:
            action[11] = self.control_scale
        if '=' in self.keys_pressed:
            action[11] = -self.control_scale
        if '[' in self.keys_pressed:
            action[12] = self.control_scale
        if ']' in self.keys_pressed:
            action[12] = -self.control_scale
        if ';' in self.keys_pressed:
            action[13] = self.control_scale
        if "'" in self.keys_pressed:
            action[13] = -self.control_scale

        # Grippers
        action[14] = self.left_gripper
        action[15] = self.right_gripper

        return action

    def handle_key_press(self, key_char):
        """处理按键按下。"""
        key_upper = key_char.upper()

        # Add to pressed keys
        self.keys_pressed.add(key_upper)

        # Special keys
        if key_upper == 'Z':
            self.left_gripper = 0.0
            print("左夹爪: 打开")
        elif key_upper == 'X':
            self.left_gripper = 1.0
            print("左夹爪: 关闭")
        elif key_upper == 'N':
            self.right_gripper = 0.0
            print("右夹爪: 打开")
        elif key_upper == 'M':
            self.right_gripper = 1.0
            print("右夹爪: 关闭")
        elif key_upper == 'R':
            self.toggle_recording()
        elif key_char == ' ':
            self.reset()
        elif ord(key_char) == 27:  # ESC
            self.should_exit = True

    def handle_key_release(self, key_char):
        """处理按键释放。"""
        key_upper = key_char.upper()
        if key_upper in self.keys_pressed:
            self.keys_pressed.discard(key_upper)

    def toggle_recording(self):
        """切换录制状态。"""
        if not self.recording:
            # Start recording
            self.recording = True
            self.current_trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'success': False
            }
            print("\n🔴 开始录制...")
        else:
            # Stop recording
            self.recording = False

            # Check if trajectory is non-empty and has success
            if len(self.current_trajectory['observations']) > 0:
                # Convert to numpy arrays
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
        """主循环 - 使用MuJoCo viewer交互模式。"""
        obs, info = self.env.reset()

        def key_callback(keycode):
            """Viewer键盘回调。"""
            if 32 <= keycode < 127:  # Printable ASCII
                self.handle_key_press(chr(keycode))

        # Launch viewer with key callback
        with mujoco.viewer.launch_passive(
            self.env.model, self.env.data, key_callback=key_callback
        ) as viewer:
            step = 0

            while viewer.is_running() and not self.should_exit:
                # Get action from keyboard
                action = self.get_action_from_keys()

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

                # Sync viewer
                viewer.sync()

                # Display info periodically
                if step % 50 == 0:
                    left_dist = info.get('left_distance', 0)
                    right_dist = info.get('right_distance', 0)
                    print(f"Step {step}: L_dist={left_dist:.3f}, R_dist={right_dist:.3f}, "
                          f"reward={reward:.1f}, recording={'🔴' if self.recording else '⚪'}")

                # Handle episode end
                if terminated or truncated:
                    if terminated:
                        print(f"\n✨ Episode成功！")
                    else:
                        print(f"\n⏱️  Episode超时")

                    obs, info = self.env.reset()
                    step = 0
                else:
                    obs = next_obs

                # Small delay for stability
                time.sleep(0.02)

        # Save when viewer closes
        self.save_and_exit()

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
        sys.exit(0)


if __name__ == "__main__":
    collector = RealtimeTeleopCollector()
    collector.run()
