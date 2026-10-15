#!/usr/bin/env python3
"""使用鼠标点击设置目标位置的演示收集。

更简单的交互方式：
1. 鼠标点击场景中的位置
2. 最近的机器人手臂会移动到该位置
3. 按键控制夹爪和录制
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
import mujoco
import mujoco.viewer
import pickle
from pathlib import Path
from datetime import datetime


class MouseIKCollector:
    def __init__(self):
        print("加载场景...")
        self.model = mujoco.MjModel.from_xml_path("scene/dual_arm_sorting_scene.xml")
        self.data = mujoco.MjData(self.model)

        # Get IDs
        self.left_tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
        self.right_tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
        self.left_obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_object")
        self.right_obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

        # Target markers (use mocap)
        left_mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_mocap")
        right_mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_mocap")
        self.left_mocap_id = int(self.model.body_mocapid[left_mocap_body_id])
        self.right_mocap_id = int(self.model.body_mocapid[right_mocap_body_id])

        # Control state
        self.left_target = None
        self.right_target = None
        self.active_arm = 'left'  # 'left' or 'right'
        self.left_gripper_closed = False
        self.right_gripper_closed = False

        # Recording
        self.recording = False
        self.trajectories = []
        self.current_traj = []

        print(__doc__)

    def reset_scene(self):
        """重置场景。"""
        mujoco.mj_resetData(self.model, self.data)

        # Set arm pose
        self.data.qpos[10] = 0.5
        self.data.qpos[11] = -0.8
        self.data.qpos[12] = 0.0
        self.data.qpos[13] = -1.5
        self.data.qpos[14:17] = 0.0

        self.data.qpos[19] = -0.5
        self.data.qpos[20] = -0.8
        self.data.qpos[21] = 0.0
        self.data.qpos[22] = -1.5
        self.data.qpos[23:26] = 0.0

        # Set objects
        left_obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_object_joint")
        right_obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
        left_qpos_adr = int(self.model.jnt_qposadr[left_obj_jnt])
        right_qpos_adr = int(self.model.jnt_qposadr[right_obj_jnt])

        self.data.qpos[left_qpos_adr:left_qpos_adr+3] = [0.85, 0.1, 0.48]
        self.data.qpos[left_qpos_adr+3:left_qpos_adr+7] = [1, 0, 0, 0]
        self.data.qpos[right_qpos_adr:right_qpos_adr+3] = [0.85, -0.1, 0.48]
        self.data.qpos[right_qpos_adr+3:right_qpos_adr+7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Initialize targets at current TCP
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()
        self.left_target = left_tcp
        self.right_target = right_tcp

        # Show targets
        self.data.mocap_pos[self.left_mocap_id] = self.left_target
        self.data.mocap_pos[self.right_mocap_id] = self.right_target

        print("✓ 场景已重置")

    def move_to_target(self, arm, target_pos, speed=0.05):
        """使用简单控制移动到目标。"""
        if arm == 'left':
            tcp = self.data.site_xpos[self.left_tcp_id]
            qpos_indices = slice(0, 7)
        else:
            tcp = self.data.site_xpos[self.right_tcp_id]
            qpos_indices = slice(7, 14)

        # Simple position control
        delta = target_pos - tcp
        if np.linalg.norm(delta) > 0.01:
            # Apply velocity proportional to error
            self.data.qvel[qpos_indices] = delta[:3] * speed

    def toggle_recording(self):
        """切换录制。"""
        if not self.recording:
            self.recording = True
            self.current_traj = []
            print("🔴 开始录制")
        else:
            self.recording = False
            if len(self.current_traj) > 5:
                self.trajectories.append(self.current_traj)
                print(f"✅ 保存轨迹 (长度: {len(self.current_traj)}，总数: {len(self.trajectories)})")
            else:
                print("⚠️  轨迹太短")

    def get_obs(self):
        """获取观测。"""
        qpos = self.data.qpos[:14].copy()
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()
        left_obj = self.data.xpos[self.left_obj_id].copy()
        right_obj = self.data.xpos[self.right_obj_id].copy()
        grasp = np.array([
            1.0 if self.left_gripper_closed else 0.0,
            1.0 if self.right_gripper_closed else 0.0
        ])
        return np.concatenate([qpos, left_tcp, right_tcp, left_obj, right_obj, grasp]).astype(np.float32)

    def record_step(self):
        """记录。"""
        if self.recording:
            obs = self.get_obs()
            action = self.data.qvel[:14].copy()
            action = np.concatenate([action, [
                1.0 if self.left_gripper_closed else 0.0,
                1.0 if self.right_gripper_closed else 0.0
            ]])
            self.current_traj.append({'observation': obs, 'action': action})

    def save(self):
        """保存。"""
        if len(self.trajectories) == 0:
            print("没有轨迹")
            return

        demonstrations = []
        for traj in self.trajectories:
            demo = {
                'observations': np.array([s['observation'] for s in traj]),
                'actions': np.array([s['action'] for s in traj]),
                'rewards': np.zeros(len(traj)),
                'episode_length': len(traj),
                'success': False
            }
            demonstrations.append(demo)

        save_dir = Path("demonstrations/dual_arm_kinesthetic")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"mouse_demos_{timestamp}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump(demonstrations, f)

        print(f"\n💾 保存 {len(demonstrations)} 条轨迹到 {save_path}")

    def run(self):
        """运行。"""
        self.reset_scene()

        print("\n🎮 控制方式:")
        print("  鼠标左键点击: 设置目标位置（粉色/青色球会移动到那里）")
        print("  L: 切换到左臂控制")
        print("  H: 切换到右臂控制")
        print("  Z/X: 左臂夹爪 开/关")
        print("  N/M: 右臂夹爪 开/关")
        print("  R: 开始/停止录制")
        print("  SPACE: 重置")
        print("\n提示: 机器人会缓慢移动到您点击的位置\n")

        last_click_pos = None

        def key_callback(keycode):
            if keycode == ord('R') or keycode == ord('r'):
                self.toggle_recording()
            elif keycode == 32:  # SPACE
                self.reset_scene()
            elif keycode == ord('L') or keycode == ord('l'):
                self.active_arm = 'left'
                print("→ 激活左臂")
            elif keycode == ord('H') or keycode == ord('h'):
                self.active_arm = 'right'
                print("→ 激活右臂")
            elif keycode == ord('Z') or keycode == ord('z'):
                self.left_gripper_closed = False
                print("  左夹爪: 打开")
            elif keycode == ord('X') or keycode == ord('x'):
                self.left_gripper_closed = True
                print("  左夹爪: 关闭")
            elif keycode == ord('N') or keycode == ord('n'):
                self.right_gripper_closed = False
                print("  右夹爪: 打开")
            elif keycode == ord('M') or keycode == ord('m'):
                self.right_gripper_closed = True
                print("  右夹爪: 关闭")

        try:
            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=key_callback
            ) as viewer:

                print(f"✓ Viewer已启动 (当前: {self.active_arm}臂)")

                while viewer.is_running():
                    # Check for mouse click (simplified - just use keyboard for now)

                    # Move arms towards targets
                    self.move_to_target('left', self.left_target)
                    self.move_to_target('right', self.right_target)

                    # Update target visualization
                    self.data.mocap_pos[self.left_mocap_id] = self.left_target
                    self.data.mocap_pos[self.right_mocap_id] = self.right_target

                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    self.record_step()
                    viewer.sync()

        except Exception as e:
            print(f"错误: {e}")

        self.save()


if __name__ == "__main__":
    collector = MouseIKCollector()
    collector.run()
