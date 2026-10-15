#!/usr/bin/env python3
"""键盘控制目标位置 + 3D可视化。

最实用的方案：
- 3D窗口实时显示机器人姿态
- 键盘控制目标位置（粉色/青色球）
- 机器人会自动移动到目标位置
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
import mujoco
import mujoco.viewer
import pickle
from pathlib import Path
from datetime import datetime


class KeyboardMocapCollector:
    def __init__(self):
        print("加载场景...")
        self.model = mujoco.MjModel.from_xml_path("scene/dual_arm_sorting_scene.xml")
        self.data = mujoco.MjData(self.model)

        # Get IDs
        self.left_tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
        self.right_tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
        self.left_obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_object")
        self.right_obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

        left_mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_mocap")
        right_mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_mocap")
        self.left_mocap_id = int(self.model.body_mocapid[left_mocap_body_id])
        self.right_mocap_id = int(self.model.body_mocapid[right_mocap_body_id])

        # Get equality constraint IDs
        self.left_weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_gripper_weld")
        self.right_weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_gripper_weld")

        # Enable equality constraints (make robot follow mocap)
        # Note: eq_active is in mjData, not mjModel
        print("✓ 找到weld约束，将在仿真中激活")

        # Control
        self.active_arm = 'left'
        self.left_gripper = 0.0
        self.right_gripper = 0.0
        self.move_step = 0.05

        # Recording
        self.recording = False
        self.trajectories = []
        self.current_traj = []

        print(__doc__)
        print("✓ 机器人会跟随mocap球移动")

    def reset_scene(self):
        """重置。"""
        mujoco.mj_resetData(self.model, self.data)

        # Arm pose
        self.data.qpos[10:17] = [0.5, -0.8, 0.0, -1.5, 0.0, 0.0, 0.0]
        self.data.qpos[19:26] = [-0.5, -0.8, 0.0, -1.5, 0.0, 0.0, 0.0]

        # Objects
        left_obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_object_joint")
        right_obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
        left_adr = int(self.model.jnt_qposadr[left_obj_jnt])
        right_adr = int(self.model.jnt_qposadr[right_obj_jnt])

        self.data.qpos[left_adr:left_adr+7] = [0.85, 0.1, 0.48, 1, 0, 0, 0]
        self.data.qpos[right_adr:right_adr+7] = [0.85, -0.1, 0.48, 1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Mocap at TCP
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()
        self.data.mocap_pos[self.left_mocap_id] = left_tcp
        self.data.mocap_pos[self.right_mocap_id] = right_tcp

        # Enable weld constraints (robot will follow mocap)
        self.data.eq_active[self.left_weld_id] = 1
        self.data.eq_active[self.right_weld_id] = 1

        print("✓ 重置 (约束已激活)")

    def move_mocap(self, arm, direction):
        """移动目标位置（mocap球）。"""
        mocap_id = self.left_mocap_id if arm == 'left' else self.right_mocap_id
        current_pos = self.data.mocap_pos[mocap_id].copy()

        if direction == 'forward':
            current_pos[0] += self.move_step
        elif direction == 'backward':
            current_pos[0] -= self.move_step
        elif direction == 'left':
            current_pos[1] += self.move_step
        elif direction == 'right':
            current_pos[1] -= self.move_step
        elif direction == 'up':
            current_pos[2] += self.move_step
        elif direction == 'down':
            current_pos[2] -= self.move_step

        self.data.mocap_pos[mocap_id] = current_pos

    def toggle_recording(self):
        """切换录制。"""
        if not self.recording:
            self.recording = True
            self.current_traj = []
            print("🔴 录制")
        else:
            self.recording = False
            if len(self.current_traj) > 5:
                self.trajectories.append(self.current_traj)
                print(f"✅ 保存 (长度:{len(self.current_traj)}, 总数:{len(self.trajectories)})")
            else:
                print("⚠️  太短")

    def get_obs(self):
        qpos = self.data.qpos[:14].copy()
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()
        left_obj = self.data.xpos[self.left_obj_id].copy()
        right_obj = self.data.xpos[self.right_obj_id].copy()
        grasp = np.array([self.left_gripper, self.right_gripper])
        return np.concatenate([qpos, left_tcp, right_tcp, left_obj, right_obj, grasp]).astype(np.float32)

    def record_step(self):
        if self.recording:
            obs = self.get_obs()
            action = np.concatenate([self.data.qvel[:14].copy(), [self.left_gripper, self.right_gripper]])
            self.current_traj.append({'observation': obs, 'action': action})

    def save(self):
        if len(self.trajectories) == 0:
            print("无轨迹")
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
        save_path = save_dir / f"keyboard_demos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump(demonstrations, f)

        print(f"\n💾 {len(demonstrations)} 条 → {save_path}")

    def run(self):
        self.reset_scene()

        print("\n🎮 键盘控制:")
        print("  [Tab] 切换左/右臂")
        print("  W/S/A/D/Q/E: 移动目标位置（粉色/青色球）")
        print("  Z/X: 当前臂夹爪 开/关")
        print("  R: 录制")
        print("  空格: 重置")
        print("")

        def key_callback(keycode):
            if keycode == 9:  # Tab
                self.active_arm = 'right' if self.active_arm == 'left' else 'left'
                color = '粉色' if self.active_arm == 'left' else '青色'
                print(f"→ 切换到{self.active_arm}臂 ({color}球)")
            elif keycode == ord('W') or keycode == ord('w'):
                self.move_mocap(self.active_arm, 'forward')
            elif keycode == ord('S') or keycode == ord('s'):
                self.move_mocap(self.active_arm, 'backward')
            elif keycode == ord('A') or keycode == ord('a'):
                self.move_mocap(self.active_arm, 'left')
            elif keycode == ord('D') or keycode == ord('d'):
                self.move_mocap(self.active_arm, 'right')
            elif keycode == ord('Q') or keycode == ord('q'):
                self.move_mocap(self.active_arm, 'down')
            elif keycode == ord('E') or keycode == ord('e'):
                self.move_mocap(self.active_arm, 'up')
            elif keycode == ord('Z') or keycode == ord('z'):
                if self.active_arm == 'left':
                    self.left_gripper = 0.0
                    print("  左夹爪: 开")
                else:
                    self.right_gripper = 0.0
                    print("  右夹爪: 开")
            elif keycode == ord('X') or keycode == ord('x'):
                if self.active_arm == 'left':
                    self.left_gripper = 1.0
                    print("  左夹爪: 关")
                else:
                    self.right_gripper = 1.0
                    print("  右夹爪: 关")
            elif keycode == ord('R') or keycode == ord('r'):
                self.toggle_recording()
            elif keycode == 32:
                self.reset_scene()

        try:
            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=key_callback
            ) as viewer:
                print(f"✓ 已启动 (控制{self.active_arm}臂)\n")

                while viewer.is_running():
                    mujoco.mj_step(self.model, self.data)
                    self.record_step()
                    viewer.sync()

        except Exception as e:
            print(f"错误: {e}")

        self.save()


if __name__ == "__main__":
    collector = KeyboardMocapCollector()
    collector.run()
