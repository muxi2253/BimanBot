#!/usr/bin/env python3
"""鼠标拖拽运动学教学演示收集。

在MuJoCo GUI中直接拖拽机器人末端执行器，记录轨迹。
这是最直观的演示采集方式！
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
import mujoco
import pickle
from pathlib import Path
from datetime import datetime


class KinestheticTeaching:
    """运动学教学收集器（鼠标拖拽）。"""

    def __init__(self, scene_path="scene/dual_arm_sorting_scene.xml"):
        print("加载场景...")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Get mocap body IDs
        left_mocap_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_mocap")
        right_mocap_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_mocap")

        # Get mocap indices (for mocap_pos array)
        self.left_mocap_id = int(self.model.body_mocapid[left_mocap_body_id])
        self.right_mocap_id = int(self.model.body_mocapid[right_mocap_body_id])

        # Get TCP site IDs
        self.left_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
        self.right_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")

        # Get object IDs
        self.left_obj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_object")
        self.right_obj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_object")

        # Recording state
        self.recording = False
        self.trajectories = []
        self.current_traj = []

        print(__doc__)
        print("\n使用说明:")
        print("1. GUI会自动打开")
        print("2. 用鼠标左键拖拽粉色/青色小球（mocap点）")
        print("3. 机器人末端会跟随移动")
        print("4. 按键控制:")
        print("   R - 开始/停止录制")
        print("   SPACE - 重置环境")
        print("   ESC - 退出并保存")
        print("\n提示:")
        print("- 先拖动mocap点到物体上方")
        print("- 在录制模式下操作才会记录")
        print("- 每次按R会保存一条完整轨迹")

    def reset_scene(self):
        """重置场景。"""
        mujoco.mj_resetData(self.model, self.data)

        # Set robot arms to low ready pose (near table height)
        # Left arm joints: qpos[10:17]
        self.data.qpos[10] = 0.5      # joint1 (shoulder pan)
        self.data.qpos[11] = -0.8     # joint2 (shoulder lift) - NEGATIVE = down
        self.data.qpos[12] = 0.0      # joint3 (shoulder roll)
        self.data.qpos[13] = -1.5     # joint4 (elbow)
        self.data.qpos[14] = 0.0      # joint5 (wrist 1)
        self.data.qpos[15] = 0.0      # joint6 (wrist 2)
        self.data.qpos[16] = 0.0      # joint7 (wrist 3)

        # Right arm joints: qpos[19:26]
        self.data.qpos[19] = -0.5     # joint1 (shoulder pan)
        self.data.qpos[20] = -0.8     # joint2 (shoulder lift) - NEGATIVE = down
        self.data.qpos[21] = 0.0      # joint3 (shoulder roll)
        self.data.qpos[22] = -1.5     # joint4 (elbow)
        self.data.qpos[23] = 0.0      # joint5 (wrist 1)
        self.data.qpos[24] = 0.0      # joint6 (wrist 2)
        self.data.qpos[25] = 0.0      # joint7 (wrist 3)

        # Place objects on table (not falling)
        left_pos = np.array([0.85, 0.1, 0.48])
        right_pos = np.array([0.85, -0.1, 0.48])

        # Get freejoint qpos indices
        left_obj_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_object_joint")
        right_obj_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")

        left_qpos_adr = int(self.model.jnt_qposadr[left_obj_jnt])
        right_qpos_adr = int(self.model.jnt_qposadr[right_obj_jnt])

        self.data.qpos[left_qpos_adr:left_qpos_adr+3] = left_pos
        self.data.qpos[left_qpos_adr+3:left_qpos_adr+7] = [1, 0, 0, 0]

        self.data.qpos[right_qpos_adr:right_qpos_adr+3] = right_pos
        self.data.qpos[right_qpos_adr+3:right_qpos_adr+7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Position mocap spheres near robot TCP (not floating above)
        # Get current TCP positions
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()

        # Place mocap slightly above TCP
        self.data.mocap_pos[self.left_mocap_id] = left_tcp + np.array([0, 0, 0.1])
        self.data.mocap_pos[self.right_mocap_id] = right_tcp + np.array([0, 0, 0.1])

        mujoco.mj_forward(self.model, self.data)
        print("♻️  场景已重置")

    def get_observation(self):
        """获取当前观测。"""
        # Joint positions (14D)
        qpos = self.data.qpos[:14].copy()

        # TCP positions (6D)
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()

        # Object positions (6D)
        left_obj = self.data.xpos[self.left_obj_id].copy()
        right_obj = self.data.xpos[self.right_obj_id].copy()

        # Grasp states (2D) - simplified as 0 for now
        grasp = np.array([0.0, 0.0])

        obs = np.concatenate([qpos, left_tcp, right_tcp, left_obj, right_obj, grasp])
        return obs.astype(np.float32)

    def record_step(self):
        """记录当前状态和动作。"""
        if self.recording:
            obs = self.get_observation()
            # Action is the joint velocities
            action = self.data.qvel[:14].copy()
            # Append gripper commands (0 for now)
            action = np.concatenate([action, [0.0, 0.0]])

            self.current_traj.append({
                'observation': obs,
                'action': action,
                'qpos': self.data.qpos[:14].copy()
            })

    def toggle_recording(self):
        """切换录制状态。"""
        if not self.recording:
            self.recording = True
            self.current_traj = []
            print("🔴 开始录制！拖拽mocap点来演示...")
        else:
            self.recording = False
            if len(self.current_traj) > 5:  # At least 5 steps
                self.trajectories.append(self.current_traj)
                print(f"✅ 录制完成！轨迹长度: {len(self.current_traj)}")
                print(f"   已保存轨迹数: {len(self.trajectories)}")
            else:
                print("⚠️  轨迹太短，未保存")

    def save_demonstrations(self):
        """保存所有演示。"""
        if len(self.trajectories) == 0:
            print("没有录制任何轨迹")
            return

        # Convert to standard format
        demonstrations = []
        for traj in self.trajectories:
            obs_list = [step['observation'] for step in traj]
            act_list = [step['action'] for step in traj]

            demo = {
                'observations': np.array(obs_list),
                'actions': np.array(act_list),
                'rewards': np.zeros(len(traj)),  # Dummy rewards
                'episode_length': len(traj),
                'success': False  # User can edit later
            }
            demonstrations.append(demo)

        # Save to file
        save_dir = Path("demonstrations/dual_arm_kinesthetic")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"kinesthetic_demos_{timestamp}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump(demonstrations, f)

        print(f"\n💾 保存了 {len(demonstrations)} 条轨迹")
        print(f"   文件: {save_path}")
        print(f"   平均长度: {np.mean([d['episode_length'] for d in demonstrations]):.1f} 步")

    def run(self):
        """主循环 - 启动viewer并监听键盘。"""
        self.reset_scene()

        print("\n启动MuJoCo viewer...")
        print("寻找粉色/青色小球（mocap）并拖拽...")
        print("\n提示: macOS用户需要用 mjpython 运行:")
        print("  mjpython tools/teleop_kinesthetic.py")

        try:
            # Try importing viewer (requires mjpython on macOS)
            import mujoco.viewer

            def key_callback(keycode):
                """处理键盘输入。"""
                if keycode == ord('R') or keycode == ord('r'):
                    self.toggle_recording()
                elif keycode == 32:  # SPACE
                    self.reset_scene()

            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=key_callback
            ) as viewer:
                while viewer.is_running():
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)

                    # Record if recording
                    self.record_step()

                    # Sync viewer
                    viewer.sync()

        except (AttributeError, ImportError) as e:
            print(f"\n❌ Viewer不可用: {e}")
            print("\nmacOS解决方案:")
            print("  mjpython tools/teleop_kinesthetic.py")
            print("\n或使用其他方案:")
            print("  ./START_TELEOP.sh")
            print("  选择 2 (ASCII可视化)")
            return

        # Save on exit
        self.save_demonstrations()


if __name__ == "__main__":
    teacher = KinestheticTeaching()
    teacher.run()
