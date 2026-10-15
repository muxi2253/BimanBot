#!/usr/bin/env python3
"""简化版 mocap 拖拽演示收集。

使用说明:
1. 启动后会打开 MuJoCo viewer 窗口
2. 在窗口中右键点击，勾选 Visualization -> mocap
3. 找到粉色/青色半透明小球
4. 双击小球选中它（会高亮）
5. 拖拽小球（不需要按Ctrl）
6. 按 R 开始/停止录制
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
import mujoco
import mujoco.viewer
import pickle
from pathlib import Path
from datetime import datetime


class SimpleMocapCollector:
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

        # Recording state
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

        # Set objects on table
        left_obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_object_joint")
        right_obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")

        left_qpos_adr = int(self.model.jnt_qposadr[left_obj_jnt])
        right_qpos_adr = int(self.model.jnt_qposadr[right_obj_jnt])

        self.data.qpos[left_qpos_adr:left_qpos_adr+3] = [0.85, 0.1, 0.48]
        self.data.qpos[left_qpos_adr+3:left_qpos_adr+7] = [1, 0, 0, 0]

        self.data.qpos[right_qpos_adr:right_qpos_adr+3] = [0.85, -0.1, 0.48]
        self.data.qpos[right_qpos_adr+3:right_qpos_adr+7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Position mocap near TCP
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()

        self.data.mocap_pos[self.left_mocap_id] = left_tcp + [0, 0, 0.15]
        self.data.mocap_pos[self.right_mocap_id] = right_tcp + [0, 0, 0.15]

        mujoco.mj_forward(self.model, self.data)
        print("✓ 场景已重置")

    def toggle_recording(self):
        """切换录制状态。"""
        if not self.recording:
            self.recording = True
            self.current_traj = []
            print("🔴 开始录制！")
        else:
            self.recording = False
            if len(self.current_traj) > 5:
                self.trajectories.append(self.current_traj)
                print(f"✅ 录制停止！保存轨迹长度: {len(self.current_traj)}")
                print(f"   总轨迹数: {len(self.trajectories)}")
            else:
                print("⚠️  轨迹太短，未保存")

    def get_obs(self):
        """获取观测。"""
        qpos = self.data.qpos[:14].copy()
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()
        left_obj = self.data.xpos[self.left_obj_id].copy()
        right_obj = self.data.xpos[self.right_obj_id].copy()
        grasp = np.array([0.0, 0.0])

        return np.concatenate([qpos, left_tcp, right_tcp, left_obj, right_obj, grasp]).astype(np.float32)

    def record_step(self):
        """记录步骤。"""
        if self.recording:
            obs = self.get_obs()
            action = self.data.qvel[:14].copy()
            action = np.concatenate([action, [0.0, 0.0]])
            self.current_traj.append({'observation': obs, 'action': action})

    def save(self):
        """保存演示。"""
        if len(self.trajectories) == 0:
            print("没有录制任何轨迹")
            return

        demonstrations = []
        for traj in self.trajectories:
            obs_list = [step['observation'] for step in traj]
            act_list = [step['action'] for step in traj]

            demo = {
                'observations': np.array(obs_list),
                'actions': np.array(act_list),
                'rewards': np.zeros(len(traj)),
                'episode_length': len(traj),
                'success': False
            }
            demonstrations.append(demo)

        save_dir = Path("demonstrations/dual_arm_kinesthetic")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"kinesthetic_demos_{timestamp}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump(demonstrations, f)

        print(f"\n💾 保存了 {len(demonstrations)} 条轨迹到 {save_path}")

    def run(self):
        """运行。"""
        self.reset_scene()

        print("\n启动 viewer...")
        print("提示:")
        print("  - 双击粉色/青色小球可选中并拖拽")
        print("  - 按 R 开始/停止录制")
        print("  - 按 SPACE 重置场景")
        print("  - 关闭窗口退出并保存")
        print("")

        def key_callback(keycode):
            if keycode == ord('R') or keycode == ord('r'):
                self.toggle_recording()
            elif keycode == 32:  # SPACE
                self.reset_scene()

        try:
            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=key_callback
            ) as viewer:
                while viewer.is_running():
                    mujoco.mj_step(self.model, self.data)
                    self.record_step()
                    viewer.sync()
        except Exception as e:
            print(f"错误: {e}")
            print("请确保使用 mjpython 运行此脚本")

        self.save()


if __name__ == "__main__":
    collector = SimpleMocapCollector()
    collector.run()
