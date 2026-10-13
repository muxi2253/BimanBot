"""
通用机器人适配层 - 从配置文件加载机器人特性，让 IK 系统与具体机器人解耦

使用方法：
    adapter = RobotAdapter('robot/robot_config.yaml')
    left_joints = adapter.get_arm_joints('left')
    tcp_site = adapter.get_tcp_site('left')
"""

import os

import yaml
import mujoco
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class RobotAdapter:
    """通用机器人适配层"""

    def __init__(self, config_path: str):
        """
        加载机器人配置文件

        Args:
            config_path: 配置文件路径 (YAML)
        """
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Robot config not found: {config_path}")

        self.config_path = config_path
        self.config_dir = config_path.parent
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.robot_type = self.config['robot_type']
        self.kinematics = self.config['kinematics']
        self.grippers = self.config.get('grippers', {})
        self.tool_geometry = self.config.get('tool_geometry', {})
        self.rest_pose = self.config.get('rest_pose', {})
        self.base_config = self.config.get('mobile_base', {})
        self.assets = self.config.get('assets', {})
        self.cameras = self.config.get('cameras', {})

    def get_arm_chain(self, side: str) -> List[str]:
        """
        获取手臂运动学链（从 base 到 TCP 的 body 序列）

        Args:
            side: 'left' or 'right'

        Returns:
            Body 名称列表
        """
        arm_key = f'{side}_arm'
        if arm_key not in self.kinematics:
            raise ValueError(f"Arm '{side}' not defined in config")
        return self.kinematics[arm_key].get('chain', [])

    def get_arm_joints(self, side: str) -> List[str]:
        """
        获取手臂关节名称列表（按顺序）

        Args:
            side: 'left' or 'right'

        Returns:
            Joint 名称列表
        """
        arm_key = f'{side}_arm'
        if arm_key not in self.kinematics:
            raise ValueError(f"Arm '{side}' not defined in config")
        return self.kinematics[arm_key]['joints']

    def get_tcp_site(self, side: str) -> str:
        """
        获取 TCP (Tool Center Point) site 名称

        Args:
            side: 'left' or 'right'

        Returns:
            Site 名称
        """
        arm_key = f'{side}_arm'
        if arm_key not in self.kinematics:
            raise ValueError(f"Arm '{side}' not defined in config")
        return self.kinematics[arm_key]['tcp_site']

    def get_dof(self, side: str) -> int:
        """
        获取手臂自由度

        Args:
            side: 'left' or 'right'

        Returns:
            自由度数量
        """
        arm_key = f'{side}_arm'
        if arm_key not in self.kinematics:
            raise ValueError(f"Arm '{side}' not defined in config")
        return self.kinematics[arm_key]['dof']

    def get_rest_pose(self, side: str) -> np.ndarray:
        """
        获取静止姿态（用作 IK seed）

        Args:
            side: 'left' or 'right'

        Returns:
            关节角数组
        """
        arm_key = f'{side}_arm'
        if arm_key not in self.rest_pose:
            # 返回全零作为默认
            dof = self.get_dof(side)
            return np.zeros(dof)
        return np.array(self.rest_pose[arm_key])

    def get_gripper_joints(self, side: str) -> List[str]:
        """
        获取夹爪关节名称列表

        Args:
            side: 'left' or 'right'

        Returns:
            夹爪关节名称列表
        """
        if side not in self.grippers:
            return []
        return self.grippers[side].get('joints', [])

    def get_gripper_range(self, side: str) -> Tuple[float, float]:
        """
        获取夹爪开合范围

        Args:
            side: 'left' or 'right'

        Returns:
            (open_position, closed_position)
        """
        if side not in self.grippers:
            return (0.0, 0.0)

        grip_cfg = self.grippers[side]
        return grip_cfg.get('open', 0.09), grip_cfg.get('closed', 0.0)

    def get_base_type(self) -> str:
        """
        获取移动底盘类型

        Returns:
            'differential', 'omnidirectional', 'holonomic', 或 'fixed'
        """
        return self.base_config.get('type', 'fixed')

    def get_wheel_joints(self) -> List[str]:
        """
        获取轮子关节名称列表

        Returns:
            轮子关节名称列表
        """
        return self.base_config.get('joints', [])

    def get_base_root_joint(self) -> Optional[str]:
        """Return the optional floating-base joint used by kinematic IK."""
        return self.base_config.get('root_joint')

    def get_asset_path(self, kind: str) -> Optional[str]:
        """Return an absolute asset path resolved from the config directory.

        Asset bundles are relocatable: relative values in the YAML are relative
        to the YAML file itself, while absolute paths remain unchanged.
        Environment variables and ``~`` are expanded for private deployments.
        """
        value = self.assets.get(kind)
        if not value:
            return None
        path = Path(os.path.expandvars(str(value))).expanduser()
        if not path.is_absolute():
            path = self.config_dir / path
        return str(path.resolve())

    def get_camera_mounts(self) -> Dict[str, Dict]:
        """Return normalized camera mounts for MuJoCo camera injection."""
        mounts = {}
        for name, camera in self.cameras.items():
            mounts[name] = {
                'body': camera['parent'],
                'pos': list(camera['pos']),
                'quat': list(camera['quat']),
                'fovy': float(camera.get('fovy', 60.0)),
            }
        return mounts

    def map_to_qpos(
        self,
        model: mujoco.MjModel,
        joint_values: Dict[str, float],
        base_qpos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        将关节值字典映射到完整的 qpos 数组

        Args:
            model: MuJoCo 模型
            joint_values: {joint_name: value} 字典
            base_qpos: 如果提供，作为起始 qpos（否则用零向量）

        Returns:
            完整的 qpos 数组
        """
        if base_qpos is None:
            qpos = np.zeros(model.nq)
        else:
            qpos = base_qpos.copy()

        for joint_name, value in joint_values.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                addr = model.jnt_qposadr[jid]
                qpos[addr] = value

        return qpos

    def extract_arm_qpos(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        side: str
    ) -> np.ndarray:
        """
        从完整 qpos 中提取指定手臂的关节角

        Args:
            model: MuJoCo 模型
            qpos: 完整 qpos 数组
            side: 'left' or 'right'

        Returns:
            手臂关节角数组
        """
        joint_names = self.get_arm_joints(side)
        arm_qpos = []

        for jname in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                addr = model.jnt_qposadr[jid]
                arm_qpos.append(qpos[addr])

        return np.array(arm_qpos)

    def get_actuator_name(self, joint_name: str) -> str:
        """
        根据关节名称推导执行器名称（遵循 pos_/vel_ 命名规范）

        Args:
            joint_name: 关节名称

        Returns:
            执行器名称
        """
        # 默认策略：轮子用 vel_，其他用 pos_
        if 'wheel' in joint_name.lower():
            return f"vel_{joint_name}"
        else:
            return f"pos_{joint_name}"

    def __repr__(self):
        return (f"RobotAdapter(type={self.robot_type}, "
                f"arms={list(self.kinematics.keys())}, "
                f"base={self.get_base_type()})")


def load_default_adapter() -> RobotAdapter:
    """
    加载默认机器人适配器（从标准路径）

    Returns:
        RobotAdapter 实例
    """
    configured = os.environ.get('ROBOT_CONFIG')
    if configured:
        return RobotAdapter(configured)

    # Source-tree default. Installed applications should set ROBOT_CONFIG or
    # construct RobotAdapter with their own asset-bundle configuration.
    candidates = [
        Path(__file__).parent.parent / 'robot' / 'robot_config.yaml'
    ]

    for path in candidates:
        if Path(path).exists():
            return RobotAdapter(str(path))

    raise FileNotFoundError(
        "No robot config found. Please create robot/robot_config.yaml"
    )


if __name__ == '__main__':
    # 测试代码
    try:
        adapter = load_default_adapter()
        print(f"Loaded: {adapter}")
        print(f"Left arm joints: {adapter.get_arm_joints('left')}")
        print(f"Left TCP site: {adapter.get_tcp_site('left')}")
        print(f"Left rest pose: {adapter.get_rest_pose('left')}")
    except FileNotFoundError as e:
        print(f"Config not found: {e}")
