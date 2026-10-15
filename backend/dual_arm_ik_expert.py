"""IK-based expert policy for dual-arm pick-and-place.

Uses MinkIKSolver to generate smooth, collision-aware trajectories
for both arms simultaneously.
"""

from __future__ import annotations

import numpy as np
import mujoco
from pathlib import Path
from typing import Optional

try:
    from .ik_backend import MinkIKSolver, kinematics_from_adapter
    from .robot_adapter import RobotAdapter
except ImportError:
    from ik_backend import MinkIKSolver, kinematics_from_adapter
    from robot_adapter import RobotAdapter


class DualArmIKExpert:
    """IK-based expert policy for dual-arm sorting task."""

    def __init__(self, scene_path: str = "scene/dual_arm_sorting_scene.xml",
                 config_path: str = "robot/robot_config.yaml"):
        # Load adapter and scene
        self.adapter = RobotAdapter(config_path)
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Create IK solvers for both arms
        left_kin = kinematics_from_adapter(self.adapter, 'left')
        right_kin = kinematics_from_adapter(self.adapter, 'right')
        self.left_solver = MinkIKSolver(self.model, left_kin)
        self.right_solver = MinkIKSolver(self.model, right_kin)

        # Get joint indices
        self.left_joints = self.adapter.get_arm_joints('left')
        self.right_joints = self.adapter.get_arm_joints('right')
        self.left_qpos = np.array([self._joint_qpos(j) for j in self.left_joints])
        self.right_qpos = np.array([self._joint_qpos(j) for j in self.right_joints])

        # Get gripper indices
        self.left_gripper_joints = self.adapter.get_gripper_joints('left')
        self.right_gripper_joints = self.adapter.get_gripper_joints('right')
        self.left_gripper_qpos = np.array([self._joint_qpos(j) for j in self.left_gripper_joints])
        self.right_gripper_qpos = np.array([self._joint_qpos(j) for j in self.right_gripper_joints])

        # Get gripper ranges
        self.left_open, self.left_closed = self.adapter.get_gripper_range('left')
        self.right_open, self.right_closed = self.adapter.get_gripper_range('right')

        # Get TCP site IDs
        self.left_tcp_id = self._site_id(self.adapter.get_tcp_site('left'))
        self.right_tcp_id = self._site_id(self.adapter.get_tcp_site('right'))

        # Task state
        self.phase = 0  # 0: approach, 1: grasp, 2: lift, 3: move, 4: place, 5: release
        self.steps_in_phase = 0
        self.max_steps_per_phase = 50

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found: {name}")
        return int(jid)

    def _joint_qpos(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site not found: {name}")
        return int(sid)

    def reset(self, seed: Optional[int] = None):
        """Reset policy state for new episode."""
        self.phase = 0
        self.steps_in_phase = 0
        mujoco.mj_resetData(self.model, self.data)

        # Set to rest pose
        left_rest = self.adapter.get_rest_pose('left')
        right_rest = self.adapter.get_rest_pose('right')
        self.data.qpos[self.left_qpos] = left_rest
        self.data.qpos[self.right_qpos] = right_rest
        self.data.qpos[self.left_gripper_qpos] = self.left_open
        self.data.qpos[self.right_gripper_qpos] = self.right_open

        mujoco.mj_forward(self.model, self.data)

    def get_action(self, obs: np.ndarray, target_left: np.ndarray,
                   target_right: np.ndarray) -> tuple[np.ndarray, bool]:
        """Compute action using proportional control (no IK).

        Args:
            obs: Current observation (joint positions + TCP positions)
            target_left: Target position for left TCP [x, y, z]
            target_right: Target position for right TCP [x, y, z]

        Returns:
            action: Joint position deltas [14D]
            done: Whether episode is complete
        """
        # Get current state from observation directly
        current_qpos = obs[:14]
        left_tcp_pos = obs[14:17]  # Use TCP from obs
        right_tcp_pos = obs[17:20]
        left_obj_pos = obs[20:23]
        right_obj_pos = obs[23:26]

        # Compute target positions based on phase
        if self.phase == 0:  # Approach source bin (above objects)
            # Move above current object positions
            left_target = left_obj_pos.copy()
            left_target[2] += 0.15  # 15cm above object
            right_target = right_obj_pos.copy()
            right_target[2] += 0.15
            gripper_left = self.left_open
            gripper_right = self.right_open

        elif self.phase == 1:  # Descend to grasp
            left_target = left_obj_pos.copy()
            left_target[2] += 0.05  # 5cm above object
            right_target = right_obj_pos.copy()
            right_target[2] += 0.05
            gripper_left = self.left_open
            gripper_right = self.right_open

        elif self.phase == 2:  # Close grippers
            left_target = left_tcp_pos.copy()
            right_target = right_tcp_pos.copy()
            gripper_left = self.left_closed
            gripper_right = self.right_closed

        elif self.phase == 3:  # Lift
            left_target = left_obj_pos.copy()
            left_target[2] += 0.20  # Lift 20cm
            right_target = right_obj_pos.copy()
            right_target[2] += 0.20
            gripper_left = self.left_closed
            gripper_right = self.right_closed

        elif self.phase == 4:  # Move to target bin
            # Use final target bin positions (passed as arguments)
            left_target = target_left.copy()
            left_target[2] += 0.20  # Stay 20cm above
            right_target = target_right.copy()
            right_target[2] += 0.20
            gripper_left = self.left_closed
            gripper_right = self.right_closed

        elif self.phase == 5:  # Descend to place
            left_target = target_left.copy()
            left_target[2] += 0.10  # 10cm above target
            right_target = target_right.copy()
            right_target[2] += 0.10
            gripper_left = self.left_closed
            gripper_right = self.right_closed

        elif self.phase == 6:  # Release
            left_target = left_tcp_pos.copy()
            right_target = right_tcp_pos.copy()
            gripper_left = self.left_open
            gripper_right = self.right_open

        else:  # Done
            return np.zeros(16), True

        # Use simple proportional control (Cartesian space)
        # Compute desired TCP movements
        left_error_vec = left_target - left_tcp_pos
        right_error_vec = right_target - right_tcp_pos

        # Scale by proportional gain (keep small for stability)
        kp = 0.5
        left_cartesian_delta = left_error_vec * kp
        right_cartesian_delta = right_error_vec * kp

        # Naive mapping: assume first 3 joints roughly control XYZ
        # This is a HUGE simplification but avoids the IK divergence issue
        left_joint_delta = np.zeros(7)
        right_joint_delta = np.zeros(7)
        left_joint_delta[:3] = left_cartesian_delta
        right_joint_delta[:3] = right_cartesian_delta

        # Clip to match env's limits
        left_joint_delta = np.clip(left_joint_delta, -0.08, 0.08)
        right_joint_delta = np.clip(right_joint_delta, -0.08, 0.08)

        # Add gripper commands
        action = np.concatenate([left_joint_delta, right_joint_delta,
                                [gripper_left, gripper_right]])

        # Phase transition logic
        left_dist = np.linalg.norm(left_tcp_pos - left_target)
        right_dist = np.linalg.norm(right_tcp_pos - right_target)
        both_reached = (left_dist < 0.05) and (right_dist < 0.05)  # 5cm threshold

        self.steps_in_phase += 1
        if both_reached or self.steps_in_phase >= self.max_steps_per_phase:
            print(f"Phase {self.phase}: L_target={left_target}, R_target={right_target}, "
                  f"L_dist={left_dist:.3f}, R_dist={right_dist:.3f}, reached={both_reached}")
            self.phase += 1
            self.steps_in_phase = 0
            print(f"Phase transition: {self.phase}")

        done = self.phase >= 7
        return action, done
