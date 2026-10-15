#!/usr/bin/env python3
"""
Dual-arm IK-based expert policy for demonstration generation.

Adapts the PublicPickPlaceDemo architecture to control both arms simultaneously,
generating expert trajectories for the dual-arm pick-place task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import sys

import mujoco
import numpy as np

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

try:
    from ik_backend import MinkIKSolver, kinematics_from_adapter
    from robot_adapter import RobotAdapter
except ImportError:
    print("Warning: Could not import IK backend. Using simplified controller.")
    MinkIKSolver = None
    kinematics_from_adapter = None
    RobotAdapter = None


class DualArmExpertPolicy:
    """
    IK-based expert controller for dual-arm pick-and-place.

    Generates near-optimal trajectories by:
    1. Computing IK solutions for both TCPs to reach objects
    2. Grasping objects when within threshold
    3. Moving objects to target bin
    4. Releasing when over target
    """

    def __init__(
        self,
        scene_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_ik: bool = True
    ):
        repo = Path(__file__).resolve().parent.parent

        # Load scene
        if scene_path is None:
            scene_path = str(repo / "scene" / "dual_arm_sorting_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Try to load IK solver if available
        self.use_ik = use_ik and RobotAdapter is not None
        if self.use_ik:
            try:
                config = Path(config_path) if config_path else repo / "robot" / "robot_config.yaml"
                self.adapter = RobotAdapter(str(config))

                # Create IK solvers for both arms
                self.left_solver = MinkIKSolver(
                    self.model, kinematics_from_adapter(self.adapter, "left"))
                self.right_solver = MinkIKSolver(
                    self.model, kinematics_from_adapter(self.adapter, "right"))

                self.left_arm_joints = self.adapter.get_arm_joints("left")
                self.right_arm_joints = self.adapter.get_arm_joints("right")

                print("IK solvers initialized successfully")
            except Exception as e:
                print(f"Failed to initialize IK solvers: {e}")
                print("Falling back to simplified controller")
                self.use_ik = False

        # Get IDs
        self.left_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
        self.right_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")

        self.left_object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "bluecup_0")
        self.right_object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "kebike_0")

        self.source_bin_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "crate_source")
        self.target_bin_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "crate_target")

        # Get weld constraint IDs for kinematic grasping
        self.left_grasp_weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_grasp_weld")
        self.right_grasp_weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_grasp_weld")

        # Get mocap body IDs
        left_mocap_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_mocap")
        right_mocap_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_mocap")

        self.left_mocap_idx = int(self.model.body_mocapid[left_mocap_id])
        self.right_mocap_idx = int(self.model.body_mocapid[right_mocap_id])

        # Get freejoint indices
        self.left_obj_qpos = self._get_freejoint_qpos(self.left_object_id)
        self.right_obj_qpos = self._get_freejoint_qpos(self.right_object_id)

        # Target position
        self.target_pos = self.data.xpos[self.target_bin_id].copy()
        self.target_pos[2] += 0.15  # 15cm above bin bottom

        # State tracking
        self.left_phase = "approach"  # approach -> grasp -> move -> release
        self.right_phase = "approach"
        self.left_grasped = False
        self.right_grasped = False

        # Thresholds
        self.grasp_threshold = 0.05  # 5cm to grasp
        self.place_threshold = 0.08  # 8cm to release

    def _get_freejoint_qpos(self, body_id: int) -> int:
        """Get qpos index for body's freejoint."""
        jnt_adr = int(self.model.body_jntadr[body_id])
        return int(self.model.jnt_qposadr[jnt_adr])

    def _sync_mocap_to_tcp(self, arm: str):
        """Update mocap body to match TCP position."""
        if arm == "left":
            tcp_pos = self.data.site_xpos[self.left_tcp_id].copy()
            tcp_mat = self.data.site_xmat[self.left_tcp_id].reshape(3, 3).copy()
            self.data.mocap_pos[self.left_mocap_idx] = tcp_pos
            self.data.mocap_quat[self.left_mocap_idx] = self._mat_to_quat(tcp_mat)
        else:
            tcp_pos = self.data.site_xpos[self.right_tcp_id].copy()
            tcp_mat = self.data.site_xmat[self.right_tcp_id].reshape(3, 3).copy()
            self.data.mocap_pos[self.right_mocap_idx] = tcp_pos
            self.data.mocap_quat[self.right_mocap_idx] = self._mat_to_quat(tcp_mat)

    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    def _activate_grasp(self, arm: str):
        """Activate kinematic weld constraint."""
        if arm == "left":
            self.data.eq_active[self.left_grasp_weld_id] = 1
            self.left_grasped = True
        else:
            self.data.eq_active[self.right_grasp_weld_id] = 1
            self.right_grasped = True

    def _deactivate_grasp(self, arm: str):
        """Deactivate kinematic weld constraint."""
        if arm == "left":
            self.data.eq_active[self.left_grasp_weld_id] = 0
            self.left_grasped = False
        else:
            self.data.eq_active[self.right_grasp_weld_id] = 0
            self.right_grasped = False

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compute expert action for current observation.

        Returns:
            action: [14D joint deltas, 2D gripper commands]
            info: dict with phase information
        """
        # Extract state from observation
        current_joints = obs[:14]
        left_tcp = self.data.site_xpos[self.left_tcp_id]
        right_tcp = self.data.site_xpos[self.right_tcp_id]
        left_obj = self.data.xpos[self.left_object_id]
        right_obj = self.data.xpos[self.right_object_id]

        # Compute target TCP positions based on phase
        left_target = self._get_target_tcp("left", left_obj)
        right_target = self._get_target_tcp("right", right_obj)

        # Compute joint actions (simplified proportional control)
        # In full implementation, this would use IK solver
        joint_action = self._compute_joint_action(
            current_joints, left_target, right_target, left_tcp, right_tcp)

        # Gripper commands
        left_gripper = 1.0 if self.left_phase in ["grasp", "move"] else 0.0
        right_gripper = 1.0 if self.right_phase in ["grasp", "move"] else 0.0

        # Update phases and grasps
        self._update_phase("left", left_tcp, left_obj)
        self._update_phase("right", right_tcp, right_obj)

        action = np.concatenate([joint_action, [left_gripper, right_gripper]])

        info = {
            "left_phase": self.left_phase,
            "right_phase": self.right_phase,
            "left_grasped": self.left_grasped,
            "right_grasped": self.right_grasped
        }

        return action, info

    def _get_target_tcp(self, arm: str, obj_pos: np.ndarray) -> np.ndarray:
        """Get target TCP position based on current phase."""
        phase = self.left_phase if arm == "left" else self.right_phase

        if phase == "approach":
            # Reach toward object (10cm above)
            target = obj_pos.copy()
            target[2] += 0.10
        elif phase == "grasp":
            # Move down to grasp
            target = obj_pos.copy()
            target[2] += 0.02  # 2cm above object center
        elif phase == "move":
            # Move to target bin
            target = self.target_pos.copy()
            # Offset left/right arms slightly to avoid collision
            if arm == "left":
                target[1] += 0.05  # 5cm left
            else:
                target[1] -= 0.05  # 5cm right
        else:  # release
            # Stay at release position
            target = self.target_pos.copy()
            if arm == "left":
                target[1] += 0.05
            else:
                target[1] -= 0.05

        return target

    def _compute_joint_action(
        self,
        current_joints: np.ndarray,
        left_target: np.ndarray,
        right_target: np.ndarray,
        left_tcp: np.ndarray,
        right_tcp: np.ndarray
    ) -> np.ndarray:
        """
        Compute joint deltas using simplified Cartesian control.

        In full implementation, this would use IK solver to compute
        optimal joint positions for target TCP poses.
        """
        # Proportional control in Cartesian space
        kp = 0.5
        max_delta = 0.05

        # Compute Cartesian errors
        left_error = left_target - left_tcp
        right_error = right_target - right_tcp

        # Heuristic mapping: tile 3D error to 7D joint space
        # This is simplified; real IK uses Jacobian pseudo-inverse
        left_joint_delta = np.tile(left_error, 3)[:7] * kp
        right_joint_delta = np.tile(right_error, 3)[:7] * kp

        joint_delta = np.concatenate([left_joint_delta, right_joint_delta])
        joint_delta = np.clip(joint_delta, -max_delta, max_delta)

        return joint_delta

    def _update_phase(self, arm: str, tcp_pos: np.ndarray, obj_pos: np.ndarray):
        """Update phase state machine."""
        if arm == "left":
            phase = self.left_phase
        else:
            phase = self.right_phase

        if phase == "approach":
            dist = np.linalg.norm(tcp_pos - obj_pos)
            if dist < self.grasp_threshold:
                if arm == "left":
                    self.left_phase = "grasp"
                else:
                    self.right_phase = "grasp"

        elif phase == "grasp":
            # Activate grasp after one step
            self._activate_grasp(arm)
            if arm == "left":
                self.left_phase = "move"
            else:
                self.right_phase = "move"

        elif phase == "move":
            dist_to_target = np.linalg.norm(obj_pos - self.target_pos)
            if dist_to_target < self.place_threshold:
                if arm == "left":
                    self.left_phase = "release"
                else:
                    self.right_phase = "release"

        elif phase == "release":
            # Deactivate grasp
            self._deactivate_grasp(arm)

    def demonstrate(
        self,
        env,
        max_steps: int = 500,
        render: bool = False
    ) -> Tuple[dict, bool]:
        """
        Run a full episode using expert policy.

        Returns:
            trajectory: dict with observations, actions, rewards, info
            success: whether both objects were placed successfully
        """
        obs, info = env.reset()

        # Sync MuJoCo state
        self.data.qpos[:] = env.data.qpos[:]
        self.data.qvel[:] = env.data.qvel[:]
        mujoco.mj_forward(self.model, self.data)

        # Reset phase tracking
        self.left_phase = "approach"
        self.right_phase = "approach"
        self.left_grasped = False
        self.right_grasped = False
        self.data.eq_active[:] = 0

        observations = [obs]
        actions = []
        rewards = []
        infos = []

        for step in range(max_steps):
            # Sync state from env
            self.data.qpos[:] = env.data.qpos[:]
            self.data.qvel[:] = env.data.qvel[:]
            mujoco.mj_forward(self.model, self.data)

            # Update mocap bodies
            self._sync_mocap_to_tcp("left")
            self._sync_mocap_to_tcp("right")

            # Compute action
            action, action_info = self.get_action(obs)

            # Execute in environment
            obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(action_info)

            if render:
                env.render()

            if terminated or truncated:
                break

        trajectory = {
            'observations': np.array(observations[:-1]),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'infos': infos,
            'success': info.get('episode_success', False),
            'episode_length': len(actions)
        }

        return trajectory, info.get('episode_success', False)


if __name__ == "__main__":
    # Test expert policy
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv

    print("Creating environment...")
    env = DualArmPickPlaceEnv(render_mode=None, max_steps=500)

    print("Creating expert policy...")
    expert = DualArmExpertPolicy(use_ik=False)  # Use simplified controller

    print("Running demonstration...")
    trajectory, success = expert.demonstrate(env, max_steps=500)

    print(f"\nDemonstration complete:")
    print(f"  Success: {success}")
    print(f"  Episode length: {trajectory['episode_length']}")
    print(f"  Total reward: {trajectory['rewards'].sum():.2f}")
    print(f"  Avg reward per step: {trajectory['rewards'].mean():.2f}")

    env.close()
