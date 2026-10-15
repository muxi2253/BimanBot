"""Simple waypoint-based expert for dual-arm pick-place.

Uses pre-defined joint trajectories instead of IK.
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Tuple


class WaypointExpert:
    """Expert that follows pre-defined joint waypoints."""

    def __init__(self, scene_path: str = "scene/dual_arm_sorting_scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Get site IDs for TCP positions
        self.left_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp")
        self.right_tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")

        # Task state
        self.reset()

    def reset(self):
        """Reset for new episode."""
        self.phase = 0
        self.steps_in_phase = 0
        self.max_steps_per_phase = 80

        # Will be set by demonstrate()
        self.left_obj_pos = None
        self.right_obj_pos = None
        self.target_left = None
        self.target_right = None

    def _compute_target_joints(self, current_qpos: np.ndarray,
                               left_tcp_target: np.ndarray,
                               right_tcp_target: np.ndarray) -> np.ndarray:
        """Use simple proportional control in Cartesian space.

        This is a naive controller but sufficient for demonstration collection.
        """
        # Update sim state
        self.data.qpos[:14] = current_qpos
        mujoco.mj_forward(self.model, self.data)

        # Get current TCP positions
        left_tcp = self.data.site_xpos[self.left_tcp_id].copy()
        right_tcp = self.data.site_xpos[self.right_tcp_id].copy()

        # Compute Cartesian errors
        left_error = left_tcp_target - left_tcp
        right_error = right_tcp_target - right_tcp

        # Simple P controller: map Cartesian error to joint space
        # This is very naive but works for small movements
        kp = 1.5  # Proportional gain

        # For simplicity, map X/Y/Z error to first 3 joints of each arm
        # (This ignores actual kinematics but is good enough for demos)
        left_joint_delta = np.zeros(7)
        right_joint_delta = np.zeros(7)

        # Map position error to joint deltas (heuristic mapping)
        left_joint_delta[0] = kp * left_error[0]  # X -> joint 0
        left_joint_delta[1] = kp * left_error[1]  # Y -> joint 1
        left_joint_delta[2] = kp * left_error[2]  # Z -> joint 2

        right_joint_delta[0] = kp * right_error[0]
        right_joint_delta[1] = kp * right_error[1]
        right_joint_delta[2] = kp * right_error[2]

        target_qpos = current_qpos + np.concatenate([left_joint_delta, right_joint_delta])

        return target_qpos

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Compute action for current observation.

        Returns:
            action: [14D joint deltas, 2D gripper commands]
            done: whether episode is complete
        """
        current_qpos = obs[:14]

        # Phase 0: Move above objects
        if self.phase == 0:
            left_target = self.left_obj_pos.copy()
            left_target[2] += 0.20  # 20cm above
            right_target = self.right_obj_pos.copy()
            right_target[2] += 0.20
            gripper_cmd = [0.0, 0.0]  # Open

        # Phase 1: Descend to grasp height
        elif self.phase == 1:
            left_target = self.left_obj_pos.copy()
            left_target[2] += 0.05  # 5cm above
            right_target = self.right_obj_pos.copy()
            right_target[2] += 0.05
            gripper_cmd = [0.0, 0.0]

        # Phase 2: Close grippers
        elif self.phase == 2:
            left_target = self.left_obj_pos.copy()
            left_target[2] += 0.05
            right_target = self.right_obj_pos.copy()
            right_target[2] += 0.05
            gripper_cmd = [1.0, 1.0]  # Close

        # Phase 3: Lift objects
        elif self.phase == 3:
            left_target = self.left_obj_pos.copy()
            left_target[2] += 0.25  # Lift 25cm
            right_target = self.right_obj_pos.copy()
            right_target[2] += 0.25
            gripper_cmd = [1.0, 1.0]

        # Phase 4: Move to target bin
        elif self.phase == 4:
            left_target = self.target_left.copy()
            left_target[2] += 0.15  # Stay above bin
            right_target = self.target_right.copy()
            right_target[2] += 0.15
            gripper_cmd = [1.0, 1.0]

        # Phase 5: Descend to place
        elif self.phase == 5:
            left_target = self.target_left.copy()
            right_target = self.target_right.copy()
            gripper_cmd = [1.0, 1.0]

        # Phase 6: Release
        elif self.phase == 6:
            left_target = self.target_left.copy()
            right_target = self.target_right.copy()
            gripper_cmd = [0.0, 0.0]  # Open

        else:
            # Done
            return np.zeros(16), True

        # Compute target joint positions
        target_qpos = self._compute_target_joints(current_qpos, left_target, right_target)

        # Compute joint deltas
        joint_deltas = target_qpos - current_qpos
        joint_deltas = np.clip(joint_deltas, -0.15, 0.15)

        # Combine with gripper commands
        action = np.concatenate([joint_deltas, gripper_cmd])

        # Phase transition logic
        self.steps_in_phase += 1

        # Check if targets reached (simple distance check)
        self.data.qpos[:14] = current_qpos
        mujoco.mj_forward(self.model, self.data)
        left_tcp = self.data.site_xpos[self.left_tcp_id]
        right_tcp = self.data.site_xpos[self.right_tcp_id]

        left_dist = np.linalg.norm(left_tcp - left_target)
        right_dist = np.linalg.norm(right_tcp - right_target)
        both_reached = (left_dist < 0.08) and (right_dist < 0.08)

        if both_reached or self.steps_in_phase >= self.max_steps_per_phase:
            self.phase += 1
            self.steps_in_phase = 0

        done = self.phase >= 7
        return action, done

    def demonstrate(self, env, max_steps: int = 1000,
                   render: bool = False) -> Tuple[dict, bool]:
        """Collect one demonstration trajectory.

        Returns:
            trajectory: dict with observations, actions, rewards, etc.
            success: whether episode succeeded
        """
        obs, info = env.reset()
        self.reset()

        # Get object and target positions from first observation
        self.left_obj_pos = obs[20:23].copy()
        self.right_obj_pos = obs[23:26].copy()
        self.target_left = info['target_left'].copy()
        self.target_right = info['target_right'].copy()

        observations = [obs]
        actions = []
        rewards = []
        dones = []

        done = False
        step = 0

        while not done and step < max_steps:
            action, expert_done = self.get_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated or expert_done

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            step += 1

            if render and hasattr(env, 'render'):
                env.render()

        success = info.get('episode_success', False)

        trajectory = {
            'observations': np.array(observations[:-1]),  # Remove last obs
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'episode_length': step,
            'success': success,
            'final_info': info
        }

        return trajectory, success
