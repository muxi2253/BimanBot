"""Simple scripted policy for dual-arm pick-place demonstrations.

Uses hard-coded joint-space trajectories instead of IK.
This is specifically designed to generate demonstrations for imitation learning.
"""

import numpy as np
from typing import Optional, Tuple


class DualArmScriptedPolicy:
    """Scripted policy using pre-recorded joint trajectories."""

    def __init__(self):
        # Pre-computed joint configurations for key poses
        # Format: [7 left joints, 7 right joints]

        # Rest pose (arms apart, grippers open)
        self.rest_pose = np.array([
            -0.5, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0,  # Left arm
            0.5, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0,   # Right arm
        ])

        # Reach pose (arms extended forward, above source bin)
        self.reach_pose = np.array([
            0.0, -0.8, 0.0, -1.8, 0.0, 1.2, 0.0,  # Left arm
            0.0, -0.8, 0.0, -1.8, 0.0, 1.2, 0.0,  # Right arm
        ])

        # Grasp pose (arms lowered to grasp objects)
        self.grasp_pose = np.array([
            0.0, -1.2, 0.0, -2.0, 0.0, 0.8, 0.0,  # Left arm
            0.0, -1.2, 0.0, -2.0, 0.0, 0.8, 0.0,  # Right arm
        ])

        # Lift pose (arms lifted with objects)
        self.lift_pose = np.array([
            0.0, -0.6, 0.0, -1.6, 0.0, 1.4, 0.0,  # Left arm
            0.0, -0.6, 0.0, -1.6, 0.0, 1.4, 0.0,  # Right arm
        ])

        # Place pose (arms extended to target bin)
        self.place_pose = np.array([
            0.3, -0.9, 0.0, -1.9, 0.0, 1.3, 0.0,  # Left arm
            0.3, -0.9, 0.0, -1.9, 0.0, 1.3, 0.0,  # Right arm
        ])

        # Gripper states
        self.gripper_open = 0.0
        self.gripper_closed = 1.0

        # Phase state
        self.phase = 0
        self.phase_step = 0
        self.max_phase_steps = 50

        # Phase sequence:
        # 0: Move to reach pose (above objects)
        # 1: Move to grasp pose (lower to objects)
        # 2: Close grippers
        # 3: Move to lift pose
        # 4: Move to place pose
        # 5: Open grippers
        # 6: Return to rest pose

    def reset(self, seed: Optional[int] = None):
        """Reset policy state."""
        self.phase = 0
        self.phase_step = 0
        if seed is not None:
            np.random.seed(seed)

    def get_action(self, obs: np.ndarray, target_left: np.ndarray,
                   target_right: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Generate scripted action for current observation.

        Args:
            obs: Current observation [14 joints + 6 TCP + 6 obj + 2 grasp]
            target_left: Final target position for left object (not used in scripted policy)
            target_right: Final target position for right object (not used)

        Returns:
            action: [14 joint deltas, 2 gripper commands]
            done: Whether episode is complete
        """
        current_qpos = obs[:14]

        # Select target pose based on phase
        if self.phase == 0:
            target_qpos = self.reach_pose
            gripper_left = self.gripper_open
            gripper_right = self.gripper_open
        elif self.phase == 1:
            target_qpos = self.grasp_pose
            gripper_left = self.gripper_open
            gripper_right = self.gripper_open
        elif self.phase == 2:
            target_qpos = current_qpos  # Hold position
            gripper_left = self.gripper_closed
            gripper_right = self.gripper_closed
        elif self.phase == 3:
            target_qpos = self.lift_pose
            gripper_left = self.gripper_closed
            gripper_right = self.gripper_closed
        elif self.phase == 4:
            target_qpos = self.place_pose
            gripper_left = self.gripper_closed
            gripper_right = self.gripper_closed
        elif self.phase == 5:
            target_qpos = current_qpos  # Hold position
            gripper_left = self.gripper_open
            gripper_right = self.gripper_open
        elif self.phase == 6:
            target_qpos = self.rest_pose
            gripper_left = self.gripper_open
            gripper_right = self.gripper_open
        else:
            return np.zeros(16), True

        # Compute joint deltas with proportional control
        joint_deltas = (target_qpos - current_qpos) * 0.3  # Smooth movement
        joint_deltas = np.clip(joint_deltas, -0.08, 0.08)  # Match env limits

        # Assemble action
        action = np.concatenate([joint_deltas, [gripper_left, gripper_right]])

        # Phase transition logic
        self.phase_step += 1
        error = np.linalg.norm(target_qpos - current_qpos)

        # Move to next phase when close enough or timeout
        if error < 0.1 or self.phase_step >= self.max_phase_steps:
            if self.phase_step >= self.max_phase_steps:
                print(f"Phase {self.phase} timeout (error={error:.3f})")
            else:
                print(f"Phase {self.phase} complete (error={error:.3f})")
            self.phase += 1
            self.phase_step = 0

        done = self.phase >= 7
        return action, done

    def demonstrate(self, env, max_steps: int = 500, render: bool = False):
        """Collect one demonstration trajectory.

        Args:
            env: Gym environment
            max_steps: Maximum steps per episode
            render: Whether to render

        Returns:
            trajectory: Dictionary with observations, actions, rewards, etc.
            success: Whether the episode succeeded
        """
        obs, info = env.reset()
        self.reset()

        target_left = info['target_left']
        target_right = info['target_right']

        observations = []
        actions = []
        rewards = []

        done = False
        step = 0

        while not done and step < max_steps:
            action, policy_done = self.get_action(obs, target_left, target_right)

            observations.append(obs)
            actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            done = terminated or truncated or policy_done
            step += 1

            if render and hasattr(env, 'render'):
                env.render()

        success = info.get('episode_success', False)

        trajectory = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'episode_length': step,
            'success': success
        }

        return trajectory, success
