#!/usr/bin/env python3
"""
Expert policy for demonstration generation.

Provides a simple scripted controller that computes near-optimal actions
for the dual-arm reaching task. Used to generate expert demonstrations
for imitation learning / behavioral cloning.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


class ExpertReachingPolicy:
    """
    PD-controller based expert for dual-arm reaching.

    Computes actions by applying proportional control toward target positions.
    This is a kinematic controller that assumes direct joint position control.
    """

    def __init__(self, kp=0.5, max_action=0.1, noise_std=0.0):
        """
        Args:
            kp: Proportional gain (controls how aggressively to move toward target)
            max_action: Maximum joint delta per step (action clipping)
            noise_std: Gaussian noise std to add to actions (for data diversity)
        """
        self.kp = kp
        self.max_action = max_action
        self.noise_std = noise_std

    def get_action(self, obs, env):
        """
        Compute expert action given observation.

        Args:
            obs: Environment observation (21D: joints + TCP + stage)
            env: Environment instance (to access target positions)

        Returns:
            action: 14D joint position deltas
        """
        # Extract current joint positions (first 14 dims)
        current_joints = obs[:14]

        # Get current TCP positions
        left_tcp = env.data.site_xpos[env.left_tcp_id]
        right_tcp = env.data.site_xpos[env.right_tcp_id]

        # Get target positions
        target_left = env.target_left
        target_right = env.target_right

        # Compute position errors in Cartesian space
        left_error = target_left - left_tcp
        right_error = target_right - right_tcp

        # Simple proportional control: move TCP toward target
        # We approximate joint-space action using TCP error scaled by kp
        # This is a heuristic - true IK would be more accurate but slower

        # For simplicity: assume first 7 joints control left arm, last 7 control right
        # Scale Cartesian error to joint space (rough approximation)
        left_action = self.kp * np.tile(left_error, 3)[:7]  # Repeat 3D error to 7D
        right_action = self.kp * np.tile(right_error, 3)[:7]

        action = np.concatenate([left_action, right_action])

        # Clip to max action magnitude
        action = np.clip(action, -self.max_action, self.max_action)

        # Add optional noise for diversity
        if self.noise_std > 0:
            action += np.random.randn(14) * self.noise_std
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def demonstrate(self, env, max_steps=500, render=False):
        """
        Run a full episode using expert policy.

        Returns:
            trajectory: dict with keys ['observations', 'actions', 'rewards', 'dones']
            success: bool indicating if episode succeeded
        """
        obs, info = env.reset()

        observations = [obs]
        actions = []
        rewards = []
        dones = []

        for step in range(max_steps):
            action = self.get_action(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)

            if render:
                env.render()

            if terminated or truncated:
                break

        trajectory = {
            'observations': np.array(observations[:-1]),  # Exclude final obs
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'success': info.get('episode_success', False),
            'episode_length': len(actions)
        }

        return trajectory, info.get('episode_success', False)


class ImprovedExpertPolicy(ExpertReachingPolicy):
    """
    Enhanced expert using pseudo-inverse Jacobian for more accurate control.

    This version computes joint velocities using the robot's Jacobian,
    providing smoother and more efficient trajectories.
    """

    def __init__(self, kp=0.3, max_action=0.08, noise_std=0.0, damping=0.01):
        super().__init__(kp, max_action, noise_std)
        self.damping = damping

    def compute_jacobian(self, env, site_id, joint_ids):
        """Compute Jacobian for a specific TCP site."""
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))

        # Get Jacobian from MuJoCo
        import mujoco
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)

        # Extract columns corresponding to arm joints
        jac = jacp[:, joint_ids]
        return jac

    def get_action(self, obs, env):
        """Compute action using Jacobian pseudo-inverse."""
        # Get TCP positions and targets
        left_tcp = env.data.site_xpos[env.left_tcp_id]
        right_tcp = env.data.site_xpos[env.right_tcp_id]
        target_left = env.target_left
        target_right = env.target_right

        # Compute Cartesian errors
        left_error = self.kp * (target_left - left_tcp)
        right_error = self.kp * (target_right - right_tcp)

        # Get joint IDs for each arm (assuming first 7 are left, last 7 are right)
        left_joint_ids = list(range(7))
        right_joint_ids = list(range(7, 14))

        # Compute Jacobians
        jac_left = self.compute_jacobian(env, env.left_tcp_id, left_joint_ids)
        jac_right = self.compute_jacobian(env, env.right_tcp_id, right_joint_ids)

        # Damped pseudo-inverse solution
        def damped_pinv(J):
            return J.T @ np.linalg.inv(J @ J.T + self.damping * np.eye(3))

        # Compute joint velocities
        left_action = damped_pinv(jac_left) @ left_error
        right_action = damped_pinv(jac_right) @ right_error

        action = np.concatenate([left_action, right_action])

        # Clip and add noise
        action = np.clip(action, -self.max_action, self.max_action)
        if self.noise_std > 0:
            action += np.random.randn(14) * self.noise_std
            action = np.clip(action, -self.max_action, self.max_action)

        return action


if __name__ == "__main__":
    # Quick test of expert policy
    from gym_env_sequential import SequentialReachEnv

    env = SequentialReachEnv(
        render_mode=None,
        max_steps=500,
        reward_type="dense",
        max_stages=1
    )

    print("Testing expert policy...")
    expert = ExpertReachingPolicy(kp=0.3, max_action=0.1)

    trajectory, success = expert.demonstrate(env, max_steps=500)

    print(f"\nEpisode finished:")
    print(f"  Success: {success}")
    print(f"  Episode length: {trajectory['episode_length']}")
    print(f"  Total reward: {trajectory['rewards'].sum():.2f}")
    print(f"  Final observation shape: {trajectory['observations'].shape}")

    env.close()
