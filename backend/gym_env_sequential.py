#!/usr/bin/env python3
"""
Sequential Reaching Environment for Dual-Arm Robot

Task: Reach 3 sets of target positions in sequence (6 targets total)
- Both arms must reach current targets to advance to next stage
- Episode succeeds only if all 3 stages completed
- Demonstrates long-horizon task capability

This is an extension of DualArmReachEnv for hierarchical RL research.
"""

import numpy as np
import gymnasium as gym
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))
try:
    from .gym_env_reach import DualArmReachEnv
except ImportError:
    from gym_env_reach import DualArmReachEnv


class SequentialReachEnv(DualArmReachEnv):
    """
    3-stage sequential reaching task.

    Observation: joint pos (14D) + TCP pos (6D) + stage indicator (1D) = 21D
    Action: joint position deltas (14D)
    Reward: Stage completion bonuses + optional distance shaping

    Episode structure:
        Stage 0: Reach targets A & B
        Stage 1: Reach targets C & D
        Stage 2: Reach targets E & F

    Success: Complete all 3 stages within max_steps
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        max_steps=500,  # PLAN A: Reduced to 500 for single-stage rapid feedback
        config_path=None,
        reward_type="sparse",  # "sparse", "shaped", or "dense"
        randomize_targets=False,  # Whether to randomize target positions
        max_stages=1,  # PLAN A: Single stage only for initial learning
    ):
        """
        Args:
            max_steps: Maximum steps per episode (default 1500 for 3 stages)
            reward_type:
                - "sparse": Only stage completion bonuses
                - "shaped": Add distance-based penalties
                - "dense": Continuous distance reward + bonuses
            randomize_targets: If True, randomize target positions each episode
            max_stages: Maximum number of stages (for curriculum learning)
        """
        super().__init__(render_mode=render_mode, max_steps=max_steps, config_path=config_path)

        self.reward_type = reward_type
        self.randomize_targets = randomize_targets
        self.max_stages = max_stages  # For curriculum learning

        # Stage tracking
        self.current_stage = 0  # 0, 1, or 2
        self.stages_completed = 0

        # Define fixed target positions for each stage (reproducible)
        # FIXED: Reduced arm separation to make task learnable
        # Original had 60cm separation - too hard for random exploration
        # New design: 20-30cm separation, more natural reaching poses
        self.stage_targets_base = [
            # Stage 0: Close symmetric reaching (20cm apart)
            ([0.4, 0.10, 1.2], [0.4, -0.10, 1.2]),
            # Stage 1: Medium spread (25cm apart)
            ([0.5, 0.125, 1.3], [0.5, -0.125, 1.1]),
            # Stage 2: Wider spread (30cm apart)
            ([0.35, 0.15, 1.4], [0.35, -0.15, 1.0]),
        ]

        # Expand observation space to include stage indicator
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),  # 14 joints + 6 TCP + 1 stage
            dtype=np.float32
        )

    def _define_stage_targets(self):
        """Define or randomize target positions for current episode."""
        if not self.randomize_targets:
            # Use fixed targets (reproducible)
            self.stage_targets = [
                (np.array(left), np.array(right))
                for left, right in self.stage_targets_base
            ]
        else:
            # Randomize targets within workspace bounds
            self.stage_targets = []
            for _ in range(3):
                left_target = np.array([
                    self.np_random.uniform(0.3, 0.6),
                    self.np_random.uniform(0.1, 0.5),
                    self.np_random.uniform(1.0, 1.5)
                ])
                right_target = np.array([
                    self.np_random.uniform(0.3, 0.6),
                    self.np_random.uniform(-0.5, -0.1),
                    self.np_random.uniform(1.0, 1.5)
                ])
                self.stage_targets.append((left_target, right_target))

    def _get_obs(self):
        """Get observation with stage indicator."""
        # Get base observation (14 joints + 6 TCP positions)
        base_obs = super()._get_obs()

        # Add stage indicator (normalized to [0, 1])
        stage_indicator = np.array([self.current_stage / 2.0], dtype=np.float32)

        obs = np.concatenate([base_obs, stage_indicator])
        return obs

    def _get_info(self):
        """Extended info with stage tracking."""
        info = super()._get_info()

        # Add stage-specific info
        info['current_stage'] = self.current_stage
        info['stages_completed'] = self.stages_completed
        info['stage_success'] = info['success']  # Current stage success
        info['episode_success'] = (self.stages_completed >= min(self.max_stages, 3))

        return info

    def _check_stage_completion(self):
        """Check if current stage targets are reached."""
        left_tcp_pos = self.data.site_xpos[self.left_tcp_id]
        right_tcp_pos = self.data.site_xpos[self.right_tcp_id]

        left_dist = np.linalg.norm(left_tcp_pos - self.target_left)
        right_dist = np.linalg.norm(right_tcp_pos - self.target_right)

        # Both arms must be within threshold
        # PLAN A FIX: Relaxed from 0.10m to 0.20m for exploration feasibility
        return (left_dist < 0.20 and right_dist < 0.20)

    def _advance_stage(self):
        """Move to next stage and update targets."""
        self.current_stage += 1
        self.stages_completed += 1

        if self.current_stage < len(self.stage_targets):
            # Update targets for next stage
            self.target_left, self.target_right = self.stage_targets[self.current_stage]

        # Debug info
        if self.current_stage <= 2:
            print(f"✅ Stage {self.current_stage} reached! "
                  f"Moving to stage {self.current_stage + 1}/3")

    def reset(self, seed=None, options=None):
        """Reset environment and define stage targets."""
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)

        # Reset stage tracking
        self.current_stage = 0
        self.stages_completed = 0

        # Define targets for this episode
        self._define_stage_targets()

        # Set initial targets (stage 0)
        self.target_left, self.target_right = self.stage_targets[0]

        # Return observation with stage indicator
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one step with multi-stage logic."""
        # Execute base step
        obs_base, reward_base, terminated, truncated, info_base = super().step(action)

        # Override reward based on reward_type
        reward = 0.0

        # Get current distances
        left_tcp_pos = self.data.site_xpos[self.left_tcp_id]
        right_tcp_pos = self.data.site_xpos[self.right_tcp_id]
        left_dist = np.linalg.norm(left_tcp_pos - self.target_left)
        right_dist = np.linalg.norm(right_tcp_pos - self.target_right)
        total_dist = left_dist + right_dist

        # Reward shaping based on type
        # FIXED: Increased reward signal strength 10x for better learning
        if self.reward_type == "sparse":
            # Only stage completion gives reward
            reward = 0.0
        elif self.reward_type == "shaped":
            # Distance penalty (10x stronger than before)
            reward = -0.1 * total_dist
        elif self.reward_type == "dense":
            # Strong distance reward (10x stronger)
            reward = -10.0 * total_dist
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # Check stage completion
        stage_completed = self._check_stage_completion()

        if stage_completed:
            # Stage completion bonus
            reward += 10.0

            # Advance to next stage (if not already at max)
            if self.current_stage < min(self.max_stages, 3) - 1:
                self._advance_stage()
            else:
                # All stages completed!
                reward += 50.0  # Episode completion bonus
                terminated = True
                print(f"🎉 Episode complete! All {self.stages_completed} stages finished!")

        # Update info
        info = self._get_info()

        # Get observation with stage indicator
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info


# Convenience function for creating environment
def make_sequential_reach_env(reward_type="sparse", max_stages=3, **kwargs):
    """
    Factory function for creating Sequential Reach environment.

    Args:
        reward_type: "sparse", "shaped", or "dense"
        max_stages: 1, 2, or 3 (for curriculum learning)
        **kwargs: Additional arguments for SequentialReachEnv

    Returns:
        SequentialReachEnv instance
    """
    return SequentialReachEnv(
        reward_type=reward_type,
        max_stages=max_stages,
        **kwargs
    )


# Test script
if __name__ == "__main__":
    print("Testing Sequential Reaching Environment...")
    print("=" * 60)

    # Create environment
    env = SequentialReachEnv(reward_type="shaped", max_stages=3)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print()

    # Run a test episode
    obs, info = env.reset(seed=42)
    print(f"Initial state:")
    print(f"  Current stage: {info['current_stage']}")
    print(f"  Target left: {env.target_left}")
    print(f"  Target right: {env.target_right}")
    print()

    total_reward = 0
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(f"Step {step}: Stage {info['current_stage']}, "
                  f"Distance: {info['total_distance']:.3f}, "
                  f"Reward: {reward:.3f}")

        if terminated or truncated:
            print(f"\nEpisode finished at step {step}")
            print(f"  Stages completed: {info['stages_completed']}")
            print(f"  Episode success: {info['episode_success']}")
            print(f"  Total reward: {total_reward:.2f}")
            break

    env.close()
    print("\n✅ Sequential environment test complete!")
