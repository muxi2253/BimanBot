#!/usr/bin/env python3
"""
Curriculum Learning Wrapper for Sequential Reaching

Gradually increases task difficulty:
- Start with 1 stage
- After reaching 80% success rate, unlock 2 stages
- After 80% success on 2 stages, unlock all 3 stages

This helps the agent learn progressively rather than facing
the full 3-stage task from the beginning.
"""

import numpy as np
import gymnasium as gym
from collections import deque


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum learning wrapper that progressively increases task difficulty.

    Monitors success rate and automatically advances to the next difficulty level
    when the agent achieves consistent performance.
    """

    def __init__(
        self,
        env,
        initial_stages=1,
        success_threshold=0.8,
        window_size=100,
        min_episodes=50,
        verbose=True,
    ):
        """
        Args:
            env: Base environment (SequentialReachEnv)
            initial_stages: Starting number of stages (1, 2, or 3)
            success_threshold: Success rate needed to advance (e.g., 0.8 = 80%)
            window_size: Number of recent episodes to evaluate
            min_episodes: Minimum episodes before considering advancement
            verbose: Print curriculum progression messages
        """
        super().__init__(env)
        self.initial_stages = initial_stages
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.min_episodes = min_episodes
        self.verbose = verbose

        # Curriculum state
        self.current_max_stages = initial_stages
        self.episode_count = 0
        self.success_buffer = deque(maxlen=window_size)

        # Statistics
        self.curriculum_advances = 0
        self.episodes_per_level = []

        # Set initial max_stages on wrapped env
        if hasattr(self.env, 'max_stages'):
            self.env.max_stages = self.current_max_stages

        if self.verbose:
            print(f"🎓 Curriculum Learning initialized:")
            print(f"   Starting stages: {self.current_max_stages}")
            print(f"   Success threshold: {self.success_threshold:.1%}")
            print(f"   Window size: {self.window_size}")

    def reset(self, **kwargs):
        """Reset environment and check curriculum progression."""
        # Check if we should advance curriculum
        if len(self.success_buffer) >= self.min_episodes:
            self._check_curriculum_advancement()

        # Reset with current curriculum level
        if hasattr(self.env, 'max_stages'):
            self.env.max_stages = self.current_max_stages

        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute step and track success for curriculum."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track episode completion
        if terminated or truncated:
            self.episode_count += 1

            # Determine success based on current curriculum level
            stages_completed = info.get('stages_completed', 0)
            is_success = (stages_completed >= self.current_max_stages)

            self.success_buffer.append(is_success)

            # Add curriculum info
            info['curriculum_stage'] = self.current_max_stages
            info['curriculum_success'] = is_success

        return obs, reward, terminated, truncated, info

    def _check_curriculum_advancement(self):
        """Check if we should advance to next curriculum level."""
        if self.current_max_stages >= 3:
            # Already at max difficulty
            return

        # Calculate success rate over window
        success_rate = np.mean(self.success_buffer)

        # Check if ready to advance
        if success_rate >= self.success_threshold:
            self._advance_curriculum()

    def _advance_curriculum(self):
        """Advance to next curriculum level."""
        old_stages = self.current_max_stages
        self.current_max_stages = min(self.current_max_stages + 1, 3)

        # Record statistics
        self.curriculum_advances += 1
        self.episodes_per_level.append(self.episode_count)

        # Clear success buffer for new level
        self.success_buffer.clear()

        if self.verbose:
            success_rate = np.mean(list(self.success_buffer)[-self.window_size:]) if self.success_buffer else 0
            print(f"\n" + "=" * 60)
            print(f"📈 CURRICULUM ADVANCED!")
            print(f"   {old_stages} stages → {self.current_max_stages} stages")
            print(f"   Episodes trained: {self.episode_count}")
            print(f"   Recent success rate: {success_rate:.1%}")
            print("=" * 60 + "\n")

    def get_curriculum_stats(self):
        """Get curriculum learning statistics."""
        return {
            'current_stages': self.current_max_stages,
            'curriculum_advances': self.curriculum_advances,
            'episode_count': self.episode_count,
            'episodes_per_level': self.episodes_per_level,
            'recent_success_rate': np.mean(self.success_buffer) if self.success_buffer else 0.0,
        }


# Helper function for creating curriculum environment
def make_curriculum_env(
    reward_type="sparse",
    initial_stages=1,
    success_threshold=0.8,
    **kwargs
):
    """
    Factory function for creating curriculum-wrapped environment.

    Args:
        reward_type: "sparse", "shaped", or "dense"
        initial_stages: Starting difficulty (1, 2, or 3)
        success_threshold: Success rate needed to advance
        **kwargs: Additional CurriculumWrapper arguments

    Returns:
        CurriculumWrapper instance
    """
    from gym_env_sequential import SequentialReachEnv

    env = SequentialReachEnv(reward_type=reward_type, max_stages=3)
    env = CurriculumWrapper(
        env,
        initial_stages=initial_stages,
        success_threshold=success_threshold,
        **kwargs
    )

    return env


# Test script
if __name__ == "__main__":
    print("Testing Curriculum Learning Wrapper...")
    print("=" * 60)

    # Create curriculum environment
    env = make_curriculum_env(
        reward_type="shaped",
        initial_stages=1,
        success_threshold=0.8,
        window_size=10,  # Small window for testing
        min_episodes=5,
        verbose=True
    )

    print(f"\nObservation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Current curriculum stage: {env.current_max_stages}")
    print()

    # Simulate episodes
    print("Simulating episodes (with fake success)...")
    for ep in range(30):
        obs, info = env.reset(seed=ep)
        done = False
        step_count = 0

        while not done and step_count < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            # Fake success for testing (pretend we reached current stage)
            if done and ep % 2 == 0:  # 50% fake success rate
                info['stages_completed'] = env.current_max_stages

        if (ep + 1) % 10 == 0:
            stats = env.get_curriculum_stats()
            print(f"\nAfter {ep + 1} episodes:")
            print(f"  Current stages: {stats['current_stages']}")
            print(f"  Curriculum advances: {stats['curriculum_advances']}")
            print(f"  Recent success: {stats['recent_success_rate']:.1%}")

    print("\n" + "=" * 60)
    print("Final curriculum stats:")
    stats = env.get_curriculum_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    env.close()
    print("\n✅ Curriculum wrapper test complete!")
