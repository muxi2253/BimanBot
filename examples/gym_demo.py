#!/usr/bin/env python3
"""
Example: Using the Dual-Arm Reach environment with random policy

This demonstrates:
1. Creating the environment
2. Running episodes with random actions
3. Collecting statistics
"""

import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

# Import after adding to path
from gym_env_reach import DualArmReachEnv


def run_episode(env, render=False):
    """Run one episode with random policy"""
    obs, info = env.reset()

    episode_reward = 0
    step_count = 0

    while True:
        # Random action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        if render and step_count % 10 == 0:
            print(f"  Step {step_count}: dist={info['total_distance']:.3f}m, reward={reward:.3f}")

        if terminated or truncated:
            break

    return episode_reward, step_count, info


def main():
    print("=" * 60)
    print("Dual-Arm Reach Environment - Random Policy Demo")
    print("=" * 60)

    # Create environment
    env = DualArmReachEnv(max_steps=200)

    print(f"\nEnvironment info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max steps: {env.max_steps}")

    # Run multiple episodes
    num_episodes = 5
    results = []

    print(f"\nRunning {num_episodes} episodes with random policy...\n")

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")

        ep_reward, ep_steps, ep_info = run_episode(env, render=True)
        results.append({
            'reward': ep_reward,
            'steps': ep_steps,
            'final_distance': ep_info['total_distance'],
            'success': ep_info['success']
        })

        print(f"  Final: reward={ep_reward:.2f}, steps={ep_steps}, "
              f"dist={ep_info['total_distance']:.3f}m, "
              f"success={ep_info['success']}\n")

    # Statistics
    print("=" * 60)
    print("Statistics:")
    print("=" * 60)

    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_dist = np.mean([r['final_distance'] for r in results])
    success_rate = np.mean([r['success'] for r in results])

    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average final distance: {avg_dist:.3f}m")
    print(f"Success rate: {success_rate:.1%}")

    env.close()

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  - Train RL policy (PPO, SAC, etc.)")
    print("  - Collect demonstrations for IL")
    print("  - Add more complex tasks")
    print("=" * 60)


if __name__ == '__main__':
    main()
