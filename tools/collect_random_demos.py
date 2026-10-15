"""Collect demonstrations by random exploration with success filtering.

This is a pragmatic approach that doesn't require IK or scripted policies.
We sample random actions and only keep successful trajectories.
"""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
from tqdm import tqdm
import pickle
import json
from pathlib import Path


def collect_by_exploration(
    num_successes=10,
    max_attempts=1000,
    max_steps=500,
    save_dir="demonstrations",
    render=False
):
    """Collect demonstrations via random exploration.

    Args:
        num_successes: Number of successful trajectories to collect
        max_attempts: Maximum episodes to try
        max_steps: Max steps per episode
        save_dir: Where to save demonstrations
        render: Whether to render episodes

    Returns:
        stats: Collection statistics
    """
    env = DualArmPickPlaceEnv(
        render_mode="human" if render else None,
        max_steps=max_steps
    )

    trajectories = []
    stats = {
        'successes': 0,
        'attempts': 0,
        'episode_lengths': [],
        'episode_rewards': []
    }

    print(f"Collecting {num_successes} successful demonstrations via exploration...")
    print(f"This may take a while with random actions...")

    pbar = tqdm(total=num_successes, desc="Successful episodes")

    while stats['successes'] < num_successes and stats['attempts'] < max_attempts:
        obs, info = env.reset()

        observations = []
        actions = []
        rewards = []

        done = False
        step = 0

        while not done and step < max_steps:
            # Sample random action (biased toward smaller movements)
            action = np.random.normal(0, 0.05, size=16)
            action = np.clip(action, -0.1, 0.1)

            observations.append(obs)
            actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            done = terminated or truncated
            step += 1

        stats['attempts'] += 1
        stats['episode_lengths'].append(step)
        stats['episode_rewards'].append(sum(rewards))

        # Keep only successful trajectories
        if info.get('episode_success', False):
            trajectory = {
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'episode_length': step,
                'success': True
            }
            trajectories.append(trajectory)
            stats['successes'] += 1
            pbar.update(1)

    pbar.close()
    env.close()

    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    demo_file = save_path / "demonstrations_exploration.pkl"
    with open(demo_file, 'wb') as f:
        pickle.dump(trajectories, f)

    stats['success_rate'] = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0.0
    stats['avg_length'] = np.mean(stats['episode_lengths']) if stats['episode_lengths'] else 0
    stats['avg_reward'] = np.mean(stats['episode_rewards']) if stats['episode_rewards'] else 0

    stats_file = save_path / "exploration_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exploration complete!")
    print(f"{'='*60}")
    print(f"Attempts: {stats['attempts']}")
    print(f"Successes: {stats['successes']} ({stats['success_rate']*100:.2f}%)")
    print(f"Avg episode length: {stats['avg_length']:.1f}")
    print(f"Avg reward: {stats['avg_reward']:.2f}")
    print(f"Saved to: {demo_file}")

    return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-successes', type=int, default=10,
                        help='Number of successful demonstrations to collect')
    parser.add_argument('--max-attempts', type=int, default=1000,
                        help='Maximum episodes to attempt')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Max steps per episode')
    parser.add_argument('--save-dir', type=str, default='demonstrations',
                        help='Directory to save demonstrations')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')

    args = parser.parse_args()

    collect_by_exploration(
        num_successes=args.num_successes,
        max_attempts=args.max_attempts,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        render=args.render
    )
