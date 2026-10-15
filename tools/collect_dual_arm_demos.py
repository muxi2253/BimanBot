#!/usr/bin/env python3
"""Collect demonstrations for dual-arm pick-place using waypoint expert."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import argparse
import numpy as np
import pickle
import json
from tqdm import tqdm

from simple_dual_arm_expert import WaypointExpert
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv


def collect_demonstrations(
    num_episodes=100,
    max_steps=500,
    save_dir="demonstrations/dual_arm",
    success_only=True,
    render=False
):
    """Collect expert demonstrations using waypoint controller.

    Args:
        num_episodes: Number of episodes to collect
        max_steps: Max steps per episode
        save_dir: Directory to save demonstrations
        success_only: Only save successful trajectories
        render: Whether to render episodes

    Returns:
        stats: Dictionary of collection statistics
    """
    # Create environment and expert
    env = DualArmPickPlaceEnv(
        render_mode="human" if render else None,
        max_steps=max_steps
    )
    expert = WaypointExpert()

    # Storage
    all_trajectories = []
    stats = {
        'success_count': 0,
        'total_episodes': 0,
        'episode_lengths': [],
        'episode_rewards': [],
        'success_rate': 0.0
    }

    print(f"收集 {num_episodes} 条演示轨迹...")
    print(f"专家类型: Waypoint Controller")
    print(f"最大步数: {max_steps}")

    for episode in tqdm(range(num_episodes)):
        trajectory, success = expert.demonstrate(env, max_steps=max_steps, render=render)

        stats['total_episodes'] += 1
        stats['episode_lengths'].append(trajectory['episode_length'])
        stats['episode_rewards'].append(trajectory['rewards'].sum())

        if success:
            stats['success_count'] += 1

        # Save trajectory if success or if we're keeping all
        if success or not success_only:
            all_trajectories.append(trajectory)

    # Compute final statistics
    stats['success_rate'] = stats['success_count'] / stats['total_episodes']
    stats['avg_episode_length'] = np.mean(stats['episode_lengths'])
    stats['avg_episode_reward'] = np.mean(stats['episode_rewards'])
    stats['num_saved_trajectories'] = len(all_trajectories)

    # Save demonstrations
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save trajectories as pickle
    demo_file = save_path / "demonstrations.pkl"
    with open(demo_file, 'wb') as f:
        pickle.dump(all_trajectories, f)

    # Save statistics as JSON
    stats_file = save_path / "collection_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"收集完成!")
    print(f"{'='*60}")
    print(f"总episodes: {stats['total_episodes']}")
    print(f"成功episodes: {stats['success_count']} ({stats['success_rate']*100:.1f}%)")
    print(f"保存轨迹数: {stats['num_saved_trajectories']}")
    print(f"平均episode长度: {stats['avg_episode_length']:.1f} 步")
    print(f"平均episode回报: {stats['avg_episode_reward']:.2f}")
    print(f"\n保存至: {demo_file}")

    env.close()
    return stats


def inspect_demonstrations(demo_file="demonstrations/dual_arm/demonstrations.pkl"):
    """Load and inspect saved demonstrations."""
    with open(demo_file, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"从 {demo_file} 加载了 {len(trajectories)} 条轨迹")
    print(f"\n第一条轨迹:")
    traj = trajectories[0]
    print(f"  观测形状: {traj['observations'].shape}")
    print(f"  动作形状: {traj['actions'].shape}")
    print(f"  奖励形状: {traj['rewards'].shape}")
    print(f"  Episode长度: {traj['episode_length']}")
    print(f"  成功: {traj['success']}")
    print(f"  总回报: {traj['rewards'].sum():.2f}")

    # Aggregate statistics
    lengths = [t['episode_length'] for t in trajectories]
    rewards = [t['rewards'].sum() for t in trajectories]
    successes = [t['success'] for t in trajectories]

    print(f"\n数据集统计:")
    print(f"  总transitions: {sum(lengths)}")
    print(f"  平均episode长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  平均episode回报: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  成功率: {np.mean(successes)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集双臂抓取演示")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="要收集的episodes数量")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="每个episode的最大步数")
    parser.add_argument("--save-dir", type=str, default="demonstrations/dual_arm",
                        help="保存演示的目录")
    parser.add_argument("--keep-failures", action="store_true",
                        help="保留失败的轨迹（默认: 仅成功）")
    parser.add_argument("--render", action="store_true",
                        help="收集时渲染episodes")
    parser.add_argument("--inspect", type=str, default=None,
                        help="检查现有演示文件")

    args = parser.parse_args()

    if args.inspect:
        inspect_demonstrations(args.inspect)
    else:
        collect_demonstrations(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            save_dir=args.save_dir,
            success_only=not args.keep_failures,
            render=args.render
        )
