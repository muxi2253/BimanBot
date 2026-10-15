#!/usr/bin/env python3
"""
Demonstration collection tool for imitation learning.

Collects expert trajectories using scripted policy and saves them
in a format ready for behavioral cloning training.
"""

import argparse
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from dual_arm_scripted_policy import DualArmScriptedPolicy
    from simple_dual_arm_expert import WaypointExpert
    from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
    TASK_TYPE = "pickplace"
except ImportError:
    TASK_TYPE = None

if TASK_TYPE is None:
    try:
        from expert_policy import ExpertReachingPolicy, ImprovedExpertPolicy
        from gym_env_sequential import SequentialReachEnv
        TASK_TYPE = "reach"
    except ImportError:
        raise ImportError("Could not import either reach or pickplace task")


def collect_demonstrations(
    num_episodes=100,
    expert_type="simple",
    kp=0.3,
    max_action=0.1,
    noise_std=0.01,
    max_steps=500,
    reward_type="dense",
    save_dir="demonstrations",
    success_only=True,
    render=False,
    task="auto"
):
    """
    Collect expert demonstrations.

    Args:
        num_episodes: Number of episodes to collect
        expert_type: "simple" or "jacobian" (for reach) or "ik" (for pickplace)
        kp: Proportional gain
        max_action: Max action magnitude
        noise_std: Action noise for diversity
        max_steps: Max steps per episode
        reward_type: Environment reward type
        save_dir: Directory to save demonstrations
        success_only: Only save successful trajectories
        render: Whether to render episodes
        task: "reach" or "pickplace" (default: auto-detect)

    Returns:
        stats: Dictionary of collection statistics
    """
    # Auto-detect task type if not specified
    if task == "auto":
        task = TASK_TYPE

    # Create environment and expert based on task
    if task == "pickplace":
        env = DualArmPickPlaceEnv(
            render_mode="human" if render else None,
            max_steps=max_steps
        )
        expert = DualArmScriptedPolicy()
    else:  # reach task
        env = SequentialReachEnv(
            render_mode="human" if render else None,
            max_steps=max_steps,
            reward_type=reward_type,
            max_stages=1
        )
        if expert_type == "jacobian":
            expert = ImprovedExpertPolicy(kp=kp, max_action=max_action, noise_std=noise_std)
        else:
            expert = ExpertReachingPolicy(kp=kp, max_action=max_action, noise_std=noise_std)

    # Storage
    all_trajectories = []
    stats = {
        'success_count': 0,
        'total_episodes': 0,
        'episode_lengths': [],
        'episode_rewards': [],
        'success_rate': 0.0
    }

    print(f"Collecting {num_episodes} demonstrations...")
    print(f"Expert type: {expert_type}, kp={kp}, max_action={max_action}, noise={noise_std}")

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
    print(f"Collection complete!")
    print(f"{'='*60}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Successful episodes: {stats['success_count']} ({stats['success_rate']*100:.1f}%)")
    print(f"Saved trajectories: {stats['num_saved_trajectories']}")
    print(f"Avg episode length: {stats['avg_episode_length']:.1f} steps")
    print(f"Avg episode reward: {stats['avg_episode_reward']:.2f}")
    print(f"\nSaved to: {demo_file}")

    env.close()
    return stats


def inspect_demonstrations(demo_file="demonstrations/demonstrations.pkl"):
    """Load and inspect saved demonstrations."""
    with open(demo_file, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"Loaded {len(trajectories)} trajectories from {demo_file}")
    print(f"\nFirst trajectory:")
    traj = trajectories[0]
    print(f"  Observations shape: {traj['observations'].shape}")
    print(f"  Actions shape: {traj['actions'].shape}")
    print(f"  Rewards shape: {traj['rewards'].shape}")
    print(f"  Episode length: {traj['episode_length']}")
    print(f"  Success: {traj['success']}")
    print(f"  Total reward: {traj['rewards'].sum():.2f}")

    # Aggregate statistics
    lengths = [t['episode_length'] for t in trajectories]
    rewards = [t['rewards'].sum() for t in trajectories]
    successes = [t['success'] for t in trajectories]

    print(f"\nDataset statistics:")
    print(f"  Total transitions: {sum(lengths)}")
    print(f"  Avg episode length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Avg episode reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Success rate: {np.mean(successes)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--expert-type", choices=["simple", "jacobian", "scripted"], default="scripted",
                        help="Type of expert policy (simple/jacobian for reach, scripted for pickplace)")
    parser.add_argument("--task", choices=["reach", "pickplace", "auto"], default="auto",
                        help="Task type (auto-detect by default)")
    parser.add_argument("--kp", type=float, default=0.3,
                        help="Proportional gain")
    parser.add_argument("--max-action", type=float, default=0.1,
                        help="Maximum action magnitude")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Action noise std (for diversity)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--reward-type", choices=["sparse", "shaped", "dense"], default="dense",
                        help="Environment reward type")
    parser.add_argument("--save-dir", type=str, default="demonstrations",
                        help="Directory to save demonstrations")
    parser.add_argument("--keep-failures", action="store_true",
                        help="Keep failed trajectories (default: success only)")
    parser.add_argument("--render", action="store_true",
                        help="Render episodes during collection")
    parser.add_argument("--inspect", type=str, default=None,
                        help="Inspect existing demonstration file")

    args = parser.parse_args()

    if args.inspect:
        inspect_demonstrations(args.inspect)
    else:
        collect_demonstrations(
            num_episodes=args.num_episodes,
            expert_type=args.expert_type,
            kp=args.kp,
            max_action=args.max_action,
            noise_std=args.noise_std,
            max_steps=args.max_steps,
            reward_type=args.reward_type,
            save_dir=args.save_dir,
            success_only=not args.keep_failures,
            render=args.render,
            task=args.task
        )
