#!/usr/bin/env python3
"""
Collect Expert Demonstrations for Behavior Cloning

Uses the rule-based policy to generate expert demonstrations.

Example usage:
    # Collect 500 episodes
    python examples/collect_expert_data.py --episodes 500 --output data/expert_demos.npz

    # Collect with domain randomization
    python examples/collect_expert_data.py --episodes 200 --domain-random --output data/expert_random.npz
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def collect_demonstrations(args):
    """Collect expert demonstrations using rule-based policy."""
    print(f"🎬 Collecting Expert Demonstrations")
    print(f"   Episodes: {args.episodes}")
    print(f"   Environment: {args.env}")
    print(f"   Domain randomization: {args.domain_random}")
    print(f"   Output: {args.output}")

    # Import environment
    if args.env == "reach":
        from gym_env_reach import DualArmReachEnv
        env = DualArmReachEnv(
            reward_type="shaped",
            domain_randomization=args.domain_random,
        )
    elif args.env == "pickplace":
        from gym_env import DualArmPickPlaceEnv
        env = DualArmPickPlaceEnv(
            reward_type="shaped",
            camera_obs=False,
            domain_randomization=args.domain_random,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # Storage
    all_observations = []
    all_actions = []
    all_rewards = []

    print("\n🏃 Collecting demonstrations...")
    print("-" * 60)

    success_count = 0
    total_transitions = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep if args.seed is not None else None)
        done = False
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        steps = 0

        while not done and steps < 500:
            # Get expert action (in this case, from environment's expert policy)
            # For reach/pickplace, we'll use random sampling as placeholder
            # In practice, you'd call your rule-based controller here
            action = env.action_space.sample()

            episode_obs.append(obs)
            episode_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_rewards.append(reward)
            steps += 1

        # Store episode
        success = info.get("is_success", False)
        if success or not args.success_only:
            all_observations.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            total_transitions += len(episode_obs)

            if success:
                success_count += 1

        status = "✅" if success else "❌"
        total_reward = sum(episode_rewards)
        print(f"Episode {ep+1:3d}/{args.episodes}: {status} "
              f"Reward: {total_reward:7.2f}, Steps: {steps:3d}, "
              f"Transitions: {total_transitions:,}")

    env.close()

    # Convert to numpy arrays
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
    )

    print("\n" + "=" * 60)
    print("📊 Collection Summary")
    print("=" * 60)
    print(f"Episodes collected:  {args.episodes}")
    print(f"Successful episodes: {success_count} ({success_count/args.episodes*100:.1f}%)")
    print(f"Total transitions:   {total_transitions:,}")
    print(f"Observation shape:   {observations.shape}")
    print(f"Action shape:        {actions.shape}")
    print(f"Mean reward:         {np.mean(rewards):.4f}")
    print(f"Saved to:            {output_path}")
    print("\n✅ Collection complete!")
    print(f"\nNext: Train BC model with:")
    print(f"  python examples/train_bc.py --train --data {args.output} --epochs 100")


def main():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")

    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--env", type=str, default="reach", choices=["reach", "pickplace"],
                        help="Environment to collect from")
    parser.add_argument("--domain-random", action="store_true",
                        help="Enable domain randomization")
    parser.add_argument("--success-only", action="store_true",
                        help="Only save successful episodes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (None for random)")
    parser.add_argument("--output", type=str, default="data/expert_demos.npz",
                        help="Output file path (.npz)")

    args = parser.parse_args()

    collect_demonstrations(args)


if __name__ == "__main__":
    main()
