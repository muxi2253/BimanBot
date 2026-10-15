"""Test IK-based expert policy on dual-arm pick-place task."""

import sys
sys.path.insert(0, 'backend')

import numpy as np
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv
from dual_arm_ik_expert import DualArmIKExpert


def test_ik_expert(num_episodes=3, render=False):
    """Test IK expert on dual-arm pick-place."""
    env = DualArmPickPlaceEnv(render_mode='human' if render else None)
    expert = DualArmIKExpert()

    successes = 0
    total_reward = 0.0

    for ep in range(num_episodes):
        obs, info = env.reset()
        expert.reset()

        # Get target positions from env
        target_left = info['target_left']
        target_right = info['target_right']

        episode_reward = 0
        done = False
        step = 0

        print(f"\n=== Episode {ep + 1} ===")
        print(f"Target left: {target_left}")
        print(f"Target right: {target_right}")
        print(f"Obs shape: {obs.shape}")
        print(f"Left obj from obs: {obs[20:23]}")
        print(f"Right obj from obs: {obs[23:26]}")
        print(f"Left TCP from obs: {obs[14:17]}")
        print(f"Right TCP from obs: {obs[17:20]}")

        while not done and step < 500:
            # Get action from expert
            action, expert_done = expert.get_action(obs, target_left, target_right)

            # Debug first few steps
            if step < 3:
                print(f"\nStep {step}:")
                print(f"  Action (first 7): {action[:7]}")
                print(f"  Action norm: {np.linalg.norm(action[:14]):.4f}")
                print(f"  Left TCP: {obs[14:17]}")
                print(f"  Right TCP: {obs[17:20]}")

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated or expert_done

            episode_reward += reward
            step += 1

            if step % 50 == 0:
                left_dist = info.get('left_distance', 0.0)
                right_dist = info.get('right_distance', 0.0)
                print(f"Step {step}: L_dist={left_dist:.3f}, R_dist={right_dist:.3f}, "
                      f"phase={expert.phase}, reward={episode_reward:.2f}")

        success = info.get('success', False)
        successes += success
        total_reward += episode_reward

        print(f"\nEpisode {ep + 1} finished:")
        print(f"  Steps: {step}")
        print(f"  Success: {success}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Final distances: L={info.get('left_distance', 0):.3f}, "
              f"R={info.get('right_distance', 0):.3f}")

    env.close()

    print(f"\n{'='*50}")
    print(f"IK Expert Results ({num_episodes} episodes):")
    print(f"  Success rate: {100 * successes / num_episodes:.1f}%")
    print(f"  Avg reward: {total_reward / num_episodes:.2f}")
    print(f"{'='*50}")

    return successes / num_episodes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    success_rate = test_ik_expert(args.episodes, args.render)
    print(f"\nFinal success rate: {100 * success_rate:.1f}%")
