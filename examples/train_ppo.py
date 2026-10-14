#!/usr/bin/env python3
"""
PPO Training Script for BimanBot

Train a PPO agent from scratch on either Reach or Pick-Place environment.

Example usage:
    # Train on Reach (easier)
    python examples/train_ppo.py --env reach --train --timesteps 1000000

    # Train on Pick-Place with domain randomization
    python examples/train_ppo.py --env pickplace --train --timesteps 5000000 --domain-random

    # Evaluate trained model
    python examples/train_ppo.py --eval --checkpoint logs/ppo_reach/best_model.zip --episodes 100
"""

import argparse
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.env_util import make_vec_env
    import torch
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nInstall training dependencies with:")
    print("  pip install stable-baselines3 torch gymnasium")
    sys.exit(1)


def make_env(env_name: str, domain_random: bool = False, camera_obs: bool = False):
    """Create BimanBot environment."""
    if env_name == "reach":
        from gym_env_reach import DualArmReachEnv
        def _init():
            # DualArmReachEnv doesn't support reward_type or domain_randomization yet
            return DualArmReachEnv()
        return _init
    elif env_name == "pickplace":
        from gym_env import DualArmPickPlaceEnv
        def _init():
            return DualArmPickPlaceEnv(
                reward_type="shaped",
                camera_obs=camera_obs,
                domain_randomization=domain_random,
            )
        return _init
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def train_ppo(args):
    """Train PPO agent."""
    print(f"🚀 Training PPO on {args.env} environment")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Domain randomization: {args.domain_random}")
    print(f"   Camera observations: {args.camera_obs}")
    print(f"   Save frequency: {args.save_freq:,} steps")
    print(f"   Device: {args.device}")

    # Create log directory
    log_dir = Path(args.log_dir) / f"ppo_{args.env}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create vectorized environment
    print("\n📦 Creating environment...")
    env = make_vec_env(
        make_env(args.env, args.domain_random, args.camera_obs),
        n_envs=args.n_envs,
        vec_env_cls=DummyVecEnv,
    )
    env = VecMonitor(env, str(log_dir / "monitor"))

    # Create evaluation environment
    eval_env = make_vec_env(
        make_env(args.env, domain_random=False, camera_obs=args.camera_obs),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # PPO hyperparameters
    ppo_kwargs = {
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "verbose": 1,
        "tensorboard_log": str(log_dir),
        "device": args.device,
    }

    # Determine policy type based on observation space
    from gymnasium import spaces
    if isinstance(env.observation_space, spaces.Dict):
        policy_type = "MultiInputPolicy"
    else:
        policy_type = "MlpPolicy"

    print(f"   Using policy: {policy_type}")

    # Load checkpoint or create new model
    if args.checkpoint:
        print(f"\n📂 Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, **ppo_kwargs)
    else:
        print(f"\n🤖 Creating new PPO model with {policy_type}...")
        model = PPO(policy_type, env, **ppo_kwargs)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    print("\n🎓 Starting training...")
    print(f"📊 TensorBoard: tensorboard --logdir {log_dir}")
    print(f"💾 Models will be saved to: {log_dir}")
    print("-" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n✅ Training complete! Final model saved to: {final_path}")

    env.close()
    eval_env.close()


def evaluate_ppo(args):
    """Evaluate trained PPO agent."""
    if not args.checkpoint:
        print("ERROR: --checkpoint required for evaluation")
        sys.exit(1)

    print(f"🎯 Evaluating PPO model: {args.checkpoint}")
    print(f"   Environment: {args.env}")
    print(f"   Episodes: {args.episodes}")

    # Create environment
    env = DummyVecEnv([make_env(args.env, domain_random=False, camera_obs=args.camera_obs)])

    # Load model
    model = PPO.load(args.checkpoint, device=args.device)

    # Evaluate
    print("\n🏃 Running evaluation...")
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

            if done:
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                success = info[0].get("is_success", False)
                successes.append(success)

                status = "✅" if success else "❌"
                print(f"Episode {ep+1:3d}: {status} Reward: {total_reward:7.2f}, Steps: {steps:3d}")

    # Print summary
    import numpy as np
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    print(f"Success Rate:    {np.mean(successes)*100:6.2f}%")
    print(f"Mean Reward:     {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"Mean Length:     {np.mean(episode_lengths):8.2f} ± {np.std(episode_lengths):6.2f}")
    print(f"Min/Max Reward:  {np.min(episode_rewards):8.2f} / {np.max(episode_rewards):8.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate PPO on BimanBot")

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train a new model")
    mode_group.add_argument("--eval", action="store_true", help="Evaluate a trained model")

    # Environment
    parser.add_argument("--env", type=str, default="reach", choices=["reach", "pickplace"],
                        help="Environment to train on")
    parser.add_argument("--domain-random", action="store_true",
                        help="Enable domain randomization")
    parser.add_argument("--camera-obs", action="store_true",
                        help="Use camera observations (image-based policy)")

    # Training
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--save-freq", type=int, default=50000,
                        help="Checkpoint save frequency (steps)")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency (steps)")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Optimization epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")

    # Evaluation
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")

    # I/O
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load (for resume or eval)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, mps, auto)")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    if args.train:
        train_ppo(args)
    else:
        evaluate_ppo(args)


if __name__ == "__main__":
    main()
