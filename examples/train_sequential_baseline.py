#!/usr/bin/env python3
"""
Baseline Training Script for Sequential Reaching

Trains PPO on sequential reaching task with two reward variants:
1. Sparse: Only stage completion bonuses
2. Shaped: Distance-based intermediate rewards

Example usage:
    # Sparse reward baseline
    python examples/train_sequential_baseline.py --train --reward-type sparse --seed 42

    # Shaped reward baseline
    python examples/train_sequential_baseline.py --train --reward-type shaped --seed 42

    # Evaluate trained model
    python examples/train_sequential_baseline.py --eval --checkpoint logs/sequential_sparse/best_model.zip
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, BaseCallback
    )
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    import torch
    import numpy as np
except ImportError as e:
    print(f"ERROR: {e}")
    print("\nInstall with: pip install stable-baselines3 torch gymnasium")
    sys.exit(1)


def make_env(reward_type="sparse", seed=None):
    """Create Sequential Reaching environment."""
    def _init():
        from gym_env_sequential import SequentialReachEnv
        env = SequentialReachEnv(reward_type=reward_type, max_stages=3)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


class StageProgressCallback(BaseCallback):
    """
    Custom callback to log stage completion statistics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_stages = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if episode ended
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Get info from the environment
                info = self.locals["infos"][idx]
                stages = info.get("stages_completed", 0)

                self.episode_stages.append(stages)

                # Log every 10 episodes
                if len(self.episode_stages) % 10 == 0:
                    avg_stages = np.mean(self.episode_stages[-10:])
                    self.logger.record("rollout/avg_stages_completed", avg_stages)

                    # Success rate (all 3 stages)
                    success_rate = np.mean([s >= 3 for s in self.episode_stages[-10:]])
                    self.logger.record("rollout/success_rate", success_rate)

                    if self.verbose > 0 and len(self.episode_stages) % 100 == 0:
                        print(f"Episodes: {len(self.episode_stages)}, "
                              f"Avg stages: {avg_stages:.2f}, "
                              f"Success rate: {success_rate:.2%}")

        return True


def train_baseline(args):
    """Train PPO baseline on sequential reaching."""
    print(f"🚀 Training PPO Baseline on Sequential Reaching")
    print(f"   Reward type: {args.reward_type}")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Seed: {args.seed}")
    print(f"   Parallel envs: {args.n_envs}")
    print(f"   Device: {args.device}")
    print()

    # Create log directory
    log_dir = Path(args.log_dir) / f"sequential_{args.reward_type}"
    if args.seed is not None:
        log_dir = log_dir / f"seed_{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create vectorized environments
    print("📦 Creating environments...")
    env = make_vec_env(
        make_env(args.reward_type, args.seed),
        n_envs=args.n_envs,
        vec_env_cls=DummyVecEnv,
    )

    eval_env = make_vec_env(
        make_env(args.reward_type, args.seed),
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

    # Create or load model
    if args.checkpoint:
        print(f"\n📂 Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, **ppo_kwargs)
    else:
        print("\n🤖 Creating new PPO model...")
        model = PPO("MlpPolicy", env, **ppo_kwargs)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_sequential",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    stage_callback = StageProgressCallback(verbose=1)

    # Train
    print("\n🎓 Starting training...")
    print(f"📊 TensorBoard: tensorboard --logdir {log_dir}")
    print("-" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback, stage_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = log_dir / "final_model"
    model.save(final_path)
    print(f"\n💾 Final model saved to: {final_path}.zip")

    env.close()
    eval_env.close()

    print("\n✅ Training complete!")
    print(f"📂 Results saved in: {log_dir}")


def evaluate_baseline(args):
    """Evaluate trained model."""
    print(f"🔍 Evaluating model: {args.checkpoint}")
    print(f"   Episodes: {args.episodes}")
    print()

    # Load model
    from gym_env_sequential import SequentialReachEnv

    # Create environment
    env = SequentialReachEnv(reward_type="sparse")  # Use sparse for eval

    # Load model
    model = PPO.load(args.checkpoint)

    # Evaluation metrics
    episode_rewards = []
    episode_stages = []
    successes = []

    print("Running evaluation...")
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_stages.append(info["stages_completed"])
        successes.append(info["episode_success"])

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{args.episodes} - "
                  f"Stages: {info['stages_completed']}, "
                  f"Success: {info['episode_success']}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean stages completed: {np.mean(episode_stages):.2f} ± {np.std(episode_stages):.2f}")
    print(f"Success rate: {np.mean(successes):.2%}")
    print(f"  Stage 1 completion: {np.mean([s >= 1 for s in episode_stages]):.2%}")
    print(f"  Stage 2 completion: {np.mean([s >= 2 for s in episode_stages]):.2%}")
    print(f"  Stage 3 completion: {np.mean([s >= 3 for s in episode_stages]):.2%}")
    print("=" * 60)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO baseline on Sequential Reaching")

    # Mode
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")

    # Environment
    parser.add_argument("--reward-type", type=str, default="sparse",
                        choices=["sparse", "shaped", "dense"],
                        help="Reward type (sparse, shaped, or dense)")

    # Training parameters
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy coefficient")

    # Callbacks
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluation frequency")

    # Paths
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load/evaluate")

    # Evaluation
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, auto)")

    args = parser.parse_args()

    if args.train:
        train_baseline(args)
    elif args.eval:
        if not args.checkpoint:
            print("ERROR: --checkpoint required for evaluation")
            sys.exit(1)
        evaluate_baseline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
