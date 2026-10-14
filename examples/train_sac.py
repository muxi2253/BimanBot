#!/usr/bin/env python3
"""
SAC Training Script for BimanBot

SAC is an off-policy RL algorithm that is more sample-efficient than PPO.

Example usage:
    python examples/train_sac.py --train --env reach --timesteps 500000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    import torch
except ImportError as e:
    print(f"ERROR: {e}")
    print("\nInstall with: pip install stable-baselines3 torch gymnasium")
    sys.exit(1)


def make_env(env_name: str):
    """Create BimanBot environment."""
    if env_name == "reach":
        from gym_env_reach import DualArmReachEnv
        return lambda: DualArmReachEnv()
    elif env_name == "pickplace":
        from gym_env import DualArmPickPlaceEnv
        return lambda: DualArmPickPlaceEnv(reward_type="shaped", camera_obs=False)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def train_sac(args):
    """Train SAC agent."""
    print(f"🚀 Training SAC on {args.env} environment")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Device: {args.device}")

    log_dir = Path(args.log_dir) / f"sac_{args.env}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = make_vec_env(make_env(args.env), n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env(make_env(args.env), n_envs=1, vec_env_cls=DummyVecEnv)

    # SAC hyperparameters
    sac_kwargs = {
        "learning_rate": args.lr,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "gamma": args.gamma,
        "verbose": 1,
        "tensorboard_log": str(log_dir),
        "device": args.device,
    }

    # Create or load model
    if args.checkpoint:
        print(f"\n📂 Loading checkpoint: {args.checkpoint}")
        model = SAC.load(args.checkpoint, env=env, **sac_kwargs)
    else:
        print("\n🤖 Creating new SAC model...")
        model = SAC("MlpPolicy", env, **sac_kwargs)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac_model",
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
    print("-" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n✅ Training complete! Final model: {final_path}")

    env.close()
    eval_env.close()


def evaluate_sac(args):
    """Evaluate trained SAC agent."""
    if not args.checkpoint:
        print("ERROR: --checkpoint required for evaluation")
        sys.exit(1)

    print(f"🎯 Evaluating SAC model: {args.checkpoint}")

    env = DummyVecEnv([make_env(args.env)])
    model = SAC.load(args.checkpoint, device=args.device)

    print("\n🏃 Running evaluation...")
    import numpy as np
    
    episode_rewards = []
    episode_lengths = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

            if done:
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                print(f"Episode {ep+1:3d}: Reward: {total_reward:7.2f}, Steps: {steps:3d}")

    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    print(f"Mean Reward:     {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"Mean Length:     {np.mean(episode_lengths):8.2f} ± {np.std(episode_lengths):6.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate SAC")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--eval", action="store_true")

    parser.add_argument("--env", type=str, default="reach", choices=["reach", "pickplace"])
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--episodes", type=int, default=50)
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save-freq", type=int, default=50000)
    parser.add_argument("--eval-freq", type=int, default=10000)

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    if args.train:
        train_sac(args)
    else:
        evaluate_sac(args)


if __name__ == "__main__":
    main()
