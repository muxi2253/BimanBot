#!/usr/bin/env python3
"""Train dual-arm pick-place with SAC."""

import sys
sys.path.insert(0, 'backend')

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gym_env_dual_arm_pickplace import DualArmPickPlaceEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=100000)
    parser.add_argument('--eval-freq', type=int, default=5000)
    parser.add_argument('--save-freq', type=int, default=10000)
    parser.add_argument('--log-dir', type=str, default='logs/dual_arm_sac')
    parser.add_argument('--model-dir', type=str, default='models/dual_arm_sac')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    args = parser.parse_args()

    # Create directories
    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    env = Monitor(DualArmPickPlaceEnv(render_mode=None))
    eval_env = Monitor(DualArmPickPlaceEnv(render_mode=None))

    print(f"训练配置:")
    print(f"  总步数: {args.total_timesteps}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  观测维度: {env.observation_space.shape}")
    print(f"  动作维度: {env.action_space.shape}")

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=str(log_dir)
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="sac_model"
    )

    # Train
    print("\n开始训练...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100
    )

    # Save final model
    final_path = model_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n训练完成! 模型保存至: {final_path}")

    # Final evaluation
    print("\n最终评估 (10 episodes):")
    obs, _ = eval_env.reset()
    episode_rewards = []
    episode_successes = []

    for ep in range(10):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_successes.append(info['episode_success'])
        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, success={info['episode_success']}")

    print(f"\n最终结果:")
    print(f"  平均回报: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  成功率: {100*np.mean(episode_successes):.1f}%")

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
