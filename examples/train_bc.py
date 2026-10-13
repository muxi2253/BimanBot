#!/usr/bin/env python3
"""
Behavior Cloning Training Script for BimanBot

Train a policy to imitate expert demonstrations.

Example usage:
    # Collect expert data first
    python examples/collect_expert_data.py --episodes 500 --output data/expert_demos.npz

    # Train BC model
    python examples/train_bc.py --train --data data/expert_demos.npz --epochs 100

    # Evaluate trained model
    python examples/train_bc.py --eval --checkpoint logs/bc_model.pth --episodes 50
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import gymnasium as gym
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nInstall training dependencies with:")
    print("  pip install torch gymnasium numpy")
    sys.exit(1)


class ExpertDataset(Dataset):
    """Dataset of expert demonstrations."""

    def __init__(self, data_path: str):
        """Load expert data from .npz file."""
        print(f"📂 Loading expert data from: {data_path}")
        data = np.load(data_path)

        self.observations = torch.FloatTensor(data['observations'])
        self.actions = torch.FloatTensor(data['actions'])

        print(f"   Observations shape: {self.observations.shape}")
        print(f"   Actions shape: {self.actions.shape}")
        print(f"   Total transitions: {len(self.observations):,}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=[256, 256]):
        super().__init__()

        layers = []
        prev_size = obs_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Tanh())  # Assume normalized actions [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)

    def predict(self, obs, deterministic=True):
        """Stable-Baselines3 compatible predict interface."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(next(self.parameters()).device)
            action = self.forward(obs_tensor)
            return action.cpu().numpy(), None


def train_bc(args):
    """Train BC policy on expert demonstrations."""
    print(f"🚀 Training Behavior Cloning")
    print(f"   Data: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Device: {args.device}")

    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = ExpertDataset(args.data)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset):,} transitions")
    print(f"   Val:   {len(val_dataset):,} transitions")

    # Create model
    obs_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    model = BCPolicy(obs_dim, action_dim, hidden_sizes=args.hidden_sizes)
    model.to(args.device)

    print(f"\n🤖 Model architecture:")
    print(f"   Input dim: {obs_dim}")
    print(f"   Hidden: {args.hidden_sizes}")
    print(f"   Output dim: {action_dim}")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print("\n🎓 Starting training...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []

        for obs, actions in train_loader:
            obs = obs.to(args.device)
            actions = actions.to(args.device)

            pred_actions = model(obs)
            loss = criterion(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []

        with torch.no_grad():
            for obs, actions in val_loader:
                obs = obs.to(args.device)
                actions = actions.to(args.device)

                pred_actions = model(obs)
                loss = criterion(pred_actions, actions)

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1:3d}/{args.epochs}: "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), log_dir / "best_model.pth")
            print(f"   ✅ New best model saved (val_loss: {val_loss:.6f})")

    # Save final model
    torch.save(model.state_dict(), log_dir / "final_model.pth")

    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Models saved to: {log_dir}")


def evaluate_bc(args):
    """Evaluate trained BC policy."""
    if not args.checkpoint:
        print("ERROR: --checkpoint required for evaluation")
        sys.exit(1)

    print(f"🎯 Evaluating BC model: {args.checkpoint}")
    print(f"   Environment: {args.env}")
    print(f"   Episodes: {args.episodes}")

    # Create environment
    if args.env == "reach":
        from gym_env_reach import DualArmReachEnv
        env = DualArmReachEnv(reward_type="shaped")
    elif args.env == "pickplace":
        from gym_env import DualArmPickPlaceEnv
        env = DualArmPickPlaceEnv(reward_type="shaped", camera_obs=False)
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # Load model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = BCPolicy(obs_dim, action_dim, hidden_sizes=args.hidden_sizes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    model.eval()

    # Evaluate
    print("\n🏃 Running evaluation...")
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 500:
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success = info.get("is_success", False)
        successes.append(success)

        status = "✅" if success else "❌"
        print(f"Episode {ep+1:3d}: {status} Reward: {total_reward:7.2f}, Steps: {steps:3d}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    print(f"Success Rate:    {np.mean(successes)*100:6.2f}%")
    print(f"Mean Reward:     {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"Mean Length:     {np.mean(episode_lengths):8.2f} ± {np.std(episode_lengths):6.2f}")
    print(f"Min/Max Reward:  {np.min(episode_rewards):8.2f} / {np.max(episode_rewards):8.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate BC on BimanBot")

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train a new model")
    mode_group.add_argument("--eval", action="store_true", help="Evaluate a trained model")

    # Training
    parser.add_argument("--data", type=str, help="Path to expert demonstrations (.npz)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes")

    # Evaluation
    parser.add_argument("--env", type=str, default="reach", choices=["reach", "pickplace"],
                        help="Environment for evaluation")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")

    # I/O
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint (.pth)")
    parser.add_argument("--log-dir", type=str, default="logs/bc",
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
        if not args.data:
            print("ERROR: --data required for training")
            sys.exit(1)
        train_bc(args)
    else:
        evaluate_bc(args)


if __name__ == "__main__":
    main()
