#!/usr/bin/env python3
"""
LeRobot Integration for BimanBot

Use pretrained models from HuggingFace LeRobot and fine-tune on BimanBot.

Example usage:
    # List available models
    python examples/train_with_lerobot.py --list

    # Fine-tune a Diffusion Policy
    python examples/train_with_lerobot.py --model lerobot/diffusion_pusht --train --episodes 200

    # Evaluate
    python examples/train_with_lerobot.py --model lerobot/diffusion_pusht --eval --checkpoint logs/lerobot_model.pth
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import torch
    from huggingface_hub import hf_hub_download, list_models
except ImportError:
    print("ERROR: LeRobot integration requires huggingface-hub")
    print("\nInstall with:")
    print("  pip install lerobot huggingface-hub transformers")
    sys.exit(1)


POPULAR_MODELS = [
    "lerobot/diffusion_pusht",
    "lerobot/act_aloha_mobile",
    "lerobot/tdmpc_pusht",
    "lerobot/vinn_aloha",
]


def list_lerobot_models():
    """List available LeRobot models on HuggingFace."""
    print("🤗 Available LeRobot Models")
    print("=" * 60)
    print("\nPopular pretrained models:")
    for model in POPULAR_MODELS:
        print(f"  - {model}")

    print("\n💡 Note: These models are trained on different robots/tasks.")
    print("   Fine-tuning required for BimanBot compatibility.")
    print("\nFor full list, visit: https://huggingface.co/lerobot")


def train_with_lerobot(args):
    """Fine-tune LeRobot model on BimanBot."""
    print(f"🚀 Fine-tuning LeRobot Model")
    print(f"   Model: {args.model}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Device: {args.device}")

    print("\n⚠️  Note: Full LeRobot integration is under development.")
    print("   This is a placeholder script showing the integration pattern.")
    print("\n📚 For now, you can:")
    print("   1. Use train_ppo.py for reinforcement learning")
    print("   2. Use train_bc.py for behavior cloning")
    print("   3. Export data with serialize_to_lerobot.py")
    print("\n🔗 LeRobot docs: https://github.com/huggingface/lerobot")

    # TODO: Implement actual fine-tuning
    # 1. Download pretrained model
    # 2. Adapt input/output dimensions to BimanBot
    # 3. Fine-tune on collected demonstrations
    # 4. Save adapted model


def evaluate_with_lerobot(args):
    """Evaluate fine-tuned LeRobot model."""
    if not args.checkpoint:
        print("ERROR: --checkpoint required for evaluation")
        sys.exit(1)

    print(f"🎯 Evaluating LeRobot Model")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Episodes: {args.episodes}")

    print("\n⚠️  Note: Full LeRobot evaluation is under development.")
    print("   Use train_ppo.py --eval or train_bc.py --eval for now.")


def main():
    parser = argparse.ArgumentParser(description="LeRobot integration for BimanBot")

    # Mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--list", action="store_true",
                            help="List available LeRobot models")
    mode_group.add_argument("--train", action="store_true",
                            help="Fine-tune a LeRobot model")
    mode_group.add_argument("--eval", action="store_true",
                            help="Evaluate a fine-tuned model")

    # Model
    parser.add_argument("--model", type=str,
                        help="HuggingFace model ID (e.g., lerobot/diffusion_pusht)")

    # Training
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of fine-tuning episodes")

    # Evaluation
    parser.add_argument("--checkpoint", type=str,
                        help="Model checkpoint for evaluation")

    # I/O
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

    if args.list:
        list_lerobot_models()
    elif args.train:
        if not args.model:
            print("ERROR: --model required for training")
            sys.exit(1)
        train_with_lerobot(args)
    elif args.eval:
        evaluate_with_lerobot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
