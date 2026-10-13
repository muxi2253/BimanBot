# 🤖 Machine Learning Training Examples

Complete training examples for reinforcement learning and imitation learning with BimanBot.

## 📋 Overview

| Method | Script | Use Case | Data Needed |
|---|---|---|---|
| **PPO** (Proximal Policy Optimization) | [`train_ppo.py`](train_ppo.py) | Learn from scratch via trial-and-error | None (self-play) |
| **BC** (Behavior Cloning) | [`train_bc.py`](train_bc.py) | Clone expert demonstrations | 50-500 episodes |
| **LeRobot Pretrained** | [`train_with_lerobot.py`](train_with_lerobot.py) | Fine-tune on BimanBot from HuggingFace models | Optional |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core training libraries
pip install -r requirements-training.txt

# Optional: For LeRobot integration
pip install lerobot huggingface-hub
```

### 2. Train PPO from Scratch

```bash
# Train on Reach environment (easier, faster)
python examples/train_ppo.py \
    --env reach \
    --train \
    --timesteps 1000000 \
    --save-freq 50000

# Train on Pick-Place environment (harder, slower)
python examples/train_ppo.py \
    --env pickplace \
    --train \
    --timesteps 5000000 \
    --domain-random
```

### 3. Collect Expert Data + Train BC

```bash
# Collect demonstrations using rule-based policy
python examples/collect_expert_data.py \
    --episodes 500 \
    --output data/expert_demos.npz \
    --seed 42

# Train behavior cloning model
python examples/train_bc.py \
    --train \
    --data data/expert_demos.npz \
    --epochs 100 \
    --batch-size 64
```

### 4. Use LeRobot Pretrained Models

```bash
# List available pretrained models
python examples/train_with_lerobot.py --list

# Fine-tune a Diffusion Policy model
python examples/train_with_lerobot.py \
    --model lerobot/diffusion_pusht \
    --episodes 200 \
    --train
```

---

## 📊 Expected Results

### PPO Training Timeline

| Environment | Steps | Training Time | Success Rate |
|---|---|---|---|
| **Reach** | 500K | ~2 hours (CPU) | >90% |
| **Pick-Place** | 2-5M | ~1-2 days (CPU) | 60-80% |

### BC Training Timeline

| Data Size | Epochs | Training Time | Success Rate |
|---|---|---|---|
| 100 episodes | 50 | ~10 min | 40-60% |
| 500 episodes | 100 | ~30 min | 70-85% |
| 2000 episodes | 200 | ~2 hours | 85-95% |

> 💡 **Note**: Times are approximate on M1/M2 Mac or modern CPU. GPU training is 5-10× faster.

---

## 🔧 Training Configuration

### PPO Hyperparameters

```python
# Default config in train_ppo.py
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}
```

### BC Hyperparameters

```python
# Default config in train_bc.py
{
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "optimizer": "adam",
    "loss": "mse",
}
```

---

## 📈 Monitoring Training

### TensorBoard Logs

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Training metrics will be available at:
# http://localhost:6006
```

### Key Metrics to Watch

**PPO**:
- `rollout/ep_rew_mean` - Average episode reward (should increase)
- `rollout/success_rate` - Task completion rate
- `train/policy_loss` - Policy gradient loss
- `train/value_loss` - Value function loss

**BC**:
- `train/loss` - MSE between predicted and expert actions
- `eval/success_rate` - Performance on held-out validation set

---

## 🎯 Data Requirements Analysis

### Generalization to New SKUs

Based on the rule-based demo generator, here's data requirements for different generalization levels:

| Goal | SKUs in Training | Episodes per SKU | Total Data | Expected New-SKU Success |
|---|---|---|---|---|
| **Overfit to 6 SKUs** | 6 | 50 | 300 | <20% |
| **Moderate generalization** | 6 | 200 | 1200 | 40-60% |
| **Strong generalization** | 20-50 | 100 | 2000-5000 | 70-85% |

### Factors That Improve Generalization

1. **Domain Randomization** ✅
   - Friction: 0.5 - 2.0
   - Joint damping: 0.8× - 1.2×
   - Object mass: 0.8× - 1.2× (if implemented)

2. **Visual Observations** ✅
   - 3× RGB cameras (top + dual wrist)
   - Enables vision-based policies

3. **Data Augmentation**
   - Random object poses ✅ (via `--seed`)
   - Random lighting (TODO)
   - Camera jitter (TODO)

---

## 🛠️ Advanced Usage

### Multi-GPU Training

```bash
# Not yet implemented - coming soon
# mpirun -np 4 python examples/train_ppo.py --train --timesteps 5000000
```

### Hyperparameter Tuning with Optuna

```bash
# Not yet implemented - coming soon
# python examples/tune_hyperparams.py --trials 100
```

### Checkpoint Management

```bash
# Resume from checkpoint
python examples/train_ppo.py \
    --train \
    --checkpoint logs/ppo_reach/best_model.zip \
    --timesteps 2000000

# Evaluate trained model
python examples/train_ppo.py \
    --eval \
    --checkpoint logs/ppo_reach/best_model.zip \
    --episodes 100
```

---

## 📚 References

**Frameworks Used**:
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - PPO implementation
- [LeRobot](https://github.com/huggingface/lerobot) - Pretrained models
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard

**Key Papers**:
- PPO: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- Diffusion Policy: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- ACT: [Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)

---

## ❓ FAQ

**Q: Which algorithm should I start with?**  
A: PPO for Reach environment (fast feedback), or BC if you already have expert data.

**Q: How much data do I need for BC?**  
A: Start with 100-200 episodes. If success rate <60%, collect more data or add domain randomization.

**Q: Can I use pretrained models from other robots?**  
A: Yes via LeRobot, but you'll need to fine-tune. Different robots have different kinematics and action spaces.

**Q: Why is training slow on my Mac?**  
A: MuJoCo rendering is CPU-bound. Use `--no-render` during training and only render during evaluation.

**Q: My PPO training diverges after 500K steps. What to do?**  
A: Lower `learning_rate` to 1e-4, increase `n_steps` to 4096, or reduce `clip_range` to 0.1.

---

## 🤝 Contributing

Found a bug or have a training recipe to share? Open an issue or PR!

**Training scripts we'd love to add**:
- [ ] Diffusion Policy (native implementation)
- [ ] ACT (Action Chunking Transformer)
- [ ] DAgger (Dataset Aggregation)
- [ ] Multi-task learning across Reach + Pick-Place
- [ ] Vision-based policy with ResNet/ViT encoder
