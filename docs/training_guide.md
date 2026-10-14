# 🎓 BimanBot 机器学习训练指南

完整的强化学习和模仿学习训练流程，从快速验证到生产级训练。

---

## 📋 目录

1. [环境准备](#环境准备)
2. [训练方案对比](#训练方案对比)
3. [快速开始](#快速开始)
4. [详细训练步骤](#详细训练步骤)
   - [方案 A: 行为克隆 (BC)](#方案-a-行为克隆-bc)
   - [方案 B: PPO 强化学习](#方案-b-ppo-强化学习)
   - [方案 C: LeRobot 预训练模型](#方案-c-lerobot-预训练模型)
5. [训练时间估算](#训练时间估算)
6. [监控和评估](#监控和评估)
7. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装训练依赖

```bash
# 安装核心训练库
pip install -r requirements-training.txt

# 可选：安装 LeRobot（用于预训练模型）
pip install lerobot huggingface-hub transformers
```

### 2. 验证环境

```bash
# 测试 Reach 环境
python -c "from backend.gym_env_reach import DualArmReachEnv; env = DualArmReachEnv(); print('✅ Reach env OK')"

# 测试 Pick-Place 环境
python -c "from backend.gym_env import DualArmPickPlaceEnv; env = DualArmPickPlaceEnv(); print('✅ PickPlace env OK')"
```

---

## 训练方案对比

| 方案 | 数据需求 | 训练时间 | 泛化性 | 难度 | 推荐场景 |
|---|---|---|---|---|---|
| **行为克隆 (BC)** | 500-1000 轨迹 | 几小时 | ⭐⭐ | ⭐ | 快速原型验证 |
| **PPO 强化学习** | 无需专家数据 | 1-5 小时 | ⭐⭐⭐ | ⭐⭐⭐ | 主力推荐方案 |
| **Diffusion Policy** | 1000-2000 轨迹 | 1-2 天 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 高性能需求 |
| **ACT** | 1000-3000 轨迹 | 2-3 天 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 双臂协调任务 |

---

## 快速开始

### ⚡ 5 分钟快速验证

```bash
# PPO 快速训练（Reach 任务）
python examples/train_ppo.py --train --env reach --timesteps 100000 --n-envs 8
```

**预期输出**：5 分钟后，机器人能够基本到达目标位置。

---

## 详细训练步骤

## 方案 A: 行为克隆 (BC)

**特点**：最快上手，适合快速验证管道。

### 步骤 1：收集专家数据

```bash
# 收集 500 条 Reach 任务演示（约 5-10 分钟）
python examples/collect_expert_data.py \
    --episodes 500 \
    --env reach \
    --output data/expert_reach.npz

# 收集 1000 条 Pick-Place 任务演示（约 15-20 分钟）
python examples/collect_expert_data.py \
    --episodes 1000 \
    --env pickplace \
    --domain-random \
    --output data/expert_pickplace.npz
```

**记录**：
- [ ] Reach 数据收集完成
- [ ] Pick-Place 数据收集完成
- 数据文件位置：`data/expert_*.npz`

### 步骤 2：训练 BC 模型

```bash
# 训练 Reach BC 模型（约 10-15 分钟）
python examples/train_bc.py \
    --train \
    --data data/expert_reach.npz \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --log-dir logs/bc_reach

# 训练 Pick-Place BC 模型（约 20-30 分钟）
python examples/train_bc.py \
    --train \
    --data data/expert_pickplace.npz \
    --epochs 150 \
    --batch-size 128 \
    --lr 0.0005 \
    --log-dir logs/bc_pickplace
```

**记录**：
- [ ] Reach BC 训练完成
- [ ] Pick-Place BC 训练完成
- 最佳验证损失：________
- 模型位置：`logs/bc_*/best_model.pth`

### 步骤 3：评估 BC 模型

```bash
# 评估 Reach BC 模型
python examples/train_bc.py \
    --eval \
    --checkpoint logs/bc_reach/best_model.pth \
    --env reach \
    --episodes 50

# 评估 Pick-Place BC 模型
python examples/train_bc.py \
    --eval \
    --checkpoint logs/bc_pickplace/best_model.pth \
    --env pickplace \
    --episodes 50
```

**记录**：
- Reach 成功率：______%
- Pick-Place 成功率：______%
- 平均奖励：________

---

## 方案 B: PPO 强化学习

**特点**：无需专家数据，自主探索学习，推荐用于主力训练。

### 步骤 1：快速验证（10 分钟）

```bash
# Reach 任务快速验证
python examples/train_ppo.py \
    --train \
    --env reach \
    --timesteps 200000 \
    --n-envs 8 \
    --log-dir logs/ppo_reach_quick
```

**记录**：
- [ ] 快速验证完成
- 最终奖励：________
- 训练时间：________ 分钟

### 步骤 2：标准训练（30-60 分钟）

```bash
# Reach 任务标准训练
python examples/train_ppo.py \
    --train \
    --env reach \
    --timesteps 1000000 \
    --n-envs 4 \
    --save-freq 50000 \
    --eval-freq 10000 \
    --log-dir logs/ppo_reach
```

**实时监控**（新终端窗口）：
```bash
tensorboard --logdir logs/ppo_reach
# 浏览器打开 http://localhost:6006
```

**记录**：
- [ ] 标准训练完成
- 最佳奖励：________
- 成功率：______%
- 训练时间：________ 小时

### 步骤 3：完整训练（3-5 小时，过夜运行）

```bash
# Pick-Place 任务完整训练（后台运行）
nohup python examples/train_ppo.py \
    --train \
    --env pickplace \
    --timesteps 5000000 \
    --n-envs 8 \
    --domain-random \
    --save-freq 100000 \
    --eval-freq 50000 \
    --log-dir logs/ppo_pickplace \
    > logs/training_pickplace.log 2>&1 &

# 查看训练进度
tail -f logs/training_pickplace.log
```

**记录**：
- [ ] 完整训练完成
- 最佳奖励：________
- 成功率：______%
- 训练时间：________ 小时
- 模型位置：`logs/ppo_pickplace/best_model.zip`

### 步骤 4：评估 PPO 模型

```bash
# 评估 Reach PPO 模型
python examples/train_ppo.py \
    --eval \
    --checkpoint logs/ppo_reach/best_model.zip \
    --env reach \
    --episodes 100

# 评估 Pick-Place PPO 模型
python examples/train_ppo.py \
    --eval \
    --checkpoint logs/ppo_pickplace/best_model.zip \
    --env pickplace \
    --episodes 100
```

**记录**：
- Reach 成功率：______%
- Pick-Place 成功率：______%
- 平均奖励：________
- 平均步数：________

### 步骤 5：从检查点恢复训练（可选）

如果训练中断：

```bash
# 恢复 Reach 训练
python examples/train_ppo.py \
    --train \
    --env reach \
    --checkpoint logs/ppo_reach/checkpoints/ppo_model_500000_steps.zip \
    --timesteps 1000000 \
    --n-envs 4 \
    --log-dir logs/ppo_reach

# 恢复 Pick-Place 训练
python examples/train_ppo.py \
    --train \
    --env pickplace \
    --checkpoint logs/ppo_pickplace/checkpoints/ppo_model_2000000_steps.zip \
    --timesteps 5000000 \
    --n-envs 8 \
    --domain-random \
    --log-dir logs/ppo_pickplace
```

---

## 方案 C: LeRobot 预训练模型

**特点**：使用 HuggingFace 预训练模型，最前沿方法。

### 步骤 1：列出可用模型

```bash
python examples/train_with_lerobot.py --list
```

**可用模型**：
- `lerobot/diffusion_pusht` - Diffusion Policy
- `lerobot/act_aloha_mobile` - ACT（双臂机器人）
- `lerobot/tdmpc_pusht` - TD-MPC
- `lerobot/vinn_aloha` - VINN

### 步骤 2：微调预训练模型

```bash
# 微调 Diffusion Policy（示例，需要完整实现）
python examples/train_with_lerobot.py \
    --model lerobot/diffusion_pusht \
    --train \
    --episodes 200
```

**记录**：
- [ ] LeRobot 集成完成
- 使用模型：________
- 微调轮数：________

**注意**：LeRobot 集成仍在开发中，建议先使用 PPO 或 BC。

---

## 训练时间估算

### Reach 任务（100 万步）

| 硬件 | 并行环境 | 预计时间 |
|---|---|---|
| M1/M2 Mac | 4 | 30-50 分钟 |
| M1/M2 Mac | 8 | 15-25 分钟 |
| Intel i7 CPU | 4 | 50-90 分钟 |
| RTX 3080 GPU | 8 | 15-20 分钟 |
| A100 GPU | 16 | 5-10 分钟 |

### Pick-Place 任务（500 万步）

| 硬件 | 并行环境 | 预计时间 |
|---|---|---|
| M1/M2 Mac | 4 | 3-5 小时 |
| M1/M2 Mac | 8 | 1.5-2.5 小时 |
| Intel i7 CPU | 4 | 4-7 小时 |
| RTX 3080 GPU | 8 | 1-2 小时 |
| A100 GPU | 16 | 30-60 分钟 |

### 加速技巧

1. **增加并行环境数**：
   ```bash
   # 从 4 个增加到 8 个（速度翻倍）
   --n-envs 8
   ```

2. **使用 GPU**（NVIDIA GPU）：
   ```bash
   --device cuda
   ```

3. **减少评估频率**：
   ```bash
   --eval-freq 50000  # 默认 10000
   ```

4. **后台运行**：
   ```bash
   nohup python examples/train_ppo.py ... > training.log 2>&1 &
   ```

---

## 监控和评估

### 1. TensorBoard 实时监控

```bash
# 启动 TensorBoard
tensorboard --logdir logs

# 打开浏览器
open http://localhost:6006
```

**关键指标**：
- `rollout/ep_rew_mean` - 平均奖励（越高越好）
- `rollout/ep_len_mean` - 平均步数
- `time/fps` - 训练速度（每秒步数）
- `train/value_loss` - 价值损失
- `train/policy_loss` - 策略损失

### 2. 查看训练日志

```bash
# 查看实时日志
tail -f logs/training_pickplace.log

# 搜索关键信息
grep "ep_rew_mean" logs/training_pickplace.log
```

### 3. 评估模型性能

```bash
# 评估最佳模型
python examples/train_ppo.py \
    --eval \
    --checkpoint logs/ppo_pickplace/best_model.zip \
    --env pickplace \
    --episodes 100
```

### 4. 可视化评估

```bash
# 在 GUI 中运行训练好的策略（需要实现）
python tools/record_demo.py \
    --policy logs/ppo_pickplace/best_model.zip \
    --episodes 10 \
    --output recordings/trained_policy.mp4
```

---

## 常见问题

### Q1: 训练过程中断了怎么办？

**答**：PPO 会自动保存检查点，使用 `--checkpoint` 恢复：

```bash
python examples/train_ppo.py \
    --train \
    --checkpoint logs/ppo_reach/checkpoints/ppo_model_500000_steps.zip \
    --timesteps 1000000
```

### Q2: 训练速度太慢怎么办？

**答**：增加并行环境数或减少评估频率：

```bash
python examples/train_ppo.py \
    --train \
    --n-envs 8 \
    --eval-freq 50000
```

### Q3: 奖励不收敛怎么办？

**答**：可能需要调整超参数：

```bash
python examples/train_ppo.py \
    --train \
    --lr 0.0001 \
    --n-steps 4096 \
    --batch-size 128
```

### Q4: 内存不足怎么办？

**答**：减少并行环境数：

```bash
python examples/train_ppo.py \
    --train \
    --n-envs 2  # 减少到 2 个
```

### Q5: 如何选择训练方案？

**答**：
- **快速验证**：BC（1 小时内见效）
- **日常使用**：PPO Reach（1 小时，性能好）
- **生产部署**：PPO Pick-Place + Domain Randomization（过夜训练，最强泛化）

### Q6: 训练好的模型如何部署？

**答**：加载模型并在环境中运行：

```python
from stable_baselines3 import PPO
model = PPO.load("logs/ppo_pickplace/best_model.zip")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

---

## 🎯 推荐训练流程

### 第一天：快速验证
- [ ] 方案 A 步骤 1-3：BC 快速验证（1 小时）
- [ ] 方案 B 步骤 1-2：PPO 标准训练（1 小时）

### 第二天：完整训练
- [ ] 方案 B 步骤 3：PPO 完整训练（过夜）

### 第三天：评估优化
- [ ] 方案 B 步骤 4：全面评估
- [ ] 调整超参数重新训练（可选）

---

## 📊 训练记录表

| 日期 | 方案 | 环境 | 总步数 | 训练时间 | 成功率 | 平均奖励 | 备注 |
|---|---|---|---|---|---|---|---|
| 2026-10-13 | BC | reach | - | - | - | - | |
| 2026-10-13 | PPO | reach | 1M | - | - | - | |
| 2026-10-13 | PPO | pickplace | 5M | - | - | - | |

---

## 🔗 相关资源

- **Stable-Baselines3 文档**: https://stable-baselines3.readthedocs.io/
- **LeRobot GitHub**: https://github.com/huggingface/lerobot
- **PPO 论文**: https://arxiv.org/abs/1707.06347
- **Diffusion Policy 论文**: https://arxiv.org/abs/2303.04137

---

## 📝 下一步

完成训练后，你可以：

1. **导出为 LeRobot 格式**：
   ```bash
   python tools/serialize_to_lerobot.py
   ```

2. **部署到真实机器人**：
   - 加载训练好的模型
   - 在真实环境中测试
   - 微调以适应 sim-to-real gap

3. **收集更多数据**：
   - 使用训练好的策略收集更多演示
   - 迭代训练提升性能

---

**祝训练顺利！🚀**
