# 🚀 快速测试运行检查清单

## ✅ 运行前准备

### 1. 电池优化
- [ ] 屏幕亮度降到最低
- [ ] 关闭Chrome/Safari浏览器
- [ ] 关闭Slack/微信/其他聊天软件
- [ ] 关闭Spotify/音乐播放器
- [ ] 关闭蓝牙（如果不需要）
- [ ] 开启"低电量模式"（系统设置 → 电池）

### 2. 环境检查
```bash
# 当前电量
pmset -g batt

# 激活环境
conda activate dual-arm

# 验证依赖
python -c "import stable_baselines3, torch, gymnasium; print('✓ 依赖OK')"

# 进入目录
cd /Users/catherine/Developer/BimanBot
```

### 3. 开始运行
```bash
# 使用caffeinate防止休眠
caffeinate -i ./experiments/run_quick_test.sh
```

---

## ⏰ 预计时间线

**当前时间**：约20:00（假设）

| 时间 | 事件 | 电池预估 |
|------|------|---------|
| 20:00 | 开始运行 | 46% |
| 20:40 | 方法1完成（Sparse） | 40% |
| 21:20 | 方法2完成（Shaped） | 34% |
| 22:00 | 方法3完成（Curriculum） | 28% |
| 22:05 | 运行分析脚本 | 27% |
| 22:10 | 查看结果 | 27% |

**预计完成时间**：2-2.5小时后  
**预计剩余电量**：25-30%

---

## 📊 实时监控

### 在另一个终端窗口运行（可选）
```bash
# 每5分钟显示一次电池状态
while true; do 
    date "+%H:%M"
    pmset -g batt | grep -o '[0-9]*%' | head -1
    echo "---"
    sleep 300
done
```

### 查看训练进度
```bash
# 用TensorBoard监控（可选，会耗一点电）
tensorboard --logdir logs/
# 浏览器打开：http://localhost:6006
```

---

## 🚨 应急预案

### 如果电量降到20%
```bash
# Ctrl+C 停止当前实验
# 已完成的实验会保存，可以后续继续
```

### 如果中途需要停止
```bash
# Ctrl+C 优雅停止
# 检查已完成的部分
ls -lh logs/
```

### 恢复中断的实验
```bash
# 查看日志确认哪些完成了
cat experiments/quick_test_log_*.txt

# 单独运行未完成的方法
python examples/train_sequential_baseline.py \
    --train --reward-type [sparse/shaped] --seed 42 \
    --timesteps 100000 --n-envs 4
```

---

## ✅ 完成后检查

### 1. 确认所有实验完成
```bash
ls -lh logs/sequential_*/seed_42/
# 应该看到3个目录，每个都有best_model.zip
```

### 2. 运行分析
```bash
python experiments/analyze_results.py
```

### 3. 查看结果
```bash
open experiments/results/learning_curves.png
open experiments/results/final_comparison_bars.png
cat experiments/results/comparison_results.json
```

### 4. 检查最终电量
```bash
pmset -g batt
```

---

## 📝 记录结果

在这里记录你的实际数据：

**开始时间**：__________  
**结束时间**：__________  
**总耗时**：__________  

**电量变化**：
- 开始：46%
- 结束：_____% 

**成功率**：
- Sparse Reward: _____% 
- Shaped Reward: _____%
- Curriculum: _____%

**观察**：
- 哪种方法最好：__________
- 是否看到明显改进：__________
- 有没有unexpected情况：__________

---

## 🎯 运行命令（复制粘贴）

```bash
# 一键完整命令
cd /Users/catherine/Developer/BimanBot && \
conda activate dual-arm && \
caffeinate -i ./experiments/run_quick_test.sh
```

---

## 💪 加油！

- ✅ 代码已经全部实现
- ✅ 脚本已经测试过
- ✅ 只需要等待2-3小时
- ✅ 完成后就有真实数据了！

**祝实验顺利！** 🚀

---

**创建时间**：2024-10-15  
**当前电量**：46%  
**预计完成**：2-3小时后
