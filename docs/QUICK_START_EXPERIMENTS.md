# Quick Start Guide: Running Sequential Reaching Experiments

## 🚀 Complete Workflow

Follow these steps to run all experiments and generate results for your CV.

---

## Step 1: Activate Environment

```bash
cd /Users/catherine/Developer/BimanBot
conda activate dual-arm
```

**Verify dependencies:**
```bash
python -c "import stable_baselines3; import torch; import gymnasium; print('✓ All dependencies OK')"
```

If this fails, install:
```bash
pip install stable-baselines3 torch gymnasium tensorboard matplotlib
```

---

## Step 2: Run All Experiments (Overnight)

### Option A: Automated Batch Script (Recommended)

```bash
# Make script executable (first time only)
chmod +x experiments/run_all_experiments.sh

# Run all experiments
./experiments/run_all_experiments.sh
```

This will:
- Run 3 methods × 3 seeds = 9 experiments
- Take ~8-12 hours total
- Save all logs to `logs/` directory
- Log progress to `experiments/experiment_log_*.txt`

**Tip**: Run overnight, then check results in the morning!

---

### Option B: Manual Execution

If you want more control, run experiments individually:

```bash
# Method 1: Sparse Reward (3 seeds)
for seed in 42 123 456; do
    python examples/train_sequential_baseline.py \
        --train --reward-type sparse --seed $seed \
        --timesteps 500000 --n-envs 4
done

# Method 2: Shaped Reward (3 seeds)
for seed in 42 123 456; do
    python examples/train_sequential_baseline.py \
        --train --reward-type shaped --seed $seed \
        --timesteps 500000 --n-envs 4
done

# Method 3: Curriculum Learning (3 seeds)
for seed in 42 123 456; do
    python examples/train_sequential_curriculum.py \
        --train --seed $seed \
        --timesteps 500000 --n-envs 4
done
```

---

## Step 3: Monitor Training (Optional)

While experiments are running:

```bash
# In a new terminal
conda activate dual-arm
tensorboard --logdir logs/

# Open browser: http://localhost:6006
```

You'll see real-time learning curves for all methods.

---

## Step 4: Analyze Results

After all experiments finish:

```bash
python experiments/analyze_results.py
```

This generates:
- **Learning curves**: `experiments/results/learning_curves.png`
- **Bar charts**: `experiments/results/final_comparison_bars.png`
- **Statistics table**: `experiments/results/comparison_results.json`

---

## Step 5: Review Results

Check the generated files:

```bash
# View images (macOS)
open experiments/results/learning_curves.png
open experiments/results/final_comparison_bars.png

# View statistics
cat experiments/results/comparison_results.json
```

---

## Expected Timeline

| Phase | Time | When |
|-------|------|------|
| **Setup** | 10 min | Now |
| **Start experiments** | 2 min | Today evening |
| **Training (automated)** | 8-12 hours | Overnight |
| **Analysis** | 5 min | Tomorrow morning |
| **Review results** | 30 min | Tomorrow |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'stable_baselines3'"

**Solution:**
```bash
pip install stable-baselines3 torch gymnasium
```

---

### Issue: Experiments are slow

**Solution:**
- Check if GPU is being used: `nvidia-smi` (if you have GPU)
- Reduce `--timesteps` for faster pilot test:
  ```bash
  ./experiments/run_all_experiments.sh  # Edit TIMESTEPS=100000
  ```

---

### Issue: Out of disk space

**Solution:**
- Each experiment uses ~100-500 MB
- Free up space: `du -sh logs/`
- Delete old logs: `rm -rf logs/old_experiments/`

---

### Issue: Script interrupted

**Solution:**
- Experiments are independent - you can re-run failed ones
- Check log file: `cat experiments/experiment_log_*.txt`
- Re-run specific seed:
  ```bash
  python examples/train_sequential_baseline.py \
      --train --reward-type sparse --seed 42 \
      --timesteps 500000 --n-envs 4
  ```

---

## Quick Test (5 minutes)

Before running overnight, test that everything works:

```bash
# Quick 50k step test
python examples/train_sequential_baseline.py \
    --train --reward-type shaped --seed 42 \
    --timesteps 50000 --n-envs 4
```

If this completes successfully (~10-15 min), you're ready for full experiments!

---

## Next Steps After Results

Once you have results, you can:

1. **Write CV description** with real numbers
2. **Generate additional plots** (customize `analyze_results.py`)
3. **Run more seeds** for stronger statistics
4. **Try different hyperparameters**

---

## File Structure After Experiments

```
BimanBot/
├── logs/
│   ├── sequential_sparse/
│   │   ├── seed_42/
│   │   │   ├── best_model.zip
│   │   │   └── events.out.tfevents.*
│   │   ├── seed_123/
│   │   └── seed_456/
│   ├── sequential_shaped/
│   │   └── seed_*/...
│   └── sequential_curriculum/
│       └── seed_*/...
│
├── experiments/
│   ├── run_all_experiments.sh
│   ├── analyze_results.py
│   ├── experiment_log_20241015_*.txt
│   └── results/
│       ├── learning_curves.png
│       ├── final_comparison_bars.png
│       └── comparison_results.json
```

---

## Ready to Start?

1. Open terminal
2. Navigate to BimanBot directory
3. Activate conda environment
4. Run: `./experiments/run_all_experiments.sh`
5. Wait ~8-12 hours
6. Run: `python experiments/analyze_results.py`
7. Check results in `experiments/results/`

**Good luck! 🚀**
