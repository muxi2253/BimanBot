# Sequential Reaching Experiments: README

## 📁 What's in this directory?

- **`run_all_experiments.sh`** - Automated script to run all 9 experiments (3 methods × 3 seeds)
- **`analyze_results.py`** - Analysis script to generate plots and statistics from TensorBoard logs
- **`experiment_log_*.txt`** - Log files from experiment runs (auto-generated)
- **`results/`** - Output directory for plots and statistics (auto-generated)

---

## 🎯 What are we testing?

**Task**: Sequential Reaching - robot must reach 3 sets of target positions in order

**Methods**:
1. **Sparse Reward** - Only get reward when completing stages
2. **Shaped Reward** - Distance-based intermediate rewards
3. **Curriculum Learning** - Start easy (1 stage), progressively add difficulty

**Hypothesis**: Curriculum learning and reward shaping will converge faster than sparse baseline.

---

## 🚀 Quick Start

```bash
# Run all experiments (8-12 hours)
./run_all_experiments.sh

# After completion, analyze results
python analyze_results.py

# View results
open results/learning_curves.png
```

See [QUICK_START_EXPERIMENTS.md](../docs/QUICK_START_EXPERIMENTS.md) for detailed instructions.

---

## 📊 Output Files

After running `analyze_results.py`, you'll get:

### 1. Learning Curves (`results/learning_curves.png`)
- Success rate vs training steps
- Shows convergence speed for each method
- Mean ± standard deviation across seeds

### 2. Bar Charts (`results/final_comparison_bars.png`)
- Final performance comparison
- Success rate and stages completed
- Error bars show variability

### 3. Statistics Table (`results/comparison_results.json`)
```json
{
  "Sparse Reward": {
    "success_rate": [0.25, 0.08],  // mean ± std
    "avg_stages": [1.5, 0.3],
    "n_seeds": 3
  },
  ...
}
```

---

## 🔧 Customization

### Change training duration
Edit `run_all_experiments.sh`:
```bash
TIMESTEPS=500000  # Change to 250000 for faster testing
```

### Change number of seeds
Edit `run_all_experiments.sh`:
```bash
SEEDS=(42 123 456 789)  # Add more seeds
```

### Analyze specific logs
```bash
python analyze_results.py --log-dir logs/ --output-dir my_results/
```

---

## 📈 Expected Results

Based on task difficulty, we expect:

| Method | Expected Success Rate | Convergence Speed |
|--------|----------------------|-------------------|
| Sparse Reward | 10-30% | Slow/Never |
| Shaped Reward | 40-60% | ~300k steps |
| Curriculum Learning | 60-80% | ~200k steps |

**Note**: These are estimates. Real results may vary ±15%.

---

## ⏱️ Time Estimates

With 4 parallel environments:
- **Per experiment**: ~1-1.5 hours (500k steps)
- **All 9 experiments**: ~8-12 hours
- **Quick test** (50k steps): ~10-15 minutes

---

## 🐛 Troubleshooting

**Problem**: Script fails with "command not found"
```bash
chmod +x run_all_experiments.sh
```

**Problem**: "No module named 'stable_baselines3'"
```bash
conda activate dual-arm
pip install stable-baselines3 torch gymnasium tensorboard matplotlib
```

**Problem**: Analysis finds no data
- Check that experiments finished: `ls -lh logs/`
- Verify TensorBoard files exist: `find logs/ -name "events.out.tfevents.*"`
- Check experiment log: `cat experiment_log_*.txt`

---

## 📝 Citation / Usage

If you use these experiments in your research:

```
Sequential Reaching Task with Hierarchical Control
Implemented using MuJoCo, Stable-Baselines3, and PPO
Methods: Baseline, Reward Shaping, Curriculum Learning
GitHub: [your-repo-url]
```

---

## 📧 Questions?

See full documentation: [docs/OPTION1_IMPLEMENTATION_PLAN.md](../docs/OPTION1_IMPLEMENTATION_PLAN.md)
