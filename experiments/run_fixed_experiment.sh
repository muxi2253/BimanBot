#!/bin/bash
################################################################################
# Fixed Quick Test - After relaxing distance threshold from 0.05m to 0.10m
# This should show better learning compared to the previous run
################################################################################

set -e

LOG_FILE="experiments/fixed_experiment_log_$(date +%Y%m%d_%H%M%S).txt"
RESULTS_DIR="experiments/results"
TIMESTEPS=100000
N_ENVS=4

mkdir -p "$RESULTS_DIR"

echo "🔧 Fixed Quick Experiment - Relaxed Threshold (0.10m)" | tee "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Training steps per method: $TIMESTEPS" | tee -a "$LOG_FILE"
echo "Parallel environments: $N_ENVS" | tee -a "$LOG_FILE"
echo "Change: Distance threshold 0.05m → 0.10m" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Only run one seed to save time
SEED=42

# Method 1: Sparse reward
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "🎯 Method 1: Sparse Reward (baseline)" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
python examples/train_sequential_baseline.py \
    --train \
    --reward-type sparse \
    --seed $SEED \
    --timesteps $TIMESTEPS \
    --n-envs $N_ENVS 2>&1 | tee -a "$LOG_FILE"

# Method 2: Shaped reward
echo "" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "🎯 Method 2: Shaped Reward" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
python examples/train_sequential_baseline.py \
    --train \
    --reward-type shaped \
    --seed $SEED \
    --timesteps $TIMESTEPS \
    --n-envs $N_ENVS 2>&1 | tee -a "$LOG_FILE"

# Method 3: Curriculum learning
echo "" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "🎯 Method 3: Curriculum Learning" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
python examples/train_sequential_curriculum.py \
    --train \
    --seed $SEED \
    --timesteps $TIMESTEPS \
    --n-envs $N_ENVS 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ All methods completed!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📊 Results saved in:" | tee -a "$LOG_FILE"
echo "   - logs/sequential_sparse/seed_${SEED}/" | tee -a "$LOG_FILE"
echo "   - logs/sequential_shaped/seed_${SEED}/" | tee -a "$LOG_FILE"
echo "   - logs/sequential_curriculum/seed_${SEED}/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "  1. Run: python experiments/analyze_results.py" | tee -a "$LOG_FILE"
echo "  2. Compare with previous run to show improvement" | tee -a "$LOG_FILE"
