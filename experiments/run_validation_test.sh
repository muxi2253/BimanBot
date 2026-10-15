#!/bin/bash
################################################################################
# Validation Test - After fixing target positions (60cm -> 20cm separation)
# Quick 10k steps test to verify the fix works before running full experiment
################################################################################

set -e

LOG_FILE="experiments/validation_test_log_$(date +%Y%m%d_%H%M%S).txt"
RESULTS_DIR="experiments/results"
TIMESTEPS=10000  # Very short test
N_ENVS=4

mkdir -p "$RESULTS_DIR"

echo "🔍 Validation Test - Fixed Target Positions (20cm separation)" | tee "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Training steps: $TIMESTEPS (quick validation)" | tee -a "$LOG_FILE"
echo "Parallel environments: $N_ENVS" | tee -a "$LOG_FILE"
echo "Key change: Stage 0 arm separation 60cm → 20cm" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

SEED=42

# Test with shaped reward (easiest to see improvement)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "🎯 Testing: Shaped Reward (should show progress)" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
python examples/train_sequential_baseline.py \
    --train \
    --reward-type shaped \
    --seed $SEED \
    --timesteps $TIMESTEPS \
    --n-envs $N_ENVS 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ Validation test complete!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📊 Check the log for:" | tee -a "$LOG_FILE"
echo "   - avg_stages_completed should be > 0" | tee -a "$LOG_FILE"
echo "   - success_rate should increase from 0" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "If this works, run the full experiment:" | tee -a "$LOG_FILE"
echo "   ./experiments/run_quick_test.sh" | tee -a "$LOG_FILE"
