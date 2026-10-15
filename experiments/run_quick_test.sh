#!/bin/bash
################################################################################
# Quick Test Experiments (Battery-Friendly Version)
#
# Runs all 3 methods with 1 seed, 100k steps each
# Total time: ~2-3 hours (battery safe)
#
# Usage:
#   chmod +x experiments/run_quick_test.sh
#   ./experiments/run_quick_test.sh
################################################################################

set -e

# Configuration for quick test
TIMESTEPS=100000  # Reduced from 500000
N_ENVS=4
SEED=42  # Just one seed

echo "🔋 Battery-Friendly Quick Test"
echo "================================"
echo "  Timesteps: $TIMESTEPS (20% of full)"
echo "  Seed: $SEED (1 seed only)"
echo "  Estimated time: 2-3 hours"
echo ""

# Check battery
battery=$(pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%')
echo "Current battery: ${battery}%"

if [ "$battery" -lt 40 ]; then
    echo "⚠️  Warning: Battery low ($battery%)"
    echo "   This test needs ~2-3 hours"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting quick test experiments..."
echo ""

LOG_DIR="logs"
LOGFILE="experiments/quick_test_log_$(date +%Y%m%d_%H%M%S).txt"

start_time=$(date +%s)

# Method 1: Sparse Reward
echo "▶ Method 1/3: Sparse Reward"
python examples/train_sequential_baseline.py \
    --train --reward-type sparse --seed $SEED \
    --timesteps $TIMESTEPS --n-envs $N_ENVS \
    --log-dir "$LOG_DIR" \
    2>&1 | tee -a "$LOGFILE"

echo ""

# Method 2: Shaped Reward
echo "▶ Method 2/3: Shaped Reward"
python examples/train_sequential_baseline.py \
    --train --reward-type shaped --seed $SEED \
    --timesteps $TIMESTEPS --n-envs $N_ENVS \
    --log-dir "$LOG_DIR" \
    2>&1 | tee -a "$LOGFILE"

echo ""

# Method 3: Curriculum Learning
echo "▶ Method 3/3: Curriculum Learning"
python examples/train_sequential_curriculum.py \
    --train --seed $SEED \
    --timesteps $TIMESTEPS --n-envs $N_ENVS \
    --log-dir "$LOG_DIR" \
    2>&1 | tee -a "$LOGFILE"

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║         Quick Test Complete!                  ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Duration: ${hours}h ${minutes}m"
echo "Results in: $LOG_DIR/"
echo ""
echo "Next steps:"
echo "  1. Check battery: $(pmset -g batt | grep -o '[0-9]*%' | head -1)"
echo "  2. Analyze: python experiments/analyze_results.py"
echo "  3. If good, run full version when you have charger"
echo ""
