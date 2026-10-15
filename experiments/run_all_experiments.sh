#!/bin/bash
################################################################################
# Complete Experimental Suite for Sequential Reaching
#
# Runs all 3 methods with 3 random seeds each:
#   - Method 1: Sparse Reward Baseline
#   - Method 2: Shaped Reward Baseline
#   - Method 3: Curriculum Learning
#
# Total: 9 training runs (3 methods × 3 seeds)
# Estimated time: 8-12 hours (can run overnight)
#
# Usage:
#   chmod +x experiments/run_all_experiments.sh
#   ./experiments/run_all_experiments.sh
################################################################################

set -e  # Exit on error

# Configuration
TIMESTEPS=500000
N_ENVS=4
SEEDS=(42 123 456)
LOG_DIR="logs"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Sequential Reaching: Complete Experimental Suite        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Timesteps per run: $(printf "%'d" $TIMESTEPS)"
echo "  Parallel envs: $N_ENVS"
echo "  Seeds: ${SEEDS[@]}"
echo "  Total experiments: $((${#SEEDS[@]} * 3))"
echo ""
echo "Estimated time:"
echo "  Per experiment: ~1-1.5 hours"
echo "  Total: ~8-12 hours"
echo ""

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Warning: No conda environment detected${NC}"
    echo "   Recommended: conda activate dual-arm"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if required packages are available
echo "Checking dependencies..."
python -c "import stable_baselines3; import torch; import gymnasium" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Error: Required packages not found${NC}"
    echo "   Install with: pip install stable-baselines3 torch gymnasium"
    exit 1
fi
echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOGFILE="experiments/experiment_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Experiment started at $(date)" > "$LOGFILE"

# Function to run single experiment
run_experiment() {
    local method=$1
    local script=$2
    local extra_args=$3
    local seed=$4

    echo -e "${BLUE}▶ Starting: $method (seed $seed)${NC}"
    echo "  Command: python $script $extra_args --seed $seed --timesteps $TIMESTEPS --n-envs $N_ENVS"

    # Run experiment and capture output
    if python "$script" $extra_args \
        --train \
        --seed "$seed" \
        --timesteps $TIMESTEPS \
        --n-envs $N_ENVS \
        --log-dir "$LOG_DIR" \
        2>&1 | tee -a "$LOGFILE"; then
        echo -e "${GREEN}✓ Complete: $method (seed $seed)${NC}"
        return 0
    else
        echo -e "${YELLOW}✗ Failed: $method (seed $seed)${NC}"
        return 1
    fi
}

# Main experiment loop
start_time=$(date +%s)
total_experiments=0
successful_experiments=0
failed_experiments=0

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Running experiments with seed: $seed"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # Method 1: Sparse Reward
    total_experiments=$((total_experiments + 1))
    if run_experiment \
        "Sparse Reward" \
        "examples/train_sequential_baseline.py" \
        "--reward-type sparse" \
        "$seed"; then
        successful_experiments=$((successful_experiments + 1))
    else
        failed_experiments=$((failed_experiments + 1))
    fi
    echo ""

    # Method 2: Shaped Reward
    total_experiments=$((total_experiments + 1))
    if run_experiment \
        "Shaped Reward" \
        "examples/train_sequential_baseline.py" \
        "--reward-type shaped" \
        "$seed"; then
        successful_experiments=$((successful_experiments + 1))
    else
        failed_experiments=$((failed_experiments + 1))
    fi
    echo ""

    # Method 3: Curriculum Learning
    total_experiments=$((total_experiments + 1))
    if run_experiment \
        "Curriculum Learning" \
        "examples/train_sequential_curriculum.py" \
        "" \
        "$seed"; then
        successful_experiments=$((successful_experiments + 1))
    else
        failed_experiments=$((failed_experiments + 1))
    fi
    echo ""

    echo -e "${GREEN}✓ Seed $seed batch complete${NC}"
    echo ""

    # Intermediate summary
    echo "Progress: $successful_experiments/$total_experiments experiments successful"
    echo ""
done

# Final summary
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT COMPLETE                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  Total experiments: $total_experiments"
echo "  Successful: $successful_experiments"
echo "  Failed: $failed_experiments"
echo "  Duration: ${hours}h ${minutes}m"
echo ""
echo "Results saved in:"
echo "  📂 Models: $LOG_DIR/"
echo "  📊 TensorBoard: tensorboard --logdir $LOG_DIR"
echo "  📝 Log file: $LOGFILE"
echo ""

if [ $failed_experiments -eq 0 ]; then
    echo -e "${GREEN}🎉 All experiments completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results: python experiments/analyze_results.py"
    echo "  2. Generate plots: python experiments/plot_results.py"
    echo "  3. View TensorBoard: tensorboard --logdir $LOG_DIR"
else
    echo -e "${YELLOW}⚠️  Some experiments failed. Check $LOGFILE for details.${NC}"
fi

echo ""
echo "Experiment completed at $(date)" >> "$LOGFILE"
