#!/bin/bash
################################################################################
# Plan B Validation Test
# Tests the fixed environment: 20cm separation + 10x reward + 800 max steps
################################################################################

set -e

echo "🔧 Plan B Validation Test"
echo "=========================="
echo "Changes:"
echo "  - Target separation: 60cm → 20cm"
echo "  - Reward strength: 10x increase"
echo "  - Max steps: 1500 → 800"
echo ""

# Find Python with gymnasium
if command -v python3 &> /dev/null; then
    if python3 -c "import gymnasium" 2>/dev/null; then
        PYTHON=python3
    fi
fi

if [ -z "$PYTHON" ]; then
    # Try conda environments
    if [ -d "/Users/catherine/miniforge3/envs/lerobot" ]; then
        source /Users/catherine/miniforge3/bin/activate lerobot
        PYTHON=python3
    elif [ -d "/Users/catherine/miniforge3" ]; then
        source /Users/catherine/miniforge3/bin/activate base
        PYTHON=python3
    fi
fi

if [ -z "$PYTHON" ]; then
    echo "ERROR: Cannot find Python with gymnasium installed"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""

# Quick validation (10k steps, shaped reward)
echo "Running quick validation (10k steps, ~15 mins)..."
$PYTHON examples/train_sequential_baseline.py \
    --train \
    --reward-type shaped \
    --seed 99 \
    --timesteps 10000 \
    --n-envs 4

echo ""
echo "✅ Validation complete!"
echo ""
echo "Check logs/sequential_shaped/seed_99/ for results"
echo "Look for avg_stages_completed > 0 in the output above"
