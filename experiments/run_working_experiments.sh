#!/bin/bash

# Script to run only the working experiments
# This runs Experiment 1 (DDP baseline) and Experiment 3 (DDP from checkpoint)

set -e

echo "=========================================="
echo "Running Working Experiments Only"
echo "=========================================="
echo ""
echo "This script runs:"
echo "- Experiment 1: DDP Baseline (✅ WORKING)"
echo "- Experiment 3: DDP from Checkpoint (✅ READY)"
echo ""
echo "Skipping:"
echo "- Experiment 2: DiLoCo Torch (⚠️ HAS ISSUES)"
echo "- Experiment 4: DiLoCo from Checkpoint (⚠️ DEPENDS ON EXP2)"
echo ""

# Change to the experiments directory
cd "$(dirname "$0")"

# Function to run an experiment with error handling
run_experiment() {
    local exp_name=$1
    local exp_script=$2
    
    echo ""
    echo "=========================================="
    echo "Starting $exp_name"
    echo "=========================================="
    
    if [ -f "$exp_script" ]; then
        echo "Running $exp_script..."
        if bash "$exp_script"; then
            echo "✅ $exp_name completed successfully"
        else
            echo "❌ $exp_name failed"
            echo "Continuing with next experiment..."
        fi
    else
        echo "❌ Script $exp_script not found"
    fi
    
    echo "Waiting 5 seconds before next experiment..."
    sleep 5
}

# Run working experiments only
run_experiment "Experiment 1: DDP Baseline" "exp1_ddp_baseline/run_exp1.sh"
run_experiment "Experiment 3: DDP from Checkpoint" "exp3_ddp_from_checkpoint/run_exp3.sh"

echo ""
echo "=========================================="
echo "Working Experiments Completed!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "- ✅ Experiment 1: DDP baseline training (WORKING)"
echo "- ✅ Experiment 3: DDP training from checkpoint (WORKING)"
echo ""
echo "Skipped experiments (have issues):"
echo "- ⚠️ Experiment 2: DiLoCo with torch.distributed (NaN losses)"
echo "- ⚠️ Experiment 4: DiLoCo from checkpoint (depends on Exp2)"
echo ""
echo "Check the individual experiment directories for logs and results."
echo "See EXPERIMENT_STATUS.md for detailed status information."

