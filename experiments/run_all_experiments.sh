#!/bin/bash

# Master script to run all 4 experiments
# This script runs each experiment sequentially

set -e

echo "=========================================="
echo "Starting All 4 Experiments"
echo "=========================================="

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

# Run all experiments
run_experiment "Experiment 1: DDP Baseline" "exp1_ddp_baseline/run_exp1.sh"
run_experiment "Experiment 2: DiLoCo Torch" "exp2_diloco_torch/run_exp2.sh"
run_experiment "Experiment 3: DDP from Checkpoint" "exp3_ddp_from_checkpoint/run_exp3.sh"
run_experiment "Experiment 4: DiLoCo from Checkpoint" "exp4_diloco_from_checkpoint/run_exp4.sh"

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "- Experiment 1: DDP baseline training"
echo "- Experiment 2: DiLoCo with torch.distributed"
echo "- Experiment 3: DDP training from checkpoint"
echo "- Experiment 4: DiLoCo training from checkpoint"
echo ""
echo "Check the individual experiment directories for logs and results."
