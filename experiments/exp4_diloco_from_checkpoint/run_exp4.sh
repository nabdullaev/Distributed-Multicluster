#!/bin/bash

# Experiment 4: First train t steps with lower batch_size (1 GPU), then DiLoCo from checkpoint
# This involves two phases:
# Phase 1: Single GPU training for t steps
# Phase 2: DiLoCo training from the checkpoint

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=3,4,5
export PYTHONPATH=/home/nabdullaev/Distributed-Multicluster
export PATH="/home/nabdullaev/Distributed-Multicluster/.env-open-diloco/bin:$PATH"
export WANDB_MODE=online
export MASTER_PORT=29505

# Experiment parameters
PHASE1_STEPS=25  # Train for 25 steps on single GPU
PHASE2_STEPS=25  # Then train for 25 more steps with DiLoCo
PER_DEVICE_BATCH_SIZE=4
TOTAL_BATCH_SIZE=16
LR=1e-2
MODEL_PATH="/home/nabdullaev/Distributed-Multicluster/tests/models/llama-2m-fresh"
PROJECT="exp4_diloco_from_checkpoint"
CHECKPOINT_DIR="./checkpoints/exp4_phase1"
LOCAL_STEPS=5  # Sync every 5 steps
OUTER_LR=0.7

echo "Starting Experiment 4: DiLoCo training from checkpoint"
echo "Phase 1: Single GPU training for $PHASE1_STEPS steps"
echo "Phase 2: DiLoCo training for $PHASE2_STEPS steps from checkpoint"

# Phase 1: Single GPU training
echo "=== Phase 1: Single GPU Training ==="
export CUDA_VISIBLE_DEVICES=6
export MASTER_PORT=29505

torchrun \
    --standalone \
    --nproc_per_node=1 \
    ../../open_diloco/train_fsdp.py \
    --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
    --total-batch-size $PER_DEVICE_BATCH_SIZE \
    --lr $LR \
    --path-model $MODEL_PATH \
    --project "${PROJECT}_phase1" \
    --no-torch-compile \
    --fake-data \
    --max-steps $PHASE1_STEPS \
    --metric-logger-type wandb \
    --precision fp16-mixed \
    --sharding-strategy NO_SHARD \
    --ckpt.interval 25 \
    --ckpt.path $CHECKPOINT_DIR

echo "Phase 1 completed. Checkpoint saved."

# Phase 2: DiLoCo training from checkpoint
echo "=== Phase 2: DiLoCo Training from Checkpoint ==="
export CUDA_VISIBLE_DEVICES=3,4,5
export MASTER_PORT=29506

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find $CHECKPOINT_DIR -name "model_step_*" -type d | sort -V | tail -1)
echo "Resuming from checkpoint: $LATEST_CHECKPOINT"

torchrun \
    --standalone \
    --nproc_per_node=2 \
    ../../open_diloco/train_diloco_torch_simple.py \
    --batch-size $TOTAL_BATCH_SIZE \
    --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
    --lr $LR \
    --model-name-or-path $MODEL_PATH \
    --project "${PROJECT}_phase2" \
    --total-steps $PHASE2_STEPS \
    --local-steps $LOCAL_STEPS \
    --outer-lr $OUTER_LR \
    --precision fp16-mixed \
    --fake-data \
    --resume-from-checkpoint $LATEST_CHECKPOINT

echo "Experiment 4 completed!"
