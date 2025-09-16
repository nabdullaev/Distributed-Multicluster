#!/bin/bash

# Experiment 3: First train t steps with lower batch_size (1 GPU), then DDP from checkpoint
# This involves two phases:
# Phase 1: Single GPU training for t steps
# Phase 2: DDP training from the checkpoint

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=3,4,5
export PYTHONPATH=/home/nabdullaev/Distributed-Multicluster
export PATH="/home/nabdullaev/Distributed-Multicluster/.env-open-diloco/bin:$PATH"
export WANDB_MODE=online
export MASTER_PORT=29503

# Experiment parameters
PHASE1_STEPS=25  # Train for 25 steps on single GPU
PHASE2_STEPS=25  # Then train for 25 more steps with DDP
PER_DEVICE_BATCH_SIZE=4
TOTAL_BATCH_SIZE=16
LR=1e-2
MODEL_PATH="/home/nabdullaev/Distributed-Multicluster/tests/models/llama-2m-fresh"
PROJECT="exp3_ddp_from_checkpoint"
CHECKPOINT_DIR="./checkpoints/exp3_phase1"

echo "Starting Experiment 3: DDP training from checkpoint"
echo "Phase 1: Single GPU training for $PHASE1_STEPS steps"
echo "Phase 2: DDP training for $PHASE2_STEPS steps from checkpoint"

# Phase 1: Single GPU training
echo "=== Phase 1: Single GPU Training ==="
export CUDA_VISIBLE_DEVICES=6
export MASTER_PORT=29503

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

# Phase 2: DDP training from checkpoint
echo "=== Phase 2: DDP Training from Checkpoint ==="
export CUDA_VISIBLE_DEVICES=3,4,5
export MASTER_PORT=29504

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find $CHECKPOINT_DIR -name "model_step_*" -type d | sort -V | tail -1)
echo "Resuming from checkpoint: $LATEST_CHECKPOINT"

torchrun \
    --standalone \
    --nproc_per_node=2 \
    ../../open_diloco/train_fsdp.py \
    --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
    --total-batch-size $TOTAL_BATCH_SIZE \
    --lr $LR \
    --path-model $MODEL_PATH \
    --project "${PROJECT}_phase2" \
    --no-torch-compile \
    --fake-data \
    --max-steps $PHASE2_STEPS \
    --metric-logger-type wandb \
    --precision fp16-mixed \
    --sharding-strategy NO_SHARD \
    --ckpt.path $CHECKPOINT_DIR \
    --ckpt.resume true

echo "Experiment 3 completed!"
