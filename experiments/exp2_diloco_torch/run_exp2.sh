#!/bin/bash

# Experiment 2: DiLoCo training from scratch, sync every N steps
# This uses train_diloco_torch.py (pure torch.distributed DiLoCo implementation)

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=/home/nabdullaev/Distributed-Multicluster
export PATH="/home/nabdullaev/Distributed-Multicluster/.env-open-diloco/bin:$PATH"
export WANDB_MODE=online
export MASTER_PORT=29502

# Experiment parameters
NPROC_PER_NODE=2
PER_DEVICE_BATCH_SIZE=4
TOTAL_BATCH_SIZE=16
LR=1e-2
MODEL_PATH="/home/nabdullaev/Distributed-Multicluster/tests/models/llama-2m-fresh"
MAX_STEPS=50
PROJECT="exp2_diloco_torch"
LOCAL_STEPS=5  # Sync every 5 steps
OUTER_LR=0.7

echo "Starting Experiment 2: DiLoCo training from scratch, sync every $LOCAL_STEPS steps"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Batch size per device: $PER_DEVICE_BATCH_SIZE"
echo "Total batch size: $TOTAL_BATCH_SIZE"
echo "Max steps: $MAX_STEPS"
echo "Local steps (sync frequency): $LOCAL_STEPS"

# Run the training
torchrun \
    --standalone \
    --nproc_per_node=$NPROC_PER_NODE \
    ../../open_diloco/train_diloco_torch_simple.py \
    --batch-size $TOTAL_BATCH_SIZE \
    --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
    --lr $LR \
    --model-name-or-path $MODEL_PATH \
    --project $PROJECT \
    --total-steps $MAX_STEPS \
    --local-steps $LOCAL_STEPS \
    --outer-lr $OUTER_LR \
    --precision fp16-mixed \
    --fake-data \
    --metric-logger-type wandb

echo "Experiment 2 completed!"
