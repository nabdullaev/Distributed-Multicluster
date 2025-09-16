#!/bin/bash

# Experiment 1: DDP training from zero, sync every step
# This uses train_fsdp.py with no Hivemind configuration

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=/home/nabdullaev/Distributed-Multicluster
export PATH="/home/nabdullaev/Distributed-Multicluster/.env-open-diloco/bin:$PATH"
export WANDB_MODE=online
export MASTER_PORT=29501

# Experiment parameters
NPROC_PER_NODE=2
PER_DEVICE_BATCH_SIZE=4
TOTAL_BATCH_SIZE=16
LR=1e-2
MODEL_PATH="/home/nabdullaev/Distributed-Multicluster/tests/models/llama-2m-fresh"
MAX_STEPS=50
PROJECT="exp1_ddp_baseline"

echo "Starting Experiment 1: DDP training from zero, sync every step"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Batch size per device: $PER_DEVICE_BATCH_SIZE"
echo "Total batch size: $TOTAL_BATCH_SIZE"
echo "Max steps: $MAX_STEPS"

# Run the training
torchrun \
    --standalone \
    --nproc_per_node=$NPROC_PER_NODE \
    ../../open_diloco/train_fsdp.py \
    --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
    --total-batch-size $TOTAL_BATCH_SIZE \
    --lr $LR \
    --path-model $MODEL_PATH \
    --project $PROJECT \
    --no-torch-compile \
    --fake-data \
    --max-steps $MAX_STEPS \
    --metric-logger-type wandb \
    --precision fp16-mixed \
    --sharding-strategy NO_SHARD

echo "Experiment 1 completed!"
