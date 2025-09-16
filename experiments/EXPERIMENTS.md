# Experiments

## Prerequisites

```bash
# Activate environment
source /home/nabdullaev/Distributed-Multicluster/.env-open-diloco/bin/activate

# Set environment
export PYTHONPATH=/home/nabdullaev/Distributed-Multicluster
export WANDB_MODE=online  # Changed to online for proper wandb logging
```

## Experiment 1: DDP Baseline

**Description**: Standard DDP training, sync every step

```bash
cd experiments/exp1_ddp_baseline
./run_exp1.sh
```

**What it does**:
- 2 GPUs (6,7)
- 50 steps
- Batch size: 4 per device, 16 total
- Sync: Every step
- Uses wandb for logging

## Experiment 2: DiLoCo Torch

**Description**: DiLoCo training, sync every 5 steps

```bash
cd experiments/exp2_diloco_torch
./run_exp2.sh
```

**What it does**:
- 2 GPUs (6,7)
- 50 steps
- Batch size: 4 per device, 16 total
- Sync: Every 5 steps (DiLoCo)
- Local steps: 5
- Outer LR: 0.7
- Uses wandb for logging

## Experiment 3: DDP from Checkpoint

**Description**: Single GPU training (25 steps) → DDP training (25 steps)

```bash
cd experiments/exp3_ddp_from_checkpoint
./run_exp3.sh
```

**What it does**:
- Phase 1: 1 GPU (6), 25 steps, batch size 4
- Phase 2: 2 GPUs (3,4,5), 25 steps, batch size 4 per device
- Checkpoint saved between phases
- **Resume functionality**: Automatically resumes from checkpoint in phase 2
- **Wandb continuity**: Continues logging to the same wandb run across phases

## Experiment 4: DiLoCo from Checkpoint

**Description**: Single GPU training (25 steps) → DiLoCo training (25 steps)

```bash
cd experiments/exp4_diloco_from_checkpoint
./run_exp4.sh
```

**What it does**:
- Phase 1: 1 GPU (6), 25 steps, batch size 4
- Phase 2: 2 GPUs (3,4,5), 25 steps, DiLoCo sync every 5 steps
- Checkpoint saved between phases
- **Resume functionality**: Automatically resumes from checkpoint in phase 2
- **Wandb continuity**: Continues logging to the same wandb run across phases

## Stop Running Processes

```bash
# Kill all training processes
pkill -f torchrun
pkill -f train_diloco_torch_simple
pkill -f train_fsdp

# Verify nothing is running
ps aux | grep -E "(torchrun|train_diloco|train_fsdp)" | grep -v grep
```

## Expected Output

All experiments should:
- Start with "Starting Experiment X"
- Show step progress: "step: 1, loss: 6.94, lr: 1e-06"
- Show DiLoCo sync: "perform outer step at step 5" (Experiments 2 & 4 only)
- Stop at configured steps: "Reached step limit (X), stopping training"
- End with "Training completed"

**Experiments 3 & 4**: Two-phase training with checkpoint saving between phases
- Phase 1: Shows "Phase 1 completed. Checkpoint saved."
- Phase 2: Shows "Resumed from checkpoint at step 25 with loss X.XX"
- **Wandb logging**: All experiments now use wandb for proper logging and resume functionality