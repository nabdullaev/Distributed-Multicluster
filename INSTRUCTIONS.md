# Distributed Multi-Cluster Training Instructions

This repository provides distributed training capabilities using both `torch.distributed` and Hivemind for multi-cluster training scenarios.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPUs
- Linux environment

## Quick Start: Torch Distributed Training

### 1. Environment Setup

```bash
# Navigate to the project directory
cd /path/to/Distributed-Multicluster

# Create a new virtual environment
python3 -m venv .env-open-diloco

# Activate the environment
source .env-open-diloco/bin/activate

# Install dependencies
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0
pip install cyclopts==2.6.1
pip install wandb==0.16.4
pip install pydantic==2.7.4
pip install hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff9
```

### 2. Basic Torch Distributed Training

```bash
# Set environment variables
export PYTHONPATH=/path/to/Distributed-Multicluster
export WANDB_MODE=disabled

# Run basic distributed training (2 GPUs, 5 steps)
torchrun --standalone --nproc_per_node=2 train_fsdp.py \
    --per-device-train-batch-size 4 \
    --total-batch-size 16 \
    --lr 1e-2 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type dummy \
    --no-torch-compile \
    --fake-data \
    --max-steps 5
```

### 3. Training on Specific GPUs

```bash
# Use specific GPUs (e.g., GPUs 6 and 7)
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 train_fsdp.py \
    --per-device-train-batch-size 4 \
    --total-batch-size 16 \
    --lr 1e-2 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type dummy \
    --no-torch-compile \
    --fake-data \
    --max-steps 5
```

### 4. Advanced Training Configuration

```bash
# Larger batch size, more steps, real data
torchrun --standalone --nproc_per_node=4 train_fsdp.py \
    --per-device-train-batch-size 8 \
    --total-batch-size 64 \
    --lr 4e-4 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type wandb \
    --torch-compile \
    --max-steps 1000 \
    --seq-length 1024 \
    --precision fp16-mixed
```

### 5. Multi-Node Training

```bash
# For multi-node training, use MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500

# On master node
torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_fsdp.py \
    --per-device-train-batch-size 4 \
    --total-batch-size 16 \
    --lr 1e-2 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type dummy \
    --no-torch-compile \
    --fake-data \
    --max-steps 5

# On worker node
torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_fsdp.py \
    --per-device-train-batch-size 4 \
    --total-batch-size 16 \
    --lr 1e-2 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type dummy \
    --no-torch-compile \
    --fake-data \
    --max-steps 5
```

## Key Parameters Explained

- `--nproc_per_node`: Number of GPUs per node
- `--nnodes`: Number of nodes in the cluster
- `--per-device-train-batch-size`: Batch size per GPU
- `--total-batch-size`: Total batch size across all GPUs
- `--lr`: Learning rate
- `--path-model`: Path to the pre-trained model
- `--metric-logger-type`: Logging backend (dummy, wandb)
- `--fake-data`: Use synthetic data for testing
- `--max-steps`: Maximum training steps
- `--seq-length`: Sequence length for language modeling
- `--precision`: Training precision (fp16-mixed, bf16-mixed, 32-true)

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change `MASTER_PORT` if you get port binding errors
2. **GPU memory**: Reduce batch size if you encounter OOM errors
3. **CUDA version mismatch**: Ensure PyTorch CUDA version matches your system

### Debug Commands

```bash
# Check GPU availability
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Kill hanging processes
pkill -f train_fsdp
```

## Performance Tips

1. **Use mixed precision** (`--precision fp16-mixed`) for faster training
2. **Enable torch.compile** for optimized execution (remove `--no-torch-compile`)
3. **Adjust batch size** based on GPU memory availability
4. **Use appropriate sharding strategy** for your model size

## Hivemind Multi-Cluster Training

Hivemind enables decentralized multi-cluster training with automatic peer discovery and weight averaging.

### 1. Start the DHT (Distributed Hash Table)

First, start a DHT instance that will coordinate the distributed training:

```bash
# Navigate to the project directory
cd /path/to/Distributed-Multicluster

# Start DHT in background
PATH="/path/to/Distributed-Multicluster/.env-open-diloco/bin:$PATH" hivemind-dht \
    --identity_path fixed_private_key.pem \
    --host_maddrs /ip4/0.0.0.0/tcp/30001 &

# Wait a moment for DHT to start, then check the peer address
sleep 3
# The DHT will output something like:
# --initial_peers /ip4/127.0.0.1/tcp/30001/p2p/Qmd5gRA2PYYQB8gZNBau6APQTCGqEGsCggmpxay66h4sUJ
```

### 2. Basic Hivemind Training

```bash
# Set environment variables
export PYTHONPATH=/path/to/Distributed-Multicluster
export WANDB_MODE=disabled

# Run Hivemind training (replace PEER_ADDRESS with the actual address from DHT output)
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 train_fsdp.py \
    --per-device-train-batch-size 4 \
    --total-batch-size 16 \
    --lr 1e-2 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type dummy \
    --no-torch-compile \
    --fake-data \
    --max-steps 5 \
    --hv.initial-peers /ip4/127.0.0.1/tcp/30001/p2p/Qmd5gRA2PYYQB8gZNBau6APQTCGqEGsCggmpxay66h4sUJ \
    --hv.galaxy-size 1 \
    --hv.world-rank 0 \
    --hv.local-steps 2
```

### 3. Multi-Machine Hivemind Training

For training across multiple machines, set up the DHT on one machine and run training on others:

**On Machine 1 (DHT Coordinator):**
```bash
# Start DHT
hivemind-dht --identity_path fixed_private_key.pem --host_maddrs /ip4/0.0.0.0/tcp/30001

# Note the peer address from output
```

**On Machine 2 (Worker 1):**
```bash
export PEER_ADDRESS=/ip4/MACHINE1_IP/tcp/30001/p2p/PEER_ID
export WORLD_RANK=0

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_fsdp.py \
    --per-device-train-batch-size 4 \
    --total-batch-size 16 \
    --lr 1e-2 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type dummy \
    --no-torch-compile \
    --fake-data \
    --max-steps 5 \
    --hv.initial-peers $PEER_ADDRESS \
    --hv.galaxy-size 2 \
    --hv.world-rank $WORLD_RANK \
    --hv.local-steps 2
```

**On Machine 3 (Worker 2):**
```bash
export PEER_ADDRESS=/ip4/MACHINE1_IP/tcp/30001/p2p/PEER_ID
export WORLD_RANK=1

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_fsdp.py \
    --per-device-train-batch-size 4 \
    --total-batch-size 16 \
    --lr 1e-2 \
    --path-model ../tests/models/llama-2m-fresh \
    --metric-logger-type dummy \
    --no-torch-compile \
    --fake-data \
    --max-steps 5 \
    --hv.initial-peers $PEER_ADDRESS \
    --hv.galaxy-size 2 \
    --hv.world-rank $WORLD_RANK \
    --hv.local-steps 2
```

### 4. Hivemind-Specific Parameters

- `--hv.initial-peers`: DHT peer address for initial connection
- `--hv.galaxy-size`: Total number of workers across all machines
- `--hv.world-rank`: Unique rank for this worker (0, 1, 2, ...)
- `--hv.local-steps`: Number of local training steps before averaging
- `--hv.outer-lr`: Learning rate for the outer optimizer (default: 0.7)

### 5. Stopping Hivemind Training

```bash
# Stop training processes
pkill -f torchrun

# Stop DHT
pkill -f hivemind-dht
```

## Next Steps

After mastering both torch.distributed and Hivemind, explore:
- Real dataset training
- Model checkpointing and resuming
- Advanced FSDP configurations
- Production multi-cluster deployments
