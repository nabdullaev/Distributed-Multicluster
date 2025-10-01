import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader

# from open_diloco.utils import FakeTokenizedDataset, WandbLogger, DummyLogger
from utils import FakeTokenizedDataset, WandbLogger, DummyLogger

from nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot


# Function to initialize the distributed process group
def ddp_setup():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_offloaded_param(outer_optimizer: torch.optim.Optimizer):
    return [
        param.data.detach().clone().to("cpu") for group in outer_optimizer.param_groups for param in group["params"]
    ]


def main(
    batch_size: int = 512,
    per_device_train_batch_size: int = 32,
    seq_length: int = 1024,
    checkpoint_interval: Optional[int] = None,
    checkpoint_path: str = "outputs",
    warmup_steps: int = 1000,
    total_steps: int = 88_000,
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed",
    project: str = "hivemind_debug",
    model_name_or_path: str = "PrimeIntellect/llama-150m-fresh",
    lr: float = 4e-4,
    resume_from_checkpoint: Optional[str] = None,
    local_steps: int = 500,
    outer_lr: float = 0.7,
    fake_data: bool = True,
    max_steps: Optional[int] = None,
    metric_logger_type: str = "dummy",
):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert batch_size % per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    # Setup logging
    wandb_run_id = None
    if local_rank == 0:
        logger_cls = WandbLogger if metric_logger_type == "wandb" else DummyLogger
        metric_logger = logger_cls(project=project, config={}, resume=False, run_id=wandb_run_id)

    # Load model
    from transformers import LlamaConfig, LlamaForCausalLM
    
    config_model = LlamaConfig.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=config_model)
    model = model.to(local_rank)

    # Setup optimizers with more conservative settings for DiLoCo
    inner_optimizer = torch.optim.AdamW(model.parameters(), lr=lr*0.1, weight_decay=0.1, betas=(0.9, 0.95))  # Lower LR
    outer_optimizer = torch.optim.SGD(model.parameters(), lr=outer_lr*0.1, momentum=0.9, nesterov=True)  # Lower LR

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if precision not in ["fp16-mixed", "bf16-mixed", "32-true"]:
        raise ValueError(f"Invalid precision: {precision}. Please choose 'fp16-mixed', 'bf16-mixed', or '32-true'.")

    half_precision = precision == "fp16-mixed" or precision == "bf16-mixed"
    half_precision_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16-mixed")

    # Setup data
    if fake_data:
        def _collate_fake(batch):
            input_ids = torch.stack([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch])
            attention_mask = torch.stack([torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        train_dataset = FakeTokenizedDataset(seq_length, 1024)
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=_collate_fake,
            batch_size=per_device_train_batch_size,
            num_workers=0,
        )
    else:
        # For real data, we'd need to implement tokenization
        raise NotImplementedError("Real data not implemented in this simplified version")

    start_step = 0
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        # Load checkpoint using the same utilities as train_fsdp.py
        # from open_diloco.ckpt_utils import load_checkpoint
        from ckpt_utils import load_checkpoint
        last_loss, wandb_run_id = load_checkpoint(
            checkpoint_path=resume_from_checkpoint,
            model=model,
            optimizer=inner_optimizer,
            scheduler=scheduler,
            outer_optimizer=outer_optimizer,
            scaler=scaler,
        )
        start_step = scheduler.last_epoch
        print(f"Resumed from checkpoint at step {start_step} with loss {last_loss}")
        
        # Reinitialize logger with wandb run ID if resuming
        if local_rank == 0 and metric_logger_type == "wandb" and wandb_run_id:
            metric_logger = WandbLogger(project=project, config={}, resume=True, run_id=wandb_run_id)

    # Initialize parameters consistently across ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Initialize DiLoCo parameters
    params_offloaded = get_offloaded_param(outer_optimizer)
    
    # Make sure all ranks have the same initial offloaded parameters
    # Move to GPU first, then broadcast
    for param_offloaded in params_offloaded:
        param_offloaded_gpu = param_offloaded.to(local_rank)
        dist.broadcast(param_offloaded_gpu, src=0)
        param_offloaded.copy_(param_offloaded_gpu.cpu())
    
    model.train()

    start_time = time.time()
    print(f"starting from step {start_step}")

    loss_batch = 0

    real_step = 0
    for step, batch in enumerate(iterable=train_dataloader):
        step_within_grad_acc = (step + 1) % gradient_accumulation_steps

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        with torch.autocast(device_type="cuda", dtype=half_precision_dtype) if half_precision else nullcontext():
            outputs = model(**batch)
            # Handle case where model doesn't return loss directly
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss / gradient_accumulation_steps
            else:
                # Calculate loss manually for language modeling
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['input_ids'][..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) / gradient_accumulation_steps
            loss_batch += loss.detach()

        scaler.scale(loss).backward()

        if step_within_grad_acc == 0:
            scaler.unscale_(optimizer=inner_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # More aggressive clipping
            scaler.step(optimizer=inner_optimizer)
            scaler.update()
            scheduler.step()
            inner_optimizer.zero_grad()
            
            # Increment real_step after optimizer step
            real_step += 1

            # DiLoCo synchronization every local_steps
            if real_step % local_steps == 0 and real_step > 0:  # Skip step 0
                if local_rank == 0:
                    print(f"perform outer step at step {real_step}")

                main_param = [param for group in inner_optimizer.param_groups for param in group["params"]]

                for param_offloaded, param in zip(params_offloaded, main_param):
                    param_offloaded_on_device = param_offloaded.data.to(param.device)
                    # Calculate gradient as difference
                    param.grad = param_offloaded_on_device - param.data
                    # Average gradients across ranks
                    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)
                    # Update parameter
                    param.data = param_offloaded_on_device

                outer_optimizer.step()
                outer_optimizer.zero_grad()
                params_offloaded = get_offloaded_param(outer_optimizer)

            if local_rank == 0:
                dict_to_log = {
                    "Loss": loss_batch.item(),
                    "step": real_step,
                    "lr": [group["lr"] for group in inner_optimizer.param_groups][0],
                    "Perplexity": torch.exp(loss_batch).item(),
                    "effective_step": real_step * world_size,
                    "total_samples": real_step * batch_size * world_size,
                }

                metric_logger.log(dict_to_log)
                print(
                    f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in inner_optimizer.param_groups][0]}"
                )
                loss_batch = 0
                
                # Check if we should stop after completing this step
                # Use max_steps if provided, otherwise use total_steps
                steps_limit = max_steps if max_steps is not None else total_steps
                if real_step >= steps_limit:
                    print(f"Reached step limit ({steps_limit}), stopping training.")
                    break
        
        if args.use_nirvana and step % checkpoint_interval == 0:
            copy_out_to_snapshot("nirvana_checkpoints")

    print("Training completed.")
    if local_rank == 0:
        metric_logger.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=str, default="outputs")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=88_000)
    parser.add_argument("--precision", type=str, default="fp16-mixed")
    parser.add_argument("--project", type=str, default="hivemind_debug")
    parser.add_argument("--model-name-or-path", type=str, default="PrimeIntellect/llama-150m-fresh")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--local-steps", type=int, default=500)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--fake-data", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--metric-logger-type", type=str, default="dummy")
    parser.add_argument("--use-nirvana", action="store_true", default=False)
    
    args = parser.parse_args()

    if args.use_nirvana:
        os.makedirs("nirvana_checkpoints", exist_ok=True)
        copy_snapshot_to_out("nirvana_checkpoints")
    
    ddp_setup()
    main(**vars(args))
    destroy_process_group()
