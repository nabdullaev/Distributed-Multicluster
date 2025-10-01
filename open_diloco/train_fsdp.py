"""

to test quickly do 
torchrun --nproc_per_node=2 \
        train_fsdp.py --per-device-train-batch-size 8 --total-batch-size 128 --lr 1e-2 --path-model ../tests/models/llama-2m-fresh \
        --no-torch-compile --log-activations-steps 5 --fake-data --max-steps 20
"""

from functools import partial
import os
import time
from contextlib import nullcontext
import datetime
from typing import Any, Literal, Optional, List, Dict

import torch
try:
    from pydantic import model_validator  # type: ignore
    from pydantic_config import parse_argv, BaseConfig  # type: ignore
    _HAS_PYDANTIC_CONFIG = True
except Exception:
    import argparse
    _HAS_PYDANTIC_CONFIG = False

    def _bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_msg: str):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(f"--{name}", dest=name.replace('-', '_'), action="store_true", help=help_msg)
        group.add_argument(f"--no-{name}", dest=name.replace('-', '_'), action="store_false", help=f"Disable {help_msg}")
        parser.set_defaults(**{name.replace('-', '_'): default})
from torch.distributed import destroy_process_group, init_process_group

# Optional import: torchdata (only needed for non-fake data). Fallback to standard DataLoader
try:
    from torchdata.stateful_dataloader import StatefulDataLoader  # type: ignore
except Exception:  # torchdata not installed
    from torch.utils.data import DataLoader as StatefulDataLoader  # type: ignore

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.device_mesh import init_device_mesh

from ckpt_utils import (
    CKPT_PREFIX,
    CkptConfig,
    check_checkpoint_path_access,
    delete_old_checkpoints,
    get_diloco_rank_dir_name,
    get_resume_info,
    load_checkpoint,
    save_checkpoint,
)
# Lazy import hivemind components only when needed to avoid hard dependency
try:
    from hivemind_diloco import AllReduceStrategy, DiLoCoOptimizer  # type: ignore
except Exception:
    AllReduceStrategy = None  # type: ignore
    DiLoCoOptimizer = None  # type: ignore
from utils import WandbLogger, DummyLogger

try:
    from hivemind.dht.dht import DHT  # type: ignore
except Exception:
    DHT = None  # type: ignore
try:
    from hivemind.utils.networking import log_visible_maddrs  # type: ignore
except Exception:
    def log_visible_maddrs(*args, **kwargs):  # type: ignore
        return None
from hivemind.optim.optimizer import logger


from utils import (
    FakeTokenizedDataset,
    get_compression_kwargs,
    get_sharding_strategy,
    register_metrics_hooks,
)

from nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot


TIMEOUT_NCCL_MINUTES = os.environ.get("TIMEOUT_NCCL_MINUTES", 120)
TARGET_LAYER_ACTIVATIONS = ["self_attn", "lm_head"]
TEST_VOCAB_SIZE = 1024


# Function to initialize the distributed process group
def ddp_setup():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_process_group(backend=backend, timeout=datetime.timedelta(minutes=TIMEOUT_NCCL_MINUTES))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


class HvConfig(BaseConfig if _HAS_PYDANTIC_CONFIG else object):
    outer_lr: float = 0.7
    local_steps: int = 500
    initial_peers: Optional[List[str]] = None
    host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/0"]
    announce_maddrs: Optional[List[str]] = None
    matchmaking_time: Optional[float] = None
    averaging_timeout: Optional[float] = None
    hivemind_compression: Optional[Literal["fp16", "scaled-fp16", "uniform8bit", "quantile8bit", "blockwise8bit"]] = None
    all_reduce_strategy: Any = AllReduceStrategy.WAIT_FOR_ALL if AllReduceStrategy is not None else None
    timeout_waiting_for_peers: Optional[float] = None
    skip_load_from_peers: bool = False
    world_rank: int
    galaxy_size: int
    fail_rank_drop: bool = False  # fail if we lose a diloco worker

    @model_validator(mode="before")
    def cast_str_to_list(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """This allow to only pass a string and it will still be cast as a list"""
        for arg_name in ["initial_peers", "host_maddrs", "announce_maddrs"]:
            if arg_name in values.keys() and isinstance(values[arg_name], str):
                values[arg_name] = [values[arg_name]]
        return values


class Config(BaseConfig if _HAS_PYDANTIC_CONFIG else object):
    path_model: str = "PrimeIntellect/llama-150m-fresh"
    torch_compile: bool = True
    attn_implementation: str = "sdpa"
    # Data
    dataset_name_or_path: str = "allenai/c4"
    seq_length: int = 1024
    c4_tiny: bool = False
    num_workers: int = 4
    # Optimization
    lr: float = 4e-4
    total_batch_size: int = 512
    per_device_train_batch_size: int = 32
    warmup_steps: int = 1000
    total_steps: int = 88_000
    sharding_strategy: str = "NO_SHARD"
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"
    # Checkpointing and logging
    project: str = "hivemind_debug"
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    log_activations_steps: Optional[int] = None
    ckpt: CkptConfig = CkptConfig()
    # Hivemind
    hv: Optional[HvConfig] = None  # if no hv config then hivemind is disabled
    fake_data: bool = False
    max_steps: Optional[int] = None


def get_dataloader(tokenizer, world_size, rank, local_rank, config: Config) -> StatefulDataLoader:
    if config.fake_data:
        train_dataset = FakeTokenizedDataset(config.seq_length, TEST_VOCAB_SIZE)
    else:
        # Lazy import heavy deps only if needed
        from datasets import load_dataset  # type: ignore
        from datasets.distributed import split_dataset_by_node  # type: ignore

        ds = load_dataset(config.dataset_name_or_path, "en", streaming=True)

        def tokenize_function(data):
            outputs = tokenizer(
                data["text"],
                truncation=True,
                max_length=config.seq_length,
                padding="max_length",
            )
            return outputs

        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])[
            "train"
        ]

        if config.hv is not None:
            train_dataset = split_dataset_by_node(
                tokenized_datasets,
                world_size=config.hv.galaxy_size * world_size,
                rank=config.hv.world_rank * world_size + local_rank,
            )

        else:
            train_dataset = split_dataset_by_node(tokenized_datasets, world_size=world_size, rank=rank)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return StatefulDataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
    )


def get_model(config: Config) -> LlamaForCausalLM:
    # Load model
    config_model = LlamaConfig.from_pretrained(config.path_model, attn_implementation=config.attn_implementation)
    return LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=config.path_model, config=config_model)


def train(config: Config):
    sharding_strategy = get_sharding_strategy(config.sharding_strategy)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    world_messenger_hv = config.hv is not None and local_rank == 0

    # batch_size is the total batch size for all GPUs
    assert config.total_batch_size % world_size == 0
    batch_size = config.total_batch_size // world_size

    assert batch_size % config.per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // config.per_device_train_batch_size

    if config.hv is not None:
        sharding_strategy = ShardingStrategy.NO_SHARD
        log("Hivemind is used, ShardingStrategy.NO_SHARD is used")

    resume_from_ckpt, resume_path = get_resume_info(config.ckpt)

    # Get wandb run ID from checkpoint if resuming
    wandb_run_id = None
    if resume_from_ckpt:
        # We'll get the wandb run ID after loading the checkpoint
        pass
    
    if rank == 0:
        logger_cls = WandbLogger if config.metric_logger_type == "wandb" else DummyLogger
        cfg_payload = config.model_dump() if hasattr(config, "model_dump") else vars(config)
        metric_logger = logger_cls(project=config.project, config=cfg_payload, resume=resume_from_ckpt, run_id=wandb_run_id)

    if config.hv is not None:
        if AllReduceStrategy is None or DiLoCoOptimizer is None:
            raise RuntimeError("Hivemind is not installed but hv config is provided. Install hivemind or run without hv.")
        log("hivemind diloco enabled")

    if world_messenger_hv:
        # safe import within block (ensures module load only if actually used)
        from hivemind_diloco import AllReduceStrategy as _ARS, DiLoCoOptimizer as _DLO  # noqa: F401
        dht = DHT(
            start=True,
            initial_peers=config.hv.initial_peers,
            host_maddrs=config.hv.host_maddrs,
            announce_maddrs=config.hv.announce_maddrs,
        )
        log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=False)

    if local_rank == 0:
        check_checkpoint_path_access(config.ckpt.path, rank, config.hv.world_rank if config.hv else None)

    # DataLoader preparation
    if config.fake_data:
        # Avoid external downloads for tokenizer when using fake data
        def _collate_fake(batch):
            input_ids = torch.stack([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch])
            attention_mask = torch.stack([torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        train_dataset = FakeTokenizedDataset(config.seq_length, TEST_VOCAB_SIZE)
        train_dataloader = StatefulDataLoader(
            train_dataset,
            collate_fn=_collate_fake,
            batch_size=config.per_device_train_batch_size,
            num_workers=config.num_workers,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
        tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it
        train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank, config)

    model = get_model(config)
    model = model.to(local_rank)

    half_precision = config.precision == "fp16-mixed" or config.precision == "bf16-mixed"
    half_precision_dtype = torch.bfloat16 if config.precision == "bf16-mixed" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16-mixed")

    if sharding_strategy in [
        ShardingStrategy._HYBRID_SHARD_ZERO2,
        ShardingStrategy.HYBRID_SHARD,
    ]:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        nnodes = world_size // local_world_size
        device_mesh = init_device_mesh("cuda", (nnodes, local_world_size), mesh_dim_names=("global", "local"))
    else:
        device_mesh = None
    model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=MixedPrecision(param_dtype=half_precision_dtype) if half_precision else None,
            use_orig_params=config.torch_compile,
            device_mesh=device_mesh,
        )
    if config.torch_compile:
        model = torch.compile(model)

    # Setup optimizers
    inner_optimizer = partial(torch.optim.AdamW, lr=config.lr, weight_decay=0.1, betas=(0.9, 0.95))  # noqa: F821

    if config.hv is not None:
        outer_optimizer = partial(torch.optim.SGD, lr=config.hv.outer_lr, momentum=0.9, nesterov=True)

    def scheduler_fn(opt):
        return get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )

    if config.hv is not None:
        if resume_from_ckpt:
            # We need to load with a fake optimizer to set the model parameters correctly before initializing the DiLoCoOptimizer
            # This is because the DiLoCoOptimizer makes a copy of the model parameters for the state averager which is hard to update later
            # We also need to do this on follower workers so that the world_messenger has friends to talk to when it does its two loads
            # Otherwise the world messenger will get lonely and hang
            fake_optimizer = inner_optimizer(model.parameters())
            last_loss, _ = load_checkpoint(
                checkpoint_path=os.path.join(resume_path, get_diloco_rank_dir_name(config.hv.world_rank)),
                model=model,
                optimizer=fake_optimizer,
            )
            del fake_optimizer

    if resume_from_ckpt:
        if config.hv is not None:
            ckpt_path = os.path.join(resume_path, get_diloco_rank_dir_name(config.hv.world_rank))
        else:
            ckpt_path = resume_path

    if world_messenger_hv:
        diloco_args = dict(
            dht=dht,
            run_id="llama",
            batch_size=batch_size,
            num_inner_steps=config.hv.local_steps,
            outer_optimizer=outer_optimizer,
            inner_optimizer=inner_optimizer,
            scheduler=None,
            params=model.parameters(),
            delay_optimizer_step=False,
            delay_grad_averaging=False,
            verbose=True,
            all_reduce_strategy=config.hv.all_reduce_strategy,
            timeout_waiting_for_peers=config.hv.timeout_waiting_for_peers,
        )

        diloco_args.update(get_compression_kwargs(config.hv.hivemind_compression))

        if config.hv.averaging_timeout is not None:
            diloco_args["averaging_timeout"] = config.hv.averaging_timeout

        if config.hv.matchmaking_time is not None:
            diloco_args["matchmaking_time"] = config.hv.matchmaking_time

        optimizer = DiLoCoOptimizer(**diloco_args)

        scheduler = scheduler_fn(
            optimizer.inner_optimizer
        )  # scheduler(optimizer) should work but better to make it explicit here

        if resume_from_ckpt:
            last_loss, wandb_run_id = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer.inner_optimizer,
                scheduler=scheduler,
                outer_optimizer=optimizer.state_averager.optimizer,
                scaler=scaler,
                data_loader=train_dataloader,
            )
            start_step = scheduler.last_epoch
        else:
            start_step = 0

    else:
        optimizer = inner_optimizer(model.parameters())
        scheduler = scheduler_fn(optimizer)
        if resume_from_ckpt:
            last_loss, wandb_run_id = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                data_loader=train_dataloader,
            )
            start_step = scheduler.last_epoch
        else:
            start_step = 0

    if resume_from_ckpt:
        log(f"Resumed from checkpoint at step {start_step} with loss {last_loss}")

    model.train()

    if world_messenger_hv and not config.hv.skip_load_from_peers:
        optimizer.load_state_from_peers()

    current_time = time.time()
    log(f"starting from step {start_step}")

    loss_batch = 0

    if world_messenger_hv:
        max_num_peers = 0

    log_activations = {}

    for step, batch in enumerate(iterable=train_dataloader, start=start_step * gradient_accumulation_steps):
        real_step = (step + 1) // gradient_accumulation_steps
        is_accumulating = bool((step + 1) % gradient_accumulation_steps)

        logging_activations_steps = (
            config.log_activations_steps is not None and real_step % config.log_activations_steps == 0
        )

        if logging_activations_steps:
            handles = register_metrics_hooks(
                model, TARGET_LAYER_ACTIVATIONS, log_activations, gradient_accumulation_steps
            )

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        with model.no_sync() if is_accumulating else nullcontext():
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

        if logging_activations_steps:
            for handle in handles:
                handle.remove()

        if not is_accumulating:
            if world_messenger_hv:
                scaler.unscale_(optimizer=optimizer.inner_optimizer)
            else:
                scaler.unscale_(optimizer=optimizer)

            model.clip_grad_norm_(1.0)  # gradient clipping

            if world_messenger_hv:
                optimizer.step(scaler=scaler)

                # todo(sami): refactor to use built in pytorch mechanism to handle scaler manually
                # should allow to just do scaler.step(optimizer)
            else:
                scaler.step(optimizer)

            scaler.update()

            scheduler.step()
            optimizer.zero_grad()

            if config.hv is not None:
                if int(real_step) % config.hv.local_steps == 0:
                    for param in model.parameters():
                        torch.distributed.broadcast(param.data, src=0)

            if rank == 0:
                total_samples = real_step * config.total_batch_size
                effective_step = real_step

                if config.hv is not None:
                    # Note that this assumes that we have the right amount of worker since t0.
                    # Not robust to off/on ramping
                    effective_step = real_step * config.hv.galaxy_size
                    total_samples = real_step * config.total_batch_size * config.hv.galaxy_size

                metrics = {
                    "Loss": loss_batch.item(),
                    "step": real_step,
                    "lr": [group["lr"] for group in optimizer.param_groups][0],
                    "Perplexity": torch.exp(loss_batch).item(),
                    "effective_step": effective_step,  # at each step the we have compute total_batch_size. Independent of the number of GPUs
                    "total_samples": total_samples,
                    "time_taken": time.time() - current_time,
                    "tokens_per_second": config.seq_length * config.total_batch_size / (time.time() - current_time),
                }

                if world_messenger_hv:
                    outer_lr = [group["lr"] for group in optimizer.state_averager.optimizer.param_groups][0]
                    num_peers = optimizer.tracker.global_progress.num_peers

                    max_num_peers = max(max_num_peers, num_peers)

                    if num_peers == 0:
                        num_peers = 1

                    metrics["outer_lr"] = outer_lr
                    metrics["num_peers"] = num_peers

                if logging_activations_steps:
                    metrics.update(log_activations)
                    log_activations = {}

                if world_messenger_hv and num_peers < max_num_peers:
                    log(message=f"Lost a diloco worker, num_peers: {num_peers}, galaxy_size: {config.hv.galaxy_size}")
                    if config.hv.fail_rank_drop:
                        raise ValueError(
                            f"Lost a diloco worker, num_peers: {num_peers}, galaxy_size: {config.hv.galaxy_size}"
                        )

                current_time = time.time()

                metric_logger.log(metrics)

                if config.hv is None:
                    log(
                        f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in optimizer.param_groups][0]}"
                    )

            # Save checkpoint every 'checkpoint_interval' steps
            if config.ckpt.interval is not None and real_step % config.ckpt.interval == 0:
                log(f"saving at step {real_step}, step {step+1}")
                ckpt_path = os.path.join(config.ckpt.path, f"{CKPT_PREFIX}_{int(real_step)}")

                if config.hv:
                    ckpt_path = os.path.join(ckpt_path, get_diloco_rank_dir_name(config.hv.world_rank))

                if world_messenger_hv:
                    assert isinstance(optimizer, DiLoCoOptimizer)
                    with optimizer.tracker.pause_updates():
                        save_checkpoint(
                            checkpoint_path=ckpt_path,
                            model=model,
                            optimizer=optimizer.inner_optimizer,
                            scheduler=scheduler,
                            outer_optimizer=optimizer.state_averager.optimizer,
                            loss=loss_batch.item(),
                            scaler=scaler,
                            data_loader=train_dataloader,
                            save_global_state=True,
                        )
                else:
                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=loss_batch.item(),
                        scaler=scaler,
                        data_loader=train_dataloader,
                        save_global_state=rank == 0,
                    )
                
                if args.use_nirvana:
                    copy_out_to_snapshot("nirvana_checkpoints")

                if local_rank == 0:
                    # only the rank 0 deletes the checkpoints
                    if config.ckpt.topk is not None:
                        ckpt_deleted = delete_old_checkpoints(config.ckpt.path, config.ckpt.topk)
                        if ckpt_deleted:
                            log(f"Deleted old checkpoints: {ckpt_deleted}")

            loss_batch = 0

            if config.max_steps is not None and real_step >= config.max_steps:
                break

    log("Training completed.")
    if rank == 0:
        metric_logger.finish()


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "PRIME_INTELLECT_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    if _HAS_PYDANTIC_CONFIG:
        config = Config(**parse_argv())
    else:
        # minimal argparse fallback covering flags we use in our run
        import argparse as _argparse
        _p = _argparse.ArgumentParser()
        _p.add_argument("--path-model", dest="path_model", default="PrimeIntellect/llama-150m-fresh")
        _p.add_argument("--per-device-train-batch-size", type=int, dest="per_device_train_batch_size", default=32)
        _p.add_argument("--total-batch-size", type=int, dest="total_batch_size", default=512)
        _p.add_argument("--lr", type=float, dest="lr", default=4e-4)
        _p.add_argument("--seq-length", type=int, dest="seq_length", default=1024)
        _p.add_argument("--project", dest="project", default="hivemind_debug")
        _p.add_argument("--precision", dest="precision", default="fp16-mixed")
        _p.add_argument("--sharding-strategy", dest="sharding_strategy", default="NO_SHARD")
        _p.add_argument("--num-workers", type=int, dest="num_workers", default=4)
        _p.add_argument("--total-steps", type=int, dest="total_steps", default=88_000)
        _p.add_argument("--warmup-steps", type=int, dest="warmup_steps", default=1000)
        _p.add_argument("--log-activations-steps", type=int, dest="log_activations_steps", default=None)
        _p.add_argument("--max-steps", type=int, dest="max_steps", default=None)
        _bool_flag(_p, "torch-compile", default=True, help_msg="torch compile")
        _bool_flag(_p, "fake-data", default=False, help_msg="use fake data")
        _p.add_argument("--metric-logger-type", dest="metric_logger_type", default="wandb")
        _p.add_argument("--ckpt.interval", dest="ckpt_interval", type=int, default=None)
        _p.add_argument("--ckpt.path", dest="ckpt_path", default="outputs")
        _p.add_argument("--ckpt.resume", dest="ckpt_resume", type=str, default=None)
        # Hivemind arguments
        _p.add_argument("--hv.world-rank", dest="hv_world_rank", type=int, default=None)
        _p.add_argument("--hv.galaxy-size", dest="hv_galaxy_size", type=int, default=None)
        _p.add_argument("--hv.local-steps", dest="hv_local_steps", type=int, default=500)
        _p.add_argument("--hv.outer-lr", dest="hv_outer_lr", type=float, default=0.7)
        _p.add_argument("--hv.initial-peers", dest="hv_initial_peers", type=str, default=None)
        _p.add_argument("--use-nirvana", action="store_true", default=False)
        args = _p.parse_args()

        if args.use_nirvana:
            os.makedirs("nirvana_checkpoints", exist_ok=True)
            copy_snapshot_to_out("nirvana_checkpoints")

        class _C(Config):
            pass

        cfg = _C()
        cfg.path_model = args.path_model
        cfg.per_device_train_batch_size = args.per_device_train_batch_size
        cfg.total_batch_size = args.total_batch_size
        cfg.lr = args.lr
        cfg.seq_length = args.seq_length
        cfg.project = args.project
        cfg.precision = args.precision
        cfg.sharding_strategy = args.sharding_strategy
        cfg.num_workers = args.num_workers
        cfg.total_steps = args.total_steps
        cfg.warmup_steps = args.warmup_steps
        cfg.log_activations_steps = args.log_activations_steps
        cfg.max_steps = args.max_steps
        cfg.torch_compile = args.torch_compile
        cfg.fake_data = args.fake_data
        cfg.metric_logger_type = args.metric_logger_type
        cfg.ckpt.interval = args.ckpt_interval
        cfg.ckpt.path = args.ckpt_path
        cfg.ckpt.resume = args.ckpt_resume
        
        # Set Hivemind config if provided
        if args.hv_world_rank is not None:
            # from open_diloco.train_fsdp import HvConfig
            cfg.hv = HvConfig()
            cfg.hv.world_rank = args.hv_world_rank
            cfg.hv.galaxy_size = args.hv_galaxy_size or 1
            cfg.hv.local_steps = args.hv_local_steps
            cfg.hv.outer_lr = args.hv_outer_lr
            if args.hv_initial_peers:
                cfg.hv.initial_peers = [args.hv_initial_peers]
        config = cfg

    train(config)
    destroy_process_group()
