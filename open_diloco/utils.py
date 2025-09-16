import hashlib
from functools import partial
import pickle
from typing import Any, Generator, Protocol, Dict, Tuple, Union, List, Optional

import torch
from torch.utils.hooks import RemovableHandle
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import IterableDataset
try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore


_WRAPPED_NAME_TO_REMOVE = ["_forward_module.", "_fsdp_wrapped_module.", "_orig_mod."]


def _remove_fsdp_prefix(name: str) -> str:
    for prefix in _WRAPPED_NAME_TO_REMOVE:
        if prefix in name:
            name = name.replace(prefix, "")
    return name


@torch.compiler.disable()
@torch.no_grad()
def log_activations_hook(
    _mod: torch.nn.Module,
    _inp: torch.Tensor,
    outp: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    mod_name: str,
    gradient_accumulation_steps: int,
    log_activations: Dict[str, float],
) -> None:
    if isinstance(outp, tuple):
        outp = outp[0]
    norm = outp.norm(p=2) / gradient_accumulation_steps
    name = _remove_fsdp_prefix(mod_name)
    if f"activation/{name}" not in log_activations:
        log_activations[f"activation/{name}"] = norm
    else:
        log_activations[f"activation/{name}"] += norm


def register_metrics_hooks(
    model: torch.nn.Module,
    target_layers: List[str],
    log_activations: Dict[str, torch.Tensor],
    gradient_accumulation_steps: int,
) -> List[RemovableHandle]:
    """
    this function take a torch   module, a list of layer name and apply a hook function that
    monitor the output norm of the layers.
    """
    handles = []
    for name, mod in model.named_modules():
        for layer in target_layers:
            if name.endswith(layer):
                handle = mod.register_forward_hook(
                    partial(
                        log_activations_hook,
                        log_activations=log_activations,
                        mod_name=name,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                    )
                )
                handles.append(handle)

    return handles


def _round_str(x: float):
    return f"{x:.4f}"


def _round_flatten(a: torch.Tensor, max_size: int = 1000):
    bounds = int(max_size**0.5)
    return ",".join(_round_str(i) for i, _ in zip(a[:bounds, :bounds].flatten(), range(max_size)))


def hash_tensor_content(a: torch.Tensor, max_size: int = 1000) -> str:
    return hashlib.md5(_round_flatten(a, max_size=max_size).encode("utf-8")).hexdigest()


def get_compression_kwargs(hivemind_compression: Union[str, None]) -> dict:
    """Return the compression kwargs for hivemind optimizer based on the hivemind_compression argument."""
    ret_kwargs = {}

    if hivemind_compression is None:
        from hivemind import NoCompression

        ret_kwargs["grad_compression"] = NoCompression()
        ret_kwargs["state_averaging_compression"] = NoCompression()

    elif hivemind_compression == "fp16":
        from hivemind import Float16Compression

        ret_kwargs["grad_compression"] = Float16Compression()
        ret_kwargs["state_averaging_compression"] = Float16Compression()
    elif hivemind_compression == "scaled-fp16":
        from hivemind import ScaledFloat16Compression

        ret_kwargs["grad_compression"] = ScaledFloat16Compression()
        ret_kwargs["state_averaging_compression"] = ScaledFloat16Compression()
    elif hivemind_compression == "uniform8bit":
        from hivemind import Uniform8BitQuantization

        ret_kwargs["grad_compression"] = Uniform8BitQuantization()
        ret_kwargs["state_averaging_compression"] = Uniform8BitQuantization()
    elif hivemind_compression == "quantile8bit":
        from hivemind import Quantile8BitQuantization

        ret_kwargs["grad_compression"] = Quantile8BitQuantization()
        ret_kwargs["state_averaging_compression"] = Quantile8BitQuantization()

    elif hivemind_compression == "blockwise8bit":
        from hivemind import BlockwiseQuantization

        ret_kwargs["grad_compression"] = BlockwiseQuantization()
        ret_kwargs["state_averaging_compression"] = BlockwiseQuantization()
    else:
        raise ValueError(f"Invalid hivemind_compression: {hivemind_compression}")
    return ret_kwargs


def found_inf_grad(optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler) -> bool:
    """
    this function check if the scaler has found inf grad for the optimizer. It does by looking up the optimizer state
    regsited inside the scaler. Code is mostly copied/inspired by the torch GradScaler codebase.
    """
    if not scaler._enabled:
        return False

    optimizer_state = scaler._per_optimizer_states[id(optimizer)]
    assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

    return sum(v.item() for v in optimizer_state["found_inf_per_device"].values()) > 0


def get_sharding_strategy(sharding_strategy: str) -> ShardingStrategy:
    if sharding_strategy == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    elif sharding_strategy == "HYBRID_SHARD":
        return ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "_HYBRID_SHARD_ZERO2":
        return ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Please choose 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD', or '_HYBRID_SHARD_ZERO2'."
        )


class FakeTokenizedDataset(IterableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"

    def __iter__(self) -> Generator[Dict[str, Any], Any, None]:
        while True:
            input_ids = torch.randint(3, self.vocab_size, (self.seq_len,)).tolist()
            attention_mask = [1] * self.seq_len
            yield {"input_ids": input_ids, "attention_mask": attention_mask}


class Logger(Protocol):
    def __init__(self, project, config): ...

    def log(self, metrics: Dict[str, Any]): ...

    def finish(self): ...


class WandbLogger:
    def __init__(self, project, config, resume: bool, run_id: Optional[str] = None):
        if wandb is None:
            raise RuntimeError("wandb is not available in the current environment")
        
        # If resuming and we have a run_id, try to continue the same run
        if resume and run_id:
            try:
                wandb.init(project=project, config=config, resume="must", id=run_id)
            except Exception as e:
                print(f"Failed to resume wandb run {run_id}: {e}")
                print("Creating a new wandb run instead")
                wandb.init(project=project, config=config)
        elif resume:
            # Try to resume automatically, but this might create a new run
            try:
                wandb.init(project=project, config=config, resume="auto")
            except Exception as e:
                print(f"Failed to auto-resume wandb: {e}")
                print("Creating a new wandb run instead")
                wandb.init(project=project, config=config)
        else:
            wandb.init(project=project, config=config)

    def log(self, metrics: Dict[str, Any]):
        if wandb is not None:
            wandb.log(metrics)

    def finish(self):
        if wandb is not None:
            wandb.finish()


class DummyLogger:
    def __init__(self, project, config, *args, **kwargs):
        self.project = project
        self.config = config
        open(project, "a").close()  # Create an empty file at the project path

        self.data = []

    def log(self, metrics: Dict[str, Any]):
        self.data.append(metrics)

    def finish(self):
        with open(self.project, "wb") as f:
            pickle.dump(self.data, f)
