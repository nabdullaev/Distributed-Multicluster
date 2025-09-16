import fsspec
import os
import torch
from typing import Optional, Tuple, List
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp
from fsspec.generic import GenericFileSystem

# Optional imports and fallbacks to run without hivemind/pydantic_config/torchdata
try:
    from pydantic_config import BaseConfig  # type: ignore
    _HAS_PYDANTIC_CONFIG = True
except Exception:
    _HAS_PYDANTIC_CONFIG = False
    class BaseConfig(object):
        pass

try:
    from torchdata.stateful_dataloader import StatefulDataLoader  # type: ignore
except Exception:
    from torch.utils.data import DataLoader as StatefulDataLoader  # type: ignore

try:
    from hivemind.optim.optimizer import logger  # type: ignore
except Exception:
    class _Logger:
        @staticmethod
        def info(msg: str):
            print(msg)

    logger = _Logger()


GLOBAL_STATE_FILE = "global_state_dict.pt"
CKPT_PREFIX = "model_step"

if _HAS_PYDANTIC_CONFIG:
    class CkptConfig(BaseConfig):
        # Use strings to avoid evaluation under Python 3.8
        resume: "str | bool | None" = None  # type: ignore[valid-type]
        interval: "int | None" = None  # type: ignore[valid-type]
        path: str = "outputs"
        topk: "int | None" = None  # type: ignore[valid-type]
else:
    class CkptConfig(BaseConfig):
        def __init__(self):
            self.resume = None  # type: ignore
            self.interval = None
            self.path = "outputs"
            self.topk = None


def get_resume_info(ckpt_config: CkptConfig) -> Tuple[bool, Optional[str]]:
    """
    check if we should resume from a checkpoint, if yes return the path to the checkpoint, otherwise return None
    """
    if ckpt_config.resume is None:
        return False, None
    elif isinstance(ckpt_config.resume, bool) and ckpt_config.resume:
        # Using fsspec to list directory contents
        fs = GenericFileSystem()
        try:
            ckpt_files = [f for f in fs.ls(ckpt_config.path, detail=False) if filter_ckpt_files(f)]
        except FileNotFoundError:
            logger.info(f"Checkpoint path {ckpt_config.path} not found, starting from scratch")
            return False, None

        if len(ckpt_files) == 0:
            logger.info(f"No checkpoints found in {ckpt_config.path}, starting from scratch")
            return False, None

        latest_ckpt = max(ckpt_files, key=lambda f: int(f.split("_")[-1]))
        return True, latest_ckpt
    elif isinstance(ckpt_config.resume, str):
        # Handle string values like "true", "false", or actual paths
        if ckpt_config.resume.lower() in ["true", "1", "yes"]:
            # Auto-detect latest checkpoint
            fs = GenericFileSystem()
            try:
                ckpt_files = [f for f in fs.ls(ckpt_config.path, detail=False) if filter_ckpt_files(f)]
            except FileNotFoundError:
                logger.info(f"Checkpoint path {ckpt_config.path} not found, starting from scratch")
                return False, None

            if len(ckpt_files) == 0:
                logger.info(f"No checkpoints found in {ckpt_config.path}, starting from scratch")
                return False, None

            latest_ckpt = max(ckpt_files, key=lambda f: int(f.split("_")[-1]))
            return True, latest_ckpt
        elif ckpt_config.resume.lower() in ["false", "0", "no"]:
            return False, None
        else:
            # Treat as specific checkpoint path
            return True, ckpt_config.resume
    else:
        return True, ckpt_config.resume


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    loss: Optional[float] = None,
    data_loader: Optional[StatefulDataLoader] = None,
    save_global_state: bool = True,
):
    """Save the model and optimizer state to a checkpoint folderx

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to save
        optimizer: the optimizer to save
        scheduler: the scheduler to save
        outer_optimizer: the outer optimizer to save
        loss: the loss to save
        data_loader: the data loader to save
        save_global_state: whether to save the global state
    """
    rank = int(os.environ["RANK"])

    # 1. Save distributed states
    fs_storage_writer = dcp.FsspecWriter(checkpoint_path, sync_files=False)
    # for some reason sync_files = True try to call stream.fileno which is not supported with gcp ffspec storage.

    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    dcp_state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }
    dcp.save(dcp_state_dict, storage_writer=fs_storage_writer)
    if data_loader is not None:
        rank_state_dict = {}
        # Skip data_loader state_dict as it doesn't exist
        # rank_state_dict["data_loader"] = data_loader.state_dict()
        with fsspec.open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "wb") as f:
            torch.save(rank_state_dict, f)

    if not save_global_state:
        return

    # 2. Save global states
    global_state_dict = {"scheduler": scheduler.state_dict(), "loss": loss if loss is not None else 0}
    if outer_optimizer is not None:
        global_state_dict["outer_optimizer"] = outer_optimizer.state_dict()
    if scaler is not None:
        global_state_dict["scaler"] = scaler.state_dict()
    
    # Save wandb run ID if available
    try:
        import wandb
        if wandb.run is not None:
            global_state_dict["wandb_run_id"] = wandb.run.id
    except Exception:
        pass  # wandb not available or not initialized

    with fsspec.open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "wb") as f:
        torch.save(global_state_dict, f)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    data_loader: Optional[StatefulDataLoader] = None,
) -> Tuple[float, Optional[str]]:
    """Load the model and optimizer state from a checkpoint folder

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to load
        optimizer: the optimizer to load
        scheduler: the scheduler to load
        outer_optimizer: the outer optimizer to load
        data_loader: the data loader to load

    Returns:
        Tuple[loss, wandb_run_id]: the loss from the checkpoint and wandb run ID if available
    """
    rank = int(os.environ["RANK"])
    # 1. Load distributed states
    fs_storage_reader = dcp.FsspecReader(checkpoint_path)

    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    dcp_state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }
    dcp.load(dcp_state_dict, storage_reader=fs_storage_reader)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optimizer_state_dict,
    )
    if data_loader is not None:
        rank_file_path = os.path.join(checkpoint_path, f"__{rank}_0.pt")
        if os.path.exists(rank_file_path):
            with fsspec.open(rank_file_path, "rb") as f:
                rank_state_dict = torch.load(f)
            if "data_loader" in rank_state_dict and hasattr(data_loader, "load_state_dict"):
                data_loader.load_state_dict(rank_state_dict["data_loader"])
        else:
            # If rank-specific file doesn't exist, try to use rank 0 file
            rank_0_file_path = os.path.join(checkpoint_path, "__0_0.pt")
            if os.path.exists(rank_0_file_path):
                with fsspec.open(rank_0_file_path, "rb") as f:
                    rank_state_dict = torch.load(f)
                if "data_loader" in rank_state_dict and hasattr(data_loader, "load_state_dict"):
                    data_loader.load_state_dict(rank_state_dict["data_loader"])

    # 2. Load global states
    with fsspec.open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "rb") as f:
        global_state_dict = torch.load(f)
    if scheduler is not None:
        scheduler.load_state_dict(global_state_dict["scheduler"])
        optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
    if outer_optimizer is not None and "outer_optimizer" in global_state_dict:
        outer_optimizer.load_state_dict(global_state_dict["outer_optimizer"])
    if scaler is not None:
        scaler.load_state_dict(global_state_dict["scaler"])
    
    # Get wandb run ID if available
    wandb_run_id = global_state_dict.get("wandb_run_id", None)
    return global_state_dict["loss"], wandb_run_id


def filter_ckpt_files(f):
    if CKPT_PREFIX not in f:
        return False
    else:
        try:
            int(f.split("_")[-1])
            return True
        except ValueError:
            return False


def delete_old_checkpoints(checkpoint_path: str, topk: int) -> List[str]:
    fs = GenericFileSystem()
    ckpt_files = [f for f in fs.ls(checkpoint_path, detail=False) if filter_ckpt_files(f)]
    ckpt_files.sort(key=lambda x: int(x.split("_")[-1]))

    ckpt_deleted = []
    for ckpt_file in ckpt_files[:-topk]:
        fs.rm(ckpt_file, recursive=True)
        ckpt_deleted.append(ckpt_file)
    return ckpt_deleted


def check_checkpoint_path_access(checkpoint_path: str, rank: int, world_rank_hv: Optional[int] = None):
    if world_rank_hv:
        dummy_file_path = os.path.join(
            checkpoint_path, get_diloco_rank_dir_name(world_rank_hv), f"dummy_file_{rank}.txt"
        )
    else:
        dummy_file_path = os.path.join(checkpoint_path, f"dummy_file_{rank}.txt")

    with fsspec.open(dummy_file_path, "w") as f:
        f.write("This is a dummy file for testing access.")
    gfs = GenericFileSystem()
    gfs.rm(dummy_file_path)


def get_diloco_rank_dir_name(world_rank_diloco: int) -> str:
    return f"diloco_rank_{world_rank_diloco}"
