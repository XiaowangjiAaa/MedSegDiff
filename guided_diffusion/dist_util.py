"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

def setup_dist(args):
    """
    Setup a distributed process group for multi-GPU training.
    Assumes torchrun or torch.distributed.launch is used to start the script.
    """
    if dist.is_initialized():
        return

    backend = "nccl" if th.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    th.cuda.set_device(dev().index)
    print(f"[Rank {dist.get_rank()}] Initialized process group with backend '{backend}'.")


def dev():
    """
    Get the correct local device for the current process.
    This ensures each rank gets its designated GPU.
    """
    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    mpigetrank = dist.get_rank() if dist.is_initialized() else 0
    if mpigetrank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None

    # Broadcast data to all processes
    data = _broadcast_bytes(data)
    return th.load(io.BytesIO(data), **kwargs)


def _broadcast_bytes(data):
    """
    Helper to broadcast raw byte data across ranks.
    """
    length = th.tensor([len(data)], dtype=th.int64, device=dev())
    dist.broadcast(length, src=0)
    buffer = th.empty(length.item(), dtype=th.uint8, device=dev())
    if dist.get_rank() == 0:
        buffer.copy_(th.tensor(list(data), dtype=th.uint8, device=dev()))
    dist.broadcast(buffer, src=0)
    return bytes(buffer.cpu().tolist())


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, src=0)


def _find_free_port():
    """
    Not used in current setup but kept for future flexibility.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
