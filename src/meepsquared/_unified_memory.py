# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Loading a large checkpoint onto a GPU that shares the system's memory.

Transformers opens every safetensors shard with mmap and builds the GPU
tensors while the mapped pages are still in memory. On a GPU that shares the
system's memory (an NVIDIA GB10, a DGX Spark) the page cache and the GPU's
tensors come out of the same memory, so a checkpoint is held twice: a 67 GB
model needs some 134 GB and the load runs out of memory. While meepsquared loads a
model onto such a GPU, each shard is read straight to the GPU and its pages are
dropped from the page cache, so at most one shard is held twice.

On such a GPU the kernel's out-of-memory killer, not the GPU, ends a process
that takes too much: it kills whatever it picks, this process or another, and
the machine may freeze first. So before loading, PyTorch's allocator is
capped at the memory the system has available less HEADROOM: what the page
cache needs for the files a training reads and writes, and the growth of the
machine's own daemons, which do not grow with the machine. An overflow is then
PyTorch's OutOfMemoryError in this process, which a plan or a caller can act on.
"""

import contextlib
import os

HEADROOM = 4 * 2**30


def shares_system_memory(device: int = 0) -> bool:
    import torch

    return bool(torch.cuda.get_device_properties(device).is_integrated)


def available_memory() -> int:
    """The system's MemAvailable, in bytes: what the kernel can give without swapping, the page cache included."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable line")


def allocator_cap(available: int, total: int) -> int:
    """The bytes PyTorch may allocate on a GPU that shares the system's memory: the available memory less the headroom."""
    cap = available - HEADROOM
    if cap <= 0:
        raise RuntimeError(f"the system has {available / 2**30:.1f} GiB available of {total / 2**30:.1f} GiB, "
                           f"less than the {HEADROOM / 2**30:.0f} GiB kept for the page cache and the machine's daemons; "
                           "free memory on the machine before loading a model")
    return cap


def recap_allocator(device: int = 0) -> int:
    """Cap PyTorch's allocator again, at what this process holds now plus what the system has available less the
    headroom: after something else in the process (a shared vLLM) has taken memory, outside the allocator too, the cap
    set before it would let the process take more than the machine has left. Returns the cap in bytes."""
    import torch

    total = torch.cuda.get_device_properties(device).total_memory
    held = torch.cuda.memory_reserved(device)
    # Three times the headroom: the training's own host memory still grows after loading by 10 GiB and more on
    # Qwen3.6-35B-A3B (its data, its loader, its kernels' compilation)
    cap = min(held + allocator_cap(available_memory() - 2 * HEADROOM, total), total)
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    return cap


def cap_allocator(device: int = 0) -> int:
    """Cap PyTorch's allocator on a GPU that shares the system's memory; returns the cap in bytes."""
    import torch

    total = torch.cuda.get_device_properties(device).total_memory
    cap = min(allocator_cap(available_memory(), total), total)
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    return cap


class _ShardOnGpu:
    """A safetensors shard read whole to the GPU, standing in for the mmap handle Transformers opens."""

    def __init__(self, safe_open, filename: str, device: str):
        with safe_open(filename, framework="pt", device=device) as f:
            self.tensors = {key: f.get_tensor(key) for key in f.keys()}
        fd = os.open(filename, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def keys(self):
        return self.tensors.keys()

    def get_slice(self, key):
        return self.tensors[key]

    def get_tensor(self, key):
        return self.tensors[key]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.tensors = {}


@contextlib.contextmanager
def shards_to_gpu(device: str = "cuda:0"):
    """While inside, Transformers' shard loading reads each shard to the GPU and drops its pages.

    Only the call that loads model shards (it passes a backend) is replaced;
    every other use of safe_open goes to safetensors.
    """
    from transformers import modeling_utils

    real = modeling_utils.safe_open

    def safe_open(filename, *args, **kwargs):
        if "backend" in kwargs:
            return _ShardOnGpu(real, filename, device)
        return real(filename, *args, **kwargs)

    modeling_utils.safe_open = safe_open
    try:
        yield
    finally:
        modeling_utils.safe_open = real
