# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""PyTorch's CUDA allocator with expandable segments.

PyTorch keeps freed GPU memory in fixed-size segments for reuse. A training
whose tensors change shape from step to step (padding-free batches of
changing length, the routed rows of a mixture of experts) leaves those
segments in pieces that the next request does not fit, so memory that is free
cannot be used. With expandable segments a segment grows in place instead,
which keeps the reserved memory close to what is in use; auto checkpointing
counts reserved but unused memory as free, so it also reads truer.

meepsquared turns it on when it is imported, unless something else decided:
- before CUDA starts, through PyTorch's own setting, PYTORCH_ALLOC_CONF;
- once CUDA has started (importing TRL or bitsandbytes can start it), through
  torch.cuda.memory._set_allocator_settings, which changes the setting for the
  segments the allocator creates from then on (vLLM switches expandable
  segments with it);
- not when PYTORCH_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF is already set: the
  user's setting stands;
- not on Windows or under WSL, where PyTorch does not support it, nor on an
  NVIDIA Jetson board, whose driver refuses the virtual-memory calls it is
  built on. The DGX Spark's GB10 supports them.
The decision and its reason are kept in DECISION and printed in the banner.
"""

import os
import sys

VARIABLES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
SETTING = "expandable_segments:True"
JETSON_RELEASE_FILE = "/etc/nv_tegra_release"

DECISION = "not decided"


def _cuda_started() -> bool:
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except AttributeError:
        return False


def _unsupported_platform() -> str:
    """Why this platform cannot use expandable segments, or "" when it can."""
    if os.name == "nt":
        return "Windows"
    try:
        if "microsoft" in os.uname().release.lower():
            return "WSL"
    except AttributeError:
        pass
    if os.path.exists(JETSON_RELEASE_FILE):
        return "a Jetson board"
    return ""


def _set_running_allocator(setting: str) -> None:
    import torch

    torch.cuda.memory._set_allocator_settings(setting)


def decide(environ=None, cuda_started=None, unsupported=None, set_running=None) -> str:
    """Turn expandable segments on where that is right; returns what was decided, and why.

    The keyword arguments are for tests; by default the process's environment,
    CUDA state and platform are read, and a running allocator is changed in place.
    """
    environ = os.environ if environ is None else environ
    cuda_started = _cuda_started() if cuda_started is None else cuda_started
    unsupported = _unsupported_platform() if unsupported is None else unsupported
    set_running = _set_running_allocator if set_running is None else set_running
    given = [name for name in VARIABLES if name in environ]
    if given:
        return f"as set in {given[0]}"
    if unsupported:
        return f"off (not supported on {unsupported})"
    environ["PYTORCH_ALLOC_CONF"] = SETTING
    if cuda_started:
        set_running(SETTING)
    return "expandable segments"


def apply() -> None:
    global DECISION
    DECISION = decide()
