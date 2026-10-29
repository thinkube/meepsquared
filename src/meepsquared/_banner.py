# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The banner from_pretrained prints once a model is loaded: what meepsquared will
train it with. MEEPSQUARED_QUIET=1 in the environment turns it off."""

import os

MARK = ["╭────────────╮   ", "│ meep meep! │   ", "╰──┬─────────╯   ", "   ╵ ∘ ○  ·  ·  ·"]


def _gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        return "none"
    major, minor = torch.cuda.get_device_capability(0)
    return f"{torch.cuda.get_device_name(0)} (sm_{major}{minor})"


def _allocator() -> str:
    from meepsquared import _allocator as allocator

    return allocator.DECISION


def _versions() -> str:
    import torch
    import transformers
    import trl

    return f"PyTorch {torch.__version__.split('+')[0]}. Transformers {transformers.__version__}. TRL {trl.__version__}"


def text(version: str, model_type: str, weights: str, liger: bool, padding_free: bool, padding_free_reason: str) -> str:
    padding = "on" if padding_free else f"off ({padding_free_reason})"
    lines = [
        f"Meep² {version}: fast LoRA and QLoRA on Transformers, PEFT and TRL",
        f"GPU: {_gpu()}, allocator: {_allocator()}. {_versions()}",
        f"Model: {model_type}, {weights}. Liger: {'on' if liger else 'off'}. Padding-free: {padding}",
        "Apache-2.0: https://github.com/thinkube/meepsquared",
    ]
    return "\n".join(f"{mark}   {line}" for mark, line in zip(MARK, lines))


def show(**kwargs) -> None:
    if os.environ.get("MEEPSQUARED_QUIET") == "1":
        return
    from meepsquared import __version__

    print(text(__version__, **kwargs), flush=True)
