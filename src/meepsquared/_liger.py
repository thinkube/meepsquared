# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Liger Kernel for a loaded model, with the loss chunks sized for training speed.

Liger's fused linear cross entropy computes the output layer and the loss in
chunks of tokens, never holding the logits of the whole batch. It sizes a
chunk as tokens / ceil(vocabulary / (chunk_mem_const * hidden size)), with
chunk_mem_const 1. With a large vocabulary that gives chunks of a few dozen
tokens: on a 256K-vocabulary model, about 125 thin matmuls a step, which a GPU
runs far below its speed. A budget of 32 makes the chunks 32 times larger: the
same loss in a few large matmuls, for some hundred MB more memory. Liger takes
chunk_mem_const as an argument of its ops function but not from the model
patches, so the budget is set there, for every model in the process.

Liger's rotary embedding runs on its Triton kernel: Liger 0.8 picks its CuTe
DSL kernel where NVIDIA's CUTLASS DSL is installed, and that kernel failed on
an NVIDIA GB10 at step 136 of a Llama 3.1 8B training with a batch shape it
had not seen (DLTensorWrapper given None). Liger chooses by environment
variable; a value set before meepsquared runs is kept.
"""

import os

_budget = {"value": 1, "patched": False}


def set_loss_chunk_budget(value: int) -> None:
    """The chunk_mem_const every fused loss in this process uses."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"loss_chunk_budget must be a positive integer, not {value!r}")
    import liger_kernel.ops.fused_linear_cross_entropy as flce

    if not _budget["patched"]:
        forward = flce.fused_linear_cross_entropy_forward

        def with_budget(*args, **kwargs):
            kwargs["chunk_mem_const"] = _budget["value"]
            return forward(*args, **kwargs)

        flce.fused_linear_cross_entropy_forward = with_budget
        _budget["patched"] = True
    _budget["value"] = value


def prepare(model, loss_chunk_budget: int) -> None:
    """Check that Liger has patches for the model, and set the chunk budget.

    The patches themselves are applied by the trainer (use_liger_kernel), which
    then also knows that the loss leaves no logits to compute metrics from.
    """
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    model_type = model.config.model_type
    if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN:
        raise NotImplementedError(
            f"Liger Kernel has no patches for model type {model_type!r}; "
            "load it with liger=False to train it without them"
        )
    set_loss_chunk_budget(loss_chunk_budget)
    os.environ.setdefault("LIGER_KERNEL_IMPL_ROPE", "nvidia-triton")
