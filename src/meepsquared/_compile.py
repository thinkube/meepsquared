# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Each decoder layer compiled as one region: get_peft_model(compile_layers=True).

A model with many layers and many small operations per layer (Gemma 4: 58
layers) keeps the GPU waiting for the CPU to launch its kernels. torch.compile
turns the element-wise operations of a layer into a few fused kernels, as
Unsloth does. Shapes are dynamic, so padding-free batches of changing length
do not recompile. The layers are compiled after the trainer is built, once
the fused LoRA kernels are on their modules; those kernels, FlashAttention,
meepsquared's memory-efficient attention (see meepsquared._varlen) and bitsandbytes'
4-bit layers stay outside the compiled region, and Liger's
in-layer patches are left off (see meepsquared._trl), as Inductor fuses what they
fuse.
"""

import torch

from meepsquared import _checkpointing

_ORIGINAL = "_meepsquared_uncompiled_forward"
# Recorded on the model by apply: the backend its layers were compiled with
_BACKEND = "_meepsquared_compile_backend"
# Attention functions left outside the compiled region: Inductor fails on
# FlashAttention's custom op in the backward pass (KeyError in its scheduler);
# meepsquared_varlen's autograd Function calls the kernel's ops directly and reads
# the batch's boundaries as Python integers
OUTSIDE = ("flash_attention_2", "flash_attention_3", "meepsquared_varlen")


def _attention_outside(model) -> None:
    """The model's attention function marked for Dynamo to break the graph around, where it is one of OUTSIDE."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    implementation = model.config._attn_implementation
    if implementation in OUTSIDE and not getattr(ALL_ATTENTION_FUNCTIONS[implementation], "_meepsquared_outside", False):
        function = torch.compiler.disable(ALL_ATTENTION_FUNCTIONS[implementation])
        function._meepsquared_outside = True
        ALL_ATTENTION_FUNCTIONS[implementation] = function


def decision(config, attn_implementation: str) -> tuple[bool, str]:
    """Whether compiling the layers pays, for compile_layers="auto", and why.

    It pays where the GPU waits for the CPU: a model whose attention runs
    inside the compiled region (SDPA; Gemma 4 went 1.8 times faster). It
    gains nothing with FlashAttention, which stays outside the region (Llama
    3.1 8B), and a model with linear-attention layers (Qwen3.5) recompiles
    at every step, as their shapes change. meepsquared's memory-efficient
    attention also stays outside, and each layer compiles as the two pieces
    around it: the models it is chosen for (Gemma 4) have many small
    operations per layer, which the compiled pieces fuse.
    """
    from meepsquared import _varlen
    from meepsquared.models import FLASH_ATTENTION, has_linear_attention

    if not torch.cuda.is_available():
        return False, "no CUDA GPU"
    if attn_implementation in FLASH_ATTENTION:
        return False, "FlashAttention runs outside them"
    if has_linear_attention(config):
        return False, "linear-attention layers would recompile at every step"
    if attn_implementation == _varlen.NAME:
        return True, "the attention runs between two compiled pieces of each layer"
    return True, f"the attention ({attn_implementation}) runs inside them"


def wanted(config, attn_implementation: str) -> bool:
    """Whether compiling the layers pays, for compile_layers="auto" (see decision)."""
    return decision(config, attn_implementation)[0]


def _quantized_outside() -> None:
    """bitsandbytes' 4-bit layers marked for Dynamo to break the graph around.

    A QLoRA model keeps 4-bit layers that are not LoRA targets (Gemma 4's
    per-layer embedding projections); AOTAutograd cannot trace their gemm_4bit
    custom op.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        return
    if not getattr(bnb.nn.Linear4bit.forward, "_meepsquared_outside", False):
        forward = torch.compiler.disable(bnb.nn.Linear4bit.forward)
        forward._meepsquared_outside = True
        bnb.nn.Linear4bit.forward = forward


def recompile_limit_for(layer_count: int) -> int:
    """The recompile limit that lets every kind of layer of a model keep its compiled code.

    Dynamo keeps at most recompile_limit compiled versions of one function, and
    the layers of a class share their forward's code. Layers of one class can
    still differ in what Dynamo guards on: Gemma 4's layers have heads 256 or
    512 wide, some keep the full key-value history, some have no key
    projection. Each kind is one more version, and so is each piece the forward
    is cut into at a graph break. Past the default limit of 8, Dynamo runs the
    remaining kinds uncompiled, and which ones depends on the order the run met
    them. Two versions per layer covers every layer being its own kind, at two
    pieces each.
    """
    return 2 * layer_count


def _allow_every_kind(layer_count: int) -> None:
    import torch._dynamo.config as dynamo_config

    wanted = recompile_limit_for(layer_count)
    if dynamo_config.recompile_limit < wanted:
        dynamo_config.recompile_limit = wanted
    if dynamo_config.accumulated_recompile_limit < 4 * wanted:
        dynamo_config.accumulated_recompile_limit = 4 * wanted


def apply(model, backend: str = "inductor") -> int:
    """Compile the forward of the decoder layers' class; returns the layer count. Applying again changes nothing.

    One compiled function per layer class, not per layer: the layers share
    their code and differ in their weights, which Dynamo takes as inputs, so
    a new sequence length compiles once for all of them, not once per layer.
    Dynamo's recompile limit is raised so that every kind of layer within a
    class keeps its compiled code (see recompile_limit_for).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    setattr(base, _BACKEND, backend)
    _attention_outside(base)
    _quantized_outside()
    all_layers = _checkpointing.layers(base)
    _allow_every_kind(len(all_layers))
    for cls in {type(layer) for layer in all_layers}:
        if _ORIGINAL not in cls.__dict__:
            setattr(cls, _ORIGINAL, cls.forward)
            cls.forward = torch.compile(cls.forward, dynamic=True, backend=backend)
    return len(all_layers)


def uncompiled(model) -> None:
    """The layers' forward as it was, for generation after a training with compiled layers."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for cls in {type(layer) for layer in _checkpointing.layers(base)}:
        if _ORIGINAL in cls.__dict__:
            cls.forward = getattr(cls, _ORIGINAL)
            delattr(cls, _ORIGINAL)


def compiled(model) -> bool:
    """Whether the forward of the model's decoder layers is compiled now."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return any(_ORIGINAL in cls.__dict__ for cls in {type(layer) for layer in _checkpointing.layers(base)})


def recompile(model) -> bool:
    """The layers compiled again, with the backend apply used, after uncompiled; returns whether it compiled them.

    Nothing happens to a model whose layers apply never compiled (no trainer
    built yet, or compile_layers off), or whose layers are compiled now.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    backend = getattr(base, _BACKEND, None)
    if backend is None or compiled(model):
        return False
    apply(model, backend=backend)
    return True
