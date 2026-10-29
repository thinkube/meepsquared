# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Selective recomputation: a checkpointed layer that keeps the outputs of its large products.

A checkpointed layer runs its forward again in the backward pass. PyTorch's
selective activation checkpointing (torch.utils.checkpoint, context_fn from
create_selective_checkpoint_contexts) lets a policy decide per operation:
MUST_SAVE keeps the operation's output from the forward pass and returns it
when the layer is recomputed; PREFER_RECOMPUTE runs the operation again. The
policy here keeps the large products of a layer, the matrix multiplications
and the attention, and recomputes everything else: normalization, rotary
embeddings, activation functions, element-wise work, and LoRA's own products,
which are small. Memory is saved at a fraction of a whole layer's recompute.

The policy sees operations as PyTorch dispatches them (aten.mm, aten.bmm,
aten._scaled_dot_product_flash_attention, ...), and the custom operations
libraries register with torch.library (FlashAttention's forward). It does not
see inside a Triton kernel launched directly: Liger's kernels, the activation
kernels of the fused LoRA MLP and meepsquared's own run again when the layer is
recomputed, which is what the policy wants of them. A custom autograd
Function (the fused LoRA kernels, Liger's) runs its forward again when the
layer is recomputed, and the products inside it are kept or recomputed one by
one like any other.

Two things break the policy, and are refused or kept off where they occur:

- An in-place write over a kept output. The forward and the recompute would
  see different values, and PyTorch's version check stops the training. The
  fused LoRA kernels and the routed-expert LoRA add their adapter products
  out of place while a selective layer computes (in_place), and in place
  everywhere else, where it saves a pass over each projection's output.
  Liger's rotary embedding, whose kernel rotates q and k in place over the
  projections' outputs, is left off when the layout trained has a selective
  layer (see meepsquared._trl); Liger patches the model's module, so it is off for
  every layer then, and on for every layer otherwise.
- A compiled layer. The layer's forward runs as one compiled region, and the
  checkpoint around it does not reach the operations inside; a layer is
  either compiled or selectively recomputed.
"""

import functools
import importlib
import threading
from typing import Callable, Optional

import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

# FlashAttention's forward as a PyTorch op, by attention implementation: the
# op namespace, the module whose import registers it (the one Transformers
# imports), and the op names. flash-attn 2.7 and later register them;
# FlashAttention 3 installs as the module flash_attn_interface.
FLASH_ATTENTION_OPS = {
    "flash_attention_2": ("flash_attn", "flash_attn.flash_attn_interface",
                          ("_flash_attn_forward", "_flash_attn_varlen_forward")),
    "flash_attention_3": ("flash_attn_3", "flash_attn_interface", ("_flash_attn_forward",)),
}


@functools.lru_cache(maxsize=None)
def products() -> dict:
    """The product operations, each with the positions of its two factors in the op's arguments.

    A product's inner dimension is the last of its left factor and its output
    width the last of its right one; the policy reads both to tell LoRA's small
    products from the layer's large ones. Transformers' experts run their
    grouped product through its own op where PyTorch's grouped_mm does not
    apply (the CPU); it is a product like the others, where the installed
    Transformers registers it.
    """
    import transformers.integrations.moe  # noqa: F401  registers torch.ops.transformers

    aten = torch.ops.aten
    found = {aten.mm.default: (0, 1), aten.addmm.default: (1, 2), aten.bmm.default: (0, 1),
             aten.baddbmm.default: (1, 2), aten._scaled_mm.default: (0, 1), aten._grouped_mm.default: (0, 1)}
    fallback = getattr(torch.ops.transformers, "grouped_mm_fallback", None)
    if fallback is not None:
        found[fallback.default] = (0, 1)
    return found


@functools.lru_cache(maxsize=None)
def attention_ops() -> frozenset:
    """PyTorch's fused attention operations: the kernels scaled_dot_product_attention dispatches to, and the
    memory-efficient kernel's own forward, which meepsquared_varlen calls (see meepsquared._varlen)."""
    aten = torch.ops.aten
    return frozenset({aten._efficient_attention_forward.default,
                      aten._scaled_dot_product_flash_attention.default,
                      aten._scaled_dot_product_efficient_attention.default,
                      aten._scaled_dot_product_cudnn_attention.default,
                      aten._scaled_dot_product_flash_attention_for_cpu.default,
                      aten._scaled_dot_product_fused_attention_overrideable.default})


def flash_attention_ops(implementation: str) -> Optional[list]:
    """FlashAttention's forward ops for an attention implementation: [] where it is not FlashAttention, None where
    the installed FlashAttention does not register them."""
    if implementation not in FLASH_ATTENTION_OPS:
        return []
    namespace, module, names = FLASH_ATTENTION_OPS[implementation]
    try:
        importlib.import_module(module)
    except ImportError:
        return None
    try:
        return [getattr(getattr(torch.ops, namespace), name).default for name in names]
    except (AttributeError, RuntimeError):
        return None


class Policy:
    """Which operations of a layer keep their output: products wider than the LoRA rank, and attention.

    A product with an inner dimension or an output width of at most `rank` is
    one of LoRA's own (x @ A, or (x @ A) @ B): cheap, and recomputed. Every
    other product and the attention kernels keep their output. `observe`, when
    set, is called with (context, op, decision) for every operation the policy
    sees, in the forward pass and in the recompute.
    """

    def __init__(self, rank: int, saved: frozenset):
        self.rank = rank
        self.saved = saved
        self.observe: Optional[Callable] = None

    def decide(self, op, args) -> CheckpointPolicy:
        factors = products().get(op)
        if factors is not None:
            left, right = args[factors[0]], args[factors[1]]
            if min(left.shape[-1], right.shape[-1]) <= self.rank:
                return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.MUST_SAVE
        if op in self.saved:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def __call__(self, ctx, op, *args, **kwargs) -> CheckpointPolicy:
        decision = self.decide(op, args)
        if self.observe is not None:
            self.observe(ctx, op, decision)
        return decision


def lora_rank(model) -> int:
    """The largest LoRA rank on the model; 0 where it has no adapter yet."""
    from peft.tuners.lora import LoraLayer

    return max((r for module in model.modules() if isinstance(module, LoraLayer) for r in module.r.values()),
               default=0)


def _base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def refusal(model, compiled: bool) -> Optional[str]:
    """Why the model's layers cannot be recomputed selectively, or None when they can."""
    from meepsquared import _linear_attention
    from meepsquared.models import has_linear_attention

    base = _base(model)
    if compiled:
        return "the layers are compiled, and the policy does not reach inside a compiled region"
    if has_linear_attention(base.config) and _linear_attention.available():
        return ("the linear-attention layers run flash-linear-attention's delta-rule kernel, which the policy "
                "cannot see; selective recomputation is not verified on them")
    implementation = base.config._attn_implementation
    if flash_attention_ops(implementation) is None:
        return (f"this {implementation} does not register its forward as a PyTorch op "
                f"(torch.ops.{FLASH_ATTENTION_OPS[implementation][0]}), so the policy cannot keep the attention; "
                "flash-attn 2.7 and later do")
    return None


def policy_for(model) -> Policy:
    """The policy for a model: its LoRA rank, PyTorch's attention kernels and its FlashAttention's op."""
    base = _base(model)
    flash = flash_attention_ops(base.config._attn_implementation)
    if flash is None:
        raise ValueError(f"selective recomputation: {refusal(model, compiled=False)}")
    return Policy(lora_rank(model), attention_ops() | frozenset(flash))


def context_fn(policy: Policy):
    """The context_fn torch.utils.checkpoint takes: fresh caching contexts for each checkpointed call."""
    return functools.partial(create_selective_checkpoint_contexts, policy)


# How many selectively recomputed layers are computing now in this thread (their forward, or its recompute)
_RUNNING = threading.local()


def in_place() -> bool:
    """Whether a forward may add into a product's output in place.

    Everywhere but inside a selectively recomputed layer, where the policy keeps
    that output for the backward pass and a write over it would change what the
    recompute reads. Elsewhere the in-place add saves a pass over the output: an
    out-of-place torch.addmm copies its input into a new tensor before the
    product (PyTorch's addmm_out_cuda_impl). Compiled code is never selective
    (see refusal) and does not read the flag.
    """
    return torch.compiler.is_compiling() or getattr(_RUNNING, "depth", 0) == 0


def _flagged(function):
    """function, run with the flag of a selectively recomputed layer set: in the forward and in the recompute."""
    def run(*args, **kwargs):
        _RUNNING.depth = getattr(_RUNNING, "depth", 0) + 1
        try:
            return function(*args, **kwargs)
        finally:
            _RUNNING.depth -= 1

    return run


class SelectiveCheckpoint:
    """The checkpoint function of a selectively recomputed layer, in the form Transformers' layers call:
    (function, *args) checkpointed non-reentrantly with the policy's caching contexts."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self.context_fn = context_fn(policy)

    def __call__(self, function, *args, **kwargs):
        return checkpoint(_flagged(function), *args, use_reentrant=False, context_fn=self.context_fn, **kwargs)


def checkpoint_function(model) -> SelectiveCheckpoint:
    """The checkpoint function of a selectively recomputed layer, with the policy for this model."""
    return SelectiveCheckpoint(policy_for(model))


def whole_checkpoint_function():
    """The checkpoint function of a layer that recomputes its whole forward."""
    return functools.partial(checkpoint, use_reentrant=False)
