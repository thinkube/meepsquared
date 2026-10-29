# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed batches on PyTorch's memory-efficient attention kernel: attn_implementation="meepsquared_varlen".

FlashAttention 2 and 3 compute heads of at most 256 dimensions. Gemma 4's
full-attention layers have heads of 512, so with FlashAttention it cannot
train on packed (padding-free) batches, and SDPA computes every padded
position. PyTorch's memory-efficient attention kernel (the CUTLASS kernel
behind SDPA's EFFICIENT_ATTENTION backend) takes the boundaries of the
examples of a packed batch as cu_seqlens, as FlashAttention's varlen function
does, and computes heads of any width that is a multiple of 8 (16-bit inputs)
or 4 (float32). SDPA does not pass cu_seqlens to it, so meepsquared calls its
forward and backward ops, aten._efficient_attention_forward and
aten._efficient_attention_backward, through an autograd Function.

What the attention function computes:

- A packed batch (one row, position ids that restart at each example, or
  TRL 1.16's cu_seq_lens_q and max_length_q): each example attends to itself
  only. The boundaries are read from the position ids as Transformers'
  FlashAttention path reads them; trailing padding (position ids of 0) is a
  run of one-token examples, which the loss ignores.
- A padded batch (a 2D attention mask with padding, which Transformers'
  FlashAttention mask function passes on): the real tokens of each row are
  taken out, computed as one packed row, and put back; padded positions get
  zeros.
- Rows without padding: each row one sequence.

The mask is causal from the top left when queries and keys are as many, and
from the bottom right when there are fewer queries (generation with a cache).
A sliding window of W lets each query see itself and the W - 1 keys before it,
as Transformers' masks do; the kernel's window_size is W. Grouped-query
attention: the kernel takes as many key and value heads as query heads, so
the key and value heads are repeated for it and their gradients summed back
over each group. Dropout goes to the kernel, which keeps its random state for
the backward pass.

What the kernel cannot do is refused by name, never substituted: logit
soft-capping, attention sinks, a sliding window without a causal mask, a
paged cache, a 4D attention mask, inputs off a CUDA GPU or in a dtype it does
not compute, and head widths it does not align.
"""

import functools
from typing import Optional

import torch

NAME = "meepsquared_varlen"
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The kernel's mask types (sdp::CustomMaskType)
NO_MASK, CAUSAL_TOP_LEFT, CAUSAL_BOTTOM_RIGHT = 0, 1, 2


def alignment(dtype: torch.dtype) -> int:
    """The multiple a head's width must be: 16 bytes of the dtype (sdp_utils' minimum_gemm_alignment)."""
    return 4 if dtype == torch.float32 else 8


def check_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    """Refuse inputs the kernel does not compute, saying why. Shapes are (batch, heads, length, width)."""
    if query.dtype not in DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"attn_implementation={NAME!r}: the kernel computes float16, bfloat16 or float32 queries, "
                         f"keys and values of one dtype; they are {query.dtype}, {key.dtype}, {value.dtype}")
    if key.shape[-1] != query.shape[-1]:
        raise ValueError(f"attn_implementation={NAME!r}: queries have heads of {query.shape[-1]} dimensions and keys "
                         f"of {key.shape[-1]}; the kernel needs them equal")
    multiple = alignment(query.dtype)
    for name, width in (("query", query.shape[-1]), ("value", value.shape[-1])):
        if width % multiple:
            raise ValueError(f"attn_implementation={NAME!r}: {name} heads of {width} dimensions; the kernel computes "
                             f"{query.dtype} heads whose width is a multiple of {multiple}")
    if query.shape[1] % key.shape[1]:
        raise ValueError(f"attn_implementation={NAME!r}: {query.shape[1]} query heads do not share "
                         f"{key.shape[1]} key heads evenly")
    if query.device.type != "cuda":
        raise ValueError(f"attn_implementation={NAME!r}: PyTorch's memory-efficient kernel runs on a CUDA GPU, and "
                         f"the attention's inputs are on {query.device}")


def boundaries(batch_size: int, position_ids: Optional[torch.Tensor] = None, cu_seq_lens_q=None,
               cu_seq_lens_k=None, max_length_q=None, max_length_k=None):
    """The examples of a packed batch, (cu_seqlens_q, cu_seqlens_k, max_q, max_k); None for a batch not packed.

    TRL 1.16's padding-free collator gives the four; else they are read from
    the position ids, which restart at each example, as Transformers'
    FlashAttention path reads them (prepare_fa_kwargs_from_position_ids), for a
    single row whose position ids do not simply count up.
    """
    from transformers.modeling_flash_attention_utils import (_is_packed_sequence,
                                                             prepare_fa_kwargs_from_position_ids)

    given = {"cu_seq_lens_q": cu_seq_lens_q, "cu_seq_lens_k": cu_seq_lens_k, "max_length_q": max_length_q,
             "max_length_k": max_length_k}
    missing = [name for name, value in given.items() if value is None]
    if not missing:
        return (cu_seq_lens_q.to(torch.int32), cu_seq_lens_k.to(torch.int32), int(max_length_q), int(max_length_k))
    if len(missing) < len(given):
        raise ValueError(f"attn_implementation={NAME!r}: the batch gives {sorted(set(given) - set(missing))} without "
                         f"{missing}; a packed batch's boundaries need all four")
    if not _is_packed_sequence(position_ids, batch_size):
        return None
    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(position_ids)
    return cu_q, cu_k, int(max_q), int(max_k)


def _expand(x: torch.Tensor, group: int) -> torch.Tensor:
    """Key or value heads, (batch, length, heads, width), each repeated for the query heads of its group."""
    return x if group == 1 else x.repeat_interleave(group, dim=2)


def _fold(grad: torch.Tensor, group: int) -> torch.Tensor:
    """The gradient of repeated heads summed back over each group."""
    if group == 1:
        return grad
    batch, length, heads, width = grad.shape
    return grad.view(batch, length, heads // group, group, width).sum(3)


class EfficientAttention(torch.autograd.Function):
    """The kernel's forward and backward ops. Inputs and output are (batch, length, heads, width).

    With cu_seqlens the batch is one row of examples laid end to end. The
    forward saves the output, the log-sum-exp and the dropout's random state
    for the backward op; the keys and values are saved with their own heads
    and repeated again in the backward pass.
    """

    @staticmethod
    def forward(ctx, query, key, value, cu_q, cu_k, max_q: int, max_k: int, dropout: float, mask_type: int,
                scale: float, window: Optional[int]):
        group = query.shape[2] // key.shape[2]
        out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
            query, _expand(key, group), _expand(value, group), None, cu_q, cu_k, max_q, max_k, dropout, mask_type,
            True, scale=scale, window_size=window)
        ctx.save_for_backward(query, key, value, out, lse, seed, offset, cu_q, cu_k)
        ctx.settings = (group, max_q, max_k, dropout, mask_type, scale, window)
        return out

    @staticmethod
    def backward(ctx, grad):
        query, key, value, out, lse, seed, offset, cu_q, cu_k = ctx.saved_tensors
        group, max_q, max_k, dropout, mask_type, scale, window = ctx.settings
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            grad.contiguous(), query, _expand(key, group), _expand(value, group), None, out, cu_q, cu_k, max_q, max_k,
            lse, dropout, seed, offset, mask_type, False, scale=scale, window_size=window)
        return dq, _fold(dk, group), _fold(dv, group), None, None, None, None, None, None, None, None


def _mask_type(causal: bool, query_length: int, key_length: int) -> int:
    if not causal:
        return NO_MASK
    return CAUSAL_TOP_LEFT if query_length == key_length else CAUSAL_BOTTOM_RIGHT


def _padded(q, k, v, attention_mask, causal, dropout, scale, window):
    """A padded batch: the real tokens of every row as one packed row, the output put back in place."""
    batch, query_length, heads, width = q.shape
    key_length = k.shape[1]
    if attention_mask.shape != (batch, key_length):
        raise ValueError(f"attn_implementation={NAME!r}: an attention mask of shape {tuple(attention_mask.shape)} "
                         f"for {batch} rows of {key_length} keys")
    keys_kept = attention_mask.bool()
    queries_kept = keys_kept[:, -query_length:]

    def starts(kept):
        return torch.nn.functional.pad(kept.sum(-1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))

    cu_q, cu_k = starts(queries_kept), starts(keys_kept)
    max_q, max_k = int(queries_kept.sum(-1).max()), int(keys_kept.sum(-1).max())
    query_index = queries_kept.flatten().nonzero().flatten()
    key_index = keys_kept.flatten().nonzero().flatten()

    def taken(x, index):
        return x.reshape(-1, *x.shape[2:]).index_select(0, index)[None]

    out = EfficientAttention.apply(taken(q, query_index), taken(k, key_index), taken(v, key_index), cu_q, cu_k,
                                   max_q, max_k, dropout, _mask_type(causal, query_length, key_length), scale, window)
    placed = out.new_zeros(batch * query_length, heads, v.shape[-1]).index_put((query_index,), out[0])
    return placed.view(batch, query_length, heads, v.shape[-1])


def attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None,
                      sliding_window=None, softcap=None, is_causal=None, s_aux=None, cache=None, **kwargs):
    """The attention function Transformers calls, as ALL_ATTENTION_FUNCTIONS[NAME].

    query (batch, heads, length, width), key and value (batch, key heads,
    length, width), as Transformers' attention functions take them; returns
    the output as (batch, length, heads, width) and no weights.
    """
    if softcap is not None:
        raise ValueError(f"attn_implementation={NAME!r}: this model soft-caps its attention logits (softcap="
                         f"{softcap}), and the kernel does not; load it with attn_implementation='sdpa'")
    if s_aux is not None:
        raise ValueError(f"attn_implementation={NAME!r}: this model has attention sinks (s_aux), and the kernel "
                         "has none; load it with attn_implementation='sdpa'")
    if cache is not None:
        raise ValueError(f"attn_implementation={NAME!r}: a paged cache (continuous batching) is not computed by "
                         "this attention; generate with attn_implementation='sdpa'")
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    if sliding_window is not None and not causal:
        raise ValueError(f"attn_implementation={NAME!r}: the kernel's sliding window ({sliding_window}) needs a "
                         "causal mask, and this attention is bidirectional")
    if attention_mask is not None and attention_mask.dim() != 2:
        raise ValueError(f"attn_implementation={NAME!r}: a {attention_mask.dim()}D attention mask; the attention "
                         "takes a padding mask of (batch, keys) or none")
    check_inputs(query, key, value)
    q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    batch, query_length, heads, _ = q.shape
    key_length = k.shape[1]
    scale = scaling if scaling is not None else q.shape[-1] ** -0.5
    window = int(sliding_window) if sliding_window is not None else None
    if attention_mask is not None:
        return _padded(q, k, v, attention_mask, causal, dropout, scale, window), None
    packed = boundaries(batch, kwargs.get("position_ids"), kwargs.get("cu_seq_lens_q"), kwargs.get("cu_seq_lens_k"),
                        kwargs.get("max_length_q"), kwargs.get("max_length_k"))
    if packed is None:
        out = EfficientAttention.apply(q, k, v, None, None, query_length, key_length, dropout,
                                       _mask_type(causal, query_length, key_length), scale, window)
        return out, None
    cu_q, cu_k, max_q, max_k = packed
    q, k, v = (x.reshape(1, -1, *x.shape[2:]) for x in (q, k, v))
    out = EfficientAttention.apply(q, k, v, cu_q.to(q.device), cu_k.to(q.device), max_q, max_k, dropout,
                                   _mask_type(causal, query_length, key_length), scale, window)
    return out.view(batch, query_length, heads, -1), None


@functools.lru_cache(maxsize=None)
def register() -> None:
    """NAME as an attention implementation of Transformers, with FlashAttention's mask function.

    That mask function gives no mask for a batch without padding, and the 2D
    padding mask otherwise, which the attention unpads; it never builds a 4D
    mask.
    """
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask

    AttentionInterface.register(NAME, attention_forward)
    AttentionMaskInterface.register(NAME, flash_attention_mask)
