# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Example boundaries in the linear-attention layers of a padding-free batch.

A gated DeltaNet layer (Qwen3.5, Qwen3.6, Qwen3-Next) runs a short causal
convolution and then the gated delta rule. In a padding-free batch both must
restart at each example. Transformers gives both the boundaries
(cu_seq_lens_q, from TRL's padding-free collator): the delta rule honours them
when it runs on flash-linear-attention's kernel; the convolution honours them
only on the causal-conv1d package, and its PyTorch version runs across them, so
the first tokens of an example would see the end of the one before.

meepsquared replaces the convolution of such a model's module with one that, given
boundaries, runs flash-linear-attention's causal convolution over them, and
otherwise runs the module's own.
"""

import sys

_PATCHED = "_meepsquared_boundaries"


def available() -> bool:
    """flash-linear-attention's kernels import: the gated delta rule and the causal convolution.

    The package flash-linear-attention ships them in a second package, fla-core;
    without it "fla" imports but its kernels do not, and Transformers quietly runs
    the PyTorch delta rule, which ignores the boundaries.
    """
    try:
        from fla.modules.conv import causal_conv1d  # noqa: F401
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    except ImportError:
        return False
    return True


def _with_boundaries(original):
    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        cu_seqlens = kwargs.get("cu_seq_lens_q")
        if cu_seqlens is None:
            return original(hidden_states, weight, bias, activation=activation, **kwargs)
        from fla.modules.conv import causal_conv1d

        # Transformers lays the input out as [batch, channels, time]; fla as [batch, time, channels]
        out, _ = causal_conv1d(hidden_states.transpose(1, 2).contiguous(), weight=weight, bias=bias,
                               activation=activation, cu_seqlens=cu_seqlens)
        return out.transpose(1, 2)

    setattr(causal_conv1d_fn, _PATCHED, True)
    return causal_conv1d_fn


def patch(model) -> int:
    """The convolution with boundaries in every module of the model that has
    linear-attention layers; returns how many modules were patched now."""
    patched = 0
    for module_name in {type(m).__module__ for m in model.modules()}:
        source = sys.modules.get(module_name)
        convolution = getattr(source, "causal_conv1d_fn", None)
        if convolution is None or getattr(convolution, _PATCHED, False):
            continue
        source.causal_conv1d_fn = _with_boundaries(convolution)
        patched += 1
    return patched
