# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused LoRA kernels on a PEFT model: get_peft_model(fused_lora=True).

The kernels (Axolotl's, in _vendor/axolotl) compute a projection and its LoRA
adapter in one autograd function with a hand-written backward pass, and an
MLP's gate, up and down projections with its gated activation as one: SiLU
(Llama, Qwen), the exact GELU, or the tanh GELU of Gemma (meepsquared._geglu_tanh).

They are attached to modules, not to source code: an MLP with LoRA on
gate_proj, up_proj and down_proj gets the fused MLP as its forward, an
attention with LoRA on q_proj, k_proj and v_proj computes the three in one
kernel, and every other LoRA projection on a linear layer gets the fused
projection. A model whose MLP has another activation is refused.

Each forward is a function of the module bound to it, and what it needs
beyond the module is kept on the module: a deep copy of a fused model
computes on its own parameters, as copy.deepcopy rebinds a bound method
to the copy. A forward that closed over the module would run on the
original.
"""

import types

import torch
from torch import nn

MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _kernels():
    try:
        from meepsquared._vendor.axolotl import lora
    except ImportError as e:
        raise ImportError(f"fused_lora needs bitsandbytes and Triton, which meepsquared installs ({e})") from e
    return lora


def _mlp_geglu_tanh(self, X, inplace: bool = True):
    """Axolotl's fused LoRA MLP with the tanh GELU: apply_lora_mlp_swiglu, the activation kernels changed."""
    kernels = _kernels()
    from meepsquared import _geglu_tanh

    projections = (self.gate_proj, self.up_proj, self.down_proj)
    if any(kernels._requires_projection_forward(p) for p in projections):
        return kernels._projection_forward_mlp(self, X, "geglu_tanh")
    gate, up, down = (kernels.get_lora_parameters(p) for p in projections)
    X_drop = kernels._apply_dropout(gate[7], X, self.training)
    # Each projection's parameters, in the order LoRA_MLP takes them: W, bias, quant state, A, B, scale, LoRA bias, magnitude
    return kernels.LoRA_MLP.apply(X, X_drop, *gate[:7], gate[8], *up[:7], up[8], *down[:7], down[8],
                                  _geglu_tanh.forward, _geglu_tanh.backward, inplace)


def _mlp_forward(kernels, activation: str):
    """The fused MLP forward for an activation name, as Transformers' configs give it."""
    forwards = {"silu": kernels.apply_lora_mlp_swiglu, "gelu": kernels.apply_lora_mlp_geglu,
                "gelu_pytorch_tanh": _mlp_geglu_tanh}
    return forwards.get(activation)


def available() -> bool:
    """Whether the kernels can be imported (bitsandbytes and Triton)."""
    try:
        _kernels()
    except ImportError:
        return False
    return True


def supported_activation(config) -> tuple[bool, str]:
    """Whether the model's MLP activation has a fused kernel, and the activation's name."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    activation = getattr(text, "hidden_act", None) or getattr(text, "hidden_activation", None)
    return activation in ("silu", "gelu", "gelu_pytorch_tanh"), str(activation)


def _on_linear(module) -> bool:
    """A LoRA layer over a linear layer: full precision, FP8 or 4-bit."""
    from peft.tuners.lora import LoraLayer

    from meepsquared._fp8 import Fp8Linear

    if not isinstance(module, LoraLayer):
        return False
    base = module.get_base_layer()
    if isinstance(base, (nn.Linear, Fp8Linear)):
        return True
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return isinstance(base, bnb.nn.Linear4bit)


def _outside_compile(forward):
    """The forward left out of a compiled region: Dynamo cannot trace the kernels' autograd functions."""
    return torch.compiler.disable(forward)


_MLP_FORWARDS = {}


def _three_dimensional(forward):
    """The kernels take (batch, sequence, hidden); a mixture of experts gives its shared expert (tokens, hidden)."""
    def wrapped(self, X, *args, **kwargs):
        if X.dim() == 2:
            return forward(self, X.unsqueeze(0), *args, **kwargs).squeeze(0)
        return forward(self, X, *args, **kwargs)

    return wrapped


def _disabled_mlp_forward(activation: str, forward):
    """The fused MLP forward of an activation left out of compiled regions, made once so a module keeps the same one."""
    if activation not in _MLP_FORWARDS:
        _MLP_FORWARDS[activation] = _outside_compile(_three_dimensional(forward))
    return _MLP_FORWARDS[activation]


def _attach(module, forward, **marks) -> None:
    """forward, a function of (self, x, ...), as the module's own, with the marks as the module's attributes."""
    module.forward = types.MethodType(forward, module)
    for name, value in marks.items():
        setattr(module, name, value)


def _own_forward(module, x, *args, **kwargs):
    """The module's forward as its class defines it: PEFT's LoRA layer."""
    return type(module).forward(module, x, *args, **kwargs)


@_outside_compile
def _projection_forward(self, x, *args, **kwargs):
    kernels = _kernels()
    if args or kwargs or kernels._requires_projection_forward(self):
        return _own_forward(self, x, *args, **kwargs)
    return kernels.apply_lora_linear(self, x)


QKV_PROJECTIONS = ("q_proj", "k_proj", "v_proj")


class _QkvState:
    """What the three projections of one attention module share: the module, and the K and V q_proj computed."""

    def __init__(self, attention):
        self.attention = attention
        self.pending = {}


@_outside_compile
def _q_forward(self, x, *args, **kwargs):
    kernels = _kernels()
    state = self._meepsquared_qkv
    projections = [getattr(state.attention, name) for name in QKV_PROJECTIONS]
    if args or kwargs or any(kernels._requires_projection_forward(p) for p in projections):
        return _own_forward(self, x, *args, **kwargs)
    state.pending.clear()
    Q, K, V = kernels.apply_lora_qkv(state.attention, x)
    state.pending.update(x=x, k_proj=K, v_proj=V)
    return Q


def _handed_over(name):
    @_outside_compile
    def forward(self, x, *args, **kwargs):
        pending = self._meepsquared_qkv.pending
        if not args and not kwargs and pending.get("x") is x and name in pending:
            out = pending.pop(name)
            if "k_proj" not in pending and "v_proj" not in pending:
                pending.clear()
            return out
        return _own_forward(self, x, *args, **kwargs)

    return forward


_K_FORWARD, _V_FORWARD = _handed_over("k_proj"), _handed_over("v_proj")


def _fused_qkv(attention):
    """The three adapters of an attention module in one kernel, through its q_proj.

    Transformers' attention calls q_proj, k_proj and v_proj in turn on the same
    hidden states. q_proj computes the three with Axolotl's fused QKV kernel
    and keeps K and V; k_proj and v_proj hand them over when they are called
    with that same input, and compute their own otherwise.
    """
    state = _QkvState(attention)
    for name, forward in zip(QKV_PROJECTIONS, (_q_forward, _K_FORWARD, _V_FORWARD)):
        _attach(getattr(attention, name), forward, _meepsquared_qkv=state, _meepsquared_fused=True)


def _marked(module, name) -> bool:
    return module.__dict__.get(name, False)


def apply(model) -> dict:
    """Attach the kernels to the model's LoRA layers; returns how many of each.

    Applying again replaces the forwards set by another patch (Liger's SwiGLU
    MLP, which the trainer applies) and leaves the kernels' own in place.
    """
    kernels = _kernels()
    if len(model.active_adapters) != 1:
        raise NotImplementedError("fused_lora works with one active adapter")
    text = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    in_mlp = set()
    counts = {"mlp": 0, "qkv": 0, "projections": 0}
    for module in model.modules():
        projections = [getattr(module, name, None) for name in MLP_PROJECTIONS]
        if not all(p is not None and _on_linear(p) for p in projections):
            continue
        # Most configs name the activation hidden_act; Gemma's name it hidden_activation
        activation = getattr(text, "hidden_act", None) or getattr(text, "hidden_activation", None)
        forward = _mlp_forward(kernels, activation)
        if forward is None:
            raise NotImplementedError(
                f"fused_lora fuses MLPs with the silu, gelu or gelu_pytorch_tanh activation; "
                f"{model.config.model_type} uses {activation!r}. Train it with fused_lora=False"
            )
        if "forward" not in module.__dict__ or _marked(module, "_meepsquared_mlp") is not forward:
            _attach(module, _disabled_mlp_forward(activation, forward), _meepsquared_mlp=forward)
        in_mlp.update(id(p) for p in projections)
        counts["mlp"] += 1
    for module in model.modules():
        if id(module) in in_mlp or not _on_linear(module):
            continue
        if not _marked(module, "_meepsquared_fused"):
            _attach(module, _projection_forward, _meepsquared_fused=True)
        counts["projections"] += 1
    for module in model.modules():
        projections = [getattr(module, name, None) for name in QKV_PROJECTIONS]
        if not all(p is not None and _on_linear(p) for p in projections):
            continue
        if not _marked(projections[0], "_meepsquared_qkv"):
            _fused_qkv(module)
        counts["qkv"] += 1
        counts["projections"] -= 3
    if not any(counts.values()):
        raise ValueError("fused_lora found no LoRA layer on a linear layer to fuse")
    return counts
