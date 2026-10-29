# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA on the experts of a mixture of experts, computed on the routed tokens only.

PEFT puts LoRA on an expert stack (target_parameters) by registering, at each
forward, a parametrization that turns the stack's weight into W + scale * lhs @ rhs
for all experts. On Qwen3.6-35B-A3B that builds and copies about 1 GB of weights
per layer at every forward and backward, though a token reaches 8 of the 256
experts: half the GPU time of a training step.

Here the outermost of PEFT's wrappers around the experts runs the experts
itself, as Transformers' grouped_mm implementation does, on the base weights
unchanged, and adds each adapter's change to the tokens routed to each expert:
x @ W^T + scale * (x @ rhs^T) @ lhs^T, two grouped matmuls whose inner size is
the LoRA rank. Gradients reach the adapters through those products. The
forward is the same function of the inputs and the weights; only the
intermediate full weights are not built.
"""

import os
import re
import types

import torch

from meepsquared import _grouped, _selective

_PATCHED = "_meepsquared_routed_lora"
# The attributes of a Transformers experts module (transformers.integrations.moe)
EXPERTS_ATTRIBUTES = ("num_experts", "has_gate", "has_bias", "is_transposed")
# The stacks of weights an experts module holds, by name: a gated MLP's gate and up projections
# in one stack, or an up projection alone, and the down projection
STACKS = ("gate_up_proj", "up_proj", "down_proj")


def _wrapper_chain(outer):
    """PEFT's ParamWrappers around an experts module, outermost first, and the module."""
    from peft.tuners.lora.layer import ParamWrapper

    chain, module = [], outer
    while isinstance(module, ParamWrapper):
        chain.append(module)
        module = module.base_layer
    return chain, module


def _grouped_lora(x, lhs, rhs, scaling, offsets, is_transposed):
    """scale * x @ delta per group, delta = lhs @ rhs in the layout of the experts' weight.

    The scale goes into the small factor: a multiply of the full-size result
    would cost as much as the product itself.
    """
    from transformers.integrations.moe import _grouped_mm

    # The grouped matmul needs every row stride to be a multiple of 16 bytes; the
    # rank dimension is padded with zeros to that, which leaves lhs @ rhs unchanged
    pad = -lhs.shape[-1] % (16 // lhs.element_size())
    if pad:
        lhs = torch.nn.functional.pad(lhs, (0, pad))
        rhs = torch.nn.functional.pad(rhs, (0, 0, 0, pad))
    if is_transposed:  # weight (experts, in, out): x @ delta
        low = _grouped_mm(x, (lhs * scaling).contiguous(), offs=offsets)
        return _grouped_mm(low, rhs.contiguous(), offs=offsets)
    # weight (experts, out, in): x @ delta^T
    low = _grouped_mm(x, (rhs * scaling).transpose(-2, -1).contiguous(), offs=offsets)
    return _grouped_mm(low, lhs.transpose(-2, -1).contiguous(), offs=offsets)


def _routed_forward(experts, hidden_states, top_k_index, top_k_weights, factors):
    """Transformers' grouped_mm experts forward (5.17), with the adapters' change added per projection."""
    from transformers.integrations.moe import _grouped_linear

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.size(0), hidden_states.size(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    selected = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]
    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=experts.num_experts, min=0, max=experts.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    sentinel = (expert_ids_g >= experts.num_experts).unsqueeze(-1)
    expert_ids_g.clamp_(max=experts.num_experts - 1)
    selected = selected.masked_fill(sentinel, 0.0)

    kernel = _grouped.wanted(device)

    def project(x, name, bias_name):
        weight = getattr(experts, name)
        fp8 = weight.dtype == torch.float8_e4m3fn  # a stack held as FP8 (meepsquared._fp8), with one scale per expert
        scale = getattr(experts, name + "_scale_inv") if fp8 else None
        lhs, rhs, scaling = factors.get(name, (None, None, 1.0))
        if kernel:
            # one launch for the base product, the adapters and the bias (meepsquared._grouped)
            bias = getattr(experts, bias_name) if experts.has_bias else None
            out = _grouped.lora_linear(x.to(hidden_states.dtype), weight, scale, offsets, experts.is_transposed,
                                       lhs, rhs, scaling, bias)
            return out.masked_fill_(sentinel, 0.0)
        bias = getattr(experts, bias_name)[expert_ids_g] if experts.has_bias else None
        if fp8:
            from meepsquared import _fp8

            out = _fp8.grouped_product(x, weight, scale, offsets, bias, experts.is_transposed)
        else:
            out = _grouped_linear(x, weight, offsets, bias=bias, is_transposed=experts.is_transposed)
        # In place, except in a selectively recomputed layer (meepsquared._selective): there
        # the base product's output is kept for the backward pass, and a write over it
        # would change what the recompute reads
        inplace = _selective.in_place()
        if lhs is not None:
            delta = _grouped_lora(x.to(lhs.dtype), lhs, rhs, scaling, offsets, experts.is_transposed)
            out = out.add_(delta) if inplace else out + delta
        return out.masked_fill_(sentinel, 0.0) if inplace else out.masked_fill(sentinel, 0.0)

    if experts.has_gate:
        out = experts._apply_gate(project(selected, "gate_up_proj", "gate_up_proj_bias"))
    else:
        out = experts.act_fn(project(selected, "up_proj", "up_proj_bias"))
    out = project(out, "down_proj", "down_proj_bias")
    weighted = out * sample_weights_g.unsqueeze(-1)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.size(0), device=device)
    return weighted[inverse].view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def _forward(self, hidden_states, top_k_index, top_k_weights, *args, **kwargs):
    chain, experts = _wrapper_chain(self)
    adapters = self.active_adapters
    fp8 = getattr(experts, "_meepsquared_fp8", False)
    plain = (not args and not kwargs and not any(w.merged for w in chain) and len(adapters) == 1
             and all(adapters[0] in w.lora_A for w in chain))
    if self.disable_adapters and fp8 and not args and not kwargs:
        return _routed_forward(experts, hidden_states, top_k_index, top_k_weights, {})
    if self.disable_adapters or not plain:
        if fp8:
            # PEFT's forward builds W + delta from the stack, which is FP8 now
            raise NotImplementedError("expert stacks held as FP8 (load_in_fp8) compute with one active, unmerged "
                                      "adapter and no extra arguments")
        return self._meepsquared_peft_forward(hidden_states, top_k_index, top_k_weights, *args, **kwargs)
    factors = {w.parameter_name: _delta_factors(w, adapters[0], hidden_states.dtype) for w in chain}
    return _routed_forward(experts, hidden_states, top_k_index, top_k_weights, factors)


setattr(_forward, _PATCHED, True)


def _delta_factors(wrapper, adapter: str, dtype: torch.dtype):
    """(lhs, rhs, scaling) with delta = scaling * lhs @ rhs in the layout of the stack, as PEFT's get_delta_factors.

    PEFT casts the factors to the stack's dtype; a stack held as FP8 would take
    the adapters with it, so they are cast to the compute dtype here.
    """
    a, b = wrapper.lora_A[adapter].weight, wrapper.lora_B[adapter].weight
    a = a.reshape(wrapper.num_experts, -1, a.shape[-1])  # (experts, rank, in)
    b = b.reshape(b.shape[0], -1, wrapper.num_experts).permute(2, 0, 1)  # (experts, out, rank)
    if not wrapper._did_swap_in_out_features:  # the stack is (experts, in, out)
        lhs, rhs = a.transpose(-2, -1), b.transpose(-2, -1)
    else:
        lhs, rhs = b, a
    return lhs.to(dtype), rhs.to(dtype), wrapper.scaling[adapter]


def expert_targets(model) -> tuple[list, int]:
    """The target_parameters that name every expert stack of the model, and how many stacks they reach.

    A name is the stack's path inside its decoder layer ("mlp.experts.gate_up_proj"),
    which PEFT matches in every layer. A model with no Transformers experts
    module gives ([], 0).
    """
    names, stacks = set(), 0
    for name, module in model.named_modules():
        if not all(hasattr(module, attribute) for attribute in EXPERTS_ATTRIBUTES):
            continue
        inside_layer = re.search(r"(?:^|\.)layers\.\d+\.(.+)$", name)
        if inside_layer is None:
            raise NotImplementedError(f"{name}: an experts module outside a numbered decoder layer")
        for stack in STACKS:
            if isinstance(getattr(module, stack, None), torch.nn.Parameter):
                names.add(f"{inside_layer.group(1)}.{stack}")
                stacks += 1
    return sorted(names), stacks


def kernel_text(device) -> str:
    """Which code computes the experts' products on this device, and why, for a plan."""
    forced = os.environ.get("MEEPSQUARED_GROUPED_KERNEL")
    if forced in ("0", "1"):
        return f"meepsquared's grouped kernel {'on' if forced == '1' else 'off'}, as MEEPSQUARED_GROUPED_KERNEL={forced} sets"
    if device.type != "cuda":
        return "Transformers' grouped product, as the model is not on a CUDA GPU"
    major, minor = torch.cuda.get_device_capability(device)
    if _grouped.wanted(device):
        return (f"meepsquared's grouped kernel, one launch per product: PyTorch's grouped product loops over the "
                f"experts on compute capability {major}.{minor}")
    if major in _grouped.TORCH_HAS_GROUPED_MM:
        return f"PyTorch's grouped product, which has kernels of its own on compute capability {major}.{minor}"
    return "PyTorch's grouped product, a loop over the experts: meepsquared's kernel needs Triton"


def routed(module) -> bool:
    """Whether the module is an experts wrapper whose forward meepsquared computes."""
    return getattr(module.forward, _PATCHED, False) if hasattr(module, "forward") else False


def apply(model) -> int:
    """The routed LoRA on every expert stack PEFT has wrapped; returns how many."""
    from peft.tuners.lora.layer import ParamWrapper

    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, ParamWrapper) or name.endswith(".base_layer"):
            continue  # an inner wrapper: its outermost one runs the chain
        chain, experts = _wrapper_chain(module)
        if not all(hasattr(experts, a) for a in ("num_experts", "has_gate", "has_bias", "is_transposed")):
            raise NotImplementedError(
                f"{name}: LoRA on parameters of {type(experts).__name__}, which is not a Transformers experts module")
        for w in chain:
            for dropout in w.lora_dropout.values():
                if not isinstance(dropout, torch.nn.Identity):
                    raise NotImplementedError(f"{name}: LoRA dropout on expert parameters is not supported")
        if not getattr(module.forward, _PATCHED, False):
            module._meepsquared_peft_forward = module.forward
            module.forward = types.MethodType(_forward, module)
        patched += 1
    return patched
