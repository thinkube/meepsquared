# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The expert stacks of a mixture of experts held as NF4: from_pretrained(load_in_4bit=True).

bitsandbytes' 4-bit loading replaces nn.Linear layers. Transformers stores the
routed experts of a mixture of experts as 3D parameters, one stack of all the
experts per projection (gate_up_proj, down_proj), which it leaves in 16 bits.
The experts are most of such a model: Gemma 4 26B-A4B's are 22.8 of its 26
billion parameters, 46 GB in BF16, about 13 GB in NF4.

So the stacks are not read with the rest of the checkpoint (without_stacks):
they stay on the meta device while Transformers loads and quantizes every
other weight. Then quantize_stacks reads them from the checkpoint one at a
time, to the device of their layer, and quantizes each with bitsandbytes
(NF4, blocks of 64, the double quantization of the linear layers or not):
the peak is one stack in 16 bits. A stack is kept as a non-persistent
buffer <name>_nf4 with its QuantState; the parameter is gone.

The experts then run through Transformers' experts interface as
"meepsquared_nf4": Transformers' grouped forward, with each product reading
its stack dequantized for that product and dropped again. The backward pass
needs only the input's gradient (the stacks are frozen) and dequantizes the
stack again, so no dequantized stack is kept between a product and its
gradient: the transient memory is one stack in the compute dtype.

LoRA on the stacks (target_parameters) needs them as parameters, in 16 bits
or FP8: with it, load the model with experts_4bit=False.

A checkpoint that stores each stack as one tensor, as Transformers holds it
(Gemma 4, Qwen3.5 and 3.6), is read this way. A checkpoint of one tensor
per expert is fused into stacks by Transformers while it loads, so its
stacks are loaded in 16 bits first and quantized after (quantize_loaded):
the load takes the memory of the 16-bit stacks, the training that of NF4.
"""

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch

NAME = "meepsquared_nf4"
STACKS = ("gate_up_proj", "up_proj", "down_proj")
PACKED = "_nf4"
BLOCK = 64
# Parts of a multimodal model whose layers are not the language model's
TOWERS = ("vision", "audio", "image")
_LAYER_TAIL = re.compile(r"layers\.\d+\..*$")


class Nf4ExpertsError(ValueError):
    """The experts cannot be held as NF4; the message says why."""


def is_stack(key: str) -> bool:
    """A checkpoint or parameter name of an expert stack: <...>.experts.<stack>."""
    head, _, last = key.rpartition(".")
    return last in STACKS and head.endswith("experts")


def _index_file(model_name: str, token: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    from transformers.utils import cached_file

    found = {name: cached_file(model_name, name, token=token, _raise_exceptions_for_missing_entries=False)
             for name in ("model.safetensors.index.json", "model.safetensors")}
    return found["model.safetensors.index.json"], found["model.safetensors"]


def checkpoint_stacks(model_name: str, token: Optional[str] = None) -> dict:
    """The expert stacks the checkpoint stores as one tensor each: {key: safetensors file}; empty when it has none."""
    from safetensors import safe_open

    index, single = _index_file(model_name, token)
    if index is not None:
        weight_map = json.loads(Path(index).read_text())["weight_map"]
        folder = Path(index).parent
        return {key: str(folder / file) for key, file in weight_map.items() if is_stack(key)}
    if single is not None:
        with safe_open(single, framework="pt") as f:
            return {key: single for key in f.keys() if is_stack(key)}
    # Not safetensors: Transformers reads the whole checkpoint, and quantize_loaded holds its stacks as NF4
    return {}


class _ShardWithoutStacks:
    """The handle Transformers opens on a shard, without the shard's expert stacks."""

    def __init__(self, handle):
        self.handle = handle
        tensors = getattr(handle, "tensors", None)
        if isinstance(tensors, dict):
            # A shard another loader read whole (meepsquared._unified_memory): its stacks are dropped now
            for key in [k for k in tensors if is_stack(k)]:
                del tensors[key]

    def keys(self):
        return [k for k in self.handle.keys() if not is_stack(k)]

    def get_slice(self, key):
        return self.handle.get_slice(key)

    def get_tensor(self, key):
        return self.handle.get_tensor(key)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self.handle.__exit__(*exc)


@contextlib.contextmanager
def without_stacks():
    """While inside, Transformers loads a model without reading its expert stacks, which stay on meta.

    It wraps the shard loader in place when entered, so a loader set before
    (meepsquared._unified_memory.shards_to_gpu) still reads the shards."""
    from transformers import modeling_utils

    inner_open = modeling_utils.safe_open
    real_move = modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device

    def safe_open(filename, *args, **kwargs):
        handle = inner_open(filename, *args, **kwargs)
        # Only the call that loads model shards passes a backend
        return _ShardWithoutStacks(handle) if "backend" in kwargs else handle

    def move(self, missing_keys, *args, **kwargs):
        return real_move(self, {k for k in missing_keys if not is_stack(k)}, *args, **kwargs)

    modeling_utils.safe_open = safe_open
    modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = move
    try:
        yield
    finally:
        modeling_utils.safe_open = inner_open
        modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = real_move


def _in_tower(name: str) -> bool:
    return any(tower in part for part in name.split(".") for tower in TOWERS)


def _checkpoint_key(name: str, keys: list) -> str:
    """The checkpoint key of a model's stack: the same name, or the one key that ends as it does from its layer."""
    if name in keys:
        return name
    tail = _LAYER_TAIL.search(name)
    found = [k for k in keys if tail is not None and k.endswith(tail.group(0)) and not _in_tower(k)]
    if len(found) != 1:
        raise Nf4ExpertsError(f"the expert stack {name} matches {len(found)} checkpoint keys "
                              f"({found[:3]}{'...' if len(found) > 3 else ''}), not one")
    return found[0]


def _layer_device(model, module_name: str) -> torch.device:
    """Where the experts' layer is: the device of the first tensor of the experts module, or of the module that
    holds it, that is not on meta."""
    while True:
        module = model.get_submodule(module_name)
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.device.type != "meta":
                return tensor.device
        if not module_name:
            raise Nf4ExpertsError("no weight of the model is on a device")
        module_name = module_name.rpartition(".")[0]


def _drop_pages(filename: str) -> None:
    fd = os.open(filename, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def quantize_stacks(model, files: dict, dtype: torch.dtype, double_quant: bool) -> dict:
    """Read every expert stack left on meta from its checkpoint file and hold it as NF4; returns what was done.

    files: {checkpoint key: safetensors file}, from checkpoint_stacks."""
    from safetensors import safe_open

    meta = [(name, p) for name, p in model.named_parameters() if is_stack(name) and p.device.type == "meta"]
    if not meta:
        raise Nf4ExpertsError("the model has no expert stack left to read: it is not a mixture of experts as "
                              "Transformers stores one, or its stacks were loaded already")
    keys = list(files)
    by_file: dict = {}
    for name, p in meta:
        key = _checkpoint_key(name, keys)
        by_file.setdefault(files[key], []).append((name, key, p))
    bytes_nf4 = 0
    params = 0
    for file, items in by_file.items():
        with safe_open(file, framework="pt") as f:
            for name, key, p in items:
                module_name, _, attr = name.rpartition(".")
                experts = model.get_submodule(module_name)
                weight = f.get_tensor(key).to(device=_layer_device(model, module_name), dtype=dtype)
                if tuple(weight.shape) != tuple(p.shape):
                    raise Nf4ExpertsError(f"{key} is {tuple(weight.shape)} in the checkpoint and {name} is "
                                          f"{tuple(p.shape)} in the model")
                params += weight.numel()
                bytes_nf4 += _hold(experts, attr, weight, double_quant)
                del weight
        _drop_pages(file)
    return {"stacks": len(meta), "parameters": params, "bytes": bytes_nf4}


def _hold(experts, attr: str, weight: torch.Tensor, double_quant: bool) -> int:
    """The stack held as NF4 in place of the parameter; returns the bytes it takes."""
    import bitsandbytes.functional as bnbf

    packed, state = bnbf.quantize_4bit(weight.contiguous(), blocksize=BLOCK, compress_statistics=double_quant,
                                       quant_type="nf4")
    del experts._parameters[attr]
    experts.register_buffer(attr + PACKED, packed, persistent=False)
    experts.__dict__.setdefault("_meepsquared_nf4", {})[attr] = state
    return packed.numel() * packed.element_size() + _state_bytes(state)


def quantize_loaded(model, dtype: torch.dtype, double_quant: bool) -> Optional[dict]:
    """Every expert stack the model holds as a 16-bit parameter held as NF4 instead, one at a time; None when the
    model has none (it is not a mixture of experts)."""
    found = [name for name, p in model.named_parameters()
             if is_stack(name) and p.dim() == 3 and p.is_floating_point() and p.device.type != "meta"]
    if not found:
        return None
    bytes_nf4 = params = 0
    for name in found:
        # Fetched one at a time, so that each 16-bit stack is freed once its NF4 one is held
        module_name, _, attr = name.rpartition(".")
        experts = model.get_submodule(module_name)
        weight = getattr(experts, attr).detach().to(dtype)
        params += weight.numel()
        bytes_nf4 += _hold(experts, attr, weight, double_quant)
        del weight
    return {"stacks": len(found), "parameters": params, "bytes": bytes_nf4}


def _state_bytes(state) -> int:
    total = state.absmax.numel() * state.absmax.element_size()
    if getattr(state, "state2", None) is not None:
        total += state.state2.absmax.numel() * state.state2.absmax.element_size()
    return total


def nf4_modules(model) -> list:
    """The experts modules whose stacks are held as NF4, with their names."""
    return [(name, m) for name, m in model.named_modules() if m.__dict__.get("_meepsquared_nf4")]


def dequantized(experts, attr: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """A stack of the module, in the dtype it was quantized from unless given; the result is not kept."""
    import bitsandbytes.functional as bnbf

    weight = bnbf.dequantize_4bit(getattr(experts, attr + PACKED), experts._meepsquared_nf4[attr])
    return weight if dtype is None else weight.to(dtype)


def dequantized_stacks(experts, dtype: torch.dtype):
    """(name, stack) of every NF4 stack of the module, dequantized: for a merged save."""
    for attr in experts.__dict__.get("_meepsquared_nf4", {}):
        yield attr, dequantized(experts, attr, dtype)


class _Nf4GroupedProduct(torch.autograd.Function):
    """The grouped product of x with a stack held as NF4, as Transformers' _grouped_linear computes it.

    The stack is dequantized, in the dtype it was quantized from, for the
    forward product and again for the backward one, which gives x's gradient
    only: the stack is frozen, and keeping it dequantized would cost what NF4
    saves. As in Transformers, the product computes in the stack's dtype.
    """

    @staticmethod
    def forward(ctx, x, experts, attr, offsets, bias, is_transposed):
        from transformers.integrations.moe import _grouped_linear

        ctx.experts, ctx.attr, ctx.offsets, ctx.is_transposed = experts, attr, offsets, is_transposed
        ctx.x_dtype = x.dtype
        return _grouped_linear(x, dequantized(experts, attr), offsets, bias=bias, is_transposed=is_transposed)

    @staticmethod
    def backward(ctx, grad_out):
        from transformers.integrations.moe import _grouped_linear

        weight = dequantized(ctx.experts, ctx.attr)
        # x @ W^T has the gradient grad @ W, which is the transposed form of the same product, and the other way round
        grad_x = _grouped_linear(grad_out.contiguous(), weight, ctx.offsets, is_transposed=not ctx.is_transposed)
        return grad_x.to(ctx.x_dtype), None, None, None, None, None


@torch.compiler.disable  # Dynamo cannot trace the product's autograd function nor bitsandbytes' kernels
def experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                    top_k_weights: torch.Tensor) -> torch.Tensor:
    """Transformers' grouped experts forward (transformers.integrations.moe.grouped_mm_experts_forward), its two
    products on the NF4 stacks."""
    if self._is_expert_parallel:
        raise Nf4ExpertsError("NF4 expert stacks are not sharded across processes (expert parallelism)")
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    selected = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]
    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    up = "gate_up_proj" if self.has_gate else "up_proj"
    up_bias = getattr(self, f"{up}_bias")[expert_ids_g] if self.has_bias else None
    out = _Nf4GroupedProduct.apply(selected, self, up, offsets, up_bias, self.is_transposed)
    out = self._apply_gate(out) if self.has_gate else self.act_fn(out)
    down_bias = self.down_proj_bias[expert_ids_g] if self.has_bias else None
    out = _Nf4GroupedProduct.apply(out, self, "down_proj", offsets, down_bias, self.is_transposed)
    if self.has_post_expert_norm:
        out = self.post_expert_norm(out)

    weighted = out * sample_weights_g.unsqueeze(-1)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted = weighted[inv_perm]
    return weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def register() -> None:
    """experts_forward in Transformers' experts interface, under NAME."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    if NAME not in ALL_EXPERTS_FUNCTIONS:
        ALL_EXPERTS_FUNCTIONS.register(NAME, experts_forward)


def use(model) -> int:
    """The model's NF4 experts run as NAME; returns how many experts modules. Every one must take the interface."""
    modules = nf4_modules(model)
    for name, experts in modules:
        missing = [a for a in ("has_gate", "has_bias", "is_transposed", "num_experts", "_is_expert_parallel")
                   if not hasattr(experts, a)]
        if missing:
            raise Nf4ExpertsError(f"{name} ({type(experts).__name__}) does not run through Transformers' experts "
                                  f"interface (it has no {', '.join(missing)}); its NF4 stacks cannot be used")
    register()
    model.set_experts_implementation(NAME)
    wrong = [name for name, experts in modules if experts.config._experts_implementation != NAME]
    if wrong:
        raise Nf4ExpertsError(f"the experts implementation of {wrong[0]} stayed "
                              f"{model.get_submodule(wrong[0]).config._experts_implementation!r}, not {NAME!r}")
    return len(modules)


def load(model, model_name: str, files: dict, dtype: torch.dtype, double_quant: bool) -> Optional[dict]:
    """After a load (under without_stacks when files names the checkpoint's stacks): the stacks held as NF4 and the
    experts running on them; None for a model without expert stacks."""
    done = (quantize_stacks(model, files, dtype, double_quant) if files
            else quantize_loaded(model, dtype, double_quant))
    if done is None:
        return None
    done["modules"] = use(model)
    left = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    if left:
        raise Nf4ExpertsError(f"{model_name}: {len(left)} weights were not read, among them {left[0]}")
    return done
