# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Saving a trained LoRA model: the adapter alone, or merged into 16-bit weights.

A merged save writes the base model with W + scaling * B @ A in place of each
adapted weight, computed in float32 and stored in the model's dtype. A 4-bit
base (QLoRA) or an FP8 one is dequantized first, every such layer of it, so
the result is an ordinary 16-bit model. An expert stack of a mixture of
experts is written the same way, every expert with its adapter merged in and
an FP8 stack dequantized with its per-expert scale. The model is written
tensor by tensor into safetensors shards of at most MAX_SHARD_BYTES, so the
save needs the memory of one shard, not of the whole model. The model in
memory is not changed: training or generation can go on after the save.

An adapter save takes the adapter's parameters from the model's parameters,
never from its state dict: the state dict reads every frozen weight too, and
on a mixture of experts with LoRA on its FP8 expert stacks that builds full
copies of the stacks, more memory than the training itself took.

In a distributed run every process calls save(); the first one writes. When
the model is sharded across the processes (FSDP2), the adapter's parameters
are gathered from all of them first, and a merged save is refused: the base
weights are not in one place.
"""

import copy
import json
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import torch

SAVE_METHODS = ("lora", "merged_16bit")
MAX_SHARD_BYTES = 5 * 2**30  # Transformers' shard size
_INDEX = "model.safetensors.index.json"


def _dequantized(layer) -> Optional[torch.Tensor]:
    """The dense weight of a bitsandbytes 4-bit or an FP8 linear layer, or None for another layer."""
    from meepsquared._fp8 import Fp8Linear

    if isinstance(layer, Fp8Linear):
        return layer.dequantized()
    try:
        import bitsandbytes as bnb
    except ImportError:
        return None
    if not isinstance(layer, bnb.nn.Linear4bit):
        return None
    return bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)


def _the_adapter(module) -> str:
    if len(module.active_adapters) != 1:
        raise NotImplementedError("a merged save takes one active adapter")
    adapter = module.active_adapters[0]
    if module.use_dora.get(adapter, False):
        raise NotImplementedError("a merged save of a DoRA adapter is not supported")
    return adapter


def _merged_weight(module, dtype) -> torch.Tensor:
    adapter = _the_adapter(module)
    base = module.get_base_layer()
    weight = _dequantized(base)
    if weight is None:
        if not isinstance(base, torch.nn.Linear):
            raise NotImplementedError(f"a merged save of LoRA on {type(base).__name__} is not supported")
        weight = base.weight
    a, b = module.lora_A[adapter].weight, module.lora_B[adapter].weight
    delta = (b.float() @ a.float()) * module.scaling[adapter]
    if getattr(module, "fan_in_fan_out", False):
        delta = delta.t()
    return (weight.float() + delta).to(dtype)


def _merged_stack(experts, name: str, wrapper, dtype) -> torch.Tensor:
    """An expert stack with its adapter merged: W + scaling * lhs @ rhs for every expert, in the stack's layout.

    The factors are the ones the training computes with (meepsquared._moe),
    in float32; an FP8 stack is dequantized with its per-expert scale first.
    """
    from meepsquared import _fp8, _moe

    stack = getattr(experts, name)
    if stack.dtype == torch.float8_e4m3fn:
        weight = _fp8.dequantize_stack(stack.detach(), getattr(experts, name + _fp8.STACK_SCALE), torch.float32)
    else:
        weight = stack.detach().float()
    if wrapper is not None:
        lhs, rhs, scaling = _moe._delta_factors(wrapper, _the_adapter(wrapper), torch.float32)
        weight = weight + scaling * torch.bmm(lhs, rhs)
    return weight.to(dtype)


def _merged_experts(wrapper, prefix: str, dtype) -> Iterator[tuple[str, torch.Tensor]]:
    """The tensors of an experts module wrapped by PEFT: each stack merged, the rest as they are, no FP8 scale."""
    from meepsquared import _fp8, _moe

    chain, experts = _moe._wrapper_chain(wrapper)
    wrappers = {w.parameter_name: w for w in chain}
    for pname, value in list(experts.named_parameters(recurse=False)) + list(experts.named_buffers(recurse=False)):
        if pname in getattr(experts, "_non_persistent_buffers_set", set()):
            continue
        if pname.endswith(_fp8.STACK_SCALE) and pname[: -len(_fp8.STACK_SCALE)] in dict(experts.named_parameters(recurse=False)):
            continue  # the scale of an FP8 stack, which is written dequantized
        if pname in wrappers or value.dtype == torch.float8_e4m3fn:
            yield f"{prefix}{pname}", _merged_stack(experts, pname, wrappers.get(pname), dtype)
        else:
            yield f"{prefix}{pname}", value.detach().to(dtype) if value.is_floating_point() else value.detach()


def merged_tensors(peft_model) -> Iterator[tuple[str, torch.Tensor]]:
    """The base model's weights with the adapters merged and every 4-bit or FP8 layer dequantized, one at a time, on the CPU.

    They are stored in the dtype of the input embeddings, which are never
    quantized: the dtype the model was loaded in, or computes in when its other
    layers are 4-bit. A tensor shared by two names (tied embeddings) is
    written once, under the first name.
    """
    from peft.tuners.lora import LoraLayer
    from peft.tuners.lora.layer import ParamWrapper

    base = peft_model.get_base_model()
    dtype = base.get_input_embeddings().weight.dtype
    done = []  # modules whose parameters are written as a whole: a LoRA layer, a 4-bit layer, wrapped experts
    seen = set()
    for name, module in base.named_modules():
        if any(name.startswith(prefix + ".") for prefix in done):
            continue
        prefix = f"{name}." if name else ""
        if isinstance(module, ParamWrapper):
            for key, value in _merged_experts(module, prefix, dtype):
                yield key, value.cpu()
            done.append(name)
            continue
        if isinstance(module, LoraLayer):
            yield f"{prefix}weight", _merged_weight(module, dtype).cpu()
            bias = module.get_base_layer().bias
            if bias is not None:
                yield f"{prefix}bias", bias.detach().to(dtype).cpu()
            done.append(name)
            continue
        dense = _dequantized(module)
        if dense is not None:
            yield f"{prefix}weight", dense.to(dtype).cpu()
            if module.bias is not None:
                yield f"{prefix}bias", module.bias.detach().to(dtype).cpu()
            done.append(name)
            continue
        for pname, value in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
            if pname in getattr(module, "_non_persistent_buffers_set", set()):
                continue
            storage = (value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), value.dtype)
            if storage in seen:
                continue
            seen.add(storage)
            yield f"{prefix}{pname}", (value.detach().to(dtype) if value.is_floating_point() else value.detach()).cpu()


def write_shards(tensors, directory: Path, max_shard_bytes: int = MAX_SHARD_BYTES) -> None:
    """Write the tensors as Transformers reads them: model.safetensors, or numbered shards and an index.

    Each tensor is taken as it comes, so the memory held is at most one shard.
    A model written earlier into the same folder is replaced, shards and index.
    """
    from safetensors.torch import save_file

    for old in list(directory.glob("model*.safetensors")) + [directory / _INDEX]:
        if old.exists():
            old.unlink()
    shards, current, size, total = [], {}, 0, 0

    def flush():
        nonlocal current, size
        if current:
            path = directory / f"shard-{len(shards):05d}.safetensors"  # renamed once the count is known
            save_file(current, str(path), metadata={"format": "pt"})
            shards.append((path, list(current)))
            current, size = {}, 0

    for name, tensor in tensors:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current and size + tensor_bytes > max_shard_bytes:
            flush()
        current[name] = tensor.contiguous()
        size, total = size + tensor_bytes, total + tensor_bytes
    flush()
    if len(shards) == 1:
        shards[0][0].rename(directory / "model.safetensors")
        return
    weight_map = {}
    for i, (path, names) in enumerate(shards, 1):
        final = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        path.rename(directory / final)
        weight_map.update({name: final for name in names})
    (directory / _INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map},
                                               indent=2, sort_keys=True) + "\n")


def _first_process() -> bool:
    """Whether this is the only process, or the first one of a distributed run."""
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _sharded(peft_model) -> bool:
    """Whether the model's parameters are sharded across processes, as FSDP2 holds them."""
    from torch.distributed.tensor import DTensor

    return any(isinstance(p, DTensor) for p in peft_model.parameters())


def adapter_state_dict(peft_model) -> dict:
    """The trainable parameters, the adapter, as whole tensors on the CPU, from the parameters themselves.

    A sharded parameter is gathered from every process, so every process of
    the run must call this; the result is the same in each.
    """
    from torch.distributed.tensor import DTensor

    return {name: (p.full_tensor() if isinstance(p, DTensor) else p).detach().cpu()
            for name, p in peft_model.named_parameters() if p.requires_grad}


def save(peft_model, save_directory, tokenizer=None, save_method: str = "merged_16bit", **kwargs) -> None:
    """Save the model; in a distributed run every process calls this, and the first one writes."""
    if save_method not in SAVE_METHODS:
        raise ValueError(f"save_method must be one of {SAVE_METHODS}, not {save_method!r}; "
                         "a 4-bit merged save would lose precision and is not offered")
    sharded = _sharded(peft_model)
    if sharded and save_method != "lora":
        raise ValueError("a merged save needs the whole model in one process, and this one is sharded across processes "
                         "(FSDP2): save the adapter with save_method='lora', then load the base model and the adapter "
                         "on one GPU and save merged from there")
    state = adapter_state_dict(peft_model) if save_method == "lora" else None
    if not _first_process():
        return
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if save_method == "lora":
        peft_model.save_pretrained(directory, state_dict=state, **kwargs)
    else:
        if kwargs:
            raise TypeError(f"a merged save takes no further arguments; got {sorted(kwargs)}")
        base = peft_model.get_base_model()
        config = copy.deepcopy(base.config)
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        config.dtype = base.get_input_embeddings().weight.dtype
        config.save_pretrained(directory)
        if getattr(base, "generation_config", None) is not None:
            base.generation_config.save_pretrained(directory)
        write_shards(merged_tensors(peft_model), directory, MAX_SHARD_BYTES)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)


def push(peft_model, repo_id: str, tokenizer=None, save_method: str = "merged_16bit", token: Optional[str] = None,
         private: Optional[bool] = None, commit_message: str = "Upload model") -> str:
    """Save as save() does, into a new or existing Hugging Face Hub repository.

    Returns the repository's URL; in a distributed run, every process calls
    this, the first one uploads and returns the URL, the others return "".
    """
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as directory:
        save(peft_model, directory, tokenizer=tokenizer, save_method=save_method)
        if not _first_process():
            return ""
        api = HfApi(token=token)
        url = api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=directory, commit_message=commit_message)
    return str(url)
