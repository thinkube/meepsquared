# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Settings for training on several GPUs.

    SFTConfig(..., **meepsquared.fsdp_config(model))

With FSDP2 each GPU holds a share of every decoder layer's weights, and the
GPUs send each other the rest of a layer's weights as it runs: a model about
twice the size of one GPU trains on two. Load the model on the CPU
(device_map="cpu", or "balanced" for one process with the layers spread), so
that each GPU only receives its share. Between consumer GPUs over PCIe the
weights sent at every step cost time; more examples per step spread that cost.
For a model that fits on one GPU, DDP (torchrun, no settings) is faster.
"""


def fsdp_config(model, reshard_after_forward: bool = True, cpu_ram_efficient_loading: bool = False) -> dict:
    """SFTConfig's fsdp and fsdp_config for FSDP2, sharded by the model's decoder layer.

    cpu_ram_efficient_loading: the first process reads the checkpoint and the
    others receive their share from it; set ACCELERATE_USE_FSDP=true and
    FSDP_CPU_RAM_EFFICIENT_LOADING=1 in the environment before loading too.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # _no_split_modules names the layer classes of the whole checkpoint; a model loaded for text
    # from a multimodal checkpoint has no vision blocks, and Accelerate refuses a class it cannot find
    present = {type(module).__name__ for module in base.modules()}
    layers = sorted(set(getattr(base, "_no_split_modules", None) or []) & present)
    if not layers:
        raise ValueError(f"{type(base).__name__} names no decoder layer class (_no_split_modules) to shard by")
    return {"fsdp": "full_shard auto_wrap",
            "fsdp_config": {"fsdp_version": 2, "transformer_layer_cls_to_wrap": layers,
                            "reshard_after_forward": reshard_after_forward,
                            "cpu_ram_efficient_loading": cpu_ram_efficient_loading}}
