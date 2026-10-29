# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A model whose layers are spread over several GPUs (device_map="balanced", "auto").

Triton kernels (Liger's, TRL's, flash-linear-attention's) launch on the
process's current GPU. Between consumer GPUs without peer-to-peer access, a
kernel launched on one GPU cannot read another's memory, so every kernel must
launch on the GPU that holds its tensors. Each module of the model's device
map runs its forward with its own GPU as the current device, and the model's
outer forward, where the trainer computes the loss on the output layer's
tensors, with the output layer's GPU; the trainer's count of the batch's tokens
goes to that GPU, and the loss comes back to the trainer's. The backward pass needs nothing: PyTorch runs each
GPU's part of it on that GPU.
"""

import functools

import torch


def gpus(model) -> list:
    device_map = getattr(model, "hf_device_map", None) or {}
    return sorted({torch.device(d) if not isinstance(d, int) else torch.device("cuda", d)
                   for d in device_map.values() if str(d) not in ("cpu", "disk", "meta")},
                  key=lambda d: d.index or 0)


def _on_device(module, device):
    forward = module.forward
    if getattr(forward, "_meepsquared_device", None) is not None:
        return

    @functools.wraps(forward)
    def forward_on_device(*args, **kwargs):
        with torch.cuda.device(device):
            return forward(*args, **kwargs)

    forward_on_device._meepsquared_device = device
    module.forward = forward_on_device


def apply(model) -> bool:
    """For a model spread over several GPUs, each part runs on its own GPU; True when it is spread."""
    device_map = getattr(model, "hf_device_map", None) or {}
    if len(gpus(model)) < 2:
        return False
    modules = dict(model.named_modules())
    output_device = None
    for name, device in device_map.items():
        if str(device) in ("cpu", "disk", "meta"):
            continue
        device = device if isinstance(device, torch.device) else torch.device("cuda", device) \
            if isinstance(device, int) else torch.device(device)
        _on_device(modules[name], device)
        if name.endswith("lm_head"):
            output_device = device
    model._meepsquared_output_device = output_device or gpus(model)[-1]
    _on_device(model, model._meepsquared_output_device)
    return True


def count_on_output_device(trainer, output_device) -> None:
    """The loss computed on the output layer's GPU, and handed back on the trainer's.

    The trainer counts the batch's tokens on its own device, the first GPU, and
    expects the loss there; TRL's fused loss divides by the count on the GPU it
    computes the loss on. The count goes to that GPU, the loss comes back.
    """
    compute_loss = trainer.compute_loss
    trainer_device = trainer.args.device

    @functools.wraps(compute_loss)
    def compute_loss_on_output_device(model, inputs, return_outputs=False, num_items_in_batch=None):
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(output_device)
        result = compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if return_outputs:
            loss, outputs = result
            return loss.to(trainer_device), outputs
        return result.to(trainer_device)

    trainer.compute_loss = compute_loss_on_output_device
