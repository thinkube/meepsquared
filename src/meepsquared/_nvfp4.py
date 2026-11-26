# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The frozen weights and their inputs in NVFP4, on Blackwell's FP4 tensor cores: from_pretrained(load_in_nvfp4=True).

A LoRA training reads the base model's linear layers and never updates them.
Each one is replaced, once the adapters are added, by a Transformer Engine
layer whose weight is held as NVFP4 (FP4 e2m1 values, one FP8 scale per block
of 16 and one float32 scale per tensor) and whose input is quantized to NVFP4
as it comes, both under Transformer Engine's NVFP4 recipe: the forward
product runs on the FP4 tensor cores. The weight gets no gradient. The
adapters stay in BF16, as does everything that is not a frozen linear layer
of the decoder: embeddings, norms, routers, the output layer and a mixture of
experts' expert stacks.

The input's gradient is computed in the layer's dtype (BF16), as the
output's gradient times the weight dequantized for that product: the
gradient of the product the forward computed, its input's quantization read
as the identity. Transformer Engine's own backward quantizes the output's
gradient to NVFP4 first, and its recipe rounds that gradient stochastically
so that the rounding has no bias. The Transformer Engine of NVIDIA's PyTorch
26.03 container has no stochastic-rounding FP4 kernel for compute capability
12.1 (it faults with "FP4 cvt PTX instructions are architecture-specific"),
and a gradient rounded to the nearest FP4 value in every frozen layer it
crosses carries a bias that adds up over the layers: the adapters and the
head then train on a wrong gradient.

Transformer Engine holds the weight twice, as it is and transposed: 1.125
bytes a weight, 56% of BF16's memory, where FP8 holds one byte. The recipe is
Transformer Engine's NVFP4BlockScaling with stochastic rounding off, which
concerns gradients only and no gradient is quantized here; the weights' 2D
blocks are the recipe's own.

Transformer Engine takes a product whose rows (every dimension of the input
but the last) are a multiple of 8: the rows are padded with zeros to a
multiple of ROW_MULTIPLE and cut back after. Zero rows change no scale of the
other rows, so the result is the same.
"""

import torch
from torch import nn

# Transformer Engine's NVFP4 products take a row count that is a multiple of 8; 16 is FP4's block
ROW_MULTIPLE = 16
MIN_CAPABILITY = (10, 0)


def available(device=None) -> tuple[bool, str]:
    """Whether NVFP4 products run here, and why not when they do not."""
    if not torch.cuda.is_available():
        return False, "NVFP4 products run on a CUDA GPU"
    try:
        from transformer_engine.pytorch.quantization import check_nvfp4_support
    except ImportError:
        return False, ("NVFP4 needs NVIDIA Transformer Engine 2.13 or later (transformer_engine.pytorch), as NVIDIA's "
                       "PyTorch containers carry it")
    capability = torch.cuda.get_device_capability(device)
    if capability < MIN_CAPABILITY:
        return False, (f"FP4 tensor cores need compute capability {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]} or later "
                       f"(Blackwell); this GPU is {capability[0]}.{capability[1]}")
    ok, why = check_nvfp4_support()
    return (True, "") if ok else (False, why)


def recipe():
    from transformer_engine.common.recipe import NVFP4BlockScaling

    return NVFP4BlockScaling(disable_stochastic_rounding=True)


class _Nvfp4Product(torch.autograd.Function):
    """y = rows @ W^T with W and the rows in NVFP4, by the Transformer Engine layer; the backward gives the rows'
    gradient only, W being frozen: the output's gradient times the dequantized W, in the gradient's dtype."""

    @staticmethod
    def forward(ctx, rows, layer):
        import transformer_engine.pytorch as te

        ctx.layer = layer
        with te.autocast(enabled=True, recipe=layer.recipe):
            return layer.inner(rows)

    @staticmethod
    def backward(ctx, grad_out):
        weight = ctx.layer.inner.weight.dequantize().to(grad_out.dtype)
        return grad_out @ weight, None


class Nvfp4Linear(nn.Module):
    """A frozen linear layer held as NVFP4 by a Transformer Engine layer; its bias, if any, stays as it was."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        import transformer_engine.pytorch as te

        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.dtype = linear.weight.dtype
        self.recipe = recipe()
        with te.quantized_model_init(enabled=True, recipe=self.recipe):
            self.inner = te.Linear(self.in_features, self.out_features, bias=False, params_dtype=self.dtype,
                                   device=linear.weight.device)
        with torch.no_grad():
            self.inner.weight.copy_(linear.weight.detach())
        self.inner.weight.requires_grad_(False)
        self.bias = linear.bias

    @torch.compiler.disable  # Dynamo traces neither Transformer Engine's layers nor the product's autograd function
    def forward(self, x):
        shape = x.shape
        rows = x.reshape(-1, shape[-1]).to(self.dtype)
        count = rows.shape[0]
        padding = -count % ROW_MULTIPLE
        if padding:
            rows = nn.functional.pad(rows, (0, 0, 0, padding))
        out = _Nvfp4Product.apply(rows, self)
        out = out[:count].reshape(*shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def dequantized(self) -> torch.Tensor:
        """The weight back in its dtype, for a merged save."""
        return self.inner.weight.dequantize().to(self.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, NVFP4 (Transformer Engine)"


def apply(model) -> int:
    """Every frozen linear layer of the decoder layers held as NVFP4; returns how many. Applying again changes
    nothing."""
    from meepsquared import _checkpointing, _fp8

    ok, why = available(next(model.parameters()).device)
    if not ok:
        raise RuntimeError(f"load_in_nvfp4=True: {why}")
    trainable = {name.rsplit(".", 1)[0] for name, p in model.named_parameters() if p.requires_grad}
    converted = 0
    for layer in _checkpointing.layers(model):
        for name, module in list(layer.named_modules()):
            # The same layers FP8 holds: frozen, not an adapter, not a router nor the output layer, sizes of 16
            if not _fp8.convertible(module, name, trainable):
                continue
            parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
            setattr(parent, name.rsplit(".", 1)[-1], Nvfp4Linear(module))
            converted += 1
    if not converted:
        raise ValueError("load_in_nvfp4=True found no frozen linear layer to hold as NVFP4")
    return converted
