# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The frozen weights in FP8, their products on the FP8 tensor cores: from_pretrained(load_in_fp8=True).

A LoRA training changes the adapters only; the base model's linear layers are
read, never updated. Stored as FP8 (e4m3) with one scale per tensor, they take
half the memory of BF16, and each product with them runs on the GPU's FP8
tensor cores, about 1.6 times faster than BF16 on an NVIDIA GB10, its input
quantized on the way in by one fused kernel. The backward pass needs only the
input's gradient, another FP8 product with the same weight, so no activation
is kept for the base product. The adapters stay in BF16, as does everything
that is not a frozen linear layer of the decoder: embeddings, norms, the
output layer. The fused LoRA kernels (meepsquared._fused) run their base products
through the same FP8 product, so fused_lora and load_in_fp8 combine.

One scale per tensor is NVIDIA's own recipe (Transformer Engine's current
scaling) and the fast path of torch._scaled_mm on the GB10, where per-row
scales fall to a slower kernel. The numbers differ slightly from a BF16
training, as any quantization does: the benchmarks compare the losses.

FP8 tensor cores exist from compute capability 8.9 (Ada, Hopper, Blackwell);
an older GPU is refused.

The expert stacks of a mixture of experts that carry adapters (meepsquared._moe)
are held as FP8 too, with one scale per expert. meepsquared's grouped kernel
(meepsquared._grouped) reads such a stack as it is and converts each tile in
registers. Where that kernel does not run, a stack is dequantized to the
compute dtype for each product and dropped again, in the backward pass as
well, where the input's gradient is the only one needed: PyTorch's grouped
FP8 matmul (_scaled_grouped_mm) runs only on compute capability 9.0 and
10.0. Either way the memory the experts take halves.
"""

import math

import torch
from torch import nn

E4M3_MAX = 448.0
# torch._scaled_mm takes a row count that is a multiple of this
ROW_MULTIPLE = 16
MIN_CAPABILITY = (8, 9)

try:
    import triton
    import triton.language as tl
except ImportError:  # the kernels need Triton; the CPU reference path does not
    triton = None

if triton is not None:

    @triton.jit
    def _amax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        v = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_max(out_ptr, tl.max(tl.abs(v), axis=0))

    @triton.jit
    def _cast_kernel(x_ptr, scale_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        v = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr)
        tl.store(out_ptr + offsets, (v / s).to(tl.float8e4nv), mask=mask)

    @triton.jit
    def _grouped_scale_kernel(x_ptr, scale_ptr, out_ptr, n, per_group, DIVIDE: tl.constexpr, BLOCK: tl.constexpr):
        """out = x / scale[group] (DIVIDE: quantize) or x * scale[group] (dequantize), stored in out's dtype."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        v = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + offsets // per_group, mask=mask, other=1.0)
        if DIVIDE:
            v = v / s
        else:
            v = v * s
        tl.store(out_ptr + offsets, v, mask=mask)


def available(device=None) -> tuple[bool, str]:
    """Whether FP8 products run here, and why not when they do not."""
    if not torch.cuda.is_available():
        return False, "FP8 products run on a CUDA GPU"
    if triton is None:
        return False, "the FP8 kernels need Triton"
    capability = torch.cuda.get_device_capability(device)
    if capability < MIN_CAPABILITY:
        return False, (f"FP8 tensor cores need compute capability {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]} or later "
                       f"(Ada, Hopper, Blackwell); this GPU is {capability[0]}.{capability[1]}")
    return True, ""


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x as FP8 e4m3 with one scale: x ≈ fp8 * scale. On the GPU, two fused kernels; elsewhere, PyTorch."""
    x = x.contiguous()
    if x.is_cuda and triton is not None:
        n = x.numel()
        amax = torch.zeros(1, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(n, 8192),)
        _amax_kernel[grid](x, amax, n, BLOCK=8192)
        scale = torch.clamp(amax, min=1e-12) / E4M3_MAX
        out = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
        _cast_kernel[grid](x, scale, out, n, BLOCK=8192)
        return out, scale.reshape(())
    scale = torch.clamp(x.detach().abs().amax().float(), min=1e-12) / E4M3_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def _grouped_scale(x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, divide: bool) -> torch.Tensor:
    """out = x / scale per leading group (divide) or x * scale; x is (groups, ...), scale (groups,)."""
    n, per_group = x.numel(), x[0].numel()
    if x.is_cuda and triton is not None:
        grid = (triton.cdiv(n, 8192),)
        _grouped_scale_kernel[grid](x, scale, out, n, per_group, DIVIDE=divide, BLOCK=8192)
        return out
    s = scale.float().view(-1, *([1] * (x.dim() - 1)))
    return out.copy_((x.float() / s) if divide else (x.float() * s))


def quantize_stack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A stack of expert weights (experts, ...) as FP8 e4m3 with one scale per expert: weight ≈ fp8 * scale[expert]."""
    weight = weight.contiguous()
    amax = weight.detach().abs().flatten(1).amax(dim=1).float()
    scale = torch.clamp(amax, min=1e-12) / E4M3_MAX
    out = torch.empty(weight.shape, device=weight.device, dtype=torch.float8_e4m3fn)
    return _grouped_scale(weight.detach(), scale, out, divide=True), scale


def dequantize_stack(weight8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The stack back in a compute dtype, for one product; the result is not kept."""
    out = torch.empty(weight8.shape, device=weight8.device, dtype=dtype)
    return _grouped_scale(weight8, scale, out, divide=False)


class _Fp8GroupedProduct(torch.autograd.Function):
    """The grouped product of x with a stack held as FP8, as Transformers' _grouped_linear computes it.

    The stack is dequantized for the forward product and again for the
    backward one, which gives x's gradient only: the stack is frozen, and
    keeping it dequantized would cost what FP8 saves.
    """

    @staticmethod
    def forward(ctx, x, weight8, scale, offsets, bias, is_transposed):
        from transformers.integrations.moe import _grouped_linear

        ctx.weight8, ctx.scale, ctx.offsets, ctx.is_transposed = weight8, scale, offsets, is_transposed
        return _grouped_linear(x, dequantize_stack(weight8, scale, x.dtype), offsets, bias=bias, is_transposed=is_transposed)

    @staticmethod
    def backward(ctx, grad_out):
        from transformers.integrations.moe import _grouped_linear

        weight = dequantize_stack(ctx.weight8, ctx.scale, grad_out.dtype)
        # x @ W^T has the gradient grad @ W, which is the transposed form of the same product, and the other way round
        grad_x = _grouped_linear(grad_out, weight, ctx.offsets, is_transposed=not ctx.is_transposed)
        return grad_x, None, None, None, None, None


def grouped_product(x, weight8, scale, offsets, bias, is_transposed: bool) -> torch.Tensor:
    """x against a stack held as FP8, per expert group; see _Fp8GroupedProduct."""
    return _Fp8GroupedProduct.apply(x, weight8, scale, offsets, bias, is_transposed)


STACK_SCALE = "_scale_inv"


def convert_stacks(experts) -> int:
    """Every frozen stack of an experts module held as FP8, its scale as <name>_scale_inv; returns how many."""
    converted = 0
    for name, parameter in list(experts.named_parameters(recurse=False)):
        if parameter.dim() != 3 or parameter.requires_grad or not parameter.is_floating_point():
            continue
        if parameter.dtype == torch.float8_e4m3fn and hasattr(experts, name + STACK_SCALE):
            # FP8 already, with its scale: the stacks a shared vLLM holds (meepsquared._shared_vllm)
            converted += 1
            continue
        weight8, scale = quantize_stack(parameter.detach())
        setattr(experts, name, nn.Parameter(weight8, requires_grad=False))
        experts.register_buffer(name + STACK_SCALE, scale)
        converted += 1
    if converted:
        experts._meepsquared_fp8 = True
    return converted


def _rows_padded(x8: torch.Tensor):
    """x8 with its rows padded to a multiple of ROW_MULTIPLE, and the row count to cut back to."""
    rows = x8.shape[0]
    padded = math.ceil(rows / ROW_MULTIPLE) * ROW_MULTIPLE
    if padded == rows:
        return x8, rows
    return torch.nn.functional.pad(x8.view(torch.uint8), (0, 0, 0, padded - rows)).view(torch.float8_e4m3fn), rows


def scaled_mm(x: torch.Tensor, weight8: torch.Tensor, weight_scale: torch.Tensor, transpose_weight: bool) -> torch.Tensor:
    """x @ W^T (transpose_weight) or x @ W, x quantized on the way in, in BF16 out.

    weight8 is (out, in) row-major; torch._scaled_mm takes it as is for x @ W,
    and its transposed view for x @ W^T.
    """
    if not x.is_cuda:  # the reference path, for CPUs and tests
        w = weight8.float() * weight_scale
        return (x.float() @ (w.t() if transpose_weight else w)).to(torch.bfloat16)
    x8, x_scale = quantize(x.reshape(-1, x.shape[-1]))
    x8, rows = _rows_padded(x8)
    b = weight8.t() if transpose_weight else weight8
    out = torch._scaled_mm(x8, b, scale_a=x_scale.reshape(1), scale_b=weight_scale.reshape(1), out_dtype=torch.bfloat16)
    return out[:rows].reshape(*x.shape[:-1], out.shape[-1])


class _Fp8Product(torch.autograd.Function):
    """y = x @ W^T with W in FP8; the backward gives x's gradient only, W being frozen."""

    @staticmethod
    def forward(ctx, x, weight8, weight_scale):
        ctx.weight8, ctx.weight_scale = weight8, weight_scale
        return scaled_mm(x, weight8, weight_scale, transpose_weight=True)

    @staticmethod
    def backward(ctx, grad_out):
        return scaled_mm(grad_out, ctx.weight8, ctx.weight_scale, transpose_weight=False), None, None


class Fp8Linear(nn.Module):
    """A frozen linear layer held as FP8 with one scale; its bias, if any, stays as it was.

    weight is the FP8 tensor and weight_scale_inv its scale (W ≈ weight *
    weight_scale_inv), the names the fused LoRA kernels read them by.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        weight8, scale = quantize(linear.weight.detach())
        self.register_buffer("weight", weight8)
        self.register_buffer("weight_scale_inv", scale)
        self.bias = linear.bias
        self.dtype = linear.weight.dtype

    @torch.compiler.disable  # Dynamo cannot trace the product's autograd function
    def forward(self, x):
        out = _Fp8Product.apply(x, self.weight, self.weight_scale_inv)
        if self.bias is not None:
            out = out + self.bias
        return out

    def dequantized(self) -> torch.Tensor:
        """The weight back in its dtype, for a merged save."""
        return (self.weight.float() * self.weight_scale_inv).to(self.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8 e4m3, one scale"


# The routers of a mixture of experts: their decisions stay as the BF16 weights make them
ROUTERS = ("gate", "router")
# torch._scaled_mm takes a weight whose two sizes are multiples of this
SIZE_MULTIPLE = 16


def convertible(module, name: str, trainable: set) -> bool:
    """A frozen nn.Linear of the decoder, not an adapter, not the output layer, not a router, its sizes multiples of 16."""
    return (type(module) is nn.Linear and module.weight.dim() == 2 and not module.weight.requires_grad
            and "lora_" not in name and not name.endswith("lm_head") and name not in trainable
            and name.rsplit(".", 1)[-1] not in ROUTERS
            and module.in_features % SIZE_MULTIPLE == 0 and module.out_features % SIZE_MULTIPLE == 0)


def apply(model) -> int:
    """Every frozen linear layer of the decoder layers held as FP8, and every expert stack whose adapters meepsquared._moe
    computes; returns how many. Applying again changes nothing."""
    from meepsquared import _checkpointing, _moe

    ok, why = available(next(model.parameters()).device)
    if not ok:
        raise RuntimeError(f"load_in_fp8=True: {why}")
    trainable = {name.rsplit(".", 1)[0] for name, p in model.named_parameters() if p.requires_grad}
    converted = 0
    for layer in _checkpointing.layers(model):
        for name, module in list(layer.named_modules()):
            if _moe.routed(module):
                converted += convert_stacks(_moe._wrapper_chain(module)[1])
            if not convertible(module, name, trainable):
                continue
            parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
            setattr(parent, name.rsplit(".", 1)[-1], Fp8Linear(module).to(module.weight.device))
            converted += 1
    if not converted:
        raise ValueError("load_in_fp8=True found no frozen linear layer to hold as FP8")
    return converted
