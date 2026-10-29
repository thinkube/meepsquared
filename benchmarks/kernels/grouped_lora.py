# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The expert products at Qwen3.6-35B-A3B's sizes: meepsquared's grouped kernel against PyTorch's grouped matmul.

    python grouped_lora.py [--rows 262144] [--experts 256] [--skew]

Times, in milliseconds, one grouped product of `rows` rows over `experts`
experts for the two projections of a layer (2048 -> 1024 and 512 -> 2048),
with the stack in BF16 and in FP8, for PyTorch's grouped matmul (a loop of
one product per expert on a GB10) and meepsquared's kernel. --skew routes the rows
unevenly, as a trained router does.
"""

import argparse
import time

import torch

from meepsquared import _grouped

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=262144)
parser.add_argument("--experts", type=int, default=256)
parser.add_argument("--skew", action="store_true", help="uneven rows per expert")
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--block-m", type=int, default=_grouped.BLOCK_M, help="rows per tile of the kernel")
a = parser.parse_args()

device = "cuda"
torch.manual_seed(0)
if a.skew:
    weights = torch.rand(a.experts) ** 3
    counts = (weights / weights.sum() * a.rows).round().to(torch.int32)
    counts[-1] += a.rows - counts.sum()
else:
    counts = torch.full((a.experts,), a.rows // a.experts, dtype=torch.int32)
    counts[-1] += a.rows - counts.sum()
offsets = torch.cumsum(counts, 0, dtype=torch.int32).to(device)


def timed(f):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(a.repeats):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / a.repeats * 1000


def torch_grouped(x, weight, is_transposed):
    from transformers.integrations.moe import _grouped_linear

    return _grouped_linear(x, weight, offsets, is_transposed=is_transposed)


# The ceiling: the kernel with one expert is a dense product, against torch.mm (cuBLAS)
xd = torch.randn(a.rows, 2048, device=device, dtype=torch.bfloat16)
wd = torch.randn(1, 1024, 2048, device=device, dtype=torch.bfloat16) * 0.02
one = torch.tensor([a.rows], dtype=torch.int32, device=device)
ms_mm = timed(lambda: xd @ wd[0].t())
ms_dense = timed(lambda: _grouped.grouped_linear(xd, wd, one, False, block_m=a.block_m))
print(f"dense {a.rows} x 2048 x 1024: torch.mm {ms_mm:.2f} ms, the kernel as one expert {ms_dense:.2f} ms")

print(f"{a.rows} rows over {a.experts} experts{' (skewed)' if a.skew else ''}, {a.repeats} repeats, tiles of {a.block_m} rows; milliseconds per product")
print(f"{'product':34s} {'torch, BF16':>12s} {'kernel, BF16':>13s} {'FP8 weights':>12s} {'FP8 products':>13s}")
for name, K, N, is_transposed in (("gate_up: 2048 -> 1024 (out, in)", 2048, 1024, False),
                                  ("down: 512 -> 2048 (out, in)", 512, 2048, False),
                                  ("backward of gate_up: 1024 -> 2048", 1024, 2048, True)):
    x = torch.randn(a.rows, K, device=device, dtype=torch.bfloat16)
    shape = (a.experts, K, N) if is_transposed else (a.experts, N, K)
    weight = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.02
    weight8 = weight.to(torch.float8_e4m3fn)
    out = _grouped.grouped_linear(x, weight, offsets, is_transposed, block_m=a.block_m)
    reference = torch_grouped(x, weight, is_transposed).float()
    error = (out.float() - reference).abs().max().item()
    out8 = _grouped.grouped_linear(x, weight8, offsets, is_transposed, block_m=a.block_m, products_fp8=True).float()
    error8 = ((out8 - reference).norm() / reference.norm()).item()
    ms_torch = timed(lambda: torch_grouped(x, weight, is_transposed))
    ms_kernel = timed(lambda: _grouped.grouped_linear(x, weight, offsets, is_transposed, block_m=a.block_m))
    ms_w8 = timed(lambda: _grouped.grouped_linear(x, weight8, offsets, is_transposed, block_m=a.block_m, products_fp8=False))
    ms_p8 = timed(lambda: _grouped.grouped_linear(x, weight8, offsets, is_transposed, block_m=a.block_m, products_fp8=True))
    print(f"{name:34s} {ms_torch:12.2f} {ms_kernel:13.2f} {ms_w8:12.2f} {ms_p8:13.2f}   BF16 max |diff| {error:.4f}; FP8 products relative error {error8:.4f}")

# The whole product with rank-16 adapters, forward and backward: the kernel against meepsquared's PyTorch path
from meepsquared._moe import _grouped_lora  # noqa: E402

print("\nwith rank-16 adapters, forward + backward (gradients of x and the adapters); milliseconds")
print(f"{'product':34s} {'torch path':>12s} {'kernel':>12s} {'FP8 weights':>12s} {'FP8 products':>13s}")
for name, K, N, is_transposed in (("gate_up: 2048 -> 1024 (out, in)", 2048, 1024, False),
                                  ("down: 512 -> 2048 (out, in)", 512, 2048, False)):
    x = torch.randn(a.rows, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    shape = (a.experts, K, N) if is_transposed else (a.experts, N, K)
    weight = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.02
    weight8 = weight.to(torch.float8_e4m3fn)
    scale8 = torch.full((a.experts,), 1.0, device=device)
    lhs = (torch.randn(a.experts, N, 16, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_()
    rhs = (torch.randn(a.experts, 16, K, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_()

    grad = torch.randn(a.rows, N, device=device, dtype=torch.bfloat16)  # as the next layer gives it: a full tensor

    def torch_path():
        out = torch_grouped(x, weight, is_transposed)
        out.add_(_grouped_lora(x, lhs, rhs, 2.0, offsets, is_transposed))
        out.backward(grad)

    def kernel_path(w, s, products_fp8=None):
        _grouped.lora_linear(x, w, s, offsets, is_transposed, lhs, rhs, 2.0, products_fp8=products_fp8).backward(grad)

    ms_t = timed(torch_path)
    ms_k = timed(lambda: kernel_path(weight, None))
    ms_w8 = timed(lambda: kernel_path(weight8, scale8, False))
    ms_p8 = timed(lambda: kernel_path(weight8, scale8, True))
    print(f"{name:34s} {ms_t:12.2f} {ms_k:12.2f} {ms_w8:12.2f} {ms_p8:13.2f}")
