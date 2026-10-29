# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The experts' products with their LoRA adapters, in one kernel launch each.

The rows of x are sorted by expert; offsets[e] is the end of expert e's rows.
For every expert e and its rows x_e the forward computes

    out_e = x_e @ W_e^T + s · (x_e @ A_e^T) @ B_e^T + bias_e

with the stack W as (experts, out, in) or, when is_transposed, (experts, in,
out), in BF16 or in FP8 (e4m3) with one scale per expert, and the adapters A
(rank, in) and B (out, rank) in BF16: PEFT's lhs and rhs factors, in the
layout of the stack. The product x_e @ A_e^T ("low", rows by rank) is kept for
the backward pass.

PyTorch's grouped matmul computes the base product, but on a GPU without a
kernel for it (compute capability below 9.0, or 12.x such as the GB10) it
runs as a loop of one product per expert, and a training step comes to be
bound by launching them; the adapters cost two more loops. Here one launch
computes everything: the low-rank product is accumulated in the main loop
from the x tile already in registers, and applied in the epilogue as a
product of rank depth.

The work is tiled: each expert's rows are cut into tiles of BLOCK_M rows, so
no tile spans two experts and an expert with no rows has no tile. A program
computes one (rows, out) tile; the tile's expert and first row come from two
small arrays built on the GPU (tile_map), with no host round trip: the grid
is sized from an upper bound on the number of tiles, and programs beyond the
real count return at once.

The backward pass is three launches: the same kernel for the input's
gradient (the stack read the other way round, the adapters swapping roles),
and a reduction kernel, once per adapter, for the adapters' gradients
(sums over each expert's rows of grad_e^T @ low_e). The stack is frozen and
gets no gradient.

See design/grouped-lora-kernel.md.
"""

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the kernels need Triton; the references below do not
    triton = None

BLOCK_M = 128
RANK_MULTIPLE = 16  # tl.dot takes sizes that are multiples of 16
# PyTorch has grouped matmul kernels of its own for these; elsewhere it loops over the experts
TORCH_HAS_GROUPED_MM = (9, 10)


def wanted(device) -> bool:
    """Whether the kernels run the experts here: Triton, a CUDA GPU, and no grouped matmul of PyTorch's own.

    MEEPSQUARED_GROUPED_KERNEL=0 or 1 overrides the decision, for a measurement.
    """
    forced = os.environ.get("MEEPSQUARED_GROUPED_KERNEL")
    if forced in ("0", "1"):
        return forced == "1"
    if triton is None or device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device)[0] not in TORCH_HAS_GROUPED_MM


def fp8_products(device) -> bool:
    """Whether an FP8 stack's products run on the FP8 tensor cores, the activations quantized per row and tile.

    Off unless MEEPSQUARED_FP8_PRODUCTS=1: measured on a GB10 (compute capability
    12.1, Triton 3.7), the FP8 product is slower than the BF16 one with the
    weights converted in registers (20 ms against 14 for 262,144 rows by
    2048 by 1024 over 256 experts) and carries a 4% relative error from the
    quantization of the activations. The weights stay FP8 either way; the
    switch exists for a measurement on another GPU or toolchain.
    """
    return os.environ.get("MEEPSQUARED_FP8_PRODUCTS") == "1"


def max_tiles(rows: int, experts: int, block_m: int = BLOCK_M) -> int:
    """An upper bound on the tiles: every expert's last tile may be partial."""
    return rows // block_m + experts


def tile_map(offsets: torch.Tensor, rows: int, block_m: int = BLOCK_M) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(tile_expert, tile_row, tiles) for `rows` rows sorted by expert.

    offsets[e] is the end of expert e's rows (int32, on the device). The two
    arrays have max_tiles entries, computed on the device; tiles, a 0-d
    tensor on the device, says how many are real. Entries beyond that hold
    the last expert and are never used.
    """
    experts = offsets.numel()
    starts = torch.cat([offsets.new_zeros(1), offsets[:-1]])
    tiles_per_expert = (offsets - starts + block_m - 1) // block_m
    tile_ends = torch.cumsum(tiles_per_expert, 0)
    t = torch.arange(max_tiles(rows, experts, block_m), device=offsets.device, dtype=tile_ends.dtype)
    tile_expert = torch.searchsorted(tile_ends, t, right=True).clamp(max=experts - 1)
    tile_row = starts[tile_expert] + (t - (tile_ends - tiles_per_expert)[tile_expert]) * block_m
    return tile_expert.to(torch.int32), tile_row.to(torch.int32), tile_ends[-1].to(torch.int32)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "GROUP_T": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_T": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_T": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 32, "GROUP_T": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_T": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 32, "GROUP_T": 8}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "GROUP_T": 8}, num_warps=8, num_stages=3),
        ],
        key=["K", "N", "R", "W_FP8", "X_FP8", "TRANSPOSED", "HAS_LORA", "BLOCK_M"],
    )
    @triton.jit
    def _grouped_gemm_kernel(
        x_ptr, w_ptr, scale_ptr, a_ptr, b_ptr, bias_ptr, out_ptr, low_ptr,
        tile_expert_ptr, tile_row_ptr, offsets_ptr, tiles_ptr,
        K, N, scaling,
        stride_xm, stride_xk, stride_we, stride_wk, stride_wn,
        stride_ae, stride_ak, stride_ar, stride_be, stride_br, stride_bn, stride_bias_e,
        stride_om, stride_on, stride_lm,
        R: tl.constexpr, W_FP8: tl.constexpr, X_FP8: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_LORA: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        STORE_LOW: tl.constexpr, TRANSPOSED: tl.constexpr, EVEN_K: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_T: tl.constexpr,
    ):
        """out[rows of tile t, n-block] = scale[e] · x[rows] @ W[e] + scaling · (x[rows] @ A[e]) @ B[e] + bias[e].

        W is addressed as (k, n), A as (k, r) and B as (r, n) through their
        strides, whatever their layout in memory. The programs are ordered so
        that GROUP_T consecutive tiles (which share an expert, hence its
        weight) run next to each other over all n-blocks: the weight tiles
        stay in the L2 cache between them.

        With W_FP8 the weight tile arrives as FP8. With X_FP8 as well, each
        x tile is scaled per row to FP8 by its own largest value, the two
        multiply on the FP8 tensor cores, and the row scales are undone on
        the partial product; without X_FP8 the weight tile is converted to
        BF16 in registers. The adapters always multiply the BF16 x tile.
        """
        pid = tl.program_id(0)
        n_blocks = tl.cdiv(N, BLOCK_N)
        programs_per_group = GROUP_T * n_blocks
        group = pid // programs_per_group
        first_t = group * GROUP_T
        group_size = tl.minimum(tl.num_programs(0) // n_blocks - first_t, GROUP_T)
        t = first_t + (pid % programs_per_group) % group_size
        n_block = (pid % programs_per_group) // group_size
        if t >= tl.load(tiles_ptr):
            return
        expert = tl.load(tile_expert_ptr + t)
        row0 = tl.load(tile_row_ptr + t)
        row_end = tl.load(offsets_ptr + expert)
        n0 = n_block * BLOCK_N

        rows = row0 + tl.arange(0, BLOCK_M)
        cols = n0 + tl.arange(0, BLOCK_N)
        ranks = tl.arange(0, R)
        row_mask = rows < row_end
        col_mask = cols < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_low = tl.zeros((BLOCK_M, R), dtype=tl.float32)
        e64 = expert.to(tl.int64)
        w_base = w_ptr + e64 * stride_we
        a_base = a_ptr + e64 * stride_ae
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            if EVEN_K:
                x_tile = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk, mask=row_mask[:, None], other=0.0)
            else:
                x_tile = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :] * stride_xk,
                                 mask=row_mask[:, None] & (ks < K)[None, :], other=0.0)
            if TRANSPOSED:
                # the stack is (experts, in, out): a (k, n) tile is contiguous along n
                if EVEN_K:
                    w_tile = tl.load(w_base + ks[:, None] * stride_wk + cols[None, :] * stride_wn, mask=col_mask[None, :], other=0.0)
                else:
                    w_tile = tl.load(w_base + ks[:, None] * stride_wk + cols[None, :] * stride_wn,
                                     mask=(ks < K)[:, None] & col_mask[None, :], other=0.0)
            else:
                # the stack is (experts, out, in): an (n, k) tile is contiguous along k, and transposed in registers
                if EVEN_K:
                    w_tile = tl.load(w_base + cols[:, None] * stride_wn + ks[None, :] * stride_wk, mask=col_mask[:, None], other=0.0)
                else:
                    w_tile = tl.load(w_base + cols[:, None] * stride_wn + ks[None, :] * stride_wk,
                                     mask=col_mask[:, None] & (ks < K)[None, :], other=0.0)
                w_tile = tl.trans(w_tile)
            if X_FP8:
                row_scale = tl.maximum(tl.max(tl.abs(x_tile.to(tl.float32)), axis=1), 1e-12) / 448.0  # e4m3's largest value
                x8 = (x_tile.to(tl.float32) / row_scale[:, None]).to(tl.float8e4nv)
                acc += tl.dot(x8, w_tile) * row_scale[:, None]
            else:
                if W_FP8:
                    w_tile = w_tile.to(tl.bfloat16)
                acc = tl.dot(x_tile, w_tile, acc)
            if HAS_LORA:
                if EVEN_K:
                    a_tile = tl.load(a_base + ks[:, None] * stride_ak + ranks[None, :] * stride_ar)
                else:
                    a_tile = tl.load(a_base + ks[:, None] * stride_ak + ranks[None, :] * stride_ar,
                                     mask=(ks < K)[:, None], other=0.0)
                acc_low = tl.dot(x_tile, a_tile, acc_low)
        if HAS_SCALE:
            acc = acc * tl.load(scale_ptr + expert)
        if HAS_LORA:
            low = acc_low.to(tl.bfloat16)
            if STORE_LOW:
                tl.store(low_ptr + rows[:, None] * stride_lm + ranks[None, :], low, mask=row_mask[:, None])
            b_tile = tl.load(b_ptr + e64 * stride_be + ranks[:, None] * stride_br + cols[None, :] * stride_bn,
                             mask=col_mask[None, :], other=0.0)
            acc += tl.dot(low, b_tile) * scaling
        if HAS_BIAS:
            acc += tl.load(bias_ptr + e64 * stride_bias_e + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on, acc.to(out_ptr.dtype.element_ty),
                 mask=row_mask[:, None] & col_mask[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 128, "BLOCK_M": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 128, "BLOCK_M": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_M": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "R"],
    )
    @triton.jit
    def _grouped_reduce_kernel(
        g_ptr, l_ptr, out_ptr, offsets_ptr,
        N, scaling,
        stride_gm, stride_gn, stride_lm, stride_oe, stride_on, stride_or,
        R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """out[e, n, r] = scaling · sum over expert e's rows of g[row, n] · l[row, r]: an adapter's gradient."""
        expert = tl.program_id(0)
        n0 = tl.program_id(1) * BLOCK_N
        start = tl.load(offsets_ptr + expert - 1, mask=expert > 0, other=0)
        end = tl.load(offsets_ptr + expert)
        cols = n0 + tl.arange(0, BLOCK_N)
        ranks = tl.arange(0, R)
        col_mask = cols < N
        acc = tl.zeros((BLOCK_N, R), dtype=tl.float32)
        for row0 in range(start, end, BLOCK_M):
            rows = row0 + tl.arange(0, BLOCK_M)
            row_mask = rows < end
            g_tile = tl.load(g_ptr + rows[:, None] * stride_gm + cols[None, :] * stride_gn,
                             mask=row_mask[:, None] & col_mask[None, :], other=0.0)
            l_tile = tl.load(l_ptr + rows[:, None] * stride_lm + ranks[None, :], mask=row_mask[:, None], other=0.0)
            acc = tl.dot(tl.trans(g_tile), l_tile, acc)
        out = out_ptr + expert.to(tl.int64) * stride_oe + cols[:, None] * stride_on + ranks[None, :] * stride_or
        tl.store(out, (acc * scaling).to(out_ptr.dtype.element_ty), mask=col_mask[:, None])


def available() -> bool:
    """Whether the kernels can run here: Triton and a CUDA GPU."""
    return triton is not None and torch.cuda.is_available()


def _stack_strides(weight: torch.Tensor, K: int, is_transposed: bool):
    """(N, stride_e, stride_k, stride_n) of a stack addressed as (k, n)."""
    if is_transposed:
        if weight.shape[1] != K:
            raise ValueError(f"a transposed stack (experts, in, out) with in={weight.shape[1]} does not take x with {K} columns")
        return weight.shape[2], weight.stride(0), weight.stride(1), weight.stride(2)
    if weight.shape[2] != K:
        raise ValueError(f"a stack (experts, out, in) with in={weight.shape[2]} does not take x with {K} columns")
    return weight.shape[1], weight.stride(0), weight.stride(2), weight.stride(1)


def _padded_rank(factor: torch.Tensor, dim: int) -> torch.Tensor:
    """The factor with its rank dimension padded with zeros to a multiple of RANK_MULTIPLE."""
    pad = -factor.shape[dim] % RANK_MULTIPLE
    if not pad:
        return factor
    shape = list(factor.shape)
    shape[dim] = pad
    return torch.cat([factor, factor.new_zeros(shape)], dim=dim)


class _Adapters:
    """PEFT's factors as the kernel addresses them: a as (e, k, r), b as (e, r, n), both with the rank padded.

    delta = lhs @ rhs in the stack's layout: (out, in) means lhs (out, rank)
    and rhs (rank, in); (in, out) means lhs (in, rank) and rhs (rank, out).
    """

    def __init__(self, lhs, rhs, is_transposed):
        self.rank = lhs.shape[2]
        if is_transposed:
            self.a, self.b = _padded_rank(lhs, 2), _padded_rank(rhs, 1)  # a[k, r] = lhs[in, rank]; b[r, n] = rhs[rank, out]
            self.a_strides = (self.a.stride(0), self.a.stride(1), self.a.stride(2))
            self.b_strides = (self.b.stride(0), self.b.stride(1), self.b.stride(2))
        else:
            self.a, self.b = _padded_rank(rhs, 1), _padded_rank(lhs, 2)  # a[k, r] = rhs[rank, in]^T; b[r, n] = lhs[out, rank]^T
            self.a_strides = (self.a.stride(0), self.a.stride(2), self.a.stride(1))
            self.b_strides = (self.b.stride(0), self.b.stride(2), self.b.stride(1))
        self.padded = self.b.shape[1] if is_transposed else self.b.shape[2]

    def swapped(self):
        """The adapters for the backward product: b takes a's role, addressed (e, k=n, r), and a b's, addressed (e, r, n=k)."""
        other = object.__new__(_Adapters)
        other.rank, other.padded = self.rank, self.padded
        other.a, other.a_strides = self.b, (self.b_strides[0], self.b_strides[2], self.b_strides[1])
        other.b, other.b_strides = self.a, (self.a_strides[0], self.a_strides[2], self.a_strides[1])
        return other


def _launch(x, weight, scale, offsets, is_transposed, adapters, scaling, bias, store_low, block_m, products_fp8=None):
    """One launch of the product kernel. Returns (out, low), low being None without adapters or store_low.

    products_fp8 says whether an FP8 stack's products run in FP8 (fp8_products decides when None).
    """
    rows, K = x.shape
    w_fp8 = weight.dtype == torch.float8_e4m3fn
    x_fp8 = w_fp8 and (fp8_products(x.device) if products_fp8 is None else products_fp8)
    N, stride_we, stride_wk, stride_wn = _stack_strides(weight, K, is_transposed)
    out = torch.empty((rows, N), device=x.device, dtype=x.dtype)
    R = adapters.padded if adapters is not None else RANK_MULTIPLE
    low = torch.empty((rows, R), device=x.device, dtype=torch.bfloat16) if (adapters is not None and store_low) else None
    if rows == 0:
        return out, low
    tile_expert, tile_row, tiles = tile_map(offsets, rows, block_m)
    grid = lambda meta: (tile_expert.numel() * triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    none = x  # for a pointer argument the kernel never reads
    _grouped_gemm_kernel[grid](
        x, weight, scale if scale is not None else none,
        adapters.a if adapters is not None else none, adapters.b if adapters is not None else none,
        bias if bias is not None else none, out, low if low is not None else none,
        tile_expert, tile_row, offsets, tiles,
        K, N, float(scaling),
        x.stride(0), x.stride(1), stride_we, stride_wk, stride_wn,
        *(adapters.a_strides if adapters is not None else (0, 0, 0)),
        *(adapters.b_strides if adapters is not None else (0, 0, 0)),
        bias.stride(0) if bias is not None else 0,
        out.stride(0), out.stride(1), low.stride(0) if low is not None else 0,
        R=R, W_FP8=w_fp8, X_FP8=x_fp8, HAS_SCALE=scale is not None, HAS_LORA=adapters is not None,
        HAS_BIAS=bias is not None, STORE_LOW=low is not None, TRANSPOSED=is_transposed, EVEN_K=K % 128 == 0,
        BLOCK_M=block_m,
    )
    return out, low


def grouped_linear(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor, is_transposed: bool,
                   block_m: int = BLOCK_M, products_fp8=None) -> torch.Tensor:
    """x (rows, in) by the stack (experts, out, in), or (experts, in, out) when is_transposed, per expert group.

    offsets (experts,) int32 on the device: the end of each expert's rows.
    Returns (rows, out) in x's dtype. A stack in FP8 is converted in registers
    without a scale: the caller folds the scale in. The base product alone,
    for the tests and the benchmark.
    """
    if triton is None:
        raise RuntimeError("the grouped kernel needs Triton")
    return _launch(x, weight, None, offsets, is_transposed, None, 1.0, None, False, block_m, products_fp8)[0]


def _reduce(g: torch.Tensor, low: torch.Tensor, offsets: torch.Tensor, scaling: float) -> torch.Tensor:
    """(experts, N, R): scaling · sum over each expert's rows of g^T @ low."""
    experts = offsets.numel()
    N, R = g.shape[1], low.shape[1]
    out = torch.empty((experts, N, R), device=g.device, dtype=torch.bfloat16)
    grid = lambda meta: (experts, triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    _grouped_reduce_kernel[grid](g, low, out, offsets, N, float(scaling),
                                 g.stride(0), g.stride(1), low.stride(0), out.stride(0), out.stride(1), out.stride(2), R=R)
    return out


class _LoraGroupedProduct(torch.autograd.Function):
    """The experts' product with the adapters; gradients for x and the adapters, none for the frozen stack."""

    @staticmethod
    def forward(ctx, x, weight, scale, offsets, is_transposed, lhs, rhs, scaling, bias, products_fp8):
        adapters = _Adapters(lhs, rhs, is_transposed) if lhs is not None else None
        out, low = _launch(x, weight, scale, offsets, is_transposed, adapters, scaling, bias, True, BLOCK_M, products_fp8)
        ctx.save_for_backward(x, weight, offsets, *((low,) if low is not None else ()))
        ctx.adapters, ctx.scale, ctx.is_transposed, ctx.scaling = adapters, scale, is_transposed, scaling
        ctx.products_fp8 = products_fp8
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, offsets, *rest = ctx.saved_tensors
        adapters = ctx.adapters
        grad_out = grad_out.contiguous()
        if adapters is None:
            grad_x, _ = _launch(grad_out, weight, ctx.scale, offsets, not ctx.is_transposed, None, 1.0, None, False, BLOCK_M,
                                ctx.products_fp8)
            return grad_x, None, None, None, None, None, None, None, None, None
        low = rest[0]
        # grad_x = grad @ W, the stack read the other way, + scaling · (grad @ B^T) @ A^T: the adapters swap roles
        grad_x, low2 = _launch(grad_out, weight, ctx.scale, offsets, not ctx.is_transposed, adapters.swapped(),
                               ctx.scaling, None, True, BLOCK_M, ctx.products_fp8)
        grad_a = _reduce(x, low2, offsets, ctx.scaling)        # (experts, in, R): the gradient of a, addressed (k, r)
        grad_b = _reduce(grad_out, low, offsets, ctx.scaling)  # (experts, out, R): the gradient of b, addressed (n, r)
        rank = adapters.rank
        if ctx.is_transposed:  # a = lhs (in, rank), b = rhs (rank, out)
            grad_lhs, grad_rhs = grad_a[:, :, :rank], grad_b.transpose(1, 2)[:, :rank, :]
        else:  # a = rhs (rank, in) read transposed, b = lhs (out, rank) read transposed
            grad_lhs, grad_rhs = grad_b[:, :, :rank], grad_a.transpose(1, 2)[:, :rank, :]
        return grad_x, None, None, None, None, grad_lhs, grad_rhs, None, None, None


def lora_linear(x, weight, scale, offsets, is_transposed: bool, lhs=None, rhs=None, scaling: float = 1.0, bias=None,
                products_fp8=None):
    """The experts' product of x with the stack, the adapters (PEFT's lhs, rhs, scaling) and the bias stack (experts, out).

    scale (experts,) is the FP8 stack's per-expert scale, None for a BF16 stack. products_fp8 says whether an FP8
    stack's products run on the FP8 tensor cores; None lets fp8_products decide for the device.
    """
    if triton is None:
        raise RuntimeError("the grouped kernel needs Triton")
    return _LoraGroupedProduct.apply(x, weight, scale, offsets, is_transposed, lhs, rhs, scaling, bias, products_fp8)


def reference(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor, is_transposed: bool) -> torch.Tensor:
    """The base product, one expert at a time, in float32, for the tests."""
    out = torch.empty((x.shape[0], weight.shape[2] if is_transposed else weight.shape[1]), device=x.device, dtype=x.dtype)
    start = 0
    for e, end in enumerate(offsets.tolist()):
        w = weight[e].float()
        out[start:end] = (x[start:end].float() @ (w if is_transposed else w.t())).to(x.dtype)
        start = end
    return out


def reference_lora(x, weight, scale, offsets, is_transposed, lhs, rhs, scaling, bias=None):
    """The whole product with PyTorch operations, differentiable, for the tests."""
    out = []
    start = 0
    for e, end in enumerate(offsets.tolist()):
        w = weight[e].float() * (scale[e].float() if scale is not None else 1.0)
        if lhs is not None:
            w = w + scaling * (lhs[e].float() @ rhs[e].float())
        y = x[start:end].float() @ (w if is_transposed else w.t())
        if bias is not None:
            y = y + bias[e].float()
        out.append(y)
        start = end
    return torch.cat(out).to(x.dtype)
