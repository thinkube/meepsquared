# A grouped matmul kernel for the experts, with the adapters and FP8 weights inside

Design note. Status: cuts 1, 2 and 4 implemented in `src/meepsquared/_grouped.py`, measured, and on by default where PyTorch loops (`_moe` uses it). Step-level, Qwen3.6-35B-A3B, 30 steps measured from the eleventh, against the same runs without the kernel: BF16 13.3 → 11.7 s per step (706 → 799 tokens/s); FP8 stacks 14.0 → 10.9 s (669 → 857 tokens/s, 61 GiB); PEFT's computation 16.0 s (587). Cut 3 as FP8 tensor-core products (the activations quantized per row and tile in the main loop, Triton's FP8 `dot`) was implemented and measured on the GB10: slower than the BF16 product with the weights converted in registers — 20.5 ms against 14.3 (gate/up), 11.9 against 9.2 (down), 23.5 against 13.1 (the backward shape), 112 ms against 59 with adapters forward and backward — and with a 4.1% relative error on the product. It stays in the code behind `MEEPSQUARED_FP8_PRODUCTS=1`, off by default. Not done: fusing the activation between the two projections; a measurement on Hopper or B200.

## Measured so far (GB10, 262,144 rows over 256 experts, `benchmarks/kernels/grouped_lora.py`)

| Product | PyTorch's loop | kernel, BF16 | kernel, FP8 weights |
|---|---|---|---|
| gate/up 2048 → 1024, base only | 15.6 ms | 13.6 | 14.8 (20.5 as FP8 tensor-core products) |
| down 512 → 2048, base only | 13.2 | 9.0 | 9.1 |
| 1024 → 2048 (the backward shape), base only | 18.1 | 13.0 | 15.2 |
| gate/up with rank-16 adapters, forward + backward | 102.8 (Meep²'s PyTorch path) | 59.1 | 70.9 |
| down with rank-16 adapters, forward + backward | 71.5 | 32.2 | 37.8 |

The kernel as a single dense product: 12.5 ms against `torch.mm`'s 11.9 (cuBLAS). FP8 weights converted in registers are not faster than BF16: at about 1,000 rows per expert these products are bound by arithmetic, not by reading weights (the kernel runs at about 80 TFLOPS), so halving the weight bytes buys nothing. FP8 pays in memory, and the kernel removes the dequantization pass that cost 5% of a step; a speed gain from FP8 would need FP8 tensor-core products, the open question of a later cut.

## The problem, as measured

On an NVIDIA GB10 (compute capability 12.1) training LoRA on Qwen3.6-35B-A3B
(40 layers, 256 experts of 2048 → 2×512 → 2048, 8 experts per token):

- PyTorch's grouped matmul (`torch._grouped_mm`) has no kernel for this chip.
  It runs as a loop over the experts, one `mm` per expert: about 223,000
  `mm` launches per step. The step is bound by launching kernels, not by the
  GPU: 15.8 s of CPU time against 14.5 s of GPU time per step
  (`benchmarks/profiles/moe-meepsquared.txt`).
- The routed LoRA (`meepsquared._moe`) adds two more grouped products per
  projection, so two more loops of 256 launches each, and ends level with
  PEFT's own computation (which builds the full weight of every expert).
- PyTorch's FP8 grouped matmul (`torch._scaled_grouped_mm`) refuses the chip
  (it is for compute capability 9.0 and 10.0). FP8 expert stacks are
  dequantized to BF16 for each product, which costs 5–10% of a step at equal
  checkpointing; the memory saved (31 GiB) is the gain.
- One grouped product of 262,144 routed rows (32 × 1,024 tokens × 8
  experts) by 2048 → 1024 takes 15.8 ms as the loop runs it; each LoRA
  product (rank 16) 4–6 ms; PEFT's weight build 20.5 ms.

Each expert's weight (2048 × 1024 in BF16) is 4 MB and serves about 300
rows per step: the products are bound by reading weights, not by
arithmetic. Halving the weight bytes (FP8) is worth up to 2× on them; one
launch per product instead of 256 removes the CPU bound.

## Goal

One Triton kernel that computes, for every expert `e` and the rows routed
to it,

    out_e = x_e @ W_e^T + s · (x_e @ A_e^T) @ B_e^T

with `W_e` in BF16 or FP8 (e4m3, one scale per expert), the adapters
`A_e` (rank × in) and `B_e` (out × rank) in BF16, in **one launch** for the
forward pass and a small number of launches for the backward pass. The
result equals the current routed path to BF16 rounding.

Non-goals for the first version: FP8 activations (true FP8 tensor-core
products), fusing the gate activation between the two projections, expert
parallelism across GPUs, Hopper/B200 (PyTorch has kernels there).

## The kernel

**Layout.** The rows are sorted by expert (as `_routed_forward` already
does); `offsets[e]` is the end of expert `e`'s rows. The weight stack is
`(experts, out, in)` or `(experts, in, out)` (`is_transposed`); both
layouts are handled by strides, not by copies.

**Work assignment.** The output is tiled in `BM × BN` tiles. A tile belongs
to one expert and must not straddle two: each expert's row range is cut
into `ceil(rows_e / BM)` tiles. A small preparation kernel (or a prefix
sum on the GPU) builds two arrays, `tile_expert[t]` and `tile_row[t]`
(first row of tile `t`), so the main kernel's program `(t, n)` knows its
expert and rows without a host round trip. Experts with no rows produce no
tiles. The grid is `(tiles, out / BN)`.

**Main loop.** For a tile, loop over `K = in` in steps of `BK`:

    x_tile  = load x[rows, k:k+BK]                      (BM × BK, BF16)
    w_tile  = load W_e[n:n+BN, k:k+BK]                  (BN × BK)
    acc    += dot(x_tile, w_tile^T)                     (BM × BN, FP32)
    a_tile  = load A_e[:, k:k+BK]                       (R × BK)
    acc_low += dot(x_tile, a_tile^T)                    (BM × R,  FP32)

`R` is the LoRA rank padded to 16 (Triton's `dot` wants multiples of 16).
`acc_low` costs one extra small `dot` per step and `BM × R` registers; it
is the whole first LoRA product, computed from the `x_tile` already in
registers, so `x` is read once for the base and the adapter together.

**Epilogue.**

    b_tile = load B_e[n:n+BN, :]                        (BN × R)
    acc   += s · dot(acc_low.to(bf16), b_tile^T)       (BM × BN)
    acc   += bias_e[n:n+BN]                             (if the stack has biases)
    store out[rows, n:n+BN] = acc.to(bf16)

The second LoRA product is a `dot` of rank-16 depth per tile: negligible.

**FP8 weights.** When `W` is `float8_e4m3fn`, `w_tile` is loaded as FP8 and
converted in registers: `w_tile.to(bf16) * scale_e`. The product stays a
BF16 `dot`, so the numbers equal the dequantized path, and the weight bytes
read halve. This is the only FP8 the first version does. A later version
can quantize `x_tile` per tile and use FP8 `dot` (the tensor cores Triton
exposes on this chip); whether it pays is a measurement, not a plan.

**Backward.** The frozen weight needs no gradient. Three products:

1. `grad_x_e = grad_out_e @ W_e + s · (grad_out_e @ B_e) @ A_e`: the same
   kernel with the roles of `in` and `out` swapped (`W` read through its
   other layout, `B` where `A` was and `A` where `B` was). One launch.
2. `grad_B_e = s · grad_out_e^T @ low_e`, with `low_e = x_e @ A_e^T`
   (`out × R` per expert): a reduction over each expert's rows. One launch
   of a second kernel whose program is `(expert, n)`, looping over the
   expert's rows in `BM` chunks. `low_e` is recomputed from `x` and `A`
   inside that loop (rank 16: cheap), so the forward saves nothing for it.
3. `grad_A_e = s · (grad_out_e @ B_e)^T @ x_e` (`R × in` per expert): the
   same reduction kernel with the roles swapped. One launch.

Four launches per projection per backward pass instead of about 1,300.

**Autotuning.** `BM`, `BN`, `BK`, warps and stages are autotuned once per
shape class; the keys are `(in, out, dtype of W, is_transposed)`, not the
number of rows, so a run autotunes a handful of times, not every step.
Candidate tiles for this chip start from `BM ∈ {64, 128}`,
`BN ∈ {64, 128}`, `BK ∈ {32, 64}`, 4–8 warps, 3–4 stages; the measurement
decides.

**Numerics.** FP32 accumulation; `acc_low` cast to BF16 before the epilogue
`dot`, as the current path casts `low`. The tests compare against the
current routed path (which is PEFT's result to BF16 rounding) with the
tolerances `tests/test_moe.py` uses.

## Where it plugs in

`meepsquared._moe._routed_forward.project` calls the kernel's autograd Function
when the stack is on a CUDA device whose PyTorch grouped matmul is the loop
(compute capability below 9.0, or 12.0 and above), Triton is importable, and
the dtypes are BF16 (adapters) with BF16 or FP8 (stack). On compute
capability 9.x and 10.x PyTorch has grouped kernels of its own and the
current path stays; that is a stated rule with its reason, and a benchmark
on such a GPU can change it. The kernel never silently replaces a failure:
a shape or dtype it does not take is an error that names it.

`meepsquared._fp8.grouped_product` (the dequantize-per-product path) remains for
GPUs where the kernel does not run; with the kernel, FP8 stacks never leave
FP8.

## Tests

- Forward equals the current routed path on the tiny MoE fixture
  (`tests/test_moe.py`), for BF16 and FP8 stacks, both layouts, with and
  without biases, with adapters disabled (base product only).
- Gradients of `x`, `A`, `B` equal the current path's.
- Edge cases: experts with no rows; experts with fewer rows than `BM`; row
  counts and sizes that are not multiples of the tiles; a single expert.
- A CPU reference implementation of the tile mapping for the preparation
  step.

## Benchmark

`benchmarks/kernels/grouped_lora.py`: the three products at Qwen3.6-35B-A3B
shapes (262,144 rows, 256 experts, 2048 ↔ 1024 and 512 ↔ 2048), BF16 and
FP8 weights, against the current path (torch's loop plus the two LoRA
products) and PEFT's build: milliseconds per product, forward and backward.
Then the MoE benchmark (`benchmarks/moe.ipynb`) as it is, with the kernel
on: that is the number that goes into the docs.

## What to expect, and when to stop

Expectations, to be replaced by measurements:

- Launches per layer and pass: from about 1,500 to about 10.
- The base product: at or below the 15.8 ms of the loop in BF16 (cuBLAS's
  per-expert `mm` is efficient; the gain is the launches, the fused LoRA
  and the single read of `x`); with FP8 weights, toward half of that.
- A step on Qwen3.6-35B-A3B: from about 13 s toward 9–10 s, if the expert
  products are the 40–50% of GPU time the profile suggests and the CPU is
  no longer the bound.

The work is cut so that each cut is measured before the next:

1. Forward kernel, BF16, base product only, with the tile mapping and the
   tests. Measure against the loop. If it is not faster, stop here and
   say so.
2. The fused LoRA (`acc_low` and the epilogue). Measure.
3. FP8 weights in the main loop. Measure.
4. The backward kernels. Measure the full step with the MoE benchmark.
5. Only then: the docs and the FP8 activations question.

## Risks

- Triton on compute capability 12.1: the toolchain in the `meepsquared` venv is
  Triton 3.7 with PyTorch 2.11 (NVIDIA's build); Meep²'s own Triton kernels
  (`_geglu_tanh`, `_fp8`) run there, so the compiler does. Performance
  tuning for this chip is the unknown.
- Routing skew: a few experts receive many rows and most few. The tile
  mapping handles it (tiles follow rows), but a `BM` of 128 wastes work on
  experts with 20 rows; the autotuner may prefer smaller tiles than a dense
  GEMM would.
- The sort, histogram and gather around the products
  (`_routed_forward`) are unchanged and become a larger share of the step
  once the products are fast; they are PyTorch operations, a few launches,
  and can be fused later if the profile says so.
