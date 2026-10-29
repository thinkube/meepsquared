# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The gated GELU with the tanh approximation (gelu_pytorch_tanh), as Gemma's MLP computes it.

    h = gelu_tanh(gate) * up,  gelu_tanh(x) = 0.5 x (1 + tanh(k (x + 0.044715 x^3))),  k = sqrt(2 / pi)

Triton kernels in the form Axolotl's fused LoRA MLP takes (see _vendor/axolotl/lora.py):
the forward returns h; the backward writes, in place, h over grad_output, the
gradient of gate over gate and the gradient of up over up, and returns the
three. Axolotl's own GEGLU kernels compute the exact GELU (erf), which is
another function. reference() is the same computation in PyTorch, for a check.
"""

import torch

K = 0.7978845608028654  # sqrt(2 / pi)
C = 0.044715


def reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(gate.float(), approximate="tanh").to(up.dtype) * up


try:
    import triton
    import triton.language as tl
except ImportError:  # the kernels need Triton; reference() does not
    triton = None

if triton is not None:
    from meepsquared._vendor.axolotl.op_registry import register_kernel_op

    @triton.jit
    def _fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0)
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        # tanh(u) = 2 sigmoid(2u) - 1
        tanh = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        gelu_gate = (0.5 * gate * (1.0 + tanh)).to(up.dtype)
        tl.store(out_ptr + offsets, gelu_gate * up, mask=mask)

    @register_kernel_op("geglu_tanh_fwd")
    def forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(gate)
        n_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
        _fwd_kernel[grid](gate_ptr=gate, up_ptr=up, out_ptr=out, n_elements=n_elements, BLOCK_SIZE=1024)
        return out

    @forward.register_fake
    def _(gate, up):
        return torch.empty_like(gate)

    @triton.jit
    def _bwd_kernel(grad_out_ptr, gate_ptr, up_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0)
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        tanh = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        half_one_plus_tanh = 0.5 * (1.0 + tanh)
        gelu_gate = (gate * half_one_plus_tanh).to(grad_out.dtype)
        h = gelu_gate * up
        grad_up = grad_out * gelu_gate
        # d gelu_tanh / d gate
        d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * gate * gate)
        d_gelu = half_one_plus_tanh + 0.5 * gate * (1.0 - tanh * tanh) * d_inner
        grad_gate = ((grad_out * up).to(tl.float32) * d_gelu).to(grad_out.dtype)
        tl.store(grad_out_ptr + offsets, h, mask=mask)
        tl.store(gate_ptr + offsets, grad_gate, mask=mask)
        tl.store(up_ptr + offsets, grad_up, mask=mask)

    @register_kernel_op("geglu_tanh_bwd", mutates_args=("grad_output", "gate", "up"))
    def _bwd_op(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> None:
        n_elements = grad_output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
        _bwd_kernel[grid](grad_out_ptr=grad_output, gate_ptr=gate, up_ptr=up, n_elements=n_elements,
                          BLOCK_SIZE=1024)

    @_bwd_op.register_fake
    def _(grad_output, gate, up):
        return None

    def backward(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
        """In place: h over grad_output, the gate's gradient over gate, the up's over up; returns the three."""
        _bwd_op(grad_output, gate, up)
        return grad_output, gate, up
