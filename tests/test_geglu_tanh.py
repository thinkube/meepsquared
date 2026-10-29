# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from meepsquared import _geglu_tanh


def test_reference_is_the_tanh_gelu_times_up():
    gate, up = torch.randn(4, 8, 16), torch.randn(4, 8, 16)
    expected = torch.nn.functional.gelu(gate, approximate="tanh") * up
    torch.testing.assert_close(_geglu_tanh.reference(gate, up), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")
def test_kernels_match_the_reference_and_its_gradients():
    gate = torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    h = _geglu_tanh.reference(gate, up)
    grad = torch.randn_like(h)
    h.backward(grad)

    fused = _geglu_tanh.forward(gate.detach(), up.detach())
    torch.testing.assert_close(fused, h.detach(), atol=2e-2, rtol=2e-2)
    h2, grad_gate, grad_up = _geglu_tanh.backward(grad.clone(), gate.detach().clone(), up.detach().clone())
    torch.testing.assert_close(h2, h.detach(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grad_gate, gate.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(grad_up, up.grad, atol=2e-2, rtol=2e-2)
