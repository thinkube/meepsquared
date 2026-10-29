# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The grouped product kernel computes what one product per expert computes."""

import pytest
import torch

from meepsquared import _grouped

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on a GPU")


def offsets_of(rows_per_expert, device="cpu"):
    return torch.cumsum(torch.tensor(rows_per_expert, dtype=torch.int32, device=device), 0, dtype=torch.int32)


def test_tile_map_cuts_each_experts_rows_into_tiles_and_skips_empty_experts():
    offsets = offsets_of([5, 0, 130, 64])  # 1 + 0 + 3 + 1 tiles of 64
    tile_expert, tile_row, tiles = _grouped.tile_map(offsets, rows=199, block_m=64)
    assert tiles.item() == 5
    assert tile_expert[:5].tolist() == [0, 2, 2, 2, 3]
    assert tile_row[:5].tolist() == [0, 5, 69, 133, 135]
    assert tile_expert.numel() == _grouped.max_tiles(199, 4, 64) >= 5
    assert tile_expert.max() <= 3  # the spare entries name a real expert


def test_tile_map_with_every_expert_empty_but_one():
    offsets = offsets_of([0, 0, 7])
    tile_expert, tile_row, tiles = _grouped.tile_map(offsets, rows=7, block_m=64)
    assert tiles.item() == 1 and tile_expert[0].item() == 2 and tile_row[0].item() == 0


@cuda
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("rows_per_expert,K,N", [
    ([37, 0, 200, 64, 1], 96, 80),      # uneven groups, an empty one, sizes that are no multiples of the tiles
    ([300] * 8, 256, 128),              # even groups
    ([3], 2048, 1024),                  # one expert, Qwen3.6's sizes
])
def test_the_kernel_matches_one_product_per_expert(is_transposed, rows_per_expert, K, N):
    torch.manual_seed(0)
    device = "cuda"
    rows = sum(rows_per_expert)
    experts = len(rows_per_expert)
    x = torch.randn(rows, K, device=device, dtype=torch.bfloat16)
    shape = (experts, K, N) if is_transposed else (experts, N, K)
    weight = (torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.05)
    offsets = offsets_of(rows_per_expert, device)
    out = _grouped.grouped_linear(x, weight, offsets, is_transposed)
    expected = _grouped.reference(x, weight, offsets, is_transposed)
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)


@cuda
@pytest.mark.parametrize("is_transposed", [False, True])
def test_the_kernel_takes_an_fp8_stack_as_weights_to_convert(is_transposed):
    torch.manual_seed(0)
    x = torch.randn(70, 64, device="cuda", dtype=torch.bfloat16)
    shape = (3, 64, 48) if is_transposed else (3, 48, 64)
    weight = (torch.randn(shape, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    offsets = offsets_of([10, 40, 20], "cuda")
    out = _grouped.grouped_linear(x, weight, offsets, is_transposed, products_fp8=False)
    expected = _grouped.reference(x, weight, offsets, is_transposed)
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)


@cuda
@pytest.mark.parametrize("is_transposed", [False, True])
def test_the_fp8_product_quantizes_the_activations_per_row_and_stays_close(is_transposed):
    torch.manual_seed(0)
    x = torch.randn(300, 128, device="cuda", dtype=torch.bfloat16)
    x[::7] *= 20  # rows of different sizes: the scale is per row
    shape = (4, 128, 96) if is_transposed else (4, 96, 128)
    weight = (torch.randn(shape, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    offsets = offsets_of([100, 0, 150, 50], "cuda")
    expected = _grouped.reference(x, weight, offsets, is_transposed).float()
    out = _grouped.grouped_linear(x, weight, offsets, is_transposed, products_fp8=True).float()
    # e4m3 keeps 3 bits of mantissa: the activations quantized per row give a relative error of about 4%
    assert (out - expected).abs().max() <= 0.1 * expected.abs().max()
    assert (out - expected).norm() / expected.norm() < 0.06


@cuda
def test_a_mismatched_stack_is_refused():
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="in=32"):
        _grouped.grouped_linear(x, weight, offsets_of([4, 4], "cuda"), is_transposed=False)


def adapters(experts, K, N, rank, is_transposed, device):
    """PEFT's factors in the stack's layout: lhs (out or in, rank), rhs (rank, in or out)."""
    if is_transposed:
        lhs = torch.randn(experts, K, rank, device=device, dtype=torch.bfloat16) * 0.1
        rhs = torch.randn(experts, rank, N, device=device, dtype=torch.bfloat16) * 0.1
    else:
        lhs = torch.randn(experts, N, rank, device=device, dtype=torch.bfloat16) * 0.1
        rhs = torch.randn(experts, rank, K, device=device, dtype=torch.bfloat16) * 0.1
    return lhs.requires_grad_(), rhs.requires_grad_()


@cuda
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("fp8", [False, "weights", "products"])
@pytest.mark.parametrize("rank", [4, 16])
def test_the_product_with_adapters_and_its_gradients_match_pytorch(is_transposed, fp8, rank):
    torch.manual_seed(0)
    device = "cuda"
    rows_per_expert = [37, 0, 200, 64, 1]
    K, N = 96, 80
    experts = len(rows_per_expert)
    offsets = offsets_of(rows_per_expert, device)
    x = (torch.randn(sum(rows_per_expert), K, device=device, dtype=torch.bfloat16)).requires_grad_()
    shape = (experts, K, N) if is_transposed else (experts, N, K)
    weight = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.05
    scale = None
    if fp8:
        from meepsquared import _fp8

        weight, scale = _fp8.quantize_stack(weight)
    lhs, rhs = adapters(experts, K, N, rank, is_transposed, device)
    bias = torch.randn(experts, N, device=device, dtype=torch.bfloat16)
    scaling = 2.0

    out = _grouped.lora_linear(x, weight, scale, offsets, is_transposed, lhs, rhs, scaling, bias, products_fp8=fp8 == "products")
    grad = torch.randn_like(out)
    out.backward(grad)
    got = (out.detach(), x.grad.clone(), lhs.grad.clone(), rhs.grad.clone())
    x.grad = lhs.grad = rhs.grad = None

    expected_out = _grouped.reference_lora(x, weight, scale, offsets, is_transposed, lhs, rhs, scaling, bias)
    expected_out.backward(grad)
    expected = (expected_out.detach(), x.grad, lhs.grad, rhs.grad)
    for name, a, b in zip(("out", "grad_x", "grad_lhs", "grad_rhs"), got, expected):
        assert a.shape == b.shape, name
        if fp8 == "products" and name in ("out", "grad_x"):
            # the activations, and the gradient in the backward pass, are quantized per row to e4m3: about 4% error
            assert (a.float() - b.float()).norm() <= 0.06 * b.float().norm(), name
            continue
        # the low-rank products are BF16 in the kernel as in meepsquared's PyTorch path, and the reference's are float32:
        # a sum over an expert's rows differs by its rounding, relative to the tensor's size
        torch.testing.assert_close(a.float(), b.float(), atol=1e-2 * b.float().abs().max().item(), rtol=0, msg=name)


@cuda
def test_the_product_without_adapters_gives_the_base_and_its_input_gradient():
    torch.manual_seed(0)
    device = "cuda"
    offsets = offsets_of([10, 30], device)
    x = torch.randn(40, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(2, 48, 64, device=device, dtype=torch.bfloat16) * 0.05
    out = _grouped.lora_linear(x, weight, None, offsets, False)
    out.sum().backward()
    torch.testing.assert_close(out.float(), _grouped.reference(x.detach(), weight, offsets, False).float(), atol=2e-2, rtol=2e-2)
    expected = (weight.sum(dim=1))[torch.repeat_interleave(torch.arange(2, device=device), torch.tensor([10, 30], device=device))]
    torch.testing.assert_close(x.grad.float(), expected.float(), atol=5e-2, rtol=3e-2)


def test_fp8_products_are_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("MEEPSQUARED_FP8_PRODUCTS", raising=False)
    assert _grouped.fp8_products(torch.device("cuda", 0)) is False
    monkeypatch.setenv("MEEPSQUARED_FP8_PRODUCTS", "1")
    assert _grouped.fp8_products(torch.device("cuda", 0)) is True


def test_wanted_follows_the_override_and_the_device(monkeypatch):
    monkeypatch.setenv("MEEPSQUARED_GROUPED_KERNEL", "0")
    assert _grouped.wanted(torch.device("cuda", 0)) is False
    monkeypatch.setenv("MEEPSQUARED_GROUPED_KERNEL", "1")
    assert _grouped.wanted(torch.device("cpu")) is True
    monkeypatch.delenv("MEEPSQUARED_GROUPED_KERNEL")
    assert _grouped.wanted(torch.device("cpu")) is False
