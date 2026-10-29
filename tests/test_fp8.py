# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from torch import nn

from meepsquared import MeepLanguageModel, _fp8

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 products run on a GPU")


def test_quantize_round_trips_within_e4m3_precision():
    x = torch.randn(64, 32) * 3
    x8, scale = _fp8.quantize(x)
    assert x8.dtype == torch.float8_e4m3fn and scale.dim() == 0
    torch.testing.assert_close(x8.float() * scale, x, atol=0.1, rtol=0.07)


def test_fp8_linear_computes_the_linear_layer_on_the_cpu_reference_path():
    linear = nn.Linear(32, 48, bias=True).to(torch.bfloat16)
    fp8 = _fp8.Fp8Linear(linear)
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(fp8(x).float(), linear(x).float(), atol=0.3, rtol=0.1)
    torch.testing.assert_close(fp8.dequantized().float(), linear.weight.float(), atol=0.1, rtol=0.07)
    assert fp8.weight.numel() == linear.weight.numel() and fp8.weight.element_size() == 1
    # a copy computes with its own weight: the forward is the class's, not set on the instance
    twin = copy.deepcopy(fp8)
    assert "forward" not in twin.__dict__ and twin.forward.__self__ is twin


def test_convertible_leaves_adapters_the_output_layer_and_trainable_layers():
    frozen = nn.Linear(32, 16)
    frozen.weight.requires_grad_(False)
    assert _fp8.convertible(frozen, "self_attn.q_proj", set())
    assert not _fp8.convertible(frozen, "self_attn.q_proj.lora_A.default", set())
    assert not _fp8.convertible(frozen, "lm_head", set())
    assert not _fp8.convertible(frozen, "mlp.down_proj", {"mlp.down_proj"})


def test_convertible_leaves_routers_and_layers_the_fp8_product_cannot_take():
    frozen = nn.Linear(32, 16)
    frozen.weight.requires_grad_(False)
    assert not _fp8.convertible(frozen, "mlp.gate", set())  # the router of a mixture of experts
    assert not _fp8.convertible(frozen, "mlp.router", set())
    assert _fp8.convertible(frozen, "mlp.gate_proj", set())
    narrow = nn.Linear(32, 1)  # a shared expert's gate: the FP8 product takes sizes that are multiples of 16
    narrow.weight.requires_grad_(False)
    assert not _fp8.convertible(narrow, "mlp.shared_expert_gate", set())
    assert not _fp8.convertible(nn.Linear(8, 8), "mlp.up_proj", set())


def test_fp8_is_refused_without_a_capable_gpu(tiny_model, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        MeepLanguageModel.from_pretrained("no/such-model", load_in_fp8=True)
    with pytest.raises(ValueError, match="choose one"):
        MeepLanguageModel.from_pretrained("no/such-model", load_in_fp8=True, load_in_4bit=True)


@cuda
def test_fp8_product_and_its_gradient_match_bf16(tiny_model):
    linear = nn.Linear(256, 512, bias=False).to("cuda", torch.bfloat16)
    fp8 = _fp8.Fp8Linear(linear).cuda()
    x = torch.randn(3, 37, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)  # 111 rows, not a multiple of 16
    y_ref = linear(x)
    x2 = x.detach().clone().requires_grad_(True)
    y = fp8(x2)
    torch.testing.assert_close(y.float(), y_ref.float(), atol=0.5, rtol=0.1)
    grad = torch.randn_like(y)
    y_ref.backward(grad)
    y.backward(grad)
    torch.testing.assert_close(x2.grad.float(), x.grad.float(), atol=0.5, rtol=0.1)


@cuda
def test_get_peft_model_holds_the_frozen_decoder_linears_as_fp8(tiny_model):
    tiny_model._meepsquared["fp8"] = True  # as from_pretrained(load_in_fp8=True) marks it
    model = MeepLanguageModel.get_peft_model(tiny_model.to("cuda", torch.bfloat16), r=4,
                                              use_gradient_checkpointing=False, target_modules=["q_proj"])
    assert model._meepsquared_fused_lora is True  # the fused kernels run the FP8 products
    layer = model.base_model.model.model.layers[0]
    assert isinstance(layer.self_attn.q_proj.base_layer, _fp8.Fp8Linear)
    assert isinstance(layer.self_attn.k_proj, _fp8.Fp8Linear)
    assert sum(isinstance(m, _fp8.Fp8Linear) for m in model.modules()) == 2 * 7


@cuda
def test_fused_projections_run_the_fp8_product(tiny_model):
    import copy

    from meepsquared import _fused

    tiny_model._meepsquared["fp8"] = True
    plain = MeepLanguageModel.get_peft_model(copy.deepcopy(tiny_model).to("cuda", torch.bfloat16), r=4,
                                              use_gradient_checkpointing=False,
                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], fused_lora=False)
    fused = copy.deepcopy(plain)
    assert _fused.apply(fused) == {"mlp": 0, "qkv": 2, "projections": 2}
    ids = torch.tensor([[3, 4, 5, 6, 7]], device="cuda")
    torch.testing.assert_close(fused(input_ids=ids).logits.float(), plain(input_ids=ids).logits.float(), atol=0.3, rtol=0.1)


@cuda
def test_fp8_defaults_keep_the_fused_kernels(tiny_model):
    tiny_model._meepsquared["fp8"] = True
    model = MeepLanguageModel.get_peft_model(tiny_model.to("cuda", torch.bfloat16), r=4, use_gradient_checkpointing=False)
    assert model._meepsquared_fused_lora is True
    ids = torch.tensor([[3, 4, 5, 6]], device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    ids = torch.tensor([[3, 4, 5, 6]], device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.grad is not None


def test_a_stack_quantizes_per_expert_and_comes_back_within_e4m3_precision():
    torch.manual_seed(0)
    stack = torch.randn(4, 6, 8, dtype=torch.bfloat16) * torch.tensor([0.01, 0.1, 1.0, 10.0]).view(4, 1, 1).to(torch.bfloat16)
    weight8, scale = _fp8.quantize_stack(stack)
    assert weight8.dtype == torch.float8_e4m3fn and weight8.shape == stack.shape and scale.shape == (4,)
    back = _fp8.dequantize_stack(weight8, scale, torch.bfloat16)
    assert back.dtype == torch.bfloat16
    for expert in range(4):  # one scale per expert: each expert's error is relative to its own amax
        amax = stack[expert].float().abs().amax()
        assert (back[expert].float() - stack[expert].float()).abs().max() <= amax * 0.07


def test_convert_stacks_holds_frozen_stacks_only():
    experts = nn.Module()
    experts.gate_up_proj = nn.Parameter(torch.randn(3, 8, 4, dtype=torch.bfloat16), requires_grad=False)
    experts.down_proj = nn.Parameter(torch.randn(3, 4, 4, dtype=torch.bfloat16), requires_grad=False)
    experts.trained = nn.Parameter(torch.randn(3, 4, 4, dtype=torch.bfloat16))
    assert _fp8.convert_stacks(experts) == 2
    assert experts.gate_up_proj.dtype == torch.float8_e4m3fn and experts.gate_up_proj_scale_inv.shape == (3,)
    assert experts.down_proj.dtype == torch.float8_e4m3fn and experts.trained.dtype == torch.bfloat16
    assert experts._meepsquared_fp8 is True
