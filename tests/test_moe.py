# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA on expert stacks, computed on the routed tokens, trains as PEFT's does."""

import copy

import pytest
import torch

from meepsquared import _moe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped matmuls run on a GPU")


@pytest.fixture
def moe():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(0)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=2, layer_types=["full_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, max_position_embeddings=64,
        initializer_range=0.2,
    )
    config._experts_implementation = "grouped_mm"
    model = Qwen3_5MoeForCausalLM(config).to("cuda", torch.bfloat16)
    peft_model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.2)
    return peft_model


def run(model, ids):
    model.zero_grad()
    logits = model(input_ids=ids).logits.float()
    logits.pow(2).mean().backward()
    grads = {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def relative(x, reference):
    return ((x - reference).abs().max() / reference.abs().max().clamp(min=1e-6)).item()


def test_routed_lora_matches_peft(moe):
    routed = copy.deepcopy(moe)
    assert _moe.apply(routed) == 2
    ids = torch.randint(0, 64, (2, 24), device="cuda")
    out_peft, grads_peft = run(moe, ids)
    out_routed, grads_routed = run(routed, ids)
    assert relative(out_routed, out_peft) < 0.02
    # The adapters change the output far more than that tolerance: the comparison can tell
    with torch.no_grad(), moe.disable_adapter():
        assert relative(moe(input_ids=ids).logits.float(), out_peft) > 0.1
    expert_grads = [n for n in grads_peft if "experts" in n and "lora_" in n]
    assert expert_grads and grads_peft.keys() == grads_routed.keys()
    for name in grads_peft:
        assert relative(grads_routed[name], grads_peft[name]) < 0.05, name


def test_disabled_adapters_give_the_base_model(moe):
    routed = copy.deepcopy(moe)
    _moe.apply(routed)
    ids = torch.randint(0, 64, (1, 16), device="cuda")
    with torch.no_grad(), routed.disable_adapter(), moe.disable_adapter():
        assert relative(routed(input_ids=ids).logits.float(), moe(input_ids=ids).logits.float()) < 1e-3


def stacks(model):
    """The experts modules under PEFT's wrappers, in order."""
    return [_moe._wrapper_chain(m)[1] for m in model.modules() if _moe.routed(m)]


def with_fp8_stacks(routed):
    """A copy with its expert stacks held as FP8, and a BF16 twin holding the same weights dequantized.

    The twin tells the implementation's error apart from the quantization's:
    the FP8 path computes exactly what the twin computes.
    """
    from meepsquared import _fp8

    fp8, twin = copy.deepcopy(routed), copy.deepcopy(routed)
    for experts in stacks(fp8):
        assert _fp8.convert_stacks(experts) == 2
    for held, plain in zip(stacks(fp8), stacks(twin)):
        for name in ("gate_up_proj", "down_proj"):
            scale = getattr(held, name + _fp8.STACK_SCALE)
            getattr(plain, name).data = _fp8.dequantize_stack(getattr(held, name), scale, torch.bfloat16)
    return fp8, twin


def test_fp8_expert_stacks_compute_what_their_dequantized_weights_compute(moe):
    routed = copy.deepcopy(moe)
    _moe.apply(routed)
    fp8, twin = with_fp8_stacks(routed)
    experts = stacks(fp8)[0]
    assert experts.gate_up_proj.dtype == torch.float8_e4m3fn and experts.gate_up_proj_scale_inv.shape == (8,)
    assert experts.down_proj.dtype == torch.float8_e4m3fn and experts._meepsquared_fp8 is True
    ids = torch.randint(0, 64, (2, 24), device="cuda")
    out_twin, grads_twin = run(twin, ids)
    out_fp8, grads_fp8 = run(fp8, ids)
    assert relative(out_fp8, out_twin) < 2e-2
    assert grads_twin.keys() == grads_fp8.keys() and any("experts" in n for n in grads_fp8)
    for name in grads_twin:
        assert relative(grads_fp8[name], grads_twin[name]) < 3e-2, name
    # the quantization itself is small next to what the adapters do
    out_bf16, _ = run(routed, ids)
    with torch.no_grad(), routed.disable_adapter():
        without = routed(input_ids=ids).logits.float()
    assert relative(out_fp8, out_bf16) < relative(without, out_bf16)


def test_fp8_stacks_with_the_adapters_disabled_give_the_base_model(moe):
    routed = copy.deepcopy(moe)
    _moe.apply(routed)
    fp8, twin = with_fp8_stacks(routed)
    ids = torch.randint(0, 64, (1, 16), device="cuda")
    with torch.no_grad(), twin.disable_adapter(), fp8.disable_adapter():
        assert relative(fp8(input_ids=ids).logits.float(), twin(input_ids=ids).logits.float()) < 2e-2
    # PEFT's own forward cannot run on FP8 stacks: a call it would take is refused
    wrapper = next(m for m in fp8.modules() if _moe.routed(m))
    with pytest.raises(NotImplementedError, match="FP8"):
        wrapper(torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16),
                torch.zeros(1, 2, dtype=torch.long, device="cuda"), torch.ones(1, 2, device="cuda", dtype=torch.bfloat16), "extra")


def test_fp8_apply_holds_the_stacks_and_the_frozen_linears_of_a_mixture_of_experts(moe):
    from meepsquared import _fp8

    fp8 = copy.deepcopy(moe)
    _moe.apply(fp8)
    converted = _fp8.apply(fp8)
    layer = fp8.base_model.model.model.layers[0]
    assert stacks(fp8)[0].gate_up_proj.dtype == torch.float8_e4m3fn
    assert isinstance(layer.self_attn.q_proj.base_layer, _fp8.Fp8Linear) and isinstance(layer.self_attn.o_proj, _fp8.Fp8Linear)
    assert all(p.dtype == torch.bfloat16 for p in layer.mlp.gate.parameters())  # the router stays in BF16
    assert type(layer.mlp.shared_expert_gate) is torch.nn.Linear  # one output: not a size the FP8 product takes
    assert converted >= 2 * 2 + 2 * 4
    ids = torch.randint(0, 64, (1, 16), device="cuda")
    out, grads = run(fp8, ids)
    assert torch.isfinite(out).all() and grads
