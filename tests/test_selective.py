# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Selective recomputation: the policy keeps the products, the training is the same, the layouts are chosen."""

import copy

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from meepsquared import MeepLanguageModel, _checkpointing, _selective

GIB = 2**30
IDS = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 3], [4, 5, 3, 6, 7, 3, 8, 9]])


def with_adapters(model, **options):
    """The PEFT model with non-zero B matrices, so the adapters change the output and get gradients that matter."""
    peft_model = MeepLanguageModel.get_peft_model(model, r=4, **options)
    torch.manual_seed(1)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.1)
    return peft_model


def loss_and_gradients(model):
    model.zero_grad()
    loss = model(input_ids=IDS, labels=IDS).loss
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def assert_same_training(reference, other):
    loss_ref, grads_ref = loss_and_gradients(reference)
    loss, grads = loss_and_gradients(other)
    torch.testing.assert_close(loss, loss_ref, atol=1e-6, rtol=1e-6)
    assert grads.keys() == grads_ref.keys()
    for name in grads_ref:
        torch.testing.assert_close(grads[name], grads_ref[name], atol=1e-6, rtol=1e-6, msg=name)


def test_selective_lays_out_every_layer(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing="selective")
    assert _checkpointing.kinds(model) == ["selective", "selective"]
    assert model.base_model.model.is_gradient_checkpointing
    MeepLanguageModel.for_inference(model)
    assert _checkpointing.kinds(model) == ["none", "none"]
    MeepLanguageModel.for_training(model)
    assert _checkpointing.kinds(model) == ["selective", "selective"]


def test_a_layout_mixes_the_three_ways(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    _checkpointing.set_checkpointed(model, _checkpointing.Recompute(whole=1, selective=1))
    assert _checkpointing.kinds(model) == ["whole", "selective"]
    _checkpointing.set_checkpointed(model, _checkpointing.Recompute(whole=0, selective=1))
    assert _checkpointing.kinds(model) == ["selective", "none"]
    _checkpointing.set_checkpointed(model, 1)
    assert _checkpointing.kinds(model) == ["whole", "none"]


@pytest.mark.parametrize("setting, layout", [
    ("selective", _checkpointing.Recompute(0, 2)), (True, _checkpointing.Recompute(2, 0)),
    (1, _checkpointing.Recompute(1, 0)), (False, _checkpointing.Recompute(0, 0)),
    (_checkpointing.Recompute(1, 1), _checkpointing.Recompute(1, 1))])
def test_recompute_of_a_setting(setting, layout):
    assert _checkpointing.recompute(setting, 2) == layout
    assert _checkpointing.count(setting, 2) == layout.checkpointed


def test_selective_trains_like_no_checkpointing(tiny_model):
    reference = with_adapters(copy.deepcopy(tiny_model), use_gradient_checkpointing=False)
    selective = with_adapters(copy.deepcopy(tiny_model), use_gradient_checkpointing="selective")
    assert_same_training(reference, selective)


def test_selective_trains_like_no_checkpointing_with_the_fused_kernels(tiny_model):
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    reference = with_adapters(copy.deepcopy(tiny_model), use_gradient_checkpointing=False, target_modules=targets)
    selective = with_adapters(copy.deepcopy(tiny_model), use_gradient_checkpointing="selective",
                              target_modules=targets, fused_lora=True)
    assert _checkpointing.kinds(selective) == ["selective", "selective"]
    assert_same_training(reference, selective)


def test_a_mixed_layout_trains_like_no_checkpointing(tiny_model):
    reference = with_adapters(copy.deepcopy(tiny_model), use_gradient_checkpointing=False)
    mixed = with_adapters(copy.deepcopy(tiny_model), use_gradient_checkpointing=False)
    _checkpointing.set_checkpointed(mixed, _checkpointing.Recompute(whole=1, selective=1))
    assert_same_training(reference, mixed)


def test_selective_trains_the_experts_like_no_checkpointing(tiny_moe):
    targets = {"target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]}
    reference = with_adapters(copy.deepcopy(tiny_moe), use_gradient_checkpointing=False, **targets)
    selective = with_adapters(copy.deepcopy(tiny_moe), use_gradient_checkpointing="selective", **targets)
    assert_same_training(reference, selective)


def observe(model):
    """Record (is_recompute, op, decision) for every operation the policy of the layers sees."""
    seen = []
    for layer in _checkpointing.layers(model):
        policy = layer._gradient_checkpointing_func.policy
        policy.observe = lambda ctx, op, decision: seen.append((ctx.is_recompute, op, decision))
    return seen


def test_the_policy_keeps_the_products_and_recomputes_the_rest(tiny_model):
    model = with_adapters(tiny_model, use_gradient_checkpointing="selective")
    seen = observe(model)
    loss_and_gradients(model)
    aten = torch.ops.aten
    forward = [(op, decision) for recompute, op, decision in seen if not recompute]
    recomputed = [(op, decision) for recompute, op, decision in seen if recompute]
    # The recompute runs again (in reverse layer order, and stopping once the backward has what it needs),
    # and decides every operation it meets as the forward did
    assert recomputed and set(recomputed) <= set(forward)
    kept = {op for op, decision in forward if decision is CheckpointPolicy.MUST_SAVE}
    assert aten.mm.default in kept or aten.addmm.default in kept
    assert kept & {aten._scaled_dot_product_flash_attention_for_cpu.default, aten.bmm.default}
    assert all("mm" in str(op) or "attention" in str(op) for op in kept), kept
    # LoRA's own products (rank 4 wide) are recomputed beside the kept ones
    assert any(decision is CheckpointPolicy.PREFER_RECOMPUTE for op, decision in forward if op is aten.mm.default)
    for op in (aten.rsqrt.default, aten.silu.default, aten.mul.Tensor, aten.add.Tensor):
        decisions = {decision for seen_op, decision in forward if seen_op is op}
        assert decisions == {CheckpointPolicy.PREFER_RECOMPUTE}, op


def test_the_policy_recomputes_loras_own_products():
    policy = _selective.Policy(rank=4, saved=_selective.attention_ops())
    aten = torch.ops.aten
    x, a, b, w = torch.zeros(8, 32), torch.zeros(32, 4), torch.zeros(4, 64), torch.zeros(32, 64)
    assert policy.decide(aten.mm.default, (x, w)) is CheckpointPolicy.MUST_SAVE
    assert policy.decide(aten.mm.default, (x, a)) is CheckpointPolicy.PREFER_RECOMPUTE  # x @ A: rank wide
    assert policy.decide(aten.addmm.default, (torch.zeros(8, 64), x @ a, b)) is CheckpointPolicy.PREFER_RECOMPUTE
    assert policy.decide(aten.bmm.default, (torch.zeros(2, 8, 16), torch.zeros(2, 16, 8))) is CheckpointPolicy.MUST_SAVE
    assert policy.decide(aten._scaled_dot_product_flash_attention_for_cpu.default, ()) is CheckpointPolicy.MUST_SAVE
    assert policy.decide(aten.silu.default, (x,)) is CheckpointPolicy.PREFER_RECOMPUTE


def test_the_rank_is_read_from_the_adapters(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    assert _selective.lora_rank(model) == 4
    assert _selective.policy_for(model).rank == 4


def test_selective_is_refused_with_compiled_layers(tiny_model):
    with pytest.raises(ValueError, match="compiled"):
        MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing="selective",
                                           compile_layers=True)


def test_selective_is_refused_on_linear_attention_with_its_kernel(tiny_model, monkeypatch):
    from meepsquared import _linear_attention

    monkeypatch.setattr(_linear_attention, "available", lambda: True)
    tiny_model.config.layer_types = ["linear_attention", "full_attention"]
    with pytest.raises(ValueError, match="linear-attention"):
        MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing="selective")


def test_selective_is_refused_when_flash_attention_has_no_op(tiny_model, monkeypatch):
    monkeypatch.setattr(_selective, "flash_attention_ops", lambda implementation: None)
    tiny_model.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="torch.ops.flash_attn"):
        MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing="selective")


def test_sdpa_needs_no_flash_attention_op():
    assert _selective.flash_attention_ops("sdpa") == []


def test_time_layer_times_each_setting_on_the_cpu(tiny_model):
    model = with_adapters(tiny_model, use_gradient_checkpointing="selective")
    seconds = _checkpointing.time_layer(model, _checkpointing.Shape(2, 8), ("none", "selective", "whole"), repeats=2)
    assert set(seconds) == {"none", "selective", "whole"} and all(s > 0 for s in seconds.values())
    assert _checkpointing.kinds(model) == ["selective", "selective"]  # left as it was
    assert all(p.grad is None for p in model.parameters())


# A 32-layer model on a GPU with 24 GiB, 20 free: 16.4 GiB usable beyond the margin
FREE, TOTAL, LAYERS = 20 * GIB, 24 * GIB, 32
TIMES = {"none": 0.030, "selective": 0.033, "whole": 0.040}


def choose(per_layer, per_selective=None, times=TIMES, free=FREE):
    return _checkpointing.choose(free, TOTAL, per_layer, per_selective, times, LAYERS)


def test_nothing_is_recomputed_when_everything_fits():
    assert choose(GIB // 2, GIB // 4) == _checkpointing.Recompute(0, 0)


def test_selective_on_as_few_layers_as_needed():
    # 32 layers of 1 GiB hold 32; selective halves a layer: s selective layers hold 32 - s / 2 <= 16.4 at s = 32
    assert choose(GIB, GIB // 2) == _checkpointing.Recompute(0, 32)
    # a quarter: 32 - 0.75 s <= 16.4 at s = 21
    assert choose(GIB, GIB // 4) == _checkpointing.Recompute(0, 21)


def test_selective_on_every_layer_plus_whole_on_as_few_as_fit():
    # 32 selective layers hold 24, too much; w whole layers leave (32 - w) * 0.75 <= 16.4 at w = 11
    # The rule without selective recomputation: 16 layers keeping everything fit, 16 recompute whole
    assert _checkpointing.layers_to_checkpoint(FREE, TOTAL, GIB, LAYERS) == 16
    # Without times, the order of preference holds: every layer selective plus the fewest whole
    assert choose(GIB, 3 * GIB // 4, times=None) == _checkpointing.Recompute(11, 21)
    # 11 whole + 21 selective at 31 ms cost 11 x 40 + 21 x 31 = 1091 ms; 16 whole + 16 keeping, 16 x 40 + 16 x 30 = 1120
    assert choose(GIB, 3 * GIB // 4, times={"none": 0.030, "selective": 0.031, "whole": 0.040}) == _checkpointing.Recompute(11, 21)


def test_the_time_measured_decides_between_the_two_mixed_layouts():
    # Selective at 33 ms: 11 x 40 + 21 x 33 = 1133 ms, more than the 1120 of 16 whole layers beside 16 keeping
    assert choose(GIB, 3 * GIB // 4) == _checkpointing.Recompute(16, 0)
    # Selective as slow as whole: the same
    assert choose(GIB, 3 * GIB // 4, times={"none": 0.030, "selective": 0.040, "whole": 0.040}) == _checkpointing.Recompute(16, 0)


def test_selective_refused_leaves_the_old_rule():
    assert choose(GIB, None) == _checkpointing.Recompute(16, 0)
    assert choose(GIB // 2, None) == _checkpointing.Recompute(0, 0)


def test_nothing_fits_recomputes_every_layer_whole():
    assert choose(GIB, GIB // 2, free=2 * GIB) == _checkpointing.Recompute(32, 0)


def test_fewest_whole_beside_the_users_selective_layers():
    # 10 selective layers (0.5 GiB each) and 22 keeping everything hold 27; w whole leave 27 - w <= 16.4 at w = 11
    assert _checkpointing.fewest_whole(FREE, TOTAL, GIB, GIB // 2, LAYERS, selective=10) == 11
    assert _checkpointing.fewest_whole(FREE, TOTAL, GIB // 4, GIB // 8, LAYERS, selective=10) == 0


LENGTHS = [5, 300, 1024, 40, 900]


def test_largest_batch_is_the_planned_batch_of_the_most_tokens():
    # Padding-free: the examples of each batch end to end; [2, 4] holds 1,924 tokens, [1, 3] 340
    shape = _checkpointing.largest_batch([[1, 3], [2, 4], [0]], LENGTHS, padding_free=True)
    assert (shape.rows, shape.length, shape.lengths, shape.padding) == (1, 1924, (1024, 900), 0)
    assert shape.tokens == 1924 and shape.text() == "1,924 tokens (2 examples, end to end)"
    inputs = shape.inputs(torch.device("cpu"))
    assert inputs["input_ids"].shape == (1, 1924)
    assert inputs["position_ids"][0, :3].tolist() == [0, 1, 2] and inputs["position_ids"][0, 1024].item() == 0


def test_largest_batch_padded_is_a_row_per_example_of_its_longest():
    # Padded: [0, 1, 3] is 3 rows of 300 (900 tokens), [2] one row of 1,024
    shape = _checkpointing.largest_batch([[0, 1, 3], [2]], LENGTHS, padding_free=False)
    assert (shape.rows, shape.length, shape.lengths) == (1, 1024, None) and shape.text() == "1 x 1024 tokens"
    rounded = _checkpointing.largest_batch([[0, 1, 3], [2]], LENGTHS, padding_free=False, pad_to_multiple_of=128)
    assert (rounded.rows, rounded.length) == (3, 384)  # 300 rounded up to 384: 1,152 tokens, more than 1,024
    assert "position_ids" not in rounded.inputs(torch.device("cpu"))


def test_the_median_and_mean_batch_of_the_planned_batches():
    # Padding-free: [0] holds 5 tokens, [1, 3] 340, [2, 4] 1,924
    shapes = _checkpointing.batch_shapes([[1, 3], [2, 4], [0], []], LENGTHS, padding_free=True)
    assert [s.tokens for s in shapes] == [340, 1924, 5]
    assert _checkpointing.median_batch(shapes).tokens == 340
    assert _checkpointing.mean_tokens(shapes) == pytest.approx((340 + 1924 + 5) / 3)
    # An even count: the lower of the two middle batches
    assert _checkpointing.median_batch(shapes[:2]).tokens == 340


def test_largest_batch_padding_free_rounds_the_whole_batch():
    shape = _checkpointing.largest_batch([[0, 1]], [100, 50], padding_free=True, pad_to_multiple_of=64)
    assert (shape.length, shape.lengths, shape.padding) == (192, (100, 50), 42)
    positions = shape.inputs(torch.device("cpu"))["position_ids"][0]
    assert positions[150].item() == 0 and positions[191].item() == 41


def test_largest_batch_of_no_examples_is_refused():
    with pytest.raises(ValueError, match="no examples"):
        _checkpointing.largest_batch([[]], [], padding_free=True)


def test_the_longest_examples_bound_any_order():
    shape = _checkpointing.longest_examples([5, 300, 1200, 40, 900], micro=2, max_length=1024, padding_free=True)
    assert (shape.rows, shape.length, shape.lengths) == (1, 1924, (1024, 900))
    padded = _checkpointing.longest_examples([5, 300, 1200, 40, 900], micro=4, max_length=1024, padding_free=False)
    assert (padded.rows, padded.length) == (4, 1024)
    # Fewer examples than the micro-batch: all of them
    assert _checkpointing.longest_examples([7, 9], micro=8, max_length=1024, padding_free=True).lengths == (9, 7)


def in_place_during_forward(model):
    """_selective.in_place() as each call of the first layer's MLP saw it, in the forward and in any recompute."""
    seen = []
    mlp = _checkpointing.layers(model)[0].mlp
    # Before the MLP runs: a recompute stops once the backward has its tensors, which can be inside the MLP
    mlp.register_forward_pre_hook(lambda module, args: seen.append(_selective.in_place()))
    loss_and_gradients(model)
    return seen


@pytest.mark.parametrize("layout, expected", [
    (_checkpointing.Recompute(0, 2), [False, False]),  # the forward and the recompute of a selective layer
    (_checkpointing.Recompute(2, 0), [True, True]),    # a layer recomputed whole adds in place, twice
    (_checkpointing.Recompute(0, 0), [True]),          # a layer keeping everything, once
    (_checkpointing.Recompute(1, 1), [True, True]),    # the first layer whole, the second selective
])
def test_only_a_selective_layer_adds_out_of_place(tiny_model, layout, expected):
    model = with_adapters(tiny_model, use_gradient_checkpointing=False)
    _checkpointing.set_checkpointed(model, layout)
    assert in_place_during_forward(model) == expected
    assert _selective.in_place()


def test_matmul_lora_adds_in_place_outside_a_selective_layer(monkeypatch):
    from meepsquared._vendor.axolotl import lora

    torch.manual_seed(0)
    X, W, A, B = torch.randn(6, 8), torch.randn(16, 8), torch.randn(4, 8), torch.randn(16, 4)
    expected = X @ W.t() + 0.5 * (X @ A.t()) @ B.t()
    bases = []

    def base_mm(X, W, W_quant, transpose_weight, out=None):
        bases.append(X @ W.t())
        return bases[-1]

    monkeypatch.setattr(lora, "_base_mm", base_mm)
    out = lora.matmul_lora(X, W, None, None, A, B, 0.5)
    torch.testing.assert_close(out, expected)
    assert out.data_ptr() == bases[-1].data_ptr()  # written into the base product's output
    out = _selective._flagged(lambda: lora.matmul_lora(X, W, None, None, A, B, 0.5))()
    torch.testing.assert_close(out, expected)
    assert out.data_ptr() != bases[-1].data_ptr()  # the base product's output left as it was
    torch.testing.assert_close(bases[-1], X @ W.t())


def test_the_flag_is_put_back_after_an_error():
    def fails():
        raise RuntimeError("inside")

    with pytest.raises(RuntimeError, match="inside"):
        _selective._flagged(fails)()
    assert _selective.in_place()
