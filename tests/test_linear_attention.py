# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A padding-free batch through a model with linear-attention layers gives
each example what it gets alone, in the forward and the backward pass.

The model's weights start at a standard deviation of 0.5: at the default 0.02
a leak across the boundary changes the activations by about as much as BF16
rounding, and the test could not tell one from the other.
"""

import importlib.util

import pytest
import torch

from meepsquared import _linear_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None
    or importlib.util.find_spec("flash_attn") is None,
    reason="flash-linear-attention and FlashAttention kernels run on a GPU",
)

A, B = 9, 7


@pytest.fixture
def hybrid():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, max_position_embeddings=64, initializer_range=0.5,
    )
    config._attn_implementation = "flash_attention_2"
    model = Qwen3_5ForCausalLM(config).to("cuda", torch.bfloat16)
    _linear_attention.patch(model)
    return model


@pytest.fixture
def examples():
    torch.manual_seed(1)
    return torch.randint(0, 64, (A,), device="cuda"), torch.randint(0, 64, (B,), device="cuda")


def packed(a, b, boundaries=True):
    kwargs = {"position_ids": torch.cat([torch.arange(A), torch.arange(B)])[None].cuda()}
    if boundaries:
        cu = torch.tensor([0, A, A + B], dtype=torch.int32, device="cuda")
        kwargs.update(cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=A, max_length_k=A,
                      seq_idx=torch.cat([torch.zeros(A), torch.ones(B)]).int()[None].cuda())
    return torch.cat([a, b])[None], kwargs


def after_linear_attention(model, ids, **kwargs):
    return model(input_ids=ids, output_hidden_states=True, **kwargs).hidden_states[1][0].float()


def relative(x, reference):
    return ((x - reference).abs().max() / reference.abs().max()).item()


def test_packed_examples_match_alone(hybrid, examples):
    a, b = examples
    with torch.no_grad():
        alone = [after_linear_attention(hybrid, x[None]) for x in (a, b)]
        together = after_linear_attention(hybrid, *packed(a, b)[:1], **packed(a, b)[1])
        leaking = after_linear_attention(hybrid, *packed(a, b, False)[:1], **packed(a, b, False)[1])
    assert relative(together[:A], alone[0]) < 0.05
    assert relative(together[A:], alone[1]) < 0.05
    # Without boundaries the second example starts with the end of the first:
    # the comparison above can tell
    assert relative(leaking[A:], alone[1]) > 0.3


def test_the_convolution_restarts_at_each_example():
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5

    assert getattr(qwen3_5.causal_conv1d_fn, "_meepsquared_boundaries", False)


def test_packed_examples_give_the_same_gradients(hybrid, examples):
    a, b = examples

    def gradients(loss):
        hybrid.zero_grad()
        loss.backward()
        return {n: p.grad.float().clone() for n, p in hybrid.named_parameters() if p.grad is not None}

    alone = gradients(sum(hybrid(input_ids=x[None]).logits.float().pow(2).mean() * len(x) for x in (a, b)))
    ids, kwargs = packed(a, b)
    together = gradients(hybrid(input_ids=ids, **kwargs).logits.float().pow(2).mean() * (A + B))
    assert alone.keys() == together.keys()
    for name in alone:
        assert relative(together[name], alone[name]) < 0.05, name

