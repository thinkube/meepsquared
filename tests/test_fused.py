# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from meepsquared import MeepLanguageModel, MeepModel

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the SwiGLU kernel runs on a GPU")


def with_adapters(model, **options):
    """The PEFT model with non-zero B matrices, so the adapters change the output."""
    peft_model = MeepLanguageModel.get_peft_model(model, r=4, use_gradient_checkpointing=False, **options)
    torch.manual_seed(1)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.1)
    return peft_model


def outputs_and_gradients(model, ids):
    model.zero_grad()
    logits = model(input_ids=ids).logits
    logits.float().pow(2).mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def assert_same_training(reference, fused, ids, tol):
    out_ref, grad_ref = outputs_and_gradients(reference, ids)
    out_fused, grad_fused = outputs_and_gradients(fused, ids)
    torch.testing.assert_close(out_fused, out_ref, atol=tol, rtol=tol)
    assert grad_ref.keys() == grad_fused.keys()
    for name in grad_ref:
        torch.testing.assert_close(grad_fused[name], grad_ref[name], atol=tol, rtol=tol, msg=name)


def test_fused_projections_train_like_peft(tiny_model):
    from meepsquared import _fused

    reference = with_adapters(copy.deepcopy(tiny_model), target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    fused = copy.deepcopy(reference)
    assert _fused.apply(fused) == {"mlp": 0, "qkv": 2, "projections": 2}
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])
    assert_same_training(reference, fused, ids, tol=1e-5)


def test_a_mlp_with_another_activation_is_refused(tiny_model):
    tiny_model.config.hidden_act = "relu"
    with pytest.raises(NotImplementedError, match="relu"):
        MeepModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, fused_lora=True)


def test_an_activation_named_hidden_activation_is_read(tiny_model, monkeypatch):
    import types

    # Gemma's text configs name the MLP activation hidden_activation, and have no hidden_act
    from meepsquared import _fused

    monkeypatch.setattr(type(tiny_model.config), "get_text_config",
                        lambda self, **kwargs: types.SimpleNamespace(hidden_activation="gelu_pytorch_tanh"))
    model = MeepModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, fused_lora=True)
    mlp = model.base_model.model.model.layers[0].mlp
    assert mlp._meepsquared_mlp is _fused._mlp_geglu_tanh


def test_applying_again_keeps_the_kernels(tiny_model):
    from meepsquared import _fused

    model = with_adapters(tiny_model, target_modules=["q_proj", "o_proj"], fused_lora=True)
    q = model.base_model.model.model.layers[0].self_attn.q_proj
    forward = q.forward
    _fused.apply(model)
    assert q.forward is forward


@cuda
def test_a_copy_of_a_fused_model_computes_on_its_own_parameters(tiny_model):
    fused = with_adapters(tiny_model.to("cuda", torch.bfloat16), fused_lora=True)
    twin = copy.deepcopy(fused)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]], device="cuda")
    _, grads = outputs_and_gradients(twin, ids)
    assert grads and all(p.grad is None for p in fused.parameters())
    attention = twin.base_model.model.model.layers[0].self_attn
    assert attention.q_proj.forward.__self__ is attention.q_proj
    assert attention.q_proj._meepsquared_qkv.attention is attention
    assert attention.q_proj._meepsquared_qkv is attention.k_proj._meepsquared_qkv


def test_k_proj_alone_computes_its_own(tiny_model):
    from meepsquared import _fused

    reference = with_adapters(copy.deepcopy(tiny_model), target_modules=["q_proj", "k_proj", "v_proj"])
    fused = copy.deepcopy(reference)
    _fused.apply(fused)
    x = torch.randn(1, 3, tiny_model.config.hidden_size)
    attention = fused.base_model.model.model.layers[0].self_attn
    expected = reference.base_model.model.model.layers[0].self_attn
    torch.testing.assert_close(attention.k_proj(x), expected.k_proj(x), atol=1e-5, rtol=1e-5)
    q = attention.q_proj(x)
    torch.testing.assert_close(q, expected.q_proj(x), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(attention.v_proj(x), expected.v_proj(x), atol=1e-5, rtol=1e-5)


@cuda
def test_fused_mlp_and_projections_train_like_peft(tiny_model):
    model = tiny_model.to("cuda", torch.bfloat16)
    reference = with_adapters(copy.deepcopy(model))
    fused = copy.deepcopy(reference)
    from meepsquared import _fused

    assert _fused.apply(fused) == {"mlp": 2, "qkv": 2, "projections": 2}
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]], device="cuda")
    assert_same_training(reference, fused, ids, tol=2e-2)


@cuda
def test_the_fused_mlp_takes_a_two_dimensional_input(tiny_model):
    """A mixture of experts gives its shared expert (tokens, hidden); the kernels compute on (batch, sequence, hidden)."""
    model = with_adapters(tiny_model.to("cuda", torch.bfloat16), target_modules=["gate_proj", "up_proj", "down_proj"],
                          fused_lora=True)
    mlp = model.base_model.model.model.layers[0].mlp
    x = torch.randn(7, tiny_model.config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    flat = mlp(x)
    assert flat.shape == x.shape
    torch.testing.assert_close(flat, mlp(x.unsqueeze(0)).squeeze(0))
    flat.float().pow(2).mean().backward()
    assert x.grad is not None and x.grad.shape == x.shape
