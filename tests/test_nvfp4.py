# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Frozen layers held as NVFP4 by Transformer Engine: refused where they cannot run, and on a Blackwell GPU computing
as the 16-bit layer does within FP4's error, for any count of rows."""

import pytest
import torch

from meepsquared import MeepLanguageModel, _nvfp4

blackwell = pytest.mark.skipif(not _nvfp4.available()[0], reason="NVFP4 runs on Blackwell with Transformer Engine")


@pytest.mark.skipif(torch.cuda.is_available(), reason="the refusal of a machine without a CUDA GPU")
def test_nvfp4_is_refused_without_a_gpu_before_the_model_loads():
    with pytest.raises(RuntimeError, match="load_in_nvfp4=True: NVFP4 products run on a CUDA GPU"):
        MeepLanguageModel.from_pretrained("no/such-model", load_in_nvfp4=True)


def test_one_way_to_hold_the_frozen_weights():
    with pytest.raises(ValueError, match="choose one"):
        MeepLanguageModel.from_pretrained("no/such-model", load_in_4bit=True, load_in_nvfp4=True)


@pytest.mark.parametrize("method", ["fp8", "nvfp4"])
def test_a_choice_job_loads_the_frozen_layers_for_its_method(monkeypatch, method):
    from meepsquared import _choice_job, models
    from meepsquared.job import TrainingJob

    seen = {}

    class Loaded(Exception):
        pass

    def from_pretrained(name, **kwargs):
        seen.update(kwargs)
        raise Loaded

    monkeypatch.setattr(models.MeepLanguageModel, "from_pretrained", staticmethod(from_pretrained))
    with pytest.raises(Loaded):
        _choice_job._load(TrainingJob(model="org/model", task="choice", method=method, data="d.jsonl"), [])
    assert seen[f"load_in_{method}"] and not seen["load_in_4bit"]
    assert not seen["load_in_fp8" if method == "nvfp4" else "load_in_nvfp4"]


def relative(x, reference):
    return ((x.float() - reference.float()).norm() / reference.float().norm()).item()


@blackwell
@pytest.mark.parametrize("shape", [(1, 1357, 256), (4, 37, 256), (1, 1, 256), (7, 256)])
def test_an_nvfp4_layer_computes_as_its_16_bit_layer_for_any_rows(shape):
    torch.manual_seed(0)
    dense = torch.nn.Linear(256, 512, bias=True, device="cuda", dtype=torch.bfloat16).requires_grad_(False)
    layer = _nvfp4.Nvfp4Linear(dense)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    out.float().pow(2).mean().backward()
    x_ref = x.detach().clone().requires_grad_()
    out_ref = dense(x_ref)
    out_ref.float().pow(2).mean().backward()
    assert out.shape == out_ref.shape
    # NVFP4's error on random values is about 15% of the norm
    assert relative(out, out_ref) < 0.25 and relative(x.grad, x_ref.grad) < 0.3
    assert layer.dequantized().shape == dense.weight.shape and not layer.inner.weight.requires_grad


@blackwell
@pytest.mark.parametrize("shape", [(1, 1357, 256), (7, 256)])
def test_the_input_gradient_is_the_16_bit_product_with_the_dequantized_weight(shape):
    torch.manual_seed(0)
    dense = torch.nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16).requires_grad_(False)
    layer = _nvfp4.Nvfp4Linear(dense)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    expected = grad_out.reshape(-1, 512) @ layer.dequantized()
    # no quantization of the gradient: only bfloat16's rounding of the product stands between the two
    assert relative(x.grad.reshape(-1, 256), expected) < 1e-2


@blackwell
def test_the_adapters_train_on_nvfp4_layers(tiny_model):
    from peft import LoraConfig, get_peft_model

    model = tiny_model.to("cuda", torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    peft = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    assert _nvfp4.apply(peft) > 0
    ids = torch.randint(0, 10, (2, 13), device="cuda")
    peft(input_ids=ids).logits.float().pow(2).mean().backward()
    grads = [p.grad for n, p in peft.named_parameters() if "lora_" in n]
    assert grads and all(g is not None for g in grads)
