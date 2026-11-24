# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Expert stacks held as NF4: read from the checkpoint one at a time, run through Transformers' experts interface,
computing as their dequantized stacks do, forward and backward."""

import copy

import pytest
import torch

from meepsquared import _nf4_experts

pytest.importorskip("bitsandbytes")


def tiny():
    """A random two-layer Qwen3.5 mixture of experts in bfloat16: Transformers holds and its checkpoints store each
    expert stack as one tensor, as Gemma 4's do."""
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(0)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=2, layer_types=["full_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, max_position_embeddings=64,
        initializer_range=0.2,
    )
    config._experts_implementation = "grouped_mm"
    return Qwen3_5MoeForCausalLM(config).to(torch.bfloat16)


@pytest.fixture
def checkpoint(tmp_path):
    """The tiny model written as a checkpoint of the names it holds, and the model."""
    from safetensors.torch import save_model

    model = tiny()
    model.config.save_pretrained(tmp_path)
    save_model(model, str(tmp_path / "model.safetensors"))
    return str(tmp_path), model


def load_without_stacks(path):
    from transformers import AutoModelForCausalLM

    with _nf4_experts.without_stacks():
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)


def stack_names(model):
    return sorted(n for n, _ in model.named_parameters() if _nf4_experts.is_stack(n))


def test_a_stack_is_named_by_its_experts_module():
    assert _nf4_experts.is_stack("model.layers.3.mlp.experts.gate_up_proj")
    assert _nf4_experts.is_stack("model.language_model.layers.0.experts.down_proj")
    assert not _nf4_experts.is_stack("model.layers.3.mlp.down_proj.weight")
    assert not _nf4_experts.is_stack("model.layers.3.mlp.shared_expert.down_proj.weight")
    assert not _nf4_experts.is_stack("model.layers.3.mlp.experts.gate_up_proj_bias")


def test_the_stacks_stay_unread_while_the_rest_loads(checkpoint):
    path, original = checkpoint
    files = _nf4_experts.checkpoint_stacks(path)
    assert sorted(files) == stack_names(original) and len(files) == 4
    model = load_without_stacks(path)
    params = dict(model.named_parameters())
    assert all(params[k].device.type == "meta" for k in files)
    reference = dict(original.named_parameters())
    assert all(torch.equal(p, reference[n]) for n, p in params.items() if n not in files)


def test_the_stacks_are_read_into_nf4_and_leave_the_parameters(checkpoint):
    path, original = checkpoint
    model = load_without_stacks(path)
    done = _nf4_experts.load(model, path, _nf4_experts.checkpoint_stacks(path), torch.bfloat16, double_quant=False)
    assert (done["stacks"], done["modules"]) == (4, 2)
    assert done["parameters"] == sum(p.numel() for n, p in original.named_parameters() if _nf4_experts.is_stack(n))
    # Half a byte a weight, and one float32 scale for each block of 64
    assert done["bytes"] == done["parameters"] // 2 + done["parameters"] // _nf4_experts.BLOCK * 4
    assert stack_names(model) == []
    assert model.config._experts_implementation == _nf4_experts.NAME
    # The packed stacks are not in the state dict: a save of the base model does not write them as they are
    assert not any(k.endswith(_nf4_experts.PACKED) for k in model.state_dict())


def run(model, ids):
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
    logits = model(inputs_embeds=embeds).logits.float()
    logits.pow(2).mean().backward()
    return logits.detach(), embeds.grad.float()


def relative(x, reference):
    return ((x - reference).abs().max() / reference.abs().max().clamp(min=1e-6)).item()


def test_nf4_experts_compute_as_their_dequantized_stacks_forward_and_backward(checkpoint):
    path, original = checkpoint
    model = load_without_stacks(path)
    _nf4_experts.load(model, path, _nf4_experts.checkpoint_stacks(path), torch.bfloat16, double_quant=False)
    # The reference: the 16-bit model with each stack replaced by its NF4 stack dequantized, on Transformers' own
    # grouped forward
    reference = copy.deepcopy(original)
    for name, experts in _nf4_experts.nf4_modules(model):
        for attr, stack in _nf4_experts.dequantized_stacks(experts, torch.bfloat16):
            getattr(reference.get_submodule(name), attr).data.copy_(stack)
    ids = torch.randint(0, 64, (2, 12))
    out, grad = run(model, ids)
    out_reference, grad_reference = run(reference, ids)
    assert relative(out, out_reference) < 1e-2
    assert relative(grad, grad_reference) < 2e-2
    # Each stack held is the model's own stack through bitsandbytes' NF4: the right tensor was read and quantized.
    # (The round trip is compared, not the original: bitsandbytes' CPU kernels differ by the CPU's instructions.)
    import bitsandbytes.functional as bnbf

    stacks = dict(original.named_parameters())
    for name, experts in _nf4_experts.nf4_modules(model):
        for attr, stack in _nf4_experts.dequantized_stacks(experts, torch.bfloat16):
            weight = stacks[f"{name}.{attr}"].detach()
            packed, state = bnbf.quantize_4bit(weight.contiguous(), blocksize=_nf4_experts.BLOCK,
                                               compress_statistics=False, quant_type="nf4")
            assert torch.equal(stack, bnbf.dequantize_4bit(packed, state)) and not torch.equal(stack, weight)


def test_stacks_loaded_in_16_bits_are_quantized_as_the_checkpoints(checkpoint):
    path, original = checkpoint
    from_checkpoint = load_without_stacks(path)
    _nf4_experts.load(from_checkpoint, path, _nf4_experts.checkpoint_stacks(path), torch.bfloat16, False)
    loaded = copy.deepcopy(original)
    done = _nf4_experts.load(loaded, path, {}, torch.bfloat16, double_quant=False)
    assert (done["stacks"], done["modules"]) == (4, 2)
    packed = {n: b for n, b in from_checkpoint.named_buffers() if n.endswith(_nf4_experts.PACKED)}
    assert len(packed) == 4
    assert all(torch.equal(b, packed[n]) for n, b in loaded.named_buffers() if n.endswith(_nf4_experts.PACKED))


def test_a_model_without_expert_stacks_is_left_as_it_is(tiny_model):
    assert _nf4_experts.load(tiny_model, "llama", {}, torch.float32, double_quant=False) is None


def test_a_merged_save_writes_the_nf4_stacks_dequantized(checkpoint):
    from peft import LoraConfig, get_peft_model

    from meepsquared import _save

    path, original = checkpoint
    model = load_without_stacks(path)
    _nf4_experts.load(model, path, _nf4_experts.checkpoint_stacks(path), torch.bfloat16, double_quant=False)
    peft_model = get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj"]))
    written = dict(_save.merged_tensors(peft_model))
    shapes = {n: tuple(p.shape) for n, p in original.named_parameters() if _nf4_experts.is_stack(n)}
    assert {n: tuple(written[n].shape) for n in shapes} == shapes
    assert not any(n.endswith(_nf4_experts.PACKED) for n in written)


def test_lora_on_nf4_stacks_is_refused(checkpoint):
    from meepsquared import MeepLanguageModel

    path, _ = checkpoint
    model = load_without_stacks(path)
    model._meepsquared = {"nf4_experts": _nf4_experts.load(model, path, _nf4_experts.checkpoint_stacks(path),
                                                           torch.bfloat16, double_quant=False)}
    with pytest.raises(ValueError, match="experts_4bit=False"):
        MeepLanguageModel.get_peft_model(model, target_parameters=["mlp.experts.gate_up_proj"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes' 4-bit loading runs on a GPU")
def test_from_pretrained_in_4bit_holds_the_expert_stacks_as_nf4(checkpoint, tokenizer, tmp_path):
    from meepsquared import MeepLanguageModel

    path, _ = checkpoint
    tokenizer.save_pretrained(path)
    model, _ = MeepLanguageModel.from_pretrained(path, load_in_4bit=True, attn_implementation="sdpa", liger=False)
    assert model._meepsquared["nf4_experts"]["stacks"] == 4 and stack_names(model) == []
    kept, _ = MeepLanguageModel.from_pretrained(path, load_in_4bit=True, attn_implementation="sdpa", liger=False,
                                                experts_4bit=False)
    assert kept._meepsquared["nf4_experts"] is None and len(stack_names(kept)) == 4
    ids = torch.randint(0, 64, (1, 8), device="cuda")
    assert torch.isfinite(model(input_ids=ids).logits).all()
