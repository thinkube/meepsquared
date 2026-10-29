# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from transformers import AutoModelForCausalLM

from meepsquared import MeepLanguageModel

IDS = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])


@pytest.fixture
def trained(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.1)
    return model.eval()


def test_the_merged_model_computes_what_the_adapted_one_does(trained, tokenizer, tmp_path):
    with torch.no_grad():
        expected = trained(input_ids=IDS).logits
    trained.save_pretrained_merged(tmp_path, tokenizer, save_method="merged_16bit")
    merged = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert not any("lora" in name for name, _ in merged.named_parameters())
    with torch.no_grad():
        torch.testing.assert_close(merged(input_ids=IDS).logits, expected, atol=1e-5, rtol=1e-5)
    assert (tmp_path / "tokenizer_config.json").exists()


def test_the_save_leaves_the_model_as_it_was(trained, tokenizer, tmp_path):
    with torch.no_grad():
        before = trained(input_ids=IDS).logits
    trained.save_pretrained_merged(tmp_path, tokenizer)
    with torch.no_grad():
        torch.testing.assert_close(trained(input_ids=IDS).logits, before)
    assert any("lora_A" in name for name, _ in trained.named_parameters())


def test_lora_saves_the_adapter_alone(trained, tmp_path):
    trained.save_pretrained_merged(tmp_path, save_method="lora")
    assert (tmp_path / "adapter_config.json").exists() and not (tmp_path / "config.json").exists()


def test_an_adapter_save_never_reads_the_whole_state_dict(trained, tmp_path, monkeypatch):
    """On a mixture of experts with FP8 expert stacks, the model's state dict costs more memory than the training."""
    from peft import PeftModel

    with torch.no_grad():
        expected = trained(input_ids=IDS).logits

    def whole_state(*args, **kwargs):
        raise AssertionError("the adapter save read the whole model's state dict")

    monkeypatch.setattr(type(trained), "state_dict", whole_state)
    trained.save_pretrained_merged(tmp_path, save_method="lora")
    monkeypatch.undo()
    loaded = PeftModel.from_pretrained(trained.get_base_model(), tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids=IDS).logits, expected)


@pytest.mark.parametrize("method", ["merged_4bit", "gguf"])
def test_other_save_methods_are_refused(trained, tmp_path, method):
    with pytest.raises(ValueError):
        trained.save_pretrained_merged(tmp_path, save_method=method)


def test_only_the_first_process_writes(trained, tmp_path, monkeypatch):
    from meepsquared import _save

    monkeypatch.setattr(_save, "_first_process", lambda: False)
    trained.save_pretrained_merged(tmp_path / "adapter", save_method="lora")
    assert not (tmp_path / "adapter").exists()


def test_a_sharded_model_saves_the_adapter_gathered(trained, tiny_model, tmp_path, monkeypatch):
    """The sharded path, without the gather: the gathered state dict of one process is the adapter."""
    from peft import PeftModel

    from meepsquared import _save

    with torch.no_grad():
        expected = trained(input_ids=IDS).logits
    monkeypatch.setattr(_save, "_sharded", lambda model: True)
    state = _save.adapter_state_dict(trained)
    assert state and all("lora_" in name for name in state) and all(v.device.type == "cpu" for v in state.values())
    trained.save_pretrained_merged(tmp_path, save_method="lora")
    assert (tmp_path / "adapter_config.json").exists()
    loaded = PeftModel.from_pretrained(trained.get_base_model(), tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids=IDS).logits, expected)


def test_a_merged_save_of_a_sharded_model_is_refused(trained, tmp_path, monkeypatch):
    from meepsquared import _save

    monkeypatch.setattr(_save, "_sharded", lambda model: True)
    with pytest.raises(ValueError, match="sharded"):
        trained.save_pretrained_merged(tmp_path, save_method="merged_16bit")


@pytest.fixture
def trained_moe(tiny_moe):
    """A mixture of experts with LoRA on the attention and on every expert stack, computed as the training computes it."""
    from peft import LoraConfig, get_peft_model

    from meepsquared import _moe

    model = get_peft_model(tiny_moe, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                                                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
    assert _moe.apply(model) == 2
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.2)
    return model.eval()


def test_the_merged_experts_compute_what_the_adapted_ones_do(trained_moe, tmp_path):
    from meepsquared import _save

    with torch.no_grad():
        expected = trained_moe(input_ids=IDS).logits
        with trained_moe.disable_adapter():
            base = trained_moe(input_ids=IDS).logits
    assert not torch.allclose(expected, base, atol=1e-3)  # the adapters on the experts change the output
    _save.save(trained_moe, tmp_path, save_method="merged_16bit")
    merged = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert not any("lora" in name for name, _ in merged.named_parameters())
    assert not any(name.endswith("_scale_inv") for name, _ in merged.named_buffers())
    with torch.no_grad():
        torch.testing.assert_close(merged(input_ids=IDS).logits, expected, atol=1e-4, rtol=1e-4)


def test_fp8_expert_stacks_are_merged_from_their_dequantized_weights(trained_moe, tmp_path):
    from meepsquared import _fp8, _moe, _save

    for module in list(trained_moe.modules()):
        if _moe.routed(module):
            assert _fp8.convert_stacks(_moe._wrapper_chain(module)[1]) == 2
    with torch.no_grad():
        expected = trained_moe(input_ids=IDS).logits
    _save.save(trained_moe, tmp_path, save_method="merged_16bit")
    merged = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    stack = merged.model.layers[0].mlp.experts.gate_up_proj
    assert stack.dtype == torch.float32 and not any(name.endswith("_scale_inv") for name, _ in merged.named_buffers())
    with torch.no_grad():
        torch.testing.assert_close(merged(input_ids=IDS).logits, expected, atol=1e-4, rtol=1e-4)


def test_a_merged_model_is_written_in_shards_the_size_asks_for(trained, tmp_path, monkeypatch):
    from meepsquared import _save

    with torch.no_grad():
        expected = trained(input_ids=IDS).logits
    (tmp_path / "model-00001-of-00009.safetensors").write_bytes(b"left by an earlier save")
    monkeypatch.setattr(_save, "MAX_SHARD_BYTES", 16 * 1024)
    trained.save_pretrained_merged(tmp_path, save_method="merged_16bit")
    shards = sorted(p.name for p in tmp_path.glob("model-*.safetensors"))
    assert len(shards) > 1 and shards[0] == f"model-00001-of-{len(shards):05d}.safetensors"
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == set(shards)
    merged = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(merged(input_ids=IDS).logits, expected, atol=1e-5, rtol=1e-5)
