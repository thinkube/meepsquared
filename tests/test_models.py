# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from transformers import LlamaConfig, PretrainedConfig

from meepsquared import MeepLanguageModel, MeepModel
from meepsquared.models import (ATTENTION_TARGETS, DEFAULT_TARGETS, attention_by_part, check_flash_attention_heads,
                           language_model_targets, padding_free_ok)


def test_padding_free_needs_flash_attention():
    config = LlamaConfig()
    assert padding_free_ok(config, "flash_attention_2")
    assert padding_free_ok(config, "flash_attention_3")
    assert not padding_free_ok(config, "sdpa")
    assert not padding_free_ok(config, "eager")


def test_attention_by_part_for_a_model_loaded_with_its_towers():
    from transformers import AutoConfig

    # Gemma 4's causal-LM class is the whole model, audio and vision towers included
    parts = attention_by_part(AutoConfig.for_model("gemma4"), "flash_attention_2")
    assert parts == {"": "flash_attention_2", "text_config": "flash_attention_2",
                     "vision_config": None, "audio_config": None}


def test_attention_as_given_for_a_model_loaded_as_its_language_model():
    from transformers import AutoConfig

    assert attention_by_part(LlamaConfig(), "flash_attention_2") == "flash_attention_2"
    # Qwen3.5's checkpoint is multimodal, but its causal-LM class is the language model alone
    assert attention_by_part(AutoConfig.for_model("qwen3_5"), "flash_attention_2") == "flash_attention_2"


def test_flash_attention_refused_for_heads_wider_than_256():
    config = PretrainedConfig(head_dim=256, global_head_dim=512)
    with pytest.raises(ValueError, match="global_head_dim"):
        check_flash_attention_heads(config, "flash_attention_2")
    check_flash_attention_heads(config, "sdpa")
    check_flash_attention_heads(PretrainedConfig(head_dim=256), "flash_attention_2")
    check_flash_attention_heads(LlamaConfig(), "flash_attention_2")


class WithVisionTower(torch.nn.Module):
    """A language model and a vision tower whose layers have the same names, as in Gemma 4."""

    def __init__(self, config):
        super().__init__()
        from transformers import LlamaModel

        self.model = torch.nn.Module()
        self.model.language_model = LlamaModel(config)
        self.model.vision_tower = LlamaModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_decoder(self):
        return self.model.language_model


def test_targets_leave_out_the_towers(tiny_model):
    model = WithVisionTower(tiny_model.config)
    targets = language_model_targets(model, ["q_proj", "lm_head"])
    assert "lm_head" in targets
    assert [t for t in targets if t.endswith("q_proj")] == [
        f"model.language_model.layers.{i}.self_attn.q_proj" for i in range(2)]


def test_targets_as_given_without_towers(tiny_model):
    assert language_model_targets(tiny_model, ["q_proj", "lm_head"]) == ["q_proj", "lm_head"]
    assert language_model_targets(tiny_model, r".*q_proj") == r".*q_proj"


@pytest.mark.parametrize("trl_version, fla, expected", [("0.29.1", True, False), ("1.15.0", True, False),
                                                        ("1.16.0", False, False), ("1.16.0", True, True)])
def test_padding_free_with_linear_attention_needs_trl_1_16_and_fla(monkeypatch, trl_version, fla, expected):
    import trl

    from meepsquared import _linear_attention

    monkeypatch.setattr(trl, "__version__", trl_version)
    monkeypatch.setattr(_linear_attention, "available", lambda: fla)
    config = PretrainedConfig(layer_types=["linear_attention", "linear_attention", "full_attention"])
    assert padding_free_ok(config, "flash_attention_2") is expected


def lora_names(model):
    return {name.split(".lora_")[0].split(".")[-1] for name, _ in model.named_parameters() if ".lora_" in name}


def test_get_peft_model_trains_only_the_adapters(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in name for name in trainable)
    assert lora_names(model) == set(DEFAULT_TARGETS)
    assert model._meepsquared == {"padding_free": True, "liger": False}


def test_random_state_fixes_the_initial_adapters(tiny_model):
    a = MeepLanguageModel.get_peft_model(tiny_model, r=4, random_state=7, use_gradient_checkpointing=False)
    first = {n: p.detach().clone() for n, p in a.named_parameters() if "lora_A" in n}
    a.unload()
    b = MeepLanguageModel.get_peft_model(tiny_model, r=4, random_state=7, use_gradient_checkpointing=False)
    for name, p in b.named_parameters():
        if "lora_A" in name:
            assert torch.equal(p, first[name])


def test_gradient_checkpointing_follows_the_argument(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    assert model.base_model.model.is_gradient_checkpointing
    MeepLanguageModel.for_inference(model)
    assert not model.base_model.model.is_gradient_checkpointing and model.config.use_cache
    MeepLanguageModel.for_training(model)
    assert model.base_model.model.is_gradient_checkpointing and not model.config.use_cache


def test_get_peft_model_refuses_a_model_meepsquared_did_not_load(tiny_model):
    del tiny_model._meepsquared
    with pytest.raises(TypeError):
        MeepLanguageModel.get_peft_model(tiny_model)


def test_fast_model_chooses_targets_by_kind(tiny_model):
    model = MeepModel.get_peft_model(tiny_model, r=4, finetune_mlp_modules=False, use_gradient_checkpointing=False)
    assert lora_names(model) == set(ATTENTION_TARGETS)


def test_fast_model_refuses_vision_layers_and_empty_choices(tiny_model):
    with pytest.raises(NotImplementedError):
        MeepModel.get_peft_model(tiny_model, finetune_vision_layers=True)
    with pytest.raises(ValueError):
        MeepModel.get_peft_model(tiny_model, finetune_attention_modules=False, finetune_mlp_modules=False)


def test_fast_inference_is_refused_before_anything_loads():
    with pytest.raises(NotImplementedError, match="use_vllm"):
        MeepLanguageModel.from_pretrained("no/such-model", fast_inference=True)


def test_defaults_decide_for_the_user(tiny_model, monkeypatch):
    from meepsquared import _compile, _fused, models

    monkeypatch.setattr(_fused, "available", lambda: True)
    assert "CUDA" in models._fused_wanted(tiny_model)[1]  # a model on the CPU
    if torch.cuda.is_available():
        tiny_model.to("cuda")
        assert models._fused_wanted(tiny_model) == (True, "")
        tiny_model.config.hidden_act = "relu"
        wanted, why = models._fused_wanted(tiny_model)
        assert not wanted and "relu" in why
        tiny_model.config.hidden_act = "silu"
        monkeypatch.setattr(_fused, "available", lambda: False)
        assert "bitsandbytes" in models._fused_wanted(tiny_model)[1]
        tiny_model.to("cpu")

    assert _compile.wanted(LlamaConfig(), "flash_attention_2") is False
    assert _compile.wanted(LlamaConfig(), "sdpa") is torch.cuda.is_available()
    linear = PretrainedConfig(layer_types=["linear_attention", "full_attention"])
    assert _compile.wanted(linear, "sdpa") is False


def test_get_peft_model_defaults(tiny_model, monkeypatch):
    from meepsquared import _fused

    monkeypatch.setattr(_fused, "available", lambda: False)
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4)  # on the CPU: no kernels, no compile
    assert model._meepsquared_gradient_checkpointing == "auto"
    assert model._meepsquared_fused_lora is False
    assert model._meepsquared_compile_layers is False


def test_pad_token_is_a_padding_token_of_the_vocabulary_else_eos(tokenizer):
    from meepsquared.models import pad_token_for

    assert pad_token_for(tokenizer) == "<pad>"
    bare = copy.deepcopy(tokenizer)
    bare.pad_token = None
    assert pad_token_for(bare) == "<pad>"  # still in the vocabulary, as Llama 3.1's <|finetune_right_pad_id|>
    bare.add_special_tokens({"additional_special_tokens": ["<|finetune_right_pad_id|>"]})
    assert "pad" in pad_token_for(bare)
