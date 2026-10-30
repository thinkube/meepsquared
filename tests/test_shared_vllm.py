# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The shared vLLM of the milestones, in what runs without vLLM: the check's comparison, the adapter renamed to the
checkpoint's names, the refusals, FP8 stacks left as vLLM holds them, and router logits off while generating. The
engine itself is tested on a GPU (design/milestones-vllm-shared-weights.md, §6 and §7)."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

import meepsquared as m2
from meepsquared import _fp8, _milestones, _shared_vllm, _vllm
from meepsquared.job import TrainingJob


def test_the_check_takes_ties_at_rounding_as_the_same_top_token():
    mine = [torch.log_softmax(torch.tensor([2.0, 1.9, -1.0]), 0), torch.log_softmax(torch.tensor([3.0, 0.0, 0.0]), 0)]
    theirs = [{1: float(mine[0][1]), 0: float(mine[0][0])}, {0: float(mine[1][0]) + 0.01, 1: float(mine[1][1])}]
    found = _shared_vllm.compare(mine, theirs)
    assert found["top_same"] and found["ties"] == 1 and found["max_logprob_diff"] == pytest.approx(0.01, abs=1e-6)
    # A top token the trainer ranks far below its own is a difference
    assert not _shared_vllm.compare([mine[1]], [{1: 0.0, 0: -5.0}])["top_same"]


def test_the_adapter_is_renamed_to_the_checkpoints_names(tmp_path, monkeypatch):
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    monkeypatch.setattr(_vllm, "checkpoint_name",
                        lambda model, name: name.replace("model.layers.", "model.language_model.layers."))
    weights = {"base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight": torch.ones(2, 2),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(2, 2)}
    save_file(weights, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2}))
    with _shared_vllm.checkpoint_named(Base(), tmp_path) as named:
        renamed = sorted(load_file(str(named / "adapter_model.safetensors")))
        assert json.loads((named / "adapter_config.json").read_text()) == {"r": 2}
    assert renamed == ["base_model.model.model.language_model.layers.0.mlp.experts.base_layer.lora_A.weight",
                       "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert not named.exists()


def test_other_architectures_are_refused_by_name(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    with pytest.raises(_shared_vllm.SharedError, match="tested on .*Qwen3_5MoeForConditionalGeneration"):
        _shared_vllm.check_architecture(str(tmp_path))


def test_the_engine_is_for_milestones_with_an_evaluation_and_not_for_qlora():
    with pytest.raises(ValueError, match="'vllm' or not given"):
        m2.Overrides(milestone_engine="hf")
    with pytest.raises(ValueError, match="give milestones and milestone_eval"):
        TrainingJob(model="m", data="d.jsonl", overrides=m2.Overrides(milestone_engine="vllm"))
    with pytest.raises(ValueError, match="'qlora' holds them in 4 bits"):
        TrainingJob(model="m", data="d.jsonl", method="qlora", milestones=0.1, milestone_eval="e.jsonl",
                    milestone_scorer="s.py:score", overrides=m2.Overrides(milestone_engine="vllm"))


def test_fp8_leaves_the_stacks_a_shared_vllm_holds_as_they_are():
    experts = torch.nn.Module()
    stack = torch.randn(4, 6, 8).to(torch.float8_e4m3fn)
    experts.gate_up_proj = torch.nn.Parameter(stack, requires_grad=False)
    experts.register_buffer("gate_up_proj" + _fp8.STACK_SCALE, torch.full((4,), 0.5))
    assert _fp8.convert_stacks(experts) == 1
    assert experts.gate_up_proj.data_ptr() == stack.data_ptr()
    assert torch.equal(getattr(experts, "gate_up_proj" + _fp8.STACK_SCALE), torch.full((4,), 0.5))


def test_router_logits_are_off_while_generating_and_back_after():
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(vocab_size=32, hidden_size=32, num_hidden_layers=1, layer_types=["full_attention"],
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=16, num_experts=4,
                                  num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
                                  output_router_logits=True)
    model = Qwen3_5MoeForCausalLM(config)
    model.train()
    with _milestones._Inference(model):
        assert model.config.output_router_logits is False and not model.training
    assert model.config.output_router_logits is True and model.training
