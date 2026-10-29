# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from meepsquared import MeepLanguageModel

TEXT = Dataset.from_dict({"text": ["the hare runs fast", "and the sloth sleeps"]})


def trainer(model, tokenizer, tmp_path, **config):
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1, **config)
    return SFTTrainer(model=model, args=args, train_dataset=TEXT, processing_class=tokenizer)


def test_padding_free_and_checkpointing_come_from_the_model(tiny_model, tokenizer, tmp_path):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.padding_free is True
    assert t.args.gradient_checkpointing is False
    assert t.args.use_liger_kernel is False


def test_a_model_that_cannot_go_padding_free_is_padded(tiny_model, tokenizer, tmp_path):
    tiny_model._meepsquared = {"padding_free": False, "liger": False}
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.padding_free is False
    assert t.args.gradient_checkpointing is True


def test_the_config_wins_when_it_sets_them(tiny_model, tokenizer, tmp_path):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    t = trainer(model, tokenizer, tmp_path, padding_free=False, gradient_checkpointing=False)
    assert t.args.padding_free is False
    assert not model.base_model.model.is_gradient_checkpointing


def test_fused_lora_keeps_its_mlp_from_liger(tiny_model, tokenizer, tmp_path):
    # The Trainer applies Liger in train(); its SwiGLU would replace the fused MLP forward
    tiny_model._meepsquared = {"padding_free": True, "liger": True}
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, fused_lora=True)
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.use_liger_kernel is True
    assert t.args.liger_kernel_config["swiglu"] is False and t.args.liger_kernel_config["geglu"] is False


def auto_trainer(model, tokenizer, tmp_path, monkeypatch, layout, **config):
    """A trainer for use_gradient_checkpointing="auto", its measurement replaced by `layout`; returns it and the
    arguments resolve_auto was called with."""
    from meepsquared import _checkpointing

    calls = []

    def resolve_auto(model, shape, compiled=False, refused=None, batch=None):
        calls.append({"compiled": compiled, "refused": refused})
        _checkpointing.set_checkpointed(model, layout)
        return layout

    monkeypatch.setattr(_checkpointing, "resolve_auto", resolve_auto)
    return trainer(model, tokenizer, tmp_path, **config), calls


def test_auto_keeps_ligers_rope_unless_it_chose_a_selective_layer(tiny_model, tokenizer, tmp_path, monkeypatch):
    import copy

    from meepsquared import _checkpointing

    tiny_model._meepsquared = {"padding_free": True, "liger": True}
    tiny_model.max_seq_length = 16  # as from_pretrained sets it
    model = MeepLanguageModel.get_peft_model(copy.deepcopy(tiny_model), r=4, use_gradient_checkpointing="auto")
    t, calls = auto_trainer(model, tokenizer, tmp_path, monkeypatch, _checkpointing.Recompute(0, 0))
    assert calls == [{"compiled": False, "refused": None}]
    assert (t.args.liger_kernel_config or {}).get("rope") is not False
    assert _checkpointing.kinds(model) == ["none", "none"]

    model = MeepLanguageModel.get_peft_model(copy.deepcopy(tiny_model), r=4, use_gradient_checkpointing="auto")
    t, _ = auto_trainer(model, tokenizer, tmp_path, monkeypatch, _checkpointing.Recompute(0, 1))
    assert t.args.liger_kernel_config["rope"] is False
    assert _checkpointing.kinds(model) == ["selective", "none"]


def test_auto_leaves_selective_out_when_the_config_keeps_ligers_rope(tiny_model, tokenizer, tmp_path, monkeypatch):
    from meepsquared import _checkpointing, _trl

    tiny_model._meepsquared = {"padding_free": True, "liger": True}
    tiny_model.max_seq_length = 16  # as from_pretrained sets it
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing="auto")
    t, calls = auto_trainer(model, tokenizer, tmp_path, monkeypatch, _checkpointing.Recompute(0, 0),
                            liger_kernel_config={"rms_norm": True})
    assert calls[0]["refused"] == _trl.ROPE_KEPT
    assert t.args.liger_kernel_config.get("rope") is not False


def test_selective_recomputation_leaves_ligers_rope_off(tiny_model, tokenizer, tmp_path):
    # Liger's rotary embedding rotates q and k in place, over the projection outputs a selective layer keeps
    tiny_model._meepsquared = {"padding_free": True, "liger": True}
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing="selective")
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.use_liger_kernel is True and t.args.liger_kernel_config["rope"] is False
    assert t.args.gradient_checkpointing is False
    with pytest.raises(ValueError, match="rope"):
        trainer(model, tokenizer, tmp_path, liger_kernel_config={"rms_norm": True})
    t = trainer(model, tokenizer, tmp_path, liger_kernel_config={"rope": False})
    assert t.args.liger_kernel_config["rope"] is False


def test_a_trainer_of_ones_own_is_measured_on_the_longest_examples(tiny_model, tokenizer, tmp_path):
    from datasets import Dataset

    from meepsquared import _trl

    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=1)
    t = trainer(model, tokenizer, tmp_path, max_length=3, per_device_train_batch_size=8)
    # TRL 0.29 cuts the examples to max_length while tokenizing, TRL 1.x leaves that to the collator
    lengths = _trl.example_lengths(t.train_dataset)
    assert len(lengths) == 2 and all(n >= 3 for n in lengths)
    # Padding-free: the 2 examples of the data, each cut to 3 tokens, end to end in one row; the trainer's
    # own sampler orders them, so the bound is the longest examples
    shape, text = _trl.measured_shape(t, "sft", 3)
    assert (shape.rows, shape.length, shape.lengths) == (1, 6, (3, 3))
    assert text == "the data's 2 longest examples, 6 tokens (2 examples, end to end): the bound, as the trainer " \
                   "orders its batches itself"
    # GRPO's sequences are generated: measured at their bound
    shape, text = _trl.measured_shape(t, "grpo", 3)
    assert shape == _checkpointing_shape(8, 3)
    assert text == "8 x 3 tokens, the bound: GRPO generates its completions as it trains"
    with pytest.raises(ValueError, match="input_ids"):
        _trl.example_lengths(Dataset.from_dict({"text": ["a"]}))


def _checkpointing_shape(rows, length):
    from meepsquared import _checkpointing

    return _checkpointing.Shape(rows, length)


def test_some_layers_checkpointed_are_set_by_meepsquared_not_the_trainer(tiny_model, tokenizer, tmp_path):
    from meepsquared import _checkpointing

    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=1)
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.gradient_checkpointing is False
    assert [layer.gradient_checkpointing for layer in _checkpointing.layers(model)] == [True, False]
    with pytest.raises(ValueError, match="leave gradient_checkpointing out"):
        trainer(model, tokenizer, tmp_path, gradient_checkpointing=True)


def test_compiled_layers_pad_to_a_multiple(tiny_model, tokenizer, tmp_path):
    tiny_model._meepsquared = {"padding_free": False, "liger": False}
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, compile_layers=True)
    t = trainer(model, tokenizer, tmp_path)
    assert t.args.pad_to_multiple_of == 128
    t = trainer(model, tokenizer, tmp_path, pad_to_multiple_of=16)
    assert t.args.pad_to_multiple_of == 16



    del tiny_model._meepsquared
    t = trainer(tiny_model, tokenizer, tmp_path, gradient_checkpointing=False)
    assert t.args.padding_free is False


def test_examples_are_cut_to_max_length(tiny_model, tokenizer, tmp_path):
    long = Dataset.from_dict({"text": ["the hare runs fast and the sloth sleeps"] * 2})
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1, max_length=3)
    t = SFTTrainer(model=model, args=args, train_dataset=long, processing_class=tokenizer)
    assert t.args.padding_free is True and t.args.max_length == 3
    batch = t.data_collator([t.train_dataset[0], t.train_dataset[1]])
    # padding-free: the two examples end to end in one row, each cut to 3 tokens
    assert batch["input_ids"].shape == (1, 6)
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]


def test_padding_free_given_in_the_config_is_cut_to_max_length(tiny_model, tokenizer, tmp_path):
    """padding_free=True with a max_length, given in the config: TRL 1.x refuses the pair; meepsquared cuts instead."""
    long = Dataset.from_dict({"text": ["the hare runs fast and the sloth sleeps"] * 2})
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1, max_length=3,
                     padding_free=True)
    t = SFTTrainer(model=model, args=args, train_dataset=long, processing_class=tokenizer)
    assert t.args.padding_free is True and t.args.max_length == 3
    batch = t.data_collator([t.train_dataset[0], t.train_dataset[1]])
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]


def test_grpo_takes_its_settings_from_the_model(tiny_model, tokenizer, tmp_path):
    from trl import GRPOConfig, GRPOTrainer

    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    prompts = Dataset.from_dict({"prompt": ["the hare", "the sloth"]})
    args = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1,
                      per_device_train_batch_size=2, num_generations=2, max_completion_length=4)
    # reward_funcs by position, as GRPOTrainer's signature puts it before args
    t = GRPOTrainer(model, lambda completions, **kw: [0.0] * len(completions), args=args,
                    train_dataset=prompts, processing_class=tokenizer)
    assert t.args.gradient_checkpointing is True
    assert t.args.use_liger_kernel is False


def test_fsdp_refuses_bitsandbytes_optimizers_and_keeps_bf16_weights(tiny_model, tokenizer):
    from meepsquared._trl import _fsdp_settings

    args = SFTConfig(output_dir="/tmp/x", use_cpu=True, report_to=[], optim="adamw_8bit")
    with pytest.raises(ValueError, match="adamw_torch_fused"):
        _fsdp_settings(args, {"optim"})
    args = SFTConfig(output_dir="/tmp/x", use_cpu=True, report_to=[], optim="adamw_torch")
    _fsdp_settings(args, {"optim"})
    assert args.bf16 is False
    args.bf16 = True
    with pytest.raises(ValueError, match="FP32"):
        _fsdp_settings(args, {"optim", "bf16"})
