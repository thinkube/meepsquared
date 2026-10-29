# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from meepsquared import MeepLanguageModel, _checkpointing, _compile


def test_the_layer_class_is_compiled_once(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, compile_layers=True)
    assert model._meepsquared_compile_layers is True
    layer_cls = type(_checkpointing.layers(model)[0])
    original = layer_cls.forward
    try:
        assert _compile.apply(model, backend="eager") == 2
        compiled = layer_cls.forward
        assert compiled is not original
        assert _compile.apply(model, backend="eager") == 2
        assert layer_cls.forward is compiled
        ids = torch.tensor([[3, 4, 5, 6]])
        out = model(input_ids=ids, labels=ids)
        assert torch.isfinite(out.loss)
    finally:
        _compile.uncompiled(model)
    assert layer_cls.forward is original


def test_bitsandbytes_layers_stay_outside_the_compiled_region(tiny_model):
    bnb = pytest.importorskip("bitsandbytes")
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, compile_layers=True)
    try:
        _compile.apply(model, backend="eager")
        assert getattr(bnb.nn.Linear4bit.forward, "_meepsquared_outside", False)
    finally:
        _compile.uncompiled(model)


def test_the_recompile_limit_lets_every_kind_of_layer_compile(tiny_model):
    """Dynamo's limit is raised to two versions per layer, and never lowered."""
    import torch._dynamo.config as dynamo_config

    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, compile_layers=True)
    before = dynamo_config.recompile_limit, dynamo_config.accumulated_recompile_limit
    try:
        dynamo_config.recompile_limit = 1
        _compile.apply(model, backend="eager")
        assert dynamo_config.recompile_limit == _compile.recompile_limit_for(2) == 4
        dynamo_config.recompile_limit = 1000
        _compile.apply(model, backend="eager")
        assert dynamo_config.recompile_limit == 1000
    finally:
        _compile.uncompiled(model)
        dynamo_config.recompile_limit, dynamo_config.accumulated_recompile_limit = before
