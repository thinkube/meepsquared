# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from meepsquared import MeepLanguageModel, _checkpointing


def flags(model):
    return [layer.gradient_checkpointing for layer in _checkpointing.layers(model)]


@pytest.mark.parametrize("setting, expected", [(True, 2), (False, 0), (None, 0), ("auto", 2), ("unsloth", 2), ("memory", 2),
                                               ("selective", 2), (1, 1), (5, 2), (0.5, 1), (0.0, 0), (1.0, 2)])
def test_count(setting, expected):
    assert _checkpointing.count(setting, 2) == expected


@pytest.mark.parametrize("setting", ["some", -1, 1.5, 2.5])
def test_count_refuses_other_settings(setting):
    with pytest.raises(ValueError):
        _checkpointing.count(setting, 2)


def test_a_number_of_layers_checkpoints_the_first_ones(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=1)
    assert flags(model) == [True, False]
    assert model.base_model.model.is_gradient_checkpointing
    MeepLanguageModel.for_inference(model)
    assert flags(model) == [False, False]
    MeepLanguageModel.for_training(model)
    assert flags(model) == [True, False]


def test_auto_checkpoints_every_layer_until_measured(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing="auto")
    assert flags(model) == [True, True]
    model._meepsquared_checkpointed_layers = 1
    MeepLanguageModel.for_training(model)
    assert flags(model) == [True, False]
    model._meepsquared_checkpointed_layers = _checkpointing.Recompute(whole=0, selective=1)
    MeepLanguageModel.for_training(model)
    assert _checkpointing.kinds(model) == ["selective", "none"]


def test_layers_to_checkpoint():
    gib = 2**30
    # 100 GiB free of 128: 15% kept back leaves 80.8 GiB, room for 40 layers of 2 GiB
    assert _checkpointing.layers_to_checkpoint(100 * gib, 128 * gib, 2 * gib, 32) == 0
    assert _checkpointing.layers_to_checkpoint(100 * gib, 128 * gib, 4 * gib, 32) == 12
    assert _checkpointing.layers_to_checkpoint(10 * gib, 128 * gib, 4 * gib, 32) == 32
    assert _checkpointing.layers_to_checkpoint(100 * gib, 128 * gib, 0, 32) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures GPU memory")
def test_one_layer_holds_more_than_none(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model.cuda(), r=4, use_gradient_checkpointing="auto")
    per_layer = _checkpointing.layer_activation_bytes(model, batch=2, length=16)
    assert per_layer > 0
    assert flags(model) == [True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures GPU memory")
def test_a_selective_layer_holds_less_than_a_kept_one(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model.cuda(), r=4, use_gradient_checkpointing="auto")
    shape = _checkpointing.Shape(2, 16)
    figures = _checkpointing.measure_figures(model, shape)
    assert 0 < figures.per_selective < figures.per_layer
    assert set(figures.times) == {"none", "selective", "whole"}
    assert flags(model) == [True, True]


def test_meminfo_reads_available_and_total():
    text = "MemTotal:       127598544 kB\nMemFree:        36457492 kB\nMemAvailable:   94337588 kB\nCached:  57563764 kB\n"
    assert _checkpointing.meminfo_bytes(text) == (94337588 * 1024, 127598544 * 1024)
    with pytest.raises(ValueError):
        _checkpointing.meminfo_bytes("MemTotal: 1 kB\n")



@pytest.mark.skipif(not torch.cuda.is_available(), reason="the allocator's cache is measured on a GPU")
def test_the_allocators_free_cache_counts_as_free():
    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()
    before, total = _checkpointing.free_memory(device)
    block = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    del block  # freed to the allocator's cache, not to the system
    torch.cuda.synchronize()
    assert _checkpointing.cached_bytes(device) >= 256 << 20
    after, _ = _checkpointing.free_memory(device)
    assert after >= before - (16 << 20)  # what the cache holds is still free to this process


def test_only_the_decoders_layers_count():
    from transformers.modeling_layers import GradientCheckpointingLayer

    class Layer(GradientCheckpointingLayer):
        pass

    class Multimodal(torch.nn.Module):
        """A vision tower ahead of the decoder in module order, as Gemma 4 has it."""

        def __init__(self):
            super().__init__()
            self.vision_tower = torch.nn.ModuleList([Layer(), Layer()])
            self.language_model = torch.nn.ModuleList([Layer(), Layer(), Layer()])

        def get_decoder(self):
            return self.language_model

    model = Multimodal()
    assert _checkpointing.layers(model) == list(model.language_model)
