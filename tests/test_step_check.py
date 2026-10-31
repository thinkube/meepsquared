# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The plan counts the trainable parameters' memory, and checks the timed steps' real peak against the budget."""

from types import SimpleNamespace

import pytest
import torch

from meepsquared import MeepLanguageModel, _checkpointing, _sessions, job as job_module

GIB = 2**30


@pytest.mark.parametrize("optimizer, per_parameter", [
    ("adamw_8bit", 2), ("paged_adamw_8bit", 2), ("adamw_torch", 8), ("adamw_hf", 8), ("lion_8bit", 1),
    ("lion_32bit", 4), ("sgd", 0), ("adafactor", 4),
])
def test_optimizer_state_bytes_per_parameter(optimizer, per_parameter):
    assert _checkpointing.optimizer_state_bytes(optimizer) == per_parameter


def test_trainable_bytes_count_the_gradients_and_the_optimizer_state(tiny_model):
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in trainable)
    assert count > 0 and all(p.dtype == torch.float32 for p in trainable)
    assert _checkpointing.trainable_bytes(model, "adamw_8bit") == count * (4 + 2)
    assert _checkpointing.trainable_bytes(model, "adamw_torch") == count * (4 + 8)


def timing(peak):
    return _sessions.Timing(1.0, None, None, 0, None, peak)


def checked(monkeypatch, tiny_model, peaks, layout, given=False, per_layer=GIB):
    """_timed_steps_fitting with a budget of 10 GiB and timed steps peaking as `peaks` says, one per timing."""
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    layers = len(_checkpointing.layers(model))
    peaks = list(peaks)
    timed = []

    def fake_time_steps(trainer, shape, accumulation, steps, eval_shape=None):
        timed.append(getattr(model, "_meepsquared_gradient_checkpointing", None))
        peak = peaks.pop(0)
        if peak == "oom":
            raise torch.OutOfMemoryError("out of memory")
        return timing(peak)

    monkeypatch.setattr(_sessions, "time_steps", fake_time_steps)
    monkeypatch.setattr(_checkpointing, "peak_budget", lambda device, margin: 10 * GIB)
    _checkpointing.set_checkpointed(model, layout)
    model._meepsquared_gradient_checkpointing = layout
    shape = _checkpointing.Shape(2, 8)
    result = job_module._timed_steps_fitting(None, model, shape, 1, 3, None, layout, SimpleNamespace(per_layer=per_layer),
                                             layers, torch.device("cpu"), given, 0)
    return result, timed, layers


def test_steps_within_the_budget_change_nothing(monkeypatch, tiny_model):
    (result, layout, corrected), timed, layers = checked(monkeypatch, tiny_model, [9 * GIB], _checkpointing.Recompute(0, 0))
    assert result.peak_gpu_allocated_bytes == 9 * GIB and layout == _checkpointing.Recompute(0, 0) and corrected == ""
    assert len(timed) == 1


def test_a_peak_over_the_budget_adds_layers_by_the_overshoot_then_looks_for_the_fewest(monkeypatch, tiny_model):
    # 2 GiB over at 1 GiB a layer: two more layers recompute whole (the tiny model has two) and fit; the layout
    # between, one layer, is timed too and does not fit, so two it is
    (result, layout, corrected), timed, layers = checked(
        monkeypatch, tiny_model, [12 * GIB, 8 * GIB, 11 * GIB], _checkpointing.Recompute(0, 0))
    assert layout == _checkpointing.Recompute(2, 0)
    assert result.peak_gpu_allocated_bytes == 8 * GIB
    assert "2.00 GiB over the budget of 10.00 GiB" in corrected and "timed again" in corrected
    assert "fit, the timed steps peaking at 8.00 GiB" in corrected
    assert [t.whole for t in timed] == [0, 2, 1]
    assert all(layer.gradient_checkpointing for layer in _checkpointing.layers(tiny_model)[:layout.whole])


def test_the_fewest_layers_that_fit_are_kept(monkeypatch, tiny_model):
    # Two layers fit, then one fits too: one is kept, and the model is laid out for it
    (result, layout, corrected), timed, layers = checked(
        monkeypatch, tiny_model, [12 * GIB, 8 * GIB, 9 * GIB], _checkpointing.Recompute(0, 0))
    assert layout == _checkpointing.Recompute(1, 0) and result.peak_gpu_allocated_bytes == 9 * GIB
    assert [t.whole for t in timed] == [0, 2, 1]
    assert [bool(layer.gradient_checkpointing) for layer in _checkpointing.layers(tiny_model)] == [True, False]


def test_an_out_of_memory_step_tries_halfway_to_every_layer(monkeypatch, tiny_model):
    (result, layout, corrected), timed, layers = checked(monkeypatch, tiny_model, ["oom", 7 * GIB], _checkpointing.Recompute(0, 0))
    assert layout == _checkpointing.Recompute(1, 0) and "ran out of memory" in corrected
    assert result.peak_gpu_allocated_bytes == 7 * GIB and [t.whole for t in timed] == [0, 1]


def test_out_of_memory_steps_end_with_every_layer_and_then_stop(monkeypatch, tiny_model):
    with pytest.raises(RuntimeError, match="every layer recomputed whole"):
        checked(monkeypatch, tiny_model, ["oom", "oom", "oom"], _checkpointing.Recompute(0, 0))


def test_steps_that_never_fit_stop_with_the_figures(monkeypatch, tiny_model):
    with pytest.raises(RuntimeError, match="does not fit"):
        checked(monkeypatch, tiny_model, [12 * GIB, 12 * GIB, 12 * GIB], _checkpointing.Recompute(0, 0), per_layer=GIB)


def test_a_layout_given_by_hand_is_not_changed(monkeypatch, tiny_model):
    with pytest.raises(RuntimeError, match="given by hand"):
        checked(monkeypatch, tiny_model, [12 * GIB], _checkpointing.Recompute(1, 0), given=True)
