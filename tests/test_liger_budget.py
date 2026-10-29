# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import os

import pytest


@pytest.fixture
def fresh(monkeypatch):
    """meepsquared._liger with no patch applied, over a recording stand-in for Liger's chunked forward."""
    import liger_kernel.ops.fused_linear_cross_entropy as flce

    import meepsquared._liger as module

    calls = []
    monkeypatch.setattr(flce, "fused_linear_cross_entropy_forward", lambda *a, **kw: calls.append(kw))
    module = importlib.reload(module)
    return module, flce, calls


def test_the_budget_reaches_every_call(fresh):
    module, flce, calls = fresh
    module.set_loss_chunk_budget(32)
    flce.fused_linear_cross_entropy_forward(chunk_mem_const=1)
    module.set_loss_chunk_budget(8)
    flce.fused_linear_cross_entropy_forward()
    assert [c["chunk_mem_const"] for c in calls] == [32, 8]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "32"])
def test_the_budget_is_a_positive_integer(fresh, bad):
    module, _, _ = fresh
    with pytest.raises(ValueError):
        module.set_loss_chunk_budget(bad)


def test_a_model_type_without_patches_is_refused(fresh, tiny_model):
    module, _, _ = fresh
    tiny_model.config.model_type = "no_such_model"
    with pytest.raises(NotImplementedError):
        module.prepare(tiny_model, 32)


def test_rope_runs_on_ligers_triton_kernel(tiny_model, monkeypatch):
    from meepsquared import _liger

    monkeypatch.delenv("LIGER_KERNEL_IMPL_ROPE", raising=False)
    _liger.prepare(tiny_model, 1)
    assert os.environ["LIGER_KERNEL_IMPL_ROPE"] == "nvidia-triton"


def test_a_rope_kernel_chosen_by_the_user_is_kept(tiny_model, monkeypatch):
    from meepsquared import _liger

    monkeypatch.setenv("LIGER_KERNEL_IMPL_ROPE", "nvidia-cutedsl")
    _liger.prepare(tiny_model, 1)
    assert os.environ["LIGER_KERNEL_IMPL_ROPE"] == "nvidia-cutedsl"
