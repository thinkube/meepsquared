# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

from meepsquared import _linear_attention


def test_fla_without_its_kernels_is_not_available(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_fla_ops(name, *args, **kwargs):
        if name.startswith("fla.ops"):
            raise ModuleNotFoundError("No module named 'fla.ops'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_fla_ops)
    assert _linear_attention.available() is False
