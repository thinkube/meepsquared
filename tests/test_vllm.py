# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.conversion_mapping import get_model_conversion_mapping

from meepsquared._vllm import checkpoint_name


def test_a_text_model_out_of_a_multimodal_checkpoint_gets_its_checkpoint_names():
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
                               num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
                               linear_num_key_heads=2, linear_num_value_heads=4)
    model = Qwen3_5ForCausalLM(config)
    # What from_pretrained records when it loads the text model out of the multimodal checkpoint
    model._weight_conversions = get_model_conversion_mapping(model)
    assert checkpoint_name(model, "model.layers.0.linear_attn.in_proj_qkv.weight") == \
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert checkpoint_name(model, "model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert checkpoint_name(model, "lm_head.weight") == "lm_head.weight"


def test_a_model_loaded_without_renamings_keeps_its_names(tiny_model):
    tiny_model._weight_conversions = None
    for name, _ in tiny_model.named_parameters():
        assert checkpoint_name(tiny_model, name) == name


MAPS = [
    "ffff8000-ffff9000 r-xp 00000000 00:00 0  [vdso]",
    "e000-f000 r-xp 00000000 fd:01 1  /venv/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so\n",
    "a000-b000 r-xp 00000000 fd:01 2  /usr/local/cuda/lib64/libcudart.so.13\n",
    "c000-d000 r-xp 00000000 fd:01 3  /opt/stubs/libcuda.so\n",
]


def test_the_stub_lines_are_dropped_and_the_real_runtime_kept():
    """What FlashInfer reads: tilelang's stub comes first in the maps, the real runtime after it."""
    from meepsquared._vllm import without_stub_libraries

    lines = without_stub_libraries(MAPS)
    assert [l for l in lines if "libcudart" in l] == [MAPS[2]]
    assert MAPS[0] in lines and MAPS[3] in lines, "a directory or an unrelated library named stubs is not a stub"


def test_flashinfer_gets_the_maps_without_stubs_as_its_module_loads(tmp_path, monkeypatch):
    """A module named like FlashInfer's, loaded through the hook, reads /proc/self/maps without stub libraries."""
    from meepsquared import _vllm

    pkg = tmp_path / "fakeinfer" / "comm"
    pkg.mkdir(parents=True)
    (tmp_path / "fakeinfer" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "cuda_ipc.py").write_text(
        "def first_cudart():\n"
        "    with open('/proc/self/maps') as f:\n"
        "        return next((l for l in f if 'libcudart' in l), None)\n"
        "FIRST = first_cudart()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_vllm.builtins, "open", _fake_open(MAPS, _vllm.builtins.open))
    assert _vllm.install_stub_free_maps_for("fakeinfer.comm.cuda_ipc")

    import fakeinfer.comm.cuda_ipc as cuda_ipc

    assert "libcudart.so.13" in cuda_ipc.FIRST and "stub" not in cuda_ipc.FIRST
    assert cuda_ipc.first_cudart() == cuda_ipc.FIRST


def test_the_hook_is_not_installed_for_an_imported_module():
    import sys

    from meepsquared._vllm import install_stub_free_maps_for

    assert install_stub_free_maps_for("os") is False
    assert not any(getattr(f, "module_name", None) == "os" for f in sys.meta_path)


def _fake_open(maps_lines, real_open):
    import io

    def fake_open(file, *args, **kwargs):
        if file == "/proc/self/maps":
            return io.StringIO("".join(maps_lines))
        return real_open(file, *args, **kwargs)

    return fake_open
