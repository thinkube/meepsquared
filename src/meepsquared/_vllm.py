# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""GRPO generation through vLLM: the weights reach vLLM under their checkpoint names.

TRL starts vLLM on the model's files, so vLLM builds the architecture the
checkpoint names, and after each step TRL sends it the trained weights by the
names they have in the training model. The two can differ: Transformers loads
a text model out of a multimodal checkpoint (Qwen3.5) by renaming its weights
(model.language_model.X becomes model.X), and vLLM, holding the multimodal
architecture, knows only the checkpoint's names. Each name TRL sends is
renamed back as Transformers' own save_pretrained renames it, with the
renamings recorded on the model when it was loaded. A weight that a converter
(a merge or split of tensors) produced cannot be renamed alone, and is refused.

TRL 1.x keeps this in trl.generation.vllm_generation; meepsquared's vLLM support
needs TRL 1.16 or later.

The CUDA runtime FlashInfer finds: FlashInfer (vLLM's dependency) takes the
first line of /proc/self/maps that holds "libcudart" as the CUDA runtime, and
tilelang (another dependency of vLLM) maps a stub of that name that lacks
most of the runtime; loaded after PyTorch's real runtime, the stub sits at a
lower address and comes first. FlashInfer looks the runtime up while its
module flashinfer.comm.cuda_ipc is being imported, so nothing can be patched
on the module: an import hook gives that module, as it loads, an open() that
reads /proc/self/maps without stub libraries. The hook is installed when
meepsquared is imported and FlashInfer is installed but not yet imported.
"""

import builtins
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import os
import sys

_PATCHED = "_meepsquared_checkpoint_names"


def _reverse_transforms(model):
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    conversions = getattr(model, "_weight_conversions", None) or []
    reverse = [conversion.reverse_transform() for conversion in conversions[::-1]]
    return ([t for t in reverse if isinstance(t, WeightRenaming)],
            [t for t in reverse if isinstance(t, WeightConverter)])


def _base_model(model):
    model = getattr(model, "module", model)
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def checkpoint_name(model, name: str) -> str:
    """The name the weight has in the checkpoint the model was loaded from."""
    from transformers.core_model_loading import rename_source_key

    renamings, converters = _reverse_transforms(model)
    if not renamings and not converters:
        return name
    renamed, converter_pattern = rename_source_key(name, renamings, converters, reverse=True)
    if converter_pattern is not None:
        raise NotImplementedError(
            f"{name} was converted from several checkpoint tensors when the model was loaded "
            "(a merge or split); meepsquared cannot send it to vLLM by name"
        )
    return renamed


MAPS = "/proc/self/maps"
FLASHINFER_CUDART_MODULE = "flashinfer.comm.cuda_ipc"


def is_stub_library(maps_line: str) -> bool:
    """Whether a line of /proc/self/maps maps a stub shared library (libcudart_stub.so, say)."""
    if "/" not in maps_line:
        return False
    filename = os.path.basename(maps_line[maps_line.index("/"):].strip())
    return ".so" in filename and "stub" in filename.rpartition(".so")[0]


def without_stub_libraries(maps_lines):
    """The lines of /proc/self/maps that do not map a stub library."""
    return [line for line in maps_lines if not is_stub_library(line)]


def open_without_stub_libraries(file, *args, **kwargs):
    """open(), except that /proc/self/maps reads without the lines of stub libraries."""
    if file == MAPS and not args and not kwargs:
        with builtins.open(MAPS) as maps:
            return io.StringIO("".join(without_stub_libraries(maps)))
    return builtins.open(file, *args, **kwargs)


class _StubFreeMapsFinder(importlib.abc.MetaPathFinder):
    """Loads one module with open() replaced by open_without_stub_libraries (see the module)."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    def find_spec(self, fullname, path, target=None):
        if fullname != self.module_name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_module_with_stub_free_maps(module):
            module.open = open_without_stub_libraries
            exec_module(module)

        loader.exec_module = exec_module_with_stub_free_maps
        return spec


def install_stub_free_maps_for(module_name: str) -> bool:
    """Install the import hook for module_name, once. False when the module is already imported: nothing to do."""
    if module_name in sys.modules:
        return False
    if not any(isinstance(f, _StubFreeMapsFinder) and f.module_name == module_name for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFreeMapsFinder(module_name))
    return True


def patch_flashinfer_library_lookup() -> None:
    """FlashInfer's CUDA runtime lookup skips a stub (see the module). Nothing happens without FlashInfer."""
    if importlib.util.find_spec("flashinfer") is None:
        return
    install_stub_free_maps_for(FLASHINFER_CUDART_MODULE)


def patch_vllm_generation() -> None:
    patch_flashinfer_library_lookup()
    try:
        from trl.generation.vllm_generation import VLLMGeneration
    except ImportError:
        return
    fix = VLLMGeneration._fix_param_name_to_vllm
    if getattr(fix, _PATCHED, False):
        return

    @functools.wraps(fix)
    def fix_to_checkpoint_name(self, name, extra_prefixes=None):
        name = fix(self, name, extra_prefixes=extra_prefixes)
        return checkpoint_name(_base_model(self.model), name)

    setattr(fix_to_checkpoint_name, _PATCHED, True)
    VLLMGeneration._fix_param_name_to_vllm = fix_to_checkpoint_name
