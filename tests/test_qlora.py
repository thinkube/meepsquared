# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest
import torch

from meepsquared import MeepLanguageModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("bitsandbytes") is None,
                                reason="4-bit weights need bitsandbytes and a GPU")


def test_quant_storage_packs_the_4bit_weights_in_the_given_dtype(tiny_model, tokenizer, tmp_path):
    import bitsandbytes as bnb

    del tiny_model._meepsquared
    tiny_model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    model, _ = MeepLanguageModel.from_pretrained(str(tmp_path), load_in_4bit=True, attn_implementation="sdpa",
                                                 quant_storage=torch.bfloat16, liger=False)
    layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert layers and all(m.weight.dtype == torch.bfloat16 for m in layers)
