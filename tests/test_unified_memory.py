# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForCausalLM

from meepsquared import _unified_memory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not _unified_memory.shares_system_memory(),
                                reason="the shard-by-shard load is for a GPU that shares the system's memory")


def test_a_sharded_checkpoint_loads_shard_by_shard_to_the_same_weights(tiny_model, tmp_path, monkeypatch):
    tiny_model.save_pretrained(tmp_path, max_shard_size="20KB")
    shards = list(tmp_path.glob("*.safetensors"))
    assert len(shards) > 1
    read = []
    real = _unified_memory._ShardOnGpu.__init__

    def counting(self, safe_open, filename, device):
        read.append(filename)
        real(self, safe_open, filename, device)

    monkeypatch.setattr(_unified_memory._ShardOnGpu, "__init__", counting)
    with _unified_memory.shards_to_gpu():
        loaded = AutoModelForCausalLM.from_pretrained(tmp_path, device_map={"": 0})
    assert sorted(map(str, read)) == sorted(map(str, shards))
    expected = tiny_model.state_dict()
    for name, value in loaded.state_dict().items():
        assert value.device.type == "cuda"
        torch.testing.assert_close(value.cpu(), expected[name], msg=name)
