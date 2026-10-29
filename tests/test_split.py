# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A model spread over two GPUs trains as on one, Triton kernels included."""

import pytest
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from meepsquared import MeepLanguageModel

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
SPREAD = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.layers.0": 0, "model.layers.1": 1,
          "model.norm": 1, "lm_head": 1}


def first_loss(path, tokenizer, device_map, tmp_path):
    model, _ = MeepLanguageModel.from_pretrained(str(path), dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map=device_map)
    model = MeepLanguageModel.get_peft_model(model, r=4, use_gradient_checkpointing=True, random_state=1)
    args = SFTConfig(output_dir=str(tmp_path / str(len(set(device_map.values())))), report_to=[], max_steps=1,
                     per_device_train_batch_size=2, logging_steps=1, save_strategy="no", seed=0)
    trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                         train_dataset=Dataset.from_dict({"text": ["the hare runs fast and the sloth sleeps"] * 2}))
    trainer.train()
    return trainer.state.log_history[0]["loss"], {p.device for p in model.parameters()}


def test_spread_over_two_gpus_trains_as_on_one(tiny_model, tokenizer, tmp_path):
    del tiny_model._meepsquared
    tiny_model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    one, devices_one = first_loss(tmp_path / "model", tokenizer, {"": 0}, tmp_path)
    two, devices_two = first_loss(tmp_path / "model", tokenizer, SPREAD, tmp_path)
    assert devices_one == {torch.device("cuda", 0)}
    assert devices_two == {torch.device("cuda", 0), torch.device("cuda", 1)}
    assert two == pytest.approx(one, rel=1e-2)
