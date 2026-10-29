# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

# Run by test_ddp.py under torchrun with 2 CPU processes: one training step of the
# tiny model, then each process prints what its trainer was set up with.
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from meepsquared import MeepLanguageModel

words = ["<unk>", "<pad>", "</s>", "the", "hare", "runs", "fast", "and", "sloth", "sleeps"]
backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
torch.manual_seed(0)
model = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64))
model._meepsquared = {"padding_free": False, "liger": False}
model = MeepLanguageModel.get_peft_model(model, r=4, use_gradient_checkpointing=True)
args = SFTConfig(output_dir=os.environ["OUT"], use_cpu=True, bf16=False, report_to=[], max_steps=1,
                 per_device_train_batch_size=1, save_strategy="no")
trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                     train_dataset=Dataset.from_dict({"text": ["the hare runs fast", "and the sloth sleeps"] * 2}))
trainer.train()
# A file per process: the processes' printed lines can interleave
Path(os.environ["OUT"], f"result-{os.environ['RANK']}.json").write_text(json.dumps(
    {"rank": int(os.environ["RANK"]), "world": int(os.environ["WORLD_SIZE"]),
     "find_unused": trainer.args.ddp_find_unused_parameters, "steps": trainer.state.global_step}))
