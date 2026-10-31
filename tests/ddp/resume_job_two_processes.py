# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

# Run by test_ddp.py under torchrun with 2 CPU processes: a TrainingJob of four steps over two epochs,
# stopped by the stop file after step 3, within the second epoch, then resumed by a new job. Each process
# records the batches it trains in both sessions, and prints whether they are its share of the
# micro-batches meepsquared laid out, as an uninterrupted training trains them.
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, TrainerCallback

import meepsquared as m2
from meepsquared import _batches

words = ["<unk>", "<pad>", "</s>", "the", "hare", "runs", "fast", "and", "sloth", "sleeps"]
backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
data = Dataset.from_dict({"text": ["the hare runs fast", "and the sloth sleeps", "the sloth runs", "the hare sleeps",
                                   "the hare", "the sloth runs fast"]})
out = os.environ["OUT"]
stop = Path(out) / "STOP"
rank = int(os.environ.get("RANK", 0))


def job():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                         max_position_embeddings=64))
    model._meepsquared = {"padding_free": True, "liger": False}
    # 4 examples per update, 2 on each process, one per micro-batch: 3 micro-batches an epoch on each
    # process, 2 updates an epoch; 4 steps take two epochs
    return m2.TrainingJob(model=model, tokenizer=tokenizer, data=data, steps=4, output=out,
                          preprocessed=os.environ.get("PREPROCESSED"),
                          recipe=m2.Recipe(effective_batch=4, optimizer="adamw_torch"),
                          overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0))


if os.environ.get("PREPROCESS_ONLY"):
    # Run alone, before the launch: the data preprocessed, its folder printed
    print(job().preprocess(os.environ["PREPROCESS_ONLY"]).folder)
    raise SystemExit(0)


class StopAfter3(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 3:
            stop.touch()


def recorded(training):
    trainer = training.trainer()
    seen = []
    step = trainer.training_step

    def recording(model, inputs, *args, **kwargs):
        seen.append(inputs["input_ids"].tolist())
        return step(model, inputs, *args, **kwargs)

    trainer.training_step = recording
    return trainer, seen


first = job()
trainer, seen_first = recorded(first)
trainer.add_callback(StopAfter3())
paused = first.train(stop_file=stop)
dist.barrier()
if rank == 0:
    stop.unlink()
dist.barrier()
second = job()
_, seen_second = recorded(second)
resumed = second.train(resume=True, stop_file=stop)

laid = trainer._meepsquared_layout
shares = [_batches.share(batches, laid.batch_size, laid.loading.drop_last, laid.loading, rank)[1]
          for batches in laid.epochs]
mine, steps = _batches.trained_by(shares, laid.per_epoch, 2, 4)
expected = [trainer.data_collator([trainer.train_dataset[i] for i in batch])["input_ids"].tolist() for batch in mine]
# A file per process: the processes' printed lines can interleave
Path(out, f"result-{rank}.json").write_text(json.dumps(
    {"rank": rank, "same": seen_first + seen_second == expected, "first": len(seen_first),
     "second": len(seen_second), "paused": paused.paused, "paused_at": paused.steps, "steps": resumed.steps,
     "planned": steps}))
