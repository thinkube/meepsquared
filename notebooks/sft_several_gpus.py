# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA or QLoRA fine-tuning with meepsquared on the GPUs of one machine; notebook 05 runs it.

    torchrun --nproc_per_node 2 sft_several_gpus.py --mode ddp  --model M --output out/05   # a copy per GPU
    torchrun --nproc_per_node 2 sft_several_gpus.py --mode fsdp --model M --output out/05   # the layers sharded
    python                      sft_several_gpus.py --mode split --model M --output out/05  # the layers spread, one process

The data is conversations as {"messages": [...]}: a Hub dataset, a Parquet
file or a JSONL file. Each process trains --batch conversations per step
(the whole batch, in split mode). The adapter is saved by the first process.

A training job (m2.TrainingJob) trains with DDP under torchrun, but not
with FSDP2 or with the layers spread over the GPUs, so this script uses
meepsquared's lower-level calls for all three modes.
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["ddp", "fsdp", "split"], required=True)
parser.add_argument("--model", required=True, help="a model id, or the path of a mirrored model")
parser.add_argument("--qlora", action="store_true", help="a 4-bit base")
parser.add_argument("--dataset", default="HuggingFaceH4/no_robots")
parser.add_argument("--rows", type=int, help="conversations to use; all of them without it")
parser.add_argument("--epochs", type=float, help="epochs over the rows; --steps without it")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--batch", type=int, default=8, help="conversations per GPU per step")
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--output", required=True, help="the folder for the adapter")
parser.add_argument("--report-to", default="none", help="where the trainer reports its metrics: none, or mlflow")
a = parser.parse_args()

import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

import meepsquared as m2  # noqa: E402

rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))


def conversations(source, rows=None):
    if source.endswith(".jsonl"):
        data = Dataset.from_list([json.loads(line) for line in open(source) if line.strip()])
    elif source.endswith(".parquet"):
        data = load_dataset("parquet", data_files=source, split="train")
    else:
        data = load_dataset(source, split="train")
    if "messages" not in data.column_names:
        raise ValueError(f"{source} has no 'messages' column; its columns are {data.column_names}")
    if rows is not None:
        if len(data) < rows:
            raise ValueError(f"{source} has {len(data)} conversations; --rows asks for {rows}")
        data = data.select(range(rows))
    return data.map(lambda r: {"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]]},
                    remove_columns=data.column_names)


load = {}
if a.mode == "split":
    load["device_map"] = "balanced"  # the layers spread over the GPUs
elif a.mode == "fsdp" and not a.qlora:
    load["device_map"] = {"": "cpu"}  # FSDP2 gives each GPU its share from the CPU copy
if a.qlora:
    load.update(load_in_4bit=True, **({"quant_storage": torch.bfloat16} if a.mode == "fsdp" else {}))
model, tokenizer = m2.MeepLanguageModel.from_pretrained(a.model, max_seq_length=a.max_length, **load)
# "auto" checkpointing measures one GPU's memory, which says nothing about a sharded or spread model: every layer there
model = m2.MeepLanguageModel.get_peft_model(model, r=a.rank, lora_alpha=a.rank,
                                              use_gradient_checkpointing="auto" if a.mode == "ddp" else True)
data = conversations(a.dataset, a.rows)

config = dict(output_dir=a.output, per_device_train_batch_size=a.batch, learning_rate=a.learning_rate,
              lr_scheduler_type="cosine", warmup_steps=max(1, a.steps // 20) if not a.epochs else 10, logging_steps=1, save_strategy="no",
              report_to=[] if a.report_to == "none" else [a.report_to], max_length=a.max_length, seed=0,
              **({"num_train_epochs": a.epochs} if a.epochs else {"max_steps": a.steps}))
if a.mode == "fsdp":
    # bitsandbytes' optimizers cannot update sharded parameters
    config.update(optim="adamw_torch_fused", **m2.fsdp_config(model))
else:
    config.update(optim="adamw_8bit", bf16=True)
trainer = SFTTrainer(model=model, args=SFTConfig(**config), train_dataset=data, processing_class=tokenizer)
result = trainer.train()
if rank == 0:
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"RESULT {json.dumps({'mode': a.mode, 'processes': world, 'steps': result.global_step, 'train_runtime_s': round(result.metrics['train_runtime'], 1), 'loss_first': losses[0], 'loss_last': losses[-1], 'peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 1)})}", flush=True)
# every process calls the save and the first one writes; under FSDP the adapter is gathered from all of them first
model.save_pretrained_merged(f"{a.output}/adapter", tokenizer, save_method="lora")
if rank == 0:
    print(f"adapter saved to {a.output}/adapter", flush=True)
