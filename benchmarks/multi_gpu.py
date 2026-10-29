# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA and QLoRA with meepsquared on several GPUs: DDP, FSDP2, or the layers spread.

    torchrun --nproc_per_node 1 multi_gpu.py --mode ddp --model M     # one GPU, the reference
    torchrun --nproc_per_node 2 multi_gpu.py --mode ddp --model M     # DDP
    torchrun --nproc_per_node 2 multi_gpu.py --mode fsdp --model M    # FSDP2
    python multi_gpu.py --mode split --model M                        # the layers spread, one process

LoRA rank 16 on the seven projections, padding-free, no_robots conversations
of strict user/assistant turns, --batch examples per GPU per step (per step in
split mode). DDP uses 8-bit AdamW; FSDP2 uses PyTorch's fused AdamW, as
bitsandbytes' optimizers cannot update sharded parameters, and loads BF16
LoRA on the CPU and QLoRA on each GPU, where bitsandbytes quantizes. The
result is the tokens per second of all GPUs together and each GPU's peak memory.
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["ddp", "fsdp", "split"], required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--source", choices=["huggingface", "thinkube"], default="huggingface", help="where the model comes from: Hugging Face, or the cluster's mirror on Thinkube")
parser.add_argument("--qlora", action="store_true")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--results", default="results/multi_gpu.jsonl")
parser.add_argument("--progress")
a = parser.parse_args()

import torch  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

import meepsquared as m2  # noqa: E402
from common import Clock, chat_dataset, model_location, peak_gib, record  # noqa: E402

world = int(os.environ.get("WORLD_SIZE", 1))
load = {}
if a.mode == "split":
    load["device_map"] = "balanced"
elif a.mode == "fsdp" and not a.qlora:
    load["device_map"] = {"": "cpu"}
if a.qlora:
    load.update(load_in_4bit=True, **({"quant_storage": torch.bfloat16} if a.mode == "fsdp" else {}))
model, tokenizer = m2.MeepLanguageModel.from_pretrained(model_location(a.model, a.source), max_seq_length=a.max_length,
                                                          **load)
model = m2.MeepLanguageModel.get_peft_model(model, r=16, lora_alpha=16, use_gradient_checkpointing=True,
                                              random_state=0)
data = chat_dataset("HuggingFaceH4/no_robots", a.batch * world * a.steps, alternating=True)

config = dict(output_dir="/tmp/meepsquared-multi-gpu", per_device_train_batch_size=a.batch, max_steps=a.steps,
              learning_rate=1e-4, lr_scheduler_type="constant", logging_steps=1, save_strategy="no", report_to=[],
              max_length=a.max_length, seed=0, include_num_input_tokens_seen="non_padding", dataloader_num_workers=0)
if a.mode == "fsdp":
    config.update(optim="adamw_torch_fused", **m2.fsdp_config(model))
else:
    config.update(optim="adamw_8bit", bf16=True)
clock = Clock(a.progress)
trainer = SFTTrainer(model=model, args=SFTConfig(**config), train_dataset=data, processing_class=tokenizer,
                     callbacks=[clock])
torch.cuda.reset_peak_memory_stats()
trainer.train()
tokens_per_s, s_per_step = clock.rate(a.warmup, a.steps)
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
# each process sees its own GPU: the peaks of every rank, in rank order, make the row
peaks = peak_gib() if world == 1 else [None] * world
if world > 1:
    torch.distributed.all_gather_object(peaks, round(torch.cuda.max_memory_allocated() / 2**30, 1))
record(a.results, {
    "benchmark": "multi-gpu", "mode": a.mode, "processes": world, "model": a.model, "source": a.source,
    "method": "qlora" if a.qlora else "lora", "batch_per_gpu": a.batch, "tokens_per_s": round(tokens_per_s),
    "s_per_step": round(s_per_step, 2), "peak_gib": peaks, "loss_last5": round(sum(losses[-5:]) / 5, 4),
})
