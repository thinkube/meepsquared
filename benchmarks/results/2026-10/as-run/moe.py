# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA on a mixture of experts, its routed experts included, with llebre.

    python moe.py --model Qwen/Qwen3.6-35B-A3B --data examples.jsonl

LoRA rank 16 (alpha 16) on full attention, the gated DeltaNet projections,
the shared expert and every routed expert (target_parameters), 32 examples
per step, padding-free, 8-bit AdamW; the loss on the answers only. --data is
a JSONL file of {"prompt": ..., "answer": ...}, each prompt a user turn with
thinking off; without it, the first user and assistant turns of no_robots.
--peft-experts trains the experts as PEFT computes them, every expert's full
weight at every pass, for comparison. --fp8 holds the frozen weights, the
expert stacks included, as FP8 (load_in_fp8).
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--source", choices=["huggingface", "thinkube"], default="huggingface", help="where the model comes from: Hugging Face, or the cluster's mirror on Thinkube")
parser.add_argument("--data", help="a JSONL file of prompt and answer; no_robots without it")
parser.add_argument("--start", type=int, default=0, help="the first line of --data to use")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--peft-experts", action="store_true")
parser.add_argument("--fp8", action="store_true", help="the frozen weights held as FP8, the expert stacks included")
parser.add_argument("--checkpointing", default="auto",
                    help="gradient checkpointing: auto, true (every layer), false, a number of layers or a fraction")
parser.add_argument("--label", help="the arm's name in the result; llebre, llebre-fp8 or llebre-peft-experts without it")
parser.add_argument("--profile", help="a file for the GPU time per operation of the step after --warmup; a mixture of experts has too many operations per step for more")
parser.add_argument("--results", default="results/moe.jsonl")
parser.add_argument("--progress")
parser.add_argument("--memory-snapshot", help="a file for PyTorch's allocation history, written at each new allocated peak, to see what is alive there")
a = parser.parse_args()

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

import llebre as lb  # noqa: E402
from llebre import _checkpointing  # noqa: E402
from common import Clock, Profile, chat_dataset, memory_snapshot, model_location, peak_gib, record, setting  # noqa: E402

model, tokenizer = lb.LlebreModel.from_pretrained(model_location(a.model, a.source), max_seq_length=a.max_length,
                                                  dtype=torch.bfloat16, load_in_fp8=a.fp8)
model = lb.LlebreModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                    "gate_proj", "up_proj", "down_proj"],
    target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    use_gradient_checkpointing=setting(a.checkpointing), random_state=0)
if a.peft_experts:
    for module in model.modules():
        if hasattr(module, "_llebre_peft_forward"):
            module.forward = module._llebre_peft_forward
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

tok = getattr(tokenizer, "tokenizer", tokenizer)
head, tail = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False).split("\x00")
end = tok.eos_token if "<|im_end|>" not in tok.get_vocab() else "<|im_end|>"
if a.data:
    pairs = []
    with open(a.data) as f:
        for i, line in enumerate(f):
            if i >= a.start:
                ex = json.loads(line)
                pairs.append((ex["prompt"], ex["answer"]))
            if len(pairs) == a.batch * a.steps:
                break
else:
    pairs = [(r["messages"][0]["content"], r["messages"][1]["content"])
             for r in chat_dataset("HuggingFaceH4/no_robots", a.batch * a.steps, alternating=True)]
rows = []
for prompt, answer in pairs:
    p = tok(head + prompt + tail, add_special_tokens=False)["input_ids"]
    t = tok(answer + end, add_special_tokens=False)["input_ids"]
    rows.append({"input_ids": (p + t)[:a.max_length], "labels": ([-100] * len(p) + t)[:a.max_length]})

clock = Clock(a.progress)
args = SFTConfig(output_dir="/tmp/llebre-moe", per_device_train_batch_size=a.batch, max_steps=a.steps,
                 learning_rate=1e-4, optim="adamw_8bit", lr_scheduler_type="constant", bf16=True, logging_steps=1,
                 save_strategy="no", report_to=[], max_length=a.max_length, completion_only_loss=False,
                 loss_type="nll", remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
                 include_num_input_tokens_seen="non_padding", dataloader_num_workers=0, seed=0)
trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok,
                     callbacks=[clock] + ([Profile(a.profile, a.warmup + 1, steps=1)] if a.profile else []))
torch.cuda.reset_peak_memory_stats()
with memory_snapshot(a.memory_snapshot, trainer):
    trainer.train()
tokens_per_s, s_per_step = clock.rate(a.warmup, a.steps)
# How many layers trained recomputing their whole forward, recomputing selectively, or keeping everything
trained = _checkpointing.kinds(model)
record(a.results, {
    "benchmark": "moe", "arm": a.label or ("llebre-peft-experts" if a.peft_experts else "llebre-fp8" if a.fp8 else "llebre"),
    "model": a.model, "source": a.source, "fp8": a.fp8, "checkpointing": a.checkpointing,
    "checkpointed_layers": {kind: trained.count(kind) for kind in ("whole", "selective", "none")},
    "trainable_m": round(trainable / 1e6, 1), "batch": a.batch, "tokens_per_step": round(sum(len(r["input_ids"]) for r in rows) / a.steps),
    "s_per_step": round(s_per_step, 2), "s_by_step": clock.by_step(), "steps_per_hour": round(3600 / s_per_step), "tokens_per_s": round(tokens_per_s),
    "peak_gib": peak_gib(), "losses": [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h],
})
