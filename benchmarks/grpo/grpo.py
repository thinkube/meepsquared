# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""GRPO on the zebra task: meepsquared or Unsloth, with or without vLLM.

    python grpo.py --arm meepsquared --model Qwen/Qwen3.5-4B
    torchrun --nproc_per_node 1 grpo.py --arm meepsquared --model Qwen/Qwen3.5-4B --vllm

The settings of the zebra-grpo recipe at its demo scale: LoRA rank 16 on the
seven projections, 4 prompts of 8 completions of up to 800 tokens per step,
the recipe's sampling and optimizer. The recipe's per_device_train_batch_size
of 1 is refused by TRL, whose generation batch holds whole groups; Unsloth
raises it to num_generations, and here it is given as that. --vllm generates
through vLLM in the training process (colocate, TRL 1.16 or later), which
vLLM wants under a distributed launcher, torchrun even for one GPU. The result
is the first step, the mean of the later steps, the time of each step spent
generating, and the rewards.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["meepsquared", "unsloth"], required=True)
parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
parser.add_argument("--source", choices=["huggingface", "thinkube"], default="huggingface", help="where the model comes from: Hugging Face, or the cluster's mirror on Thinkube")
parser.add_argument("--vllm", action="store_true")
parser.add_argument("--vllm-memory", type=float, default=0.3)
parser.add_argument("--attn", default="flash_attention_2")
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--puzzles", type=int, default=64)
parser.add_argument("--results", default=str(Path(__file__).parent.parent / "results" / "grpo.jsonl"))
parser.add_argument("--progress")
a = parser.parse_args()

if a.arm == "unsloth":
    from unsloth import FastLanguageModel as LanguageModel
else:
    from meepsquared import MeepLanguageModel as LanguageModel

import torch  # noqa: E402
import trl  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

import zebra_task  # noqa: E402
from common import model_location, record  # noqa: E402

NUM_GENERATIONS, MAX_SEQ_LEN, MAX_COMPLETION = 8, 3072, 800
model, tokenizer = LanguageModel.from_pretrained(model_name=model_location(a.model, a.source), max_seq_length=MAX_SEQ_LEN,
                                                 load_in_4bit=False, dtype=torch.bfloat16, fast_inference=False,
                                                 attn_implementation=a.attn)
model = LanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth" if a.arm == "unsloth" else True)
data = Dataset.from_list([zebra_task.puzzle_to_row(p) for p in zebra_task.make_puzzles(a.puzzles, 1)])

config = dict(
    learning_rate=1e-5, lr_scheduler_type="cosine", warmup_steps=max(1, 60 // 10), optim="paged_adamw_8bit",
    adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1, per_device_train_batch_size=NUM_GENERATIONS,
    gradient_accumulation_steps=4, num_generations=NUM_GENERATIONS, max_completion_length=MAX_COMPLETION,
    max_steps=a.steps, max_grad_norm=0.1, beta=0.0, temperature=1.0, top_p=1.0, top_k=40,
    chat_template_kwargs={"enable_thinking": False}, logging_steps=1, save_strategy="no", report_to="none",
    output_dir="/tmp/meepsquared-grpo", seed=3407, bf16=True)
if "pad_to_multiple_of" in GRPOConfig.__dataclass_fields__:
    config["pad_to_multiple_of"] = 64
if a.vllm:
    config.update(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_memory,
                  vllm_max_model_length=MAX_SEQ_LEN)
trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[zebra_task.zebra_reward_func],
                      args=GRPOConfig(**config), train_dataset=data)

generation = {}
_generate = trainer._generate_and_score_completions


def timed_generate(*args, **kwargs):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = _generate(*args, **kwargs)
    torch.cuda.synchronize()
    step = trainer.state.global_step + 1
    generation[step] = generation.get(step, 0.0) + time.perf_counter() - t
    return out


trainer._generate_and_score_completions = timed_generate


class Clock(TrainerCallback):
    marks = {}

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.marks[state.global_step] = time.perf_counter()
        if a.progress and int(os.environ.get("RANK", 0)) == 0:
            with open(a.progress, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} step {state.global_step}/{a.steps}: "
                        f"{self.marks[state.global_step] - self.marks.get(state.global_step - 1, start):.0f} s, "
                        f"generation {generation.get(state.global_step, 0.0):.0f} s\n")


trainer.add_callback(Clock())
start = time.perf_counter()
trainer.train()
marks = Clock.marks
record(a.results, {
    "benchmark": "grpo-zebra", "arm": a.arm + ("-vllm" if a.vllm else ""), "model": a.model, "source": a.source, "steps": a.steps,
    "first_step_s": round(marks[1] - start, 1),
    "s_per_step_after_first": round((marks[a.steps] - marks[1]) / (a.steps - 1), 1),
    "generation_s_by_step": {k: round(v, 1) for k, v in sorted(generation.items())},
    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    "rewards": [round(h["reward"], 3) for h in trainer.state.log_history if "reward" in h],
    "trl": trl.__version__,
})
