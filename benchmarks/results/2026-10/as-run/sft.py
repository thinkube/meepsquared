# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA and QLoRA SFT on one GPU: llebre, Unsloth, and Transformers + PEFT + TRL as they come.

    python sft.py --arm llebre --model meta-llama/Llama-3.1-8B-Instruct --method lora

Every arm trains the same training, on the same conversations in the same
order. With --plan, it is llebre's: the training decisions of a llebre plan
(plan.py writes one) are taken by every arm (the LoRA rank, alpha, dropout and
targets, the effective batch, the optimizer, the learning rate, its schedule
and warm-up), so the engines differ only in how they run it; each keeps its own
hardware decisions (the micro-batch, with gradient accumulation to the
effective batch, and the checkpointing). A plan with a token budget
(plan.py --tokens-per-update) is trained by the llebre arm alone, through
llebre's layout of the batches (llebre._batches); the other arms refuse it,
as they are not compared on it. Without --plan, every arm trains LoRA
rank 16 (alpha 16, no dropout) on the seven projections of each layer, with
8-bit AdamW, a constant learning rate of 1e-4 and batches of 8. The
result is the tokens per second (padding not counted) from step --warmup to
the last, the peak GPU memory, and the mean loss of the first and last ten
steps, which tells whether the arms trained on the same tokens.
"""

import argparse
import math
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["llebre", "unsloth", "trl"], required=True)
parser.add_argument("--model", required=True, help="a Hugging Face model id or a local folder")
parser.add_argument("--source", choices=["huggingface", "thinkube"], default="huggingface", help="where the model comes from: Hugging Face, or the cluster's mirror on Thinkube")
parser.add_argument("--method", choices=["lora", "qlora", "fp8"], default="lora",
                    help="fp8: LoRA on frozen weights held as FP8 (llebre only)")
parser.add_argument("--attn", default="flash_attention_2", help="llebre: the attention implementation, or auto for from_pretrained's choice")
parser.add_argument("--label", help="the arm's name in the result, when it differs from --arm")
parser.add_argument("--checkpointing", default="auto",
                    help="llebre: gradient checkpointing: plan (the plan's layout, trained in the order the plan laid out), auto, selective (every layer selective), memory, true (every layer whole), false, a number of layers or a fraction; unsloth: auto (its default, every layer with the activations offloaded) or false (none)")
parser.add_argument("--compile", choices=["auto", "true", "false"], default="auto", help="llebre: compile_layers")
parser.add_argument("--fused", choices=["auto", "true", "false"], default="auto", help="llebre: fused_lora")
parser.add_argument("--data", default="HuggingFaceH4/no_robots", help="a dataset id or a Parquet file of conversations")
parser.add_argument("--alternating", action="store_true", help="only conversations of strict user/assistant turns")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--batch", type=int, default=8, help="examples per update, without --plan")
parser.add_argument("--plan", help="a llebre plan (plan.py): every arm takes its training decisions")
parser.add_argument("--micro-batch", type=int, help="examples per forward pass, with gradient accumulation to the effective batch; the effective batch when not given")
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0, help="the trainer's seed; a plan's order is laid out with the job's (42)")
parser.add_argument("--eval-rows", type=int, default=0, help="conversations after the trained ones, never trained on, whose loss is measured at the end")
parser.add_argument("--schedule", choices=["constant", "cosine"], help="the learning-rate schedule in place of the plan's, to compare schedules")
parser.add_argument("--eval-every", type=int, help="with --eval-rows: the held-out loss every this many steps, with the training seconds so far (time spent evaluating not counted)")
parser.add_argument("--minutes", type=float, help="stop after this many minutes of training (evaluation not counted); --steps stays the planned length the schedule spans")
parser.add_argument("--eval-only", action="store_true", help="with --eval-rows: the held-out loss of the untrained model, set up as the arm trains it; nothing is trained or recorded")
parser.add_argument("--save-adapter", help="a folder to save the trained LoRA adapter in, at the end of training")
parser.add_argument("--adapter", help="with --eval-only: a saved LoRA adapter to evaluate, loaded into this arm's model")
parser.add_argument("--results", default="results/sft.jsonl")
parser.add_argument("--progress", help="a file for a line per step")
parser.add_argument("--profile", help="a file for the GPU time per operation of three steps after --warmup")
parser.add_argument("--memory-snapshot", help="a file for PyTorch's allocation history, written at each new allocated peak, to see what is alive there")
a = parser.parse_args()

if a.arm == "unsloth":
    import unsloth  # noqa: F401  patches transformers and TRL before they load

import torch  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from common import (Clock, Profile, QualityClock, chat_dataset, memory_snapshot, model_location, peak_gib, record,  # noqa: E402
                    setting, plan_decision, schedule, training_decisions)

training = training_decisions(a.plan) if a.plan else {
    "rank": 16, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "effective_batch": a.batch, "optimizer": "adamw_8bit", "learning_rate": 1e-4, "schedule": "constant", "warmup_steps": 0}
if a.schedule:
    training["schedule"] = a.schedule
TARGETS = list(training["targets"])
budget = training.get("tokens_per_update")
if budget is not None and a.arm != "llebre":
    raise SystemExit(f"{a.plan}: the plan batches by tokens ({budget:,} per update), which only the llebre arm "
                     "trains; the other arms are not compared on a token budget")
batch = training.get("effective_batch")
if budget is not None:
    # The budget in forward passes as llebre plans it: rows of --max-length per pass, given or
    # the plan's, and each pass an equal share of the budget's tokens
    micro = a.micro_batch or plan_decision(a.plan, "micro_batch")
    accumulation = math.ceil(math.ceil(budget / a.max_length) / micro)
    micro_tokens = math.ceil(budget / accumulation)
else:
    # The micro-batch is a hardware decision: given, or llebre's own from the plan for the llebre arm, or the effective batch
    micro = a.micro_batch or (plan_decision(a.plan, "micro_batch") if a.plan and a.arm == "llebre" else batch)
    if batch % micro:
        raise SystemExit(f"--micro-batch {micro} does not divide the effective batch {batch}")
    accumulation = batch // micro
qlora = a.method == "qlora"
fp8 = a.method == "fp8"
if fp8 and a.arm != "llebre":
    raise SystemExit(f"--method fp8 is llebre's: the {a.arm} arm has no FP8 weights")


location = model_location(a.model, a.source)
# The held-out conversations are the data's last --eval-rows, the same for every arm and batch
if budget is not None or a.eval_rows:
    # How many conversations a budget's steps take is known only once they are tokenized: every
    # usable conversation is offered, the last --eval-rows held out
    rows = chat_dataset(a.data, None, alternating=a.alternating)
    usable = len(rows) - a.eval_rows
    trained = usable if budget is not None else batch * a.steps
    if trained > usable:
        raise SystemExit(f"{a.data}: {a.steps} steps of {batch} take {trained} conversations, and {usable} come "
                         f"before the {a.eval_rows} held out")
else:
    rows = chat_dataset(a.data, batch * a.steps, alternating=a.alternating)
    trained = batch * a.steps
data = rows.select(range(trained))
held_out = rows.select(range(len(rows) - a.eval_rows, len(rows))) if a.eval_rows else None
# gradient_accumulation_steps is given: Unsloth raises it to 2 when it is not, so that a step holds twice the examples
config = dict(output_dir="/tmp/llebre-sft", max_steps=a.steps, per_device_train_batch_size=micro,
              gradient_accumulation_steps=accumulation, learning_rate=training["learning_rate"],
              optim=training["optimizer"], warmup_steps=training["warmup_steps"], bf16=True, logging_steps=1,
              save_strategy="no", report_to=[], seed=a.seed, max_length=a.max_length,
              include_num_input_tokens_seen="non_padding", dataloader_num_workers=0, per_device_eval_batch_size=micro,
              **schedule(training))
peft_config = None

if a.checkpointing == "plan" and not (a.arm == "llebre" and a.plan):
    raise SystemExit("--checkpointing plan takes the layout of a plan (--plan), on the llebre arm")
if a.arm == "llebre":
    import llebre as lb
    from llebre import _checkpointing

    if a.checkpointing == "plan":
        # The plan's layout of whole and selective layers, measured on the largest batch of its order
        checkpointing = _checkpointing.Recompute(plan_decision(a.plan, "checkpointed_layers"),
                                                 plan_decision(a.plan, "selective_layers") or 0)
    else:
        checkpointing = setting(a.checkpointing)

    model, tokenizer = lb.LlebreLanguageModel.from_pretrained(location, max_seq_length=a.max_length, load_in_4bit=qlora,
                                                            load_in_fp8=fp8,
                                                            attn_implementation=None if a.attn == "auto" else a.attn)
    model = lb.LlebreLanguageModel.get_peft_model(model, r=training["rank"], lora_alpha=training["alpha"],
                                                lora_dropout=training["dropout"], target_modules=TARGETS,
                                                use_gradient_checkpointing=checkpointing, random_state=0,
                                                fused_lora=setting(a.fused), compile_layers=setting(a.compile))
elif a.arm == "unsloth":
    from unsloth import FastLanguageModel

    def unsloth_checkpointing(value: str):
        """Unsloth's use_gradient_checkpointing for --checkpointing: its default, or none."""
        if value == "auto":
            return "unsloth"
        if value == "false":
            return False
        raise SystemExit(f"--checkpointing {value!r}: the unsloth arm takes auto or false")

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=location, max_seq_length=a.max_length,
                                                         load_in_4bit=qlora, dtype=torch.bfloat16)
    model = FastLanguageModel.get_peft_model(model, r=training["rank"], lora_alpha=training["alpha"],
                                             lora_dropout=training["dropout"], target_modules=TARGETS,
                                             use_gradient_checkpointing=unsloth_checkpointing(a.checkpointing), random_state=0)
    config["gradient_checkpointing"] = False  # Unsloth checkpoints itself
else:
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(location)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16) if qlora else None
    # Transformers' default attention: SDPA where a part of the model supports it, eager elsewhere
    model = AutoModelForCausalLM.from_pretrained(location, dtype=torch.bfloat16, quantization_config=quantization,
                                                 device_map={"": 0})
    # The seven projections, not those of a vision or audio tower (Gemma 4), whose layers PEFT refuses
    targets = rf"^(?!.*(vision|audio)_tower).*\.({'|'.join(TARGETS)})$"
    peft_config = LoraConfig(r=training["rank"], lora_alpha=training["alpha"], lora_dropout=training["dropout"],
                             target_modules=targets, task_type="CAUSAL_LM")
    config["gradient_checkpointing"] = True
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if a.eval_every and not a.eval_rows:
    raise SystemExit("--eval-every needs --eval-rows: the conversations the loss is measured on")
clock = Clock(a.progress)
quality = QualityClock(a.eval_every, a.minutes)
trainer = SFTTrainer(model=model, args=SFTConfig(**config), train_dataset=data, processing_class=tokenizer,
                     eval_dataset=held_out,
                     peft_config=peft_config, callbacks=[clock, quality] + ([Profile(a.profile, a.warmup + 1)] if a.profile else []))
if budget is not None or (a.arm == "llebre" and a.plan):
    # llebre's order: the batches its plan laid out and measured, the same for every llebre arm of a plan
    from llebre import _batches

    lengths = _batches.lengths(trainer.train_dataset, a.max_length)
    if budget is not None:
        laid = _batches.lay_out(lengths, _batches.loading(trainer), accumulation, a.steps, None, budget=micro_tokens)
        if laid.per_epoch < a.steps * accumulation:
            raise SystemExit(f"{a.data}: {laid.per_epoch} forward passes of up to {micro_tokens:,} tokens; "
                             f"{a.steps} steps of {accumulation} need {a.steps * accumulation}")
    else:
        laid = _batches.lay_out(lengths, _batches.loading(trainer), accumulation, a.steps, None, micro=micro)
    _batches.install(trainer, laid)
if a.eval_only:
    if held_out is None:
        raise SystemExit("--eval-only needs --eval-rows: the conversations the loss is measured on")
    if a.adapter:
        # Every weight of the saved adapter must find its module in this arm's model, or the loss is not the adapter's
        from peft import load_peft_weights, set_peft_model_state_dict
        saved = load_peft_weights(a.adapter)
        loaded = set_peft_model_state_dict(trainer.model, saved)
        if loaded.unexpected_keys:
            raise SystemExit(f"{a.adapter}: {len(loaded.unexpected_keys)} of {len(saved)} adapter weights have no module "
                             f"in this model, such as {loaded.unexpected_keys[0]}")
        missing = [k for k in loaded.missing_keys if "lora_" in k]
        if missing:
            raise SystemExit(f"{a.adapter}: {len(missing)} LoRA weights of this model are not in the adapter, such as {missing[0]}")
        print(f"loaded {len(saved)} adapter weights from {a.adapter}", flush=True)
    print("RESULT eval-only", json.dumps({"arm": a.label or a.arm, "model": a.model, "adapter": a.adapter,
                                          "eval_loss": round(trainer.evaluate()["eval_loss"], 6)}), flush=True)
    raise SystemExit(0)
torch.cuda.reset_peak_memory_stats()
with memory_snapshot(a.memory_snapshot, trainer):
    trainer.train()
steps_done = trainer.state.global_step
tokens_per_s, s_per_step = clock.rate(a.warmup, steps_done)
# The held-out set as the trainer prepared it at the start; a dataset passed to evaluate() is not tokenized by
# every TRL (Unsloth's 1.15 is not)
eval_loss = round(trainer.evaluate()["eval_loss"], 4) if held_out is not None else None
if a.save_adapter:
    trainer.model.save_pretrained(a.save_adapter)
    print(f"adapter saved in {a.save_adapter}", flush=True)
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]


def layout(model) -> dict:
    """How many layers trained recomputing their whole forward, recomputing selectively, or keeping everything."""
    trained = _checkpointing.kinds(model)
    return {kind: trained.count(kind) for kind in ("whole", "selective", "none")}


record(a.results, {
    "benchmark": "sft", "arm": a.label or a.arm, "attn": model.config._attn_implementation if a.arm == "llebre" else None,
    "checkpointing": a.checkpointing if a.arm in ("llebre", "unsloth") else None,
    "checkpointed_layers": layout(model) if a.arm == "llebre" else None,
    "fused": getattr(model, "_llebre_fused_lora", None) if a.arm == "llebre" else None,
    "compiled": getattr(model, "_llebre_compile_layers", None) if a.arm == "llebre" else None, "model": a.model, "source": a.source, "method": a.method,
    "batch": batch, "tokens_per_update": budget, "micro_batch": micro, "plan": a.plan, "training": training, "steps": a.steps, "tokens_per_s": round(tokens_per_s), "tokens": trainer.state.num_input_tokens_seen, "s_per_step": round(s_per_step, 3),
    "s_by_step": clock.by_step(), "peak_gib": peak_gib(), "loss_first10": round(sum(losses[:10]) / 10, 4),
    "loss_last10": round(sum(losses[-10:]) / 10, 4), "eval_rows": a.eval_rows, "eval_loss": eval_loss,
    "steps_done": steps_done, "minutes": a.minutes, "eval_every": a.eval_every, "quality": quality.points, "padding_free": getattr(trainer.args, "padding_free", None),
})
