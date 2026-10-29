# Benchmarks

The measurements behind the tables in Meep²'s README, as scripts anyone can
run. Each script appends one line per run to a JSONL file in `results/`:
the settings, the numbers, the command, and the environment (GPUs, Python,
and the versions of PyTorch, Transformers, TRL, PEFT, Liger, bitsandbytes,
FlashAttention, flash-linear-attention, vLLM, Unsloth and Meep²).

`results/2026-10/` holds the runs behind the README's tables, measured in
October 2026 on an NVIDIA GB10 (DGX Spark) and an RTX 3090: three runs per arm,
the held-out loss against TRL, the command of every arm, and the scripts as they
ran. Its README says what each file holds. The files directly in `results/` are
earlier single runs; the README still cites them for FP8 weights, NeMo
AutoModel and the two RTX 3090s. Every line carries its command, its
environment (versions, and the Meep² commit) and its numbers.

The files in `results/` are kept as they were recorded. Meep² was called
llebre then, so their labels, commands and paths say `llebre`.

Requirements: Meep², `datasets`, and for each arm what
it compares against (Unsloth in its own environment, vLLM for `--vllm`,
`python-constraint` for GRPO). `--progress FILE` writes a line per step.
`--source thinkube` loads a model from the cluster's mirror on Thinkube;
without it, from the Hugging Face Hub.

## SFT on one GPU: `sft.py` and `sft.ipynb`

LoRA or QLoRA, rank 16 on the seven projections, 8-bit AdamW, 8 conversations
of no_robots per step, at most 1,024 tokens, one batch per step; tokens per
second (padding not counted) from step 10 to step 30, peak memory, and the
loss of the first and last ten steps, the same in every arm when they train on
the same tokens.

```bash
python sft.py --arm trl     --model meta-llama/Llama-3.1-8B-Instruct --method lora
python sft.py --arm unsloth --model meta-llama/Llama-3.1-8B-Instruct --method lora
python sft.py --arm meepsquared  --model meta-llama/Llama-3.1-8B-Instruct --method lora
python sft.py --arm meepsquared  --model meta-llama/Llama-3.1-8B-Instruct --method lora --checkpointing true --label meepsquared-checkpointed
python sft.py --arm meepsquared  --model meta-llama/Llama-3.1-8B-Instruct --method fp8
```

`--arm trl` is Transformers, PEFT and TRL as they come: SDPA and padded
batches. `--method` is `lora`, `qlora` (4-bit weights) or `fp8` (the frozen
weights held as FP8, Meep² only). `--arm meepsquared` runs Meep² with its defaults; `--checkpointing`,
`--fused`, `--compile` and `--attn` set what the defaults decide (Gemma 4 needs
`--attn sdpa`: its heads are too wide for FlashAttention). `--label` names the
arm in the result. `sft.ipynb` runs the whole table; as a Thinkube notebook
job, it reads the models from the cluster's mirror.

## Mixture of experts: `moe.py`

LoRA rank 16 on Qwen3.6-35B-A3B's attention, gated DeltaNet, shared expert and
256 routed experts (941.9 M parameters), 32 examples per step, padding-free.

```bash
python moe.py --model Qwen/Qwen3.6-35B-A3B
python moe.py --model Qwen/Qwen3.6-35B-A3B --fp8
python moe.py --model Qwen/Qwen3.6-35B-A3B --peft-experts
```

`--fp8` holds the frozen weights as FP8, the expert stacks included
(`load_in_fp8=True`); `--peft-experts` computes the experts as PEFT does, the
full weight of every expert at every pass; `--checkpointing` sets Meep²'s
gradient checkpointing (default `auto`). Without `--data`, the first user and
assistant turns of no_robots; `--data` takes a JSONL file of
`{"prompt": ..., "answer": ...}` lines. `moe.ipynb` runs the three arms; as a
Thinkube notebook job, it reads the model from the cluster's mirror.

## GRPO: `grpo/grpo.py`

The zebra task: the model translates each clue of a zebra puzzle into a rule,
a constraint solver deduces the grid, and the reward is the share of correct
cells. Qwen3.5-4B, LoRA rank 16, 4 prompts of 8 completions of up to 800 tokens
per step.

```bash
cd grpo
python grpo.py --arm unsloth --model Qwen/Qwen3.5-4B
python grpo.py --arm meepsquared  --model Qwen/Qwen3.5-4B
torchrun --nproc_per_node 1 grpo.py --arm meepsquared --model Qwen/Qwen3.5-4B --vllm
```

`--vllm` needs vLLM and the training packages in one environment, and TRL 1.16
or later. `grpo.ipynb` runs the arms, skipping the vLLM one with a note where
vLLM is not installed; as a Thinkube notebook job, it reads the model from
the cluster's mirror.

## Several GPUs: `multi_gpu.py`

LoRA or QLoRA rank 16, padding-free, no_robots conversations of strict
user/assistant turns.

```bash
torchrun --nproc_per_node 1 multi_gpu.py --mode ddp   --model meta-llama/Llama-3.1-8B-Instruct   # one GPU
torchrun --nproc_per_node 2 multi_gpu.py --mode ddp   --model meta-llama/Llama-3.1-8B-Instruct
torchrun --nproc_per_node 2 multi_gpu.py --mode ddp   --model Qwen/Qwen3.8-27B --qlora
torchrun --nproc_per_node 2 multi_gpu.py --mode fsdp  --model Qwen/Qwen3-14B      # 29.5 GB of weights: not one 24 GB GPU
python                      multi_gpu.py --mode split --model Qwen/Qwen3-14B
```

Two questions, each with the modes that answer it: for a model that fits one
GPU, what a second one buys (DDP); for a model whose weights do not fit one
GPU, which of FSDP2 and the layer split train it, and how fast. `multi_gpu.ipynb` runs the list; as a Thinkube
notebook job on a node that shares its GPUs, it asks for every GPU of the
node. Measured on two RTX 3090 (24 GB) on PCIe, without NVLink.
