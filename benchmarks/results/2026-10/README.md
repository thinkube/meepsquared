# October 2026: the runs behind the README's tables

Measured 25 to 28 October 2026 on one NVIDIA GB10 (DGX Spark) and one RTX 3090,
with Meep² at commit `1375994` (the speed runs) and `7971242` (`parity.jsonl`), when it was still called llebre. The files here
are as the runs wrote them, so their labels, commands and paths say `llebre`;
the `--arm llebre` of these commands is `--arm meepsquared` in today's
`benchmarks/sft.py`.

## Files

| File | What it holds |
|---|---|
| `epoch.jsonl` | One epoch (1,137 steps), GB10: Gemma 4 E4B and Qwen 3.5 9B with TRL at its defaults, Unsloth at its fastest and Meep²; held-out loss every 100 steps |
| `epoch-shared-evaluation.jsonl` | The held-out loss of each epoch run's adapter, measured by TRL's evaluation |
| `epoch-smoke.jsonl` | The 20-step runs that checked saving and the shared evaluation before the epoch runs |
| `speed-gb10.jsonl` | LoRA, 30 steps, three runs per arm, GB10: Llama 3.1 8B, Qwen 3.5 9B, Gemma 4 E4B |
| `qlora-gb10.jsonl` | QLoRA, 30 steps, three runs per arm, GB10: Llama 3.1 8B, Gemma 4 E4B |
| `moe-gb10.jsonl` | LoRA on every expert of Qwen3.6-35B-A3B, three runs per arm, GB10 |
| `speed-3090.jsonl` | LoRA, 30 steps, three runs per arm, one RTX 3090 |
| `parity.jsonl` | LoRA, 200 steps, held-out loss on 400 conversations every 50 steps, seeds 42 and 43 |
| `parity-eval-only.jsonl` | The held-out loss of the untrained models through each engine's evaluation |
| `number-formats-llama.jsonl` | Llama 3.1 8B, 200 steps on an unfused setup, held-out loss: BF16, FP8 and QLoRA weights, and three NVFP4 prototypes |
| `plans/` | The plans every arm took its training decisions from (batch, micro-batch, learning rate) |
| `queue/` | The scripts that submitted every run, with their logs |
| `as-run/` | `sft.py`, `plan.py`, `moe.py` and their helpers as they ran. `sft.py` is the version of the epoch runs; it adds `--save-adapter`, `--adapter` and `--eval-only` to the version of the short runs and evaluates on the held-out set the trainer prepared, with the same training code. `long-run-stream.ipynb` ran each epoch job, printing while it ran |

Each line of a `.jsonl` file holds the run's command, its environment (GPU,
Python, and the versions of PyTorch, Transformers, TRL, PEFT and the rest), its
settings, the seconds of every step (`s_by_step`) and its numbers.

## What to know when reading them

- Tokens per second count the tokens trained, padding not counted, from step 11
  to step 30. In `parity.jsonl` they include the pauses for evaluation, so they
  are not speeds; use the other files for speed.
- The Unsloth arms on Qwen 3.5 ran with `UNSLOTH_COMPILE_DISABLE=1`, set through
  `run_env.py` in the queue scripts. The result lines do not record it.
- The Unsloth arms ran in an environment with Unsloth 2026.10.7, transformers
  5.17.0 and TRL 0.29.1; the TRL and Meep² arms with transformers 5.20.2 and
  TRL 1.16.0. Unsloth's releases up to 2026.10.10 require transformers 5.17 or
  older.
- The first Unsloth Llama arm of seed 42 in `parity.jsonl` failed at its final
  evaluation, from a fault in our script (it passed untokenized rows to
  `evaluate`); it was fixed and the arm run again. Both are in `queue/parity_gb10.log`.

## Rerun them, and tell us what you get

Every command below runs from `benchmarks/` with today's script (replace
`--arm llebre` with `--arm meepsquared`, and drop `--source thinkube` to load
the models from Hugging Face). If your numbers differ, or you think a
comparison is unfair to any engine, open an issue with your GPU, the versions
of your libraries, the command and the result line. We will correct the tables
and say who found it.

## The command of every arm

The commands are as they ran: their `results/round2/` paths are this folder, under the name it had then.

| File | Arm | Command |
|---|---|---|
| `epoch.jsonl` | `gemma-meep` | `sft.py --arm llebre --label gemma-meep --attn auto --checkpointing plan --model google/gemma-4-E4B-it --method lora --source thinkube --steps 1137 --warmup 10 --seed 42 --plan results/round2/plans/gemma-epoch.json --eval-rows 400 --eval-every 100 --save-adapter results/round2/adapters/gemma-meep --results results/round2/epoch.jsonl --progress results/round2/progress-gb10.txt` |
| `epoch.jsonl` | `qwen-meep` | `sft.py --arm llebre --label qwen-meep --attn auto --checkpointing plan --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 1137 --warmup 10 --seed 42 --plan results/round2/plans/qwen-epoch.json --eval-rows 400 --eval-every 100 --save-adapter results/round2/adapters/qwen-meep --results results/round2/epoch.jsonl --progress results/round2/progress-gb10.txt` |
| `epoch.jsonl` | `gemma-unsloth-fastest` | `sft.py --arm unsloth --label gemma-unsloth-fastest --checkpointing false --model google/gemma-4-E4B-it --method lora --source thinkube --steps 1137 --warmup 10 --seed 42 --plan results/round2/plans/gemma-epoch.json --eval-rows 400 --eval-every 100 --save-adapter results/round2/adapters/gemma-unsloth --results results/round2/epoch.jsonl --progress results/round2/progress-gb10.txt` |
| `epoch.jsonl` | `qwen-unsloth-fastest` | `sft.py --arm unsloth --label qwen-unsloth-fastest --checkpointing false --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 1137 --warmup 10 --seed 42 --plan results/round2/plans/qwen-epoch.json --eval-rows 400 --eval-every 100 --save-adapter results/round2/adapters/qwen-unsloth --results results/round2/epoch.jsonl --progress results/round2/progress-gb10.txt` |
| `epoch.jsonl` | `gemma-trl` | `sft.py --arm trl --label gemma-trl --model google/gemma-4-E4B-it --method lora --source thinkube --steps 1137 --warmup 10 --seed 42 --plan results/round2/plans/gemma-epoch.json --eval-rows 400 --eval-every 100 --save-adapter results/round2/adapters/gemma-trl --results results/round2/epoch.jsonl --progress results/round2/progress-gb10.txt` |
| `epoch.jsonl` | `qwen-trl` | `sft.py --arm trl --label qwen-trl --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 1137 --warmup 10 --seed 42 --plan results/round2/plans/qwen-epoch.json --eval-rows 400 --eval-every 100 --save-adapter results/round2/adapters/qwen-trl --results results/round2/epoch.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `llama-hand-all` | `sft.py --arm llebre --label llama-hand-all --attn flash_attention_2 --checkpointing true --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `llama-hand-none` | `sft.py --arm llebre --label llama-hand-none --attn flash_attention_2 --checkpointing false --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `llama-auto` | `sft.py --arm llebre --label llama-auto --attn flash_attention_2 --checkpointing plan --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `gemma-today-sdpa` | `sft.py --arm llebre --label gemma-today-sdpa --attn sdpa --checkpointing false --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `gemma-hand-all` | `sft.py --arm llebre --label gemma-hand-all --attn auto --checkpointing true --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `gemma-auto` | `sft.py --arm llebre --label gemma-auto --attn auto --checkpointing plan --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `qwen-hand-all` | `sft.py --arm llebre --label qwen-hand-all --attn auto --checkpointing true --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/qwen-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `qwen-hand-none` | `sft.py --arm llebre --label qwen-hand-none --attn auto --checkpointing false --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/qwen-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `qwen-auto` | `sft.py --arm llebre --label qwen-auto --attn auto --checkpointing plan --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/qwen-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `llama-trl` | `sft.py --arm trl --label llama-trl --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `llama-unsloth` | `sft.py --arm unsloth --label llama-unsloth --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `llama-unsloth-no-checkpointing` | `sft.py --arm unsloth --label llama-unsloth-no-checkpointing --checkpointing false --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `gemma-trl` | `sft.py --arm trl --label gemma-trl --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `gemma-unsloth` | `sft.py --arm unsloth --label gemma-unsloth --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `gemma-unsloth-no-checkpointing` | `sft.py --arm unsloth --label gemma-unsloth-no-checkpointing --checkpointing false --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `qwen-trl` | `sft.py --arm trl --label qwen-trl --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/qwen-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `qwen-unsloth` | `sft.py --arm unsloth --label qwen-unsloth --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/qwen-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-gb10.jsonl` | `qwen-unsloth-no-checkpointing` | `sft.py --arm unsloth --label qwen-unsloth-no-checkpointing --checkpointing false --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/qwen-lora.json --results results/round2/speed-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `llama-qlora-auto` | `sft.py --arm llebre --label llama-qlora-auto --attn flash_attention_2 --checkpointing plan --model meta-llama/Llama-3.1-8B-Instruct --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `llama-qlora-trl` | `sft.py --arm trl --label llama-qlora-trl --model meta-llama/Llama-3.1-8B-Instruct --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `llama-qlora-unsloth` | `sft.py --arm unsloth --label llama-qlora-unsloth --model meta-llama/Llama-3.1-8B-Instruct --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `llama-qlora-unsloth-no-checkpointing` | `sft.py --arm unsloth --label llama-qlora-unsloth-no-checkpointing --checkpointing false --model meta-llama/Llama-3.1-8B-Instruct --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/llama-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `gemma-qlora-auto` | `sft.py --arm llebre --label gemma-qlora-auto --attn auto --checkpointing plan --model google/gemma-4-E4B-it --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `gemma-qlora-trl` | `sft.py --arm trl --label gemma-qlora-trl --model google/gemma-4-E4B-it --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `gemma-qlora-unsloth` | `sft.py --arm unsloth --label gemma-qlora-unsloth --model google/gemma-4-E4B-it --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `qlora-gb10.jsonl` | `gemma-qlora-unsloth-no-checkpointing` | `sft.py --arm unsloth --label gemma-qlora-unsloth-no-checkpointing --checkpointing false --model google/gemma-4-E4B-it --method qlora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round2/plans/gemma-qlora.json --results results/round2/qlora-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `moe-gb10.jsonl` | `moe-llebre` | `moe.py --model Qwen/Qwen3.6-35B-A3B --source thinkube --steps 30 --warmup 10 --label moe-llebre --results results/round2/moe-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `moe-gb10.jsonl` | `moe-peft-experts` | `moe.py --model Qwen/Qwen3.6-35B-A3B --source thinkube --steps 30 --warmup 10 --label moe-peft-experts --peft-experts --results results/round2/moe-gb10.jsonl --progress results/round2/progress-gb10.txt` |
| `speed-3090.jsonl` | `llama-today-auto` | `sft.py --arm llebre --label llama-today-auto --attn flash_attention_2 --checkpointing auto --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/llama-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `llama-auto` | `sft.py --arm llebre --label llama-auto --attn flash_attention_2 --checkpointing plan --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/llama-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `llama-hand-16` | `sft.py --arm llebre --label llama-hand-16 --attn flash_attention_2 --checkpointing 16 --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/llama-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `gemma-today-sdpa` | `sft.py --arm llebre --label gemma-today-sdpa --attn sdpa --checkpointing auto --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/gemma-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `gemma-auto` | `sft.py --arm llebre --label gemma-auto --attn auto --checkpointing plan --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/gemma-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `llama-trl` | `sft.py --arm trl --label llama-trl --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/llama-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `llama-unsloth` | `sft.py --arm unsloth --label llama-unsloth --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/llama-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `gemma-trl` | `sft.py --arm trl --label gemma-trl --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/gemma-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `speed-3090.jsonl` | `gemma-unsloth` | `sft.py --arm unsloth --label gemma-unsloth --model google/gemma-4-E4B-it --method lora --source thinkube --steps 30 --warmup 10 --seed 42 --plan results/round/../round2/plans/gemma-lora-3090.json --results results/round/../round2/speed-3090.jsonl --progress results/round/../round2/progress-3090.txt` |
| `parity.jsonl` | `llama-trl` | `sft.py --arm trl --label llama-trl --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 200 --warmup 10 --seed 42 --plan results/round2/plans/llama-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
| `parity.jsonl` | `llama-auto` | `sft.py --arm llebre --label llama-auto --attn flash_attention_2 --checkpointing plan --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 200 --warmup 10 --seed 42 --plan results/round2/plans/llama-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
| `parity.jsonl` | `gemma-unsloth` | `sft.py --arm unsloth --label gemma-unsloth --model google/gemma-4-E4B-it --method lora --source thinkube --steps 200 --warmup 10 --seed 42 --plan results/round2/plans/gemma-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
| `parity.jsonl` | `gemma-auto` | `sft.py --arm llebre --label gemma-auto --attn auto --checkpointing plan --model google/gemma-4-E4B-it --method lora --source thinkube --steps 200 --warmup 10 --seed 42 --plan results/round2/plans/gemma-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
| `parity.jsonl` | `qwen-unsloth` | `sft.py --arm unsloth --label qwen-unsloth --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 200 --warmup 10 --seed 42 --plan results/round2/plans/qwen-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
| `parity.jsonl` | `llama-unsloth` | `sft.py --arm unsloth --label llama-unsloth --model meta-llama/Llama-3.1-8B-Instruct --method lora --source thinkube --steps 200 --warmup 10 --seed 43 --plan results/round2/plans/llama-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
| `parity.jsonl` | `qwen-auto` | `sft.py --arm llebre --label qwen-auto --attn auto --checkpointing plan --model Qwen/Qwen3.5-9B --method lora --source thinkube --steps 200 --warmup 10 --seed 42 --plan results/round2/plans/qwen-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
| `parity.jsonl` | `gemma-trl` | `sft.py --arm trl --label gemma-trl --model google/gemma-4-E4B-it --method lora --source thinkube --steps 200 --warmup 10 --seed 42 --plan results/round2/plans/gemma-parity.json --eval-rows 400 --eval-every 50 --results results/round2/parity.jsonl --progress results/round2/progress-gb10.txt` |
