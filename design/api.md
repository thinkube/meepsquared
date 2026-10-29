# Meep²'s own API: say what to train, Meep² decides how

Design note. Status: decided 26 October (PLAN F5); being built.

## Why

Meep²'s value is auto mode: it measures the model, the batch and the machine
and makes the hardware decisions (how the model loads, how batches are packed,
which kernels run, how many layers are recomputed), then says why. Its current
calls follow Unsloth's shape: `from_pretrained(max_seq_length=…,
load_in_4bit=…)` and `get_peft_model(use_gradient_checkpointing=…)`. That shape
asks for those decisions before Meep² has seen the batch or the machine, so it
invites the user to make them by hand. It also makes Meep² read as a copy of
another engine, and it does not describe a training job in a way a no-code
component can fill in.

## The shape

One object describes the job: what to train, on what, by which method. It can
be written in Python or as plain data (a dict, YAML or JSON), so that a form in
Thinkube can produce it.

```python
import meepsquared as m2

job = m2.TrainingJob(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="HuggingFaceH4/no_robots",        # a dataset id, a file, or a datasets.Dataset
    method="lora",                          # "lora", "qlora" or "fp8"
    steps=1000,                             # or epochs=1
)

print(job.plan())     # what meepsquared will do on this machine, and why; nothing trained yet
result = job.train()  # trains with TRL underneath; prints the plan first
result.save("out")   # the adapter; result.save("out", merged=True) for a 16-bit model
```

`job.plan()` loads the model, measures one layer on the real batch, reads the
free memory, and returns the decisions with their reasons, for example:

```
meepsquared plan for meta-llama/Llama-3.1-8B-Instruct, LoRA, on NVIDIA GB10 (122 GiB free)
  loading      file by file to the GPU (the GPU shares the system's memory)
  batches      packed, no padding (FlashAttention 2)
  kernels      fused LoRA and MLP, Liger's loss; layers not compiled (FlashAttention runs outside them)
  checkpoint   0 of 32 layers (one layer: 0.91 GiB for 8 x 1,024 tokens; free: 90 GiB)
  allocator    expandable segments
```

### What the user gives

| Field | Meaning | Default |
|---|---|---|
| `model` | a model id, a local path, or a loaded model | required |
| `data` | conversations (`messages`), prompt and completion pairs, or plain text | required for SFT |
| `task` | `"sft"` or `"grpo"` | `"sft"` |
| `method` | `"lora"`, `"qlora"` or `"fp8"` | `"lora"` |
| `steps` or `epochs` | how long | one epoch |
| `rewards` | the reward functions, for GRPO | required for GRPO |
| `output` | where the adapter goes | a folder named after the model |

### What Meep² decides, and the user may override

Names: `TrainingJob`, `plan()`, `train()` and the command `meepsquared train
job.yaml`. `train()` is the verb Transformers' and TRL's trainers use and the
command Axolotl and LLaMA-Factory use; a job is what hosted fine-tuning
services call one, and what Thinkube schedules. "Trainer" is left to TRL,
whose trainers Meep² builds underneath.

The batch is chosen for the job, not fixed: LoRA learns less from large
batches than full fine-tuning does, whatever its rank (Thinking Machines,
"LoRA Without Regret", 2025: both reach their best loss at smaller batches),
and QLoRA's runs used 16 per update up to 13B parameters and 32 to 64 above
(Dettmers et al., 2023, table 9). Auto mode counts the batch in examples.
For SFT, a token budget per update is an option (PLAN W1), off until the
measurement round decides (PLAN 0b): each update takes as many conversations
as fit it, so every update learns from about the same amount of text and the
GPU does the same work each step. `Recipe(tokens_per_update="auto")` takes the
largest budget that fits one forward pass, up to a ceiling of 32 x 1024 tokens
that the time-to-quality study (PLAN W2) will set; with a budget, the learning
rate scales from its stated value at 8 x 1024 tokens with the square root of
the ratio. Speed comes from the micro-batch and
the gradient accumulation, which auto mode fits to the memory, and from
checkpointing.

The learning rate depends on LoRA's scaling, alpha / rank: the update is the
learning rate times that scaling. Meep²'s default is alpha = rank (scaling 1).
At that scaling, published rates for 7B to 13B models are in the range of
1e-4 (QLoRA's 2e-4 at scaling 0.25; Thinking Machines: LoRA's best rate is
about 10 times full fine-tuning's for the same model, with alpha 32), and
lower for larger models (QLoRA halves it from 33B; Thinking Machines scales it
with the model's hidden size). The default follows that: 1e-4 up to 13B,
half from 30B, stated in the plan with its source. The schedule: both studies
trained at a constant rate; the 35B translation run trained with 100 warm-up
steps and cosine decay to a tenth over the planned steps. Which is better is
measured in the measurement round (constant against warm-up and cosine, same
job), and the default follows the measurement. The schedule always spans the
planned steps, so a job trained in several sessions continues one schedule.

The hardware decisions are auto mode's. Any of them can be set by hand, in one place, after Meep² has
measured:

```python
job = m2.TrainingJob(model=..., data=..., method="lora",
                  recipe=m2.Recipe(rank=32, learning_rate=2e-4),
                  overrides=m2.Overrides(checkpointed_layers=8, compile_layers=False))
```

An override is reported in the plan as the user's choice, beside what auto mode
would have chosen.

### Leaving the API

`job.trainer()` returns TRL's own `SFTTrainer` or `GRPOTrainer`, set up as
`train()` would run it, for anyone who wants TRL's full control. The current
calls (`MeepLanguageModel.from_pretrained`, `get_peft_model`) stay, as the way
for code coming from Unsloth, and are documented on that page only.

### As data

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
data: HuggingFaceH4/no_robots
method: lora
steps: 1000
```

`m2.TrainingJob.from_dict(...)` and `job.to_dict()`; `python -m meepsquared train job.yaml`.
The plan and the result are data too (`plan.to_dict()`), so a platform can show
them.

## What changes

- New: `m2.TrainingJob`, `m2.Recipe`, `m2.Overrides`, the plan and the result, the
  `meepsquared train` command. Built on the code that exists: loading, auto mode, the
  TRL patches, saving.
- The notebooks, the README, the get-started page and the benchmarks move to it.
- The current calls stay and keep their tests; the docs present them under
  "Coming from Unsloth".

## Open question

1. Whether `plan()` should run without loading the model fully (an estimate
   from the config), so a platform can show a plan before a GPU is free; the
   first version loads the model.
