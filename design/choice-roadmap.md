# The choice task: the features asked for, in order

Status: asked for on 2026-11-24, in this order. Each item says what is known
and what is done. The goal behind them: a model trained with the choice task
must run on one 24 GB GPU (an RTX 3090), and training may use the DGX Spark.

## First: the DGX Spark (GB10, Blackwell)

### A. Native low-precision training

- FP8. Meep² already holds the frozen linear layers in FP8
  (`load_in_fp8`, `meepsquared._fp8`): dense layers and, for LoRA on the
  experts, the expert stacks. SFT uses it as method "fp8". The choice task
  takes only "lora" and "qlora": FP8 is to be added there, and to Forge's
  choice runs.
- NVFP4 through Transformer Engine's NVFP4 recipe: weights and activations in
  FP4, not weight-only. Not started. To find out first: whether Transformer
  Engine's NVFP4 recipe runs on the GB10 (compute capability 12.1) and on
  arm64, and how a frozen base with LoRA adapters runs under it.
- Measured on the Spark against NF4: the speed, and the answer key (the
  milestones' count of right answers).

### B. Milestones answered on the serving base

Train in one format (bf16, FP8 or NVFP4 on the Spark) and answer each
milestone's evaluation set on another: the NF4 base a 3090 will serve. The
milestone count then measures the model as it will be served.

## Then, by usefulness

1. The plan fits the memory itself: on out of memory it tries the safe
   settings (fewer `candidates_per_pass`, more recomputation) and reports its
   choice. Known case: run bc0e9d4f (Gemma 4 26B-A4B, choice, up to 86
   candidates) failed its plan on the 3090, short by about 250 MiB.
2. The choice task on several GPUs, with the existing layer split
   (`meepsquared._split`): the layers over two 3090s.
3. A passage read once for all its verbs: the shared text of the questions of
   one passage reused. Serving first; in training it changes the order of the
   questions.
4. A stop when the milestone curve goes flat (the person's rule: stop a
   training that does not learn).
5. Serving generation and the choice head from one base: see
   `serving-generation-and-choice.md`.

## Limits

- No change to the meepsquared venv on a node while a training runs there.
- No two GPU trainings at once.
