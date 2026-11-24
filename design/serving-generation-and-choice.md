# Serving generation and a choice head from one copy of the base

Status: requested, not started. Training comes first (see "Order of work").

## The need

One GPU serves a base model for two kinds of requests at the same time:

- generation: translation and labeling, the base alone or with a LoRA adapter;
- choice: a question with candidates, answered by an adapter and NanoJev's set
  head (Meep²'s choice task), from the hidden state at the end of each
  candidate.

The base is held on the GPU once. Two copies (a vLLM server and a separate
choice process) fit only small models and waste the memory a larger one needs.

## Why vLLM alone does not do it

- vLLM does not return generated text and prompt hidden states from one
  request; generate and encode are separate calls
  (vllm-project/vllm RFC #24288).
- LLM.encode accepts a LoRA request, and the reward pooler returns every
  token's hidden state; whether one running instance serves both generate and
  encode was not found in its documentation. PR #38565 says the alternative
  today is two vLLM instances.
- SGLang's /generate takes lora_path and return_hidden_states ("last" gives
  the last token's state), which is what the head reads. Its tracker reports
  missing prompt rows on prefix-cache hits (#8066), a broken "last" mode
  (PR #51475) and a slowdown (#8003); LoRA together with hidden states is not
  documented. Not tested here.

## What Meep² has

meepsquared._shared_vllm (design/milestones-vllm-shared-weights.md): vLLM in
the training process on the same expert stacks as the Transformers model, the
adapter on both, checked by first-token log-probabilities. Limits today:

- the qwen3_5_moe family only, refused by name otherwise;
- 16-bit or FP8 stacks, not 4-bit;
- only the routed expert stacks are shared; every other weight is held by each
  side.

The choice task's answering code (the milestone evaluation of _choice_job:
the shared text read once, the candidates read after it, the set head on the
last state of each) is the choice side.

## What to build

1. Share every weight between vLLM and the Transformers model, not only expert
   stacks, so a dense model (Gemma 4 E4B) is held once.
2. The Gemma 4 family (gemma4, dense and the 26B-A4B mixture of experts).
3. A choice entry point on the shared model: adapter and head loaded once, the
   adapter on for a choice question; the text shared by the verbs of one
   passage read once and kept for the next verb.
4. A server process around it: generation requests go to vLLM (with an
   adapter or without), choice requests to the choice entry point.

16-bit or FP8 first: vLLM's and bitsandbytes' 4-bit layouts differ. Gemma 4
E4B is 16 GB in bf16 and fits a 24 GB GPU for serving. A choice adapter
trained on an NF4 base is checked on the answer key on the served base before
it is used there.

## Order of work

Training comes first: Meep² holding the experts of a mixture-of-experts model
in NF4 (stacks quantized one expert at a time while loading, and a product in
the grouped expert code that reads them), so Gemma 4 26B-A4B can be trained on
one 24 GB GPU. Serving follows.

## Numbers this rests on

- Gemma 4 E4B, NF4, choice task on the 3090: 201 answer-key questions answered
  in 128.8 s (Forge run c30b4cd1, milestones), about 0.64 s per question; peak
  14.03 GiB on the GPU while training (its plan).
- Gemma 4 26B-A4B: experts 22.8 B parameters (45.7 GB bf16, 22.8 GB FP8,
  about 12.8 GB NF4), the rest about 3.5 GB; computed from config.json, not
  measured. Loading in NF4 today fails with out of memory at 23.4 GiB, because
  bitsandbytes leaves the 3D expert parameters in bf16 (Forge run d559cbe5).
