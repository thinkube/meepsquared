# Milestone translations through vLLM, on the weights the trainer already holds

Specification for Meep². Status: for review.

## 1. How this specification was written

- It was written from these sources only, all Apache-2.0:
  - vLLM v0.22.1 (tag `v0.22.1`, commit 0decac0), the vLLM that Meep²'s
    Containerfile pairs with NGC PyTorch 26.03;
  - transformers 5.17.0;
  - Meep²'s own source code.
- It holds no code from Unsloth or `unsloth_zoo`. The only thing taken from
  Unsloth is a fact: that sharing a model's weights between a trainer and an
  in-process vLLM has been done for `qwen3_5_moe`.
- Whoever implements it works from this text and the sources above, and does
  not open Unsloth's or `unsloth_zoo`'s source code.
  The commit that adds the work says so.

## 2. The goal

Today a milestone generates its translations with Transformers'
`model.generate` inside the training process (`_milestones.py`).
For Qwen3.6-35B-A3B with 400 sentences, that takes about 6.5 minutes per
milestone, or about 5.5 hours over 50 milestones.

The goal is to generate them with vLLM, in the same process, without a second
copy of the routed expert stacks in memory. The base weights are frozen during
LoRA training, so vLLM and the trainer can read the same tensors.

| Item | Size |
|---|---|
| Routed experts (40 layers × 256 experts × 3 × 2048 × 512) | 32.2 G parameters: 64.4 GB in BF16, 32.2 GB in FP8 |
| Everything else (attention, linear attention, shared experts, embeddings, output layer) | about 2.8 G parameters |

Sharing the expert stacks is required. Sharing the rest is optional: a
second copy of it costs a few GB.

## 3. Facts from the sources

### 3.1 The model in vLLM

- The checkpoint's `architectures` is `Qwen3_5MoeForConditionalGeneration`.
  vLLM builds that class (`vllm/model_executor/models/qwen3_5.py`), with
  `is_3d_moe_weight = True`.
- `language_model_only` (`vllm/config/model.py`) builds it without the
  vision encoder.
- Each decoder layer has either `linear_attn` (gated delta-net) or
  `self_attn` (full attention), by `config.layer_types`, and an `mlp` of
  class `Qwen3NextSparseMoeBlock` (`qwen3_next.py`). That block holds:
  - `gate`: the router;
  - `shared_expert` and `shared_expert_gate`;
  - `experts`: a `FusedMoE`.
- vLLM fuses these projections, by `stacked_params_mapping` in
  `Qwen3_5Model.load_weights`:

  | vLLM parameter | Made of (transformers names), in this order |
  |---|---|
  | `qkv_proj` | `q_proj`, `k_proj`, `v_proj` |
  | `gate_up_proj` (shared expert) | `gate_proj`, `up_proj` |
  | `in_proj_qkvz` | `in_proj_qkv`, `in_proj_z` |
  | `in_proj_ba` | `in_proj_b`, `in_proj_a` |

  These are fused along the output dimension, so each transformers weight is
  a block of rows of the fused vLLM weight. A block of rows of a row-major
  matrix is a contiguous view, as long as vLLM keeps the fused weight
  unsplit (no tensor parallelism).

### 3.2 The expert stacks

| | transformers 5.17 (`Qwen3_5MoeExperts`) | vLLM 0.22.1 (`FusedMoE`) |
|---|---|---|
| gate and up | `gate_up_proj`: `(experts, 2 × intermediate, hidden)`, gate rows first | `w13_weight`: `(experts, 2 × intermediate, hidden)`, w1 (gate) then w3 (up) |
| down | `down_proj`: `(experts, hidden, intermediate)` | `w2_weight`: `(experts, hidden, intermediate)` |

`Qwen3_5Model.load_weights` loads `gate_up_proj` into `w13_weight` by
splitting it in two along dimension −2, without transposing ("qwen3.5 no
need to transpose"). The two layouts are the same.

### 3.3 Whether vLLM keeps that layout after loading

- After loading, vLLM converts the expert weights to the layout of the kernel
  it chose (`process_weights_after_loading`). For unquantized experts,
  `convert_to_unquantized_kernel_format`
  (`fused_moe/oracle/unquantized.py`) does this:
  - FlashInfer CUTLASS and TRT-LLM swap the halves of w13 or re-block it;
  - Triton keeps the layout.
- With LoRA enabled, `select_unquantized_moe_backend` always picks Triton.
  So for BF16 experts with LoRA, vLLM keeps the transformers layout.
- For FP8 experts, `select_fp8_moe_backend` (`oracle/fp8.py`) does not look
  at LoRA. It can pick Marlin, CUTLASS or FlashInfer, which repack the
  weights. `KernelConfig.moe_backend = "triton"` (`vllm/config/kernel.py`)
  forces Triton.
- `TritonExperts` (`experts/triton_moe.py`) supports LoRA (it mixes in
  `LoRAExpertsMixin`). It also accepts FP8 weights with a static per-tensor
  scale (`kFp8StaticTensorSym`).

### 3.4 FP8 on both sides

- Meep² (`_fp8.py`) stores weights as `float8_e4m3fn`, where weight =
  FP8 value × scale and scale = absolute maximum / 448:
  - expert stacks: one scale per expert;
  - other frozen linear layers: one scale per tensor.
- vLLM's FP8 path dequantizes the same way, weight = FP8 value × scale
  (`quantization/fp8.py`).
- For a fused weight, vLLM requantizes the logical shards to one scale. For
  example, `w13` is loaded with a scale for w1 and one for w3, then
  requantized with a single scale per expert.

### 3.5 LoRA on the experts in vLLM

- For a model with `is_3d_moe_weight`, vLLM reads a PEFT adapter whose
  expert LoRA is in PEFT's `target_parameters` layout
  (`vllm/lora/lora_model.py`, `lora/model_manager.py`):

  | Key | Holds | lora_A | lora_B |
  |---|---|---|---|
  | `….experts.base_layer.lora_A/B` | gate_up | `(rank × experts, hidden)` | `(2 × intermediate, rank × experts)` |
  | `….experts.lora_A/B` | down | `(rank × experts, intermediate)` | `(hidden, rank × experts)` |

  Gate and up share one lora_A.
- In `model_manager.py`, lora_A is reshaped as
  `(experts, rank, in)`, and lora_B as `(out, rank, experts)` and then
  permuted to `(experts, out, rank)`.
- Meep²'s routed LoRA (`_moe.py`) sits on PEFT's `ParamWrapper` with
  `target_parameters`. Its gate/up adapter is a single product with one A per
  expert (`design/grouped-lora-kernel.md`). A milestone's
  `peft/adapter_model.safetensors` is written in PEFT's format.
- The adapter's names must be the checkpoint's names
  (`model.language_model.…`), not the names transformers gives the text model
  after loading. `_vllm.py` already renames weights back for GRPO
  (`checkpoint_name`).

### 3.6 vLLM in the same process

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` (`vllm/envs.py`) runs the engine core in
  the calling process.
- `LLM.apply_model(func)` (`entrypoints/llm.py`) runs a function on the
  `nn.Module` of the model.
- `CacheConfig.kv_cache_memory_bytes` sets the cache size in bytes, instead
  of a share of GPU memory (`gpu_memory_utilization`).
- `GPUModelRunner.reload_weights(weights_iterator=…, is_checkpoint_format=…)`
  (`v1/worker/gpu_model_runner.py`) loads weights into a model that already
  exists, and `load_format="dummy"` builds the model without reading a
  checkpoint.

## 4. Requirements

1. **Opt-in.** A recipe or override turns this on for milestones (a name such
   as `Overrides(milestone_engine="vllm")`). Without it, milestones work as
   today.
2. **No fallback.** When it is on and any step fails, the job stops with an
   error that names the step:
   - vLLM is not installed;
   - a tensor cannot be shared;
   - the adapter is refused;
   - the check of §6.1 fails.

   It never switches silently to `model.generate`.
3. **One copy of the expert stacks.** After setup, every routed expert stack
   of the training model and the matching vLLM parameter use the same
   storage: the same `data_ptr()` and the same `untyped_storage()`. The
   job checks this for every layer, and refuses to start otherwise.
4. **Frozen means frozen.** No step of training writes to a shared tensor.
   The job checks once, at a milestone, that a checksum of the shared stacks
   is unchanged since setup.
5. **BF16 and FP8.** Both `Recipe` weight modes work:
   - BF16: the shared stacks are BF16;
   - FP8: the shared stacks are FP8 with one scale per expert, and vLLM runs
     with `moe_backend="triton"`.
6. **The adapter of the milestone.** Each milestone gives vLLM the adapter it
   has just saved (`<run>/output/milestones/step-N/peft`). It is loaded as a
   new LoRA request with a new id each time, and the previous one is removed.
7. **Same output as today.** Greedy decoding with the same budgets:
   - the per-sentence limit of `max_new_tokens`;
   - the cut at the first loop;
   - the prompt built by the same chat template, with thinking off.
8. **Memory.** The setup states, in the job's log, the memory used before
   vLLM starts, after vLLM starts, and at the peak of a milestone. The
   training step's peak memory must not rise by more than the vLLM cache plus
   the parameters that are not shared.
9. **Unified memory.** Loading must not hold the checkpoint twice: once in
   the page cache and once in tensors. Meep² loads shard by shard and drops
   each shard's pages (`posix_fadvise(DONTNEED)`) for this reason. The vLLM
   path must keep that property, and the log states the peak during
   loading.

## 5. Design outline (the implementer may change it, with a reason)

- **Who allocates.** vLLM allocates the weights; the trainer's parameters
  become views of vLLM's tensors.
  - vLLM fuses projections (§3.1), and a block of rows of a fused weight is a
    contiguous view. The reverse is not possible: separate transformers
    tensors cannot be viewed as one fused tensor without a copy.
  - For the expert stacks the layouts are equal (§3.2), so either side could
    allocate them.
- **Loading.** Build vLLM with `load_format="dummy"`, then stream the
  checkpoint into it shard by shard through Meep²'s own reader, with the page
  cache dropped after each shard (requirement 9).
  - For FP8, vLLM quantizes on load (`quantization="fp8"`). Whether the
    dummy-then-reload path runs vLLM's FP8 processing once and correctly is
    question Q3.
- **Building the trainer.** Build the training model without allocating its
  frozen weights (meta device). Then attach views of vLLM's tensors to it,
  and add LoRA as today.
  - Meep²'s FP8 expert stacks and scales are then vLLM's FP8 stacks and
    scales (§3.4).
  - Parameters that are not shared are loaded as today.
- **Generation.** At a milestone:
  1. put the trainer in eval mode;
  2. free the cached activation memory;
  3. load the saved adapter into vLLM;
  4. generate;
  5. remove the adapter;
  6. go back to training.
- **Settings.**
  - Set vLLM's cache size with `kv_cache_memory_bytes` from the milestone's
    needs (sentences × (prompt + budget) tokens), not from a share of GPU
    memory.
  - `enforce_eager=True` unless CUDA graphs are measured to be worth their
    memory.

## 6. Questions to answer by test before relying on it

| # | Question | How to answer it |
|---|---|---|
| Q1 | Does vLLM's reading of the 3D adapter (§3.5) order experts and ranks the way PEFT's `ParamWrapper` wrote them? | §6.1 |
| Q2 | Does `process_weights_after_loading` on GB10 (compute capability 12.1) keep each shared tensor in place, with the same storage, layout and dtype, for BF16 with LoRA and for FP8 with `moe_backend="triton"`? | Compare `data_ptr()` and a checksum before and after |
| Q3 | Does the dummy load followed by a reload of the checkpoint produce the same weights as a normal vLLM load, in BF16 and in FP8? | Compare checksums of all parameters on a small MoE model of the same family, then on the 35B |
| Q4 | Does the fused-row view of `qkv_proj`, `in_proj_qkvz`, `in_proj_ba` and the shared expert's `gate_up_proj` match transformers' split order exactly? This includes `q_proj` being twice the head dimension (query and gate) in full attention, and the order inside `in_proj_qkv`. | Compare outputs of one layer, with transformers on views and with transformers on its own weights |
| Q5 | Does vLLM's gated delta-net (`gqa_interleaved_layout=False`) read `conv1d`, `A_log`, `dt_bias` and the norm in transformers' layout? | Same as Q4, for a `linear_attention` layer |
| Q6 | Does vLLM's memory profiling at start-up behave on unified memory, with the trainer's memory already in use? | Log the memory figures of requirement 8 |

### 6.1 The correctness check (runs at setup, every job)

On 8 fixed prompts from the evaluation set, with the current adapter (the
initial one at setup):

- Compare the logits of the first generated token from the trainer
  (transformers forward, adapter on) and from vLLM (adapter loaded).
- Pass, for each of the 8: the top-1 token is the same, or the two top
  tokens are a tie, which is when either engine ranks them within 1.0 of
  log-probability of each other (measured on Qwen3.6-35B-A3B: a prompt had
  its two candidates 0.63 apart in the trainer and the other first in vLLM;
  rounding decides between them); and where the top tokens are the same,
  its log-probability differs by at most 0.5 (measured 0.16 in BF16, 0.24
  to 0.40 in FP8 from run to run). The tail of the top 20 is not compared:
  two implementations of a mixture of experts in 16 bits route some tokens
  to other experts, and the unlikely tokens move by 1 to 4.
- Fail: the job stops with the numbers.

At setup a fresh LoRA's B is zero, so the adapter changes nothing and the
check above cannot detect a wrong adapter layout. The check must therefore
also run once with an adapter whose A and B are random and nonzero, saved
and loaded through the same path. Its bound on the top token's
log-probability is the larger of 0.5 and a tenth of the adapter's effect
(measured 0.30 to 0.38 with an effect of 5.1); a wrong layout errs by about
the effect itself and changes the top token.

## 7. Acceptance on tkspark

Qwen3.6-35B-A3B, routed LoRA rank 16, the run 6 data and milestone settings
(400 sentences, Spanish and Catalan into Aranese, FLORES+ dev), in BF16 and
in FP8:

- §6.1 passes.
- Requirement 3 holds for every layer.
- Translations are compared with `model.generate` on the same adapter:
  the share of identical outputs, and chrF of each direction. The two chrF
  values differ by at most 0.5.
- Measured and reported:
  - time of one milestone (target: below 1.5 minutes, against about 6.5
    today);
  - peak memory in training and during a milestone;
  - peak memory while loading;
  - training tokens per second, against the same run without vLLM.

## 8. Out of scope

- GRPO's use of vLLM, which stays as it is (TRL starts its own engine).
- Tensor or expert parallelism across GPUs.
- Models other than the `qwen3_5_moe` family. The code should refuse other
  architectures by name until they are tested.
