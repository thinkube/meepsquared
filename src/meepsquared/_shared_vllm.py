# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Milestones answered by vLLM inside the training process, on the expert stacks the trainer already holds.

Built from design/milestones-vllm-shared-weights.md, from vLLM v0.22.1,
Transformers and PEFT; no code of Unsloth's.

- The model: the qwen3_5_moe family alone (ARCHITECTURES), refused by name
  otherwise. Its routed expert stacks are the size of the model: in vLLM
  they are w13_weight (experts, 2 x intermediate, hidden) and w2_weight
  (experts, hidden, intermediate), in Transformers gate_up_proj and
  down_proj, the same layout. They are held once, by vLLM; the trainer's
  stacks are vLLM's tensors. Every other weight is held by each side apart:
  a few GB, which keeps vLLM's fused projections out of the trainer.
- The order: the trainer loads first, its expert stacks never read and left
  on the meta device (trainer_without_stacks); vLLM, once imported, changes
  Transformers' configuration classes, after which Transformers cannot build
  this model. vLLM then loads the whole checkpoint, one shard at a time read
  to the GPU and its pages dropped (meepsquared._unified_memory's reason), and
  processes the weights once as it does for any load; then the trainer's
  stacks are attached to vLLM's (Engine.attach), the same storage checked for
  every layer.
- FP8: vLLM quantizes the stacks as it loads (quantization="fp8"), one scale
  per expert, and runs them on its Triton kernel (moe_backend="triton"),
  which keeps the layout; the trainer's FP8 stacks (meepsquared._fp8) are
  those tensors and scales.
- An answer: vLLM's greedy decoding of the prompt's tokens, with the
  milestone's adapter as a LoRA request of a new id, the previous one
  removed; the caller cuts it as model.generate's answers are cut.
- The check: on fixed prompts, vLLM's log-probabilities of the first
  answer token against the trainer's, with the adapter on both
  (first_token_logprobs).

On any failure the job stops with the step that failed: there is no
switch to model.generate.
"""

import contextlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

ARCHITECTURES = ("Qwen3_5MoeForConditionalGeneration",)
# The stacks as the checkpoint names them, and as Transformers' text model does
CHECKPOINT_STACKS = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
MODEL_STACKS = {"gate_up_proj": "w13_weight", "down_proj": "w2_weight"}
LOGPROBS = 20


class SharedError(RuntimeError):
    """A step of the shared engine failed; the message names it."""


def check_architecture(model_name: str) -> None:
    path = Path(model_name) / "config.json"
    if not path.is_file():
        raise SharedError(f"milestone_engine='vllm' needs a local checkpoint with a config.json; {model_name} has none")
    found = json.loads(path.read_text()).get("architectures") or []
    if not any(a in ARCHITECTURES for a in found):
        raise SharedError(f"milestone_engine='vllm' is tested on {list(ARCHITECTURES)} only; the model is {found}")


def _drop_pages(filename) -> None:
    fd = os.open(filename, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class _ShardWithoutStacks:
    """A safetensors shard read to the GPU without its expert stacks, its pages dropped; stands in for the handle
    Transformers opens."""

    def __init__(self, safe_open, filename: str, device: str):
        with safe_open(filename, framework="pt", device=device) as f:
            self.tensors = {k: f.get_tensor(k) for k in f.keys() if not k.endswith(CHECKPOINT_STACKS)}
        _drop_pages(filename)

    def keys(self):
        return self.tensors.keys()

    def get_slice(self, key):
        return self.tensors[key]

    def get_tensor(self, key):
        return self.tensors[key]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.tensors = {}


@contextlib.contextmanager
def trainer_without_stacks(device: str):
    """While inside, Transformers loads a model without reading its expert stacks, which stay on meta."""
    from transformers import modeling_utils

    real_open = modeling_utils.safe_open
    real_move = modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device

    def safe_open(filename, *args, **kwargs):
        if "backend" in kwargs:
            return _ShardWithoutStacks(real_open, filename, device)
        return real_open(filename, *args, **kwargs)

    def move(self, missing_keys, *args, **kwargs):
        return real_move(self, [k for k in missing_keys if not k.endswith(tuple(MODEL_STACKS))], *args, **kwargs)

    modeling_utils.safe_open = safe_open
    modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = move
    try:
        yield
    finally:
        modeling_utils.safe_open = real_open
        modeling_utils.PreTrainedModel._move_missing_keys_from_meta_to_device = real_move


def _gib(n) -> str:
    return f"{n / 2**30:.2f} GiB"


class Engine:
    """vLLM in this process on the checkpoint at `model_name`, its expert stacks the trainer's."""

    def __init__(self, model_name: str, dtype: torch.dtype, fp8: bool, max_lora_rank: int, kv_cache_bytes: int,
                 max_model_len: int, device: torch.device):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        try:
            from vllm import LLM
            from vllm.model_executor.model_loader import default_loader
        except ImportError as e:
            raise SharedError(f"milestone_engine='vllm': vLLM is not installed ({e})")
        self.device, self.fp8, self.lora_id = device, fp8, 0
        self.memory = {"before_vllm": torch.cuda.memory_allocated(device)}
        torch.cuda.reset_peak_memory_stats(device)
        read = []

        def shard_by_shard(files, *args, **kwargs):
            from safetensors import safe_open

            for f in sorted(files):
                with safe_open(f, framework="pt", device=str(device)) as s:
                    tensors = {k: s.get_tensor(k) for k in s.keys()}
                _drop_pages(f)
                read.append(f)
                yield from tensors.items()
                del tensors

        real = default_loader.safetensors_weights_iterator
        default_loader.safetensors_weights_iterator = shard_by_shard
        kwargs = dict(model=model_name, dtype=str(dtype).removeprefix("torch."), enable_lora=True,
                      max_lora_rank=max_lora_rank, max_loras=1, enforce_eager=True,
                      kv_cache_memory_bytes=int(kv_cache_bytes), max_model_len=int(max_model_len),
                      language_model_only=True, seed=0)
        if fp8:
            kwargs.update(quantization="fp8", kernel_config={"moe_backend": "triton"})
        try:
            self.llm = LLM(**kwargs)
        except Exception as e:
            raise SharedError(f"milestone_engine='vllm': vLLM did not start: {type(e).__name__}: {e}") from e
        finally:
            default_loader.safetensors_weights_iterator = real
        if not read:
            raise SharedError("milestone_engine='vllm': vLLM did not read the checkpoint through meepsquared's reader")
        self.memory["after_vllm"] = torch.cuda.memory_allocated(device)
        self.memory["loading_peak"] = torch.cuda.max_memory_allocated(device)
        self.model = self.llm.apply_model(lambda m: m)[0]
        self.params = dict(self.model.named_parameters())
        self.stacks = {}
        for name, tensor in self.params.items():
            if name.endswith(".mlp.experts.base_layer.w13_weight") or name.endswith(".mlp.experts.w13_weight"):
                layer = int(name.split(".layers.")[1].split(".")[0])
                prefix = name[: -len("w13_weight")]
                found = {"gate_up_proj": tensor, "down_proj": self.params[prefix + "w2_weight"]}
                if fp8:
                    found["gate_up_proj_scale"] = self.params[prefix + "w13_weight_scale"]
                    found["down_proj_scale"] = self.params[prefix + "w2_weight_scale"]
                self.stacks[layer] = found
        if not self.stacks:
            raise SharedError("milestone_engine='vllm': vLLM's model has no expert stacks to share")

    def attach(self, model) -> int:
        """The trainer's expert stacks become vLLM's tensors; refused unless every layer shares their storage."""
        from meepsquared import _fp8

        layers = _decoder_layers(model)
        if sorted(self.stacks) != list(range(len(layers))):
            raise SharedError(f"a tensor cannot be shared: vLLM has expert stacks in layers {sorted(self.stacks)}, "
                              f"the trainer has {len(layers)} layers")
        for i, layer in enumerate(layers):
            experts = layer.mlp.experts
            for name, vllm_name in MODEL_STACKS.items():
                mine, theirs = getattr(experts, name), self.stacks[i][name]
                if tuple(mine.shape) != tuple(theirs.shape):
                    raise SharedError(f"a tensor cannot be shared: layer {i} {name} is {tuple(mine.shape)} in the "
                                      f"trainer and {tuple(theirs.shape)} in vLLM ({vllm_name})")
                if not mine.is_meta:
                    raise SharedError(f"a tensor cannot be shared: layer {i} {name} was loaded by the trainer")
                setattr(experts, name, torch.nn.Parameter(theirs, requires_grad=False))
                if self.fp8:
                    scale = self.stacks[i][name + "_scale"].reshape(-1).float()
                    if scale.numel() != theirs.shape[0]:
                        raise SharedError(f"a tensor cannot be shared: layer {i} {name} has {scale.numel()} scales in "
                                          f"vLLM, one per expert ({theirs.shape[0]}) is meepsquared's FP8 layout")
                    experts.register_buffer(name + _fp8.STACK_SCALE, scale)
            if self.fp8:
                experts._meepsquared_fp8 = True
        self.check_shared(model)
        self.memory["shared_stacks"] = sum(t.numel() * t.element_size() for s in self.stacks.values()
                                           for k, t in s.items() if not k.endswith("_scale"))
        return len(layers)

    def check_shared(self, model) -> None:
        """Every layer's stacks in the trainer and in vLLM use the same storage; raises otherwise."""
        for i, layer in enumerate(_decoder_layers(model)):
            experts = _base_experts(layer.mlp.experts)
            for name in MODEL_STACKS:
                mine, theirs = getattr(experts, name), self.stacks[i][name]
                if (mine.data_ptr() != theirs.data_ptr()
                        or mine.untyped_storage().data_ptr() != theirs.untyped_storage().data_ptr()):
                    raise SharedError(f"a tensor cannot be shared: layer {i} {name} is not vLLM's tensor")

    def checksum(self) -> float:
        """A sum over every shared stack, to show that training did not write to them."""
        with torch.no_grad():
            return float(sum(t.float().sum(dtype=torch.float64).item() for s in self.stacks.values()
                             for k, t in s.items()))

    def _lora(self, adapter_dir):
        from vllm.lora.request import LoRARequest

        if adapter_dir is None:
            return None
        if self.lora_id:
            self.llm.llm_engine.remove_lora(self.lora_id)
        self.lora_id += 1
        return LoRARequest(f"milestone-{self.lora_id}", self.lora_id, str(adapter_dir))

    def generate(self, prompts: list, caps: list, ends: set, adapter_dir) -> list:
        """The greedy answer to each prompt (token ids), up to its cap or an end token, with the adapter."""
        from vllm import SamplingParams, TokensPrompt

        params = [SamplingParams(temperature=0, max_tokens=cap, stop_token_ids=sorted(ends), detokenize=False)
                  for cap in caps]
        request = self._lora(adapter_dir)
        try:
            out = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, lora_request=request,
                                    use_tqdm=False)
        except Exception as e:
            raise SharedError(f"vLLM did not generate: {type(e).__name__}: {e}") from e
        return [list(o.outputs[0].token_ids) for o in out]

    def first_token_logprobs(self, prompts: list, adapter_dir) -> list:
        """For each prompt, {token: log-probability} of the LOGPROBS most likely first answer tokens."""
        from vllm import SamplingParams, TokensPrompt

        request = self._lora(adapter_dir)
        try:
            out = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                    SamplingParams(temperature=0, max_tokens=1, logprobs=LOGPROBS, detokenize=False),
                                    lora_request=request, use_tqdm=False)
        except Exception as e:
            raise SharedError(f"the adapter is refused: vLLM did not answer with it: {type(e).__name__}: {e}") from e
        return [{t: lp.logprob for t, lp in o.outputs[0].logprobs[0].items()} for o in out]


PEFT_PREFIX = "base_model.model."


@contextlib.contextmanager
def checkpoint_named(model, adapter_dir):
    """The adapter saved in PEFT's format at `adapter_dir`, its weights renamed to the checkpoint's names in a
    temporary folder, while inside: vLLM maps the checkpoint's names (model.language_model.layers...) to its modules,
    and an adapter under Transformers' text model's names (model.layers...) would match none of them."""
    from safetensors.torch import load_file, save_file

    from meepsquared import _vllm

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    probe = next(n for n, _ in base.named_parameters() if ".layers.0." in n)
    head, tail = probe.split(".layers.0.", 1)
    renamed_head = _vllm.checkpoint_name(base, probe)[: -len(".layers.0." + tail)]
    weights = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    out = {}
    for key, tensor in weights.items():
        if not key.startswith(PEFT_PREFIX + head + "."):
            raise SharedError(f"the adapter is refused: {key} is not under {PEFT_PREFIX + head}")
        out[PEFT_PREFIX + renamed_head + key[len(PEFT_PREFIX + head):]] = tensor
    folder = Path(tempfile.mkdtemp(prefix="meepsquared-vllm-adapter-"))
    try:
        save_file(out, str(folder / "adapter_model.safetensors"), metadata={"format": "pt"})
        shutil.copy(Path(adapter_dir) / "adapter_config.json", folder / "adapter_config.json")
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _decoder_layers(model):
    from meepsquared import _checkpointing

    return _checkpointing.layers(model)


def _base_experts(experts):
    """The experts module under PEFT's wrappers."""
    while hasattr(experts, "base_layer"):
        experts = experts.base_layer
    return experts


def kv_cache_bytes(config, tokens: int) -> int:
    """The KV cache of `tokens` tokens of the model's full-attention layers, in the model's dtype, with half as much
    again for vLLM's blocks of the linear-attention layers' state."""
    text = getattr(config, "text_config", config)
    full = sum(1 for t in text.layer_types if t == "full_attention")
    per_token = full * 2 * text.num_key_value_heads * text.head_dim * 2
    return int(tokens * per_token * 1.5)


def trainer_logprobs(model, prompts: list, ends: Optional[set] = None) -> list:
    """The trainer's log-probabilities of the first answer token for each prompt, adapter on, as
    first_token_logprobs gives vLLM's."""
    from meepsquared import _milestones

    found = []
    with _milestones._Inference(model), torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p], device=next(model.parameters()).device)
            logits = model(input_ids=ids).logits[0, -1].float()
            found.append(torch.log_softmax(logits, dim=-1))
    return found


def compare(trainer: list, vllm: list) -> dict:
    """The check: the same top token for every prompt, and the largest difference of log-probability over the tokens
    vLLM ranks first. Two top tokens that differ are a tie, not a difference, when the trainer gives vLLM's within
    LOGPROB_BOUND of its own: rounding decides between them."""
    worst, top_same, ties = 0.0, True, 0
    for mine, theirs in zip(trainer, vllm):
        best = max(theirs, key=theirs.get)
        if int(mine.argmax()) != best:
            if float(mine.max() - mine[best]) <= LOGPROB_BOUND:
                ties += 1
            else:
                top_same = False
        worst = max(worst, max(abs(float(mine[t]) - lp) for t, lp in theirs.items()))
    return {"top_same": top_same, "ties": ties, "max_logprob_diff": worst}


# The check's bound on the difference of log-probability between the trainer and vLLM, over the tokens vLLM ranks
# first, measured on Qwen3.6-35B-A3B (see design/milestones-vllm-shared-weights.md); and the spread of the random
# adapter's A and B, which must change the model's answer by more than the bound for the check to test anything
LOGPROB_BOUND = 0.25
RANDOM_STD = 0.02
CHECK_PROMPTS = 8


def _lora_parameters(model) -> dict:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def correctness(engine, model, tokenizer, prompts: list) -> dict:
    """The check of §6.1: on `prompts` (token ids), the first answer token's log-probabilities from the trainer and
    from vLLM, with the current adapter and with a random nonzero one saved and loaded the same way; the trainer's
    adapter is put back as it was. Raises SharedError with the figures when either differs."""
    from meepsquared import _save

    found = {}
    lora = _lora_parameters(model)
    if not lora:
        raise SharedError("the check needs the model's LoRA adapter; call it after get_peft_model")
    kept = {n: p.detach().clone() for n, p in lora.items()}
    base = None
    with tempfile.TemporaryDirectory(prefix="meepsquared-check-") as tmp:
        for label in ("current", "random"):
            if label == "random":
                generator = torch.Generator(device=next(iter(lora.values())).device).manual_seed(0)
                with torch.no_grad():
                    for p in lora.values():
                        p.copy_(torch.randn(p.shape, generator=generator, device=p.device, dtype=torch.float32)
                                .mul_(RANDOM_STD).to(p.dtype))
            try:
                folder = Path(tmp) / label
                _save.save(model, folder, save_method="lora")
                mine = trainer_logprobs(model, prompts)
                with checkpoint_named(model, folder) as named:
                    theirs = engine.first_token_logprobs(prompts, named)
            finally:
                if label == "random":
                    with torch.no_grad():
                        for n, p in lora.items():
                            p.copy_(kept[n])
            result = compare(mine, theirs)
            if base is None:
                base = mine
            result["effect"] = max(float((m - b).abs().max()) for m, b in zip(mine, base))
            found[label] = result
    failed = [label for label, r in found.items() if not r["top_same"] or r["max_logprob_diff"] > LOGPROB_BOUND]
    if found["random"]["effect"] <= LOGPROB_BOUND:
        raise SharedError(f"the check of the adapter tests nothing: the random adapter changed the trainer's "
                          f"log-probabilities by {found['random']['effect']:.3f}, not more than the bound "
                          f"{LOGPROB_BOUND}")
    if failed:
        raise SharedError("the check of §6.1 failed: vLLM and the trainer differ with the "
                          + " and the ".join(f"{label} adapter ({_figures(found[label])})" for label in failed)
                          + f"; the bound is {LOGPROB_BOUND} of log-probability and the same top token")
    return found


def _figures(result: dict) -> str:
    return (f"top token {'the same' if result['top_same'] else 'different'}"
            + (f" ({result['ties']} ties)" if result.get("ties") else "") + ", largest difference "
            f"{result['max_logprob_diff']:.4f}, the adapter's effect {result['effect']:.3f}")


def check_text(found: dict) -> str:
    return "; ".join(f"{label} adapter: {_figures(r)}" for label, r in found.items())


def log(memory: dict, label: str) -> None:
    if not os.environ.get("MEEPSQUARED_QUIET"):
        print(f"meepsquared: shared vLLM {label}: " + ", ".join(f"{k} {_gib(v)}" for k, v in memory.items()),
              flush=True)


def timer():
    return time.perf_counter()
