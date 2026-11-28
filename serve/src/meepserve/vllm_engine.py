# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The vllm engine: a causal language model base held once by vLLM, for generation and for a choice head's states.

One vLLM instance holds the base, with LoRA enabled and
prefix caching on, and with vLLM's hidden-state extraction (the speculative
method extract_hidden_states, its states written per request by
ExampleHiddenStatesConnector): every request saves the hidden states of its
prompt's tokens after the last layer, of which meepserve.vllm_connector
keeps the block that holds the last token. A choice part reads, for each
candidate, the state of the last token of the shared text and the
candidate's own text, read with the part's adapter: the shared text is
computed once and found in the prefix cache by the other candidates. The
state is the last layer's output before the final norm, which this engine
applies as Transformers' text model does (meepserve.choice reads the
normed state).

vLLM's engine core runs in its own process, as vLLM runs it by default: the
connector writes a request's states in an engine step after the request
ends, and an engine core inside this process takes no step once generate
has returned, so a reader would wait on the states for ever.

Weights (the part's meepserve.json): "nf4", bitsandbytes' 4-bit NF4
quantized as vLLM loads the base (vllm-bnb-plugin), each weight as a QLoRA
training's Transformers model quantizes it (see register), computing in
bfloat16; "bf16", bfloat16.

Generation (generate): the base alone, or with an adapter, greedy or
sampled, as vLLM generates.

vLLM 0.32's interfaces this rests on: LLM(speculative_config=
{"method": "extract_hidden_states", "num_speculative_tokens": 1, ...},
kv_transfer_config=KVTransferConfig(kv_connector=
"ExampleHiddenStatesConnector", ...)); per-request paths through
SamplingParams.extra_args["kv_transfer_params"]["hidden_states_path"];
example_hidden_states_connector.load_hidden_states.
"""

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

import torch

from meepserve import PartError

WEIGHTS = ("nf4", "bf16")
# The cache vLLM keeps for keys, values and the extracted states, and the longest prompt it reads: a choice
# question of the subject finder reads up to about 10,000 tokens; one prompt of MAX_MODEL_LEN tokens of Gemma 4 12B
# takes 6.25 GiB of it (measured by vLLM 0.32 at start)
KV_CACHE_BYTES = 7 * 2 ** 30
MAX_MODEL_LEN = 16384
MAX_LORA_RANK = 64
MAX_LORAS = 4
# The block of weights each NF4 scale covers in a Transformers model loaded in 4 bits (bitsandbytes' Linear4bit)
NF4_BLOCKSIZE = 64
# The final norm's weight in the checkpoint, by the ends of the names a causal LM gives it
FINAL_NORMS = ("language_model.norm.weight", "model.norm.weight")


def _drop(path: str, cleanup) -> None:
    """A request's saved states removed once the connector's write of them is done (it holds the lock file)."""
    import fcntl

    lock = path + ".lock"
    if os.path.exists(lock):
        with open(lock) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
    cleanup(path)


def register() -> None:
    """meepserve's vLLM plugin (entry point vllm.general_plugins), which vLLM runs in each of its processes, the
    engine core's among them, so that these hold where the base is loaded:

    - the bitsandbytes plugin's loader is not run on the extraction's model (ExtractHiddenStatesModel): vLLM loads it
      with the base's loader, and the bitsandbytes loader refuses a model with no linear layer to quantize; it holds
      no weights, and its loading is nothing to do;
    - while that loader runs, bitsandbytes quantizes a weight as a QLoRA training's Transformers model does, NF4 in
      blocks of 64 with each block's scale in 32 bits (not quantized again): the plugin's in-flight quantization
      asks for the scales quantized, which gives other weights than the ones a part was trained on.

    Without the bitsandbytes plugin installed there is no loader to change: vLLM then refuses a 4-bit base itself."""
    try:
        from vllm_bnb_plugin.bitsandbytes_loader import BitsAndBytesModelLoader
    except ImportError:
        return
    if getattr(BitsAndBytesModelLoader, "_meepserve", False):
        return
    import bitsandbytes.functional as bnb_functional
    from vllm.model_executor.models.extract_hidden_states import ExtractHiddenStatesModel

    loading = threading.local()
    real_quantize = bnb_functional.quantize_4bit

    def quantize_4bit(A, *args, **kwargs):
        if getattr(loading, "active", False):
            if kwargs.get("quant_type", "fp4") != "nf4":
                raise RuntimeError(f"the bitsandbytes plugin asked for {kwargs.get('quant_type')}; meepserve's "
                                   "vllm engine holds NF4")
            kwargs["compress_statistics"] = False
            kwargs["blocksize"] = NF4_BLOCKSIZE
        return real_quantize(A, *args, **kwargs)

    real_load = BitsAndBytesModelLoader.load_weights

    def load_weights(self, model, model_config):
        if isinstance(model, ExtractHiddenStatesModel):
            return None
        loading.active = True
        try:
            return real_load(self, model, model_config)
        finally:
            loading.active = False

    bnb_functional.quantize_4bit = quantize_4bit
    BitsAndBytesModelLoader.load_weights = load_weights
    BitsAndBytesModelLoader._meepserve = True


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


class VllmBase:
    """A causal language model base held by vLLM, and its tokenizer."""

    name = "vllm"

    def __init__(self, path, weights: str, device: Optional[str] = None):
        if weights not in WEIGHTS:
            raise PartError(f"the vllm engine holds a base with weights {list(WEIGHTS)}, not {weights!r}")
        if not torch.cuda.is_available():
            raise PartError("the vllm engine runs on a CUDA GPU")
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.config.kv_transfer import KVTransferConfig

        self.path, self.weights = Path(path), weights
        self.device = torch.device(device if device is not None else "cuda")
        config = json.loads((self.path / "config.json").read_text())
        text = _text_config(config)
        self.layers, self.hidden = int(text["num_hidden_layers"]), int(text["hidden_size"])
        self.eps = float(text.get("rms_norm_eps", 1e-6))
        # On the CPU until vLLM has started its engine core: CUDA set up in this process first makes vLLM start it by
        # spawn, which imports the server's module again
        norm = self._final_norm()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), legacy=False)
        if self.tokenizer.pad_token_id is None:
            raise PartError(f"the tokenizer of {path} has no pad token")
        self.states_dir = Path(tempfile.mkdtemp(prefix="meepserve-states-", dir="/dev/shm"))
        kwargs = dict(
            model=str(self.path), dtype="bfloat16", enable_lora=True, max_lora_rank=MAX_LORA_RANK,
            max_loras=MAX_LORAS, enable_prefix_caching=True, kv_cache_memory_bytes=KV_CACHE_BYTES,
            max_model_len=MAX_MODEL_LEN, language_model_only=True, seed=0,
            speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1,
                                "draft_model_config": {"hf_config": {
                                    "eagle_aux_hidden_state_layer_ids": [self.layers]}}},
            kv_transfer_config=KVTransferConfig(
                kv_connector="LastStateConnector", kv_connector_module_path="meepserve.vllm_connector",
                kv_role="kv_producer",
                kv_connector_extra_config={"shared_storage_path": str(self.states_dir),
                                           "allow_custom_save_path": True}))
        if weights == "nf4":
            kwargs["quantization"] = "bitsandbytes"
            # meepserve's vLLM plugin (register) makes vLLM's 4-bit loading the training's, in vLLM's processes
        try:
            self.llm = LLM(**kwargs)
        except Exception as e:
            shutil.rmtree(self.states_dir, ignore_errors=True)
            raise PartError(f"vLLM did not start on {path}: {type(e).__name__}: {e}") from e
        self.norm = norm.to(self.device)
        self.adapters = {}  # name -> (id, folder)
        self._made = 0
        self._lock = threading.Lock()

    def _final_norm(self) -> torch.Tensor:
        from safetensors import safe_open

        for shard in sorted(self.path.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    if key.endswith(FINAL_NORMS) and "vision" not in key and "audio" not in key:
                        return f.get_tensor(key).float()
        raise PartError(f"{self.path} has no final norm weight ({' or '.join(FINAL_NORMS)}) in its safetensors files")

    # Adapters

    def add_lora(self, folder, rank: int, alpha, names: list, state: dict) -> str:
        """The LoRA adapter saved in PEFT's format in `folder`, under a name the engine gives; vLLM reads it from the
        folder at its first request (rank, alpha, names and state are the part's, checked by the kind)."""
        if rank > MAX_LORA_RANK:
            raise PartError(f"the adapter's rank is {rank}; the vllm engine takes up to {MAX_LORA_RANK}")
        if len(self.adapters) >= MAX_LORAS:
            raise PartError(f"the vllm engine holds {MAX_LORAS} adapters already")
        self._made += 1
        name = f"part-{self._made}"
        self.adapters[name] = (self._made, str(folder))
        return name

    def remove_lora(self, name: str) -> None:
        if name not in self.adapters:
            raise PartError(f"the engine holds no adapter named {name}")
        lora_id, _ = self.adapters.pop(name)
        self.llm.llm_engine.remove_lora(lora_id)

    def names(self) -> list:
        return list(self.adapters)

    def _request(self, name: Optional[str]):
        if name is None:
            return None
        from vllm.lora.request import LoRARequest

        if name not in self.adapters:
            raise PartError(f"the engine holds no adapter named {name}")
        lora_id, folder = self.adapters[name]
        return LoRARequest(name, lora_id, folder)

    # Reading and generating

    def candidate_states(self, adapter: str, shared: list, own: list, per_pass: int) -> torch.Tensor:
        """The state of the last token of the shared text and each candidate's own text: each read whole, the shared
        text found in the prefix cache after its first reading (per_pass does not apply: vLLM batches them)."""
        return self.last_states(adapter, [shared + o for o in own])

    def last_states(self, adapter: Optional[str], prompts: list) -> torch.Tensor:
        """The normed last-layer state of the last token of each prompt (token ids), read with `adapter`, in
        bfloat16 as Transformers' text model gives it."""
        from vllm import SamplingParams, TokensPrompt
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            cleanup_hidden_states, load_hidden_states)

        with self._lock:
            paths = [str(self.states_dir / f"{os.getpid()}-{threading.get_ident()}-{i}.safetensors")
                     for i in range(len(prompts))]
            params = [SamplingParams(temperature=0, max_tokens=1, detokenize=False,
                                     extra_args={"kv_transfer_params": {"hidden_states_path": p}}) for p in paths]
            try:
                out = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params,
                                        lora_request=self._request(adapter), use_tqdm=False)
            except Exception as e:
                raise PartError(f"vLLM did not read the candidates: {type(e).__name__}: {e}") from e
            found = []
            for prompt, o, path in zip(prompts, out, paths):
                saved = (o.kv_transfer_params or {}).get("hidden_states_path", path)
                try:
                    data = load_hidden_states(saved)
                finally:
                    cleanup_hidden_states(saved)
                # The rows of the block that holds the last prompt token, up to it (meepserve.vllm_connector)
                states, ids = data["hidden_states"], data["token_ids"]
                if (states.shape[0] != ids.shape[0] or states.shape[-1] != self.hidden or not ids.numel()
                        or ids.tolist() != prompt[len(prompt) - ids.shape[0]:]):
                    raise PartError(f"vLLM saved states of shape {tuple(states.shape)} for tokens that are not the end "
                                    f"of the prompt of {len(prompt)} tokens")
                found.append(states[-1, 0])
        last = torch.stack(found).to(self.device).float()
        normed = last * torch.pow(last.pow(2).mean(-1, keepdim=True) + self.eps, -0.5) * self.norm
        return normed.to(torch.bfloat16)

    def generate(self, prompts: list, max_tokens: int, temperature: float = 0.0, adapter: Optional[str] = None,
                 stop: Optional[list] = None) -> list:
        """The text vLLM generates for each prompt (text, the chat template already applied)."""
        from vllm import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            cleanup_hidden_states)

        with self._lock:
            paths = [str(self.states_dir / f"gen-{os.getpid()}-{threading.get_ident()}-{i}.safetensors")
                     for i in range(len(prompts))]
            params = [SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=stop or None,
                                     extra_args={"kv_transfer_params": {"hidden_states_path": p}}) for p in paths]
            try:
                out = self.llm.generate(prompts, params, lora_request=self._request(adapter), use_tqdm=False)
            except Exception as e:
                raise PartError(f"vLLM did not generate: {type(e).__name__}: {e}") from e
            for o, path in zip(out, paths):
                _drop((o.kv_transfer_params or {}).get("hidden_states_path", path), cleanup_hidden_states)
        return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                 "prompt_tokens": len(o.prompt_token_ids), "completion_tokens": len(o.outputs[0].token_ids)}
                for o in out]

    def close(self) -> None:
        shutil.rmtree(self.states_dir, ignore_errors=True)
