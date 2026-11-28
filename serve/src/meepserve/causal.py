# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The causal engine: a causal language model base with Transformers, and LoRA adapters switched by name.

The base is loaded as a choice training of Meep² loads it, so that an adapter
trained on its weights reads the same weights here:

- weights "nf4" (Meep²'s method qlora): 4-bit NF4 without double
  quantization, computing in bfloat16, with bitsandbytes;
- weights "bf16" (Meep²'s method lora): bfloat16.

Attention is PyTorch's SDPA, as the choice training reads its candidates
after the shared cache with a mask; a choice part reads them through
candidate_states. The adapters are LoRA (PEFT) on the language model's
linear layers named by the part, outside the vision, audio and image
modules; each part's adapter is kept under a name the engine gives, so
several parts share one copy of the base.
"""

import contextlib
import copy
from typing import Optional

import torch
from torch import nn

from meepserve import PartError

WEIGHTS = ("nf4", "bf16")
# The modules of a multimodal base a choice adapter leaves out (their names contain one of these)
TOWERS = ("vision", "audio", "image")


class CausalBase:
    """A causal language model base and its tokenizer, on `device` (the first GPU when there is one)."""

    name = "causal"

    def __init__(self, path, weights: str, device: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if weights not in WEIGHTS:
            raise PartError(f"the causal engine holds a base with weights {list(WEIGHTS)}, not {weights!r}")
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights = weights
        if weights == "nf4":
            if self.device.type != "cuda":
                raise PartError("weights nf4 are bitsandbytes' 4-bit weights, which run on a CUDA GPU")
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.bfloat16,
                                              bnb_4bit_use_double_quant=False)
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                              quantization_config=quantization,
                                                              device_map={"": index})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                              attn_implementation="sdpa").to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, legacy=False)
        if self.tokenizer.pad_token_id is None:
            raise PartError(f"the tokenizer of {path} has no pad token, which pads the candidates read together")
        self.peft = None
        self._made = 0

    @property
    def hidden(self) -> int:
        return self.decoder().config.hidden_size

    def decoder(self):
        """The text model inside the base (and inside its adapters, when it has some)."""
        model = self.peft.get_base_model() if self.peft is not None else self.model
        return model.get_decoder()

    def targets(self, names: list) -> list:
        """The full names of the linear layers named one of `names`, outside the vision, audio and image modules. A
        layer that holds adapters already is named as it was before PEFT wrapped it (its linear layer is then its
        base_layer)."""
        found = []
        for n, m in self.model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            n = n.removesuffix(".base_layer")
            if n.rsplit(".", 1)[-1] in names and not any(tower in n for tower in TOWERS):
                found.append(n)
        return found

    def add_lora(self, folder, rank: int, alpha, names: list, state: dict) -> str:
        """A LoRA adapter of `rank` and `alpha` on the layers `names` (see targets), its weights `state` as PEFT
        saves them; every name of the state must be the adapter's and every one of the adapter's in the state.
        Returns the adapter's name in the engine."""
        from peft import LoraConfig, get_peft_model
        from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

        targets = self.targets(names)
        if not targets:
            raise PartError(f"no linear layer of the base's language model is named one of {names}")
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=targets,
                            task_type="CAUSAL_LM")
        self._made += 1
        name = f"part-{self._made}"
        if self.peft is None:
            self.peft = get_peft_model(self.model, config, adapter_name=name)
        else:
            self.peft.add_adapter(name, config)
        expected = set(get_peft_model_state_dict(self.peft, adapter_name=name))
        if set(state) != expected:
            missing, unknown = sorted(expected - set(state)), sorted(set(state) - expected)
            self.remove_lora(name)
            raise PartError(f"the adapter does not fit the base: missing {missing[:4]}, unknown {unknown[:4]}")
        result = set_peft_model_state_dict(self.peft, state, adapter_name=name)
        if result.unexpected_keys:
            self.remove_lora(name)
            raise PartError(f"the adapter has weights the base has not: {result.unexpected_keys[:4]}")
        for p in self.peft.parameters():
            p.requires_grad_(False)
        self.peft.eval()
        return name

    def remove_lora(self, name: str) -> None:
        """The adapter `name` removed from the base; the last one removed leaves the base as it was loaded."""
        if self.peft is None or name not in self.peft.peft_config:
            raise PartError(f"the engine holds no adapter named {name}")
        if len(self.peft.peft_config) == 1:
            self.model = self.peft.unload()
            self.peft = None
        else:
            self.peft.delete_adapter(name)

    def names(self) -> list:
        """The names of the adapters the engine holds."""
        return list(self.peft.peft_config) if self.peft is not None else []

    def _suffix(self, decoder, cache, prefix_len: int, own: list) -> torch.Tensor:
        c = copy.copy(cache)
        c.layers = [copy.copy(layer) for layer in cache.layers]
        c.batch_repeat_interleave(len(own))
        width = max(map(len, own))
        pad = self.tokenizer.pad_token_id
        x = torch.tensor([o + [pad] * (width - len(o)) for o in own], device=self.device)
        m = torch.tensor([[1] * (prefix_len + len(o)) + [0] * (width - len(o)) for o in own], device=self.device)
        h = decoder(input_ids=x, attention_mask=m, past_key_values=c, use_cache=True).last_hidden_state
        return h[torch.arange(len(own), device=self.device), torch.tensor([len(o) - 1 for o in own],
                                                                           device=self.device)]

    def candidate_states(self, adapter: Optional[str], shared: list, own: list, per_pass: int) -> torch.Tensor:
        """The last state of each candidate's reading with `adapter`: the shared tokens read once with the cache on,
        the cache copied for each group of per_pass candidates, read after it right-padded."""
        with self.reading(adapter) as decoder:
            out = decoder(input_ids=torch.tensor([shared], device=self.device), use_cache=True)
            cache = out.past_key_values
            return torch.cat([self._suffix(decoder, cache, len(shared), own[s:s + per_pass])
                              for s in range(0, len(own), per_pass)])

    @contextlib.contextmanager
    def reading(self, name: Optional[str]):
        """While inside: the adapter `name` active (None: the base alone, every adapter off), no gradients, and
        bfloat16 autocast on a GPU; gives the decoder."""
        autocast = torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")
        if name is None:
            off = self.peft.disable_adapter() if self.peft is not None else contextlib.nullcontext()
            with off, torch.no_grad(), autocast:
                yield self.decoder()
            return
        if self.peft is None or name not in self.peft.peft_config:
            raise PartError(f"the engine holds no adapter named {name}")
        self.peft.set_adapter(name)
        for p in self.peft.parameters():
            p.requires_grad_(False)
        with torch.no_grad(), autocast:
            yield self.decoder()
