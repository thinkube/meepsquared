# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The transformers engine: an encoder base with Transformers, and bottleneck adapters switched by name.

The base is loaded once, in float32 as the parts were trained on it; on a
GPU its products run under bfloat16 autocast, as they did when the parts were
evaluated. Bottleneck adapters (x + up(GELU(down(x))), Pfeiffer et al., 2020)
go after the feed-forward output projection of every layer; each part's
adapters are kept under its name, and the part names the ones active for its
reading, so several parts share one copy of the base.
"""

import contextlib
from typing import Optional

import torch
from torch import nn

# The model types whose layers have the feed-forward output projection (layer.output.dense) the adapters follow
BOTTLENECK_TYPES = ("xlm-roberta", "xlm-roberta-xl", "roberta", "bert", "camembert")


class Bottleneck(nn.Module):
    def __init__(self, hidden: int, size: int):
        super().__init__()
        self.down = nn.Linear(hidden, size)
        self.up = nn.Linear(size, hidden)

    def forward(self, x):
        return x + self.up(nn.functional.gelu(self.down(x)))


class _Slot(nn.Module):
    """A layer's feed-forward output projection, then the bottleneck adapter of the active name, if any."""

    def __init__(self, dense: nn.Linear):
        super().__init__()
        self.dense = dense
        self.adapters = nn.ModuleDict()
        self.active: Optional[str] = None

    def forward(self, x):
        out = self.dense(x)
        return self.adapters[self.active](out) if self.active is not None else out


class TransformersEncoder:
    """An encoder base (XLM-RoBERTa and its kind) and its tokenizer, on `device` (the first GPU when there is one)."""

    name = "transformers"

    def __init__(self, path, device: Optional[str] = None):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        config = AutoConfig.from_pretrained(path)
        if config.model_type not in BOTTLENECK_TYPES:
            raise ValueError(f"the transformers engine holds encoders whose layers have a feed-forward output "
                             f"projection ({', '.join(BOTTLENECK_TYPES)}); {path} is {config.model_type}")
        self.model = AutoModel.from_pretrained(path, config=config, add_pooling_layer=False).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.hidden = config.hidden_size
        self.slots = None

    def _install(self) -> list:
        if self.slots is None:
            self.slots = []
            for layer in self.model.encoder.layer:
                slot = _Slot(layer.output.dense)
                layer.output.dense = slot
                self.slots.append(slot)
        return self.slots

    def add_bottlenecks(self, name: str, size: int, state: dict) -> None:
        """Adapters of `size` under `name`, one per layer, from state {"<layer>.down.weight": tensor, ...}; every
        name of the state must be one of theirs and every one of theirs in the state."""
        slots = self._install()
        if name in slots[0].adapters:
            raise ValueError(f"the engine holds adapters named {name} already")
        expected = {f"{i}.{p}" for i in range(len(slots)) for p in ("down.weight", "down.bias", "up.weight", "up.bias")}
        if set(state) != expected:
            raise ValueError(f"the adapters of {name} do not fit the base's {len(slots)} layers: missing "
                             f"{sorted(expected - set(state))[:4]}, unknown {sorted(set(state) - expected)[:4]}")
        for i, slot in enumerate(slots):
            adapter = Bottleneck(self.hidden, size)
            adapter.load_state_dict({p: state[f"{i}.{p}"] for p in ("down.weight", "down.bias", "up.weight",
                                                                     "up.bias")})
            slot.adapters[name] = adapter.to(self.device).eval()

    @contextlib.contextmanager
    def reading(self, name: Optional[str]):
        """While inside: the adapters of `name` active (None: the base alone), no gradients, and bfloat16 autocast
        on a GPU."""
        slots = self.slots or []
        for slot in slots:
            slot.active = name
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda"
                    else contextlib.nullcontext())
        try:
            with torch.no_grad(), autocast:
                yield
        finally:
            for slot in slots:
                slot.active = None

    def states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The base's last hidden states; call it inside reading()."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
