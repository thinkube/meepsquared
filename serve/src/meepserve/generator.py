# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The generator kind: text generation from a base held by the vllm engine, beside the choice parts on it.

Format 1: the part's folder holds its meepserve.json, and optionally a LoRA
adapter in PEFT's format (adapter_config.json, adapter_model.safetensors)
that generation reads with. meepserve.json names the base and its weights as
a choice part's does, and the engine "vllm": a generator and a choice part on
the same base share one copy of it.

    {"kind": "generator", "format": 1, "base": "google/gemma-4-12B-it", "weights": "nf4", "engine": "vllm"}

A request is an OpenAI chat completion (GeneratorPart.chat): messages
rendered by the base's chat template, max_tokens, temperature, top_p, stop,
chat_template_kwargs; the answer is OpenAI's chat.completion, without
streaming.
"""

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from meepserve import PartError

FORMATS = (1,)
ADAPTER_CONFIG, ADAPTER_FILE = "adapter_config.json", "adapter_model.safetensors"
# OpenAI's default sampling temperature, when a request names none
TEMPERATURE = 1.0
TOP_P = 1.0
FIELDS = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stop", "stream",
          "chat_template_kwargs", "n"}


class GeneratorPart:
    def __init__(self, name: str, adapter, engine):
        self.name, self.adapter, self.engine = name, adapter, engine
        self.closed = False

    def close(self) -> None:
        if self.closed:
            raise PartError("the part is closed already")
        if self.adapter is not None:
            self.engine.remove_lora(self.adapter)
        self.closed = True

    def answer(self, request: dict) -> dict:
        raise PartError("a generator answers chat completions (POST /v1/chat/completions), not /v1/heads")

    def chat(self, request: dict) -> dict:
        """An OpenAI chat completion: the base's answer to the messages, its chat template applied."""
        if self.closed:
            raise PartError("the part is closed")
        unknown = sorted(set(request) - FIELDS)
        if unknown:
            raise PartError(f"a generator takes {sorted(FIELDS)}, not {unknown}")
        if request.get("stream"):
            raise PartError("a generator answers without streaming: 'stream' is false or left out")
        if request.get("n", 1) != 1:
            raise PartError("a generator gives one answer: 'n' is 1 or left out")
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages or not all(
                isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                for m in messages):
            raise PartError("'messages' must be chat messages, each with a role and a text content")
        cap = request.get("max_completion_tokens", request.get("max_tokens"))
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise PartError("'max_tokens' is a whole number of 1 or more")
        temperature = request.get("temperature", TEMPERATURE)
        top_p = request.get("top_p", TOP_P)
        for field, value in (("temperature", temperature), ("top_p", top_p)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise PartError(f"'{field}' is a number of 0 or more")
        stop = request.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if stop is not None and (not isinstance(stop, list) or not all(isinstance(s, str) for s in stop)):
            raise PartError("'stop' is a text or a list of texts")
        kwargs = request.get("chat_template_kwargs") or {}
        if not isinstance(kwargs, dict):
            raise PartError("'chat_template_kwargs' is a mapping")
        prompt = self.engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                           **kwargs)
        out = self.engine.generate([prompt], cap, float(temperature), self.adapter, stop, float(top_p))[0]
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": request.get("model", self.name),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": out["text"]},
                             "finish_reason": out["finish_reason"]}],
                "usage": {"prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
                          "total_tokens": out["prompt_tokens"] + out["completion_tokens"]}}


def load(folder: Path, contract: dict, engine) -> GeneratorPart:
    """A generator of format 1 from `folder`, with its adapter when the folder holds one."""
    if not hasattr(engine, "generate"):
        raise PartError(f"{folder}: a generator runs on an engine that generates (vllm), not {engine.name}")
    if engine.weights != contract.get("weights"):
        raise PartError(f"{folder}: meepserve.json names the weights {contract.get('weights')!r}, and the engine holds "
                        f"the base with weights {engine.weights}")
    adapter = None
    if (folder / ADAPTER_CONFIG).is_file() or (folder / ADAPTER_FILE).is_file():
        from safetensors.torch import load_file

        config = json.loads((folder / ADAPTER_CONFIG).read_text())
        adapter = engine.add_lora(folder, int(config["r"]), config["lora_alpha"], config["target_modules"],
                                  load_file(str(folder / ADAPTER_FILE)))
    return GeneratorPart(contract.get("name", folder.name), adapter, engine)


@dataclass(frozen=True)
class _Kind:
    name: str
    formats: tuple
    engine: tuple

    def load(self, folder: Path, contract: dict, engine) -> GeneratorPart:
        return load(folder, contract, engine)


KIND = _Kind("generator", FORMATS, ("vllm",))
