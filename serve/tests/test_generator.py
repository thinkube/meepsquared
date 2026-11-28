# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The generator kind's requests and answers, on a stand-in engine that generates (the vllm engine needs a GPU)."""

import json

import pytest

from meepserve import PartError
from meepserve.generator import load


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + ("|model:" if add_generation_prompt else "")


class Engine:
    name, weights = "vllm", "nf4"

    def __init__(self):
        self.tokenizer, self.asked = Tokenizer(), []

    def generate(self, prompts, max_tokens, temperature, adapter, stop, top_p):
        self.asked.append((prompts, max_tokens, temperature, adapter, stop, top_p))
        return [{"text": "hola", "finish_reason": "stop", "prompt_tokens": 7, "completion_tokens": 2}]


def part(tmp_path, engine=None):
    folder = tmp_path / "g"
    folder.mkdir()
    contract = {"kind": "generator", "format": 1, "base": "test/causal", "weights": "nf4", "engine": "vllm"}
    (folder / "meepserve.json").write_text(json.dumps(contract))
    return load(folder, contract, engine or Engine())


def test_a_chat_completion(tmp_path):
    engine = Engine()
    g = part(tmp_path, engine)
    out = g.chat({"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5,
                  "temperature": 0, "stop": "\n"})
    assert out["object"] == "chat.completion" and out["model"] == "m"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hola"}
    assert out["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    assert engine.asked == [(["user:hello|model:"], 5, 0.0, None, ["\n"], 1.0)]


def test_refusals(tmp_path):
    g = part(tmp_path)
    for request, message in (({"messages": []}, "chat messages"),
                             ({"messages": [{"role": "user", "content": "a"}], "stream": True}, "streaming"),
                             ({"messages": [{"role": "user", "content": "a"}], "max_tokens": 0}, "max_tokens"),
                             ({"messages": [{"role": "user", "content": "a"}], "temperature": -1}, "temperature"),
                             ({"messages": [{"role": "user", "content": "a"}], "tools": []}, "not \\['tools'\\]")):
        with pytest.raises(PartError, match=message):
            g.chat(request)
    with pytest.raises(PartError, match="chat completions"):
        g.answer({"questions": []})


def test_a_generator_needs_an_engine_that_generates(tmp_path):
    class Encoder:
        name, weights = "transformers", "nf4"

    with pytest.raises(PartError, match="engine that generates"):
        part(tmp_path, Encoder())
