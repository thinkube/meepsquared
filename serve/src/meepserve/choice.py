# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The choice kind: a LoRA adapter and a head on a causal language model, choosing among candidates written after
one shared text.

Format 1, the files Meep²'s choice task saves (a milestone's or the
training's output folder):

- choice.json: "head" ("set" or "scalar"), "set_layers", "none_output",
  "method" ("qlora" or "lora", the weights the base was trained on) and
  "candidates_per_pass";
- adapter_config.json and adapter_model.safetensors: the LoRA adapter, in
  PEFT's format; its r, lora_alpha and target_modules are read, and the
  adapter goes on the language model's linear layers of those names, outside
  the vision, audio and image modules, as the training put it;
- head.safetensors: the head, every name required and none other allowed.

meepserve.json names the base's weights, which must be the training's:
"weights" "nf4" for method qlora, "bf16" for method lora (see
meepserve.causal).

A question is read as the training reads it: the chat template is rendered
with the last user message's content followed by "\\x00" and the generation
prompt; the text before the marker is the shared text, and each candidate
followed by the text after the marker is that candidate's own. The shared
tokens are read once with the cache on; the cache is copied for each group of
candidates_per_pass candidates, read after it right-padded, and each
candidate's score comes from the final hidden state of its last token. The
language model runs under bfloat16 autocast on a GPU; the head in float32.

Request and answer (see ChoicePart.answer):

    {"questions": [{"prompt": "..." or [messages], "candidates": ["...", ...], "none": index}, ...]}
    -> {"answers": [{"chosen": index, "probability": p, "probabilities": [p, ...]}, ...]}

"none", the index of the candidate that says none applies, is required by a
head with a none output and refused by one without.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from meepserve import CONTRACT, PartError

FORMATS = (1,)
CONFIG_FILE, HEAD_FILE = "choice.json", "head.safetensors"
ADAPTER_CONFIG, ADAPTER_FILE = "adapter_config.json", "adapter_model.safetensors"
HEADS = ("set", "scalar")
# The weights of the base each training method trained on
METHOD_WEIGHTS = {"qlora": "nf4", "lora": "bf16"}
# The marker the last user message's content ends with when the chat template is rendered
MARKER = "\x00"
# The set head's width and attention heads, as Meep² builds it
SET_WIDTH, SET_HEADS = 128, 4
# The none output's probability above which a question's answer is its none candidate
NONE_THRESHOLD = 0.5


# The head, as Meep²'s choice task builds and reads it

def set_attention_name(layer: int) -> str:
    return "set_attention" if layer == 1 else f"set_attention_{layer}"


def build_head(hidden: int, kind: str, set_layers: int = 1, none_output: bool = False) -> nn.ModuleDict:
    if kind not in HEADS:
        raise PartError(f"the head is one of {HEADS}, not {kind!r}")
    if set_layers < 1 or (kind != "set" and set_layers != 1):
        raise PartError(f"set_layers is 1 or more, and only the set head has more than 1, not {set_layers}")
    modules = {"norm": nn.LayerNorm(hidden), "scalar": nn.Linear(hidden, 1)}
    if kind == "set":
        modules["set_project"] = nn.Linear(hidden + 1, SET_WIDTH)
        for layer in range(1, set_layers + 1):
            modules[set_attention_name(layer)] = nn.MultiheadAttention(SET_WIDTH, SET_HEADS, dropout=0.0,
                                                                       batch_first=True)
        modules["set_output"] = nn.Linear(SET_WIDTH, 1)
    if none_output:
        modules["none_output"] = nn.Linear(2 * (SET_WIDTH if kind == "set" else hidden), 1)
    return nn.ModuleDict(modules)


def outputs(head: nn.ModuleDict, leaves: torch.Tensor) -> tuple:
    """(scores, states) of the candidates of one question from their last states, in float32."""
    with torch.autocast(leaves.device.type, enabled=False):
        h = head["norm"](leaves.float())
        z = head["scalar"](h).squeeze(-1)
        states = h
        if "set_project" in head:
            log_k = torch.full((len(h), 1), math.log(len(h)), device=h.device)
            u = head["set_project"](torch.cat([h, log_k], dim=-1))[None]
            layer = 1
            while set_attention_name(layer) in head:
                mixed, _ = head[set_attention_name(layer)](u, u, u, need_weights=False)
                u = torch.tanh(u + mixed)
                layer += 1
            z = z + head["set_output"](u).squeeze(-1)[0]
            states = u[0]
    return z, states


def none_logit(head: nn.ModuleDict, states: torch.Tensor, none: int) -> torch.Tensor:
    others = torch.cat([states[:none], states[none + 1:]])
    with torch.autocast(states.device.type, enabled=False):
        return head["none_output"](torch.cat([states[none], others.max(0).values]).float()).squeeze(-1)


def probabilities(head: nn.ModuleDict, z: torch.Tensor, states: torch.Tensor, none: Optional[int]) -> list:
    """Each candidate's probability: without a none output the softmax; with it, the none candidate's is the none
    output's probability, and each other's its softmax among them times the probability that one applies."""
    if "none_output" not in head:
        return [float(x) for x in z.float().softmax(-1)]
    p_none = float(torch.sigmoid(none_logit(head, states, none)))
    keep = [i for i in range(len(z)) if i != none]
    p = z[keep].float().softmax(-1)
    found = [0.0] * len(z)
    found[none] = p_none
    for i, x in zip(keep, p):
        found[i] = (1 - p_none) * float(x)
    return found


def choose(head: nn.ModuleDict, z: torch.Tensor, states: torch.Tensor, none: Optional[int]) -> tuple:
    """(the chosen candidate, its probability): the softmax's top candidate; with a none output, the none candidate
    when its probability is above 0.5, else the top of the others, its probability times the probability that one
    applies."""
    if "none_output" not in head:
        p = z.softmax(-1)
        best = int(p.argmax())
        return best, float(p[best])
    p_none = float(torch.sigmoid(none_logit(head, states, none)))
    if p_none > NONE_THRESHOLD:
        return none, p_none
    keep = [i for i in range(len(z)) if i != none]
    p = z[keep].softmax(-1)
    best = int(p.argmax())
    return keep[best], (1 - p_none) * float(p[best])


# The questions

def encode(tokenizer, prompt, candidates: list) -> tuple:
    """(shared tokens, each candidate's own tokens) of a question (see the module's docstring)."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else [dict(m) for m in prompt]
    messages[-1]["content"] = messages[-1]["content"] + MARKER
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    parts = full.split(MARKER)
    if len(parts) != 2:
        raise PartError(f"the chat template wrote the marker of the candidate {len(parts) - 1} times, not once")
    before, after = parts
    shared = tokenizer(before, add_special_tokens=False)["input_ids"]
    own = [tokenizer(c + after, add_special_tokens=False)["input_ids"] for c in candidates]
    if not shared or not all(own):
        raise PartError("a question's shared text or a candidate makes no tokens")
    return shared, own


def check_question(question, i: int, none_output: bool) -> None:
    at = f"question {i}"
    if not isinstance(question, dict):
        raise PartError(f"{at}: a question is a mapping, not {type(question).__name__}")
    unknown = sorted(set(question) - {"prompt", "candidates", "none"})
    if unknown:
        raise PartError(f"{at}: a question has 'prompt', 'candidates' and 'none', not {unknown}")
    prompt = question.get("prompt")
    if isinstance(prompt, list):
        if not prompt or not all(isinstance(m, dict) and isinstance(m.get("role"), str)
                                 and isinstance(m.get("content"), str) for m in prompt):
            raise PartError(f"{at}: 'prompt' as a list must be chat messages, each with a role and a text content")
        if prompt[-1]["role"] != "user":
            raise PartError(f"{at}: the last message of 'prompt' must be a user message, not {prompt[-1]['role']!r}")
        texts = [m["content"] for m in prompt]
    elif isinstance(prompt, str) and prompt:
        texts = [prompt]
    else:
        raise PartError(f"{at}: 'prompt' must be a text or a list of chat messages")
    candidates = question.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2 or not all(isinstance(c, str) and c
                                                                          for c in candidates):
        raise PartError(f"{at}: 'candidates' must be a list of at least 2 texts")
    if any(MARKER in t for t in texts + candidates):
        raise PartError(f"{at}: the texts hold the character \\x00, which marks where the candidate is written")
    none = question.get("none")
    if none_output:
        if isinstance(none, bool) or not isinstance(none, int) or not 0 <= none < len(candidates):
            raise PartError(f"{at}: 'none' is required by this head's none output: the index of one of the "
                            f"{len(candidates)} candidates, the one that says none applies")
    elif none is not None:
        raise PartError(f"{at}: 'none' is for a head with a none output; this one has none")


class ChoicePart:
    """A choice adapter and head on its engine: answer(request) gives each question's chosen candidate."""

    def __init__(self, adapter: str, head: nn.ModuleDict, per_pass: int, engine):
        self.adapter, self.head, self.per_pass, self.engine = adapter, head, per_pass, engine
        self.none_output = "none_output" in head

    def ask(self, questions: list, give_states: bool = False, adapter: bool = True) -> list:
        tokenizer = self.engine.tokenizer
        found = []
        with torch.no_grad():
            for question in questions:
                shared, own = encode(tokenizer, question["prompt"], question["candidates"])
                leaves = self.engine.candidate_states(self.adapter if adapter else None, shared, own, self.per_pass)
                z, states = outputs(self.head, leaves)
                none = question.get("none")
                best, probability = choose(self.head, z, states, none)
                answer = {"chosen": best, "probability": probability,
                          "probabilities": probabilities(self.head, z, states, none)}
                if give_states:
                    answer["scores"] = [float(x) for x in z]
                    answer["states"] = leaves.float().cpu().tolist()
                found.append(answer)
        return found

    def close(self) -> None:
        """The part's adapter removed from its engine; the part answers no more."""
        if self.head is None:
            raise PartError("the part is closed already")
        self.engine.remove_lora(self.adapter)
        self.head = None

    def answer(self, request: dict) -> dict:
        """{"questions": [...]} -> {"answers": [...]} (see the module's docstring). To compare two engines' reading of
        the same part: "states": true adds to each answer the head's scores and the state it read of each candidate
        (the base's normed last state, float32); "adapter": false reads with the base alone, the adapter off."""
        if self.head is None:
            raise PartError("the part is closed")
        given = {k: v for k, v in request.items() if k != "model"}
        unknown = sorted(set(given) - {"questions", "states", "adapter"})
        if unknown:
            raise PartError(f"a choice part takes 'questions', 'states' and 'adapter', not {unknown}")
        give_states, adapter = given.get("states", False), given.get("adapter", True)
        if not isinstance(give_states, bool) or not isinstance(adapter, bool):
            raise PartError("'states' and 'adapter' are true or false")
        questions = given.get("questions")
        if not isinstance(questions, list) or not questions:
            raise PartError("'questions' must be a list of questions")
        for i, question in enumerate(questions):
            check_question(question, i, self.none_output)
        return {"answers": self.ask(questions, give_states, adapter)}


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load(folder: Path, contract: dict, engine) -> ChoicePart:
    """A choice part of format 1 from `folder`, its adapter added to `engine`."""
    from safetensors.torch import load_file

    missing = [n for n in (CONFIG_FILE, HEAD_FILE, ADAPTER_CONFIG, ADAPTER_FILE) if not (folder / n).is_file()]
    if missing:
        raise PartError(f"{folder} has no {', '.join(missing)}: a choice part is the folder Meep²'s choice task saves")
    config = json.loads((folder / CONFIG_FILE).read_text())
    method = config.get("method")
    if method not in METHOD_WEIGHTS:
        raise PartError(f"{folder}/{CONFIG_FILE}: 'method' is one of {sorted(METHOD_WEIGHTS)}, not {method!r}")
    if contract.get("weights") != METHOD_WEIGHTS[method]:
        raise PartError(f"{folder}: trained with method {method}, on weights {METHOD_WEIGHTS[method]}, and "
                        f"{CONTRACT} names the weights {contract.get('weights')!r}")
    from meepserve import engine_name

    if engine.name != engine_name(contract):
        raise PartError(f"{folder}: {CONTRACT} names the engine {engine_name(contract)}, and the base is held by the "
                        f"{engine.name} engine")
    if engine.weights != METHOD_WEIGHTS[method]:
        raise PartError(f"{folder}: trained on weights {METHOD_WEIGHTS[method]}, and the engine holds the base with "
                        f"weights {engine.weights}")
    kind, set_layers, per_pass = config.get("head"), config.get("set_layers", 1), config.get("candidates_per_pass")
    none_output = config.get("none_output", False)
    if not _whole(set_layers) or not _whole(per_pass) or per_pass < 1 or not isinstance(none_output, bool):
        raise PartError(f"{folder}/{CONFIG_FILE}: 'set_layers' and 'candidates_per_pass' are whole numbers, "
                        f"candidates_per_pass 1 or more, and 'none_output' true or false")
    adapter_config = json.loads((folder / ADAPTER_CONFIG).read_text())
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("use_dora") or adapter_config.get(
            "use_rslora"):
        raise PartError(f"{folder}/{ADAPTER_CONFIG}: a choice part's adapter is plain LoRA (no DoRA, no rsLoRA)")
    rank, alpha, names = adapter_config.get("r"), adapter_config.get("lora_alpha"), adapter_config.get(
        "target_modules")
    if not _whole(rank) or not isinstance(alpha, (int, float)) or not isinstance(names, list) or not names:
        raise PartError(f"{folder}/{ADAPTER_CONFIG}: 'r', 'lora_alpha' and 'target_modules' are required")
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise PartError(f"{folder}/{ADAPTER_CONFIG}: an adapter with a rank or alpha per layer is not read here")
    head = build_head(engine.hidden, kind, set_layers, none_output)
    saved = load_file(str(folder / HEAD_FILE))
    own = dict(head.named_parameters())
    if set(saved) != set(own):
        raise PartError(f"{folder}: the saved head is not a {kind} head of {set_layers} layers: missing "
                        f"{sorted(set(own) - set(saved))[:4]}, unknown {sorted(set(saved) - set(own))[:4]}")
    with torch.no_grad():
        for name, value in saved.items():
            if value.shape != own[name].shape:
                raise PartError(f"{folder}: the head's {name} is {tuple(value.shape)}, and the base makes it "
                                f"{tuple(own[name].shape)}")
            own[name].copy_(value.float())
    adapter = engine.add_lora(folder, rank, alpha, names, load_file(str(folder / ADAPTER_FILE)))
    return ChoicePart(adapter, head.to(engine.device).eval(), per_pass, engine)


@dataclass(frozen=True)
class _Kind:
    name: str
    formats: tuple
    engine: str

    def load(self, folder: Path, contract: dict, engine) -> ChoicePart:
        return load(folder, contract, engine)


# The engines a choice part runs on; its meepserve.json's "engine" names one, "causal" when it names none
KIND = _Kind("choice", FORMATS, ("causal", "vllm"))
