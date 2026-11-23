# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Choosing among candidates written after one shared text: a causal language model with LoRA and a head that
scores each candidate (TrainingJob(task="choice"), see meepsquared._choice_job).

A row is one question:

- "prompt": the shared text, a string (one user message), or a list of chat
  messages whose last one is a user message;
- "candidates": K >= 2 strings; candidate c is appended to the content of the
  last user message;
- "target": K numbers of at least 0 that sum to 1, the probability the
  training moves the scores towards;
- "right": K booleans, which candidates are right (a milestone counts how
  often the top candidate is one of them);
- "group": optional, a text that groups the rows of a milestone's score.

A row is encoded once (encode): the chat template is rendered with the last
user message's content followed by "\\x00" and the generation prompt; the
text before the marker is the shared text, and each candidate followed by the
text after the marker is that candidate's own. The score of a candidate comes
from the final hidden state of the last token of shared + own.

Reading (Chooser.leaves): the shared tokens are read once with the cache on;
the cache is copied for each group of candidates_per_pass candidates, which
are read after it, right-padded. When training, the shared reading and each
group's reading run under torch.utils.checkpoint (non-reentrant), so both are
read again in the backward pass instead of keeping their activations. The
language model runs under bf16 autocast on a GPU; the head runs in float32.

The head ("set", NanoJev's DecisionModel, or "scalar"): a LayerNorm and a
linear score; "set" adds an attention over the candidates of the question,
whose change starts at 0. The loss of a question is the cross-entropy of the
softmax of its scores against its target, over all its candidates.

What is saved (save, load): the adapter in PEFT's format, the head
(HEAD_FILE) and CONFIG_FILE, the head kind and the model.
"""

import json
import math
import os
import statistics
from pathlib import Path
from typing import Optional

import torch
from torch import nn

HEADS = ("set", "scalar")
# The set attention's width and heads (NanoJev)
SET_WIDTH = 128
SET_HEADS = 4
# The files of a saved chooser, beside the adapter's
HEAD_FILE, CONFIG_FILE = "head.safetensors", "choice.json"
# The files of a preprocessing folder (see preprocess), beside meepsquared._preprocess's key and report
TRAIN_FILE, EVAL_FILE = "train.jsonl", "eval.jsonl"
# A milestone's answers
ANSWERS = "answers.jsonl"
# How far a target's sum may be from 1
SUM_TOLERANCE = 1e-6
# The marker the last user message's content ends with when the chat template is rendered
MARKER = "\x00"
# The shape of a preprocessed folder's files and report; part of its key, so a folder of another shape is not read
FORMAT = 2
# The prefix of the head's parameters among the trainable ones, in a saved state
HEAD_PREFIX = "head."
# The keys of a start_from file, and of each parameter's optimizer state in it
START_KEYS = ("examples", "trainable", "optimizer")
ADAM_KEYS = ("step", "exp_avg", "exp_avg_sq")
# The language models' modules a choice adapter leaves out (their names contain one of these)
TOWERS = ("vision", "audio", "image")
GROUP = "all"


# The rows

def read_rows(data, where: str) -> tuple:
    """(rows, source): the rows of a .jsonl or .parquet file, or of a list of such files, in their order."""
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError(f"{where}: an empty list of files")
        rows, sources = [], []
        for item in data:
            if not isinstance(item, (str, os.PathLike)):
                raise ValueError(f"{where}: a list holds files, not {type(item).__name__}")
            part, source = read_rows(item, where)
            rows += part
            sources.append(source)
        return rows, " + ".join(sources)
    if not isinstance(data, (str, os.PathLike)):
        raise ValueError(f"{where}: a .jsonl or .parquet file, or a list of them, not {type(data).__name__}")
    path = Path(data)
    if not path.is_file():
        raise FileNotFoundError(f"{where}: no file {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(str(path)).to_pylist()
    else:
        raise ValueError(f"{where}: {path} is not a .jsonl or .parquet file")
    if not rows:
        raise ValueError(f"{where}: {path} has no rows")
    return rows, path.name


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_row(row, i: int, where: str, target: bool, right: bool) -> None:
    """Refuse a row that is not a question as the module's docstring says; `target` and `right` say which of those
    fields the row must have."""
    at = f"{where} row {i}"
    if not isinstance(row, dict):
        raise ValueError(f"{at}: a row is a mapping, not {type(row).__name__}")
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        if not prompt or not all(isinstance(m, dict) and isinstance(m.get("role"), str)
                                 and isinstance(m.get("content"), str) for m in prompt):
            raise ValueError(f"{at}: 'prompt' as a list must be chat messages, each with a role and a text content")
        if prompt[-1]["role"] != "user":
            raise ValueError(f"{at}: the last message of 'prompt' must be a user message, not {prompt[-1]['role']!r}")
    elif not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{at}: 'prompt' must be a text or a list of chat messages")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2 or not all(
            isinstance(c, str) and c for c in candidates):
        raise ValueError(f"{at}: 'candidates' must be a list of at least 2 texts")
    texts = [prompt] if isinstance(prompt, str) else [m["content"] for m in prompt]
    if any(MARKER in t for t in texts + candidates):
        raise ValueError(f"{at}: the texts hold the character \\x00, which marks where the candidate is written")
    k = len(candidates)
    given = row.get("target")
    if given is None:
        if target:
            raise ValueError(f"{at}: 'target' is required, {k} numbers of at least 0 that sum to 1")
    elif not isinstance(given, list) or len(given) != k or not all(_number(x) and x >= 0 for x in given):
        raise ValueError(f"{at}: 'target' must be {k} numbers of at least 0, one per candidate")
    elif abs(sum(given) - 1) > SUM_TOLERANCE:
        raise ValueError(f"{at}: 'target' sums to {sum(given)!r}, not 1")
    marks = row.get("right")
    if marks is None:
        if right:
            raise ValueError(f"{at}: 'right' is required, {k} booleans saying which candidates are right")
    elif not isinstance(marks, list) or len(marks) != k or not all(isinstance(x, bool) for x in marks):
        raise ValueError(f"{at}: 'right' must be {k} booleans, one per candidate")
    if row.get("group") is not None and not isinstance(row["group"], str):
        raise ValueError(f"{at}: 'group' must be a text")


def encode(tokenizer, row: dict) -> dict:
    """The row's tokens: the shared ones and each candidate's own (see the module's docstring), and its target, right
    candidates and group as given."""
    prompt = row["prompt"]
    messages = ([{"role": "user", "content": prompt}] if isinstance(prompt, str)
                else [dict(m) for m in prompt])
    messages[-1]["content"] = messages[-1]["content"] + MARKER
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    parts = full.split(MARKER)
    if len(parts) != 2:
        raise ValueError(f"the chat template wrote the marker of the candidate {len(parts) - 1} times, not once")
    before, after = parts
    shared = tokenizer(before, add_special_tokens=False)["input_ids"]
    own = [tokenizer(c + after, add_special_tokens=False)["input_ids"] for c in row["candidates"]]
    if not shared or not all(own):
        raise ValueError("a question's shared text or a candidate makes no tokens")
    return {"shared": shared, "own": own, "target": row.get("target"), "right": row.get("right"),
            "group": row.get("group")}


def read_encoded(data, tokenizer, where: str, target: bool, right: bool) -> tuple:
    """(encoded rows, source) of `data` (see read_rows), each row checked; `data` may also be a list of rows."""
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        rows, source = data, "the rows given"
    else:
        rows, source = read_rows(data, where)
    encoded = []
    for i, row in enumerate(rows):
        check_row(row, i, where, target, right)
        try:
            encoded.append(encode(tokenizer, row))
        except ValueError as error:
            raise ValueError(f"{where} row {i}: {error}") from error
    return encoded, source


def tokens_read(row: dict) -> int:
    """The tokens the model reads for a question: the shared ones once, and every candidate's own."""
    return len(row["shared"]) + sum(len(o) for o in row["own"])


# The head and the model

def build_head(hidden: int, kind: str) -> nn.ModuleDict:
    """The head of `kind` for states of `hidden` dimensions, its modules named as a saved state names them."""
    if kind not in HEADS:
        raise ValueError(f"the head is one of {HEADS}, not {kind!r}")
    modules = {"norm": nn.LayerNorm(hidden), "scalar": nn.Linear(hidden, 1)}
    if kind == "set":
        modules.update(set_project=nn.Linear(hidden + 1, SET_WIDTH),
                       set_attention=nn.MultiheadAttention(SET_WIDTH, SET_HEADS, dropout=0.0, batch_first=True),
                       set_output=nn.Linear(SET_WIDTH, 1))
    head = nn.ModuleDict(modules)
    nn.init.normal_(head["scalar"].weight, std=0.02)
    nn.init.zeros_(head["scalar"].bias)
    if kind == "set":
        nn.init.zeros_(head["set_output"].weight)
        nn.init.zeros_(head["set_output"].bias)
    return head


def scores(head: nn.ModuleDict, leaves: torch.Tensor) -> torch.Tensor:
    """The head on the last states of the candidates of one question: the scalar score, plus with "set" the change
    the attention over the candidates makes. In float32."""
    with torch.autocast(leaves.device.type, enabled=False):
        h = head["norm"](leaves.float())
        z = head["scalar"](h).squeeze(-1)
        if "set_project" in head:
            log_k = torch.full((len(h), 1), math.log(len(h)), device=h.device)
            u = head["set_project"](torch.cat([h, log_k], dim=-1))[None]
            mixed, _ = head["set_attention"](u, u, u, need_weights=False)
            z = z + head["set_output"](torch.tanh(u + mixed)).squeeze(-1)[0]
    return z


class Chooser(nn.Module):
    """The language model with its adapter (a PEFT model), its decoder, and the head.

    peft and head are its modules; the decoder (the text model inside peft) is
    held apart, not registered twice. adapter_names: the adapter's parameters,
    by their PEFT names, in named_parameters() order.
    """

    def __init__(self, peft, decoder, head: nn.ModuleDict, kind: str, per_pass: int, pad_id: int):
        super().__init__()
        self.peft = peft
        self.head = head
        object.__setattr__(self, "decoder", decoder)
        self.kind, self.per_pass, self.pad_id = kind, per_pass, pad_id
        self.adapter_names = [n for n, p in peft.named_parameters() if p.requires_grad]
        if not self.adapter_names:
            raise ValueError("the adapter has no trainable parameters: no module matched the LoRA targets")
        parameters = dict(peft.named_parameters())
        # A plain list: the parameters are registered once, under peft
        self._adapter = [parameters[n] for n in self.adapter_names]
        base = peft.get_base_model()
        # Accelerate leaves a 4-bit model where it was loaded, by these attributes
        self.is_loaded_in_4bit = bool(getattr(base, "is_loaded_in_4bit", False))
        if getattr(base, "hf_device_map", None) is not None:
            self.hf_device_map = base.hf_device_map

    @property
    def device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def adapter(self) -> list:
        return list(self._adapter)

    def trainable(self) -> dict:
        """{name: parameter} of everything that trains: the adapter by its PEFT names, the head as head.<path>."""
        parameters = dict(self.peft.named_parameters())
        found = {n: parameters[n] for n in self.adapter_names}
        found.update({HEAD_PREFIX + n: p for n, p in self.head.named_parameters()})
        return found

    def forward(self, shared: list, own: list, target: list, grad: bool) -> torch.Tensor:
        """The loss of one question (see question_loss)."""
        return self.question_loss(shared, own, target, grad)

    def _suffix(self, cache, prefix_len: int, own: list) -> torch.Tensor:
        """The last states of the candidates `own`, read after the shared cache."""
        import copy

        device = self.device
        c = copy.copy(cache)
        c.layers = [copy.copy(layer) for layer in cache.layers]
        c.batch_repeat_interleave(len(own))
        width = max(map(len, own))
        x = torch.tensor([o + [self.pad_id] * (width - len(o)) for o in own], device=device)
        m = torch.tensor([[1] * (prefix_len + len(o)) + [0] * (width - len(o)) for o in own], device=device)
        h = self.decoder(input_ids=x, attention_mask=m, past_key_values=c, use_cache=True).last_hidden_state
        return h[torch.arange(len(own), device=device), torch.tensor([len(o) - 1 for o in own], device=device)]

    def leaves(self, shared: list, own: list, grad: bool) -> torch.Tensor:
        """The last state of each candidate's reading, the shared text read once (see the module's docstring)."""
        from torch.utils.checkpoint import checkpoint

        device = self.device
        holder = []

        def prefix(*_):
            out = self.decoder(input_ids=torch.tensor([shared], device=device), use_cache=True)
            holder[:] = [out.past_key_values]
            return tuple(t for layer in out.past_key_values.layers for t in (layer.keys, layer.values)
                         if t is not None)

        if grad:  # the shared text is read again in the backward pass, not kept
            dummy = torch.zeros(1, requires_grad=True, device=device)
            checkpoint(prefix, dummy, use_reentrant=False)
        else:
            prefix()
        cache = holder[0]
        parts = []
        for s in range(0, len(own), self.per_pass):
            group = own[s:s + self.per_pass]
            if grad:
                tensors = [t for layer in cache.layers for t in (layer.keys, layer.values) if t is not None]
                parts.append(checkpoint(lambda *_, g=group: self._suffix(cache, len(shared), g), *tensors,
                                        use_reentrant=False))
            else:
                parts.append(self._suffix(cache, len(shared), group))
        return torch.cat(parts)

    def scores(self, shared: list, own: list, grad: bool) -> torch.Tensor:
        """The head's scores of the candidates of one question; with grad False, the model is read without
        gradients."""
        device = self.device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if grad:
                leaves = self.leaves(shared, own, grad=True)
            else:
                with torch.no_grad():
                    leaves = self.leaves(shared, own, grad=False)
        return scores(self.head, leaves)

    def question_loss(self, shared: list, own: list, target: list, grad: bool) -> torch.Tensor:
        """-(target * log_softmax(scores)).sum(); with grad False the language model reads without gradients and
        only the head learns."""
        z = self.scores(shared, own, grad)
        t = torch.tensor(target, device=z.device, dtype=z.dtype)
        return -(t * z.log_softmax(-1)).sum()


def answer(chooser: Chooser, rows: list) -> list:
    """(the top candidate, its probability) of each encoded row, the model in evaluation mode and without gradients;
    its mode is put back after."""
    training = chooser.training
    chooser.eval()
    found = []
    try:
        with torch.no_grad():
            for row in rows:
                p = chooser.scores(row["shared"], row["own"], grad=False).softmax(-1)
                best = int(p.argmax())
                found.append((best, float(p[best])))
    finally:
        chooser.train(training)
    return found


def score(rows: list, found: list) -> dict:
    """{group: {"right": count, "of": rows}}: how many rows' top candidate is a right one, by group ("all" when the
    rows have none)."""
    groups = {}
    for row, (best, _) in zip(rows, found):
        counts = groups.setdefault(row.get("group") or GROUP, {"right": 0, "of": 0})
        counts["right"] += int(bool(row["right"][best]))
        counts["of"] += 1
    return dict(sorted(groups.items()))


# Saving and loading

def save(chooser: Chooser, folder, config: dict, tokenizer=None) -> None:
    """The adapter (PEFT's save_pretrained), the head (HEAD_FILE) and the config (CONFIG_FILE) into `folder`."""
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    parameters = dict(chooser.peft.named_parameters())
    # The adapter by name, whether or not it trains now (it does not during the head-only questions)
    chooser.peft.save_pretrained(str(folder), state_dict={n: parameters[n].detach().cpu()
                                                         for n in chooser.adapter_names})
    save_file({n: p.detach().float().contiguous().cpu() for n, p in chooser.head.named_parameters()},
              str(folder / HEAD_FILE))
    (folder / CONFIG_FILE).write_text(json.dumps(config, indent=1))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(folder))


def load(chooser: Chooser, folder) -> None:
    """The adapter and the head saved in `folder` (see save), copied into `chooser`; refused when they are not its."""
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    folder = Path(folder)
    parameters = dict(chooser.peft.named_parameters())
    expected = set(get_peft_model_state_dict(chooser.peft, state_dict={n: parameters[n]
                                                                      for n in chooser.adapter_names}))
    adapter = load_file(str(folder / "adapter_model.safetensors"))
    if set(adapter) != expected:
        raise ValueError(f"{folder}: the saved adapter is not this model's: missing "
                         f"{sorted(expected - set(adapter))[:5]}, unknown {sorted(set(adapter) - expected)[:5]}")
    result = set_peft_model_state_dict(chooser.peft, adapter)
    if result.unexpected_keys:
        raise ValueError(f"{folder}: the saved adapter has weights the model has not: {result.unexpected_keys[:5]}")
    head = load_file(str(folder / HEAD_FILE))
    own = dict(chooser.head.named_parameters())
    if set(head) != set(own):
        raise ValueError(f"{folder}: the saved head is not this head: missing {sorted(set(own) - set(head))[:5]}, "
                         f"unknown {sorted(set(head) - set(own))[:5]}")
    with torch.no_grad():
        for name, value in head.items():
            if value.shape != own[name].shape:
                raise ValueError(f"{folder}: the head's {name} is {tuple(value.shape)}, and this head's "
                                 f"{tuple(own[name].shape)}")
            own[name].copy_(value.to(own[name].device, own[name].dtype))


def read_start(path) -> dict:
    """A start_from file: {"examples": N, "trainable": {name: tensor}, "optimizer": {name: {"step", "exp_avg",
    "exp_avg_sq"}}, "milestones": [...] optional}, checked for its form."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"start_from: no file {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"start_from: {path} holds {type(state).__name__}, not a mapping")
    missing = [k for k in START_KEYS if k not in state]
    unknown = sorted(set(state) - set(START_KEYS) - {"milestones"})
    if missing or unknown:
        raise ValueError(f"start_from: {path} must hold {', '.join(START_KEYS)} and optionally milestones; "
                         f"missing {missing}, unknown {unknown}")
    examples = state["examples"]
    if isinstance(examples, torch.Tensor) and examples.numel() == 1 and not examples.is_floating_point():
        examples = int(examples)
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < 0:
        raise ValueError(f"start_from: examples must be a whole number of questions, not {examples!r}")
    trainable = state["trainable"]
    if not isinstance(trainable, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                  for k, v in trainable.items()):
        raise ValueError("start_from: trainable must map parameter names to tensors")
    optimizer = state["optimizer"]
    if not isinstance(optimizer, dict):
        raise ValueError("start_from: optimizer must map parameter names to AdamW's state")
    for name, entry in optimizer.items():
        if not isinstance(entry, dict) or sorted(entry) != sorted(ADAM_KEYS) or not all(
                isinstance(v, torch.Tensor) for v in entry.values()):
            raise ValueError(f"start_from: the optimizer state of {name!r} must be the tensors {', '.join(ADAM_KEYS)}")
    milestones = state.get("milestones")
    if milestones is not None and (not isinstance(milestones, list) or not all(isinstance(m, dict)
                                                                               for m in milestones)):
        raise ValueError("start_from: milestones must be a list of records")
    return {"examples": examples, "trainable": trainable, "optimizer": optimizer, "milestones": milestones or []}


def _names(names) -> str:
    names = sorted(names)
    return ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")


def check_start(chooser: Chooser, start: dict) -> None:
    """Refuse a start_from state that does not name exactly the chooser's trainable parameters, in their shapes."""
    trainable = chooser.trainable()
    missing, unknown = set(trainable) - set(start["trainable"]), set(start["trainable"]) - set(trainable)
    if missing or unknown:
        raise ValueError("start_from: the saved parameters are not this model's trainable ones: "
                         + "; ".join(part for part in (f"not in the file: {_names(missing)}" if missing else "",
                                                       f"not trainable here: {_names(unknown)}" if unknown else "")
                                     if part))
    for name, value in start["trainable"].items():
        if value.shape != trainable[name].shape:
            raise ValueError(f"start_from: {name} is {tuple(value.shape)} in the file, and "
                             f"{tuple(trainable[name].shape)} here")
    unknown = set(start["optimizer"]) - set(trainable)
    if unknown:
        raise ValueError(f"start_from: optimizer state of parameters that are not trainable here: {_names(unknown)}")
    for name, entry in start["optimizer"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            if entry[key].shape != trainable[name].shape:
                raise ValueError(f"start_from: the optimizer's {key} of {name} is {tuple(entry[key].shape)}, and the "
                                 f"parameter {tuple(trainable[name].shape)}")


def apply_start(chooser: Chooser, start: dict, optimizer=None) -> None:
    """The start_from tensors copied into the chooser's parameters; with `optimizer` (AdamW over them), its state
    set by name."""
    trainable = chooser.trainable()
    with torch.no_grad():
        for name, value in start["trainable"].items():
            trainable[name].copy_(value.to(trainable[name].device, trainable[name].dtype))
    if optimizer is not None:
        for name, entry in start["optimizer"].items():
            p = trainable[name]
            optimizer.state[p] = {"step": entry["step"].to(torch.float32).cpu(),
                                  "exp_avg": entry["exp_avg"].to(p.device, p.dtype),
                                  "exp_avg_sq": entry["exp_avg_sq"].to(p.device, p.dtype)}


# Preprocessing: the rows encoded once, into a folder keyed by what made them (see meepsquared._preprocess)

def _identity(data) -> object:
    from meepsquared import _preprocess

    if isinstance(data, (list, tuple)):
        return [_identity(d) for d in data]
    return _preprocess.data_identity(data, None, None)


def inputs(job, tokenizer) -> dict:
    """Every input a choice preprocessing depends on (the data files' bytes, the tokenizer as loaded, the versions)."""
    from meepsquared import _preprocess

    for where, data in (("data", job.data), ("eval_data", job.eval_data)):
        for item in (data if isinstance(data, (list, tuple)) else [data] if data is not None else []):
            if not isinstance(item, (str, os.PathLike)) or not Path(item).is_file():
                raise FileNotFoundError(f"{where}: no file {item}")
    return {"key_version": _preprocess.KEY_VERSION, "task": "choice", "format": FORMAT, "data": _identity(job.data),
            "eval_data": _identity(job.eval_data) if job.eval_data is not None else None,
            "tokenizer": _preprocess.tokenizer_identity(tokenizer), "versions": _preprocess.versions()}


def _facts(rows: list) -> dict:
    counts = [len(r["own"]) for r in rows]
    sizes = [tokens_read(r) for r in rows]
    return {"candidates": {"least": min(counts), "median": statistics.median(counts), "most": max(counts)},
            "tokens_read": {"least": min(sizes), "median": statistics.median(sizes), "most": max(sizes)}}


def _write(rows: list, path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess(job, cache_dir, say=print):
    """The job's rows checked and encoded into `<cache_dir>/<key>`, found there when made before from the same
    inputs. Returns meepsquared._preprocess's Outcome."""
    import shutil

    from meepsquared import _preprocess
    from meepsquared.models import load_tokenizer

    if job.task != "choice":
        raise ValueError(f"this preprocessing is for task 'choice', and the job's task is {job.task!r}")
    tokenizer = job.tokenizer if job.tokenizer is not None else load_tokenizer(job.model)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_inputs = inputs(job, tokenizer)
    key = _preprocess.key_of(key_inputs)
    target = cache_dir / key
    if _preprocess.found(target, key_inputs):
        report = _preprocess.report_of(target)
        say(f"preprocessed data found in {target}: {report['decision']['text']}")
        return _preprocess.Outcome(str(target), key, True, report)
    rows, source = read_encoded(job.data, tokenizer, "data", target=True, right=False)
    eval_rows = eval_source = None
    if job.eval_data is not None:
        eval_rows, eval_source = read_encoded(job.eval_data, tokenizer, "eval_data", target=True, right=False)
    facts = _facts(rows)
    text = (f"{len(rows):,} questions of {facts['candidates']['least']} to {facts['candidates']['most']} candidates, "
            f"{facts['tokens_read']['least']:,} to {facts['tokens_read']['most']:,} tokens read per question")
    value = {"rows": len(rows), "eval_rows": len(eval_rows) if eval_rows else None, **facts}
    reason = ("every row has a prompt, 2 or more candidates and a target that sums to 1; the rows are kept in the "
              "file's order, which is the order they train in")
    report = {"decision": {"name": "data_checks", "label": "data checks", "value": value, "text": text,
                           "reason": reason},
              "kind": "choice questions", "rows": len(rows), "eval_rows": value["eval_rows"],
              "source": source, "eval_source": eval_source, **facts}
    building = cache_dir / f".building-{os.getpid()}"
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir()
    try:
        _write(rows, building / TRAIN_FILE)
        if eval_rows is not None:
            _write(eval_rows, building / EVAL_FILE)
        (building / _preprocess.REPORT_FILE).write_text(json.dumps(report, indent=2))
        (building / _preprocess.KEY_FILE).write_text(json.dumps(key_inputs, indent=2, sort_keys=True))
        try:
            building.rename(target)
        except OSError:
            if not _preprocess.found(target, key_inputs):
                raise
            return _preprocess.Outcome(str(target), key, True, report)
    finally:
        shutil.rmtree(building, ignore_errors=True)
    say(f"preprocessed data built in {target}: {text}")
    return _preprocess.Outcome(str(target), key, False, report)


def open_folder(folder, key_inputs: dict) -> tuple:
    """(rows, eval rows or None, report) of a choice preprocessing folder whose key is `key_inputs`; refused
    otherwise."""
    from meepsquared import _preprocess

    folder = Path(folder)
    key_file = folder / _preprocess.KEY_FILE
    if not key_file.is_file() or not (folder / TRAIN_FILE).is_file():
        raise ValueError(f"preprocessed: {folder} is not a choice preprocessing folder (no {_preprocess.KEY_FILE} and "
                         f"{TRAIN_FILE})")
    kept = json.loads(key_file.read_text())
    if kept != key_inputs:
        raise ValueError(f"preprocessed: {folder} was made from other inputs than this job's: "
                         f"{', '.join(_preprocess.differences(kept, key_inputs))} differ; preprocess the job again")
    report = _preprocess.report_of(folder)

    def read(name: str) -> list:
        return [json.loads(line) for line in (folder / name).read_text().splitlines() if line]

    return read(TRAIN_FILE), read(EVAL_FILE) if (folder / EVAL_FILE).is_file() else None, report
