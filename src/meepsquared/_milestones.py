# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Milestones: at a share of the planned steps, the adapter kept apart and the model scored on what it writes.

- steps(planned, share): the milestones' steps, every `share` of the planned
  steps and the last one.
- A milestone is a folder, <output>/milestones/step-<N>: the adapter in PEFT's
  format (adapter_config.json, adapter_model.safetensors) and the tokenizer,
  which Transformers' checkpoints do not keep apart and a checkpoint rotation
  never removes; with an evaluation, the generations (generations.jsonl) and
  the scores; milestone.json last, so a folder without it is unfinished and is
  made again. It is written as step-<N>.partial and renamed when whole.
- The evaluation: rows of {"messages": [...], "reference": "...", "group":
  optional, "source": optional}. Each prompt is the chat template's rendering
  of the messages followed by an answer holding the reference, cut where the
  reference begins: what the training reads before an answer, a thinking block
  the template writes included. The model writes greedily, in left-padded
  batches of rows of similar length; each generation ends at the end token, at
  its length cap (NEW_RATIO x the tokens of `source`, or of the last user
  message, + NEW_MARGIN) and at its first loop (one copy kept).
- The scorer, scorer(hypotheses, references) -> {metric: number}, scores each
  group apart ("all" when the rows have no group). The trainer logs the scores
  as milestone/<group>/<metric>, so with report_to="mlflow" MLflow records them,
  and the status file lists each milestone.
- time_generation: the seconds of the evaluation's generation, from one timed
  batch, for the plan.

A milestone runs at the end of its optimizer step, before a stop asked for at
that step takes effect: a stop never cuts one in half. It runs on one process.
"""

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from transformers import TrainerCallback

from meepsquared import _save

# The length cap of a generation: NEW_RATIO x the source's tokens + NEW_MARGIN. On FLORES+ the largest ratio of a
# reference's tokens to its source's is 3.4 (English to Aranese), and 2 x n + 32 cuts no reference.
NEW_RATIO = 2
NEW_MARGIN = 32
# A loop: the generation ends with LOOP_MIN_REPEATS or more copies in a row of one block of 1 to LOOP_MAX_PERIOD
# tokens, covering at least LOOP_MIN_SPAN tokens
LOOP_MAX_PERIOD = 32
LOOP_MIN_SPAN = 16
LOOP_MIN_REPEATS = 4
BATCH = 16
FOLDER = "milestones"
GROUP = "all"


def steps(planned: int, share: float) -> list:
    """The steps of the milestones: every `share` of the planned steps, rounded up, and the last step."""
    count = max(1, round(1 / share))
    return sorted({min(planned, math.ceil(planned * k / count)) for k in range(1, count + 1)})


def folder(output, step: int) -> Path:
    return Path(output) / FOLDER / f"step-{step}"


def done(output, step: int) -> Optional[dict]:
    """The record of a finished milestone, or None."""
    path = folder(output, step) / "milestone.json"
    return json.loads(path.read_text()) if path.is_file() else None


def rows(data) -> list:
    """The evaluation's rows from a .jsonl file, a list of mappings or a datasets.Dataset, checked."""
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        if path.suffix.lower() != ".jsonl" or not path.is_file():
            raise ValueError(f"milestone_eval: {path} is not a .jsonl file")
        found = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif isinstance(data, list):
        found = data
    else:
        from datasets import Dataset

        if not isinstance(data, Dataset):
            raise TypeError(f"milestone_eval must be a .jsonl file, a list of rows or a datasets.Dataset, not "
                            f"{type(data).__name__}")
        found = data.to_list()
    if not found:
        raise ValueError("milestone_eval has no rows")
    for i, row in enumerate(found):
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or not messages or not all(
                isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                for m in messages):
            raise ValueError(f"milestone_eval row {i}: 'messages' must be a list of turns with a role and a content")
        if messages[-1]["role"] == "assistant":
            raise ValueError(f"milestone_eval row {i}: the messages end with an answer; the model writes it")
        if not isinstance(row.get("reference"), str) or not row["reference"]:
            raise ValueError(f"milestone_eval row {i}: 'reference' must be the expected answer, a text")
        for name in ("group", "source"):
            if row.get(name) is not None and not isinstance(row[name], str):
                raise ValueError(f"milestone_eval row {i}: {name!r} must be a text")
    return found


def prompt(tokenizer, row: dict) -> str:
    """The text the training reads before an answer: the conversation rendered with the reference as the answer,
    cut where the reference begins."""
    rendered = tokenizer.apply_chat_template(row["messages"] + [{"role": "assistant", "content": row["reference"]}],
                                             tokenize=False)
    at = rendered.rfind(row["reference"])
    if at < 0:
        raise ValueError(f"the chat template does not write the reference as it is ({row['reference'][:80]!r}); "
                         "the prompt before an answer cannot be found")
    return rendered[:at]


def budget(tokenizer, row: dict) -> int:
    """The length cap of the row's generation, in new tokens."""
    source = row.get("source")
    if source is None:
        source = next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user")
    return NEW_RATIO * len(tokenizer(source, add_special_tokens=False)["input_ids"]) + NEW_MARGIN


def loop_end(ids: list) -> Optional[tuple]:
    """(period, repeats) when `ids` ends with a loop as LOOP_* define it, else None."""
    for period in range(1, LOOP_MAX_PERIOD + 1):
        repeats = max(LOOP_MIN_REPEATS, -(-LOOP_MIN_SPAN // period))
        if len(ids) < repeats * period:
            break
        block = ids[len(ids) - period:]
        if all(ids[len(ids) - (k + 1) * period:len(ids) - k * period] == block for k in range(1, repeats)):
            return period, repeats
    return None


def cut(ids: list, cap: int, end_ids: set) -> list:
    """A generation up to its first end token, its length cap and its first loop, one copy of the loop kept."""
    ids = list(ids)
    for i, token in enumerate(ids):
        if token in end_ids:
            ids = ids[:i]
            break
    ids = ids[:cap]
    for n in range(1, len(ids) + 1):
        loop = loop_end(ids[:n])
        if loop:
            period, repeats = loop
            return ids[:n - (repeats - 1) * period]
    return ids


def end_ids(model, tokenizer) -> set:
    """The tokens that end an answer: the generation config's end tokens and the tokenizer's."""
    found = set()
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for value in (configured, tokenizer.eos_token_id):
        if isinstance(value, int):
            found.add(value)
        elif isinstance(value, (list, tuple)):
            found.update(value)
    if not found:
        raise ValueError("the model and its tokenizer name no end token; a generation could not end")
    return found


def _batches(tokenizer, found: list, size: int) -> list:
    """The rows' indices in batches of rows of similar prompt length, each with their prompts and caps."""
    prompts = [prompt(tokenizer, row) for row in found]
    caps = [budget(tokenizer, row) for row in found]
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(found)), key=lambda i: (lengths[i], caps[i]))
    return [[(i, prompts[i], caps[i]) for i in order[start:start + size]] for start in range(0, len(order), size)]


def _generate_batch(model, tokenizer, batch: list, ends: set) -> list:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Done(StoppingCriteria):
        """Each row is done at its own cap, at an end token, or at a loop."""

        def __init__(self, start, caps):
            self.start, self.caps = start, caps

        def __call__(self, input_ids, scores, **kwargs):
            finished = []
            for row, cap in zip(input_ids.tolist(), self.caps):
                out = row[self.start:]
                finished.append(len(out) >= cap or any(t in ends for t in out) or loop_end(out) is not None)
            return torch.tensor(finished, dtype=torch.bool, device=input_ids.device)

    caps = [cap for _, _, cap in batch]
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer([p for _, p, _ in batch], add_special_tokens=False, return_tensors="pt",
                            padding=True).to(next(model.parameters()).device)
    finally:
        tokenizer.padding_side = side
    start = encoded["input_ids"].shape[1]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(ends)
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max(caps), do_sample=False, use_cache=True,
                             eos_token_id=sorted(ends), pad_token_id=pad,
                             stopping_criteria=StoppingCriteriaList([Done(start, caps)]))
    return [cut(row, cap, ends) for row, cap in zip(out[:, start:].tolist(), caps)]


class _Inference:
    """The model in evaluation mode with its cache on; put back as it was on exit."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.training = self.model.training
        config = getattr(self.model, "config", None)
        self.cache = getattr(config, "use_cache", None)
        self.model.eval()
        if config is not None:
            config.use_cache = True
        return self.model

    def __exit__(self, *exc):
        config = getattr(self.model, "config", None)
        if config is not None and self.cache is not None:
            config.use_cache = self.cache
        self.model.train(self.training)
        return False


def generate(model, tokenizer, found: list, size: int = BATCH) -> list:
    """The model's greedy answer to each row, in the rows' order, cut as `cut` cuts it."""
    ends = end_ids(model, tokenizer)
    answers = [None] * len(found)
    with _Inference(model):
        for batch in _batches(tokenizer, found, size):
            for (i, _, _), ids in zip(batch, _generate_batch(model, tokenizer, batch, ends)):
                answers[i] = " ".join(tokenizer.decode(ids, skip_special_tokens=True).split())
    return answers


def time_generation(model, tokenizer, found: list, size: int = BATCH) -> tuple:
    """(seconds of the whole evaluation's generation, how it was priced), from its longest batch, timed.

    Each batch's decoding runs as many steps as its longest answer; a
    reference's tokens, within its cap, stand for that answer. The batch with
    the most such steps is generated whole, and each batch is priced from it by
    its steps."""
    ends = end_ids(model, tokenizer)
    batches = _batches(tokenizer, found, size)

    def decode_steps(batch):
        return max(min(len(tokenizer(found[i]["reference"], add_special_tokens=False)["input_ids"]) + 1, cap)
                   for i, _, cap in batch)

    longest = max(batches, key=decode_steps)
    with _Inference(model):
        _generate_batch(model, tokenizer, longest[:1], ends)  # warm-up
        began = time.perf_counter()
        _generate_batch(model, tokenizer, longest, ends)
        seconds = time.perf_counter() - began
    per_step = seconds / max(1, max(cap for _, _, cap in longest))
    total = sum(decode_steps(b) for b in batches) * per_step
    why = (f"{len(found)} rows in {len(batches)} batches of up to {size}; the batch of the most decoding steps "
           f"generated to its cap, {per_step * 1000:.0f} ms a step; each batch priced by its longest reference")
    return total, why


def score(scorer: Callable, found: list, answers: list) -> dict:
    """{group: {metric: number}}: the scorer's figures for each group of rows."""
    groups = {}
    for row, answer in zip(found, answers):
        groups.setdefault(row.get("group") or GROUP, ([], []))
        groups[row.get("group") or GROUP][0].append(answer)
        groups[row.get("group") or GROUP][1].append(row["reference"])
    scores = {}
    for group, (hypotheses, references) in sorted(groups.items()):
        figures = scorer(hypotheses, references)
        if not isinstance(figures, dict) or not figures or not all(
                isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
                for k, v in figures.items()):
            raise ValueError(f"the milestone scorer must answer a mapping of metric names to numbers, not {figures!r}")
        scores[group] = {k: float(v) for k, v in figures.items()}
    return scores


class Milestones(TrainerCallback):
    """At the end of each milestone's step: the adapter saved apart and, with an evaluation, the model's answers
    scored and logged. A milestone finished in an earlier session is not made again."""

    def __init__(self, trainer, output, at: list, found: Optional[list], scorer: Optional[Callable],
                 size: int = BATCH, status=None, report=None):
        self.trainer, self.output, self.at = trainer, Path(output), set(at)
        self.found, self.scorer, self.size = found, scorer, size
        self.status, self.report = status, report
        self.seconds = None

    def records(self) -> list:
        return [record for record in (done(self.output, step) for step in sorted(self.at)) if record is not None]

    def _report(self, state) -> None:
        if self.status is None:
            return
        upcoming = [s for s in sorted(self.at) if s > state.global_step]
        self.status.update(milestones=self.records(), milestone_seconds=self.seconds,
                           next_milestone=upcoming[0] if upcoming else None)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._report(state)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step not in self.at or not state.is_world_process_zero or done(self.output, step) is not None:
            return
        began = time.perf_counter()
        model = model if model is not None else self.trainer.model
        tokenizer = self.trainer.processing_class
        final = folder(self.output, step)
        partial = final.with_name(final.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        # meepsquared's adapter save reads the adapter's parameters alone, never the frozen expert stacks
        _save.save(model, partial, tokenizer=tokenizer, save_method="lora")
        record = {"step": step, "planned_steps": state.max_steps, "folder": str(final)}
        if self.found is not None:
            answers = generate(model, tokenizer, self.found, self.size)
            with (partial / "generations.jsonl").open("w") as f:
                for row, answer in zip(self.found, answers):
                    f.write(json.dumps({"group": row.get("group") or GROUP, "reference": row["reference"],
                                        "answer": answer}, ensure_ascii=False) + "\n")
            record["scores"] = score(self.scorer, self.found, answers)
            record["rows"] = len(self.found)
        record["seconds"] = round(time.perf_counter() - began, 1)
        (partial / "milestone.json").write_text(json.dumps(record, indent=2))
        shutil.rmtree(final, ignore_errors=True)
        partial.rename(final)
        self.seconds = record["seconds"]
        if self.report is not None:
            # The milestone runs after the step's end; its time is not the steps'
            self.report.evaluating += time.perf_counter() - began
        if record.get("scores"):
            self.trainer.log({f"milestone/{group}/{metric}": value
                              for group, figures in record["scores"].items() for metric, value in figures.items()})
        self._report(state)
