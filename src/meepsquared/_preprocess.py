# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""SFT's data tokenized, checked and cut once, kept as Arrow files keyed by what made them.

The data of a training job goes through the model's chat template and
tokenizer (TRL's own tokenization, run without a trainer), the labels of the
loss, the checks of meepsquared._data_checks and the cut to max_length once, in
passes over chunks of rows on the cores the process may use. The result is a
folder named by a key: the hash of every input that changes it (the data's
bytes, the tokenizer as loaded, max_length, where the loss counts, the
evaluation share and seed, the versions of the libraries that tokenize). A
plan or a training reads the folder memory-mapped, after checking that its own
inputs make the same key; a folder of another key is never read. The plan and
the training then hold no copy of the data in Python objects, which matters on
a GPU that shares the system's memory, and a dataset of millions of rows is
tokenized once, on a machine with cores to spare.
"""

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

KEY_VERSION = 1
KEY_FILE, REPORT_FILE = "key.json", "report.json"
TRAIN_FOLDER, EVAL_FOLDER = "train", "eval"
CACHE_FOLDER = "cache"
# TRL's default seed (TrainingArguments.seed): the trainer's loader and the held-out share draw with it
SEED = 42
# The columns the tokenization reads; the tokens stand for them in the files kept
TEXT_COLUMNS = ("messages", "text", "prompt", "completion", "conversations", "tools", "chat_template_kwargs")
# Per-row facts kept beside the tokens: the row's place in the data as given, its origin kind, and its length
# after the cut
INDEX, KIND, LENGTH = "_index", "_kind", "_length"
FACT_COLUMNS = (INDEX, KIND, LENGTH)
# The passes' own columns, dropped before the files are written
EMPTY, HAS_LOSS, CUT_LOSS, LONGER, FAILURE, NO_ANSWER, HASH = ("_empty", "_has_loss", "_cut_loss", "_longer",
                                                               "_failure", "_no_answer", "_hash")
PASS_COLUMNS = (EMPTY, HAS_LOSS, CUT_LOSS, LONGER, FAILURE, NO_ANSWER, HASH)
CHUNK = 1000
HASH_BLOCK = 64 * 2**20


@dataclasses.dataclass
class Outcome:
    """A preprocessing: the folder, its key, whether it was found (made before with the same key) or built now."""

    folder: str
    key: str
    found: bool
    report: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def file_hash(path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as f:
        while block := f.read(HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=32).hexdigest()


def data_identity(data, split: Optional[str], loaded) -> dict:
    """What identifies the data: a file by its bytes, a loaded datasets.Dataset by its fingerprint, a dataset id by
    the fingerprint of its split as loaded."""
    if isinstance(data, (str, os.PathLike)) and Path(data).exists():
        path = Path(data)
        return {"file": str(path), "bytes": path.stat().st_size, "hash": file_hash(path)}
    if isinstance(data, (str, os.PathLike)):
        return {"id": str(data), "split": split, "fingerprint": loaded._fingerprint}
    return {"fingerprint": loaded._fingerprint}


def tokenizer_identity(tokenizer) -> dict:
    """The tokenizer as loaded: its backend (vocabulary, merges, normalization), chat template and special tokens."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        vocabulary = text_hash(backend.to_str())
    else:
        vocabulary = text_hash(json.dumps(sorted(tokenizer.get_vocab().items())))
    template = tokenizer.chat_template
    if template is not None and not isinstance(template, str):
        template = json.dumps(template, sort_keys=True)
    return {"class": type(tokenizer).__name__, "vocabulary": vocabulary,
            "chat_template": text_hash(template) if template is not None else None,
            "special_tokens": {k: str(v) for k, v in sorted(tokenizer.special_tokens_map.items())}}


def versions() -> dict:
    import datasets
    import tokenizers
    import transformers
    import trl

    import meepsquared

    return {"meepsquared": meepsquared.__version__, "trl": trl.__version__, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "datasets": datasets.__version__}


def inputs(data: dict, eval_data: Optional[dict], tokenizer, kind: str, answer_args: dict, markers,
           max_length: int, eval_share: Optional[float]) -> dict:
    """Every input the preprocessing's result depends on, hashed where large."""
    return {"key_version": KEY_VERSION, "data": data, "eval_data": eval_data, "kind": kind,
            "tokenizer": tokenizer_identity(tokenizer), "answers": dict(sorted(answer_args.items())),
            "markers": list(markers) if markers is not None else None, "max_length": max_length,
            "eval_share": eval_share, "seed": SEED, "versions": versions()}


def key_of(inputs: dict) -> str:
    return hashlib.blake2b(json.dumps(inputs, sort_keys=True).encode(), digest_size=16).hexdigest()


def differences(kept: dict, now: dict) -> list:
    """The inputs that differ between a folder's key and the job's, by name."""
    return [name for name in sorted(set(kept) | set(now)) if kept.get(name) != now.get(name)]


def found(folder: Path, inputs: dict) -> bool:
    """Whether the folder holds a preprocessing of exactly these inputs."""
    key_file = folder / KEY_FILE
    if not key_file.exists():
        return False
    with open(key_file) as f:
        return json.load(f) == inputs


def report_of(folder) -> dict:
    """The report of a preprocessing folder; refused for a folder that is not one."""
    folder = Path(folder)
    if not (folder / KEY_FILE).exists() or not (folder / REPORT_FILE).exists():
        raise ValueError(f"preprocessed: {folder} is not a preprocessing folder (no {KEY_FILE} and {REPORT_FILE})")
    with open(folder / REPORT_FILE) as f:
        return json.load(f)


def open_folder(folder, inputs: dict) -> tuple:
    """(train, eval or None, report) of a preprocessing folder whose key is `inputs`, the data memory-mapped;
    refused otherwise."""
    from datasets import load_from_disk

    folder = Path(folder)
    key_file = folder / KEY_FILE
    if not key_file.exists():
        raise ValueError(f"preprocessed: {folder} is not a preprocessing folder (no {KEY_FILE})")
    with open(key_file) as f:
        kept = json.load(f)
    if kept != inputs:
        raise ValueError(f"preprocessed: {folder} was made from other inputs than this job's: "
                         f"{', '.join(differences(kept, inputs))} differ; preprocess the job again")
    with open(folder / REPORT_FILE) as f:
        report = json.load(f)
    train = load_from_disk(str(folder / TRAIN_FOLDER))
    eval_dataset = load_from_disk(str(folder / EVAL_FOLDER)) if (folder / EVAL_FOLDER).exists() else None
    return train, eval_dataset, report


class _Tokenizing:
    """Stands in for TRL's SFTTrainer in its _prepare_dataset, which reads three things of the trainer."""

    def __init__(self, tokenizer, args, sample: dict):
        from trl.chat_template_utils import get_training_chat_template, has_generation_markers

        self._tokenizer = tokenizer
        self.chat_template = (get_training_chat_template(tokenizer)
                              if args.assistant_only_loss and not has_generation_markers(tokenizer.chat_template)
                              else None)
        self.completion_only_loss = (("prompt" in sample and "completion" in sample)
                                     if args.completion_only_loss is None else args.completion_only_loss)


def tokenized(dataset, tokenizer, args, name: str):
    """The dataset tokenized as TRL's SFTTrainer tokenizes it (input_ids and labels; a completion mask when the loss
    is not on the completions alone), without a trainer or a model."""
    import contextlib

    from trl import SFTTrainer
    from trl.trainer import sft_trainer

    stand_in = _Tokenizing(tokenizer, args, dataset[0])
    # TRL runs its tokenization main process first, behind process-group barriers every process must enter; a
    # preprocessing runs in one process (see preprocess), so the barriers are left out
    ordered = sft_trainer.global_then_local_main_first
    sft_trainer.global_then_local_main_first = contextlib.nullcontext
    try:
        return SFTTrainer._prepare_dataset(stand_in, dataset, tokenizer, args, False, None, name)
    finally:
        sft_trainer.global_then_local_main_first = ordered


def column(dataset, name: str):
    """One column of a datasets.Dataset as one pyarrow array, the dataset's selection respected."""
    return dataset.select_columns([name]).with_format("arrow")[name].combine_chunks()


def where(dataset, name: str, value=True) -> list:
    """The positions of the rows whose column `name` equals `value`."""
    import numpy as np
    import pyarrow.compute as pc

    return np.flatnonzero(pc.equal(column(dataset, name), value).to_numpy(zero_copy_only=False)).tolist()


def _cache(folder: Path, name: str) -> str:
    return str(folder / CACHE_FOLDER / f"{name.replace(' ', '_').replace(':', '')}.arrow")


def _map(dataset, function, folder: Path, name: str, cores: int, **kwargs):
    """A batched pass over the dataset on the cores, its result cached in the folder being built."""
    return dataset.map(function, batched=True, batch_size=CHUNK, num_proc=cores if cores > 1 else None,
                       cache_file_name=_cache(folder, name), desc=name, **kwargs)


def named(dataset, positions: list, count: Optional[int] = None) -> dict:
    """{"count", "rows", "ids"}: how many rows, and the first EXAMPLES by their place in the data as given, with
    their _row ids where the data has them."""
    from meepsquared import _data_checks

    shown = positions[:_data_checks.EXAMPLES]
    chosen = dataset.select(shown) if shown else None
    has_ids = _data_checks.ROW in dataset.column_names
    return {"count": len(positions) if count is None else count,
            "rows": [int(i) for i in chosen[INDEX]] if chosen is not None else [],
            "ids": (list(chosen[_data_checks.ROW]) if chosen is not None else []) if has_ids else None}


def merged(first: dict, second: dict) -> dict:
    """Two named sets of rows as one, the first EXAMPLES by their place in the data."""
    from meepsquared import _data_checks

    pairs = sorted(zip(first["rows"] + second["rows"],
                       (first["ids"] or [None] * len(first["rows"])) + (second["ids"] or [None] * len(second["rows"]))))
    shown = pairs[:_data_checks.EXAMPLES]
    has_ids = first["ids"] is not None or second["ids"] is not None
    return {"count": first["count"] + second["count"], "rows": [row for row, _ in shown],
            "ids": [id_ for _, id_ in shown] if has_ids else None}


def rows_text(named_rows: dict) -> str:
    from meepsquared import _data_checks

    return _data_checks.rows_text(named_rows["rows"], named_rows["ids"], named_rows["count"])


class Building:
    """One preprocessing being built: its passes, in order, over the training data and the evaluation data."""

    def __init__(self, folder: Path, tokenizer, answer_args: dict, markers, max_length: int, cores: int):
        self.folder, self.tokenizer = folder, tokenizer
        self.answer_args, self.max_length, self.cores = answer_args, max_length, cores
        # meepsquared's turn markers as token ids, as its collator reads them (see meepsquared.chat_templates)
        plain = getattr(tokenizer, "tokenizer", tokenizer)
        self.markers = (tuple(plain(part, add_special_tokens=False)["input_ids"] for part in markers)
                        if markers is not None else None)

    def kinds(self, dataset, where_: str):
        """The dataset with its _origin read into a _kind column, refused for a row without a kind."""
        from meepsquared import _data_checks

        if _data_checks.ORIGIN not in dataset.column_names:
            return dataset
        return _map(dataset, _data_checks.kinds_of, self.folder, f"{where_}: origin kinds", self.cores,
                    with_indices=True, fn_kwargs={"where": where_}, remove_columns=[_data_checks.ORIGIN])

    def tokenize(self, dataset, where_: str, args) -> tuple:
        """(the dataset tokenized, the rows left out for having no answer token, named)."""
        from meepsquared import _data_checks

        try:
            return tokenized(dataset, self.tokenizer, args, where_), named(dataset, [])
        except Exception as error:  # noqa: BLE001 (TRL tokenizes a set at once: each row is rendered alone)
            rendered = _map(dataset, _data_checks.render_failures, self.folder, f"{where_}: rendering", self.cores,
                            fn_kwargs={"tokenizer": self.tokenizer,
                                       "assistant_masks": bool(self.answer_args.get("assistant_only_loss"))})
            failed = where(rendered, FAILURE, "")
            failed = [i for i in range(len(rendered)) if i not in set(failed)]
            if failed:
                raise ValueError(f"examples that do not render through the chat template or the tokenizer: {where_} "
                                 f"rows {rows_text(named(rendered, failed))} ({rendered[failed[0]][FAILURE]})") from error
            no_answer = where(rendered, NO_ANSWER)
            if not no_answer:
                raise
            # TRL refuses an example with no assistant token: such rows train nothing, and are left out
            dropped = named(dataset, no_answer)
            kept = sorted(set(range(len(dataset))) - set(no_answer))
            return tokenized(dataset.select(kept), self.tokenizer, args, where_), dropped

    def facts(self, dataset, where_: str, dropped: dict) -> tuple:
        """(the dataset without its rows of no counted token, with its _length, the facts of the checks)."""
        from meepsquared import _data_checks

        counted = _map(dataset, _data_checks.loss_facts, self.folder, f"{where_}: loss and length", self.cores,
                       fn_kwargs={"completion_only": bool(self.answer_args.get("completion_only_loss")),
                                  "markers": self.markers, "max_length": self.max_length})
        empty = where(counted, EMPTY)
        if empty:
            raise ValueError(f"examples that do not render through the chat template or the tokenizer: {where_} "
                             f"rows {rows_text(named(counted, empty))} make no tokens")
        none = where(counted, HAS_LOSS, False)
        if len(none) == len(counted):
            raise ValueError(f"{where_}: no example has a token the loss counts")
        facts = {"rows": len(counted) + dropped["count"],
                 "longer": named(counted, where(counted, LONGER)),
                 "answers_cut": named(counted, where(counted, CUT_LOSS)),
                 "no_answer": merged(named(counted, none), dropped)}
        if none:
            counted = counted.select(sorted(set(range(len(counted))) - set(none)))
        return counted.remove_columns([EMPTY, HAS_LOSS, CUT_LOSS, LONGER]), facts

    def hashes(self, dataset, where_: str):
        from meepsquared import _data_checks

        return _map(dataset, _data_checks.text_hashes, self.folder, f"{where_}: text hashes", self.cores,
                    fn_kwargs={"tokenizer": self.tokenizer})

    def cut(self, dataset, where_: str):
        """Every per-token column cut to max_length, as TRL and meepsquared's collator cut them, where a row is longer."""
        import pyarrow.compute as pc

        from meepsquared import _trl

        if pc.max(pc.list_value_length(column(dataset, "input_ids"))).as_py() <= self.max_length:
            return dataset
        return _map(dataset, _trl._cut_batch, self.folder, f"{where_}: cut to max_length", self.cores,
                    fn_kwargs={"max_length": self.max_length})


def preprocess(job, cache_dir, cores: Optional[int] = None, say=print) -> Outcome:
    """The job's SFT data preprocessed into `<cache_dir>/<key>`: found there when made before with the same key,
    else built now. `say` is told what happens. Of the processes of a launch (torchrun), process 0 preprocesses
    while the others wait at a barrier, then find its folder."""
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return _preprocess_here(job, cache_dir, cores, say)
    from accelerate import PartialState

    state = PartialState()
    if state.is_main_process:
        try:
            return _preprocess_here(job, cache_dir, cores, say)
        finally:
            state.wait_for_everyone()
    state.wait_for_everyone()
    return _preprocess_here(job, cache_dir, cores, say)


def _preprocess_here(job, cache_dir, cores: Optional[int], say) -> Outcome:
    from meepsquared import _unified_memory, job as _job
    from meepsquared.models import load_tokenizer

    if job.task != "sft":
        raise ValueError(f"preprocessing is for SFT's data; the job's task is {job.task!r}")
    if cores is None:
        cores = _unified_memory.job_cores()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = job.tokenizer if job.tokenizer is not None else load_tokenizer(job.model)
    building = cache_dir / f"building-{os.getpid()}"
    shutil.rmtree(building, ignore_errors=True)
    (building / CACHE_FOLDER).mkdir(parents=True)
    try:
        dataset, source = job._dataset(cache_dir=str(building / CACHE_FOLDER))
        kind, columns = _job.data_format(dataset.column_names, "sft")
        eval_dataset = eval_source = None
        if job.eval_data is not None:
            eval_dataset, eval_source = job._eval_dataset(cache_dir=str(building / CACHE_FOLDER))
            eval_kind, _ = _job.data_format(eval_dataset.column_names, "sft")
            if eval_kind != kind:
                raise ValueError(f"eval_data holds {eval_kind}, and the training data {kind}; give both alike")
        answer_args = {}
        markers = job._answers(kind, tokenizer, answer_args, [])
        max_length = job.recipe.max_length if job.recipe.max_length is not None else _job.MAX_LENGTH
        key_inputs = inputs(data_identity(job.data, job.split, dataset),
                            data_identity(job.eval_data, job.eval_split, eval_dataset) if eval_dataset is not None
                            else None, tokenizer, kind, answer_args, markers, max_length, job.eval_share)
        key = key_of(key_inputs)
        target = cache_dir / key
        if found(target, key_inputs):
            report = report_of(target)
            say(f"preprocessed data found in {target}: {report['decision']['text']}")
            return Outcome(str(target), key, True, report)
        say(f"preprocessing {len(dataset):,} rows of {source} on {cores} core{'s' if cores != 1 else ''} into {target}")
        build = Building(building, tokenizer, answer_args, markers, max_length, cores)
        report = _build(job, build, dataset, eval_dataset, source, eval_source, kind, columns, answer_args)
        with open(building / REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        shutil.rmtree(building / CACHE_FOLDER)
        with open(building / KEY_FILE, "w") as f:
            json.dump(key_inputs, f, indent=2, sort_keys=True)
        try:
            building.rename(target)
        except OSError:
            if not found(target, key_inputs):
                raise
            # Built meanwhile by another process of a launch, from the same inputs
            return Outcome(str(target), key, True, report)
        say(f"preprocessed data built in {target}: {report['decision']['text']}")
        return Outcome(str(target), key, False, report)
    finally:
        shutil.rmtree(building, ignore_errors=True)


def _build(job, build: Building, dataset, eval_dataset, source: str, eval_source: Optional[str], kind: str,
           columns: list, answer_args: dict) -> dict:
    from trl import SFTConfig

    from meepsquared import _batches, _data_checks

    checks = _data_checks
    args = SFTConfig(output_dir=str(build.folder / CACHE_FOLDER), max_length=None, use_cpu=True, report_to=[],
                     dataset_num_proc=build.cores if build.cores > 1 else None, **answer_args)

    def prepared(data, where_: str):
        import numpy as np

        data = data.add_column(INDEX, np.arange(len(data)))
        data = build.kinds(data, where_)
        data = data.select_columns([c for c in data.column_names if c in columns or c in (INDEX, KIND, checks.ROW)])
        data, dropped = build.tokenize(data, where_, args)
        return build.facts(data, where_, dropped)

    train, facts = prepared(dataset, "data")
    eval_prepared = eval_facts = None
    if eval_dataset is not None:
        eval_prepared, eval_facts = prepared(eval_dataset, "eval_data")
    cut_share = facts["answers_cut"]["count"] / facts["rows"]
    if job.recipe.max_cut_share is not None and cut_share > job.recipe.max_cut_share:
        raise ValueError(f"max_length {build.max_length} cuts answer tokens of {facts['answers_cut']['count']} of "
                         f"{facts['rows']} examples ({cut_share:.1%}), more than max_cut_share "
                         f"{job.recipe.max_cut_share:g}: rows {rows_text(facts['answers_cut'])}")

    # The evaluation share, held out of the rows that train
    held = 0
    has_kinds = KIND in train.column_names
    if job.eval_share is not None:
        # Held out of the real rows where the data marks each row's origin: an evaluation is on real data
        if has_kinds:
            candidates = where(train, KIND, "real")
            if max(1, round(job.eval_share * len(candidates))) >= len(candidates):
                raise ValueError(f"eval_share={job.eval_share:g} holds out real rows only (the data has _origin), "
                                 f"and the data has {len(candidates)} real rows: too few to hold out any and "
                                 "keep one to train")
        else:
            candidates = list(range(len(train)))
        chosen = set(candidates[i] for i in _batches.held_out(len(candidates), job.eval_share, SEED)[1])
        held_rows = sorted(chosen)
        eval_prepared, train = train.select(held_rows), train.select([i for i in range(len(train)) if i not in chosen])
        held = len(held_rows)
        eval_source = f"{source}, held out with seed {SEED}"
    if job.recipe.source_weights is not None and not has_kinds:
        raise ValueError("recipe.source_weights weights the loss by each row's origin, and the data has no "
                         "_origin column")
    if eval_prepared is not None:
        overlap = _overlap(build, train, eval_prepared)
        if overlap:
            named_overlap = ", ".join(f"evaluation row {e} is training row {t} ({how})"
                                      for e, t, how in overlap[:checks.EXAMPLES])
            raise ValueError(f"{len(overlap)} evaluation examples are training examples: {named_overlap}")
        if KIND in eval_prepared.column_names:
            synthetic = where(eval_prepared, KIND, "synthetic")
            if synthetic:
                raise ValueError(f"{len(synthetic)} evaluation examples are synthetic (_origin kind 'synthetic'): "
                                 f"rows {rows_text(named(eval_prepared, synthetic))}; evaluate on real data")
    synthetic_share = len(where(train, KIND, "synthetic")) / len(train) if has_kinds else None

    value = {**{name: facts[name] if name == "rows" else {"count": facts[name]["count"], "rows": facts[name]["rows"]}
                for name in ("rows", "longer", "answers_cut", "no_answer")},
             "eval_no_answer": ({"count": eval_facts["no_answer"]["count"], "rows": eval_facts["no_answer"]["rows"]}
                                if eval_facts else None),
             "overlap": 0 if eval_prepared is not None else None, "synthetic_share": synthetic_share}
    text = (f"{facts['rows']:,} rows render; {facts['longer']['count']:,} longer than {build.max_length} tokens, "
            f"{facts['answers_cut']['count']:,} of them with answer tokens cut; {facts['no_answer']['count']:,} "
            "without answer tokens, left out")
    if eval_facts is not None and eval_facts["no_answer"]["count"]:
        text += f" (and {eval_facts['no_answer']['count']:,} of the evaluation data)"
    if eval_prepared is not None:
        text += "; no evaluation example is a training example"
    if synthetic_share is not None:
        text += f"; {synthetic_share:.0%} of the training rows synthetic"
    named_parts = [f"{label} {rows_text(facts[name])}"
                   for name, label in (("answers_cut", "answers cut: rows"), ("no_answer", "no answer: rows"))
                   if facts[name]["count"]]
    reason = ("the cut keeps each example's start" + (f"; max_cut_share {job.recipe.max_cut_share:g} refuses more"
                                                       if job.recipe.max_cut_share is not None else
                                                       "; Recipe(max_cut_share=...) refuses more than a share")
              + "; a row of no answer token trains nothing" + "".join(f"; {part}" for part in named_parts))
    train = build.cut(train, "data")
    _save(train, build.folder / TRAIN_FOLDER, build.cores)
    if eval_prepared is not None:
        _save(build.cut(eval_prepared, "eval_data"), build.folder / EVAL_FOLDER, build.cores)
    return {"decision": {"name": "data_checks", "label": "data checks", "value": value, "text": text,
                         "reason": reason},
            "kind": kind, "columns": columns, "rows": facts["rows"], "trained_rows": len(train),
            "eval_rows": len(eval_prepared) if eval_prepared is not None else None, "held": held,
            "has_kinds": has_kinds, "seed": SEED, "source": source, "eval_source": eval_source}


def _save(dataset, folder: Path, cores: int) -> None:
    """The tokens and the facts kept beside them; the text columns, the passes' own and the _row ids (metadata read
    for the checks' messages) are not written."""
    from meepsquared import _data_checks

    kept = [c for c in dataset.column_names
            if c not in TEXT_COLUMNS and c not in PASS_COLUMNS and c != _data_checks.ROW]
    dataset.select_columns(kept).save_to_disk(str(folder), num_proc=min(cores, max(1, len(dataset) // CHUNK)) or None)


def _overlap(build: Building, train, eval_prepared) -> list:
    """(evaluation row, training row, how) of each evaluation example that is a training example: by the rendered
    text, normalized and hashed, and by the _row id when both sets have one."""
    import pyarrow.compute as pc

    from meepsquared import _data_checks

    def first_match(values, value_set) -> list:
        return pc.index_in(values, value_set=value_set).fill_null(-1).to_numpy().tolist()

    by_text = first_match(column(build.hashes(eval_prepared, "eval_data"), HASH),
                          column(build.hashes(train, "data"), HASH))
    by_row = None
    if _data_checks.ROW in train.column_names and _data_checks.ROW in eval_prepared.column_names:
        by_row = first_match(column(eval_prepared, _data_checks.ROW), column(train, _data_checks.ROW))
        eval_rows = column(eval_prepared, _data_checks.ROW).to_pylist()
    eval_index, train_index = column(eval_prepared, INDEX).to_pylist(), column(train, INDEX).to_pylist()
    found_rows = []
    for e in range(len(eval_index)):
        if by_text[e] >= 0:
            found_rows.append((eval_index[e], train_index[by_text[e]], "the same text"))
        elif by_row is not None and by_row[e] >= 0:
            found_rows.append((eval_index[e], train_index[by_row[e]], f"the same _row {eval_rows[e]!r}"))
    return found_rows
