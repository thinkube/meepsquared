# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""TrainingJob(task="tagging"): token tagging and dependency parsing on an encoder (see meepsquared._tagging).

The plan reads the sentences, chooses the tasks, loads the encoder with its
adapter (bottleneck adapters or LoRA) and the heads, and decides the batch
as for SFT: a token budget per update, the sentences sorted by length and
grouped so that a padded batch pads little (meepsquared._batches, by_length);
the largest batch the training runs is measured, the accumulation doubled
until a share of the budget fits, the layers to recompute chosen
(meepsquared._checkpointing) and the steps timed, which also checks their real
peak. One evaluation pass on the evaluation data is timed too.

Training runs Transformers' Trainer. At the end of every epoch the evaluation
data is predicted and scored (each task's accuracy; UAS and LAS with the
parser), and the job keeps the best epoch's weights, the last's, or the
average of the last epochs' (Recipe.keep). The scores are logged with the
training's metrics, and a resumed training continues the record.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

from transformers import Trainer, TrainerCallback, TrainingArguments

from meepsquared import _tagging

# A tagger's defaults, each stated in the plan with its source
TOKENS_PER_UPDATE = 4096
TAGGING_LEARNING_RATE = 5e-4
TAGGING_WARMUP_SHARE = 0.06
TAGGING_WEIGHT_DECAY = 0.01
TAGGING_OPTIMIZER = "adamw_torch"
BOTTLENECK = 256
HEAD_DROPOUT = 0.2
LORA_TARGETS = ["query", "key", "value", "dense"]
KEEPS = ("best", "last", "averaged")
TEXPLICIT = "the setting texplicit trained its XLM-RoBERTa taggers and parsers with; not yet compared with others"
# The files a training writes beside its checkpoints
SCORES, BEST, AVERAGE_SUM = "tagging-scores.jsonl", "best", "average-sum.safetensors"


def rows_of(data, words: str, where: str) -> tuple:
    """(rows, source): the sentences of a CoNLL-U file, a .jsonl, .json or .parquet file with a 'tokens' column, a
    list of such files, or a datasets.Dataset."""
    from meepsquared.job import DATA_SUFFIXES

    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError(f"{where}: an empty list of files")
        rows, sources = [], []
        for item in data:
            if not isinstance(item, (str, os.PathLike)):
                raise ValueError(f"{where}: a list holds files, not {type(item).__name__}")
            part, source = rows_of(item, words, where)
            rows += part
            sources.append(source)
        return rows, " + ".join(sources)
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        if not path.is_file():
            raise FileNotFoundError(f"{where}: no file {path}")
        if path.suffix == ".conllu":
            return _tagging.read_conllu(path, words), f"{path.name} ({words} words)"
        if path.suffix in DATA_SUFFIXES:
            from datasets import load_dataset

            return load_dataset(DATA_SUFFIXES[path.suffix], data_files=str(path), split="train").to_list(), path.name
        raise ValueError(f"{where}: {path} is not a .conllu, .jsonl, .json or .parquet file")
    import datasets

    if isinstance(data, datasets.Dataset):
        return data.to_list(), "the dataset given"
    raise ValueError(f"{where}: a CoNLL-U file, a .jsonl, .json or .parquet file, a list of them, or a "
                     f"datasets.Dataset, not {type(data).__name__}")


class TaggingArguments(TrainingArguments):
    """Transformers' TrainingArguments with the two fields meepsquared._batches reads of a batch: its batches are
    padded, to no multiple."""

    padding_free = False
    pad_to_multiple_of = None


class TaggingTrainer(Trainer):
    """Transformers' Trainer for a tagger: a checkpoint holds the adapter and the heads only (meepsquared._tagging.save),
    and a resumed training loads them back."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.args.should_save:
            _tagging.save(self.model, output_dir, self._meepsquared_tagger)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        load_into(model if model is not None else self.model, Path(resume_from_checkpoint) / _tagging.WEIGHTS)


def load_into(tagger, path) -> None:
    """The adapter and heads saved in the file `path` (meepsquared._tagging.save), copied into `tagger`."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    parameters = {name: p for name, p in tagger.named_parameters() if p.requires_grad}
    if set(state) != set(parameters):
        raise ValueError(f"{path}: the saved weights are not this tagger's")
    import torch

    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


class EpochScores(TrainerCallback):
    """At the end of every epoch, the evaluation data predicted and scored; the kept weights put in place at the
    end (Recipe.keep).

    The record of each epoch is appended to SCORES in the output folder and
    logged with the trainer's metrics as eval_<score>. With keep "best", the
    best epoch's weights are saved in <output>/best; with "averaged", the sum
    of the last `average_last` epochs' weights in AVERAGE_SUM. A resumed
    training reads them back.
    """

    def __init__(self, trainer, examples: list, rows: list, pad_id: int, budget: int, decode: str, keep: str,
                 average_last: int, epochs: int, output: str, config: dict):
        self.trainer, self.examples, self.rows = trainer, examples, rows
        self.pad_id, self.budget, self.decode = pad_id, budget, decode
        self.keep, self.average_last, self.epochs = keep, average_last, epochs
        self.output, self.config = Path(output), config
        self.records = []
        if (self.output / SCORES).is_file():
            self.records = [json.loads(line) for line in (self.output / SCORES).read_text().splitlines() if line]

    def evaluate(self, tagger) -> dict:
        tasks = tagger.tasks
        began = time.perf_counter()
        found = _tagging.predict_examples(tagger, self.examples, self.rows, self.pad_id, self.budget, self.decode)
        scores = _tagging.scores_of(found, self.examples, self.rows, tasks)
        scores["score"] = _tagging.score(scores, tasks)
        scores["seconds"] = time.perf_counter() - began
        return scores

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        from meepsquared import _sessions

        tagger = self.trainer.model
        epoch = len(self.records) + 1
        record = {"epoch": epoch, "step": state.global_step}
        if self.examples:
            record.update(self.evaluate(tagger))
            self.trainer.log({f"eval_{k}": v for k, v in record.items() if k not in ("epoch", "step", "words")})
        if _sessions.main_process():
            self.output.mkdir(parents=True, exist_ok=True)
            with open(self.output / SCORES, "a") as f:
                f.write(json.dumps(record) + "\n")
        best = max((r["score"] for r in self.records if "score" in r), default=None)
        self.records.append(record)
        if self.keep == "best" and (best is None or record["score"] > best) and _sessions.main_process():
            _tagging.save(tagger, self.output / BEST, self.config)
        if self.keep == "averaged" and epoch > self.epochs - self.average_last and _sessions.main_process():
            from safetensors.torch import load_file, save_file

            state_now = {k: v.float().cpu() for k, v in _tagging.trainable_state(tagger).items()}
            path = self.output / AVERAGE_SUM
            if path.is_file() and epoch > self.epochs - self.average_last + 1:
                summed = load_file(str(path))
                state_now = {k: summed[k] + v for k, v in state_now.items()}
            save_file(state_now, str(path))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """The kept weights put in place, so that the job's Result saves them."""
        if self.keep == "best":
            load_into(self.trainer.model, self.output / BEST / _tagging.WEIGHTS)
        elif self.keep == "averaged":
            import torch
            from safetensors.torch import load_file

            summed = load_file(str(self.output / AVERAGE_SUM))
            counted = min(self.average_last, len(self.records))
            parameters = {n: p for n, p in self.trainer.model.named_parameters() if p.requires_grad}
            with torch.no_grad():
                for name, value in summed.items():
                    parameters[name].copy_((value / counted).to(parameters[name].device, parameters[name].dtype))
        return control


def step_inputs(tagger):
    """How a batch of a shape is made for the timed steps: every subword but the two special tokens a word, so that
    the parser scores the most words the shape can hold, every label 0."""
    import torch

    def inputs_of(shape, device) -> dict:
        words = max(1, shape.length - 2)
        batch = {"input_ids": torch.zeros((shape.rows, shape.length), dtype=torch.long, device=device),
                 "attention_mask": torch.ones((shape.rows, shape.length), dtype=torch.long, device=device),
                 "first": torch.arange(1, words + 1, device=device).expand(shape.rows, -1).contiguous()}
        for column in tagger.label_columns:
            batch[column] = torch.zeros((shape.rows, words), dtype=torch.long, device=device)
        return batch

    return inputs_of


def prepare(job):
    """A tagging job's plan and trainer (see the module's docstring); returns the job's _Prepared."""
    import datasets
    import torch
    from transformers import AutoModel, AutoTokenizer

    from meepsquared import _allocator, _batches, _checkpointing, _selective, _sessions, _unified_memory
    from meepsquared.job import (RANK, TIMED_STEPS, Decision, Plan, _decide, _default, _gib, _measure,
                                 _measure_figures, _ordered, _Prepared, _recapped, _timed_steps_fitting)

    recipe, overrides = job.recipe, job.overrides
    output = job._output(job.model)
    first = {d["name"]: d for d in job._first["decisions"]} if job._first is not None else None
    if first is None and (Path(output) / SCORES).is_file():
        raise ValueError(f"{output} holds the epoch scores of another training ({SCORES}); give another output "
                         "folder, or resume that training")
    conllu = any(str(p).endswith(".conllu") for p in _paths(job.data) + _paths(job.eval_data))
    if recipe.words is not None and not conllu:
        raise ValueError("recipe.words chooses how a CoNLL-U file is read, and no .conllu file is given")
    words = recipe.words if recipe.words is not None else "syntactic"
    decisions = []

    # The sentences, and the tasks
    rows, source = rows_of(job.data, words, "data")
    eval_rows, eval_source = None, None
    if job.eval_data is not None:
        eval_rows, eval_source = rows_of(job.eval_data, words, "eval_data")
    columns = list(rows[0])
    if _tagging.TOKENS not in columns:
        raise ValueError(f"data has the columns {columns}; a tagger takes the words of each sentence as 'tokens'")
    names = _tagging.tasks_of([c for c in columns], recipe.labels)
    _tagging.check_rows(rows, names, "data")
    held = 0
    if job.eval_share is not None:
        trained, held_out = _batches.held_out(len(rows), job.eval_share, _default(TrainingArguments, "seed"))
        eval_rows, eval_source = [rows[i] for i in held_out], f"{source}, held out"
        rows = [rows[i] for i in trained]
        held = len(held_out)
    if eval_rows is not None:
        _tagging.check_rows(eval_rows, names, "eval_data")
    tasks = _tagging.build_tasks(rows, names)

    # The encoder, its tokenizer, and the adapter
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    shared = cuda and _unified_memory.shares_system_memory(local_rank)
    cap = _unified_memory.cap_allocator(local_rank) if shared else None
    if cap:
        decisions.append(Decision("allocator_cap", "allocator cap", cap, f"{cap / 2**30:.1f} GiB",
                                  "the system's available memory less 4 GiB for the page cache and the machine's "
                                  "daemons; an overflow is this process's error, not the machine's"))
    tokenizer = AutoTokenizer.from_pretrained(job.model)
    load_args = {"dtype": torch.bfloat16 if cuda else torch.float32}
    if overrides.attention is not None:
        load_args["attn_implementation"] = overrides.attention
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(job.model)
    if config.model_type in _tagging.BOTTLENECK_TYPES:
        load_args["add_pooling_layer"] = False
    encoder = AutoModel.from_pretrained(job.model, config=config, **load_args)
    if cuda:
        encoder.to(f"cuda:{local_rank}")
    device = next(encoder.parameters()).device
    decisions.append(Decision("loading", "loading", "whole", f"the encoder to {device}",
                              f"{config.model_type}, {sum(p.numel() for p in encoder.parameters()) / 1e6:,.0f}M "
                              "parameters, frozen"))
    decisions.append(Decision("weights", "weights", job.method, str(load_args["dtype"]).removeprefix("torch."),
                              "the encoder's weights; the adapter and heads train in float32 under autocast"))
    attention = encoder.config._attn_implementation
    decisions.append(_decide("attention", "attention", overrides.attention, attention, str,
                             "Transformers' default for the encoder"))

    tokenizer_limit = _tagging.max_subwords(tokenizer, config)
    max_length = recipe.max_length if recipe.max_length is not None else tokenizer_limit
    if max_length > tokenizer_limit:
        raise ValueError(f"recipe.max_length={max_length}, and the encoder takes at most {tokenizer_limit} subwords")
    examples, dropped = _tagging.encode(tokenizer, rows, tasks, max_length, "data")
    eval_examples, eval_dropped = ([], [])
    if eval_rows is not None:
        eval_examples, eval_dropped = _tagging.encode(tokenizer, eval_rows, tasks, max_length, "eval_data")
    if not examples:
        raise ValueError(f"no sentence of the data fits {max_length} subwords")
    decisions.append(Decision("max_length", "subwords", max_length,
                              f"up to {max_length} per sentence, the special tokens counted; "
                              f"{len(dropped):,} training sentences longer are left out"
                              + (f", and {len(eval_dropped):,} evaluation sentences" if eval_rows is not None else ""),
                              "as given" if recipe.max_length is not None else "the tokenizer's limit"))

    method_config = {"method": job.method, "model": job.model, "words": words,
                     "dropout": recipe.head_dropout if recipe.head_dropout is not None else HEAD_DROPOUT}
    if job.method == "bottleneck":
        method_config["bottleneck"] = recipe.bottleneck if recipe.bottleneck is not None else BOTTLENECK
        decisions.append(_decide("bottleneck", "adapters", recipe.bottleneck, BOTTLENECK,
                                 lambda n: f"bottleneck adapters of {n} after every layer's feed-forward block",
                                 f"Pfeiffer et al. (2020)'s place; {TEXPLICIT}"))
    else:
        rank = recipe.rank if recipe.rank is not None else RANK
        method_config.update(rank=rank, alpha=recipe.alpha if recipe.alpha is not None else rank,
                             lora_dropout=recipe.dropout if recipe.dropout is not None else 0.0,
                             targets=recipe.targets if recipe.targets is not None else LORA_TARGETS)
        decisions.append(_decide("rank", "LoRA rank", recipe.rank, RANK, str, "meepsquared's default"))
        decisions.append(_decide("alpha", "LoRA alpha", recipe.alpha, rank, str, "alpha = rank, a scaling of 1"))
        decisions.append(_decide("dropout", "LoRA dropout", recipe.dropout, 0.0, str, "get_peft_model's default"))
        decisions.append(_decide("targets", "targets", recipe.targets, LORA_TARGETS, ", ".join,
                                 "the attention's projections and every dense layer"))
    tagger = _tagging.build(encoder, method_config, tasks)
    tagger._meepsquared = {"tagger": method_config}
    decisions.append(_decide("head_dropout", "dropout", recipe.head_dropout, HEAD_DROPOUT, lambda d: f"{d:g}",
                             f"on the words' vectors and in the parser; {TEXPLICIT}"))

    def tasks_text(chosen) -> str:
        def text(name: str) -> str:
            if name in tasks.tags:
                return f"{name} ({len(tasks.tags[name]):,} labels)"
            if name == _tagging.PARSE:
                return (f"the parser ({len(tasks.relations):,} relations)" if tasks.relations is not None
                        else "the parser")
            return name

        return ", ".join(text(name) for name in chosen)

    decisions.append(_decide("labels", "tasks", recipe.labels, _tagging.tasks_of(columns, None), tasks_text,
                             "every column of labels the data has; the lemma as an edit rule from the word"))
    if conllu:
        decisions.append(_decide("words", "words", recipe.words, "syntactic", str,
                                 "the CoNLL-U files read as UD's words; 'surface' keeps a multi-word token as one"))

    # The trainer
    optimizer = recipe.optimizer if recipe.optimizer is not None else TAGGING_OPTIMIZER
    rate = recipe.learning_rate if recipe.learning_rate is not None else TAGGING_LEARNING_RATE
    schedule = recipe.schedule if recipe.schedule is not None else "linear"
    tokens = recipe.tokens_per_update if recipe.tokens_per_update is not None else TOKENS_PER_UPDATE
    world = int(os.environ.get("WORLD_SIZE", 1))
    if tokens % world:
        raise ValueError(f"tokens_per_update {tokens} does not divide among {world} processes")
    args = {"output_dir": output, "bf16": cuda, "use_cpu": not cuda, "per_device_train_batch_size": 1,
            "learning_rate": rate, "optim": optimizer, "logging_steps": 1, "remove_unused_columns": False,
            "weight_decay": recipe.weight_decay if recipe.weight_decay is not None else TAGGING_WEIGHT_DECAY,
            "lr_scheduler_type": {"constant": "constant_with_warmup", "cosine": "cosine_with_min_lr",
                                  "linear": "linear"}[schedule],
            "label_names": tagger.label_columns}
    if schedule == "cosine":
        from meepsquared.job import COSINE_FLOOR

        args["lr_scheduler_kwargs"] = {"min_lr_rate": COSINE_FLOOR}
    if job.steps is not None:
        args["max_steps"] = job.steps
    else:
        args["num_train_epochs"] = job.epochs if job.epochs is not None else 1
    if recipe.max_grad_norm is not None:
        args["max_grad_norm"] = recipe.max_grad_norm
    if job.report_to is not None:
        args["report_to"] = list(job.report_to) if isinstance(job.report_to, tuple) else job.report_to
    train_dataset = datasets.Dataset.from_list(examples)
    trainer = TaggingTrainer(model=tagger, args=TaggingArguments(**args), train_dataset=train_dataset,
                             data_collator=_tagging.Collator(tokenizer.pad_token_id, tagger.label_columns))
    trainer._meepsquared_tagger = method_config
    loading = _batches.loading(trainer)
    lengths = [len(e["input_ids"]) for e in examples]

    # The batch: a share of the budget per micro-batch, halved until the largest the training runs fits
    trained_bytes = _checkpointing.trainable_bytes(tagger, optimizer)
    trained_count = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
    layer_count = len(_checkpointing.layers(tagger))
    measure = cuda and first is None
    accumulation = first["accumulation"]["value"] if first is not None else 1

    def laid_out(accumulation: int):
        return _batches.lay_out(lengths, loading, accumulation, job.steps, job.epochs,
                                budget=math.ceil(tokens / world / accumulation), by_length=True)

    measured = {}
    while True:
        layout = laid_out(accumulation)
        shape = _checkpointing.largest_batch(layout.trained, lengths, False)
        if not measure:
            break
        if not measured and shared:
            cap = _unified_memory.recap_allocator(local_rank)
            decisions[:] = [_recapped(d, cap) if d.name == "allocator_cap" else d for d in decisions]
        every, per_layer, free, total = measured[accumulation] = _measure(tagger, shape)
        margin = _checkpointing.margin(device, free, total)
        if _checkpointing.fits(free - trained_bytes, margin, every, per_layer):
            break
        if shape.rows == 1:
            raise RuntimeError(f"the largest batch, {shape.text()}, one sentence, does not fit with every layer "
                               f"checkpointed: it holds {_gib(every + per_layer)}, and {_gib(free - trained_bytes - margin)} "
                               "is free beyond the margin and the trainable parameters; lower recipe.max_length")
        accumulation *= 2
    trainer.args.gradient_accumulation_steps = accumulation
    _batches.install(trainer, layout)
    decisions.append(_decide("tokens_per_update", "batch", recipe.tokens_per_update, TOKENS_PER_UPDATE,
                             lambda n: f"{n:,} subwords per update, padded, sentences of about one length together",
                             TEXPLICIT))
    decisions.append(Decision("accumulation", "accumulation", accumulation,
                              f"{accumulation} micro-batches per update of up to {math.ceil(tokens / world / accumulation):,} "
                              "subwords each", "the largest share of the budget that fits with every layer checkpointed"
                              if measure else "the model is not on a CUDA GPU" if first is None
                              else "a resumed training: the first session's"))
    decisions.append(Decision("order", "order", loading.seed,
                              f"shuffled with seed {loading.seed}, sorted by length and grouped, the groups shuffled",
                              "the trainer's seed; the training runs the micro-batches measured"))

    # The layers to recompute
    compiled = False
    selective_refusal = _selective.refusal(tagger, compiled)
    figures = margin_bytes = free_bytes = None
    if first is not None and first["checkpointed_layers"]["value"] is not None:
        layout_chosen = _checkpointing.Recompute(first["checkpointed_layers"]["value"],
                                                 first["selective_layers"]["value"] or 0)
        auto_layout, checkpoint_why = layout_chosen, "a resumed training: the first session's"
    elif measure:
        every, per_layer, free, total = measured[accumulation]
        free_bytes = free
        figures = _measure_figures(tagger, shape, every, per_layer, compiled)
        margin_bytes = _checkpointing.margin(device, free, total)
        auto_layout = _checkpointing.choose(free - trained_bytes, margin_bytes, figures.per_layer,
                                            figures.per_selective, figures.times, layer_count)
        checkpoint_why = (f"{figures.text(shape.text())}; free: {free / 2**30:.0f} of {total / 2**30:.0f} GiB, "
                          f"{_gib(trained_bytes)} of it for the gradients and the {optimizer} state of "
                          f"{trained_count:,} trainable parameters")
    else:
        auto_layout, checkpoint_why = _checkpointing.Recompute(0, 0), "the model is not on a CUDA GPU"
    if overrides.checkpointed_layers is not None:
        layout_chosen = _checkpointing.Recompute(overrides.checkpointed_layers, overrides.selective_layers or 0)
    elif first is None:
        layout_chosen = auto_layout
    _checkpointing.set_checkpointed(tagger, layout_chosen)

    def layers_text(n) -> str:
        return f"{n} of {layer_count} layers" if n is not None else "not measured"

    decisions.append(_decide("checkpointed_layers", "checkpoint", overrides.checkpointed_layers, auto_layout.whole,
                             layers_text, checkpoint_why))
    decisions.append(_decide("selective_layers", "selective",
                             overrides.selective_layers if overrides.selective_layers is not None
                             else 0 if overrides.checkpointed_layers is not None else None,
                             auto_layout.selective, layers_text,
                             selective_refusal or "recompute their cheap operations only, keeping the outputs of "
                                                  "their products"))

    # Learning
    decisions.append(_decide("optimizer", "optimizer", recipe.optimizer, TAGGING_OPTIMIZER, str,
                             f"{trained_count:,} trainable parameters: an 8-bit state would save little"))
    decisions.append(_decide("learning_rate", "learning rate", recipe.learning_rate, TAGGING_LEARNING_RATE,
                             lambda r: f"{r:g}", TEXPLICIT))
    planned = layout.steps
    warmup_auto = int(TAGGING_WARMUP_SHARE * planned)
    warmup = recipe.warmup_steps if recipe.warmup_steps is not None else warmup_auto
    trainer.args.warmup_steps = warmup
    decisions.append(_decide("schedule", "schedule", recipe.schedule, "linear",
                             lambda s: {"linear": "linear decay to 0 after the warm-up",
                                        "constant": "constant after the warm-up",
                                        "cosine": "cosine decay to a tenth"}[s], TEXPLICIT))
    decisions.append(_decide("warmup_steps", "warm-up", recipe.warmup_steps, warmup_auto, lambda n: f"{n} steps",
                             f"{TAGGING_WARMUP_SHARE:.0%} of the {planned} planned steps; {TEXPLICIT}"))
    decisions.append(_decide("weight_decay", "weight decay", recipe.weight_decay, TAGGING_WEIGHT_DECAY,
                             lambda n: f"{n:g}", TEXPLICIT))
    decisions.append(_decide("max_grad_norm", "clipping", recipe.max_grad_norm,
                             _default(TrainingArguments, "max_grad_norm"),
                             lambda n: f"gradient norm at most {n:g}", "Transformers' default"))
    epochs_laid = len(layout.epochs)
    length = (f"{job.steps} steps" if job.steps is not None
              else f"{job.epochs or 1:g} epoch{'s' if (job.epochs or 1) != 1 else ''}, {planned} steps")
    decisions.append(Decision("length", "length", planned, length,
                              "as given" if job.steps or job.epochs else "one epoch, as no steps were given"))

    # What is kept, and the evaluation
    keep = recipe.keep if recipe.keep is not None else ("best" if eval_rows is not None else "last")
    average_last = recipe.average_last if recipe.average_last is not None else 0
    if keep == "best" and eval_rows is None:
        raise ValueError("recipe.keep='best' chooses the epoch by the evaluation data; give eval_data or eval_share")
    if keep == "averaged" and not 1 <= average_last <= epochs_laid:
        raise ValueError(f"recipe.keep='averaged' averages the last recipe.average_last epochs, 1 to the "
                         f"{epochs_laid} the training runs, not {average_last}")
    if keep != "averaged" and recipe.average_last is not None:
        raise ValueError("recipe.average_last is the number of epochs recipe.keep='averaged' averages")
    keep_text = {"best": "the weights of the epoch of the best evaluation score",
                 "last": "the weights of the last step",
                 "averaged": f"the average of the weights of the last {average_last} epochs"}
    decisions.append(_decide("keep", "keep", recipe.keep, "best" if eval_rows is not None else "last",
                             keep_text.get, "the score: the mean of each task's accuracy and, with the parser, LAS"
                             if eval_rows is not None else "no evaluation data"))
    decode = recipe.decode if recipe.decode is not None else "mst"
    decisions.append(_decide("decode", "trees", recipe.decode if tasks.relations is not None else None,
                             "mst" if tasks.relations is not None else None,
                             lambda d: {"mst": "a tree with one root (Chu-Liu-Edmonds)",
                                        "argmax": "each word's best head, not always a tree", None: "no parser"}[d],
                             "UD's rule: every sentence a tree with one root" if tasks.relations is not None else ""))
    budget = math.ceil(tokens / world / accumulation)
    scorer = EpochScores(trainer, eval_examples, eval_rows or [], tokenizer.pad_token_id, budget, decode, keep,
                         average_last, epochs_laid, output, method_config)
    trainer.add_callback(scorer)
    if eval_rows is not None:
        decisions.append(Decision("evaluation", "evaluation", {"data": eval_source, "rows": len(eval_examples)},
                                  f"{len(eval_examples):,} sentences of {eval_source}, predicted and scored at the end "
                                  f"of each of the {epochs_laid} epochs",
                                  "each task's accuracy over all words; with the parser UAS and LAS"))
    else:
        decisions.append(Decision("evaluation", "evaluation", None, "none", "no eval_data or eval_share given"))
    decisions.append(Decision("data", "data", source, f"{source}: {len(rows):,} sentences, "
                              f"{sum(len(r[_tagging.TOKENS]) for r in rows):,} words"
                              + (f", {held:,} of them held out for evaluation" if held else "")))
    decisions.append(Decision("output", "output", output, output,
                              "as given" if job.output is not None else "named after the model and the task"))
    reported = list(trainer.args.report_to or [])
    decisions.append(Decision("report_to", "tracking", reported, ", ".join(reported) or "none",
                              "as given" if job.report_to is not None else "Transformers' default"))
    decisions.append(Decision("allocator", "allocator", _allocator.DECISION, _allocator.DECISION))

    # The time per step, which also checks the memory; one evaluation pass timed
    timed = job.timed_steps if job.timed_steps is not None else TIMED_STEPS
    seconds_per_step = estimated = timing = None
    eval_seconds = 0.0
    if timed == 0:
        time_why = "timed_steps=0"
    elif first is not None:
        time_why = "a resumed training: the first session's timing"
    elif not cuda and job.timed_steps is None:
        time_why = "the model is not on a CUDA GPU; timed_steps set by hand times the steps there too"
    else:
        shapes = _checkpointing.batch_shapes(layout.trained, lengths, False)
        largest = max(shapes, key=lambda s: s.tokens)
        median = _checkpointing.median_batch(shapes)
        mean = _checkpointing.mean_tokens(shapes)
        inputs_of = step_inputs(tagger)
        timing, layout_chosen, corrected = _timed_steps_fitting(
            trainer, tagger, largest, accumulation, timed, None, layout_chosen, figures, layer_count, device,
            overrides.checkpointed_layers is not None, margin_bytes, inputs_of)
        if corrected:
            for i, d in enumerate(decisions):
                if d.name == "checkpointed_layers":
                    decisions[i] = Decision(d.name, d.label, layout_chosen.whole, layers_text(layout_chosen.whole),
                                            f"{corrected}; before the check: {d.text}, {d.reason}")
        small = timing
        if median.tokens != largest.tokens:
            small = _sessions.time_steps(trainer, median, accumulation, timed, None, inputs_of)
        seconds_per_step = _sessions.on_the_line(median.tokens, small.step_seconds, largest.tokens,
                                                 timing.step_seconds, mean)
        estimated = seconds_per_step * planned
        if eval_examples:
            eval_seconds = scorer.evaluate(tagger)["seconds"]
            estimated += eval_seconds * epochs_laid
        time_why = (f"the median of {timed} optimizer steps timed after {_sessions.TIMING_WARMUP} to warm up, on the "
                    f"largest micro-batch, {largest.text()}, and on the median one, {median.text()}; each step priced "
                    f"on the line through the two at the mean micro-batch, {mean:,.0f} subwords"
                    + ("; one evaluation pass timed" if eval_examples else ""))
    duration = "not timed"
    if seconds_per_step is not None:
        duration = f"about {seconds_per_step:.3g} s per step, {_sessions.duration_text(estimated)} for {planned} steps"
        if eval_examples:
            duration += (f" with {epochs_laid} evaluations ({_sessions.duration_text(eval_seconds * epochs_laid)} "
                         "of it)")
    decisions.append(Decision("duration", "time", seconds_per_step, duration, time_why))
    peak_gpu = peak_host = request = None
    if timing is None:
        decisions.append(Decision("memory", "memory", None, "not measured",
                                  f"measured over the timed steps, and none ran: {time_why}"))
    else:
        peak_gpu, peak_host = timing.peak_gpu_bytes, timing.peak_host_bytes
        request, request_why = _sessions.memory_request(timing)
        decisions.append(Decision("memory", "memory", {"peak_gpu_bytes": peak_gpu, "peak_host_bytes": peak_host,
                                                       "memory_request_bytes": request},
                                  (f"peak {_gib(peak_gpu)} on the GPU, " if peak_gpu is not None else "")
                                  + f"{_gib(peak_host)} on the host; a job should ask for {request // 2**30} GiB "
                                    f"({request_why})", "the most PyTorch reserved over the timed steps"))
    if first is not None:
        from meepsquared.job import RESUMED

        decisions = [Decision(**first[d.name]) if d.name in RESUMED and d.name in first else d for d in decisions]
        free_bytes, seconds_per_step, estimated = (job._first["free_bytes"], job._first["seconds_per_step"],
                                                   job._first["estimated_seconds"])
        peak_gpu, peak_host, request = (job._first["peak_gpu_bytes"], job._first["peak_host_bytes"],
                                        job._first["memory_request_bytes"])
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "the CPU"
    plan = Plan(model=job.model, task=job.task, method=job.method, device=device_name, free_bytes=free_bytes,
                decisions=_ordered(decisions), seconds_per_step=seconds_per_step, estimated_seconds=estimated,
                peak_gpu_bytes=peak_gpu, peak_host_bytes=peak_host, memory_request_bytes=request)
    prepared = _Prepared(tagger, tokenizer, trainer, plan, output)
    prepared.scores = scorer
    return prepared


def _paths(data) -> list:
    if data is None:
        return []
    return list(data) if isinstance(data, (list, tuple)) else [data]
