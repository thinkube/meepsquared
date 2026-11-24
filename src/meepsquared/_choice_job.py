# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""TrainingJob(task="choice"): candidates after one shared text, scored against a target distribution (see
meepsquared._choice).

The plan reads the rows as preprocessed (encoded once, in a folder keyed by
what made it), loads the model as an SFT job does, with "lora", "qlora", "fp8" or "nvfp4",
puts LoRA on the attention projections of the language model (its towers
left out) and builds the head. Then it times training steps on the
question the model reads the most tokens of, which also measures the peak
memory a job should ask for.

Training runs Transformers' Trainer, one question per forward pass,
effective_batch questions per optimizer step, the rows in the order of the
data. AdamW updates the adapter at learning_rate and the head at
head_learning_rate; the schedule "ramped_linear" multiplies both at
optimizer step s (from 0) by min(1, (s + 1) / warmup_steps) * (1 - s / total
steps). For the first head_only questions of the data the adapter is frozen
and the language model reads without gradients; the head learns. The loss of
a step is the sum of its questions' losses, each divided by effective_batch.

A checkpoint holds the adapter, the head, the optimizer, the scheduler and the
trainer's state; a resumed training goes on at its question and step.
start_from starts a new training from a state made outside meepsquared (see
meepsquared._choice.read_start) at question N and optimizer step N /
effective_batch: the trainer is resumed from a checkpoint folder made for it,
whose trainer state says that step, and the tensors and AdamW's state are
copied in by name.

Milestones save the adapter and the head apart and, with milestone_eval,
answer its rows: each row's top candidate, and the count of right ones.
"""

import functools
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from meepsquared import _choice, _milestones

# The defaults of a choice job, each stated in the plan with its source
RANK = 16
ALPHA = 32
DROPOUT = 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 1e-3
EFFECTIVE_BATCH = 4
WARMUP_STEPS = 400
HEAD_ONLY = 500
CANDIDATES_PER_PASS = 4
HEAD = "set"
SET_LAYERS = 1
# The plan times one milestone on this many of its rows, spread over them, and scales the time by the tokens read:
# the rows' time is the tokens they read, and answering every row once can take as long as the rest of the plan
MILESTONE_SAMPLE = 64
NONE_OUTPUT = False
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
SCHEDULE = "ramped_linear"
ATTENTION = "sdpa"
# The 4-bit quantization of method "qlora": the reference training's, NF4 computing in bfloat16 without double
# quantization, so that an adapter trained on its base weights trains on the same weights here
QLORA_DTYPE = torch.bfloat16
QLORA_DOUBLE_QUANT = False
# The seed of the head's initial weights (Transformers' default seed)
SEED = 42
REFERENCE = "the setting of the reference choice training on Gemma 4 E4B"
# The format of the measurements a choice plan keeps for a later plan to reuse (measured_key): raised when what the
# time of a step or the memory depends on changes in Meep², so that measurements of other code are not reused
MEASUREMENTS = 1
# The lines of the plan that need the model loaded, which a plan reusing measurements takes from the plan that
# measured them: what they depend on is in the measurements' key
MODEL_DECISIONS = ("loading", "allocator_cap", "weights")


def ramp(step: int, warmup: int, total: int) -> float:
    """The factor of the learning rates at optimizer step `step` (from 0) of `total`: a linear warm-up over `warmup`
    steps, times a linear decay to 0."""
    rise = min(1.0, (step + 1) / warmup) if warmup else 1.0
    return rise * (1 - step / total)


class ChoiceArguments(TrainingArguments):
    """Transformers' TrainingArguments for a chooser: the model stays where it was loaded."""

    @property
    def place_model_on_device(self):
        return False


class Questions(torch.utils.data.Dataset):
    """The questions of a training in order: item i is row i % rows, at position i of the training."""

    def __init__(self, rows: int, count: int, evaluation: bool = False):
        self.rows, self.count, self.evaluation = rows, count, evaluation

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, i: int) -> dict:
        return {"row": i % self.rows, "position": i, "eval": self.evaluation}


def one(batch: list) -> dict:
    """A batch of one question, as it is."""
    if len(batch) != 1:
        raise ValueError(f"a choice batch is one question, not {len(batch)}")
    return batch[0]


class ChoiceTrainer(Trainer):
    """Transformers' Trainer for a chooser (see the module's docstring). settings: rows, eval_rows, head_only,
    effective_batch, learning_rate, head_learning_rate, weight_decay, warmup_steps, config, start (a start_from
    state or None), start_step."""

    def __init__(self, *args, settings: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self._start_folder = None
        # compute_loss divides each question's loss by effective_batch itself
        self.loss_is_scaled_for_ga = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        s = self.settings
        chooser = self.model
        row = (s["eval_rows"] if inputs["eval"] else s["rows"])[int(inputs["row"])]
        trains = int(inputs["position"]) >= s["head_only"]
        for p in chooser.adapter():
            p.requires_grad_(trains)
        loss = chooser.question_loss(row["shared"], row["own"], row["target"], trains,
                                     row.get("none")) / s["effective_batch"]
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        row = self.settings["eval_rows"][int(inputs["row"])]
        with torch.no_grad():
            loss = self.model.question_loss(row["shared"], row["own"], row["target"], False, row.get("none"))
        return loss.detach(), None, None

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            s = self.settings
            self.optimizer = torch.optim.AdamW(
                [{"params": self.model.adapter(), "lr": s["learning_rate"]},
                 {"params": list(self.model.head.parameters()), "lr": s["head_learning_rate"]}],
                lr=s["learning_rate"], weight_decay=s["weight_decay"])
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer if optimizer is not None else self.optimizer,
                functools.partial(ramp, warmup=self.settings["warmup_steps"], total=num_training_steps))
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _save(self, output_dir=None, state_dict=None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.args.should_save:
            _choice.save(self.model, output_dir, self.settings["config"])

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
        if resume_from_checkpoint == self._start_folder:
            _choice.apply_start(self.model, self.settings["start"])
        else:
            _choice.load(self.model, resume_from_checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint) -> None:
        if checkpoint is None or checkpoint != self._start_folder:
            super()._load_optimizer_and_scheduler(checkpoint)
            return
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        _choice.apply_start(self.model, self.settings["start"], optimizer)
        # The schedule at the start's step: the rates of that step, and the step the next one counts from
        step = self.settings["start_step"]
        scheduler = self.lr_scheduler
        scheduler.last_epoch = step
        rates = [base * scheduler.lr_lambdas[i](step) for i, base in enumerate(scheduler.base_lrs)]
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate
        scheduler._last_lr = rates

    def train(self, resume_from_checkpoint=None, **kwargs):
        """Trains; a training with a start_from state and no checkpoint to resume starts from that state."""
        if resume_from_checkpoint is not None or self.settings["start"] is None:
            return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
        from transformers.trainer import TRAINER_STATE_NAME
        from transformers.trainer_callback import ExportableState, TrainerState

        with tempfile.TemporaryDirectory(prefix="meepsquared-start-") as folder:
            # The callbacks' states as the Trainer records them in a state of its own
            state = TrainerState(stateful_callbacks=[cb for cb in self.callback_handler.callbacks + [self.control]
                                                     if isinstance(cb, ExportableState)])
            state.compute_steps(self.args, self.args.max_steps)
            state.global_step = self.settings["start_step"]
            state.max_steps = self.args.max_steps
            state.num_train_epochs = 1
            state.train_batch_size = 1
            state.save_to_json(os.path.join(folder, TRAINER_STATE_NAME))
            self._start_folder = folder
            try:
                return super().train(resume_from_checkpoint=folder, **kwargs)
            finally:
                self._start_folder = None


class ChoiceMilestones(_milestones.Milestones):
    """At the end of each milestone's step: the adapter and the head saved apart in <output>/milestones/step-<N> and,
    with rows, each row's top candidate (ANSWERS) and the count of right ones by group, recorded in milestone.json
    and logged as milestone/<group>/right and milestone/<group>/of. earlier: the milestone records of a start_from
    state, listed before this training's."""

    def __init__(self, trainer, output, at: list, found, config: dict, earlier: list, status=None, report=None,
                 answering=None):
        super().__init__(trainer, output, at, found, None, status=status, report=report)
        self.config, self.earlier = config, list(earlier)
        # (method, chooser): the second copy of the base that answers the rows (TrainingJob.milestone_weights)
        self.answering = answering

    def records(self) -> list:
        return self.earlier + super().records()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step not in self.at or not state.is_world_process_zero or _milestones.done(self.output, step) is not None:
            return
        began = time.perf_counter()
        chooser = self.trainer.model
        final = _milestones.folder(self.output, step)
        partial = final.with_name(final.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        _choice.save(chooser, partial, self.config)
        parts = {"saving": round(time.perf_counter() - began, 1)}
        record = {"step": step, "planned_steps": state.max_steps, "folder": str(final)}
        if self.found is not None:
            answerer = chooser
            if self.answering is not None:
                method, answerer = self.answering
                _choice.copy_trainable(chooser, answerer)
                record["weights"] = method
            found = _choice.answer(answerer, self.found)
            with (partial / _choice.ANSWERS).open("w") as f:
                for row, (best, probability) in zip(self.found, found):
                    f.write(json.dumps({"group": row.get("group") or _choice.GROUP, "chosen": best,
                                        "probability": round(probability, 6), "right": bool(row["right"][best])})
                            + "\n")
            record["scores"] = _choice.score(self.found, found)
            record["rows"] = len(self.found)
            parts["answering"] = round(time.perf_counter() - began - parts["saving"], 1)
        record["seconds"] = round(time.perf_counter() - began, 1)
        record["seconds_of"] = parts
        (partial / "milestone.json").write_text(json.dumps(record, indent=2))
        shutil.rmtree(final, ignore_errors=True)
        partial.rename(final)
        self.seconds = record["seconds"]
        if self.report is not None:
            self.report.evaluating += time.perf_counter() - began
        if record.get("scores"):
            # Logging resets the step's own log, which the trainer writes after this callback: it is kept
            logs_the_step = control.should_log
            self.trainer.log({f"milestone/{group}/{metric}": value
                              for group, figures in record["scores"].items() for metric, value in figures.items()})
            self.trainer.control.should_log = logs_the_step
        self._report(state)


def _language_model_targets(model, names: list) -> list:
    """The full names of the linear layers named one of `names`, outside the vision, audio and image modules."""
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in names
            and not any(tower in n for tower in _choice.TOWERS)]


def _load(job, decisions: list):
    """The model as an SFT job loads it, with the choice's attention and without Liger; and the loading
    decisions."""
    from meepsquared.job import Decision, _decide
    from meepsquared.models import MeepLanguageModel

    overrides = job.overrides
    if not isinstance(job.model, str):
        model = job.model
        if getattr(model, "_meepsquared", None) is None:
            raise TypeError("model: a loaded model must come from MeepLanguageModel.from_pretrained")
        if hasattr(model, "peft_config"):
            raise ValueError("model already has adapters; give it without them, and the job adds LoRA")
        if overrides.attention is not None:
            raise ValueError("overrides.attention is a loading choice, and the model is loaded already")
        loaded_as = ("qlora" if getattr(model, "is_loaded_in_4bit", False)
                     else "fp8" if model._meepsquared.get("fp8", False)
                     else "nvfp4" if model._meepsquared.get("nvfp4", False) else "lora")
        if loaded_as != job.method:
            raise ValueError(f"method={job.method!r}, and the model was loaded for {loaded_as!r}")
        decisions.append(Decision("loading", "loading", "given", "the model as given, loaded by the caller"))
        decisions.append(Decision("attention", "attention", model.config._attn_implementation,
                                  str(model.config._attn_implementation), "as the model was loaded"))
        return model
    attention = overrides.attention if overrides.attention is not None else ATTENTION
    four_bit = {"dtype": QLORA_DTYPE, "double_quant": QLORA_DOUBLE_QUANT} if job.method == "qlora" else {}
    model, _ = MeepLanguageModel.from_pretrained(job.model, load_in_4bit=job.method == "qlora",
                                                 load_in_fp8=job.method == "fp8",
                                                 load_in_nvfp4=job.method == "nvfp4", liger=False,
                                                 attn_implementation=attention, **four_bit)
    device = next(model.parameters()).device
    decisions.append(Decision("loading", "loading", "whole", f"to {device}", "as an SFT job loads it, without Liger: "
                              "the hidden states are read, not the language model's loss"))
    if model._meepsquared.get("allocator_cap"):
        cap = model._meepsquared["allocator_cap"]
        decisions.append(Decision("allocator_cap", "allocator cap", cap, f"{cap / 2**30:.1f} GiB",
                                  "the system's available memory less 4 GiB for the page cache and the machine's "
                                  "daemons; an overflow is this process's error, not the machine's"))
    decisions.append(_decide("attention", "attention", overrides.attention, ATTENTION, str,
                             "the candidates are read after the shared cache, right-padded, with a mask"))
    return model


def milestone_sample(rows: list) -> list:
    """At most MILESTONE_SAMPLE of the rows, spread evenly over them."""
    if len(rows) <= MILESTONE_SAMPLE:
        return list(rows)
    return [rows[i * len(rows) // MILESTONE_SAMPLE] for i in range(MILESTONE_SAMPLE)]


def attention_of(job) -> str:
    return job.overrides.attention if job.overrides.attention is not None else ATTENTION


def _answering_chooser(job, method: str, attention: str, rank: int, alpha: int, names: list, kind: str,
                       set_layers: int, none_output: bool, per_pass: int, pad_id: int):
    """A second copy of the base loaded as `method`, with the same adapter's shape and head, frozen, in evaluation
    mode: the milestones' rows are answered on it once the training's adapter and head are copied in."""
    from meepsquared.models import MeepLanguageModel

    four_bit = {"dtype": QLORA_DTYPE, "double_quant": QLORA_DOUBLE_QUANT} if method == "qlora" else {}
    model, _ = MeepLanguageModel.from_pretrained(job.model, load_in_4bit=method == "qlora",
                                                 load_in_fp8=method == "fp8", load_in_nvfp4=method == "nvfp4",
                                                 liger=False, attn_implementation=attention, **four_bit)
    for p in model.parameters():
        p.requires_grad_(False)
    peft = MeepLanguageModel.get_peft_model(model, r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                            target_modules=_language_model_targets(model, names),
                                            use_gradient_checkpointing=False, fused_lora=False, compile_layers=False)
    decoder = peft.get_base_model().get_decoder()
    head = _choice.build_head(decoder.config.hidden_size, kind, set_layers,
                              none_output).to(next(decoder.parameters()).device)
    answering = _choice.Chooser(peft, decoder, head, kind, per_pass, pad_id)
    for p in answering.parameters():
        p.requires_grad_(False)
    return answering.eval()


def measured_key(job) -> dict:
    """What the time of a step and the memory depend on, besides the GPU and the data's longest question: a later
    plan reuses them only when its job gives the same."""
    recipe = job.recipe
    kind = recipe.choice_head if recipe.choice_head is not None else HEAD
    return {"format": MEASUREMENTS,
            "model": job.model if isinstance(job.model, str) else str(job.model.config._name_or_path),
            "method": job.method, "attention": attention_of(job),
            "rank": recipe.rank if recipe.rank is not None else RANK,
            "targets": list(recipe.targets if recipe.targets is not None else TARGETS), "head": kind,
            "set_layers": ((recipe.set_layers if recipe.set_layers is not None else SET_LAYERS) if kind == "set"
                           else None),
            "none_output": recipe.none_output if recipe.none_output is not None else NONE_OUTPUT,
            "candidates_per_pass": (recipe.candidates_per_pass if recipe.candidates_per_pass is not None
                                    else CANDIDATES_PER_PASS)}


KEY_NAMES = {"format": "Meep²'s measurement format", "model": "the model", "method": "the method",
             "attention": "the attention", "rank": "the LoRA rank", "targets": "the LoRA targets", "head": "the head",
             "set_layers": "the set layers", "none_output": "the none output",
             "candidates_per_pass": "the candidates per pass"}


def reuse(job, earlier, reused_from: str = ""):
    """The plan of `job` with the measurements of the earlier plan, without loading the model or measuring (see
    TrainingJob.plan). MeasureAgain when they do not hold: what they depend on changed (measured_key); the data's
    longest question reads more tokens than the one measured, whose memory bounds the job's; or the job answers
    milestone rows with weights the earlier plan timed none with."""
    from meepsquared.job import MeasureAgain

    found = earlier.measurements
    key = measured_key(job)
    changed = [KEY_NAMES[k] for k in key if key[k] != found["key"].get(k)]
    if changed:
        raise MeasureAgain(f"{', '.join(changed)} changed, which the time of a step and the memory depend on")
    return prepare(job, found, reused_from).plan


def _weights_decision(job, model):
    from meepsquared.job import Decision

    dtype_name = str(model.get_input_embeddings().weight.dtype).removeprefix("torch.")
    if job.method == "qlora":
        nf4 = model._meepsquared.get("nf4_experts")
        experts = (f"; its {nf4['stacks']} expert stacks NF4 too, {nf4['bytes'] / 2**30:.1f} GiB, each dequantized "
                   "for its product" if nf4 else "")
        return Decision("weights", "weights", job.method,
                        f"4-bit NF4 without double quantization, computing in {dtype_name}{experts}; the head in "
                        "float32", "the quantization the reference training used")
    if job.method == "fp8":
        return Decision("weights", "weights", job.method,
                        f"the frozen linear layers of the decoder in FP8 (e4m3, one scale per tensor), their products "
                        f"on the FP8 tensor cores, computing in {dtype_name}; the head in float32",
                        "method fp8 (see meepsquared._fp8); the adapters, embeddings, norms and routers stay in "
                        f"{dtype_name}")
    if job.method == "nvfp4":
        return Decision("weights", "weights", job.method,
                        f"the frozen linear layers of the decoder and their inputs in NVFP4 (Transformer Engine), "
                        f"their products on the FP4 tensor cores, computing in {dtype_name}; the head in float32",
                        "method nvfp4 (see meepsquared._nvfp4); the adapters, embeddings, norms and routers stay in "
                        f"{dtype_name}")
    return Decision("weights", "weights", job.method, dtype_name, f"method {job.method}; the head in float32")


def prepare(job, reusing: dict = None, reused_from: str = ""):
    """A choice job's plan and trainer (see the module's docstring); returns the job's _Prepared.

    reusing: the measurements of an earlier plan whose key is this job's (see reuse). The plan is then made without
    the model, which is not loaded: the lines that need it are the measuring plan's, the times and the memory are its
    measurements, and the rest is decided as with the model. The _Prepared holds no model and no trainer."""
    from meepsquared import _allocator, _sessions
    from meepsquared.job import (EVALUATIONS, PREPROCESSED, RESUMED, TIMED_STEPS, Decision, MeasureAgain, Plan,
                                 _decide, _gib, _ordered, _Prepared, _reused_decision, _say, _steps_text)
    from meepsquared.models import MeepLanguageModel, load_tokenizer

    recipe, overrides = job.recipe, job.overrides
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        raise ValueError("a choice job trains in one process, and this one is one of a launch of several")
    output = job._output(job.model)
    first = {d["name"]: d for d in job._first["decisions"]} if job._first is not None else None
    decisions = []

    # The rows, encoded once (see meepsquared._choice.preprocess)
    tokenizer = job.tokenizer if job.tokenizer is not None else load_tokenizer(job.model)
    if tokenizer.pad_token_id is None:
        raise ValueError("the tokenizer has no pad token, which pads the candidates read together")
    folder = job.preprocessed
    if folder is None:
        folder = _choice.preprocess(job, Path(output) / PREPROCESSED, say=_say).folder
    rows, eval_rows, report = _choice.open_folder(folder, _choice.inputs(job, tokenizer))
    none_output = recipe.none_output if recipe.none_output is not None else NONE_OUTPUT
    if none_output:
        for where, found in (("data", rows), ("eval_data", eval_rows or [])):
            missing = next((i for i, r in enumerate(found) if r.get("none") is None), None)
            if missing is not None:
                raise ValueError(f"{where} row {missing}: 'none' is required with the head's none output: the index "
                                 "of the candidate that says no candidate applies")
    sizes = [_choice.tokens_read(r) for r in rows]
    if reusing is not None and max(sizes) > reusing["steps"]["largest_tokens"]:
        raise MeasureAgain(f"the data's longest question reads {max(sizes):,} tokens, more than the "
                           f"{reusing['steps']['largest_tokens']:,} of the one measured, whose memory bounds the job's")

    # The model, the adapter and the head
    rank = recipe.rank if recipe.rank is not None else RANK
    alpha = recipe.alpha if recipe.alpha is not None else ALPHA
    dropout = recipe.dropout if recipe.dropout is not None else DROPOUT
    names = recipe.targets if recipe.targets is not None else TARGETS
    kind = recipe.choice_head if recipe.choice_head is not None else HEAD
    set_layers = recipe.set_layers if recipe.set_layers is not None else SET_LAYERS
    per_pass = recipe.candidates_per_pass if recipe.candidates_per_pass is not None else CANDIDATES_PER_PASS
    chooser = model = None
    if reusing is None:
        model = _load(job, decisions)
        decisions.append(_weights_decision(job, model))
        targets = _language_model_targets(model, names)
        if not targets:
            raise ValueError(f"no linear layer of the language model is named one of {names}")
        for p in model.parameters():
            p.requires_grad_(False)
        peft = MeepLanguageModel.get_peft_model(model, r=rank, lora_alpha=alpha, lora_dropout=dropout,
                                                target_modules=targets, use_gradient_checkpointing=False,
                                                fused_lora=False, compile_layers=False)
        decoder = peft.get_base_model().get_decoder()
        device = next(decoder.parameters()).device
        torch.manual_seed(SEED)
        head = _choice.build_head(decoder.config.hidden_size, kind, set_layers, none_output).to(device)
        chooser = _choice.Chooser(peft, decoder, head, kind, per_pass, tokenizer.pad_token_id)
        model_name = job.model if isinstance(job.model, str) else str(model.config._name_or_path)
        target_count, cuda = len(targets), device.type == "cuda"
    else:
        # Reusing measurements, the lines that need the model are the measuring plan's: what they depend on is in
        # the measurements' key
        decisions.extend(Decision(**d) for d in reusing["facts"]["decisions"])
        if isinstance(job.model, str):
            decisions.append(_decide("attention", "attention", overrides.attention, ATTENTION, str,
                                     "the candidates are read after the shared cache, right-padded, with a mask"))
        else:
            decisions.append(Decision("attention", "attention", attention_of(job), attention_of(job),
                                      "as the model was loaded"))
        model_name, target_count, cuda = reusing["key"]["model"], reusing["facts"]["targets"], reusing["facts"]["cuda"]
    config = {"head": kind, "set_layers": set_layers, "none_output": none_output, "model": model_name,
              "method": job.method,
              "candidates_per_pass": per_pass}
    if chooser is not None:
        chooser._meepsquared = {"choice": config}
    decisions.append(_decide("rank", "LoRA rank", recipe.rank, RANK, str, REFERENCE))
    decisions.append(_decide("alpha", "LoRA alpha", recipe.alpha, ALPHA, str, REFERENCE))
    decisions.append(_decide("dropout", "LoRA dropout", recipe.dropout, DROPOUT, str, REFERENCE))
    decisions.append(_decide("targets", "targets", recipe.targets, TARGETS, ", ".join,
                             f"the attention's projections of the language model, its vision, audio and image "
                             f"modules left out: {target_count} layers"))
    head_text = {"set": "the set head: a LayerNorm, a scalar score, and an attention over the question's candidates",
                 "scalar": "a LayerNorm and a scalar score of each candidate alone"}
    decisions.append(_decide("choice_head", "head", recipe.choice_head, HEAD, head_text.get,
                             "the score of a candidate from the last state of its reading"))
    if kind == "set":
        decisions.append(_decide("set_layers", "set layers", recipe.set_layers, SET_LAYERS,
                                 lambda n: f"{n} attention layer{'s' if n != 1 else ''} over the candidates"
                                           + (", each over the previous one's output" if n > 1 else ""),
                                 REFERENCE))
    decisions.append(_decide("candidates_per_pass", "per pass", recipe.candidates_per_pass, CANDIDATES_PER_PASS,
                             lambda n: f"{n} candidates read together after the shared text, which is read once",
                             REFERENCE))
    decisions.append(_decide("none_output", "none output", recipe.none_output, NONE_OUTPUT,
                             lambda on: ("a separate output of the probability that no candidate applies, from the "
                                         "none candidate's state beside the others' largest; above 0.5 the answer is "
                                         "the none candidate, else the choice among the others" if on else "none: "
                                         "the none candidate, if any, is one of the candidates"),
                             "the head of the reference choice training has none"))

    # The length, and a start from a saved state
    effective = recipe.effective_batch if recipe.effective_batch is not None else EFFECTIVE_BATCH
    questions = job.steps * effective if job.steps is not None else math.ceil((job.epochs or 1) * len(rows))
    total = math.ceil(questions / effective)
    start, start_step = None, 0
    if job.start_from is not None:
        start = _choice.read_start(job.start_from)
        # Without the model the state's tensors are not checked against it here; the training checks them
        if chooser is not None:
            _choice.check_start(chooser, start)
        examples = start["examples"]
        if examples % effective:
            raise ValueError(f"start_from: {examples} questions are not whole optimizer steps of {effective}")
        if examples >= questions:
            raise ValueError(f"start_from: {examples} questions done, and the training has {questions}")
        start_step = examples // effective
        value = {"file": str(job.start_from), "examples": examples, "step": start_step}
        if first is not None:
            text = (f"a resumed session: it goes on from its checkpoint in {output}; the training began at question "
                    f"{examples:,}, step {start_step}, of {job.start_from}")
        else:
            text = (f"question {examples:,}, step {start_step} of {total}, from {job.start_from}: its adapter, head "
                    "and AdamW state; the questions before it are not read")
        decisions.append(Decision("start_from", "start", value, text, "start_from"))
    else:
        decisions.append(Decision("start_from", "start", None, "the first question, the head and adapter new",
                                  "no start_from given"))

    # Learning
    rate = recipe.learning_rate if recipe.learning_rate is not None else LEARNING_RATE
    head_rate = recipe.head_learning_rate if recipe.head_learning_rate is not None else HEAD_LEARNING_RATE
    warmup = recipe.warmup_steps if recipe.warmup_steps is not None else WARMUP_STEPS
    head_only = recipe.head_only if recipe.head_only is not None else HEAD_ONLY
    weight_decay = recipe.weight_decay if recipe.weight_decay is not None else WEIGHT_DECAY
    max_grad_norm = recipe.max_grad_norm if recipe.max_grad_norm is not None else MAX_GRAD_NORM
    decisions.append(_decide("effective_batch", "batch", recipe.effective_batch, EFFECTIVE_BATCH,
                             lambda n: f"{n} questions per update, one per forward pass", REFERENCE))
    decisions.append(Decision("order", "order", "data", "the rows in the order of the data, not shuffled",
                              "the data is written in the order to train"))
    decisions.append(Decision("optimizer", "optimizer", "adamw_torch",
                              "AdamW (PyTorch): the adapter's parameters, then the head's", REFERENCE))
    decisions.append(_decide("learning_rate", "learning rate", recipe.learning_rate, LEARNING_RATE,
                             lambda r: f"{r:g} for the adapter", REFERENCE))
    decisions.append(_decide("head_learning_rate", "head rate", recipe.head_learning_rate, HEAD_LEARNING_RATE,
                             lambda r: f"{r:g} for the head", REFERENCE))
    decisions.append(_decide("schedule", "schedule", recipe.schedule, SCHEDULE,
                             lambda s: "linear warm-up, then linear decay to 0 over the planned steps",
                             f"min(1, (s + 1) / warm-up) x (1 - s / {total}) at step s; {REFERENCE}"))
    decisions.append(_decide("warmup_steps", "warm-up", recipe.warmup_steps, WARMUP_STEPS, lambda n: f"{n} steps",
                             REFERENCE))
    decisions.append(_decide("head_only", "head only", recipe.head_only, HEAD_ONLY,
                             lambda n: f"the first {n:,} questions of the data train the head alone, the adapter "
                                       "frozen", REFERENCE))
    decisions.append(_decide("max_grad_norm", "clipping", recipe.max_grad_norm, MAX_GRAD_NORM,
                             lambda n: f"gradient norm at most {n:g}, the adapter and the head together", REFERENCE))
    decisions.append(_decide("weight_decay", "weight decay", recipe.weight_decay, WEIGHT_DECAY, lambda n: f"{n:g}",
                             REFERENCE))
    length = (f"{job.steps} steps" if job.steps is not None
              else f"{job.epochs or 1:g} epoch{'s' if (job.epochs or 1) != 1 else ''}")
    decisions.append(Decision("length", "length", total, f"{length}: {questions:,} questions, {total} steps",
                              "as given" if job.steps or job.epochs else "one epoch, as no steps were given"))

    # The trainer, built only with the model; its arguments are built either way, and say where it reports
    args = {"output_dir": output, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": effective, "max_steps": total, "learning_rate": rate,
            "weight_decay": weight_decay, "max_grad_norm": max_grad_norm, "optim": "adamw_torch", "logging_steps": 1,
            "remove_unused_columns": False, "label_names": [], "train_sampling_strategy": "sequential",
            "bf16": False, "fp16": False, "use_cpu": not cuda, "dataloader_num_workers": 0}
    if job.report_to is not None:
        args["report_to"] = list(job.report_to) if isinstance(job.report_to, tuple) else job.report_to
    every = None
    if eval_rows is not None:
        every_auto = max(1, math.ceil(total / EVALUATIONS))
        every = job.eval_every if job.eval_every is not None else every_auto
        args.update(eval_strategy="steps", eval_steps=every, prediction_loss_only=True)
    training_args = ChoiceArguments(**args)
    settings = {"rows": rows, "eval_rows": eval_rows, "head_only": head_only, "effective_batch": effective,
                "learning_rate": rate, "head_learning_rate": head_rate, "weight_decay": weight_decay,
                "warmup_steps": warmup, "config": config, "start": start, "start_step": start_step}
    trainer = None
    if chooser is not None:
        # One process trains on its one GPU: the Trainer's DataParallel would split a question
        if training_args.n_gpu > 1:
            training_args._n_gpu = 1
        trainer = ChoiceTrainer(model=chooser, args=training_args, train_dataset=Questions(len(rows), questions),
                                eval_dataset=(Questions(len(eval_rows), len(eval_rows), evaluation=True)
                                              if eval_rows is not None else None),
                                data_collator=one, processing_class=tokenizer, settings=settings)
    if eval_rows is not None:
        value = {"data": report["eval_source"], "rows": len(eval_rows), "every": every}
        evaluation_text = (f"{len(eval_rows):,} questions of {report['eval_source']}, every {every} steps, the loss "
                           "counted as the training's")
        if job.eval_every is None:
            decisions.append(Decision("evaluation", "evaluation", value, evaluation_text,
                                      f"about {EVALUATIONS} evaluations over the {total} planned steps"))
        else:
            decisions.append(Decision("evaluation", "evaluation", value, evaluation_text, "", "user",
                                      {**value, "every": every_auto}, f"every {every_auto} steps"))
    else:
        decisions.append(Decision("evaluation", "evaluation", None, "none", "no eval_data given"))
    decisions.append(Decision("data", "data", report["source"],
                              f"{report['source']}: {report['decision']['text']}", f"preprocessed in {folder}"))
    decisions.append(Decision("output", "output", output, output,
                              "as given" if job.output is not None else "named after the model and the task"))
    reported = list(training_args.report_to or [])
    decisions.append(Decision("report_to", "tracking", reported, ", ".join(reported) or "none",
                              "as given" if job.report_to is not None else "Transformers' default"))
    decisions.append(Decision("allocator", "allocator", _allocator.DECISION, _allocator.DECISION))

    # The milestones, and the weights that answer them
    answering = None
    answering_method = job.milestone_weights if job.milestone_weights is not None else job.method
    if job.milestone_weights is not None and job.milestone_weights != job.method:
        if reusing is None:
            answering = (job.milestone_weights,
                         _answering_chooser(job, job.milestone_weights, attention_of(job), rank, alpha, names, kind,
                                            set_layers, none_output, per_pass, tokenizer.pad_token_id))
        decisions.append(Decision("milestone_weights", "milestone weights", job.milestone_weights,
                                  f"a second copy of the base loaded as {job.milestone_weights} answers the "
                                  "milestones' rows, the adapter and the head copied into it at each milestone",
                                  "as given: the answers of the weights the model will be served on"))
    elif job.milestones is not None and job.milestone_eval is not None:
        decisions.append(Decision("milestone_weights", "milestone weights", job.method,
                                  f"the training's own weights ({job.method}) answer the milestones' rows",
                                  "as given" if job.milestone_weights is not None else "no milestone_weights given"))
    milestone_at = milestone_rows = None
    if job.milestones is not None:
        milestone_at = [s for s in _milestones.steps(total, job.milestones, job.milestone_growth) if s > start_step]
        if job.milestone_eval is not None:
            milestone_rows, _ = _choice.read_encoded(job.milestone_eval, tokenizer, "milestone_eval", target=False,
                                                     right=True, none=none_output)

    # The time per step, and the memory, from steps timed on the question of the most tokens read and on the median
    # one, each step priced on the line through the two at the mean question; reusing measurements, the measured
    # steps' times priced on the line at this data's mean question
    timed = job.timed_steps if job.timed_steps is not None else TIMED_STEPS
    seconds_per_step = estimated = timing = None
    milestone_each = None
    milestone_timing = ""
    steps = milestone = memory = None
    if reusing is not None:
        steps = reusing["steps"]
        per_question = _sessions.on_the_line(steps["median_tokens"], steps["median_seconds"], steps["largest_tokens"],
                                             steps["largest_seconds"], statistics.mean(sizes))
        seconds_per_step = per_question * effective
        estimated = seconds_per_step * (total - start_step)
        time_why = (f"{steps['how']}; each question priced on the line through the two at this data's mean, "
                    f"{statistics.mean(sizes):,.0f} tokens, {effective} questions a step, the adapter training in "
                    "each")
        memory = reusing["memory"]
        if milestone_rows is not None and milestone_at:
            milestone = reusing.get("milestone")
            if milestone is None or milestone["weights"] != answering_method:
                raise MeasureAgain(f"the milestones' rows are answered by the {answering_method} weights, and the "
                                   "earlier plan timed no milestone with them")
            milestone_each = milestone["seconds_per_token"] * sum(map(_choice.tokens_read, milestone_rows))
            milestone_timing = f"{milestone['how']}, priced by the tokens these rows read"
            estimated += milestone_each * len(milestone_at)
        elif reusing.get("milestone") is not None:
            milestone = reusing["milestone"]
    elif timed == 0:
        time_why = "timed_steps=0"
    elif first is not None:
        time_why = "a resumed training: the first session's timing"
    elif not cuda and job.timed_steps is None:
        time_why = "the model is not on a CUDA GPU; timed_steps set by hand times the steps there too"
    else:
        largest = max(range(len(rows)), key=lambda i: (sizes[i], len(rows[i]["own"])))
        median = sorted(range(len(rows)), key=lambda i: sizes[i])[len(rows) // 2]

        def inputs_of(row, device_):
            # A question past head_only: the adapter trains, the most a question holds
            return {"row": row, "position": head_only, "eval": False}

        timing = _sessions.time_steps(trainer, largest, 1, timed, None, inputs_of)
        small = timing
        if sizes[median] != sizes[largest]:
            small = _sessions.time_steps(trainer, median, 1, timed, None, inputs_of)
        per_question = _sessions.on_the_line(sizes[median], small.step_seconds, sizes[largest], timing.step_seconds,
                                             statistics.mean(sizes))
        seconds_per_step = per_question * effective
        estimated = seconds_per_step * (total - start_step)
        how = (f"the median of {timed} steps of one question timed after {_sessions.TIMING_WARMUP} to warm up, on the "
               f"question of the most tokens read, {sizes[largest]:,} tokens and {len(rows[largest]['own'])} "
               f"candidates, and on the median one, {sizes[median]:,} tokens")
        time_why = (f"{how}; each question priced on the line through the two at the mean, "
                    f"{statistics.mean(sizes):,.0f} tokens, {effective} questions a step, the adapter training in "
                    "each")
        steps = {"largest_tokens": sizes[largest], "largest_seconds": timing.step_seconds,
                 "median_tokens": sizes[median], "median_seconds": small.step_seconds, "how": how}
        if milestone_rows is not None and milestone_at:
            sample = milestone_sample(milestone_rows)
            if answering is not None:
                _choice.copy_trainable(chooser, answering[1])
            began = time.perf_counter()
            _choice.answer(answering[1] if answering is not None else chooser, sample)
            sample_tokens = sum(map(_choice.tokens_read, sample))
            milestone_each = ((time.perf_counter() - began) * sum(map(_choice.tokens_read, milestone_rows))
                              / sample_tokens)
            milestone_timing = (f"timed on {len(sample)} of the {len(milestone_rows):,} rows, spread over them, and "
                                "scaled by the tokens they read" if len(sample) < len(milestone_rows)
                                else f"timed on its {len(milestone_rows)} rows")
            milestone = {"weights": answering_method,
                         "seconds_per_token": milestone_each / sum(map(_choice.tokens_read, milestone_rows)),
                         "how": f"timed on {len(sample)} rows of {sample_tokens:,} tokens"}
            estimated += milestone_each * len(milestone_at)
    if job.milestones is not None:
        what = f"the adapter and the head saved apart in {output}/{_milestones.FOLDER}/step-<N>"
        if milestone_rows is not None:
            groups = sorted({row.get("group") or _choice.GROUP for row in milestone_rows})
            what += (f", and {len(milestone_rows):,} rows answered by their top candidate, the right ones counted "
                     f"({', '.join(groups)})")
            if milestone_each is not None:
                what += f"; each about {_sessions.duration_text(milestone_each)} ({milestone_timing})"
        decisions.append(Decision("milestones", "milestones", {"steps": milestone_at, "seconds": milestone_each},
                                  f"{len(milestone_at)} at steps "
                                  f"{_steps_text(milestone_at, job.milestone_growth is not None)}: {what}",
                                  _milestones.spacing_text(total, job.milestones, job.milestone_growth)))
    else:
        decisions.append(Decision("milestones", "milestones", None, "none", "no milestones given"))
    duration = "not timed"
    milestone_seconds = milestone_each * len(milestone_at) if milestone_each is not None and milestone_at else 0.0
    if seconds_per_step is not None:
        duration = _sessions.time_line(seconds_per_step, estimated, total - start_step,
                                       milestones=len(milestone_at or []), milestone_seconds=milestone_seconds,
                                       evaluations_counted=False)
    decisions.append(Decision("duration", "time", seconds_per_step, duration, time_why))
    peak_gpu = peak_host = request = None
    memory_why = ("the GPU: the most PyTorch reserved over the timed steps on the question of the most tokens read; "
                  "the host: the process's peak resident memory, loading included")
    if timing is not None:
        request, request_why = _sessions.memory_request(timing)
        memory = {"peak_gpu_bytes": timing.peak_gpu_bytes, "peak_host_bytes": timing.peak_host_bytes,
                  "request": request, "why": request_why}
    if memory is None:
        decisions.append(Decision("memory", "memory", None, "not measured",
                                  f"measured over the timed steps, and none ran: {time_why}"))
    else:
        peak_gpu, peak_host, request = memory["peak_gpu_bytes"], memory["peak_host_bytes"], memory["request"]
        decisions.append(Decision("memory", "memory", {"peak_gpu_bytes": peak_gpu, "peak_host_bytes": peak_host,
                                                       "memory_request_bytes": request},
                                  (f"peak {_gib(peak_gpu)} on the GPU, " if peak_gpu is not None else "")
                                  + f"{_gib(peak_host)} on the host; a job should ask for {request // 2**30} GiB "
                                    f"({memory['why']})", memory_why))

    # What was measured, kept for a later plan to reuse; or what this one reused
    if reusing is not None:
        device_name, free_bytes = reusing["device"], reusing["free_bytes"]
        reused = "the time of a step, the memory" + (" and a milestone's time" if milestone_each is not None else "")
        decisions.append(_reused_decision(
            reusing, reused_from, reused,
            "nothing they depend on changed (the model, the method, the attention, the LoRA rank and targets, the "
            f"head and the candidates per pass), and the data's longest question reads {max(sizes):,} tokens, at "
            f"most the {steps['largest_tokens']:,} measured"))
        measurements = {**reusing, "from": reusing.get("from") or reused_from or None, "milestone": milestone}
    else:
        device_name = torch.cuda.get_device_name(device) if cuda else "the CPU"
        free_bytes = torch.cuda.mem_get_info(device)[0] if cuda else None
        measurements = None
        if timing is not None and first is None:
            measurements = {"key": measured_key(job), "device": device_name, "free_bytes": free_bytes,
                            "facts": {"decisions": [d.to_dict() for d in decisions if d.name in MODEL_DECISIONS],
                                      "targets": target_count, "cuda": cuda},
                            "steps": steps, "memory": memory, "milestone": milestone}
    seconds = estimated
    if first is not None:
        decisions = [Decision(**first[d.name]) if d.name in RESUMED and d.name in first else d for d in decisions]
        free_bytes, seconds_per_step, seconds = (job._first["free_bytes"], job._first["seconds_per_step"],
                                                 job._first["estimated_seconds"])
        peak_gpu, peak_host, request = (job._first["peak_gpu_bytes"], job._first["peak_host_bytes"],
                                        job._first["memory_request_bytes"])
    plan = Plan(model=model_name, task=job.task, method=job.method, device=device_name, free_bytes=free_bytes,
                decisions=_ordered(decisions), seconds_per_step=seconds_per_step, estimated_seconds=seconds,
                peak_gpu_bytes=peak_gpu, peak_host_bytes=peak_host, memory_request_bytes=request,
                measurements=measurements)
    prepared = _Prepared(chooser, tokenizer, trainer, plan, output, milestones=milestone_at,
                         milestone_rows=milestone_rows)
    prepared.choice = {"config": config, "earlier": start["milestones"] if start is not None else [],
                       "answering": answering}
    return prepared
