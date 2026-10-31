# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""TrainingJob: say what to train, and meepsquared decides how.

    import meepsquared as m2

    job = m2.TrainingJob(model="meta-llama/Llama-3.1-8B-Instruct", data="HuggingFaceH4/no_robots", steps=1000)
    print(job.plan())     # what meepsquared will do on this machine, and why; nothing trained yet
    result = job.train()  # prints the plan, then trains with TRL
    result.save("out")    # the adapter; merged=True for a 16-bit model

A job names the model, the data, the task ("sft" or "grpo") and the method
("lora", "qlora" or "fp8"). Everything else is decided when the job is
planned: the model is loaded with MeepLanguageModel.from_pretrained, LoRA is
added with get_peft_model, the memory is measured on the real batch shape, and
TRL's trainer is built as train() runs it. A Recipe sets the training choices
(rank, learning rate, batch, schedule), Overrides the hardware ones
(checkpointed layers, micro-batch, kernels); each one set is reported in the
plan as the user's, beside what auto mode would have chosen.

A job is also plain data: TrainingJob.from_dict, to_dict, from_yaml, to_yaml,
and the command `meepsquared train job.yaml` (see meepsquared.cli).
"""

import dataclasses
import importlib
import importlib.util
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

TASKS = ("sft", "grpo")
METHODS = ("lora", "qlora", "fp8")
METHOD_NAMES = {"lora": "LoRA", "qlora": "QLoRA", "fp8": "LoRA on FP8 weights"}
SCHEDULES = ("constant", "cosine")
# The file types a local data file can have, and the datasets builder that reads each
DATA_SUFFIXES = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}

# The defaults, each stated in the plan with its source.
# LoRA: rank 16 with alpha = rank (scaling 1), on the attention and MLP projections
RANK = 16
# A model "up to 13B" has fewer parameters than this (Llama 2 13B has 13.0 billion)
UP_TO_13B = 14e9
# A model "from 30B" has at least this many
FROM_30B = 30e9
# The learning rates hold at REFERENCE_TOKENS per update; with a token budget, another
# budget scales them (scaled_learning_rate)
LEARNING_RATE = 1e-4
LEARNING_RATE_FROM_30B = 5e-5
# The batch the stated learning rates and the benchmarks were measured at: 8 conversations of up to 1024 tokens
REFERENCE_TOKENS = 8 * 1024
# A token budget is off by default until the measurement round decides on it
# (thinkube-release PLAN 0b); Recipe(tokens_per_update=AUTO) measures one.
AUTO = "auto"
# The most tokens per update a measured budget takes. LoRA learns less from large
# batches (Thinking Machines, LoRA Without Regret, 2025), so the budget is capped
# even where more fits; the ceiling is to be set by the time-to-quality study
# (thinkube-release PLAN W2). Until then: 32 x 1024 tokens.
TOKENS_PER_UPDATE_CEILING = 32 * 1024
# Examples per update of the weights, auto mode's batch: the smallest that keeps the GPU busy
EFFECTIVE_BATCH = 8
EFFECTIVE_BATCH_ABOVE_13B = 16
OPTIMIZER = "adamw_8bit"
# Tokens per example: TRL's SFTConfig default
MAX_LENGTH = 1024
# The folder of the output folder a plan preprocesses the data into when none is given
PREPROCESSED = "preprocessed"
# The schedule default will be set by the measurement round (thinkube-release
# PLAN F5: a constant rate against warm-up and cosine decay, on the same job)
SCHEDULE = "constant"
WARMUP_STEPS = 100
WARMUP_SHARE = 0.05
# Cosine decay ends at this share of the learning rate
COSINE_FLOOR = 0.1
SPLIT = "train"
# The Recipe fields that set GRPO's generation
GRPO_RECIPE = ("generations", "max_completion_length", "temperature", "top_p", "top_k", "chat_template_kwargs")
# meepsquared's vLLM support (meepsquared._vllm) needs this TRL
VLLM_TRL = (1, 16)
# The plan's time per step: optimizer steps timed on the largest micro-batch and on the median one, after one to
# warm up
TIMED_STEPS = 3
# Evaluation: the split of a dataset id given as eval_data, the largest share of the training data held
# out, and auto mode's evaluations over the planned steps
EVAL_SPLIT = "test"
EVAL_SHARE_MAX = 0.5
EVALUATIONS = 10

LR_SOURCE = ("at alpha = rank, published rates for 7B to 13B models are near 1e-4 (QLoRA's 2e-4 at scaling 0.25; "
             "Thinking Machines, LoRA Without Regret, 2025)")
LR_SOURCE_30B = "QLoRA halves the rate from 33B parameters (Dettmers et al., 2023)"
BATCH_SOURCE = ("the smallest that keeps the GPU busy: LoRA learns less from large batches (Thinking Machines, "
                "2025); QLoRA used 16 up to 13B and 32 to 64 above (Dettmers et al., 2023, table 9)")
TOKENS_SOURCE = (f"Recipe(tokens_per_update='auto'): the largest that fits one forward pass, up to "
                 f"{TOKENS_PER_UPDATE_CEILING:,} tokens: LoRA learns less from large batches (Thinking Machines, "
                 "2025); the ceiling is to be set by the time-to-quality study")
BY_EXAMPLES = "batched by examples; Recipe(tokens_per_update='auto') batches by tokens instead"
RATE_RULE = "Adam's rule: the rate grows with the square root of the batch"


def size_defaults(parameters: int) -> dict:
    """The learning rate and the effective batch auto mode takes for a model of this many parameters, with why.

    With a token budget, the learning rate holds at REFERENCE_TOKENS per update
    and scaled_learning_rate scales it; the effective batch is for a batch
    counted in examples, auto mode's.
    """
    size = f"{parameters / 1e9:.1f}B parameters"
    if parameters >= FROM_30B:
        rate = (LEARNING_RATE_FROM_30B, f"{size}, from 30B: {LR_SOURCE_30B}")
    else:
        rate = (LEARNING_RATE, f"{size}: {LR_SOURCE}")
    if parameters < UP_TO_13B:
        batch = (EFFECTIVE_BATCH, f"{size}, up to 13B: {BATCH_SOURCE}")
    else:
        batch = (EFFECTIVE_BATCH_ABOVE_13B, f"{size}, above 13B: {BATCH_SOURCE}")
    return {"learning_rate": rate, "effective_batch": batch}


def scaled_learning_rate(rate: float, tokens: int) -> float:
    """The rate stated for REFERENCE_TOKENS per update, scaled to `tokens` per update by RATE_RULE; 3 digits."""
    return float(f"{rate * math.sqrt(tokens / REFERENCE_TOKENS):.3g}")


def warmup_steps(planned: int) -> int:
    """WARMUP_STEPS, or WARMUP_SHARE of the planned steps when that is fewer."""
    return min(WARMUP_STEPS, int(WARMUP_SHARE * planned))


def batches_per_epoch(rows: int, micro: int, world: int, task: str, generations: int) -> int:
    """The micro-batches each process takes in one pass over the data, with `micro` examples in each.

    For GRPO a row is a prompt, and each prompt becomes `generations`
    completions of the batch.
    """
    examples = rows * generations if task == "grpo" else rows
    return math.ceil(examples / (micro * world))


def planned_steps(steps: Optional[int], epochs: Optional[float], batches: int, accumulation: int) -> int:
    """The updates the job makes: `steps`, or the epochs over `batches` micro-batches as the Trainer counts them."""
    if steps is not None:
        return steps
    per_epoch = max(1, math.ceil(batches / accumulation))
    return math.ceil((epochs if epochs is not None else 1) * per_epoch)


def smaller_micro_batch(per_process: int, micro: int) -> int:
    """The micro-batch after `micro`: half of it, or the next smaller divisor of per_process."""
    smaller = max(1, micro // 2)
    while per_process % smaller:
        smaller -= 1
    return smaller


PROMPTS = "prompts"
CONVERSATIONS = "conversations (messages)"
PAIRS = "prompt and completion pairs"
PLAIN_TEXT = "plain text"


def data_format(columns, task: str) -> tuple[str, list]:
    """What kind of examples the data holds, as TRL takes them, and the columns that hold them.

    Refuses data TRL cannot train on. For SFT only those columns are kept:
    TRL chooses how to tokenize an example by the columns it finds, and a
    dataset with both "messages" and a "prompt" column (no_robots) would be
    read as prompt and completion pairs.
    """
    columns = list(columns)
    if task == "grpo":
        if "prompt" in columns:
            return PROMPTS, columns
        raise ValueError(f"data has the columns {columns}; GRPO takes a 'prompt' column")
    if "messages" in columns:
        return CONVERSATIONS, ["messages"]
    if "prompt" in columns and "completion" in columns:
        return PAIRS, ["prompt", "completion"]
    if "text" in columns:
        return PLAIN_TEXT, ["text"]
    raise ValueError(f"data has the columns {columns}; SFT takes conversations ('messages'), "
                     "prompt and completion pairs ('prompt', 'completion'), or plain text ('text')")


def _gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB"


def _say(text: str) -> None:
    if os.environ.get("MEEPSQUARED_QUIET") != "1":
        print(text, flush=True)


def _check_positive(name: str, value, kind=int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, kind) or value <= 0:
        raise ValueError(f"{name} must be a positive {kind.__name__ if kind is int else 'number'}, not {value!r}")


def _fields_from_dict(cls, data: dict, where: str) -> dict:
    if not isinstance(data, dict):
        raise TypeError(f"{where} must be a mapping, not {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{where}: unknown fields {unknown}; the fields are {sorted(names)}")
    return dict(data)


def _set_fields(obj) -> dict:
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)
            if f.init and getattr(obj, f.name) is not None}


@dataclass
class Recipe:
    """The training choices. A field left as None is chosen by auto mode, and the plan says how.

    rank: LoRA's rank (16). alpha: LoRA's alpha (= rank, a scaling of 1).
    dropout: LoRA's dropout (0, get_peft_model's default). targets: the
    modules LoRA adapts (the attention and MLP projections).
    learning_rate: 1e-4 up to 13B parameters, 5e-5 from 30B; with a token
    budget these hold at 8 x 1024 tokens per update, and another budget
    scales them with the square root of the ratio.
    schedule: "constant" (after a warm-up) or "cosine" (decay to a tenth over
    the planned steps). warmup_steps: 100, or 5% of the planned steps when
    fewer. effective_batch: examples per update (8 up to 13B parameters and
    16 above; GRPO's batch, counted in completions). tokens_per_update: SFT
    only, off unless set: the batch as a token budget instead, each update
    taking as many conversations as fit it; "auto" for the largest budget
    that fits one forward pass, up to 32 x 1024 tokens, or a number of
    tokens. Give effective_batch or tokens_per_update, not both.
    optimizer: a Transformers optimizer name ("adamw_8bit").
    max_length: tokens per example (1024). max_grad_norm, weight_decay:
    Transformers' gradient clipping and weight decay (its defaults).
    experts: LoRA on every routed expert of a mixture of experts too (off),
    computed on the tokens routed to each expert (see meepsquared._moe).
    answers_only: the loss on the answers alone (on for conversations and
    prompt and completion pairs; plain text has no answers).
    max_cut_share: SFT only, the largest share of examples whose answer
    tokens max_length may cut; more is refused (unset: reported only).
    source_weights: SFT only, a weight for the loss of the rows of each
    origin kind of the data's _origin column, such as {"synthetic": 0.5}; a
    kind not given weighs 1 (see meepsquared._data_checks).
    GRPO only, each TRL's default unless set: generations (num_generations),
    max_completion_length (tokens per completion), temperature, top_p and
    top_k (the sampling), chat_template_kwargs (arguments of the chat
    template, such as {"enable_thinking": False}).
    """

    rank: Optional[int] = None
    alpha: Optional[int] = None
    dropout: Optional[float] = None
    targets: Optional[list] = None
    learning_rate: Optional[float] = None
    schedule: Optional[str] = None
    warmup_steps: Optional[int] = None
    tokens_per_update: Any = None
    effective_batch: Optional[int] = None
    optimizer: Optional[str] = None
    max_length: Optional[int] = None
    max_grad_norm: Optional[float] = None
    weight_decay: Optional[float] = None
    experts: Optional[bool] = None
    answers_only: Optional[bool] = None
    generations: Optional[int] = None
    max_completion_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    chat_template_kwargs: Optional[dict] = None
    max_cut_share: Optional[float] = None
    source_weights: Optional[dict] = None

    def __post_init__(self):
        for name in ("rank", "alpha", "effective_batch", "max_length", "generations", "max_completion_length"):
            _check_positive(name, getattr(self, name))
        if self.tokens_per_update != AUTO and (isinstance(self.tokens_per_update, bool) or not (
                self.tokens_per_update is None
                or (isinstance(self.tokens_per_update, int) and self.tokens_per_update > 0))):
            raise ValueError(f"tokens_per_update must be 'auto' or a positive number of tokens, not "
                             f"{self.tokens_per_update!r}")
        if self.tokens_per_update is not None and self.effective_batch is not None:
            raise ValueError(f"tokens_per_update={self.tokens_per_update} and effective_batch={self.effective_batch} "
                             "both state the batch per update; give one: tokens, or examples")
        for name in ("learning_rate", "max_grad_norm", "temperature"):
            _check_positive(name, getattr(self, name), (int, float))
        for name in ("experts", "answers_only"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, not {value!r}")
        if self.weight_decay is not None and (isinstance(self.weight_decay, bool)
                                              or not isinstance(self.weight_decay, (int, float))
                                              or self.weight_decay < 0):
            raise ValueError(f"weight_decay must be a number of 0 or more, not {self.weight_decay!r}")
        if self.top_p is not None and (isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float))
                                       or not 0 < self.top_p <= 1):
            raise ValueError(f"top_p must be above 0 and at most 1, not {self.top_p!r}")
        if self.top_k is not None and (isinstance(self.top_k, bool) or not isinstance(self.top_k, int)
                                       or self.top_k < 0):
            raise ValueError(f"top_k must be a whole number of tokens, 0 for no limit, not {self.top_k!r}")
        if self.chat_template_kwargs is not None and (not isinstance(self.chat_template_kwargs, dict) or not all(
                isinstance(key, str) for key in self.chat_template_kwargs)):
            raise ValueError(f"chat_template_kwargs must be a mapping of argument names to values, not "
                             f"{self.chat_template_kwargs!r}")
        if self.experts and self.dropout:
            raise ValueError("experts=True takes no LoRA dropout: meepsquared's routed LoRA computes the expert stacks "
                             "without it; leave dropout unset or 0")
        if self.warmup_steps is not None and (isinstance(self.warmup_steps, bool)
                                              or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0):
            raise ValueError(f"warmup_steps must be a whole number of steps, not {self.warmup_steps!r}")
        if self.dropout is not None and (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                                         or not 0 <= self.dropout < 1):
            raise ValueError(f"dropout must be at least 0 and below 1, not {self.dropout!r}")
        if self.schedule is not None and self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, not {self.schedule!r}")
        if self.targets is not None and (not isinstance(self.targets, list) or not self.targets):
            raise ValueError(f"targets must be a non-empty list of module names, not {self.targets!r}")
        if self.max_cut_share is not None and (isinstance(self.max_cut_share, bool)
                                               or not isinstance(self.max_cut_share, (int, float))
                                               or not 0 <= self.max_cut_share <= 1):
            raise ValueError(f"max_cut_share must be a share from 0 to 1, not {self.max_cut_share!r}")
        if self.source_weights is not None:
            from meepsquared._data_checks import KINDS

            if not isinstance(self.source_weights, dict) or not self.source_weights or any(
                    kind not in KINDS or isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or weight <= 0 for kind, weight in self.source_weights.items()):
                raise ValueError(f"source_weights must map origin kinds ({', '.join(KINDS)}) to weights above 0, "
                                 f"not {self.source_weights!r}")

    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        return cls(**_fields_from_dict(cls, data, "recipe"))

    def to_dict(self) -> dict:
        return _set_fields(self)


@dataclass
class Overrides:
    """The hardware decisions, set by hand. A field left as None is auto mode's.

    attention: from_pretrained's attn_implementation (FlashAttention 2 where
    it runs the model, else PyTorch's memory-efficient kernel with packed
    sequences, "meepsquared_varlen"; see models.choose_attention).
    liger: Liger's kernels and fused loss (on). padding_free: SFT batches laid
    end to end without padding (on where the attention keeps the examples
    apart). fused_lora: the fused LoRA kernels (on where the MLP has one).
    compile_layers: each decoder layer compiled (where it pays).
    checkpointed_layers: how many layers recompute their whole forward in the
    backward pass; selective_layers: how many recompute only their cheap
    operations, keeping the outputs of their products (see
    meepsquared._checkpointing; the cheapest layout the measured memory allows).
    With checkpointed_layers given alone, the other layers keep everything;
    with selective_layers given alone, as few layers as the memory needs
    recompute whole. micro_batch: examples per forward pass (the largest that
    fits, see TrainingJob.plan); with a token budget, rows of max_length
    tokens per forward pass, which a forward pass fills with as many
    conversations as fit.
    vllm: GRPO only, the completions generated by vLLM inside the training
    process (TRL's colocate mode) instead of Transformers' generate (off).
    milestone_engine: "vllm" answers the milestones' evaluation with vLLM
    inside the training process, holding the routed expert stacks once for
    the trainer and for vLLM (see meepsquared._shared_vllm); None, the
    default, with Transformers' generate. qwen3_5_moe models, LoRA or FP8.
    """

    attention: Optional[str] = None
    liger: Optional[bool] = None
    padding_free: Optional[bool] = None
    fused_lora: Optional[bool] = None
    compile_layers: Optional[bool] = None
    checkpointed_layers: Optional[int] = None
    selective_layers: Optional[int] = None
    micro_batch: Optional[int] = None
    vllm: Optional[bool] = None
    milestone_engine: Optional[str] = None

    def __post_init__(self):
        if self.milestone_engine not in (None, "vllm"):
            raise ValueError(f"milestone_engine is 'vllm' or not given, not {self.milestone_engine!r}")
        for name in ("liger", "padding_free", "fused_lora", "compile_layers", "vllm"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, not {value!r}")
        _check_positive("micro_batch", self.micro_batch)
        for name in ("checkpointed_layers", "selective_layers"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a whole number of layers, not {value!r}")

    @classmethod
    def from_dict(cls, data: dict) -> "Overrides":
        return cls(**_fields_from_dict(cls, data, "overrides"))

    def to_dict(self) -> dict:
        return _set_fields(self)


@dataclass
class Decision:
    """One line of the plan: what was decided, why, and by whom.

    by is "auto" or "user"; for the user's, auto_value and auto_text say
    what auto mode would have chosen, and why.
    """

    name: str
    label: str
    value: Any
    text: str
    reason: str = ""
    by: str = "auto"
    auto_value: Any = None
    auto_text: str = ""

    def line(self) -> str:
        if self.by == "user":
            return f"  {self.label:<12} {self.text} (yours; auto: {self.auto_text})"
        return f"  {self.label:<12} {self.text}" + (f" ({self.reason})" if self.reason else "")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _decide(name: str, label: str, given, auto, describe: Callable[[Any], str], reason: str,
            user_reason: str = "") -> Decision:
    """A decision: the user's value when given, else auto's; the user's is reported beside auto's."""
    if given is None:
        return Decision(name, label, auto, describe(auto), reason)
    auto_text = describe(auto) + (f", {reason}" if reason else "")
    return Decision(name, label, given, describe(given), user_reason, "user", auto, auto_text)


@dataclass
class Plan:
    """What a job will do on this machine, and why: str() is the readable plan, to_dict() the same as data."""

    model: str
    task: str
    method: str
    device: str
    free_bytes: Optional[int]
    decisions: list
    seconds_per_step: Optional[float] = None
    estimated_seconds: Optional[float] = None
    peak_gpu_bytes: Optional[int] = None
    peak_host_bytes: Optional[int] = None
    memory_request_bytes: Optional[int] = None

    def __getitem__(self, name: str) -> Decision:
        for decision in self.decisions:
            if decision.name == name:
                return decision
        raise KeyError(f"the plan has no decision {name!r}; it has {[d.name for d in self.decisions]}")

    def __str__(self) -> str:
        where = f"on {self.device}" + (f" ({self.free_bytes / 2**30:.0f} GiB free)" if self.free_bytes is not None
                                       else "")
        task = "" if self.task == "sft" else f" {self.task.upper()}"
        header = f"meepsquared plan for {self.model}, {METHOD_NAMES[self.method]}{task}, {where}"
        return "\n".join([header] + [decision.line() for decision in self.decisions])

    def to_dict(self) -> dict:
        return {"model": self.model, "task": self.task, "method": self.method, "device": self.device,
                "free_bytes": self.free_bytes, "decisions": [d.to_dict() for d in self.decisions],
                "seconds_per_step": self.seconds_per_step, "estimated_seconds": self.estimated_seconds,
                "peak_gpu_bytes": self.peak_gpu_bytes, "peak_host_bytes": self.peak_host_bytes,
                "memory_request_bytes": self.memory_request_bytes}


@dataclass
class Result:
    """What a training did. save() writes the trained adapter, or the model with it merged in.

    peak_memory_bytes is what PyTorch allocated at most on the GPU during
    training; None when the model trained on the CPU, where it is not measured.
    final_loss is the last loss the trainer logged. paused: the training
    stopped when asked, before its last step, and saved a checkpoint to resume
    from; checkpoint is the latest checkpoint in the output folder, None when
    it holds none. evaluations: each evaluation of the training, a resumed
    one's earlier sessions' too, as {"step": ..., "eval_loss": ...}.
    milestones: the record of each milestone made, earlier sessions' too:
    its step, folder, seconds and, with an evaluation, its scores by group.
    """

    steps: int
    seconds: float
    final_loss: Optional[float]
    peak_memory_bytes: Optional[int]
    output: str
    paused: bool = False
    checkpoint: Optional[str] = None
    evaluations: list = field(default_factory=list)
    milestones: list = field(default_factory=list)
    _model: Any = field(default=None, repr=False, compare=False)
    _tokenizer: Any = field(default=None, repr=False, compare=False)

    def save(self, path, merged: bool = False) -> None:
        """The adapter alone, or with merged=True the base model with the adapter merged into 16-bit weights."""
        from meepsquared import _save

        _save.save(self._model, path, tokenizer=self._tokenizer, save_method="merged_16bit" if merged else "lora")

    def to_dict(self) -> dict:
        return {"steps": self.steps, "seconds": self.seconds, "final_loss": self.final_loss,
                "peak_memory_bytes": self.peak_memory_bytes, "output": self.output, "paused": self.paused,
                "checkpoint": self.checkpoint, "evaluations": self.evaluations, "milestones": self.milestones}


def load_yaml(text: str):
    """YAML as PyYAML's safe loader reads it, with numbers such as 1e-4 read as numbers.

    YAML 1.1, which PyYAML follows, reads a number with an exponent and no
    decimal point as text; a learning rate is written that way.
    """
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_implicit_resolver("tag:yaml.org,2002:float",
                                 re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
                                 list("-+0123456789"))
    return yaml.load(text, Loader=Loader)  # noqa: S506 (a SafeLoader)


def load_reward(spec, field: str = "rewards"):
    """A function given by a field (a reward, the milestone scorer): a callable as it is, or "module:function" /
    "path/to/file.py:function" imported."""
    if callable(spec):
        return spec
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"{field}: {spec!r} is neither a function nor 'module:function' or 'file.py:function'")
    where, _, name = spec.rpartition(":")
    if where.endswith(".py"):
        path = Path(where)
        if not path.is_file():
            raise FileNotFoundError(f"{field}: {spec!r}: no file {path}")
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    if not hasattr(module, name):
        raise AttributeError(f"{field}: {spec!r}: {where} has no {name!r}")
    return getattr(module, name)


def reward_name(reward, field: str = "rewards") -> str:
    """A function of a field as data: 'module:function'; refused for one that cannot be imported by name."""
    if isinstance(reward, str):
        return reward
    module, name = getattr(reward, "__module__", None), getattr(reward, "__qualname__", "")
    if not module or module == "__main__" or "<" in name or "." in name:
        raise ValueError(f"{field}: {reward!r} cannot be written as data; give it as 'module:function' "
                         "or 'file.py:function'")
    return f"{module}:{name}"


@dataclass
class _Prepared:
    model: Any
    tokenizer: Any
    trainer: Any
    plan: Plan
    output: str
    trained: bool = False
    # The milestones' steps, the rows their evaluation answers and its scorer (meepsquared._milestones)
    milestones: Optional[list] = None
    milestone_rows: Optional[list] = None
    milestone_scorer: Any = None


@dataclass
class _Data:
    """SFT's data as the trainer trains it (see TrainingJob._data)."""

    train: Any
    eval: Any
    loading: Any
    held: int
    kinds: Optional[list]
    weights: Optional[list]
    decision: Decision
    report: dict


@dataclass
class TrainingJob:
    """A training job: what to train, on what, by which method; meepsquared decides how.

    model: a model id, a local path, or a model loaded with
        MeepLanguageModel.from_pretrained (then tokenizer is required too,
        and the method must match how it was loaded).
    data: a Hugging Face dataset id, a local .jsonl, .json or .parquet file,
        or a datasets.Dataset, holding conversations ("messages"), prompt and
        completion pairs, or plain text ("text"); for GRPO, prompts.
    task: "sft" or "grpo". method: "lora", "qlora" (4-bit weights) or "fp8".
    steps or epochs: how long; one epoch when neither is given.
    rewards: GRPO's reward functions, as TRL takes them, or their names as
        'module:function' or 'file.py:function'.
    output: the folder TRL writes checkpoints to; named after the model and
        the task when not given.
    split: the split of a Hugging Face dataset ("train").
    report_to: where the trainer reports its metrics, as Transformers'
        report_to takes it ("mlflow", or a list); Transformers' default
        ("none") when not given.
    eval_data: SFT's evaluation data, in the forms data takes, or a
        datasets.DatasetDict; eval_split: its split, EVAL_SPLIT by default
        for a dataset id, required for a DatasetDict, refused for a file or a
        datasets.Dataset. eval_share instead: the share of the training data
        held out for evaluation (above 0, at most EVAL_SHARE_MAX), chosen with
        the trainer's seed and never trained on. eval_every: the optimizer
        steps between evaluations (about EVALUATIONS over the planned steps).
        The evaluation loss is counted as the training loss is.
    timed_steps: the optimizer steps the plan times for its time per step
        (TIMED_STEPS on a GPU; 0: none). Set by hand, they are timed on the
        CPU too.
    milestones: the share of the planned steps between milestones (above 0,
        at most 1): at each, and at the last step, the adapter is saved apart
        in <output>/milestones/step-<N>. milestone_eval: rows the model answers
        at each milestone, a .jsonl file, a list of mappings or a
        datasets.Dataset, each {"messages": [...], "reference": "...", and
        optionally "group" and "source"}; milestone_scorer scores the answers
        of each group, a function (hypotheses, references) -> {metric: number}
        or its name as 'module:function' or 'file.py:function'. SFT only, on
        one process. See meepsquared._milestones.
    preprocessed: SFT only: the folder of the data preprocessed by
        job.preprocess(cache_dir) (meepsquared._preprocess), read memory-mapped
        after the check that it was made from this job's inputs. Without it,
        the plan preprocesses the data first, into <output>/preprocessed.
    recipe, overrides: see Recipe and Overrides.
    """

    model: Any
    data: Any = None
    task: str = "sft"
    method: str = "lora"
    steps: Optional[int] = None
    epochs: Optional[float] = None
    rewards: Optional[list] = None
    output: Optional[str] = None
    split: str = SPLIT
    report_to: Any = None
    eval_data: Any = None
    eval_split: Optional[str] = None
    eval_share: Optional[float] = None
    eval_every: Optional[int] = None
    timed_steps: Optional[int] = None
    milestones: Optional[float] = None
    milestone_eval: Any = None
    milestone_scorer: Any = None
    tokenizer: Any = None
    preprocessed: Any = None
    recipe: Recipe = field(default_factory=Recipe)
    overrides: Overrides = field(default_factory=Overrides)
    _prepared: Optional[_Prepared] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.model is None or (isinstance(self.model, str) and not self.model):
            raise ValueError("model: required, a model id, a local path, or a model loaded with meepsquared")
        if not isinstance(self.recipe, Recipe):
            raise TypeError(f"recipe must be an meepsquared Recipe, not {type(self.recipe).__name__}")
        if not isinstance(self.overrides, Overrides):
            raise TypeError(f"overrides must be an meepsquared Overrides, not {type(self.overrides).__name__}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, not {self.task!r}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, not {self.method!r}")
        if self.steps is not None and self.epochs is not None:
            raise ValueError("give steps or epochs, not both")
        _check_positive("steps", self.steps)
        _check_positive("epochs", self.epochs, (int, float))
        if self.timed_steps is not None and (isinstance(self.timed_steps, bool) or not isinstance(self.timed_steps, int)
                                             or self.timed_steps < 0):
            raise ValueError(f"timed_steps must be a whole number of steps, 0 for none, not {self.timed_steps!r}")
        if self.eval_data is not None and self.eval_share is not None:
            raise ValueError("give eval_data or eval_share, not both")
        if self.eval_split is not None and self.eval_data is None:
            raise ValueError("eval_split is the split of eval_data, and no eval_data is given")
        if self.eval_share is not None and (isinstance(self.eval_share, bool)
                                            or not isinstance(self.eval_share, (int, float))
                                            or not 0 < self.eval_share <= EVAL_SHARE_MAX):
            raise ValueError(f"eval_share must be a share of the training data above 0 and at most "
                             f"{EVAL_SHARE_MAX:g}, not {self.eval_share!r}")
        _check_positive("eval_every", self.eval_every)
        if self.eval_every is not None and self.eval_data is None and self.eval_share is None:
            raise ValueError("eval_every: no evaluation data; give eval_data or eval_share")
        if self.task != "sft" and (self.eval_data is not None or self.eval_share is not None):
            raise ValueError("eval_data and eval_share are SFT settings: GRPO's reward is its measure")
        if self.task != "sft" and self.preprocessed is not None:
            raise ValueError("preprocessed is an SFT setting: GRPO's prompts are not tokenized ahead")
        if self.task != "sft" and self.timed_steps:
            raise ValueError("timed_steps: the plan times SFT's steps; GRPO's steps generate their completions")
        if self.task == "grpo" and not self.rewards:
            raise ValueError("rewards: required for GRPO, a list of reward functions")
        if self.task != "grpo" and self.rewards:
            raise ValueError(f"rewards are for GRPO, and the task is {self.task!r}")
        if self.rewards is not None and not isinstance(self.rewards, (list, tuple)):
            raise ValueError("rewards must be a list of reward functions")
        if self.task != "sft" and self.overrides.padding_free is not None:
            raise ValueError("overrides.padding_free is an SFT setting; GRPO's batches are TRL's")
        if self.task != "grpo":
            for name in GRPO_RECIPE:
                if getattr(self.recipe, name) is not None:
                    raise ValueError(f"recipe.{name} is a GRPO setting")
            if self.overrides.vllm is not None:
                raise ValueError("overrides.vllm is a GRPO setting: vLLM generates GRPO's completions")
        if self.task != "sft" and (self.recipe.max_cut_share is not None or self.recipe.source_weights is not None):
            raise ValueError("recipe.max_cut_share and recipe.source_weights are SFT settings")
        if self.task != "sft" and self.recipe.answers_only is not None:
            raise ValueError("recipe.answers_only is an SFT setting; GRPO's loss is on the completions already")
        if self.task != "sft" and self.recipe.tokens_per_update is not None:
            raise ValueError("recipe.tokens_per_update is an SFT setting; GRPO's batch counts completions, "
                             "recipe.effective_batch")
        if self.milestones is not None and (isinstance(self.milestones, bool)
                                            or not isinstance(self.milestones, (int, float))
                                            or not 0 < self.milestones <= 1):
            raise ValueError(f"milestones must be the share of the planned steps between milestones, above 0 and at "
                             f"most 1, not {self.milestones!r}")
        if self.milestone_eval is not None and self.milestones is None:
            raise ValueError("milestone_eval: no milestones; give milestones, the share of the steps between them")
        if (self.milestone_eval is None) != (self.milestone_scorer is None):
            raise ValueError("milestone_eval and milestone_scorer go together: the rows to answer and what scores them")
        if self.milestones is not None and self.task != "sft":
            raise ValueError("milestones are an SFT setting")
        if self.overrides.milestone_engine is not None:
            if self.milestone_eval is None:
                raise ValueError("overrides.milestone_engine answers the milestones' evaluation; give milestones and "
                                 "milestone_eval")
            if self.method == "qlora":
                raise ValueError("overrides.milestone_engine='vllm' shares 16-bit or FP8 expert stacks; method "
                                 "'qlora' holds them in 4 bits")
        if self.report_to is not None and not (
                isinstance(self.report_to, str)
                or (isinstance(self.report_to, (list, tuple)) and all(isinstance(r, str) for r in self.report_to))):
            raise ValueError(f"report_to must be a name such as 'mlflow', or a list of names, not {self.report_to!r}")

    # The job as data

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingJob":
        fields = _fields_from_dict(cls, data, "job")
        if "recipe" in fields:
            fields["recipe"] = Recipe.from_dict(fields["recipe"])
        if "overrides" in fields:
            fields["overrides"] = Overrides.from_dict(fields["overrides"])
        return cls(**fields)

    def to_dict(self) -> dict:
        if not isinstance(self.model, str):
            raise ValueError("model is a loaded model; only a model id or a path can be written as data")
        if self.data is not None and not isinstance(self.data, (str, os.PathLike)):
            raise ValueError("data is a datasets.Dataset; only a dataset id or a file can be written as data")
        if self.tokenizer is not None:
            raise ValueError("tokenizer is a loaded tokenizer, which cannot be written as data")
        if self.eval_data is not None and not isinstance(self.eval_data, (str, os.PathLike)):
            raise ValueError("eval_data is a loaded dataset; only a dataset id or a file can be written as data")
        if self.milestone_eval is not None and not isinstance(self.milestone_eval, (str, os.PathLike)):
            raise ValueError("milestone_eval is loaded rows; only a .jsonl file can be written as data")
        out = {"model": self.model, "data": str(self.data) if self.data is not None else None, "task": self.task,
               "method": self.method, "steps": self.steps, "epochs": self.epochs,
               "rewards": [reward_name(r) for r in self.rewards] if self.rewards else None,
               "output": str(self.output) if self.output is not None else None, "split": self.split,
               "report_to": list(self.report_to) if isinstance(self.report_to, tuple) else self.report_to,
               "eval_data": str(self.eval_data) if self.eval_data is not None else None,
               "eval_split": self.eval_split, "eval_share": self.eval_share, "eval_every": self.eval_every,
               "timed_steps": self.timed_steps, "milestones": self.milestones,
               "preprocessed": str(self.preprocessed) if self.preprocessed is not None else None,
               "milestone_eval": str(self.milestone_eval) if self.milestone_eval is not None else None,
               "milestone_scorer": (reward_name(self.milestone_scorer, "milestone_scorer")
                                    if self.milestone_scorer is not None else None),
               "recipe": self.recipe.to_dict() or None,
               "overrides": self.overrides.to_dict() or None}
        return {key: value for key, value in out.items() if value is not None}

    @classmethod
    def from_yaml(cls, path) -> "TrainingJob":
        with open(path) as f:
            data = load_yaml(f.read())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a job is a YAML mapping of fields, such as 'model:' and 'data:'")
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # Planning and training

    def plan(self) -> Plan:
        """Load the model, add LoRA, measure, and build TRL's trainer as train() runs it; returns the plan."""
        return self._prepare().plan

    def trainer(self):
        """TRL's SFTTrainer or GRPOTrainer, set up as train() would run it, not trained."""
        return self._prepare().trainer

    def train(self, resume: bool = False, stop_file=None, status_file=None, stop_on_sigterm: bool = False) -> Result:
        """Print the plan, then train. Returns the Result.

        resume: continue from the latest checkpoint in the output folder, with
        its optimizer, its scheduler and the next micro-batch laid out; the
        plan must make the same micro-batches and schedule as the first
        session's. Without it, a folder holding checkpoints is refused.
        stop_file: when this file exists at the end of an optimizer step, the
        training saves a checkpoint and stops, and the Result is paused;
        stop_on_sigterm does the same on a SIGTERM. status_file: a JSON file
        of the training's state, written as it goes (see meepsquared._sessions).
        """
        from meepsquared import _sessions

        status = _sessions.Status(status_file) if status_file is not None and _sessions.main_process() else None
        if status is not None:
            status.update()
        stop = (_sessions.Stop(stop_file, stop_on_sigterm, status) if stop_file is not None or stop_on_sigterm
                else None)
        try:
            return self._train(resume, stop, status)
        except Exception as error:
            if status is not None:
                status.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if stop is not None:
                stop.close()

    def _train(self, resume: bool, stop, status) -> Result:
        import torch

        from meepsquared import _batches, _milestones, _sessions, _unified_memory

        prepared = self._prepare()
        if prepared.trained:
            raise RuntimeError("this job has trained already; make a new TrainingJob to train again")
        trainer, output = prepared.trainer, prepared.output
        checkpoint, step = None, 0
        if resume:
            checkpoint = _sessions.to_resume(output)
            _sessions.check_same_plan(output, prepared.plan)
            step = _sessions.checkpoint_step(checkpoint)[0]
            if hasattr(trainer, "_meepsquared_layout"):
                _batches.resume_at(trainer, step)
        else:
            _sessions.refuse_earlier_checkpoints(output)
            _sessions.record_plan(output, prepared.plan)
        if stop is not None:
            trainer.add_callback(stop)
        report = None
        if status is not None:
            report = _sessions.Report(status)
            trainer.add_callback(report)
            status.update(state="training", plan=prepared.plan.to_dict(), step=step,
                          total_steps=prepared.plan["length"].value)
        milestones = None
        if prepared.milestones is not None:
            # After Report: the time of a milestone is taken out of the steps'
            milestones = _milestones.Milestones(trainer, output, prepared.milestones, prepared.milestone_rows,
                                                prepared.milestone_scorer, status=status, report=report,
                                                engine=getattr(prepared.model, "_meepsquared", {}).get("engine"))
            trainer.add_callback(milestones)
        if os.environ.get("MEEPSQUARED_QUIET") != "1":
            print(prepared.plan, flush=True)
        device = next(prepared.model.parameters()).device
        on_gpu = device.type == "cuda"
        if on_gpu:
            torch.cuda.reset_peak_memory_stats(device)
        if on_gpu and _unified_memory.shares_system_memory(device.index or 0):
            _unified_memory.recap_allocator(device.index or 0)
        start = time.perf_counter()
        trainer.train(resume_from_checkpoint=checkpoint)
        seconds = time.perf_counter() - start
        prepared.trained = True
        state = trainer.state
        paused = stop is not None and stop.stopped
        latest = _sessions.latest_checkpoint(output)
        losses = [entry["loss"] for entry in state.log_history if "loss" in entry]
        evaluations = [{"step": entry["step"], "eval_loss": entry["eval_loss"]} for entry in state.log_history
                       if "eval_loss" in entry]
        if status is not None:
            status.update(state="paused" if paused else "completed", step=state.global_step, checkpoint=latest)
        return Result(steps=state.global_step, seconds=seconds, final_loss=losses[-1] if losses else None,
                      peak_memory_bytes=torch.cuda.max_memory_allocated(device) if on_gpu else None,
                      output=output, paused=paused, checkpoint=latest, evaluations=evaluations,
                      milestones=milestones.records() if milestones is not None else [],
                      _model=prepared.model,
                      _tokenizer=prepared.tokenizer)

    # What plan() does

    def _dataset(self, cache_dir: Optional[str] = None):
        """The data as a datasets.Dataset, and a description of where it came from; a file is read into
        `cache_dir` when given."""
        from datasets import Dataset, load_dataset

        if self.data is None:
            raise ValueError("data: required, a dataset id, a .jsonl, .json or .parquet file, or a datasets.Dataset")
        if isinstance(self.data, Dataset):
            return self.data, "a datasets.Dataset"
        if not isinstance(self.data, (str, os.PathLike)):
            raise TypeError(f"data must be a dataset id, a file or a datasets.Dataset, not {type(self.data).__name__}")
        path = Path(self.data)
        if path.exists():
            builder = DATA_SUFFIXES.get(path.suffix.lower())
            if builder is None:
                raise ValueError(f"data: {path} is not a .jsonl, .json or .parquet file")
            return load_dataset(builder, data_files=str(path), split="train", cache_dir=cache_dir), str(path)
        return load_dataset(str(self.data), split=self.split), f"{self.data}, split {self.split}"

    def _eval_dataset(self, cache_dir: Optional[str] = None):
        """eval_data as a datasets.Dataset, and a description of where it came from; a file is read into
        `cache_dir` when given."""
        from datasets import Dataset, DatasetDict, load_dataset

        data, split = self.eval_data, self.eval_split
        if isinstance(data, DatasetDict):
            if split is None:
                raise ValueError(f"eval_split: required, eval_data has the splits {sorted(data)}")
            if split not in data:
                raise ValueError(f"eval_split: eval_data has no split {split!r}; it has {sorted(data)}")
            return data[split], f"a datasets.DatasetDict, split {split}"
        if isinstance(data, Dataset):
            if split is not None:
                raise ValueError("eval_split: eval_data is a datasets.Dataset, which has no splits")
            return data, "a datasets.Dataset"
        if not isinstance(data, (str, os.PathLike)):
            raise TypeError(f"eval_data must be a dataset id, a file, a datasets.Dataset or a DatasetDict, not "
                            f"{type(data).__name__}")
        path = Path(data)
        if path.exists():
            if split is not None:
                raise ValueError(f"eval_split: eval_data {path} is a file, which has no splits")
            builder = DATA_SUFFIXES.get(path.suffix.lower())
            if builder is None:
                raise ValueError(f"eval_data: {path} is not a .jsonl, .json or .parquet file")
            return load_dataset(builder, data_files=str(path), split="train", cache_dir=cache_dir), str(path)
        split = split if split is not None else EVAL_SPLIT
        return load_dataset(str(data), split=split), f"{data}, split {split}"

    def _load(self, max_length: int):
        """The model and tokenizer, and the loading decisions."""
        import torch

        from meepsquared import _unified_memory
        from meepsquared.models import MeepLanguageModel

        overrides = self.overrides
        if not isinstance(self.model, str):
            model = self.model
            if getattr(model, "_meepsquared", None) is None:
                raise TypeError("model: a loaded model must come from MeepLanguageModel.from_pretrained")
            if hasattr(model, "peft_config"):
                raise ValueError("model already has adapters; give it without them, and the job adds LoRA "
                                 "as its recipe says")
            if self.tokenizer is None:
                raise ValueError("tokenizer: required when model is a loaded model")
            if overrides.attention is not None or overrides.liger is not None:
                raise ValueError("overrides.attention and overrides.liger are loading choices, and the model is "
                                 "loaded already; load it with them set")
            four_bit = bool(getattr(model, "is_loaded_in_4bit", False))
            fp8 = bool(model._meepsquared.get("fp8", False))
            loaded_as = "qlora" if four_bit else "fp8" if fp8 else "lora"
            if loaded_as != self.method:
                raise ValueError(f"method={self.method!r}, and the model was loaded for {loaded_as!r}")
            model.max_seq_length = max_length
            loading = [Decision("loading", "loading", "given", "the model as given, loaded by the caller"),
                       Decision("attention", "attention", model.config._attn_implementation,
                                str(model.config._attn_implementation), "as the model was loaded")]
            return model, self.tokenizer, loading
        if self.tokenizer is not None:
            raise ValueError("tokenizer is for a loaded model; with a model id or path meepsquared loads its tokenizer")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        shared = torch.cuda.is_available() and _unified_memory.shares_system_memory(local_rank)
        loading = [Decision("loading", "loading", "shard_by_shard" if shared else "whole",
                            f"shard by shard to GPU {local_rank}" if shared else f"to GPU {local_rank}",
                            "the GPU shares the system's memory" if shared else "")]
        kwargs = {"max_seq_length": max_length, "load_in_4bit": self.method == "qlora",
                  "load_in_fp8": self.method == "fp8"}
        if overrides.attention is not None:
            kwargs["attn_implementation"] = overrides.attention
        if overrides.liger is not None:
            kwargs["liger"] = overrides.liger
        if overrides.milestone_engine == "vllm":
            kwargs["shared_vllm"] = self._shared_vllm_settings(max_length)
        model, tokenizer = MeepLanguageModel.from_pretrained(self.model, **kwargs)
        if model._meepsquared.get("allocator_cap"):
            loading.append(Decision("allocator_cap", "allocator cap", model._meepsquared["allocator_cap"],
                                    f"{model._meepsquared['allocator_cap'] / 2**30:.1f} GiB",
                                    "the system's available memory less 4 GiB for the page cache and the machine's "
                                    "daemons; an overflow is this process's error, not the machine's"))
        attention_auto, attention_why = model._meepsquared["attention_auto"]
        loading.append(_decide("attention", "attention", overrides.attention, attention_auto, str, attention_why))
        return model, tokenizer, loading

    def _shared_vllm_settings(self, max_length: int) -> dict:
        """The shared vLLM's engine for the milestones' rows: a cache that holds every row's prompt and answer at once,
        sequences as long as the longest, and the adapter's rank."""
        from transformers import AutoConfig

        from meepsquared import _milestones, _shared_vllm
        from meepsquared.models import load_tokenizer

        if not isinstance(self.model, str):
            raise ValueError("overrides.milestone_engine='vllm' loads the model itself; give model as a path")
        tokenizer = load_tokenizer(self.model)
        rows = _milestones.rows(self.milestone_eval)
        lengths = [len(tokenizer(_milestones.prompt(tokenizer, row), add_special_tokens=False)["input_ids"])
                   + _milestones.budget(tokenizer, row) for row in rows]
        rank = self.recipe.rank if self.recipe.rank is not None else RANK
        config = AutoConfig.from_pretrained(self.model)
        sequences = min(len(rows), _shared_vllm.MAX_SEQUENCES)
        longest = max(lengths) + 16
        tokens = min(sum(lengths), sequences * longest)
        return {"max_lora_rank": max(8, 1 << (rank - 1).bit_length()),
                "kv_cache_bytes": _shared_vllm.kv_cache_bytes(config, tokens, sequences),
                "max_model_len": longest, "max_num_seqs": sequences}

    def _output(self, model) -> str:
        if self.output is not None:
            return str(self.output)
        name = (self.model if isinstance(self.model, str) else str(model.config._name_or_path)).rstrip("/")
        name = name.split("/")[-1]
        if not name:
            raise ValueError("output: required; the loaded model has no name to name the folder after")
        return f"{name}-{self.task.upper()}"

    def _check_vllm(self) -> None:
        """Refuse Overrides(vllm=True) where meepsquared cannot generate through vLLM, saying why."""
        import trl

        version = tuple(int(part) for part in trl.__version__.split(".")[:2])
        if version < VLLM_TRL:
            raise ValueError(f"overrides.vllm=True needs TRL {VLLM_TRL[0]}.{VLLM_TRL[1]} or later, and TRL "
                             f"{trl.__version__} is installed")
        if self.method == "fp8":
            raise ValueError("overrides.vllm=True with method='fp8': meepsquared does not send weights held as FP8 to "
                             "vLLM; use method 'lora' or 'qlora'")
        if importlib.util.find_spec("vllm") is None:
            raise ValueError("overrides.vllm=True, and vLLM is not installed; "
                             "https://thinkube.org/meepsquared/grpo.html says which vLLM matches your PyTorch")

    def _answers(self, kind: str, tokenizer, args: dict, decisions: list):
        """Where SFT counts the loss: TRL's settings into args and the decision; returns meepsquared's turn markers or None.

        Prompt and completion pairs: TRL's completion-only loss. Conversations:
        TRL's assistant-only loss where TRL can mask the chat template, else
        meepsquared's train_on_responses_only with the turn markers read from the
        chat template (see meepsquared.chat_templates). Plain text: every token.
        """
        from meepsquared import chat_templates

        given = self.recipe.answers_only
        if kind == PLAIN_TEXT:
            if given:
                raise ValueError("recipe.answers_only=True, and the data is plain text, which has no answers to "
                                 "single out; give conversations ('messages') or prompt and completion pairs")
            decisions.append(_decide("answers_only", "loss on", given, False, _loss_on,
                                     "plain text has no answers to single out"))
            return None
        on = given if given is not None else True
        markers, how, why = None, "", ""
        if kind == PAIRS:
            args["completion_only_loss"] = on
            how, why = "the completions only, by TRL's completion-only loss", "the prompt is context"
        elif on:
            trl_ok, trl_why = chat_templates.trl_assistant_mask(tokenizer)
            if trl_ok:
                args["assistant_only_loss"] = True
                how, why = "the answers only, by TRL's assistant-only loss", trl_why
            else:
                try:
                    markers = chat_templates.turn_markers(tokenizer)
                except ValueError as error:
                    raise ValueError(f"recipe.answers_only: TRL cannot mask these conversations ({trl_why}), and "
                                     f"meepsquared cannot read their turns from the chat template ({error}); set "
                                     "Recipe(answers_only=False) to train on every token") from error
                how = "the answers only, by meepsquared's train_on_responses_only"
                why = (f"{trl_why}; an answer starts after {markers[1]!r} and ends before {markers[0]!r}, as the "
                       "chat template writes them")
        else:
            why = "the user's turns are context"
        decisions.append(_decide("answers_only", "loss on", given, True,
                                 lambda answers: (how or "the answers only") if answers else "every token",
                                 why, why if on else ""))
        return markers

    def _generation(self, max_length: int, vllm: bool, vllm_share: float, args: dict, decisions: list) -> None:
        """GRPO's generation settings: the Recipe's into args, each one's decision with TRL's default beside it."""
        from trl import GRPOConfig

        recipe = self.recipe
        for name in ("max_completion_length", "temperature", "top_p", "top_k", "chat_template_kwargs"):
            if getattr(recipe, name) is not None:
                args[name] = getattr(recipe, name)
        completion_auto = _default(GRPOConfig, "max_completion_length")
        completion = recipe.max_completion_length if recipe.max_completion_length is not None else completion_auto
        if completion >= max_length:
            raise ValueError(f"max_completion_length {completion} leaves no room for the prompt in max_length "
                             f"{max_length}; raise recipe.max_length")
        if vllm:
            args.update(use_vllm=True, vllm_mode="colocate", vllm_max_model_length=max_length)
        decisions.append(_decide("max_completion_length", "completion", recipe.max_completion_length,
                                 completion_auto, lambda n: f"up to {n} tokens", "TRL's default"))
        for name in ("temperature", "top_p"):
            decisions.append(_decide(name, name.replace("_", " "), getattr(recipe, name),
                                     _default(GRPOConfig, name), lambda x: f"{x:g}", "TRL's default"))
        decisions.append(_decide("top_k", "top k", recipe.top_k, _default(GRPOConfig, "top_k"),
                                 lambda k: f"{k}" if k else "no limit", "TRL's default"))
        decisions.append(_decide("chat_template_kwargs", "template", recipe.chat_template_kwargs,
                                 _default(GRPOConfig, "chat_template_kwargs"),
                                 lambda d: ", ".join(f"{k}={v!r}" for k, v in d.items()) if d else "no arguments",
                                 "TRL's default"))
        decisions.append(_decide("vllm", "generation", self.overrides.vllm, False,
                                 lambda on: (f"vLLM in the training process (TRL's colocate mode), sequences of up "
                                             f"to {max_length} tokens, {vllm_share:g} of the GPU's memory"
                                             if on else "Transformers' generate, with the training model"),
                                 "vLLM is installed apart, and Overrides(vllm=True) uses it",
                                 "meepsquared sends the trained weights to vLLM under their checkpoint names"))

    def preprocess(self, cache_dir, cores: Optional[int] = None, say=print):
        """SFT's data tokenized, checked and cut, into `<cache_dir>/<key>` (see meepsquared._preprocess); found
        there when made before from the same inputs. Returns the Outcome: folder, key, found, report."""
        from meepsquared import _preprocess

        return _preprocess.preprocess(self, cache_dir, cores, say)

    def _loading(self, peft_model, tokenizer, dataset, eval_dataset, args: dict):
        """How the trainer that trains loads the preprocessed data (an meepsquared._batches.Loading).

        meepsquared lays out the micro-batches from the examples' lengths, and
        the memory is measured on the largest of them (see meepsquared._batches),
        so the trainer that trains is built after the layout; a provisional
        trainer is built first, for its Loading alone, on the data as
        preprocessed (dataset_kwargs skip_prepare_dataset). It measures nothing
        and compiles nothing; the trainer that trains pads the batches of a
        model with compiled layers to a multiple (see meepsquared._trl), and the
        Loading says so.
        """
        from trl import SFTConfig, SFTTrainer

        from meepsquared import _batches, _checkpointing, _trl

        was = peft_model._meepsquared_gradient_checkpointing, peft_model._meepsquared_compile_layers
        peft_model._meepsquared_gradient_checkpointing = len(_checkpointing.layers(peft_model))
        peft_model._meepsquared_compile_layers = False
        try:
            # A batch of 2 rows: TRL warns about padding-free training with a batch of 1 row, which a reader would
            # take for the training's
            trainer = SFTTrainer(model=peft_model,
                                 args=SFTConfig(max_length=None, per_device_train_batch_size=2,
                                                gradient_accumulation_steps=1, max_steps=1, report_to=[],
                                                dataset_kwargs={"skip_prepare_dataset": True}, **args),
                                 train_dataset=dataset, eval_dataset=eval_dataset, processing_class=tokenizer)
        finally:
            peft_model._meepsquared_gradient_checkpointing, peft_model._meepsquared_compile_layers = was
        loading = _batches.loading(trainer)
        if was[1] and loading.padded and "pad_to_multiple_of" not in args:
            loading = dataclasses.replace(loading, multiple=_trl.COMPILED_PAD_MULTIPLE)
        return loading

    def _data(self, peft_model, tokenizer, folder, kind: str, markers, answer_args: dict, max_length: int,
              args: dict) -> _Data:
        """SFT's data as preprocessed in `folder` (see meepsquared._preprocess), memory-mapped, after the check
        that the folder was made from this job's inputs."""
        from meepsquared import _preprocess

        inputs = self._preprocess_inputs(tokenizer, kind, answer_args, markers, max_length)
        train, eval_prepared, report = _preprocess.open_folder(folder, inputs)
        if report["kind"] != kind:
            raise ValueError(f"preprocessed: {folder} holds {report['kind']}, and the job's data is {kind}")
        loading = self._loading(peft_model, tokenizer, train, eval_prepared, args)
        if loading.seed != report["seed"]:
            raise ValueError(f"the trainer's seed {loading.seed} is not the seed {report['seed']} the data was "
                             "preprocessed with (its held-out share and the order of its micro-batches)")
        kinds = weights = None
        if self.recipe.source_weights is not None:
            kinds = list(train[_preprocess.KIND])
            weights = [self.recipe.source_weights.get(kind_, 1.0) for kind_ in kinds]
        # The facts kept beside the tokens are read; the trainer gets the tokens alone
        facts = [c for c in train.column_names if c in _preprocess.FACT_COLUMNS]
        train = train.remove_columns(facts)
        if eval_prepared is not None:
            eval_prepared = eval_prepared.remove_columns([c for c in eval_prepared.column_names
                                                          if c in _preprocess.FACT_COLUMNS])
        decision = Decision(**report["decision"])
        return _Data(train, eval_prepared, loading, report["held"], kinds, weights, decision, report)

    def _preprocess_inputs(self, tokenizer, kind: str, answer_args: dict, markers, max_length: int) -> dict:
        """The inputs that key this job's preprocessing (see meepsquared._preprocess.inputs), the data identified
        without loading a file."""
        from meepsquared import _preprocess

        def identity(data, split):
            if isinstance(data, (str, os.PathLike)) and Path(data).exists():
                return _preprocess.data_identity(data, split, None)
            loaded = self._dataset()[0] if data is self.data else self._eval_dataset()[0]
            return _preprocess.data_identity(data, split, loaded)

        return _preprocess.inputs(identity(self.data, self.split),
                                  identity(self.eval_data, self.eval_split) if self.eval_data is not None else None,
                                  tokenizer, kind, answer_args, markers, max_length, self.eval_share)

    def _prepare(self) -> _Prepared:
        if self._prepared is not None:
            return self._prepared
        import torch

        from trl import GRPOConfig, SFTConfig

        from meepsquared import (_allocator, _batches, _checkpointing, _compile, _data_checks, _milestones, _moe,
                            _preprocess, _selective, _sessions, _unified_memory, chat_templates)
        from meepsquared.models import DEFAULT_TARGETS, MeepLanguageModel, _fused_wanted, keeps_examples_apart

        recipe, overrides = self.recipe, self.overrides
        output = self._output(self.model)
        dataset = source = eval_source = None
        if self.task == "sft":
            # The data is preprocessed (tokenized, checked, cut) before the model loads: by the caller, into
            # the folder given as `preprocessed`, else here, into the output folder
            folder = self.preprocessed
            if folder is None:
                folder = _preprocess.preprocess(self, Path(output) / PREPROCESSED, say=_say).folder
            with open(Path(folder) / _preprocess.REPORT_FILE) as f:
                report = json.load(f)
            kind, source, eval_source, rows = report["kind"], report["source"], report["eval_source"], report["rows"]
        else:
            dataset, source = self._dataset()
            kind, columns = data_format(dataset.column_names, self.task)
            rows = len(dataset)
        rewards = [load_reward(r) for r in self.rewards] if self.rewards else None
        max_length = recipe.max_length if recipe.max_length is not None else MAX_LENGTH

        model, tokenizer, decisions = self._load(max_length)
        base_dtype = model.get_input_embeddings().weight.dtype
        dtype_name = str(base_dtype).removeprefix("torch.")
        device = next(model.parameters()).device
        weights = {"lora": f"{dtype_name}", "qlora": f"4-bit NF4, computing in {dtype_name}",
                   "fp8": f"FP8 e4m3, computing in {dtype_name}"}[self.method]
        decisions.append(Decision("weights", "weights", self.method, weights, f"method {self.method}"))

        # Counted before LoRA; a 4-bit weight packs two parameters in a byte, and its quant_state has its shape
        parameters = sum(p.quant_state.shape.numel() if getattr(p, "quant_state", None) is not None else p.numel()
                         for p in model.parameters())
        by_size = size_defaults(parameters)

        # LoRA
        rank = recipe.rank if recipe.rank is not None else RANK
        alpha = recipe.alpha if recipe.alpha is not None else rank
        dropout = recipe.dropout if recipe.dropout is not None else 0.0
        targets = recipe.targets if recipe.targets is not None else DEFAULT_TARGETS
        decisions.append(_decide("rank", "LoRA rank", recipe.rank, RANK, str, "meepsquared's default"))
        decisions.append(_decide("alpha", "LoRA alpha", recipe.alpha, rank, str, "alpha = rank, a scaling of 1"))
        decisions.append(_decide("dropout", "LoRA dropout", recipe.dropout, 0.0, str, "get_peft_model's default"))
        decisions.append(_decide("targets", "targets", recipe.targets, DEFAULT_TARGETS, ", ".join,
                                 "the attention and MLP projections"))

        # LoRA on the routed experts: PEFT's target_parameters, computed by meepsquared's routed LoRA
        expert_names, expert_stacks = _moe.expert_targets(model)
        if recipe.experts and not expert_stacks:
            raise ValueError(f"recipe.experts=True, and this {model.config.model_type} model has no expert stacks: "
                             "it is not a mixture of experts as Transformers stores one")
        expert_targets = {"target_parameters": expert_names} if recipe.experts else {}

        def experts_text(on: bool) -> str:
            if on:
                return (f"LoRA on {expert_stacks} expert stacks ({', '.join(expert_names)}), on the tokens routed "
                        "to each")
            return "frozen" if expert_stacks else "none, not a mixture of experts"

        decisions.append(_decide("experts", "experts", recipe.experts, False, experts_text,
                                 "LoRA on the modules in targets; Recipe(experts=True) adds every expert"
                                 if expert_stacks else "",
                                 f"meepsquared's routed LoRA; {_moe.kernel_text(device)}" if recipe.experts else ""))

        # The kernels auto mode would choose, before get_peft_model decides them
        fused_auto, fused_why = _fused_wanted(model)
        attention = model.config._attn_implementation
        compile_auto, compile_why = _compile.decision(model.config, attention)
        if device.type != "cuda":
            compile_auto, compile_why = False, "the model is not on a CUDA GPU"
        peft_model = MeepLanguageModel.get_peft_model(
            model, r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets,
            use_gradient_checkpointing="auto",
            fused_lora=overrides.fused_lora if overrides.fused_lora is not None else "auto",
            compile_layers=overrides.compile_layers if overrides.compile_layers is not None else "auto",
            **expert_targets)
        decisions.append(_decide("fused_lora", "fused LoRA", overrides.fused_lora, fused_auto, _on_off,
                                 fused_why or "the model's MLP has a fused kernel"))
        decisions.append(_decide("compile_layers", "compile", overrides.compile_layers, compile_auto,
                                 lambda on: "layers compiled" if on else "layers not compiled", compile_why))
        engine = model._meepsquared.get("engine")
        if engine is not None:
            # §6.1 of design/milestones-vllm-shared-weights.md: vLLM and the trainer answer alike, adapter on
            from meepsquared import _milestones, _shared_vllm

            check_rows = _milestones.rows(self.milestone_eval)[:_shared_vllm.CHECK_PROMPTS]
            check_prompts = [tokenizer(_milestones.prompt(tokenizer, row), add_special_tokens=False)["input_ids"]
                             for row in check_rows]
            checked = _shared_vllm.correctness(engine, peft_model, tokenizer, check_prompts)
            shared_gib = engine.memory["shared_stacks"] / 2**30
            decisions.append(Decision(
                "milestone_engine", "milestones by", "vllm",
                f"vLLM in this process, the {shared_gib:.1f} GiB of expert stacks held once, and "
                f"{engine.memory['shared_other'] / 2**30:.1f} GiB of the other weights ({len(engine.kept)} kept apart: "
                "they differ in vLLM)",
                f"overrides.milestone_engine; memory before vLLM {engine.memory['before_vllm'] / 2**30:.1f} GiB, "
                f"after {engine.memory['after_vllm'] / 2**30:.1f} GiB, peak while loading "
                f"{engine.memory['loading_peak'] / 2**30:.1f} GiB; the check on {len(check_prompts)} rows: "
                f"{_shared_vllm.check_text(checked)}", "user"))

        # GRPO's generation through vLLM, whose share of the GPU's memory the measurement leaves free
        vllm = bool(overrides.vllm)
        if vllm:
            self._check_vllm()
        vllm_share = _default(GRPOConfig, "vllm_gpu_memory_utilization")

        def reserved(total: int) -> int:
            return int(vllm_share * total) if vllm else 0

        # What the trainable parameters add to every step: their gradients and the optimizer's state for them.
        # Counted apart from the margin, which is for the loss, the workspaces and the passes' own buffers: an
        # adapter on every expert of a mixture of experts has hundreds of millions of parameters
        optimizer_name = recipe.optimizer if recipe.optimizer is not None else OPTIMIZER
        trained_bytes = _checkpointing.trainable_bytes(peft_model, optimizer_name)
        trained_count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        # The batch. By default it counts examples (GRPO: completions, as TRL's batches do). With
        # Recipe(tokens_per_update=...), SFT counts it in tokens instead: a budget per update, each
        # micro-batch holding as many conversations as fit its share (meepsquared._batches). SFT's
        # micro-batches are laid out ahead for the whole training, in a seeded order, and the largest
        # of them is measured; the trainer then trains that layout. GRPO's micro-batch is measured at
        # its bound, rows of max_length. The checkpointed layers follow from the batch measured.
        world = int(os.environ.get("WORLD_SIZE", 1))
        by_tokens = self.task == "sft" and recipe.tokens_per_update is not None
        measured_budget = recipe.tokens_per_update == AUTO
        effective_auto, effective_why = by_size["effective_batch"]
        layer_count = len(_checkpointing.layers(peft_model))
        if overrides.checkpointed_layers is not None and overrides.checkpointed_layers > layer_count:
            raise ValueError(f"overrides.checkpointed_layers={overrides.checkpointed_layers}, and the model has "
                             f"{layer_count} layers")
        compiled = overrides.compile_layers if overrides.compile_layers is not None else compile_auto
        selective_refusal = _selective.refusal(peft_model, compiled)
        if overrides.selective_layers:
            if selective_refusal:
                raise ValueError(f"overrides.selective_layers={overrides.selective_layers}: {selective_refusal}")
            if (overrides.checkpointed_layers or 0) + overrides.selective_layers > layer_count:
                raise ValueError(f"overrides.checkpointed_layers={overrides.checkpointed_layers} and "
                                 f"selective_layers={overrides.selective_layers} name more than the model's "
                                 f"{layer_count} layers")
        # Auto mode measures when it decides the batch, the micro-batch or the checkpointed layers,
        # and on a GPU also when all are set by hand, to report its own choice beside them
        need_auto = (overrides.micro_batch is None or overrides.checkpointed_layers is None
                     or measured_budget)
        measure = need_auto or device.type == "cuda"
        measured = {}
        # SFT: the data as preprocessed, memory-mapped (see _data); the micro-batches are laid out
        # from the examples' lengths
        prepared, lengths, loading, markers, answer_args, data = None, None, None, None, {}, None
        # The evaluation set, preprocessed alike; with eval_share, held out of the training data
        # before the micro-batches are laid out, so that none is trained on or measured
        eval_prepared, held = None, 0
        base_args = {"output_dir": output, "bf16": base_dtype == torch.bfloat16, "fp16": base_dtype == torch.float16,
                     "use_cpu": device.type == "cpu"}
        if self.task == "sft":
            markers = self._answers(kind, tokenizer, answer_args, decisions)
            data = self._data(peft_model, tokenizer, folder, kind, markers, answer_args, max_length,
                              {**base_args, **answer_args,
                               **({"padding_free": overrides.padding_free} if overrides.padding_free is not None
                                  else {})})
            prepared, eval_prepared, loading, held = data.train, data.eval, data.loading, data.held
            decisions.append(data.decision)
            weights_text = (", ".join(f"{kind} rows x{weight:g}" for kind, weight in recipe.source_weights.items())
                            + "; every other kind x1" if recipe.source_weights is not None else "none")
            decisions.append(Decision("source_weights", "weights", recipe.source_weights, weights_text,
                                      "each micro-batch holds rows of one origin kind, and its loss is multiplied "
                                      "by the kind's weight" if recipe.source_weights is not None
                                      else "every row's loss weighs the same"))
            lengths = _batches.lengths(prepared, max_length)
        # With a token budget: the budget of each process in rows (slots) of max_length tokens
        slots = None
        layouts = {}

        def batching(micro: int, budget_rows: Optional[int]) -> tuple:
            """(accumulation, tokens per micro-batch or None): for `micro` examples per micro-batch, or with a
            budget of `budget_rows` rows of max_length per process, for `micro` rows per forward pass."""
            if not by_tokens:
                return per_process // micro, None
            accumulation = math.ceil(budget_rows / micro)
            process_tokens = budget_rows * max_length if measured_budget else recipe.tokens_per_update // world
            return accumulation, math.ceil(process_tokens / accumulation)

        def laid_out(micro: int, budget_rows: Optional[int]):
            """The micro-batches of the whole training with this batch (see batching), laid out ahead."""
            if (micro, budget_rows) not in layouts:
                accumulation, budget = batching(micro, budget_rows)
                layouts[micro, budget_rows] = _batches.lay_out(lengths, loading, accumulation, self.steps,
                                                               self.epochs, micro=None if by_tokens else micro,
                                                               budget=budget,
                                                               kinds=data.kinds if data.weights is not None else None)
            return layouts[micro, budget_rows]

        def shape(micro: int, budget_rows: Optional[int]) -> _checkpointing.Shape:
            if lengths is None:
                return _checkpointing.Shape(micro, max_length)
            return _checkpointing.largest_batch(laid_out(micro, budget_rows).trained, lengths, not loading.padded,
                                                loading.multiple)

        def batch_text(micro: int, budget_rows: Optional[int]) -> str:
            if lengths is None:
                return f"{shape(micro, budget_rows).text()}, the bound: GRPO generates its completions as it trains"
            if by_tokens:
                bound = f"{micro} rows of {max_length} would be {micro * max_length:,}"
            else:
                longest = _checkpointing.longest_examples(lengths, micro, max_length, not loading.padded,
                                                          loading.multiple)
                bound = f"the data's {micro} longest would be {longest.tokens:,}"
            trained = len(laid_out(micro, budget_rows).trained)
            return (f"the largest of the {trained:,} micro-batches this job trains, "
                    f"{shape(micro, budget_rows).text()}; {bound}")

        def measurement(micro: int, budget_rows: Optional[int]):
            batch = shape(micro, budget_rows)
            if batch not in measured:
                if not measured and device.type == "cuda" and _unified_memory.shares_system_memory(device.index or 0):
                    # The data is tokenized and vLLM, if any, started, both host memory outside the allocator: the
                    # cap for the GPU work that fills the memory is set from what the machine has left now
                    cap = _unified_memory.recap_allocator(device.index or 0)
                    model._meepsquared["allocator_cap"] = cap
                    decisions[:] = [_recapped(d, cap) if d.name == "allocator_cap" else d for d in decisions]
                measured[batch] = _measure(peft_model, batch)
            return measured[batch]

        def fits(micro: int, budget_rows: Optional[int]) -> bool:
            every, per_layer, free, total = measurement(micro, budget_rows)
            return _checkpointing.fits(free - reserved(total) - trained_bytes,
                                       _checkpointing.margin(device, free, total), every, per_layer)

        def largest_fitting(start: int, smaller: Callable[[int], int], as_budget: bool = False) -> int:
            """From `start` down, by `smaller`: the first micro-batch that fits with every layer checkpointed;
            as_budget, each one with a budget of its own rows."""
            micro = start
            while not fits(micro, micro if as_budget else slots):
                if micro == 1:
                    rows = 1 if as_budget else slots
                    every, per_layer, free, total = measurement(1, rows)
                    raise RuntimeError(f"the smallest micro-batch ({batch_text(1, rows)}) does not fit with every "
                                       f"layer checkpointed: it holds {_gib(every + per_layer)}, and "
                                       f"{_gib(free - reserved(total) - trained_bytes - _checkpointing.margin(device, free, total))} "
                                       f"is free beyond the margin, the {_gib(trained_bytes)} of the trainable "
                                       "parameters' gradients and optimizer state" + (" and vLLM's share" if vllm else "")
                                       + "; "
                                       "lower recipe.max_length or use method='qlora'")
                micro = smaller(micro)
            return micro

        def half(micro: int) -> int:
            return math.ceil(micro / 2)

        if by_tokens:
            # The budget given, or the most rows that fit one forward pass up to the ceiling; the
            # micro-batch is the largest share of them that fits
            if not measured_budget:
                if recipe.tokens_per_update % world:
                    raise ValueError(f"tokens_per_update {recipe.tokens_per_update} does not divide among {world} "
                                     "processes")
                per_process_tokens = recipe.tokens_per_update // world
                slots = math.ceil(per_process_tokens / max_length)
                micro_auto = largest_fitting(slots, half) if measure else None
            else:
                slots = largest_fitting(max(1, TOKENS_PER_UPDATE_CEILING // (max_length * world)), half,
                                        as_budget=True)
                per_process_tokens = slots * max_length
                micro_auto = slots
            tokens = per_process_tokens * world
            micro = overrides.micro_batch if overrides.micro_batch is not None else micro_auto
            # Each forward pass takes an equal share of the process's tokens (see batching)
            accumulation = batching(micro, slots)[0]
            share = f"the {per_process_tokens:,} tokens per update of each process"
        else:
            effective = recipe.effective_batch if recipe.effective_batch is not None else effective_auto
            if effective % world:
                raise ValueError(f"effective batch {effective} does not divide among {world} processes")
            per_process = effective // world
            if overrides.micro_batch is not None and per_process % overrides.micro_batch:
                raise ValueError(f"overrides.micro_batch={overrides.micro_batch} does not divide the {per_process} "
                                 "examples per update of each process")
            micro_auto = (largest_fitting(per_process, lambda m: smaller_micro_batch(per_process, m)) if measure
                          else None)
            micro = overrides.micro_batch if overrides.micro_batch is not None else micro_auto
            accumulation = batching(micro, None)[0]
            share = f"{per_process} examples"

        figures = None
        if measure:
            every, per_layer, free, total = measurement(micro, slots)
            figures = _measure_figures(peft_model, shape(micro, slots), every, per_layer, compiled)
            usable = free - reserved(total) - trained_bytes
            margin_bytes = _checkpointing.margin(device, free, total)
            auto_layout = _checkpointing.choose(usable, margin_bytes, figures.per_layer, figures.per_selective,
                                                figures.times, layer_count)
            checkpointed_auto, selective_auto = auto_layout.whole, auto_layout.selective
            free_bytes = free
            micro_why = (f"the largest share of {share} that fits with every layer checkpointed: "
                         f"{_gib(sum(measurement(micro_auto, slots)[:2]))} for {batch_text(micro_auto, slots)}")
            checkpoint_why = (f"{figures.text(batch_text(micro, slots))}; free: {free / 2**30:.0f} of "
                              f"{total / 2**30:.0f} GiB, {_gib(trained_bytes)} of it for the gradients and the "
                              f"{optimizer_name} state of {trained_count:,} trainable parameters"
                              + (f", {reserved(total) / 2**30:.0f} GiB of it kept for vLLM" if vllm else "")
                              + (f"; selective recomputation not offered: {selective_refusal}"
                                 if selective_refusal else ""))
        else:
            checkpointed_auto = selective_auto = None
            free_bytes = margin_bytes = None
            micro_why = checkpoint_why = "the model is not on a CUDA GPU"
        whole_auto, whole_why = checkpointed_auto, checkpoint_why
        if overrides.checkpointed_layers is not None:
            layout = _checkpointing.Recompute(overrides.checkpointed_layers, overrides.selective_layers or 0)
        elif overrides.selective_layers is not None:
            # The user's selective layers, and as few recomputed whole as the memory needs beside them
            whole_auto = _checkpointing.fewest_whole(usable, margin_bytes, figures.per_layer, figures.per_selective,
                                                     layer_count, overrides.selective_layers)
            whole_why = (f"the fewest layers recomputed whole that fit with {overrides.selective_layers} "
                         f"selective; {checkpoint_why}")
            layout = _checkpointing.Recompute(whole_auto, overrides.selective_layers)
        else:
            layout = auto_layout
        # Set as a layout, so that the trainer takes the layers as they are and does not measure again
        _checkpointing.set_checkpointed(peft_model, layout)
        peft_model._meepsquared_gradient_checkpointing = layout
        peft_model._meepsquared_checkpointed_layers = layout

        def layers_text(n) -> str:
            return f"{n} of {layer_count} layers" if n is not None else "not measured"

        decisions.append(_decide("checkpointed_layers", "checkpoint", overrides.checkpointed_layers, whole_auto,
                                 layers_text, whole_why))
        selective_given = (overrides.selective_layers if overrides.selective_layers is not None
                           else 0 if overrides.checkpointed_layers is not None else None)
        decisions.append(_decide("selective_layers", "selective", selective_given, selective_auto, layers_text,
                                 selective_refusal or "recompute their cheap operations only, keeping the outputs "
                                                      "of their products",
                                 "checkpointed_layers given by hand: the other layers keep everything"
                                 if overrides.selective_layers is None else ""))
        if by_tokens:
            def micro_text(rows) -> str:
                if rows is None:
                    return "not measured"
                return (f"up to {math.ceil(per_process_tokens / math.ceil(slots / rows)):,} tokens per forward "
                        f"pass ({rows} x {max_length})")

            accumulation_why = "tokens per update / (tokens per forward pass x processes)"
        else:
            decisions.append(_decide("effective_batch", "batch", recipe.effective_batch, effective_auto,
                                     lambda n: f"{n} examples per update",
                                     f"{effective_why}; {BY_EXAMPLES}" if self.task == "sft" else effective_why))

            def micro_text(rows) -> str:
                return f"{rows}" if rows is not None else "not measured"

            accumulation_why = "effective batch / (micro-batch x processes)"
        decisions.append(_decide("micro_batch", "micro-batch", overrides.micro_batch, micro_auto, micro_text,
                                 micro_why))
        decisions.append(Decision("accumulation", "accumulation", accumulation,
                                  f"{accumulation} micro-batches per update"
                                  + (f", on each of {world} processes" if world > 1 else ""), accumulation_why))

        # The optimizer and the learning rate, which a token budget scales
        optimizer = recipe.optimizer if recipe.optimizer is not None else OPTIMIZER
        decisions.append(_decide("optimizer", "optimizer", recipe.optimizer, OPTIMIZER, str,
                                 "8-bit AdamW: a quarter of AdamW's state memory"))
        rate_base, size_why = by_size["learning_rate"]
        if by_tokens:
            rate_auto = scaled_learning_rate(rate_base, tokens)
            rate_why = (f"{rate_base:g} at {REFERENCE_TOKENS:,} tokens per update, x sqrt({tokens:,} / "
                        f"{REFERENCE_TOKENS:,}) = {math.sqrt(tokens / REFERENCE_TOKENS):.3g} for this batch "
                        f"({RATE_RULE}); {size_why}")
        else:
            rate_auto, rate_why = rate_base, size_why
        rate = recipe.learning_rate if recipe.learning_rate is not None else rate_auto
        decisions.append(_decide("learning_rate", "learning rate", recipe.learning_rate, rate_auto,
                                 lambda r: f"{r:g}", rate_why))
        generations = recipe.generations if recipe.generations is not None else _default(GRPOConfig, "num_generations")
        schedule = recipe.schedule if recipe.schedule is not None else SCHEDULE
        decisions.append(Decision("max_length", "tokens", max_length,
                                  f"up to {max_length} per example"
                                  + (", prompt and completion" if self.task == "grpo" else ""),
                                  "as given" if recipe.max_length is not None else "TRL's SFTConfig default"))
        if self.task == "grpo":
            decisions.append(_decide("generations", "generations", recipe.generations, generations, str,
                                     "TRL's default"))
        decisions.append(Decision("data", "data", source, f"{source}: {rows:,} rows of {kind}"
                                  + (f", {held:,} of them held out for evaluation" if held else "")))
        decisions.append(Decision("output", "output", output, output,
                                  "as given" if self.output is not None else "named after the model and the task"))

        # TRL's trainer, as train() runs it
        # The warm-up is set on the trainer's args once the steps are counted, below: with a token
        # budget the count needs the data as TRL tokenizes it, which the trainer does when built
        args = {**base_args, "per_device_train_batch_size": micro,
                "gradient_accumulation_steps": accumulation, "learning_rate": rate,
                "lr_scheduler_type": "constant_with_warmup" if schedule == "constant" else "cosine_with_min_lr",
                "optim": optimizer, "logging_steps": 1}
        if schedule == "cosine":
            args["lr_scheduler_kwargs"] = {"min_lr_rate": COSINE_FLOOR}
        if self.steps is not None:
            args["max_steps"] = self.steps
        else:
            args["num_train_epochs"] = self.epochs if self.epochs is not None else 1
        for name in ("max_grad_norm", "weight_decay"):
            if getattr(recipe, name) is not None:
                args[name] = getattr(recipe, name)
        if self.report_to is not None:
            args["report_to"] = list(self.report_to) if isinstance(self.report_to, tuple) else self.report_to
        if self.task == "sft":
            from trl import SFTTrainer

            args.update(answer_args)
            # from_pretrained decided whether the model can train padding-free, and why not
            padding_free_auto = model._meepsquared["padding_free"]
            refusal = model._meepsquared.get("padding_free_reason")
            if overrides.padding_free and not padding_free_auto:
                raise ValueError(f"overrides.padding_free=True, and this model cannot train padding-free: {refusal}")
            if overrides.padding_free is not None:
                args["padding_free"] = overrides.padding_free
            # Tokenized already, for the layout of the batches (see _tokenized)
            args["dataset_kwargs"] = {"skip_prepare_dataset": True}
            if eval_prepared is not None:
                # Evaluated every eval_every steps, in micro-batches of the training's size, the loss alone
                every_auto = max(1, math.ceil(laid_out(micro, slots).steps / EVALUATIONS))
                every = self.eval_every if self.eval_every is not None else every_auto
                args.update(eval_strategy="steps", eval_steps=every, per_device_eval_batch_size=micro,
                            prediction_loss_only=True)
            trainer = SFTTrainer(model=peft_model, args=SFTConfig(max_length=max_length, **args),
                                 train_dataset=prepared, eval_dataset=eval_prepared, processing_class=tokenizer)
            if markers is not None:
                chat_templates.train_on_responses_only(trainer, instruction_part=markers[0],
                                                       response_part=markers[1])
            decisions.append(_decide("padding_free", "batches", overrides.padding_free, padding_free_auto,
                                     lambda on: "packed, no padding" if on else "padded",
                                     keeps_examples_apart(attention) if padding_free_auto else refusal or ""))
        else:
            from trl import GRPOTrainer

            if recipe.generations is not None:
                args["num_generations"] = recipe.generations
            self._generation(max_length, vllm, vllm_share, args, decisions)
            trainer = GRPOTrainer(model=peft_model, reward_funcs=rewards, args=GRPOConfig(**args),
                                  train_dataset=dataset, processing_class=tokenizer)
            decisions.append(Decision("padding_free", "batches", False, "padded", "GRPO's batches are TRL's"))

        # The steps as the trainer will count them, and the warm-up from them. SFT trains the
        # micro-batches laid out for the measurement (meepsquared._batches)
        if self.task == "sft":
            laid = laid_out(micro, slots)
            _batches.install(trainer, laid)
            if data.weights is not None:
                _data_checks.install_weights(trainer, data.weights)
            planned = laid.steps
            epochs_laid = len(laid.epochs)
            decisions.append(Decision("order", "order", loading.seed,
                                      f"shuffled with seed {loading.seed}, the micro-batches of {epochs_laid} "
                                      f"epoch{'s' if epochs_laid != 1 else ''} laid out ahead",
                                      "the trainer's seed; the training runs the micro-batches measured"))
            if by_tokens:
                first_step = sum(len(group) for group in laid.epochs[0][:accumulation * world])
                budget_text = (f"{tokens:,} tokens per update, as many conversations as fit ({first_step} in the "
                               "first step)")
                if measured_budget:
                    decisions.append(Decision("tokens_per_update", "batch", tokens, budget_text, TOKENS_SOURCE))
                else:
                    decisions.append(Decision("tokens_per_update", "batch", tokens, budget_text, "", "user",
                                              effective_auto,
                                              f"{effective_auto} examples per update, {effective_why}"))
        else:
            planned = planned_steps(self.steps, self.epochs,
                                    batches_per_epoch(rows, micro, world, self.task, generations), accumulation)
        warmup_auto = warmup_steps(planned)
        warmup = recipe.warmup_steps if recipe.warmup_steps is not None else warmup_auto
        trainer.args.warmup_steps = warmup
        decisions.append(_decide("schedule", "schedule", recipe.schedule, SCHEDULE,
                                 lambda s: ("constant after the warm-up" if s == "constant"
                                            else f"cosine decay to {COSINE_FLOOR:g} of the rate"),
                                 f"over the {planned} planned steps; to be set by the measurement round"))
        decisions.append(_decide("warmup_steps", "warm-up", recipe.warmup_steps, warmup_auto,
                                 lambda n: f"{n} steps", f"{WARMUP_STEPS} steps or {WARMUP_SHARE:.0%} of the "
                                                         f"{planned} planned steps, whichever is fewer"))
        length = (f"{self.steps} steps" if self.steps is not None
                  else f"{self.epochs or 1:g} epoch{'s' if (self.epochs or 1) != 1 else ''}, {planned} steps")
        decisions.append(Decision("length", "length", planned, length,
                                  "as given" if self.steps or self.epochs else "one epoch, as no steps were given"))
        evaluations = eval_batches = 0
        if eval_prepared is not None:
            eval_count = len(eval_prepared)
            eval_batches = math.ceil(math.ceil(eval_count / world) / micro)
            # Transformers evaluates every `every` steps, and at the last step
            evaluations = math.ceil(planned / every)

            def evaluation_text(k: int) -> str:
                return (f"{eval_count:,} rows of {eval_source}, every {k} steps ({math.ceil(planned / k)} "
                        "evaluations), the loss counted as the training's")

            value = {"data": eval_source, "rows": eval_count, "every": every}
            every_why = f"about {EVALUATIONS} evaluations over the {planned} planned steps"
            if self.eval_every is None:
                decisions.append(Decision("evaluation", "evaluation", value, evaluation_text(every), every_why))
            else:
                decisions.append(Decision("evaluation", "evaluation", value, evaluation_text(every), "", "user",
                                          {**value, "every": every_auto},
                                          f"{evaluation_text(every_auto)}, {every_why}"))
        else:
            decisions.append(Decision("evaluation", "evaluation", None, "none", "no eval_data or eval_share given"))

        config_cls = type(trainer.args)
        decisions.append(_decide("max_grad_norm", "clipping", recipe.max_grad_norm,
                                 _default(config_cls, "max_grad_norm"), lambda n: f"gradient norm at most {n:g}",
                                 "the default of TRL's config"))
        decisions.append(_decide("weight_decay", "weight decay", recipe.weight_decay,
                                 _default(config_cls, "weight_decay"), lambda n: f"{n:g}",
                                 "the default of TRL's config"))
        reported = list(trainer.args.report_to or [])
        decisions.append(Decision("report_to", "tracking", reported, ", ".join(reported) or "none",
                                  "as given" if self.report_to is not None else "Transformers' default"))

        def liger(on: bool) -> str:
            return _liger_text(trainer.args) if on == trainer.args.use_liger_kernel else _on_off(on)

        liger_auto = True if isinstance(self.model, str) else model._meepsquared["liger"]
        decisions.append(_decide("liger", "Liger", overrides.liger, liger_auto, liger,
                                 "from_pretrained's default" if isinstance(self.model, str)
                                 else "as the model was loaded"))
        decisions.append(Decision("allocator", "allocator", _allocator.DECISION, _allocator.DECISION))

        # The time per step: optimizer steps timed on the largest micro-batch, which also measures the memory, and on
        # the median one; each step is priced on the line through the two, at the mean micro-batch the training runs
        timed = self.timed_steps if self.timed_steps is not None else TIMED_STEPS
        seconds_per_step = estimated = timing = None
        if self.task != "sft":
            time_why = "GRPO's steps generate their completions, which timed training steps would leave out"
        elif timed == 0:
            time_why = "timed_steps=0"
        elif device.type != "cuda" and self.timed_steps is None:
            time_why = "the model is not on a CUDA GPU; timed_steps set by hand times the steps there too"
        else:
            if lengths is None:
                trained_shapes = [_checkpointing.Shape(micro, max_length)]
            else:
                trained_shapes = _checkpointing.batch_shapes(laid_out(micro, slots).trained, lengths,
                                                             not loading.padded, loading.multiple)
            largest = max(trained_shapes, key=lambda s: s.tokens)
            median = _checkpointing.median_batch(trained_shapes)
            mean = _checkpointing.mean_tokens(trained_shapes)
            eval_largest = eval_median = None
            eval_mean = 0.0
            if evaluations:
                # The evaluation's batches: `micro` consecutive rows each, as its sequential loader takes them
                eval_lengths = _batches.lengths(eval_prepared, max_length)
                eval_shapes = _checkpointing.batch_shapes(
                    [range(start, min(start + micro, len(eval_lengths)))
                     for start in range(0, len(eval_lengths), micro)],
                    eval_lengths, not loading.padded, loading.multiple)
                eval_largest = max(eval_shapes, key=lambda s: s.tokens)
                eval_median = _checkpointing.median_batch(eval_shapes)
                eval_mean = _checkpointing.mean_tokens(eval_shapes)
            # The steps on the largest micro-batch are also the check of the memory decision: their real peak
            # against the budget. A peak over it puts more layers to recompute whole and times again.
            timing, layout, corrected = _timed_steps_fitting(
                trainer, peft_model, largest, accumulation, timed, eval_largest, layout, figures, layer_count,
                device, overrides.checkpointed_layers is not None, margin_bytes)
            if corrected:
                for i, d in enumerate(decisions):
                    if d.name == "checkpointed_layers":
                        decisions[i] = Decision(d.name, d.label, layout.whole, layers_text(layout.whole),
                                                f"{corrected}; before the check: {d.text}, {d.reason}")
                    elif d.name == "selective_layers":
                        decisions[i] = Decision(d.name, d.label, layout.selective, layers_text(layout.selective),
                                                d.reason)
            small = None
            if median.tokens != largest.tokens or (eval_median is not None and eval_median.tokens != eval_largest.tokens):
                small = _sessions.time_steps(trainer, median, accumulation, timed, eval_median)
            small = small or timing
            seconds_per_step = _sessions.on_the_line(median.tokens, small.step_seconds, largest.tokens,
                                                     timing.step_seconds, mean)
            eval_seconds = 0
            if evaluations:
                forward = _sessions.on_the_line(eval_median.tokens, small.forward_seconds, eval_largest.tokens,
                                                timing.forward_seconds, eval_mean)
                eval_seconds = evaluations * eval_batches * forward
            estimated = seconds_per_step * planned + eval_seconds
            time_why = (f"the median of {timed} optimizer steps timed after {_sessions.TIMING_WARMUP} to warm up, "
                        f"each {accumulation} forward and backward passes, on the largest micro-batch, "
                        f"{largest.text()}, and on the median one, {median.text()}; each step priced on the line "
                        f"through the two at the mean micro-batch of the training, {mean:,.0f} tokens")
            if evaluations:
                time_why += (f"; each evaluation {eval_batches} forward passes, priced the same way from timed "
                             f"forward passes of its largest batch, {eval_largest.text()}, and its median one, "
                             f"{eval_median.text()}, at its mean batch, {eval_mean:,.0f} tokens")
        # The milestones: their steps, their rows checked against the chat template now rather than at the first
        # milestone, and the time their generation takes, from one batch timed
        milestone_at = milestone_rows = milestone_scorer = None
        milestone_seconds = 0.0
        if self.milestones is not None:
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                raise ValueError("milestones run on one process, and this training runs on several")
            milestone_at = _milestones.steps(planned, self.milestones)
            each = None
            what = f"the adapter saved apart in {output}/{_milestones.FOLDER}/step-<N>"
            why = f"every {self.milestones:g} of the {planned} planned steps, and the last"
            if self.milestone_eval is not None:
                milestone_rows = _milestones.rows(self.milestone_eval)
                for row in milestone_rows:
                    _milestones.prompt(tokenizer, row)
                milestone_scorer = load_reward(self.milestone_scorer, "milestone_scorer")
                groups = sorted({row.get("group") or _milestones.GROUP for row in milestone_rows})
                what += f", and {len(milestone_rows):,} rows answered and scored ({', '.join(groups)})"
                if seconds_per_step is not None and model._meepsquared.get("engine") is not None:
                    import tempfile

                    from meepsquared import _save

                    with tempfile.TemporaryDirectory(prefix="meepsquared-timing-") as tmp:
                        _save.save(peft_model, tmp, save_method="lora")
                        began = time.perf_counter()
                        _milestones.engine_answers(model._meepsquared["engine"], peft_model, tokenizer,
                                                   milestone_rows, tmp)
                        each = time.perf_counter() - began
                    generation_why = f"{len(milestone_rows)} rows answered once by the shared vLLM, timed"
                elif seconds_per_step is not None:
                    each, generation_why = _milestones.time_generation(peft_model, tokenizer, milestone_rows)
                if seconds_per_step is not None:
                    milestone_seconds = each * len(milestone_at)
                    estimated += milestone_seconds
                    what += f"; each about {_sessions.duration_text(each)} of generation"
                    why += f"; the generation: {generation_why}; the scorer's time is not counted"
            decisions.append(Decision("milestones", "milestones",
                                      {"steps": milestone_at, "generation_seconds": each},
                                      f"{len(milestone_at)} at steps {_steps_text(milestone_at)}: {what}", why))
        else:
            decisions.append(Decision("milestones", "milestones", None, "none", "no milestones given"))
        duration = "not timed"
        if seconds_per_step is not None:
            duration = (f"about {seconds_per_step:.3g} s per step, {_sessions.duration_text(estimated)} for "
                        f"{planned} steps")
            if evaluations:
                duration += f" with {evaluations} evaluations ({_sessions.duration_text(eval_seconds)} of it)"
            if milestone_seconds:
                duration += (f" and {len(milestone_at)} milestones ({_sessions.duration_text(milestone_seconds)} "
                             "of it)")
        decisions.append(Decision("duration", "time", seconds_per_step, duration, time_why))

        # The memory, measured over the timed steps and the evaluation pass
        peak_gpu = peak_host = request = None
        if timing is None:
            decisions.append(Decision("memory", "memory", None, "not measured",
                                      f"measured over the timed steps, and none ran: {time_why}"))
        else:
            peak_gpu, peak_host = timing.peak_gpu_bytes, timing.peak_host_bytes
            request, request_why = _sessions.memory_request(timing)
            gpu_text = f"peak {_gib(peak_gpu)} on the GPU, " if peak_gpu is not None else ""
            held = {True: "; the host's figure counts the GPU's allocations",
                    False: "; the host's figure does not count the GPU's allocations"}.get(timing.host_holds_gpu, "")
            decisions.append(Decision(
                "memory", "memory", {"peak_gpu_bytes": peak_gpu, "peak_host_bytes": peak_host,
                                     "memory_request_bytes": request},
                f"{gpu_text}{_gib(peak_host)} on the host; a job should ask for {request // 2**30} GiB ({request_why})",
                "the GPU: the most PyTorch reserved over the timed steps" + (" and the evaluation pass"
                                                                            if evaluations else "")
                + "; the host: the process's peak resident memory since it started, loading included" + held))

        device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "the CPU"
        plan = Plan(model=self.model if isinstance(self.model, str) else str(model.config._name_or_path),
                    task=self.task, method=self.method, device=device_name, free_bytes=free_bytes,
                    decisions=_ordered(decisions), seconds_per_step=seconds_per_step, estimated_seconds=estimated,
                    peak_gpu_bytes=peak_gpu, peak_host_bytes=peak_host, memory_request_bytes=request)
        self._prepared = _Prepared(peft_model, tokenizer, trainer, plan, output, milestones=milestone_at,
                                   milestone_rows=milestone_rows, milestone_scorer=milestone_scorer)
        return self._prepared


# The most timings of the steps a plan runs while it looks for the fewest layers to recompute
TIMING_ATTEMPTS = 10

# The order of the plan's lines: how the model runs, then how it learns, then what it learns from
ORDER = ["loading", "allocator_cap", "weights", "attention", "padding_free", "liger", "fused_lora", "compile_layers",
         "checkpointed_layers", "selective_layers", "allocator", "vllm", "tokens_per_update", "effective_batch",
         "micro_batch",
         "accumulation", "order", "rank",
         "alpha", "dropout", "targets", "experts", "optimizer", "learning_rate", "schedule", "warmup_steps",
         "max_grad_norm", "weight_decay", "length", "duration", "memory", "evaluation", "milestones", "milestone_engine",
         "data_checks",
         "source_weights", "max_length", "answers_only",
         "generations",
         "max_completion_length", "temperature", "top_p", "top_k", "chat_template_kwargs", "data", "report_to",
         "output"]


def _steps_text(steps: list) -> str:
    """Steps as a short list: all of them up to six, else the first three, an ellipsis and the last."""
    shown = steps if len(steps) <= 6 else steps[:3] + ["..."] + steps[-1:]
    return ", ".join(f"{s:,}" if isinstance(s, int) else s for s in shown)


def _ordered(decisions: list) -> list:
    unknown = [d.name for d in decisions if d.name not in ORDER]
    if unknown:
        raise RuntimeError(f"the plan has no place for the decisions {unknown}; add them to ORDER in meepsquared.job")
    return sorted(decisions, key=lambda d: ORDER.index(d.name))


def _default(config_cls, name: str):
    """The default a dataclass config (TRL's SFTConfig or GRPOConfig) gives the field."""
    return config_cls.__dataclass_fields__[name].default


def _loss_on(answers: bool) -> str:
    return "the answers only" if answers else "every token"


def _on_off(on: bool) -> str:
    return "on" if on else "off"


def _liger_text(args) -> str:
    if not args.use_liger_kernel:
        return "off"
    config = args.liger_kernel_config or {}
    if config.get("rms_norm") is False:
        return "the fused loss only, as the layers are compiled"
    without = []
    if config.get("swiglu") is False:
        without.append("its SwiGLU: the fused LoRA MLP runs instead")
    if config.get("rope") is False:
        without.append("its rotary embedding: it writes over outputs selective recomputation keeps")
    return "kernels and fused loss" + "".join(f", without {part}" for part in without)


def _timed_steps_fitting(trainer, peft_model, shape, accumulation: int, timed: int, eval_shape, layout, figures,
                         layer_count: int, device, layout_given: bool, margin: Optional[int]):
    """Time the steps on the largest micro-batch (meepsquared._sessions.time_steps) and check their real peak
    against the memory.

    A peak over the budget (meepsquared._checkpointing.peak_budget), or an out-of-memory error, puts more layers to
    recompute whole and times again, then looks for the fewest that fit: a
    peak over the budget first adds as many layers as the overshoot takes at
    one layer's measured activations; a step that ran out of memory tries
    halfway to the most layers known to fit, or to every layer; once a
    layout fits, the layouts between it and the last that did not are
    timed the same way, until none is left between. A layout given by hand
    is not changed: the error says what happened. Returns (timing, layout,
    text), the text saying what was corrected, "" when nothing.
    """
    import torch

    from meepsquared import _checkpointing, _sessions

    budget = _checkpointing.peak_budget(device, margin) if margin is not None else None
    corrected = ""
    failed = None  # the most layers recomputed whole that did not fit
    fitting = None  # (layers recomputed whole, timing, layout) of the fewest that fit
    for attempt in range(TIMING_ATTEMPTS):
        try:
            timing = _sessions.time_steps(trainer, shape, accumulation, timed, eval_shape)
            peak = timing.peak_gpu_allocated_bytes
            if budget is None or peak is None or peak <= budget:
                if fitting is None or layout.whole < fitting[0]:
                    fitting = (layout.whole, timing, layout)
                if failed is None or layout.whole - failed <= 1:
                    break
                found = f"{layout.text(layer_count)} fit, the timed steps peaking at {_gib(peak)}"
                whole = (failed + layout.whole) // 2
            else:
                found = f"the timed steps peaked at {_gib(peak)}, {_gib(peak - budget)} over the budget of {_gib(budget)}"
                failed = layout.whole
                more = math.ceil((peak - budget) / figures.per_layer) if figures is not None and figures.per_layer > 0 else layer_count
                whole = min(failed + more, fitting[0] - 1 if fitting is not None else layer_count)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            found = f"the timed steps ran out of memory on {shape.text()}"
            failed = layout.whole
            whole = (failed + (fitting[0] if fitting is not None else layer_count) + 1) // 2
        if layout_given:
            raise RuntimeError(f"{found}, with the layout given by hand ({layout.text(layer_count)}); checkpoint more "
                               "layers, lower the micro-batch or recipe.max_length, or use method='qlora'")
        if fitting is None and failed >= layer_count:
            raise RuntimeError(f"{found}, with every layer recomputed whole: the micro-batch ({shape.text()}) does "
                               "not fit; lower recipe.max_length or the batch, or use method='qlora'")
        whole = max(whole, failed + 1)
        if fitting is not None and whole >= fitting[0]:
            break
        layout = _checkpointing.Recompute(whole, min(layout.selective, layer_count - whole))
        _checkpointing.set_checkpointed(peft_model, layout)
        peft_model._meepsquared_gradient_checkpointing = layout
        peft_model._meepsquared_checkpointed_layers = layout
        corrected += ("; " if corrected else "") + f"{found}: {layout.text(layer_count)} recompute, timed again"
    if fitting is None:
        raise RuntimeError(f"{corrected}; the steps still do not fit; lower recipe.max_length or the batch, or use "
                           "method='qlora'")
    whole, timing, layout = fitting
    _checkpointing.set_checkpointed(peft_model, layout)
    peft_model._meepsquared_gradient_checkpointing = layout
    peft_model._meepsquared_checkpointed_layers = layout
    return timing, layout, corrected


def _recapped(decision: Decision, cap: int) -> Decision:
    """The plan's allocator cap line after the cap was set again when the measurement started."""
    return dataclasses.replace(decision, value=cap, text=f"{cap / 2**30:.1f} GiB",
                               reason="what this process holds plus the system's available memory less 12 GiB, set "
                                      "when the measurement starts, after the data is tokenized: 4 GiB for the page "
                                      "cache and the machine's daemons, 8 GiB for what grows after it in the job and "
                                      "on the machine; an overflow is this process's error, not the machine's")


def _measure(model, shape) -> tuple[int, int, int, int]:
    """(held with every layer checkpointed, one layer's activations, free, total) in bytes for a batch of `shape`
    (an meepsquared._checkpointing.Shape), on the model's GPU."""
    from meepsquared import _checkpointing

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(base.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"auto mode measures the GPU's memory to choose the batch, the micro-batch and the "
                           f"checkpointed layers, and the model is on {device}; set Recipe(tokens_per_update=...) "
                           "or Recipe(effective_batch=...), and Overrides(micro_batch=..., checkpointed_layers=...)")
    every, per_layer = _checkpointing.measure(model, shape)
    free, total = _checkpointing.free_memory(device)
    return every, per_layer, free, total


def _measure_figures(model, shape, every: int, per_layer: int, compiled: bool):
    """One layer's three memory figures and three times for a batch of `shape` (an meepsquared._checkpointing.Figures),
    given the two figures _measure took already."""
    from meepsquared import _checkpointing, _selective

    refusal = _selective.refusal(model, compiled)
    per_selective = _checkpointing.measure_selective(model, shape) if refusal is None else None
    settings = ((_checkpointing.NONE, _checkpointing.SELECTIVE, _checkpointing.WHOLE) if refusal is None
                else (_checkpointing.NONE, _checkpointing.WHOLE))
    times = _checkpointing.time_layer(model, shape, settings)
    return _checkpointing.Figures(shape, every, per_layer, per_selective, times, len(_checkpointing.layers(model)))
