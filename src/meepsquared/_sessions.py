# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A training in sessions: stopped when asked, resumed exactly, its progress in a file, its length timed ahead.

- Stop: a TrainerCallback. At the end of each optimizer step it looks for
  the stop file, and for a SIGTERM when asked to catch one; either makes
  the trainer save a checkpoint (Transformers' own: the adapter's weights,
  the optimizer, the scheduler, the RNG states, the trainer's state) and
  stop. Under several processes they agree first, so that all stop at the
  same step. A request at the last step is not a stop: the training ends.
- Status: the status file, a JSON object written whole (a temporary file,
  then renamed over it), so that a reader never sees half of it. Report is
  the TrainerCallback that writes it after each step, log, evaluation and
  checkpoint. The time per step leaves the evaluations out.
- The checkpoints: the latest one is the one a resumed training continues
  from. A training starting afresh is refused where checkpoints of another
  are, as a later resume would continue theirs. A resumed training must
  plan what its first session planned (record_plan, check_same_plan): the
  same micro-batches, the same schedule.
- time_steps: seconds per optimizer step, from a few steps on the largest
  micro-batch, run with an optimizer of their own and the trained
  weights put back after them; seconds per forward pass alone, which an
  evaluation runs; and the peak memory of both on the GPU and the host, from
  which memory_request gives what a job should ask for.
"""

import copy
import dataclasses
import datetime
import json
import math
import os
import signal
import statistics
import time
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback

STATES = ("planning", "training", "stopping", "paused", "completed", "failed")
# The plan of the training's first session, kept in its output folder
PLAN_FILE = "meepsquared-plan.json"
# The decisions that make the micro-batches, the updates and the schedule; a resumed training keeps them
SAME_ON_RESUME = ("tokens_per_update", "effective_batch", "micro_batch", "accumulation", "order", "length",
                  "max_length", "padding_free", "answers_only", "evaluation", "data_checks", "source_weights",
                  "rank", "alpha", "dropout", "targets", "experts", "optimizer", "learning_rate", "schedule",
                  "warmup_steps", "max_grad_norm", "weight_decay", "data")
# The warm-up steps time_steps runs before it times any
TIMING_WARMUP = 1
# The memory a job asks for, over the training's measured peak
MEMORY_MARGIN = 0.10


def main_process() -> bool:
    """Whether this is the process that writes files: process 0 of a launch, or the only one."""
    return int(os.environ.get("RANK", "0")) == 0


def write_json(path: Path, data: dict) -> None:
    """`data` as JSON at `path`, whole: written to a temporary file beside it, then renamed over it."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data))
    os.replace(temporary, path)


class Status:
    """The status file of a training: its state, its step, its last loss, its speed and its last checkpoint."""

    def __init__(self, path):
        self.path = Path(path)
        self.fields = {"state": "planning", "step": 0, "total_steps": 0, "loss": None, "seconds_per_step": None,
                       "eta_seconds": None, "checkpoint": None, "plan": None, "error": None, "eval_loss": None,
                       "eval_step": None, "milestones": None, "milestone_seconds": None, "next_milestone": None}

    def update(self, **fields) -> None:
        unknown = set(fields) - set(self.fields)
        if unknown:
            raise ValueError(f"the status has no fields {sorted(unknown)}")
        if "state" in fields and fields["state"] not in STATES:
            raise ValueError(f"state must be one of {STATES}, not {fields['state']!r}")
        self.fields.update(fields)
        updated = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        write_json(self.path, {**self.fields, "updated": updated})


def mark_failed(path, error: BaseException) -> None:
    """The status file at `path` in state "failed" with the error, keeping the fields written already."""
    path = Path(path)
    status = Status(path)
    if path.exists():
        written = json.loads(path.read_text())
        status.fields.update({key: value for key, value in written.items() if key in status.fields})
    status.update(state="failed", error=f"{type(error).__name__}: {error}")


def _anyone(flag: bool, device) -> bool:
    """Whether any process has `flag` set; the flag itself with one process."""
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return flag
    value = torch.tensor([int(flag)], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


class Stop(TrainerCallback):
    """Stops the training at the end of an optimizer step, with a checkpoint, when the stop file exists or, with
    on_sigterm, once the process had a SIGTERM. close() puts the SIGTERM handler back."""

    def __init__(self, stop_file=None, on_sigterm: bool = False, status: Optional[Status] = None):
        self.stop_file = Path(stop_file) if stop_file is not None else None
        self.status = status
        self.signalled = False
        self.stopped = False
        self._handler_replaced = on_sigterm
        if on_sigterm:
            self._previous = signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum, frame) -> None:
        self.signalled = True

    def close(self) -> None:
        if self._handler_replaced:
            # getsignal's None: the handler was not set from Python; the default one is put back
            signal.signal(signal.SIGTERM, self._previous if self._previous is not None else signal.SIG_DFL)
            self._handler_replaced = False

    def asked(self) -> bool:
        return self.signalled or (self.stop_file is not None and self.stop_file.exists())

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            return
        if _anyone(self.asked(), args.device):
            control.should_save = True
            control.should_training_stop = True
            self.stopped = True
            if self.status is not None:
                self.status.update(state="stopping", step=state.global_step)


class Report(TrainerCallback):
    """Writes the status after each optimizer step, each log and each checkpoint, on the main process.

    seconds_per_step is the mean of this session's steps after its first,
    which also warms up, without the evaluations between them; it and
    eta_seconds (the steps left at that pace, the evaluations not counted) are
    None until there is one. eval_loss and eval_step are the latest
    evaluation's, a resumed training's earlier sessions' too.
    """

    def __init__(self, status: Status):
        self.status = status
        self.ends = []
        self.evaluating = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            earlier = [entry for entry in state.log_history if "eval_loss" in entry]
            latest = {"eval_loss": earlier[-1]["eval_loss"], "eval_step": earlier[-1]["step"]} if earlier else {}
            self.status.update(step=state.global_step, total_steps=state.max_steps, **latest)

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.ends.append(time.perf_counter())
        seconds = eta = None
        if len(self.ends) > 1:
            seconds = (self.ends[-1] - self.ends[0] - self.evaluating) / (len(self.ends) - 1)
            eta = seconds * (state.max_steps - state.global_step)
        self.status.update(step=state.global_step, total_steps=state.max_steps, seconds_per_step=seconds,
                           eta_seconds=eta)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs and "loss" in logs:
            self.status.update(loss=float(logs["loss"]))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.ends:
            # The evaluation runs after the step's end; its time is not the steps'
            self.evaluating += time.perf_counter() - self.ends[-1]
        if metrics and "eval_loss" in metrics:
            self.status.update(eval_loss=float(metrics["eval_loss"]), eval_step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.status.update(checkpoint=checkpoint_folder(args.output_dir, state.global_step))


def checkpoint_folder(output: str, step: int) -> str:
    """The folder of Transformers' checkpoint at `step` in `output`."""
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    return os.path.join(output, f"{PREFIX_CHECKPOINT_DIR}-{step}")


def latest_checkpoint(output: str) -> Optional[str]:
    """The checkpoint of the highest step in `output`; None when there is none."""
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(output) if os.path.isdir(output) else None


def checkpoint_step(checkpoint: str) -> tuple:
    """(step, planned steps) of a checkpoint, from its trainer state."""
    from transformers.trainer import TRAINER_STATE_NAME

    path = os.path.join(checkpoint, TRAINER_STATE_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"resume: the checkpoint {checkpoint} has no {TRAINER_STATE_NAME}")
    state = json.loads(Path(path).read_text())
    return state["global_step"], state["max_steps"]


def to_resume(output: str) -> str:
    """The checkpoint a resumed training continues from: the latest in `output`, which must exist and be short of
    the planned steps."""
    checkpoint = latest_checkpoint(output)
    if checkpoint is None:
        raise FileNotFoundError(f"resume: the output folder {output} holds no checkpoint to resume from")
    step, planned = checkpoint_step(checkpoint)
    if step >= planned:
        raise ValueError(f"resume: the latest checkpoint, {checkpoint}, is at step {step} of {planned}: that "
                         "training is complete")
    return checkpoint


def refuse_earlier_checkpoints(output: str) -> None:
    """Refuse a training that starts afresh in a folder holding another's checkpoints, which a resume would take."""
    checkpoint = latest_checkpoint(output)
    if checkpoint is not None:
        raise ValueError(f"the output folder {output} holds checkpoints of an earlier training (the latest: "
                         f"{checkpoint}); resume it (train(resume=True), meepsquared train --resume), or give another "
                         "output folder, or remove them")


def record_plan(output: str, plan) -> None:
    """The plan of a training's first session, kept in its output folder."""
    if main_process():
        os.makedirs(output, exist_ok=True)
        write_json(Path(output) / PLAN_FILE, plan.to_dict())


def check_same_plan(output: str, plan) -> None:
    """Refuse a resume whose plan makes other micro-batches, updates or schedule than the first session's did."""
    path = Path(output) / PLAN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"resume: {path} is missing: meepsquared writes it when a training starts, and resumes "
                                "only a training it started")
    first = {d["name"]: d["value"] for d in json.loads(path.read_text())["decisions"]}
    now = {d.name: json.loads(json.dumps(d.value)) for d in plan.decisions}
    differ = [f"{name}: {first.get(name)!r} then, {now.get(name)!r} now" for name in SAME_ON_RESUME
              if first.get(name) != now.get(name)]
    if differ:
        raise ValueError(f"resume: the training in {output} was planned otherwise ({'; '.join(differ)}). A resumed "
                         "training continues the same micro-batches and schedule: give the job the values it "
                         "started with, in its recipe or overrides")


@dataclasses.dataclass(frozen=True)
class Timing:
    """What time_steps measured.

    step_seconds: one optimizer step. forward_seconds: one forward pass of the
    evaluation batch without gradients; None without one. peak_gpu_bytes: the
    most memory PyTorch's allocator reserved on the GPU over the steps and the
    evaluation pass; None on the CPU. peak_host_bytes: the most resident
    memory the process has held since it started (ru_maxrss), loading the
    model included. host_holds_gpu: on a GPU that shares the system's memory,
    whether the process's resident memory grew with the GPU's allocations
    during the steps; None on another device.
    """

    step_seconds: float
    forward_seconds: Optional[float]
    peak_gpu_bytes: Optional[int]
    peak_host_bytes: int
    host_holds_gpu: Optional[bool]
    # The most PyTorch had allocated on the GPU over the steps, what the step itself held; None on the CPU
    peak_gpu_allocated_bytes: Optional[int] = None


def resident_bytes() -> int:
    """The process's resident memory now, from /proc/self/status (VmRSS)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status has no VmRSS line")


def peak_resident_bytes() -> int:
    """The most resident memory the process has held since it started: ru_maxrss, in KiB on Linux."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def time_steps(trainer, shape, accumulation: int, steps: int, eval_shape=None) -> Timing:
    """Time and measure `steps` optimizer steps after TIMING_WARMUP, and an evaluation pass (see Timing).

    A step is `accumulation` forward and backward passes of a batch of `shape`
    (an meepsquared._checkpointing.Shape), its loss computed by the trainer's
    compute_loss as the training computes it, the gradient clipped, and an
    optimizer step; its time is the median of `steps`. With `eval_shape`, the
    loss of that batch in evaluation mode without gradients, as an evaluation
    computes it, is timed the same way. The GPU's peak is reset before the
    first step.

    The steps run with an optimizer of their own, of the trainer's kind and
    settings, which is dropped after them. The trained weights are copied
    before and put back after, their gradients cleared, the RNG states and the
    trainer's logged metrics restored, so that the training starts as it would
    without them.
    """
    import torch
    from transformers import Trainer

    from meepsquared import _unified_memory

    model = trainer.model
    args = trainer.args
    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    shared = on_gpu and _unified_memory.shares_system_memory(device.index if device.index is not None else 0)
    trained = [p for p in model.parameters() if p.requires_grad]
    kept = [p.detach().clone() for p in trained]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    optimizer = optimizer_cls(trained, **optimizer_kwargs)
    inputs = shape.inputs(device)
    inputs["labels"] = inputs["input_ids"]
    eval_inputs = eval_shape.inputs(device) if eval_shape is not None else None
    if eval_inputs is not None:
        eval_inputs["labels"] = eval_inputs["input_ids"]
    # The loss is the trainer's own (compute_loss), as the training computes it: TRL's fused LM head or Liger's
    # fused loss never build the full logits. compute_loss also adds to the trainer's logged metrics and token
    # count, which are put back after.
    metrics = copy.deepcopy(getattr(trainer, "_metrics", None))
    total_tokens = getattr(trainer, "_total_train_tokens", None)

    def step() -> None:
        for _ in range(accumulation):
            with trainer.accelerator.autocast():
                loss = trainer.compute_loss(model, dict(inputs))
            (loss / accumulation).backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trained, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def forward() -> None:
        with torch.no_grad(), trainer.accelerator.autocast():
            trainer.compute_loss(model, dict(eval_inputs))

    def timed(run) -> float:
        if on_gpu:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        run()
        if on_gpu:
            torch.cuda.synchronize(device)
        return time.perf_counter() - start

    host_holds_gpu = None
    model.train()
    try:
        if on_gpu:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            reserved_before = torch.cuda.memory_reserved(device)
        if shared:
            resident_before = resident_bytes()
        for _ in range(TIMING_WARMUP):
            step()
        seconds = statistics.median(timed(step) for _ in range(steps))
        forward_seconds = None
        if eval_inputs is not None:
            model.eval()
            forward_seconds = statistics.median(timed(forward) for _ in range(steps))
        peak_gpu = torch.cuda.max_memory_reserved(device) if on_gpu else None
        peak_allocated = torch.cuda.max_memory_allocated(device) if on_gpu else None
        if shared:
            # The GPU's allocations count in the process's resident memory when it grew by at least half of them
            grown = torch.cuda.memory_reserved(device) - reserved_before
            host_holds_gpu = grown > 0 and resident_bytes() - resident_before >= grown / 2
        peak_host = peak_resident_bytes()
    finally:
        with torch.no_grad():
            for parameter, saved in zip(trained, kept):
                parameter.copy_(saved)
                parameter.grad = None
        del optimizer
        if metrics is not None:
            trainer._metrics = metrics
        if total_tokens is not None:
            trainer._total_train_tokens = total_tokens
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(was_training)
        if on_gpu:
            torch.cuda.empty_cache()
    return Timing(seconds, forward_seconds, peak_gpu, peak_host, host_holds_gpu, peak_allocated)


def on_the_line(small_tokens: int, small_seconds: float, large_tokens: int, large_seconds: float,
                tokens: float) -> float:
    """The time of a batch of `tokens`, on the line through two timed batches: a fixed part, and a part per token.

    With the two batches of the same tokens, their time is the larger one's.
    """
    if large_tokens == small_tokens:
        return large_seconds
    per_token = (large_seconds - small_seconds) / (large_tokens - small_tokens)
    return small_seconds + per_token * (tokens - small_tokens)


def memory_request(timing: Timing) -> tuple:
    """(bytes, how): the memory a job running this training should ask for, MEMORY_MARGIN over its peak, rounded up
    to a whole GiB.

    The peak is the host's, plus the GPU's where the GPU takes its memory
    from the system's and the host figure did not count it. A GPU with its
    own memory takes nothing from the job's.
    """
    peak, how = timing.peak_host_bytes, "the host's peak"
    if timing.host_holds_gpu is False:
        peak, how = peak + timing.peak_gpu_bytes, ("the host's peak plus the GPU's: the GPU takes its memory from "
                                                   "the system's, and the process's resident memory does not count it")
    elif timing.host_holds_gpu:
        how = "the host's peak, which counts the GPU's allocations: the GPU takes its memory from the system's"
    elif timing.peak_gpu_bytes is not None:
        how = "the host's peak; the GPU has memory of its own"
    request = math.ceil(peak * (1 + MEMORY_MARGIN) / 2**30) * 2**30
    return request, f"{how}, {MEMORY_MARGIN:.0%} over it, rounded up to a whole GiB"


def duration_text(seconds: float) -> str:
    """A length of time in hours and minutes, minutes, or seconds."""
    if seconds >= 3600:
        hours, rest = divmod(round(seconds / 60), 60)
        return f"{hours} h {rest} min"
    if seconds >= 60:
        return f"{round(seconds / 60)} min"
    return f"{seconds:.0f} s"
