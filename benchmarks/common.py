# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""What the benchmarks share: the data, a clock per step, and the result records."""

import contextlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

NO_ROBOTS = "HuggingFaceH4/no_robots"


def chat_dataset(source: str, rows, alternating: bool = False):
    """Conversations as {"messages": [...]}: a Hugging Face dataset id or a Parquet file with a messages column.

    rows: the first this many, or None for every usable one.
    alternating: only conversations whose turns go user, assistant, user, ...,
    for chat templates that refuse any other order.
    """
    from datasets import load_dataset

    data = (load_dataset("parquet", data_files=source, split="train") if source.endswith(".parquet")
            else load_dataset(source, split="train"))
    if alternating:
        data = data.filter(lambda r: len(r["messages"]) % 2 == 0 and [m["role"] for m in r["messages"]]
                           == ["user", "assistant"] * (len(r["messages"]) // 2))
    if rows is None:
        rows = len(data)
    if len(data) < rows:
        raise ValueError(f"{source} has {len(data)} usable rows; the run needs {rows}")
    data = data.select(range(rows))
    return data.map(lambda r: {"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]]},
                    remove_columns=data.column_names)


SOURCES = ("huggingface", "thinkube")


def setting(text: str):
    """auto, memory, selective, true, false, a number of layers or a fraction, as get_peft_model takes it."""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text in ("auto", "memory", "selective"):
        return text
    return float(text) if "." in text else int(text)


def model_location(model_id: str, source: str) -> str:
    """Where to load the model from: its Hugging Face id, or on Thinkube the folder of its mirror in the cluster."""
    if source == "huggingface":
        return model_id
    if source == "thinkube":
        import thinkube_models

        return thinkube_models.mirrored_model_path(model_id)
    raise ValueError(f"source must be one of {SOURCES}, not {source!r}")


def environment() -> dict:
    import importlib.metadata as md

    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "liger-kernel", "bitsandbytes", "flash-attn",
                    "fla-core", "vllm", "unsloth", "meepsquared"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            pass
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    # pip records the commit of a package installed from a repository
    commits = {}
    for package in versions:
        origin = md.distribution(package).read_text("direct_url.json")
        commit = json.loads(origin).get("vcs_info", {}).get("commit_id") if origin else None
        if commit:
            commits[package] = commit
    return {"gpus": gpus, "commits": commits, "machine": platform.machine(), "python": platform.python_version(), "versions": versions}


class SystemMemory:
    """The machine's used memory (MemTotal - MemAvailable), sampled every half second from import on.

    On a GPU that shares the system's memory (a GB10) this is the memory a
    training really takes: PyTorch's peak counts only its GPU tensors, not
    activations an engine offloads to CPU memory, which there is the same
    memory. Nothing else should run on the machine during a measurement.
    """

    # Where the used memory is, read at the peak: the process's resident memory,
    # and the system's anonymous, shared, mapped and file-cache memory
    PARTS = ("AnonPages", "Shmem", "Mapped", "Cached", "Mlocked")

    def __init__(self, interval: float = 0.5):
        self.start_gib = self.peak_gib = self.read()
        self.total_gib = self._meminfo()["MemTotal"]
        self.start_parts = self.at_peak = self._parts()
        thread = threading.Thread(target=self._sample, args=(interval,), daemon=True)
        thread.start()

    @staticmethod
    def _meminfo() -> dict:
        with open("/proc/meminfo") as f:
            return {line.split(":")[0]: int(line.split()[1]) / 2**20 for line in f}

    def read(self) -> float:
        info = self._meminfo()
        return info["MemTotal"] - info["MemAvailable"]

    def _parts(self) -> dict:
        info = self._meminfo()
        parts = {name: round(info.get(name, 0.0), 1) for name in self.PARTS}
        with open("/proc/self/status") as f:
            rss = next((int(line.split()[1]) for line in f if line.startswith("VmRSS:")), 0)
        parts["process_rss"] = round(rss / 2**20, 1)
        return parts

    def _sample(self, interval: float) -> None:
        while True:
            used = self.read()
            if used > self.peak_gib:
                self.peak_gib, self.at_peak = used, self._parts()
            time.sleep(interval)

    def summary(self) -> dict:
        return {"total": round(self.total_gib, 1), "used_at_start": round(self.start_gib, 1),
                "peak_used": round(self.peak_gib, 1), "peak_added": round(self.peak_gib - self.start_gib, 1),
                "parts_at_start": self.start_parts, "parts_at_peak": self.at_peak}


SYSTEM_MEMORY = SystemMemory()


def memory() -> dict:
    """What a run's memory was: PyTorch's allocated and reserved peaks per GPU, the machine's used memory, the allocator."""
    return {
        "peak_reserved_gib": [round(torch.cuda.max_memory_reserved(i) / 2**30, 1) for i in range(torch.cuda.device_count())],
        "system_gib": SYSTEM_MEMORY.summary(),
        "allocator": os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or "PyTorch's default",
    }


def record(path: str, result: dict) -> None:
    """One result line, with the command, the environment and the memory, appended to a JSONL file."""
    line = {**result, "memory": memory(), "command": " ".join(sys.argv), "environment": environment(),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print("RESULT " + json.dumps({**result, "memory": line["memory"]}), flush=True)
    if path and int(os.environ.get("RANK", 0)) == 0:
        with Path(path).open("a") as f:
            f.write(json.dumps(line) + "\n")


class QualityClock(TrainerCallback):
    """The held-out loss against training time, and a limit on training time.

    Every `every` steps the Trainer is asked to evaluate; each point records
    the step, the training seconds so far and the held-out loss. Time spent
    evaluating is not training time: it is measured and left out. With
    `minutes`, training stops once that much training time has passed.
    """

    def __init__(self, every=None, minutes=None):
        self.every, self.limit = every, minutes * 60 if minutes else None
        self.points, self.evaluating = [], 0.0
        self.eval_started = None

    def on_train_begin(self, args, state, control, **kw):
        self.start = time.perf_counter()

    def training_seconds(self) -> float:
        return time.perf_counter() - self.start - self.evaluating

    def on_step_end(self, args, state, control, **kw):
        if self.limit is not None and self.training_seconds() >= self.limit:
            control.should_training_stop = True
        if self.every and (state.global_step % self.every == 0 or control.should_training_stop):
            control.should_evaluate = True
            self.eval_started = time.perf_counter()
            self.eval_step, self.eval_at = state.global_step, self.training_seconds()

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if self.eval_started is None:
            return
        self.evaluating += time.perf_counter() - self.eval_started
        self.eval_started = None
        self.points.append({"step": self.eval_step, "training_seconds": round(self.eval_at, 1),
                            "eval_loss": round(metrics["eval_loss"], 4)})


class Clock(TrainerCallback):
    """The time and the tokens seen at each step; a line per step to `progress`, if given."""

    def __init__(self, progress: str | None = None):
        self.marks, self.progress = {}, progress

    def on_train_begin(self, args, state, control, **kw):
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.marks[state.global_step] = (now, state.num_input_tokens_seen)
        if self.progress and int(os.environ.get("RANK", 0)) == 0:
            took = now - self.marks.get(state.global_step - 1, (self.start, 0))[0]
            with open(self.progress, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} step {state.global_step}/{args.max_steps}: {took:.2f} s\n")

    def by_step(self) -> list[float]:
        """The seconds each step took, from the first; the warm-up steps show here, the rate leaves them out."""
        times = [self.start] + [self.marks[s][0] for s in sorted(self.marks)]
        return [round(b - a, 3) for a, b in zip(times, times[1:])]

    def rate(self, first: int, last: int) -> tuple[float, float]:
        """Tokens per second and seconds per step from step `first` to step `last`."""
        (t0, n0), (t1, n1) = self.marks[first], self.marks[last]
        return (n1 - n0) / (t1 - t0), (t1 - t0) / (last - first)


class Profile(TrainerCallback):
    """PyTorch's profiler over steps `first` to `first + steps - 1`; writes the GPU time per operation to `path`."""

    def __init__(self, path: str, first: int, steps: int = 3):
        self.path, self.first, self.last, self.profiler = path, first, first + steps - 1, None

    def on_step_begin(self, args, state, control, **kw):
        if state.global_step + 1 == self.first:
            from torch.profiler import ProfilerActivity, profile

            self.profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kw):
        if self.profiler is not None and state.global_step == self.last:
            torch.cuda.synchronize()
            self.profiler.__exit__(None, None, None)
            table = self.profiler.key_averages().table(sort_by="self_device_time_total", row_limit=60, max_name_column_width=90)
            Path(self.path).write_text(table)
            self.profiler = None


def peak_gib() -> list:
    return [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]


class PeakSnapshot(TrainerCallback):
    """PyTorch's allocation history written to path each time a step sets a new allocated peak.

    PyTorch keeps only the latest allocation events, so a snapshot taken at the
    end of a long run may no longer hold the step that set the peak; written
    right after that step, it does. Open the file at pytorch.org/memory_viz.
    """

    def __init__(self, path: str):
        self.path, self.peak = path, 0

    def on_step_end(self, args, state, control, **kw):
        peak = torch.cuda.max_memory_allocated()
        if peak > self.peak:
            self.peak = peak
            torch.cuda.memory._dump_snapshot(self.path)


@contextlib.contextmanager
def memory_snapshot(path: str | None, trainer=None, max_entries: int = 1_000_000):
    """PyTorch's record of every allocation while the block runs; the trainer gets a PeakSnapshot to path.

    Nothing is recorded without a path.
    """
    if not path:
        yield
        return
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    if trainer is not None:
        trainer.add_callback(PeakSnapshot(path))
    try:
        yield
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


# The decisions of a meepsquared plan that define the training itself; every arm of a benchmark takes them
TRAINING_DECISIONS = ("rank", "alpha", "dropout", "targets", "optimizer", "learning_rate", "schedule", "warmup_steps")
# The batch of a plan: examples per update (auto mode's default), or a token budget (Recipe(tokens_per_update=...))
BATCH_DECISIONS = ("effective_batch", "tokens_per_update")
# The cosine schedule of a meepsquared plan ends at this share of the learning rate (meepsquared.job.COSINE_FLOOR)
COSINE_FLOOR = 0.1


def training_decisions(path: str) -> dict:
    """The training decisions of a meepsquared plan file (plan.py), by name, with its batch: effective_batch or
    tokens_per_update, whichever the plan has. A missing decision is an error that names it."""
    with open(path) as f:
        plan = json.load(f)
    values = {d["name"]: d["value"] for d in plan["decisions"]}
    missing = [name for name in TRAINING_DECISIONS if name not in values]
    if missing:
        raise SystemExit(f"{path}: the plan has no {missing}")
    batch = [name for name in BATCH_DECISIONS if name in values]
    if len(batch) != 1:
        raise SystemExit(f"{path}: a plan has one of {list(BATCH_DECISIONS)} as its batch; this one has {batch}")
    return {name: values[name] for name in TRAINING_DECISIONS + tuple(batch)}


def plan_decision(path: str, name: str):
    """One decision of a meepsquared plan file, by name; a missing one is an error that names it."""
    with open(path) as f:
        values = {d["name"]: d["value"] for d in json.load(f)["decisions"]}
    if name not in values:
        raise SystemExit(f"{path}: the plan has no {name}")
    return values[name]


def schedule(training: dict) -> dict:
    """The Trainer's arguments for a plan's schedule: constant after the warm-up, or cosine down to COSINE_FLOOR."""
    if training["schedule"] == "constant":
        return {"lr_scheduler_type": "constant_with_warmup" if training["warmup_steps"] else "constant"}
    if training["schedule"] == "cosine":
        return {"lr_scheduler_type": "cosine_with_min_lr", "lr_scheduler_kwargs": {"min_lr_rate": COSINE_FLOOR}}
    raise SystemExit(f"schedule {training['schedule']!r}: the benchmarks take constant or cosine")
