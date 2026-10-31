# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Gradient checkpointing on as many layers as the memory needs, whole or selective.

A layer trains in one of three ways. It keeps every activation of its forward
pass for the backward pass: the fastest, the most memory. It recomputes its
whole forward in the backward pass and keeps only its input: about a third
more compute, the least memory. Or it recomputes selectively: it keeps the
outputs of its large products (the matrix multiplications and the attention)
and recomputes the cheap operations around them (see meepsquared._selective), for
memory between the two at a fraction of the whole recompute's cost.

get_peft_model(use_gradient_checkpointing=...) takes True (every layer
whole), False (none), a number of layers, a fraction of them, "memory" (every
layer, the least memory), "selective" (every layer selective), or "auto": one
layer is measured on the largest micro-batch the training runs (see
largest_batch), where meepsquared lays out the batches (a TrainingJob, see
meepsquared._batches); on a trainer of the caller's own, whose order meepsquared does
not set, on the data's longest examples (see longest_examples), when the
trainer is built. The measure is the memory the layer holds under each of
the three ways and the time of its forward and backward pass under each,
and the cheapest layout that fits the free memory is chosen (see choose), keeping
a margin free for the loss, the optimizer and the backward pass's own buffers:
MARGIN of the GPU's memory, or, on a GPU that shares the system's memory, of
the memory free for the training (see margin). There the free memory is the
system's MemAvailable, within the allocator's cap.
"""

import dataclasses
import gc
import heapq
import math
import os
import statistics
import time
from typing import Optional

import torch
from torch.utils._pytree import tree_map

from meepsquared import _selective

# The share of the GPU's memory "auto" leaves unused by activations
MARGIN = 0.15
# On a GPU that shares the system's memory the margin is a share of the memory free for the training, and at least
# this: a step's own buffers do not shrink with it (a step of Qwen3.6-35B-A3B in BF16 took 2.5 GiB over its
# measurement with 10 GiB free)
MARGIN_MIN_SHARED = 4 * 2**30
# The settings that measure: "auto", and Unsloth's name for its own checkpointing
AUTO = ("auto", "unsloth")
# The three ways a layer trains, as the measurements name them
NONE, SELECTIVE, WHOLE = "none", "selective", "whole"


@dataclasses.dataclass(frozen=True)
class Recompute:
    """How many layers recompute in the backward pass, and how.

    The first `whole` layers recompute their whole forward; the next
    `selective` layers recompute selectively; the remaining layers keep every
    activation.
    """

    whole: int = 0
    selective: int = 0

    @property
    def checkpointed(self) -> int:
        return self.whole + self.selective

    def kept(self, total: int) -> int:
        """The layers that keep every activation."""
        return total - self.checkpointed

    def text(self, total: int) -> str:
        return f"{self.whole} of {total} layers" + (f", {self.selective} selective" if self.selective else "")


@dataclasses.dataclass(frozen=True)
class Shape:
    """The batch a measurement runs.

    `rows` rows of `length` tokens; or, when `lengths` is given, a padding-free
    batch: one row holding examples of those lengths end to end, then
    `padding` tokens, with position ids that restart at each example and at
    the padding, as TRL's collator lays them.
    """

    rows: int
    length: int
    lengths: Optional[tuple] = None
    padding: int = 0

    @property
    def tokens(self) -> int:
        return self.rows * self.length

    def inputs(self, device) -> dict:
        ids = torch.zeros((self.rows, self.length), dtype=torch.long, device=device)
        if self.lengths is None:
            return {"input_ids": ids}
        pieces = [torch.arange(n, device=device) for n in self.lengths + ((self.padding,) if self.padding else ())]
        return {"input_ids": ids, "position_ids": torch.cat(pieces).unsqueeze(0)}

    def text(self) -> str:
        if self.lengths is None:
            return f"{self.rows} x {self.length} tokens"
        return f"{self.tokens:,} tokens ({len(self.lengths)} examples, end to end)"


def layers(model) -> list:
    """The model's checkpointable layers, in order: its decoder's. A multimodal model's vision and audio layers do
    not run on text, so they are not counted, measured or laid out."""
    from transformers.modeling_layers import GradientCheckpointingLayer

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = base.get_decoder() if hasattr(base, "get_decoder") else base
    return [module for module in decoder.modules() if isinstance(module, GradientCheckpointingLayer)]


def count(setting, total: int) -> int:
    """How many of `total` layers a setting checkpoints: all for True, "memory", "selective" and "auto"; none for
    False; a number, or a fraction."""
    if setting is True or setting in ("memory", SELECTIVE) or setting in AUTO:
        return total
    if setting is False or setting is None:
        return 0
    if isinstance(setting, Recompute):
        return setting.checkpointed
    if isinstance(setting, bool) or not isinstance(setting, (int, float)):
        raise ValueError(f"use_gradient_checkpointing must be 'auto', 'memory', 'selective', True, False, a number of "
                         f"layers or a fraction of them, not {setting!r}")
    if isinstance(setting, float) and 0 <= setting <= 1:
        return round(setting * total)
    if setting < 0 or setting != int(setting):
        raise ValueError(f"use_gradient_checkpointing={setting!r}: a number of layers is a whole number")
    return min(int(setting), total)


def recompute(setting, total: int) -> Recompute:
    """The layout a setting gives `total` layers: "selective" recomputes every layer selectively, the others whole."""
    if isinstance(setting, Recompute):
        return setting
    if setting == SELECTIVE:
        return Recompute(0, total)
    return Recompute(count(setting, total), 0)


def set_checkpointed(model, checkpointed) -> int:
    """Lay out the layers as `checkpointed` says: a number of layers recomputed whole, or a Recompute. Returns the
    layer count.

    Transformers gives every layer the whole-recompute checkpoint function; the
    selective layers get meepsquared's, with the policy for this model.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layout = checkpointed if isinstance(checkpointed, Recompute) else Recompute(checkpointed, 0)
    all_layers = layers(base)
    if layout.checkpointed > 0:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    elif getattr(base, "is_gradient_checkpointing", False):
        base.gradient_checkpointing_disable()
    selective = _selective.checkpoint_function(model) if layout.selective else None
    for index, layer in enumerate(all_layers):
        layer.gradient_checkpointing = index < layout.checkpointed
        layer._meepsquared_selective = layout.whole <= index < layout.checkpointed
        if layer._meepsquared_selective:
            layer._gradient_checkpointing_func = selective
    return len(all_layers)


def kinds(model) -> list:
    """How each layer trains now: "whole", "selective" or "none"."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    def kind(layer) -> str:
        if not layer.gradient_checkpointing:
            return NONE
        function = layer.__dict__.get("_gradient_checkpointing_func")
        # Transformers' gradient_checkpointing_enable gives every layer the whole-recompute function again
        selective = isinstance(function, _selective.SelectiveCheckpoint)
        return SELECTIVE if layer.__dict__.get("_meepsquared_selective", False) and selective else WHOLE

    return [kind(layer) for layer in layers(base)]


def selective_possible(model) -> bool:
    """Whether the model's setting puts a selective layer in training: "selective", or a Recompute with some, set
    or measured. "auto" before its measurement has none: the trainer patch decides again once it has measured."""
    setting = getattr(model, "_meepsquared_gradient_checkpointing", None)
    measured = getattr(model, "_meepsquared_checkpointed_layers", None)
    if isinstance(measured, Recompute):
        return measured.selective > 0
    if isinstance(setting, Recompute):
        return setting.selective > 0
    return setting == SELECTIVE


def batch_shapes(batches, lengths, padding_free: bool, pad_to_multiple_of: Optional[int] = None) -> list:
    """The Shape of each of `batches` (lists of example indices), empty ones left out.

    `lengths` are the token counts of the tokenized examples, each cut to
    max_length as the trainer cuts them. A padding-free micro-batch holds its
    examples end to end in one row; a padded one is a row per example, each of
    the longest example's length. When the trainer pads to a multiple, the row
    (padding-free: the whole batch) is rounded up to it, with the padding as one
    more example of position 0 to match the collator's position ids.
    """

    def rounded(n: int) -> int:
        return n if not pad_to_multiple_of else math.ceil(n / pad_to_multiple_of) * pad_to_multiple_of

    def shape(batch) -> Shape:
        held = sorted((lengths[index] for index in batch), reverse=True)
        if padding_free:
            total = rounded(sum(held))
            return Shape(1, total, tuple(held), total - sum(held))
        return Shape(len(held), rounded(held[0]))

    shapes = [shape(batch) for batch in batches if len(batch)]
    if not shapes:
        raise ValueError("the training data has no examples")
    return shapes


def largest_batch(batches, lengths, padding_free: bool, pad_to_multiple_of: Optional[int] = None) -> Shape:
    """The largest of `batches` (see batch_shapes), as the batch to measure: the one of the most tokens."""
    return max(batch_shapes(batches, lengths, padding_free, pad_to_multiple_of), key=lambda s: s.tokens)


def median_batch(shapes: list) -> Shape:
    """The micro-batch of the median tokens among `shapes`: the size most steps of the training are near."""
    ordered = sorted(shapes, key=lambda s: s.tokens)
    return ordered[(len(ordered) - 1) // 2]


def mean_tokens(shapes: list) -> float:
    """The mean tokens of a micro-batch among `shapes`."""
    return sum(s.tokens for s in shapes) / len(shapes)


def longest_examples(lengths, micro: int, max_length: int, padding_free: bool,
                     pad_to_multiple_of: Optional[int] = None) -> Shape:
    """A micro-batch of the data's `micro` longest examples, each cut to max_length: the largest any order of the
    data can produce, the bound where the order of the batches is not known ahead."""
    cut = [min(int(n), max_length) for n in lengths]
    return largest_batch([heapq.nlargest(micro, range(len(cut)), key=cut.__getitem__)], cut, padding_free,
                         pad_to_multiple_of)


def _held(trunk, base, layout: Recompute, shape: Shape) -> int:
    """What one forward pass of the decoder allocates and keeps, with the layers laid out as `layout`."""
    device = next(base.parameters()).device
    set_checkpointed(base, layout)
    gc.collect()  # garbage freed during the forward would count against what it allocates
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    out = trunk(**shape.inputs(device))
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_allocated(device)
    del out
    return after - before


class _Measuring:
    """The model in training mode with its layers free to be laid out; put back as they were on exit."""

    def __init__(self, model):
        self.base = model.get_base_model() if hasattr(model, "get_base_model") else model
        # The decoder alone: the output layer's logits (tokens by vocabulary, 15 GiB
        # for 32 rows of 1,024 tokens and a 250k vocabulary) hold the same with or
        # without checkpointing, and a training with the fused loss never builds them
        self.trunk = self.base.get_decoder() if hasattr(self.base, "get_decoder") else self.base
        self.layers = layers(self.base)

    def __enter__(self):
        self.state = [(layer.gradient_checkpointing, layer.__dict__.get("_meepsquared_selective", False),
                       layer.__dict__.get("_gradient_checkpointing_func")) for layer in self.layers]
        self.was_training = self.base.training
        self.base.train()
        return self

    def __exit__(self, *exc):
        for layer, (flag, selective, function) in zip(self.layers, self.state):
            layer.gradient_checkpointing = flag
            layer._meepsquared_selective = selective
            if function is not None:
                layer._gradient_checkpointing_func = function
        self.base.train(self.was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def layer_activation_bytes(model, batch: int, length: int) -> int:
    """One layer's activation memory for a batch of `batch` rows of `length` tokens, measured (see measure)."""
    return measure(model, Shape(batch, length))[1]


def measure(model, shape: Shape) -> tuple:
    """(held with every layer recomputed whole, one layer's activations) in bytes for a batch of `shape`.

    The model runs forward twice, in training mode, on a batch of that shape:
    with every layer recomputed whole, and with every layer but one. The first
    is what the forward holds at the least: the input every checkpointed layer
    keeps. The difference is what one layer keeps for its backward pass. The
    layers are left as they were.
    """
    with _Measuring(model) as m:
        total = len(m.layers)
        _held(m.trunk, m.base, Recompute(total), shape)  # the first forward allocates workspaces (cuBLAS, Triton)
        every = _held(m.trunk, m.base, Recompute(total), shape)
        but_one = _held(m.trunk, m.base, Recompute(total - 1), shape)
    return every, but_one - every


def measure_selective(model, shape: Shape) -> int:
    """What one layer keeps when recomputed selectively, in bytes, for a batch of `shape`: the outputs of its large
    products (see meepsquared._selective), measured as the difference between every layer recomputed whole and one of them
    selective."""
    with _Measuring(model) as m:
        total = len(m.layers)
        every = _held(m.trunk, m.base, Recompute(total), shape)
        one_selective = _held(m.trunk, m.base, Recompute(total - 1, 1), shape)
    return one_selective - every


def _detached(value):
    return value.detach() if isinstance(value, torch.Tensor) else value


def time_layer(model, shape: Shape, settings: tuple, repeats: int = 3) -> dict:
    """Seconds of one forward and backward pass of the first decoder layer on a batch of `shape`, per setting.

    The layer's call is captured from one forward pass of the decoder on the
    batch (its hidden states and the arguments the model passes it), then
    replayed by itself under each setting in `settings` ("none", "selective",
    "whole"): once to warm up, then `repeats` times timed, and the median
    counts. The adapter gradients it produces are discarded.
    """
    with _Measuring(model) as m:
        first = m.layers[0]
        device = next(m.base.parameters()).device
        captured = {}

        def capture(module, args, kwargs):
            captured["args"], captured["kwargs"] = args, kwargs

        handle = first.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                m.trunk(**shape.inputs(device))
        finally:
            handle.remove()
        args = list(tree_map(_detached, captured["args"]))
        kwargs = tree_map(_detached, captured["kwargs"])
        args[0] = args[0].requires_grad_(True)
        parameters = [p for p in first.parameters() if p.requires_grad]

        def run() -> None:
            out = first(*args, **kwargs)
            out = out[0] if isinstance(out, tuple) else out
            out.float().sum().backward()
            for p in parameters:
                p.grad = None
            args[0].grad = None

        def timed() -> float:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            run()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return time.perf_counter() - start

        functions = {WHOLE: _selective.whole_checkpoint_function(),
                     SELECTIVE: _selective.checkpoint_function(model) if SELECTIVE in settings else None}
        seconds = {}
        for setting in settings:
            first.gradient_checkpointing = setting != NONE
            first._meepsquared_selective = setting == SELECTIVE
            if setting != NONE:
                first._gradient_checkpointing_func = functions[setting]
            run()
            seconds[setting] = statistics.median(timed() for _ in range(repeats))
    return seconds


def margin(device, free: int, total: int) -> int:
    """The bytes kept free for the loss, the optimizer's step and the passes' own buffers, out of `free`.

    MARGIN of the GPU's memory (`total`). On a GPU that shares the system's
    memory, MARGIN of the memory free for the training (`free`), and at least
    MARGIN_MIN_SHARED: there the
    GPU's memory is the machine's, and the operating system, the page cache
    and the other processes hold much of it, which the allocator's cap
    already keeps from this process (meepsquared._unified_memory). The timed
    steps check the real peak against the budget (peak_budget) either way.
    """
    from meepsquared import _unified_memory

    if device.type == "cuda" and _unified_memory.shares_system_memory(device.index or 0):
        return max(int(MARGIN * free), MARGIN_MIN_SHARED)
    return int(MARGIN * total)


def fits(free: int, margin: int, every: int, per_layer: int) -> bool:
    """Whether a batch trains in `free` minus `margin` (see margin) with every layer checkpointed.

    It then holds every checkpointed layer's input (`every`) and, in the
    backward pass, the activations of the one layer being recomputed.
    """
    return every + per_layer <= free - margin


def layers_to_checkpoint(free: int, margin: int, per_layer: int, layer_count: int) -> int:
    """How many layers to checkpoint so that the others' activations fit in `free` minus the margin."""
    usable = free - margin
    kept = max(0, math.floor(usable / per_layer)) if per_layer > 0 else layer_count
    return max(0, layer_count - kept)


def layout_fits(free: int, margin: int, per_layer: int, per_selective: Optional[int], layer_count: int,
                layout: Recompute) -> bool:
    """Whether the activations a layout keeps fit in `free` minus the margin.

    The layers that keep everything hold `per_layer` each, the selective ones
    `per_selective` each; the inputs the checkpointed layers keep and the
    backward pass's working set are what the batch was already required to fit
    with every layer checkpointed (see fits).
    """
    held = layout.kept(layer_count) * per_layer + layout.selective * (per_selective or 0)
    return held <= free - margin


def fewest_whole(free: int, margin: int, per_layer: int, per_selective: Optional[int], layer_count: int,
                 selective: int) -> int:
    """The fewest layers recomputed whole that fit with `selective` layers recomputed selectively; the rest keep
    everything."""
    for whole in range(0, layer_count - selective + 1):
        if layout_fits(free, margin, per_layer, per_selective, layer_count, Recompute(whole, selective)):
            return whole
    return layer_count - selective


def choose(free: int, margin: int, per_layer: int, per_selective: Optional[int], times: Optional[dict],
           layer_count: int) -> Recompute:
    """The cheapest layout that fits `free` minus the margin.

    The candidates, in the order that breaks a tie: nothing recomputed; the
    fewest layers recomputed selectively, the rest keeping everything; every
    layer selective plus the fewest recomputed whole; and the fewest
    recomputed whole, the rest keeping everything (the rule without selective
    recomputation). With the measured `times` of one layer's forward and
    backward pass per setting, the candidate with the lowest total time wins;
    without them, the first that fits. per_selective None (selective
    recomputation refused on this model) leaves only the first and last
    candidates. When no candidate fits, every layer recomputes whole.
    """
    candidates = [Recompute(0, 0)]
    if per_selective is not None:
        selective = next((s for s in range(1, layer_count + 1)
                          if layout_fits(free, margin, per_layer, per_selective, layer_count, Recompute(0, s))), None)
        if selective is not None:
            candidates.append(Recompute(0, selective))
        whole = next((w for w in range(0, layer_count + 1)
                      if layout_fits(free, margin, per_layer, per_selective, layer_count, Recompute(w, layer_count - w))),
                     layer_count)
        candidates.append(Recompute(whole, layer_count - whole))
    candidates.append(Recompute(layers_to_checkpoint(free, margin, per_layer, layer_count), 0))
    fitting = []
    for candidate in candidates:
        if candidate not in fitting and layout_fits(free, margin, per_layer, per_selective, layer_count, candidate):
            fitting.append(candidate)
    if not fitting:
        return Recompute(layer_count, 0)
    if not times:
        return fitting[0]

    def cost(layout: Recompute) -> float:
        return (layout.kept(layer_count) * times[NONE] + layout.selective * times.get(SELECTIVE, math.inf)
                + layout.whole * times[WHOLE])

    return min(fitting, key=cost)


def meminfo_bytes(text: str) -> tuple[int, int]:
    """(MemAvailable, MemTotal) in bytes from the text of /proc/meminfo."""
    fields = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemAvailable", "MemTotal"):
            fields[name] = int(rest.split()[0]) * 1024
    if len(fields) != 2:
        raise ValueError("/proc/meminfo names no MemAvailable and MemTotal")
    return fields["MemAvailable"], fields["MemTotal"]


def cached_bytes(device) -> int:
    """What PyTorch's allocator holds for this process and nothing uses: free to it, used to everyone else."""
    return torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def free_memory(device) -> tuple[int, int]:
    """(free, total) bytes of the GPU's memory, as this process can use them.

    A GPU that shares the system's memory (an NVIDIA GB10) reports as free
    only the memory nothing holds; the page cache, which the kernel frees on
    demand, counts as used, and a model's files just read fill it. There the
    measure is the system's MemAvailable. A container's memory limit is not
    consulted: on a GB10, what the GPU allocates is not charged to the
    container's cgroup, which counts the page cache instead.

    Either measure leaves out what PyTorch's allocator keeps reserved for
    this process without using it, such as the BF16 weights a model held
    before they were made FP8; that is added, since this process allocates
    from it first.
    """
    from meepsquared import _unified_memory

    if _unified_memory.shares_system_memory(device.index or 0):
        with open("/proc/meminfo") as f:
            free, total = meminfo_bytes(f.read())
    else:
        free, total = torch.cuda.mem_get_info(device)
    free += cached_bytes(device)
    # The allocator's cap (meepsquared._unified_memory on a GPU that shares the system's memory, or set by
    # hand): this process may allocate at most the cap less what it holds, whatever the machine has
    cap = allocator_cap_bytes(device)
    if cap is not None:
        free = min(free, cap - torch.cuda.memory_allocated(device))
    return free, total


def allocator_cap_bytes(device) -> Optional[int]:
    """The cap on PyTorch's allocator for this process on the GPU, in bytes; None without one."""
    fraction = torch.cuda.get_per_process_memory_fraction(device)
    if fraction >= 1.0:
        return None
    return int(fraction * torch.cuda.get_device_properties(device).total_memory)


def peak_budget(device, margin: int) -> Optional[int]:
    """The most this process may hold at the peak of a step: the allocator's cap where set, else the GPU's memory,
    less `margin` (see margin). None off the GPU, where nothing measures the peak."""
    if device.type != "cuda":
        return None
    total = torch.cuda.get_device_properties(device).total_memory
    cap = allocator_cap_bytes(device)
    return (cap if cap is not None else total) - margin


# Bytes of optimizer state per trainable parameter, by the optimizer's name (Transformers' optim names): two 8-bit
# moments for the 8-bit Adam family, one for 8-bit Lion, two 32-bit moments for the others of Adam's kind, one for
# Lion, none for SGD; Adafactor's factored second moment is counted as one 32-bit moment, more than it takes
def optimizer_state_bytes(optimizer: str) -> int:
    name = optimizer.lower()
    if "sgd" in name:
        return 0
    if "lion" in name:
        return 1 if "8bit" in name else 4
    if "8bit" in name:
        return 2
    return 8 if "adafactor" not in name else 4


def trainable_bytes(model, optimizer: str) -> int:
    """What the trainable parameters add to a step beyond the activations: their gradients, in their own dtype, and
    the optimizer's state for them."""
    per_state = optimizer_state_bytes(optimizer)
    return sum(p.numel() * (p.element_size() + per_state) for p in model.parameters() if p.requires_grad)


@dataclasses.dataclass(frozen=True)
class Figures:
    """One layer measured on a batch: the bytes it keeps under each way of training, and the seconds of its forward
    and backward pass under each. selective is None where selective recomputation is refused."""

    shape: Shape
    every: int
    per_layer: int
    per_selective: Optional[int]
    times: Optional[dict]
    layer_count: int

    def text(self, batch: Optional[str] = None) -> str:
        """The three memory figures and the three times, for a plan; `batch` names the batch, else the shape does."""
        gib = [f"{self.per_layer / 2**30:.2f} GiB keeping everything"]
        if self.per_selective is not None:
            gib.append(f"{self.per_selective / 2**30:.2f} selective")
        gib.append(f"{self.every / self.layer_count / 2**30:.2f} recomputed whole")
        text = f"one layer for {batch or self.shape.text()}: {', '.join(gib)}"
        if self.times:
            ms = [f"{self.times[s] * 1000:.0f}" for s in (NONE, SELECTIVE, WHOLE) if s in self.times]
            text += f"; its forward and backward {' / '.join(ms)} ms in that order"
        return text


def measure_figures(model, shape: Shape, compiled: bool = False, refused: Optional[str] = None) -> Figures:
    """One layer's three memory figures and three times on a batch of `shape` (see Figures); `refused` leaves
    selective recomputation out, for that reason."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    every, per_layer = measure(model, shape)
    refusal = refused or _selective.refusal(model, compiled)
    per_selective = measure_selective(model, shape) if refusal is None else None
    settings = (NONE, SELECTIVE, WHOLE) if refusal is None else (NONE, WHOLE)
    times = time_layer(model, shape, settings)
    return Figures(shape, every, per_layer, per_selective, times, len(layers(base)))


def resolve_auto(model, shape: Shape, compiled: bool = False, refused: Optional[str] = None,
                 batch: Optional[str] = None) -> Recompute:
    """The layout "auto" chooses for this batch, measured on the model's GPU (see choose); sets it. `refused` leaves
    selective recomputation out of the choice, for that reason; `batch` names the batch in the line printed, else
    the shape does."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(base.parameters()).device
    figures = measure_figures(model, shape, compiled, refused)
    free, total = free_memory(device)
    layout = choose(free, margin(device, free, total), figures.per_layer, figures.per_selective, figures.times, figures.layer_count)
    set_checkpointed(model, layout)
    if not os.environ.get("MEEPSQUARED_QUIET"):
        refusal = refused or _selective.refusal(model, compiled)
        print(f"meepsquared: checkpointing {layout.text(figures.layer_count)} ({figures.text(batch)}; "
              f"free: {free / 2**30:.0f} of {total / 2**30:.0f} GiB"
              + (f"; selective recomputation not offered: {refusal}" if refusal else "") + ")", flush=True)
    return layout
