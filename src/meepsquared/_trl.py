# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""TRL's SFTTrainer and GRPOTrainer, made to train a meepsquared model as meepsquared prepared it.

For a model from meepsquared's from_pretrained, the trainer takes these settings
from the model unless its config gives them:

- padding_free (SFT): the examples of a micro-batch laid end to end as one
  sequence, with position ids that restart at each example, so no compute
  goes to padding. from_pretrained decides whether the model can train that
  way (models.padding_free_ok).
- gradient_checkpointing: as get_peft_model's use_gradient_checkpointing set it.
  With some layers checkpointed (a number, a fraction, "selective", or "auto",
  which measures, once the trainer has tokenized the data, what a micro-batch
  of the data's longest examples needs; see measured_shape), meepsquared sets the layers
  and the Trainer is told gradient_checkpointing=False, as it would otherwise
  checkpoint every layer again in train(). Where a layer may be recomputed
  selectively, Liger's rotary embedding is left off: its kernel rotates q and
  k in place, over the projection outputs such a layer keeps.
- use_liger_kernel: as from_pretrained's liger set it. The trainer applies
  Liger's patches, and skips the metrics it would compute from the logits
  that Liger's fused loss never builds. For GRPO, TRL 0.29 computes the
  GRPO loss with Liger's chunked loss; TRL 1.x applies Liger's model kernels
  without a fused loss. With LoRA on parameters (target_parameters, for the
  experts of a mixture of experts) or fused_lora, Liger's SwiGLU is left off:
  it would replace the forward of the module PEFT has wrapped, or the fused
  MLP forward.

One process training a model on one GPU uses that GPU only, never
nn.DataParallel, whatever the number of GPUs it sees.
With several processes (DDP), ddp_find_unused_parameters is False unless given.
Under FSDP, bf16 is False unless given (Accelerate would turn the BF16 weights
to FP32), and bitsandbytes' optimizers are refused (see _fsdp_settings).
A model spread over several GPUs gets its parts' devices again after the
trainer is built, which may have replaced the model's forward (see meepsquared._split).

After the trainer is built, a model with fused_lora gets its fused kernels
again, in place of the SwiGLU MLP forward that Liger set, and a model with
compile_layers gets each decoder layer compiled (see meepsquared._compile).

TRL 1.x refuses padding_free with a max_length and no packing, before its own
data preparation would cut the examples to that length. When padding-free is
on, whether meepsquared turned it on or the config gave it, the trainer is built
without max_length, and the examples are cut to it as they reach the data
collator, keeping their start, as TRL 0.29 cuts them while preparing the
dataset.
"""

import functools
import inspect
import os

from meepsquared import _checkpointing

_GIVEN = "_meepsquared_given"
# Padded batches of a model with compiled layers are padded to a multiple of this
COMPILED_PAD_MULTIPLE = 128


def _cut(example: dict, max_length: int) -> dict:
    """Every per-token field of a tokenized example cut to its first max_length tokens."""
    length = len(example["input_ids"])
    if length <= max_length:
        return example
    return {key: value[:max_length]
            if not isinstance(value, (str, bytes, dict)) and hasattr(value, "__len__") and len(value) == length
            else value
            for key, value in example.items()}


ROPE_KEPT = "liger_kernel_config keeps Liger's rotary embedding, which writes over the outputs a selective layer keeps"


def _rope_off_for_selective(args, given) -> None:
    """Liger's rotary embedding off, as a layer is recomputed selectively; refused when the config keeps it on.

    Its kernel rotates q and k in place, over the projection outputs a
    selectively recomputed layer keeps (see meepsquared._selective). Liger patches
    the model's module, not one layer, so it is off for every layer.
    """
    if not args.use_liger_kernel:
        return
    if "liger_kernel_config" in given:
        if (args.liger_kernel_config or {}).get("rope") is not False:
            raise ValueError("selective recomputation keeps the attention projections' outputs, which "
                             "Liger's rotary embedding writes over; give liger_kernel_config "
                             "{'rope': False}, or a checkpointing setting without it")
        return
    args.liger_kernel_config = {**(args.liger_kernel_config or {}), "rope": False}


def example_lengths(dataset) -> list:
    """The token count of every example of a tokenized dataset, read from the lengths of its input_ids lists in
    the Arrow table, without materializing the ids."""
    import pyarrow.compute as pc

    if "input_ids" not in dataset.column_names:
        raise ValueError("the trainer's dataset has no input_ids column; auto mode reads the examples' lengths "
                         "from the tokenized dataset, or takes the checkpointed layers as a number")
    return pc.list_value_length(dataset.data.column("input_ids")).to_pylist()


def measured_shape(trainer, kind: str, max_length: int) -> tuple:
    """(shape, text): the batch auto mode measures for a trainer of the caller's own, and its name for the line
    printed.

    The trainer orders its batches by its own sampler, which meepsquared does not
    set and does not predict; so the batch is the bound any order can reach, a
    micro-batch of the data's longest examples (see
    meepsquared._checkpointing.longest_examples). A TrainingJob lays out the
    batches itself and measures the largest it trains (see meepsquared._batches).
    An SFT trainer has tokenized its dataset (TRL 0.29 and 1.x both do so in
    __init__), so the lengths are read from it; TRL 0.29 has cut the examples
    to max_length there, TRL 1.x leaves them to the collator (CutToLength), so
    the lengths are cut here. A streamed dataset gives no lengths, and GRPO's
    sequences are prompts plus generated completions: both are measured at
    their bound, the micro-batch by max_length.
    """
    from datasets import Dataset

    args = trainer.args
    micro = args.per_device_train_batch_size
    if kind == "sft" and isinstance(trainer.train_dataset, Dataset):
        shape = _checkpointing.longest_examples(example_lengths(trainer.train_dataset), micro, max_length,
                                                bool(args.padding_free), args.pad_to_multiple_of)
        count = len(shape.lengths) if shape.lengths is not None else shape.rows
        return shape, (f"the data's {count} longest examples, {shape.text()}: the bound, as the trainer orders its "
                       "batches itself")
    why = "GRPO generates its completions as it trains" if kind == "grpo" else "a streamed dataset has no lengths ahead"
    shape = _checkpointing.Shape(micro, max_length)
    return shape, f"{shape.text()}, the bound: {why}"


class CutToLength:
    """A data collator that cuts each example to max_length before collating."""

    def __init__(self, collator, max_length: int):
        self.collator, self.max_length = collator, max_length

    def __call__(self, examples):
        return self.collator([_cut(example, self.max_length) for example in examples])


# bitsandbytes' optimizers, which update plain tensors, not FSDP2's sharded ones
_BNB_OPTIMIZERS = {"adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit", "paged_adamw_32bit", "paged_adamw",
                   "adam_8bit", "paged_lion_8bit", "lion_8bit", "paged_lion_32bit", "ademamix_8bit",
                   "paged_ademamix_8bit", "paged_ademamix_32bit", "rmsprop_bnb_8bit", "rmsprop_bnb"}


def _fsdp_settings(args, given) -> None:
    """A meepsquared model under FSDP: its BF16 weights stay BF16, and the optimizer can update shards."""
    optim = str(getattr(args.optim, "value", args.optim))
    if optim in _BNB_OPTIMIZERS:
        raise ValueError(f"optim={optim!r}: bitsandbytes' optimizers cannot update FSDP2's sharded parameters; "
                         "use optim='adamw_torch_fused' (LoRA's optimizer state is small)")
    if "bf16" not in given:
        # With mixed precision on, Accelerate turns every BF16 parameter to FP32
        # before sharding: twice the memory, and twice the weights the GPUs send
        # each other at every layer. The model is loaded in BF16 already.
        args.bf16 = False
    elif args.bf16:
        raise ValueError("bf16=True with FSDP: Accelerate then turns the model's BF16 weights to FP32 before "
                         "sharding, doubling their memory and the traffic between the GPUs; leave bf16 unset, "
                         "the model is loaded in BF16")


def _patch(trainer_cls, config_cls, kind: str, trl_1: bool) -> None:
    if getattr(trainer_cls.__init__, "_meepsquared", False):
        return

    config_init = config_cls.__init__

    @functools.wraps(config_init)
    def config_init_recording(self, *args, **kwargs):
        config_init(self, *args, **kwargs)
        object.__setattr__(self, _GIVEN, set(kwargs))

    config_cls.__init__ = config_init_recording

    trainer_init = trainer_cls.__init__
    signature = inspect.signature(trainer_init)

    @functools.wraps(trainer_init)
    def trainer_init_from_model(*call_args, **call_kwargs):
        bound = signature.bind(*call_args, **call_kwargs)
        model, args = bound.arguments.get("model"), bound.arguments.get("args")
        info = getattr(model, "_meepsquared", None)
        cut_to = None
        if info is not None:
            if args is None:
                name = str(model.config._name_or_path).rstrip("/").split("/")[-1]
                args = bound.arguments["args"] = config_cls(output_dir=f"{name}-{kind.upper()}")
            given = getattr(args, _GIVEN, set())
            if kind == "sft" and "padding_free" not in given:
                args.padding_free = info["padding_free"]
            if (kind == "sft" and trl_1 and args.padding_free and not args.packing
                    and args.max_length is not None):
                # Set by meepsquared or given in the config alike: TRL 1.x refuses the pair
                cut_to, args.max_length = args.max_length, None
            if (kind == "sft" and getattr(model, "_meepsquared_compile_layers", False) and not args.padding_free
                    and "pad_to_multiple_of" not in given):
                # Compiled layers recompile for each new sequence length, and Dynamo
                # stops compiling after 8; padded to a multiple of 128, lengths up to
                # 1024 take at most 8 values
                args.pad_to_multiple_of = COMPILED_PAD_MULTIPLE
            if "use_liger_kernel" not in given:
                args.use_liger_kernel = info["liger"]
            if (args.use_liger_kernel and "liger_kernel_config" not in given
                    and getattr(model, "_meepsquared_compile_layers", False)):
                # Inductor fuses a layer's element-wise operations itself, and Dynamo
                # cannot trace Liger's autograd functions; Liger keeps the fused loss,
                # which runs outside the layers
                args.liger_kernel_config = {**(args.liger_kernel_config or {}), "rms_norm": False, "rope": False,
                                            "swiglu": False, "geglu": False, "layer_norm": False}
            elif (args.use_liger_kernel and "liger_kernel_config" not in given
                    and (getattr(model, "_meepsquared_parameter_targets", False)
                         or getattr(model, "_meepsquared_fused_lora", False))):
                # Liger's SwiGLU, and its GeGLU for Gemma, replace the forward of the
                # MLP module. LoRA on parameters (the experts of a mixture of experts)
                # wraps that module, and Liger's forward would run on PEFT's wrapper;
                # fused_lora gives the MLP its own fused forward, which Liger's would
                # replace when the Trainer applies Liger in train() (Transformers 5.20)
                args.liger_kernel_config = {**(args.liger_kernel_config or {}), "swiglu": False, "geglu": False}
            if _checkpointing.selective_possible(model):
                _rope_off_for_selective(args, given)
            if getattr(args, "fsdp", None):
                _fsdp_settings(args, given)
            if int(os.environ.get("WORLD_SIZE", 1)) == 1 and not info.get("split") and getattr(args, "_n_gpu", 1) > 1:
                # One process, the model on one GPU, several GPUs visible: the Trainer
                # would wrap the model in nn.DataParallel, which TRL refuses
                args._n_gpu = 1
            if int(os.environ.get("WORLD_SIZE", 1)) > 1 and "ddp_find_unused_parameters" not in given:
                # Under DDP every adapter takes part in every step; looking for unused
                # parameters only costs a pass over the graph, and fails with
                # gradient checkpointing
                args.ddp_find_unused_parameters = False
            checkpointing = getattr(model, "_meepsquared_gradient_checkpointing", None)
            if checkpointing is True or checkpointing is False or checkpointing is None:
                if checkpointing is not None and "gradient_checkpointing" not in given:
                    args.gradient_checkpointing = checkpointing
                    if checkpointing and "gradient_checkpointing_kwargs" not in given:
                        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
                elif checkpointing and not args.gradient_checkpointing:
                    # The config turns off what get_peft_model turned on
                    model.gradient_checkpointing_disable()
            else:
                # Some layers checkpointed: meepsquared sets them, and the Trainer, which
                # would turn checkpointing on for every layer in train(), is told none
                if given & {"gradient_checkpointing", "gradient_checkpointing_kwargs"}:
                    raise ValueError(f"use_gradient_checkpointing={checkpointing!r} sets the checkpointed layers "
                                     "itself; leave gradient_checkpointing out of the config")
                args.gradient_checkpointing = False
                if checkpointing in _checkpointing.AUTO and (getattr(args, "fsdp", None) or info.get("split")):
                    raise ValueError("use_gradient_checkpointing='auto' measures one GPU's memory; under FSDP or "
                                     "with a model spread over GPUs, give True, False or a number of layers")
        try:
            trainer_init(*bound.args, **bound.kwargs)
        finally:
            if cut_to is not None:
                args.max_length = cut_to
        trainer = bound.arguments["self"]
        if cut_to is not None:
            trainer.data_collator = CutToLength(trainer.data_collator, cut_to)
        if info is not None and info.get("split"):
            from meepsquared import _split

            base = trainer.model.get_base_model() if hasattr(trainer.model, "get_base_model") else trainer.model
            _split.apply(base)
            _split.count_on_output_device(trainer, base._meepsquared_output_device)
        if info is not None and getattr(model, "_meepsquared_fused_lora", False):
            from meepsquared import _fused

            _fused.apply(trainer.model)
        if info is not None and getattr(model, "_meepsquared_gradient_checkpointing", None) in _checkpointing.AUTO:
            # Measured once the trainer has tokenized the data, on a micro-batch of its
            # longest examples, with the kernels that train (the fused LoRA kernels put back above)
            # and before the layers are compiled
            # Liger is applied in train(), so its rotary embedding is decided here, by the
            # layout chosen: off only when a layer is selective. A config that keeps it on
            # leaves selective recomputation out of the choice
            rope_kept = (args.use_liger_kernel and "liger_kernel_config" in given
                         and (args.liger_kernel_config or {}).get("rope") is not False)
            shape, batch = measured_shape(trainer, kind, model.max_seq_length)
            layout = _checkpointing.resolve_auto(
                model, shape, compiled=getattr(model, "_meepsquared_compile_layers", False),
                refused=ROPE_KEPT if rope_kept else None, batch=batch)
            model._meepsquared_checkpointed_layers = layout
            if layout.selective:
                _rope_off_for_selective(args, given)
        if info is not None and getattr(model, "_meepsquared_compile_layers", False):
            from meepsquared import _compile

            _compile.apply(trainer.model)

    trainer_init_from_model._meepsquared = True
    trainer_cls.__init__ = trainer_init_from_model


def patch_trainers() -> None:
    import trl

    trl_1 = int(trl.__version__.split(".")[0]) >= 1
    _patch(trl.SFTTrainer, trl.SFTConfig, "sft", trl_1)
    _patch(trl.GRPOTrainer, trl.GRPOConfig, "grpo", trl_1)
