# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The order of SFT's training data, laid out ahead as the micro-batches the training runs.

meepsquared owns the order of the examples, so that the memory is measured on a
batch the training runs, and no batch it runs is larger (see
meepsquared._checkpointing.largest_batch). Each epoch shuffles the examples with
a generator seeded from the trainer's seed plus the epoch (args.data_seed
when set, else args.seed, as Transformers' TrainingArguments define them),
then cuts them into micro-batches in that order, one of two ways:

- a number of examples per micro-batch: consecutive examples `micro` at a
  time, the last micro-batch shorter, or dropped with dataloader_drop_last,
  as PyTorch's BatchSampler cuts them;
- a token budget per micro-batch: consecutive examples grouped until the
  next would take the group over the budget (see groups). A conversation is
  never split, so a group holds at least one, even one longer than the
  budget.

The cost of a group is the tokens the GPU computes:

- packed (padding-free): the sum of the lengths, as the collator lays the
  conversations end to end in one sequence;
- padded: the count times the longest, rounded up to the padding multiple,
  as every conversation is padded to the longest one.

Under several processes, Accelerate's BatchSamplerShard gives each process
its share of the micro-batches: one in turn to each process, and with
even_batches, the end evened out with micro-batches from the start of the
epoch. meepsquared counts each process's share with that same class.

The Trainer counts the micro-batches of an epoch once, from len() before the
first epoch: ceil(count / accumulation) updates an epoch, `accumulation`
micro-batches each and the last the rest, until max_steps updates; with
epochs, max_steps is ceil(epochs x updates an epoch). trained_by() follows
that count. With a token budget, an epoch's group count can differ from the
first epoch's by a few: a shorter epoch ends early, a longer one leaves its
last groups unseen, and trained_by() leaves them out too.

install() gives the trainer the layout: its get_train_dataloader is replaced
by one that passes the layout as the DataLoader's batch sampler. TRL's
SFTTrainer, on TRL 0.29 and 1.x alike, builds its DataLoader with the
Trainer's get_train_dataloader.

A training resumed from a checkpoint continues at the next micro-batch laid
out: resume_at() starts the batch sampler in the epoch the training stopped
in, and the Trainer skips the micro-batches of the steps trained in it, on
each process its own share.
"""

import dataclasses
import functools
import itertools
import math
import types
from typing import Optional

from torch.utils.data import DataLoader, Sampler


def lengths(dataset, max_length: int) -> list:
    """The tokens of each example of a tokenized datasets.Dataset, cut to max_length as the collator cuts them."""
    import datasets
    import pyarrow.compute as pc

    if not isinstance(dataset, datasets.Dataset):
        raise TypeError(f"meepsquared lays out the batches of a datasets.Dataset with tokenized examples, not "
                        f"{type(dataset).__name__}")
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"meepsquared lays out the batches from the tokenized examples ('input_ids'); the dataset has "
                         f"{dataset.column_names}")
    counted = pc.list_value_length(dataset.with_format("arrow")["input_ids"]).to_pylist()
    return [min(n, max_length) for n in counted]


def groups(lengths: list, order, budget: int, padded: bool, multiple: int = 1) -> list:
    """Consecutive examples of `order`, grouped while their cost stays within `budget`; one at least per group."""
    out, group, longest, total = [], [], 0, 0
    for index in order:
        length = lengths[index]
        if group:
            if padded:
                cost = (len(group) + 1) * math.ceil(max(longest, length) / multiple) * multiple
            else:
                cost = total + length
            if cost > budget:
                out.append(group)
                group, longest, total = [], 0, 0
        group.append(index)
        longest, total = max(longest, length), total + length
    if group:
        out.append(group)
    return out


def shuffled(count: int, seed: int, epoch: int) -> list:
    """The examples 0 to count - 1 in the order of `epoch`: a permutation drawn by a generator seeded seed + epoch."""
    import torch

    return torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def held_out(count: int, share: float, seed: int) -> tuple:
    """(trained, held out): the examples 0 to count - 1 split for evaluation, each part in the data's order.

    The held out are the first round(share x count) of a permutation drawn by
    a generator seeded with `seed`, at least one; at least one is trained.
    """
    import torch

    held = max(1, round(share * count))
    if held >= count:
        raise ValueError(f"eval_share {share:g} of {count} examples holds out {held}, and leaves none to train on")
    chosen = set(torch.randperm(count, generator=torch.Generator().manual_seed(seed))[:held].tolist())
    return [i for i in range(count) if i not in chosen], sorted(chosen)


def cut(order: list, micro: int, drop_last: bool) -> list:
    """Consecutive examples of `order`, `micro` at a time; a shorter last one kept unless drop_last."""
    batches = [order[start:start + micro] for start in range(0, len(order), micro)]
    if drop_last and batches and len(batches[-1]) < micro:
        batches.pop()
    return batches


def padding(args) -> tuple:
    """(padded, multiple) from TRL's SFTConfig: whether batches are padded, and to what multiple of tokens (1: none)."""
    padded = not args.padding_free
    return padded, args.pad_to_multiple_of if padded and args.pad_to_multiple_of is not None else 1


@dataclasses.dataclass(frozen=True)
class Loading:
    """What decides the order of a trainer's micro-batches besides the batch: its seed, how it pads and drops, and how
    Accelerate shares the micro-batches among its processes."""

    seed: int
    drop_last: bool
    padded: bool
    multiple: int
    processes: int
    split_batches: bool
    even_batches: bool


def loading(trainer) -> Loading:
    """The Loading of a built SFT trainer."""
    args, accelerator = trainer.args, trainer.accelerator
    if accelerator.dispatch_batches:
        raise ValueError("accelerator_config dispatch_batches=True: the main process would load every batch and "
                         "split it among the processes; meepsquared lays out each process's micro-batches, leave it unset")
    if accelerator.num_processes > 1 and not accelerator.even_batches:
        raise ValueError("accelerator_config even_batches=False: the processes would train different numbers of "
                         "micro-batches; meepsquared lays out even shares, leave it unset")
    padded, multiple = padding(args)
    # TrainingArguments: data_seed, "if not set, random generators for data sampling will use the same seed as seed"
    seed = args.data_seed if args.data_seed is not None else args.seed
    return Loading(seed, bool(args.dataloader_drop_last), padded, multiple, accelerator.num_processes,
                   bool(accelerator.split_batches), bool(accelerator.even_batches))


class _Epoch:
    """One epoch's micro-batches, as a batch sampler for Accelerate's BatchSamplerShard."""

    def __init__(self, batches: list, batch_size: Optional[int], drop_last: bool):
        self.batches, self.batch_size, self.drop_last = batches, batch_size, drop_last

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def share(batches: list, batch_size: Optional[int], drop_last: bool, loading: Loading, process: int) -> tuple:
    """(count, micro-batches): what one process trains of an epoch's micro-batches, as Accelerate's
    BatchSamplerShard gives them; the count is its len(), which the Trainer counts by."""
    epoch = _Epoch(batches, batch_size, drop_last)
    if loading.processes == 1:
        return len(epoch), list(epoch)
    from accelerate.data_loader import BatchSamplerShard

    shard = BatchSamplerShard(epoch, num_processes=loading.processes, process_index=process,
                              split_batches=loading.split_batches, even_batches=loading.even_batches)
    return len(shard), list(shard)


@dataclasses.dataclass(frozen=True)
class Layout:
    """The micro-batches of every epoch a training runs, and what it trains of them.

    epochs: each epoch's micro-batches before Accelerate shares them out, lists
    of example indices. batch_size: the examples of a micro-batch, None with a
    token budget. per_epoch: the micro-batches of an epoch on each process, as
    the Trainer counts them. steps: the updates the training makes. trained:
    the micro-batches the training runs, on every process.
    """

    epochs: tuple
    batch_size: Optional[int]
    loading: Loading
    examples: int
    per_epoch: int
    steps: int
    trained: tuple


def lay_out(lengths: list, loading: Loading, accumulation: int, steps: Optional[int], epochs: Optional[float],
            micro: Optional[int] = None, budget: Optional[int] = None, kinds: Optional[list] = None) -> Layout:
    """The micro-batches of a training of `steps` updates, or `epochs` (one when neither), `accumulation`
    micro-batches an update; each micro-batch `micro` examples, or as many as fit `budget` tokens.

    With `kinds` (each example's), each micro-batch holds examples of one
    kind: the epoch's order is cut kind by kind, and its micro-batches are
    then shuffled by the same generator, so that the kinds mix in the order.
    """
    if (micro is None) == (budget is None):
        raise ValueError("a layout takes micro (examples per micro-batch) or budget (tokens per micro-batch), "
                         "one of them")
    if not lengths:
        raise ValueError("the training data has no examples")

    def batches_of(order: list) -> list:
        if micro is not None:
            return cut(order, micro, loading.drop_last)
        return groups(lengths, order, budget, loading.padded, loading.multiple)

    def epoch(index: int) -> list:
        if kinds is None:
            return batches_of(shuffled(len(lengths), loading.seed, index))
        import torch

        generator = torch.Generator().manual_seed(loading.seed + index)
        order = torch.randperm(len(lengths), generator=generator).tolist()
        made = []
        for kind in sorted(set(kinds)):
            made += batches_of([i for i in order if kinds[i] == kind])
        return [made[i] for i in torch.randperm(len(made), generator=generator).tolist()]

    drop_last = loading.drop_last if micro is not None else False
    first = epoch(0)
    per_epoch = share(first, micro, drop_last, loading, 0)[0]
    if per_epoch == 0:
        raise ValueError(f"the training data's {len(lengths)} examples make no micro-batch on each of "
                         f"{loading.processes} processes")
    updates = max(math.ceil(per_epoch / accumulation), 1)
    max_steps = steps if steps is not None else math.ceil((epochs if epochs is not None else 1) * updates)
    count = math.ceil(max_steps / updates) if steps is not None else math.ceil(epochs if epochs is not None else 1)
    laid = [first] + [epoch(index) for index in range(1, count)]
    trained, made = [], None
    for process in range(loading.processes):
        taken, made = trained_by([share(batches, micro, drop_last, loading, process)[1] for batches in laid],
                                 per_epoch, accumulation, max_steps)
        trained += taken
    return Layout(tuple(laid), micro, loading, len(lengths), per_epoch, made, tuple(trained))


def trained_by(shares: list, per_epoch: int, accumulation: int, max_steps: int) -> tuple:
    """(micro-batches, updates): what one process trains of its share of each epoch, as the Trainer takes them.

    An epoch makes ceil(per_epoch / accumulation) updates, of `accumulation`
    micro-batches each and the last of the rest; an epoch whose share runs
    out first ends there, and the training ends after max_steps updates.
    """
    updates = max(math.ceil(per_epoch / accumulation), 1)
    last = per_epoch % accumulation or accumulation
    out, step = [], 0
    for batches in shares:
        taken = iter(batches)
        for update in range(updates):
            pulled = list(itertools.islice(taken, accumulation if update != updates - 1 else last))
            if not pulled:
                break
            out += pulled
            step += 1
            if step >= max_steps:
                return out, step
    return out, step


class LaidOutBatchSampler(Sampler):
    """The micro-batches of a Layout, one epoch per pass: the epoch set_epoch names, else the one after the last
    pass, starting at `epoch`.

    len() is the first epoch's count, which the Trainer counts every epoch by.
    batch_size None (a token budget): Accelerate's BatchSamplerShard then
    takes the micro-batches as of varying size.
    """

    def __init__(self, layout: Layout, epoch: int = 0):
        self.layout = layout
        self.batch_size = layout.batch_size
        self.drop_last = layout.loading.drop_last if layout.batch_size is not None else False
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.layout.epochs[0])

    def __iter__(self):
        epoch = self.epoch
        if epoch >= len(self.layout.epochs):
            raise RuntimeError(f"the trainer started epoch {epoch + 1}, and meepsquared laid out the micro-batches of "
                               f"{len(self.layout.epochs)} epochs for {self.layout.steps} updates; the trainer's "
                               "max_steps or num_train_epochs changed after the plan")
        self.epoch = epoch + 1
        return iter(self.layout.epochs[epoch])


def _get_train_dataloader(trainer, layout: Layout) -> DataLoader:
    """The Trainer's training DataLoader, iterating the micro-batches of `layout`."""
    from transformers.trainer_utils import seed_worker

    from meepsquared import _data_checks

    args = trainer.args
    dataset = trainer._remove_unused_columns(trainer.train_dataset, description="Training")
    if len(dataset) != layout.examples:
        raise ValueError(f"the trainer's dataset has {len(dataset)} examples, and the micro-batches were laid out "
                         f"for {layout.examples}")
    dataset, collator = _data_checks.weighted_loader(trainer, dataset)
    loader = DataLoader(dataset, batch_sampler=LaidOutBatchSampler(layout, trainer._meepsquared_first_epoch),
                        collate_fn=collator,
                        num_workers=args.dataloader_num_workers, pin_memory=args.dataloader_pin_memory,
                        persistent_workers=args.dataloader_persistent_workers,
                        prefetch_factor=args.dataloader_prefetch_factor,
                        worker_init_fn=functools.partial(seed_worker, num_workers=args.dataloader_num_workers,
                                                         rank=args.process_index))
    return trainer.accelerator.prepare(loader)


def install(trainer, layout: Layout) -> None:
    """Make the trainer train the micro-batches of `layout`, kept as its _meepsquared_layout; refused when the trainer
    would load them otherwise."""
    if loading(trainer) != layout.loading:
        raise ValueError(f"the micro-batches were laid out for {layout.loading}, and the trainer loads them as "
                         f"{loading(trainer)}")
    trainer.get_train_dataloader = types.MethodType(functools.partial(_get_train_dataloader, layout=layout), trainer)
    trainer._meepsquared_layout = layout
    trainer._meepsquared_first_epoch = 0
    trainer._meepsquared_weights = None


def resume_at(trainer, step: int) -> int:
    """Make the trainer's micro-batches start in the epoch a training resumed after `step` updates continues;
    returns that epoch.

    The Trainer counts the epochs trained as step // ceil(micro-batches an
    epoch / accumulation), and skips the micro-batches of the steps trained
    in that epoch. The batch sampler starts in that epoch itself, as
    Accelerate's set_epoch does not reach it through the skipping sampler
    under several processes.
    """
    layout = trainer._meepsquared_layout
    updates = max(math.ceil(layout.per_epoch / trainer.args.gradient_accumulation_steps), 1)
    epoch = step // updates
    if epoch >= len(layout.epochs):
        raise ValueError(f"resume after {step} updates: epoch {epoch + 1}, and meepsquared laid out the micro-batches of "
                         f"{len(layout.epochs)} epochs for {layout.steps} updates")
    trainer._meepsquared_first_epoch = epoch
    return epoch
