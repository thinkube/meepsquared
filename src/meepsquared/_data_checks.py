# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The checks of SFT's data that need the model's tokenizer and chat template, made as the job is planned.

- Rendering: an example the chat template or the tokenizer refuses, or one
  of no tokens, is refused, naming the first rows. TRL tokenizes the whole
  set at once; when it fails, each row is rendered alone to find them.
- Length: the examples longer than max_length, and those of them whose
  answer tokens (the tokens the loss counts) run past it, which the cut
  drops. Reported; Recipe(max_cut_share=...) refuses a larger share.
- No answer: an example whose loss counts no token trains nothing. It is
  left out of training and evaluation, and reported.
- Overlap: an evaluation example whose rendered text, normalized (Unicode
  NFKC, case folded, white space collapsed), is a training example's, or
  whose _row id is one, is refused.
- Origin: Datasets marks each row's origin in an _origin column, a mapping
  whose kind is "real" or "synthetic". The plan states the synthetic share
  of the training rows; a synthetic evaluation row is refused.
  Recipe(source_weights={...}) weights the loss of each row by its kind:
  each micro-batch then holds rows of one kind (see meepsquared._batches), and its
  loss, the sum over its counted tokens divided by the update's count of
  counted tokens, is multiplied by the kind's weight (1 for a kind not
  given). The gradient of an update is then the weighted sum of its tokens'.

The _row and _origin columns are metadata: they are read here and never
reach the tokenizer, the trainer or the model.
"""

import unicodedata
from typing import Optional

ROW, ORIGIN = "_row", "_origin"
METADATA = (ROW, ORIGIN)
KINDS = ("real", "synthetic")
# Rows named for each check
EXAMPLES = 5
# The key a weighted micro-batch carries its weight under, from the collator to compute_loss
WEIGHT = "_meepsquared_weight"


def metadata(dataset, where: str) -> dict:
    """{"row": the _row ids or None, "kind": each row's origin kind or None}; refused for a row without a kind."""
    rows = list(dataset[ROW]) if ROW in dataset.column_names else None
    kinds = None
    if ORIGIN in dataset.column_names:
        kinds = []
        missing, unknown = [], []
        for index, origin in enumerate(dataset[ORIGIN]):
            kind = origin.get("kind") if isinstance(origin, dict) else None
            if kind is None:
                missing.append(index)
            elif kind not in KINDS:
                unknown.append((index, kind))
            kinds.append(kind)
        if missing:
            raise ValueError(f"{where}: the _origin of rows {missing[:EXAMPLES]} ({len(missing)} in all) has no kind; "
                             f"Datasets marks each row {' or '.join(KINDS)}")
        if unknown:
            raise ValueError(f"{where}: _origin kinds {unknown[:EXAMPLES]} are not one of {KINDS}")
    return {"row": rows, "kind": kinds}


def render_failures(dataset, tokenizer, assistant_masks: bool) -> tuple:
    """(failures, no_answers): rows the chat template or the tokenizer refuses, or that make no tokens, as (row,
    error); and, with TRL's assistant masks, the rows that render with no assistant token, which TRL refuses."""
    failures, no_answers = [], []
    for index, row in enumerate(dataset):
        try:
            masks = None
            if "messages" in row:
                out = tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=True,
                                                    return_assistant_tokens_mask=assistant_masks)
                ids, masks = out["input_ids"], out.get("assistant_masks") if assistant_masks else None
            elif "prompt" in row:
                if isinstance(row["prompt"], list):
                    ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True,
                                                        return_dict=True)["input_ids"]
                else:
                    ids = tokenizer(row["prompt"] + row["completion"])["input_ids"]
            else:
                ids = tokenizer(row["text"])["input_ids"]
        except Exception as error:  # noqa: BLE001 (any error of the template or the tokenizer names the row)
            failures.append((index, f"{type(error).__name__}: {error}"))
            continue
        if not ids:
            failures.append((index, "no tokens"))
        elif masks is not None and 1 not in masks:
            no_answers.append(index)
    return failures, no_answers


def counted(dataset, completion_only: bool, markers: Optional[tuple]) -> list:
    """The tokens each tokenized example's loss counts, as lists of booleans: by TRL's labels or masks (TRL 1.x
    turns its masks into labels, TRL 0.29 keeps them), and by meepsquared's turn markers (token ids) when given."""
    from meepsquared.chat_templates import response_mask

    columns = dataset.column_names
    data = dataset.select_columns([c for c in ("input_ids", "labels", "assistant_masks", "completion_mask")
                                   if c in columns]).to_dict()
    out = []
    for index, ids in enumerate(data["input_ids"]):
        keep = [True] * len(ids)
        if "labels" in data:
            keep = [k and label != -100 for k, label in zip(keep, data["labels"][index])]
        if "assistant_masks" in data:
            keep = [k and bool(m) for k, m in zip(keep, data["assistant_masks"][index])]
        if completion_only and "completion_mask" in data:
            keep = [k and bool(m) for k, m in zip(keep, data["completion_mask"][index])]
        if markers is not None:
            keep = [k and m for k, m in zip(keep, response_mask(list(ids), markers[0], markers[1]))]
        out.append(keep)
    return out


def normalized(dataset, tokenizer) -> list:
    """Each tokenized example's text, decoded, in Unicode NFKC, case folded, its white space collapsed."""
    texts = tokenizer.batch_decode(list(dataset["input_ids"]))
    return [" ".join(unicodedata.normalize("NFKC", text).casefold().split()) for text in texts]


def overlap(train_texts: list, eval_texts: list, train_rows: Optional[list], eval_rows: Optional[list]) -> list:
    """(evaluation row, training row, how) of each evaluation example that is a training example."""
    by_text = {}
    for index, text in enumerate(train_texts):
        by_text.setdefault(text, index)
    by_row = {}
    if train_rows is not None and eval_rows is not None:
        for index, row in enumerate(train_rows):
            by_row.setdefault(row, index)
    found = []
    for index, text in enumerate(eval_texts):
        if text in by_text:
            found.append((index, by_text[text], "the same text"))
        elif eval_rows is not None and eval_rows[index] in by_row:
            found.append((index, by_row[eval_rows[index]], f"the same _row {eval_rows[index]!r}"))
    return found


def cut(dataset, max_length: int):
    """Every per-token column of each example cut to its first max_length tokens, as TRL and meepsquared's collator cut
    them."""
    import pyarrow.compute as pc

    from meepsquared import _trl

    if max(pc.list_value_length(dataset.data.column("input_ids")).to_pylist(), default=0) <= max_length:
        return dataset
    from meepsquared import _unified_memory

    cores = _unified_memory.job_cores()
    return dataset.map(_trl._cut, fn_kwargs={"max_length": max_length}, desc="Cutting to max_length",
                       num_proc=cores if cores > 1 else None)


class _Weighted:
    """A map-style dataset whose items carry their row's loss weight."""

    def __init__(self, dataset, weights: list):
        self.dataset, self.weights = dataset, weights

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], WEIGHT: self.weights[index]}


class _WeightedCollator:
    """Collates items of one weight, and gives the batch that weight."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        weights = {example.pop(WEIGHT) for example in examples}
        if len(weights) != 1:
            raise RuntimeError(f"a weighted micro-batch holds rows of weights {sorted(weights)}; meepsquared lays out "
                               "each micro-batch of one origin kind")
        batch = self.collator(examples)
        batch[WEIGHT] = weights.pop()
        return batch


def weighted_loader(trainer, dataset):
    """(dataset, collator) of the training DataLoader: the rows' weights carried to compute_loss when the trainer
    has any."""
    weights = trainer._meepsquared_weights
    if weights is None:
        return dataset, trainer.data_collator
    return _Weighted(dataset, weights), _WeightedCollator(trainer.data_collator)


def install_weights(trainer, weights: list) -> None:
    """Multiply each micro-batch's loss by its rows' weight; `weights` gives each training example's."""
    compute_loss = trainer.compute_loss

    def weighted(model, inputs, return_outputs=False, num_items_in_batch=None):
        weight = inputs.pop(WEIGHT, None)
        result = compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if weight is None:
            return result
        if return_outputs:
            loss, outputs = result
            return loss * weight, outputs
        return result * weight

    trainer._meepsquared_weights = list(weights)
    trainer.compute_loss = weighted


def rows_text(rows: list, ids: Optional[list]) -> str:
    """The first EXAMPLES rows, with their _row ids when the data has them."""
    shown = [f"{row} ({ids[row]!r})" if ids is not None else str(row) for row in rows[:EXAMPLES]]
    return ", ".join(shown) + (f" and {len(rows) - EXAMPLES} more" if len(rows) > EXAMPLES else "")
