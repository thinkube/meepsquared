# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The checks of SFT's data that need the model's tokenizer and chat template, made as the data is preprocessed
(meepsquared._preprocess), each as a pass over a chunk of rows.

- Rendering: an example the chat template or the tokenizer refuses, or one
  of no tokens, is refused, naming the first rows. TRL tokenizes the whole
  set at once; when it fails, each row is rendered alone to find them.
- Length: the examples longer than max_length, and those of them whose
  answer tokens (the tokens the loss counts) run past it, which the cut
  drops. Reported; Recipe(max_cut_share=...) refuses a larger share.
- No answer: an example whose loss counts no token trains nothing. It is
  left out of training and evaluation, and reported.
- Overlap: an evaluation example whose rendered text, normalized (Unicode
  NFKC, case folded, white space collapsed) and hashed, is a training
  example's, or whose _row id is one, is refused.
- Origin: Datasets marks each row's origin in an _origin column, a mapping
  whose kind is "real" or "synthetic". The plan states the synthetic share
  of the training rows; a synthetic evaluation row is refused.
  Recipe(source_weights={...}) weights the loss of each row by its kind:
  each micro-batch then holds rows of one kind (see meepsquared._batches), and its
  loss, the sum over its counted tokens divided by the update's count of
  counted tokens, is multiplied by the kind's weight (1 for a kind not
  given). The gradient of an update is then the weighted sum of its tokens'.

A pass keeps per row only what the checks need: a length, a few booleans, a
16-byte hash. The _row and _origin columns are metadata: they are read here
and never reach the tokenizer, the trainer or the model.
"""

import hashlib
import unicodedata
from typing import Optional

ROW, ORIGIN = "_row", "_origin"
METADATA = (ROW, ORIGIN)
KINDS = ("real", "synthetic")
# Rows named for each check
EXAMPLES = 5
# The key a weighted micro-batch carries its weight under, from the collator to compute_loss
WEIGHT = "_meepsquared_weight"


def _rows_of(batch: dict) -> list:
    """A batch (columns of values) as rows."""
    names = list(batch.keys())
    return [dict(zip(names, values)) for values in zip(*(batch[name] for name in names))]


def kinds_of(batch: dict, indices: list, where: str) -> dict:
    """Each row's origin kind, from its _origin; refused for a row without a kind, or of an unknown one."""
    from meepsquared import _preprocess

    kinds, missing, unknown = [], [], []
    for index, origin in zip(indices, batch[ORIGIN]):
        kind = origin.get("kind") if isinstance(origin, dict) else None
        if kind is None:
            missing.append(index)
        elif kind not in KINDS:
            unknown.append((index, kind))
        kinds.append(kind)
    if missing:
        raise ValueError(f"{where}: the _origin of rows {missing[:EXAMPLES]} has no kind; Datasets marks each row "
                         f"{' or '.join(KINDS)}")
    if unknown:
        raise ValueError(f"{where}: _origin kinds {unknown[:EXAMPLES]} are not one of {KINDS}")
    return {_preprocess.KIND: kinds}


def render_failures(batch: dict, tokenizer, assistant_masks: bool, chat_template: Optional[str] = None) -> dict:
    """Per row: the error of the chat template or the tokenizer, or "" where it renders; and, with TRL's assistant
    masks, whether it renders with no assistant token, which TRL refuses. `chat_template`: the template TRL renders
    with, its training template where it put one in; the tokenizer's own otherwise."""
    from meepsquared import _preprocess

    failures, no_answers = [], []
    for row in _rows_of(batch):
        failure, no_answer = "", False
        try:
            masks = None
            if "messages" in row:
                out = tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=True,
                                                    return_assistant_tokens_mask=assistant_masks,
                                                    chat_template=chat_template)
                ids, masks = out["input_ids"], out.get("assistant_masks") if assistant_masks else None
            elif "prompt" in row:
                if isinstance(row["prompt"], list):
                    ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True,
                                                        return_dict=True, chat_template=chat_template)["input_ids"]
                else:
                    ids = tokenizer(row["prompt"] + row["completion"])["input_ids"]
            else:
                ids = tokenizer(row["text"])["input_ids"]
        except Exception as error:  # noqa: BLE001 (any error of the template or the tokenizer names the row)
            failure = f"{type(error).__name__}: {error}"
        else:
            if not ids:
                failure = "no tokens"
            elif masks is not None and 1 not in masks:
                no_answer = True
        failures.append(failure)
        no_answers.append(no_answer)
    return {_preprocess.FAILURE: failures, _preprocess.NO_ANSWER: no_answers}


def counted(ids: list, labels: Optional[list], assistant_masks: Optional[list], completion_mask: Optional[list],
            completion_only: bool, markers: Optional[tuple]) -> list:
    """The tokens of one tokenized example the loss counts, as booleans: by TRL's labels or masks, and by
    meepsquared's turn markers (token ids) when given."""
    from meepsquared.chat_templates import response_mask

    keep = [True] * len(ids)
    if labels is not None:
        keep = [k and label != -100 for k, label in zip(keep, labels)]
    if assistant_masks is not None:
        keep = [k and bool(m) for k, m in zip(keep, assistant_masks)]
    if completion_only and completion_mask is not None:
        keep = [k and bool(m) for k, m in zip(keep, completion_mask)]
    if markers is not None:
        keep = [k and m for k, m in zip(keep, response_mask(list(ids), markers[0], markers[1]))]
    return keep


def loss_facts(batch: dict, completion_only: bool, markers: Optional[tuple], max_length: int) -> dict:
    """Per row: its length after the cut to max_length, whether it has no token, whether its loss counts a token,
    whether a counted token lies past max_length (the cut drops it), and whether it is longer than max_length."""
    from meepsquared import _preprocess

    lengths, empty, has_loss, cut_loss, longer = [], [], [], [], []
    count = len(batch["input_ids"])
    labels = batch.get("labels", [None] * count)
    assistant = batch.get("assistant_masks", [None] * count)
    completion = batch.get("completion_mask", [None] * count)
    for i, ids in enumerate(batch["input_ids"]):
        keep = counted(ids, labels[i], assistant[i], completion[i], completion_only, markers)
        lengths.append(min(len(ids), max_length))
        empty.append(len(ids) == 0)
        has_loss.append(any(keep))
        cut_loss.append(any(keep[max_length:]))
        longer.append(len(ids) > max_length)
    return {_preprocess.LENGTH: lengths, _preprocess.EMPTY: empty, _preprocess.HAS_LOSS: has_loss,
            _preprocess.CUT_LOSS: cut_loss, _preprocess.LONGER: longer}


def normalized(text: str) -> str:
    """The text in Unicode NFKC, case folded, its white space collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hashes(batch: dict, tokenizer) -> dict:
    """Per row: a 16-byte hash of its tokens decoded and normalized, for the overlap check."""
    from meepsquared import _preprocess

    texts = tokenizer.batch_decode(batch["input_ids"])
    return {_preprocess.HASH: [hashlib.blake2b(normalized(text).encode(), digest_size=16).digest()
                               for text in texts]}


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


def rows_text(rows: list, ids: Optional[list], count: Optional[int] = None) -> str:
    """The first EXAMPLES rows (by their place in the data), with their _row ids when the data has them, and how
    many more there are of `count`."""
    count = len(rows) if count is None else count
    shown = [f"{row} ({ids[i]!r})" if ids is not None else str(row) for i, row in enumerate(rows[:EXAMPLES])]
    return ", ".join(shown) + (f" and {count - EXAMPLES} more" if count > EXAMPLES else "")
