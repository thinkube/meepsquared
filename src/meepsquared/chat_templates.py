# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Training on the model's answers only.

    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\\n",
                                      response_part="<|im_start|>assistant\\n")

The loss then counts only the tokens after each response_part, up to the next
instruction_part: the system prompt and the user's turns are context, not
targets. The masking is done on each collated batch, so it applies the same
way to padded and padding-free batches and with any TRL version; in a
padding-free row each example, from one position id 0 to the next, is masked
on its own.

A training job trains a conversation on its answers with TRL's own
assistant-only loss where TRL can mask the tokenizer's chat template
(trl_assistant_mask), else with train_on_responses_only and the turn markers
read from the chat template (turn_markers); see meepsquared.job.
"""

import re
from typing import Optional

import torch


def _find(row: list, part: list, start: int = 0) -> int:
    """The index after the first match of part in row[start:], or -1."""
    n = len(part)
    for i in range(start, len(row) - n + 1):
        if row[i:i + n] == part:
            return i + n
    return -1


def response_mask(ids: list, instruction: list, response: list) -> list:
    """True where a token belongs to an answer: after a response part, before the next instruction part."""
    mask = [False] * len(ids)
    at = 0
    while True:
        begin = _find(ids, response, at)
        if begin < 0:
            return mask
        end = _find(ids, instruction, begin)
        stop = end - len(instruction) if end >= 0 else len(ids)
        for i in range(begin, stop):
            mask[i] = True
        if end < 0:
            return mask
        at = stop


class ResponsesOnly:
    """A data collator that sets the labels of every non-answer token to -100."""

    def __init__(self, collator, instruction: list, response: list):
        self.collator, self.instruction, self.response = collator, instruction, response

    def __call__(self, examples):
        batch = self.collator(examples)
        ids, labels = batch["input_ids"], batch["labels"]
        positions = batch.get("position_ids")
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for row in range(ids.shape[0]):
            starts = [0]
            if positions is not None and ids.shape[0] == 1:
                starts = (positions[row] == 0).nonzero().flatten().tolist() or [0]
            bounds = starts + [ids.shape[1]]
            for begin, end in zip(bounds[:-1], bounds[1:]):
                mask = response_mask(ids[row, begin:end].tolist(), self.instruction, self.response)
                keep[row, begin:end] = torch.tensor(mask, dtype=torch.bool)
        batch["labels"] = labels.masked_fill(~keep, -100)
        return batch


def train_on_responses_only(trainer, instruction_part: str, response_part: str, **kwargs):
    """The trainer, its loss counted on the answers only.

    instruction_part and response_part are the text the chat template writes
    before a user turn and before an assistant turn, tokenized as the
    tokenizer does without special tokens of its own.
    """
    tokenizer = trainer.processing_class
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    instruction = tokenizer(instruction_part, add_special_tokens=False)["input_ids"]
    response = tokenizer(response_part, add_special_tokens=False)["input_ids"]
    if not instruction or not response:
        raise ValueError("instruction_part and response_part must each tokenize to at least one token")
    trainer.data_collator = ResponsesOnly(trainer.data_collator, instruction, response)
    return trainer


# Words a chat template writes as they are, to find where it puts each turn
_USER, _ANSWER, _NEXT = "meepsquareduserturn", "meepsquaredanswerturn", "meepsquarednextturn"
_GENERATION_MARKER = re.compile(r"\{%-?\s*generation\s*-?%\}")


def _template(tokenizer) -> str:
    template = tokenizer.chat_template
    if isinstance(template, dict):  # a template shipped as named variants: TRL uses the default one
        template = template["default"]
    if not template:
        raise ValueError("the tokenizer has no chat template, and conversations are written with one")
    return template


def _overlaps(offset, span) -> bool:
    return offset[0] < span[1] and span[0] < offset[1]


def _singles_out(tokenizer, text: str, instruction: str, response: str) -> bool:
    """Whether the markers, tokenized, keep every token of the answer in text and no token of a user turn."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    mask = response_mask(list(encoded["input_ids"]), instruction_ids, response_ids)
    spans = {word: (text.index(word), text.index(word) + len(word)) for word in (_USER, _ANSWER, _NEXT)
             if word in text}
    offsets = encoded["offset_mapping"]
    answer_kept = all(kept for kept, offset in zip(mask, offsets) if _overlaps(offset, spans[_ANSWER]))
    user_kept = any(kept for kept, offset in zip(mask, offsets)
                    if any(_overlaps(offset, spans[word]) for word in (_USER, _NEXT) if word in spans))
    return any(mask) and answer_kept and not user_kept


def turn_markers(tokenizer) -> tuple[str, str]:
    """(instruction_part, response_part) for train_on_responses_only, read from the tokenizer's chat template.

    response_part is the longest start of the template's generation prompt
    (what it adds for the model's turn to begin) that finds the answer. A
    thinking model's prompt may open a thinking block that an answer in the
    history does not have, so shorter starts are tried, token by token.
    instruction_part is what the template writes between the end of an
    answer and the next user turn, without the answer's own end-of-turn text,
    so that the end-of-turn token stays in the loss and the model learns to
    stop. The markers must single out the answer, and nothing of the user's
    turns, in a conversation of one answer and in one of an answer followed by
    a user turn. Raises ValueError, saying why, for a template they cannot be
    read from.
    """
    _template(tokenizer)
    user = [{"role": "user", "content": _USER}]
    answered = user + [{"role": "assistant", "content": _ANSWER}]
    following = answered + [{"role": "user", "content": _NEXT}]

    def render(messages, generation_prompt=False) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)

    prompt, with_header = render(user), render(user, True)
    if not with_header.startswith(prompt) or len(with_header) == len(prompt):
        raise ValueError("the chat template's generation prompt does not extend the conversation it ends")
    header = with_header[len(prompt):]
    alone, full = render(answered), render(following)
    if _ANSWER not in alone or _ANSWER not in full or _NEXT not in full:
        raise ValueError("the chat template does not write the turns' text as given")
    end_of_turn = alone[alone.index(_ANSWER) + len(_ANSWER):]
    between = full[full.index(_ANSWER) + len(_ANSWER):full.index(_NEXT)]
    if not between.startswith(end_of_turn):
        raise ValueError("the chat template ends an answer differently when a user turn follows it")
    instruction = between[len(end_of_turn):]
    if not instruction.strip():
        raise ValueError("the chat template writes no text that marks a user turn")
    # The starts of the generation prompt, each cut where one of its tokens begins, longest first
    starts = [token_start for token_start, _ in tokenizer(header, add_special_tokens=False,
                                                          return_offsets_mapping=True)["offset_mapping"]]
    for response in [header] + [header[:start] for start in reversed(starts[1:])]:
        if response.strip() and all(_singles_out(tokenizer, text, instruction, response) for text in (alone, full)):
            return instruction, response
    raise ValueError(f"no start of the chat template's generation prompt ({header!r}) with {instruction!r} "
                     "singles out the answer once tokenized")


def _stop_token_in_mask(tokenizer, chat_template) -> Optional[bool]:
    """Whether the template's assistant mask holds text after the answer: its end-of-turn token.

    None when the rendered text, tokenized again, does not give the tokens the
    mask was computed on, so the mask cannot be read against the text.
    """
    messages = [{"role": "user", "content": _USER}, {"role": "assistant", "content": _ANSWER},
                {"role": "user", "content": _NEXT}]
    out = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                                        chat_template=chat_template)
    text = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=chat_template)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(out["input_ids"]):
        return None
    answer_end = text.index(_ANSWER) + len(_ANSWER)
    return any(kept and start >= answer_end and text[start:end].strip()
               for kept, (start, end) in zip(out["assistant_masks"], encoded["offset_mapping"]))


def trl_assistant_mask(tokenizer) -> tuple[bool, str]:
    """Whether TRL's assistant_only_loss can mask this tokenizer's conversations, and why or why not.

    TRL masks the tokens the chat template writes between {% generation %} and
    {% endgeneration %}. A template without those markers works where TRL's
    SFTTrainer swaps in a training version of it (TRL 1.x has them for the
    common templates). The mask must also hold the answer's end-of-turn token,
    or the model would not learn to stop.
    """
    import trl
    from trl.trainer import sft_trainer

    template = _template(tokenizer)
    if _GENERATION_MARKER.search(template):
        chat_template, how = None, "the chat template marks the answers ({% generation %})"
    elif hasattr(sft_trainer, "get_training_chat_template"):
        try:
            chat_template = sft_trainer.get_training_chat_template(tokenizer)
        except ValueError:
            return False, (f"the chat template has no {{% generation %}} markers, and TRL {trl.__version__} has "
                           "no training version of it")
        how = "TRL's training version of the chat template"
    else:
        return False, (f"the chat template has no {{% generation %}} markers, and TRL {trl.__version__} does not "
                       "swap in a training version")
    stop = _stop_token_in_mask(tokenizer, chat_template)
    if stop is None:
        return False, "the tokens of TRL's mask cannot be matched to the text of the chat template"
    if not stop:
        return False, ("TRL's mask of this chat template leaves out the end-of-turn token, so the model would "
                       "not learn to stop")
    return True, how
