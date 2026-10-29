# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from meepsquared import MeepLanguageModel
from meepsquared.chat_templates import response_mask, train_on_responses_only, trl_assistant_mask, turn_markers

U, A = [1], [2]  # instruction part, response part


def test_the_mask_covers_each_answer_and_nothing_else():
    ids = [9, 1, 5, 5, 2, 7, 7, 1, 5, 2, 8]
    assert response_mask(ids, U, A) == [False, False, False, False, False, True, True, False, False, False, True]


def test_no_response_part_masks_everything():
    assert response_mask([9, 1, 5, 5], U, A) == [False] * 4


TALK = Dataset.from_dict({"text": ["the user runs and the hare fast", "user sloth hare sleeps"]})


@pytest.fixture
def words(tokenizer):
    """The test tokenizer, with "user" and "hare" standing for the instruction and response parts."""
    tokenizer.add_tokens(["user"])
    return tokenizer


@pytest.mark.parametrize("padding_free", [True, False])
def test_only_answer_tokens_keep_their_labels(tiny_model, words, tmp_path, padding_free):
    tiny_model.resize_token_embeddings(len(words))
    model = MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], max_steps=1,
                     padding_free=padding_free, max_length=None)
    trainer = train_on_responses_only(SFTTrainer(model=model, args=args, train_dataset=TALK, processing_class=words),
                                      instruction_part="user", response_part="hare")
    batch = trainer.data_collator([trainer.train_dataset[0], trainer.train_dataset[1]])
    kept = [words.convert_ids_to_tokens(int(i)) for i, label in zip(batch["input_ids"].flatten(), batch["labels"].flatten())
            if label != -100]
    # "fast" after the first example's "hare", "sleeps" after the second's (and its end-of-sequence)
    assert [t for t in kept if t != words.eos_token] == ["fast", "sleeps"]


LLAMA3 = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
          "{{ m['content'] }}<|eot_id|>{% endfor %}"
          "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")


def test_turn_markers_are_read_from_the_chat_template(chat_tokenizer):
    from conftest import CHATML

    assert turn_markers(chat_tokenizer(CHATML)) == ("<|im_start|>user\n", "<|im_start|>assistant\n")
    llama = chat_tokenizer(LLAMA3)
    llama.add_tokens(["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"], special_tokens=True)
    llama.bos_token = "<|im_start|>"  # any special token: the template writes it at the start only
    assert turn_markers(llama) == ("<|start_header_id|>user<|end_header_id|>\n\n",
                                   "<|start_header_id|>assistant<|end_header_id|>\n\n")


# A thinking model's template: its generation prompt opens a thinking block, and only the last answer has one
THINKING = ("{% for m in messages %}{% if m['role'] == 'assistant' and loop.last %}"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n{{ m['content'] }}<|im_end|>\n"
            "{% else %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}")


def test_a_thinking_templates_response_part_stops_before_the_thinking_block(chat_tokenizer):
    assert turn_markers(chat_tokenizer(THINKING)) == ("<|im_start|>user\n", "<|im_start|>assistant\n")


def test_a_template_without_a_generation_prompt_has_no_markers(chat_tokenizer):
    with pytest.raises(ValueError, match="generation prompt"):
        turn_markers(chat_tokenizer("{% for m in messages %}{{ m['role'] }} {{ m['content'] }}\n{% endfor %}"))
    with pytest.raises(ValueError, match="no chat template"):
        turn_markers(chat_tokenizer(None))


def test_trl_masks_a_template_with_generation_markers(chat_tokenizer):
    from conftest import CHATML, CHATML_MARKED

    assert trl_assistant_mask(chat_tokenizer(CHATML_MARKED))[0] is True
    ok, why = trl_assistant_mask(chat_tokenizer(CHATML))
    assert ok is False and "generation" in why
    # The end-of-turn token outside the generation markers: the model would not learn to stop
    unstoppable = CHATML_MARKED.replace("<|im_end|>{% endgeneration %}", "{% endgeneration %}<|im_end|>")
    ok, why = trl_assistant_mask(chat_tokenizer(unstoppable))
    assert ok is False and "end-of-turn" in why
