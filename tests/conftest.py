# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

WORDS = ["<unk>", "<pad>", "</s>", "the", "hare", "runs", "fast", "and", "sloth", "sleeps"]


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")


@pytest.fixture
def tiny_model():
    """A random two-layer Llama on the CPU, marked as meepsquared's from_pretrained marks it."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(config)
    model._meepsquared = {"padding_free": True, "liger": False}
    return model


# A ChatML chat template, and the same with TRL's {% generation %} markers around each answer
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
CHATML_MARKED = ("{% for m in messages %}{% if m['role'] == 'assistant' %}<|im_start|>assistant\n"
                 "{% generation %}{{ m['content'] }}<|im_end|>{% endgeneration %}\n"
                 "{% else %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


@pytest.fixture
def chat_tokenizer():
    """A function giving a tokenizer of WORDS, the roles, ChatML's and thinking's special tokens, and the template."""

    def make(template):
        words = WORDS + ["user", "assistant"]
        backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
                                            eos_token="</s>")
        tokenizer.add_tokens(["<|im_start|>", "<|im_end|>", "<think>", "</think>"], special_tokens=True)
        tokenizer.chat_template = template
        return tokenizer

    return make


@pytest.fixture
def tiny_moe():
    """A random two-layer Qwen3 mixture of experts on the CPU, marked as meepsquared's from_pretrained marks it."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, moe_intermediate_size=16, num_experts=4,
                            num_experts_per_tok=2, max_position_embeddings=64)
    model = Qwen3MoeForCausalLM(config)
    model._meepsquared = {"padding_free": True, "liger": False}
    return model
