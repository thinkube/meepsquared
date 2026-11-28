# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A tiny random XLM-RoBERTa base, and tagger parts written in the tagger kind's format 1."""

import json

import pytest
import torch

HIDDEN, LAYERS, BOTTLENECK = 32, 2, 8
SPECIAL = ["<s>", "<pad>", "</s>", "<unk>"]
PIECES = ["▁", "今", "日", "は", "晴", "れ", "。", "雨", "今日", "晴れ", "the", "hare", "runs", "fast", "sloth"]


@pytest.fixture(scope="session")
def base(tmp_path_factory):
    """The folder of a random two-layer XLM-RoBERTa with a word-level tokenizer: each word one piece."""
    from tokenizers import Tokenizer, processors
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    folder = tmp_path_factory.mktemp("base")
    vocab = {piece: i for i, piece in enumerate(SPECIAL + PIECES)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[
        ("<s>", vocab["<s>"]), ("</s>", vocab["</s>"])])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                        cls_token="<s>", sep_token="</s>", pad_token="<pad>", unk_token="<unk>")
    tokenizer.save_pretrained(folder)
    torch.manual_seed(0)
    config = XLMRobertaConfig(vocab_size=len(vocab), hidden_size=HIDDEN, num_hidden_layers=LAYERS,
                              num_attention_heads=4, intermediate_size=64, max_position_embeddings=600,
                              pad_token_id=vocab["<pad>"], bos_token_id=0, eos_token_id=2)
    XLMRobertaModel(config, add_pooling_layer=False).save_pretrained(folder)
    return folder


def write_part(folder, lang, labels, seed=1, splitter=False, base_id="test/base", state_edit=None):
    """A tagger part in format 1: tagger.json, weights.pt with random adapters and heads, and meepserve.json."""
    from meepserve.tagger import Heads

    torch.manual_seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    state = {}
    for i in range(LAYERS):
        state[f"adapters.{i}.down.weight"] = torch.randn(BOTTLENECK, HIDDEN) * 0.3
        state[f"adapters.{i}.down.bias"] = torch.randn(BOTTLENECK) * 0.3
        state[f"adapters.{i}.up.weight"] = torch.randn(HIDDEN, BOTTLENECK) * 0.3
        state[f"adapters.{i}.up.bias"] = torch.randn(HIDDEN) * 0.3
    heads = Heads(HIDDEN, labels)
    for p in heads.parameters():
        torch.nn.init.normal_(p, std=0.5)
    state.update({f"heads.{k}": v for k, v in heads.state_dict().items()})
    if state_edit is not None:
        state_edit(state)
    torch.save(state, folder / "weights.pt")
    config = {"lang": lang, "bottleneck": BOTTLENECK, "dropout": 0.2, "labels": labels, "encoder": base_id}
    if splitter:
        config["input"] = "one character per piece"
    (folder / "tagger.json").write_text(json.dumps(config, ensure_ascii=False))
    (folder / "meepserve.json").write_text(json.dumps({"kind": "tagger", "format": 1, "base": base_id}))
    return folder, state
