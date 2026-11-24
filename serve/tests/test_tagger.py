# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The tagger kind: a part of format 1 loaded on the transformers engine answers as its adapters and heads compute,
splits text by characters, and refuses what it cannot answer."""

import json

import pytest
import torch
from torch import nn

import meepserve
from meepserve import PartError, tagger
from meepserve.engine import TransformersEncoder

from conftest import HIDDEN, LAYERS, write_part

PARSE_LABELS = {"upos": ["NOUN", "VERB", "ADP", "PUNCT"], "lemma": ["l|0|", "o|0|", "l|1|x"],
                "deprel": ["root", "nsubj", "case", "punct"]}
SENTENCES = [["今日", "は", "晴れ", "。"], ["the", "hare", "runs", "fast"], ["雨"]]


class _ByHand(nn.Module):
    def __init__(self, dense, down, up):
        super().__init__()
        self.dense, self.down, self.up = dense, down, up

    def forward(self, x):
        h = self.dense(x)
        return h + self.up(nn.functional.gelu(self.down(h)))


def reference(base, state, labels, words):
    """The labels and heads of one sentence computed without the package's engine: the adapters wrapped into a
    fresh copy of the encoder by hand."""
    from transformers import AutoTokenizer, XLMRobertaModel

    model = XLMRobertaModel.from_pretrained(base, add_pooling_layer=False).eval()
    for i, layer in enumerate(model.encoder.layer):
        dense = layer.output.dense
        down = nn.Linear(HIDDEN, state[f"adapters.{i}.down.weight"].shape[0])
        up = nn.Linear(down.out_features, HIDDEN)
        down.load_state_dict({"weight": state[f"adapters.{i}.down.weight"], "bias": state[f"adapters.{i}.down.bias"]})
        up.load_state_dict({"weight": state[f"adapters.{i}.up.weight"], "bias": state[f"adapters.{i}.up.bias"]})

        layer.output.dense = _ByHand(dense, down, up)
    heads = tagger.Heads(HIDDEN, labels)
    heads.load_state_dict({k[len("heads."):]: v for k, v in state.items() if k.startswith("heads.")})
    encoded = AutoTokenizer.from_pretrained(base)(words, is_split_into_words=True, return_tensors="pt")
    first = []
    for position, word in enumerate(encoded.word_ids()):
        if word is not None and word not in {encoded.word_ids()[p] for p in first}:
            first.append(position)
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state[0, first][None]
        return heads(hidden, torch.tensor([len(words)]))


def test_a_tagger_answers_as_its_adapters_and_heads_compute(base, tmp_path):
    folder, state = write_part(tmp_path / "parse", "ja", PARSE_LABELS)
    part = meepserve.load(folder, base, device="cpu")
    answer = part.answer({"model": "x", "sentences": SENTENCES, "probabilities": ["upos"]})["sentences"]
    for words, got in zip(SENTENCES, answer):
        out = reference(base, state, PARSE_LABELS, words)
        n = len(words)
        assert got["upos"] == [PARSE_LABELS["upos"][k] for k in out["upos"][0, :n].argmax(-1).tolist()]
        heads = out["arcs"][0, :n, :n + 1].argmax(-1)
        assert got["head"] == heads.tolist()
        assert got["deprel"] == [PARSE_LABELS["deprel"][k] for k in out["relations"](heads[None])[0].argmax(-1)]
        probabilities = got["probabilities"]["upos"]
        assert len(probabilities) == n and all(sum(p.values()) == pytest.approx(1.0, abs=1e-5) for p in probabilities)
        assert list(probabilities[0]) == PARSE_LABELS["upos"]


def test_a_lemma_rule_that_strips_more_than_the_word_has_is_not_chosen(base, tmp_path):
    def prefer_the_long_rule(state):
        state["heads.out.lemma.weight"].zero_()
        state["heads.out.lemma.bias"].copy_(torch.tensor([0.0, -1.0, 9.0]))

    folder, _ = write_part(tmp_path / "lemma", "lemma", {"lemma": ["l|0|", "o|0|", "l|5|x"]},
                           state_edit=prefer_the_long_rule)
    part = meepserve.load(folder, base, device="cpu")
    got = part.answer({"sentences": [["The", "hare"]]})["sentences"][0]
    # "l|5|x" strips five characters, more than either word has: the next best rule, lowercase as written
    assert got["lemma"] == ["the", "hare"]
    assert tagger.apply_rule("Hares", "l|1|") == "hare" and tagger.apply_rule("Kyoto", "o|0|") == "Kyoto"


def test_a_tree_has_one_root_when_asked(base, tmp_path):
    def many_roots(state):
        state["heads.parser.root"].mul_(30.0)

    folder, _ = write_part(tmp_path / "roots", "roots", PARSE_LABELS, state_edit=many_roots)
    part = meepserve.load(folder, base, device="cpu")
    words = [["the", "hare", "runs", "fast", "今日", "は", "晴れ"]]
    tree = part.answer({"sentences": words, "decode": "mst"})["sentences"][0]["head"]
    assert tree.count(0) == 1 and len(tree) == len(words[0])
    assert all(0 <= h <= len(tree) and h != i + 1 for i, h in enumerate(tree))


def splitter(base, folder, begins: bool):
    def label_every_character(state):
        state["heads.out.seg.weight"].zero_()
        state["heads.out.seg.bias"].copy_(torch.tensor([9.0, 0.0] if begins else [0.0, 9.0]))

    path, _ = write_part(folder, "seg", {"seg": ["B", "I"]}, splitter=True, state_edit=label_every_character)
    return meepserve.load(path, base, device="cpu")


def test_a_splitter_reads_one_character_per_piece_and_spaces_end_words(base, tmp_path):
    every = splitter(base, tmp_path / "b", begins=True)
    assert every.answer({"texts": ["今日は 晴れ。", "", "   "]})["words"] == [["今", "日", "は", "晴", "れ", "。"], [], []]
    whole = splitter(base, tmp_path / "i", begins=False)
    # A character after a space begins a word whatever its label: the words are the runs between spaces
    assert whole.answer({"texts": ["今日は  晴れ。 雨"]})["words"] == [["今日は", "晴れ。", "雨"]]


def test_a_text_longer_than_a_reading_is_split_in_windows_and_every_character_labelled(base, tmp_path):
    every = splitter(base, tmp_path / "b", begins=True)
    text = "今日は晴れ。" * 150  # 900 characters, two readings and more
    words = every.answer({"texts": [text]})["words"][0]
    assert "".join(words) == text and len(words) == len(text)


def test_two_parts_share_one_base(base, tmp_path):
    engine = TransformersEncoder(base, "cpu")
    first, _ = write_part(tmp_path / "a", "a", PARSE_LABELS, seed=1)
    second, _ = write_part(tmp_path / "b", "b", PARSE_LABELS, seed=2)
    a = meepserve.load(first, engine=engine)
    b = meepserve.load(second, engine=engine)
    assert a.engine is b.engine
    alone = meepserve.load(first, base, device="cpu")
    request = {"sentences": SENTENCES, "probabilities": ["upos"]}
    assert a.answer(request) == alone.answer(request)
    assert a.answer(request) != b.answer(request)
    with pytest.raises(ValueError, match="holds adapters named a already"):
        meepserve.load(first, engine=engine)


def test_what_cannot_be_answered_is_refused(base, tmp_path):
    folder, _ = write_part(tmp_path / "parse", "ja", PARSE_LABELS)
    part = meepserve.load(folder, base, device="cpu")
    for request, message in [
        ({"texts": ["今日"]}, "takes 'sentences'"),
        ({"sentences": [["今日", ""]]}, "non-empty words"),
        ({"sentences": [["今日"]], "probabilities": ["feats"]}, "names heads of this tagger"),
        ({"sentences": [["今日"]], "decode": "best"}, "'decode' is one of"),
        ({"sentences": [["今日"] * 600]}, "more than the 510"),
    ]:
        with pytest.raises(PartError, match=message):
            part.answer(request)
    assert part.answer({"sentences": [[]]})["sentences"] == [{"upos": [], "lemma": [], "head": [], "deprel": []}]


def test_a_part_that_does_not_fit_is_refused_when_it_loads(base, tmp_path):
    def drop_a_head(state):
        del state["heads.out.upos.weight"]

    broken, _ = write_part(tmp_path / "broken", "broken", PARSE_LABELS, state_edit=drop_a_head)
    with pytest.raises(PartError, match="do not fit its labels"):
        meepserve.load(broken, base, device="cpu")

    def drop_an_adapter(state):
        del state[f"adapters.{LAYERS - 1}.up.bias"]

    short, _ = write_part(tmp_path / "short", "short", PARSE_LABELS, state_edit=drop_an_adapter)
    with pytest.raises(ValueError, match="do not fit the base's 2 layers"):
        meepserve.load(short, base, device="cpu")
    later, _ = write_part(tmp_path / "later", "later", PARSE_LABELS)
    (later / "meepserve.json").write_text(json.dumps({"kind": "tagger", "format": 2, "base": "test/base"}))
    with pytest.raises(PartError, match="format 2 of the kind tagger is not known"):
        meepserve.load(later, base, device="cpu")
    (later / "meepserve.json").write_text(json.dumps({"kind": "chooser", "format": 1, "base": "test/base"}))
    with pytest.raises(PartError, match="no kind 'chooser' is installed"):
        meepserve.load(later, base, device="cpu")
    (later / "meepserve.json").write_text(json.dumps({"kind": "tagger", "format": 1, "base": "other/base"}))
    with pytest.raises(PartError, match="names the encoder test/base"):
        meepserve.load(later, base, device="cpu")


def test_the_tagger_kind_is_found_by_its_entry_point():
    assert meepserve.kinds()["tagger"] is tagger.KIND
