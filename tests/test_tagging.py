# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import itertools
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

import meepsquared as m2
from meepsquared import _batches, _checkpointing, _tagging

VOCAB = ["<s>", "<pad>", "</s>", "<unk>", "the", "hare", "runs", "fast", "and", "sloth", "sleeps", "del", "de", "el",
         "camino", ".", "a", "slow"]

CONLLU = """# sent_id = 1
1-2\tdel\t_\t_\t_\t_\t_\t_\t_\t_
1\tde\tde\tADP\tSP\t_\t3\tcase\t_\t_
2\tel\tel\tDET\tDA\tDefinite=Def\t3\tdet\t_\t_
3\tcamino\tcamino\tNOUN\tNC\tGender=Masc\t0\troot\t_\t_
3.1\tél\tél\tPRON\t_\t_\t_\t_\t3:nsubj\t_

# sent_id = 2
1\tthe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\thare\thare\tNOUN\tNN\tNumber=Sing\t3\tnsubj\t_\t_
3\truns\trun\tVERB\tVBZ\t_\t0\troot\t_\t_
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

"""


@pytest.fixture
def encoder_path(tmp_path):
    """A random two-layer XLM-RoBERTa and a word-level tokenizer with its special tokens, saved."""
    torch.manual_seed(0)
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                        unk_token="<unk>", pad_token="<pad>", model_max_length=32)
    config = XLMRobertaConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                              intermediate_size=64, max_position_embeddings=34, pad_token_id=1)
    path = tmp_path / "encoder"
    XLMRobertaModel(config, add_pooling_layer=False).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture
def treebank(tmp_path):
    sentences = [["the", "hare", "runs", "fast", "."], ["the", "sloth", "sleeps", "."],
                 ["a", "slow", "sloth", "and", "the", "hare", "."], ["the", "hare", "sleeps", "."]]
    upos = {"the": "DET", "a": "DET", "hare": "NOUN", "sloth": "NOUN", "runs": "VERB", "sleeps": "VERB",
            "fast": "ADV", "slow": "ADJ", "and": "CCONJ", ".": "PUNCT"}
    rows = []
    for words in sentences * 3:
        verb = next(i for i, w in enumerate(words) if upos[w] == "VERB") + 1
        rows.append({"tokens": words, "upos": [upos[w] for w in words],
                     "lemma": [w[:-1] if w.endswith("s") and upos[w] == "VERB" else w for w in words],
                     "head": [0 if i + 1 == verb else verb for i in range(len(words))],
                     "deprel": ["root" if i + 1 == verb else "dep" for i in range(len(words))]})
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps(r) for r in rows))
    dev = tmp_path / "dev.jsonl"
    dev.write_text("\n".join(json.dumps(r) for r in rows[:4]))
    return str(train), str(dev)


def test_conllu_as_syntactic_words(tmp_path):
    path = tmp_path / "t.conllu"
    path.write_text(CONLLU)
    first, second = _tagging.read_conllu(path)
    assert first["tokens"] == ["de", "el", "camino"]
    assert first["head"] == [3, 3, 0] and first["deprel"] == ["case", "det", "root"]
    assert second["lemma"] == ["the", "hare", "run", "."]


def test_conllu_as_surface_tokens(tmp_path):
    path = tmp_path / "t.conllu"
    path.write_text(CONLLU)
    first = _tagging.read_conllu(path, words="surface")[0]
    assert first["tokens"] == ["del", "camino"]
    assert first["upos"] == ["ADP+DET", "NOUN"]
    assert first["lemma"] == ["de+el", "camino"]
    assert first["feats"] == ["Definite=Def", "Gender=Masc"]
    # The multi-word token's head is that of its first part whose head lies outside it
    assert first["head"] == [2, 0] and first["deprel"] == ["case", "root"]


@pytest.mark.parametrize("form, lemma", [("runs", "run"), ("Paris", "Paris"), ("The", "the"), ("dárselo", "dar+él+él"),
                                         ("went", "go")])
def test_lemma_rules_give_the_lemma_back(form, lemma):
    assert _tagging.apply_rule(form, _tagging.lemma_rule(form, lemma)) == lemma


def tree_score(scores, heads):
    return sum(scores[d, heads[d]] for d in range(1, len(heads)))


def is_tree(heads) -> bool:
    for start in range(1, len(heads)):
        seen, node = set(), start
        while node != 0:
            if node in seen:
                return False
            seen.add(node)
            node = heads[node]
    return True


@pytest.mark.parametrize("seed", range(20))
def test_the_tree_with_one_root_is_the_best_one(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    scores = rng.normal(size=(n + 1, n + 1))
    heads = _tagging.one_root_tree(scores)
    assert is_tree(heads) and list(heads[1:]).count(0) == 1
    best = max((h for h in itertools.product(range(n + 1), repeat=n)
                if all(h[d] != d + 1 for d in range(n)) and h.count(0) == 1 and is_tree((-1,) + h)),
               key=lambda h: tree_score(scores, (-1,) + h))
    assert tree_score(scores, heads) == pytest.approx(tree_score(scores, (-1,) + best))


def test_by_length_groups_sentences_of_about_one_length():
    lengths = [5, 30, 6, 29, 7, 31, 5, 30]
    loading = _batches.Loading(seed=0, drop_last=False, padded=True, multiple=1, processes=1, split_batches=False,
                               even_batches=True)
    layout = _batches.lay_out(lengths, loading, 1, None, 1, budget=64, by_length=True)
    for group in layout.epochs[0]:
        assert max(lengths[i] for i in group) - min(lengths[i] for i in group) <= 2
    assert sorted(i for group in layout.epochs[0] for i in group) == list(range(8))


@pytest.mark.parametrize("method", ["bottleneck", "lora"])
def test_a_tagger_learns(encoder_path, treebank, method):
    from transformers import AutoModel, AutoTokenizer

    rows = [json.loads(line) for line in open(treebank[0])]
    names = _tagging.tasks_of(list(rows[0]), None)
    assert names == ["upos", "lemma", "parse"]
    tasks = _tagging.build_tasks(rows, names)
    encoder = AutoModel.from_pretrained(encoder_path, add_pooling_layer=False)
    config = {"method": method, "dropout": 0.0, "bottleneck": 8, "rank": 4, "alpha": 4, "lora_dropout": 0.0,
              "targets": ["query", "value", "dense"]}
    tagger = _tagging.build(encoder, config, tasks)
    assert not any(p.requires_grad for n, p in tagger.named_parameters()
                   if n.startswith("encoder.") and "adapter" not in n and "lora_" not in n)
    assert len(_checkpointing.layers(tagger)) == 2
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    examples, dropped = _tagging.encode(tokenizer, rows, tasks, 32, "data")
    assert not dropped and examples[0]["first"] == [1, 2, 3, 4, 5]
    batch = _tagging.Collator(tokenizer.pad_token_id, tagger.label_columns)(examples)
    optimizer = torch.optim.AdamW([p for p in tagger.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(30):
        loss = tagger(**batch)["loss"]
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert losses[-1] < losses[0] / 2


def test_checkpointing_reaches_the_encoder_layers(encoder_path, treebank):
    from transformers import AutoModel

    rows = [json.loads(line) for line in open(treebank[0])]
    tasks = _tagging.build_tasks(rows, ["upos"])
    encoder = AutoModel.from_pretrained(encoder_path, add_pooling_layer=False)
    tagger = _tagging.build(encoder, {"method": "bottleneck", "dropout": 0.0, "bottleneck": 8}, tasks)
    _checkpointing.set_checkpointed(tagger, 1)
    assert _checkpointing.kinds(tagger) == ["whole", "none"]
    assert encoder.is_gradient_checkpointing


def test_a_tagging_job_trains_scores_saves_and_predicts(encoder_path, treebank, tmp_path):
    train, dev = treebank
    job = m2.TrainingJob(model=encoder_path, task="tagging", method="bottleneck", data=train, eval_data=dev,
                         epochs=3, output=str(tmp_path / "out"), timed_steps=0,
                         recipe=m2.Recipe(learning_rate=1e-2, bottleneck=8, tokens_per_update=64, head_dropout=0.0))
    plan = job.plan()
    assert plan["labels"].value == ["upos", "lemma", "parse"]
    assert plan["keep"].value == "best"
    assert plan["decode"].value == "mst"
    result = job.train()
    assert [r["epoch"] for r in result.evaluations] == [1, 2, 3]
    assert {"upos", "lemma", "uas", "las", "score"} <= set(result.evaluations[0])
    result.save(tmp_path / "tagger")
    tagger = m2.MeepTagger.from_pretrained(tmp_path / "tagger", encoder=encoder_path, device="cpu")
    found = tagger.predict([["the", "hare", "runs", "."], ["a", "sloth", "sleeps", "fast", "."]])
    assert len(found[1]["upos"]) == 5 and len(found[1]["head"]) == 5
    assert found[0]["head"].count(0) == 1
    with pytest.raises(ValueError, match="merged=True"):
        result.save(tmp_path / "merged", merged=True)


def test_a_tagging_job_refuses_language_model_settings(treebank):
    train, _ = treebank
    with pytest.raises(ValueError, match="effective_batch"):
        m2.TrainingJob(model="x", task="tagging", method="bottleneck", data=train,
                       recipe=m2.Recipe(effective_batch=8))
    with pytest.raises(ValueError, match="a tagger trains"):
        m2.TrainingJob(model="x", task="tagging", method="qlora", data=train)
    with pytest.raises(ValueError, match="eval_every"):
        m2.TrainingJob(model="x", task="tagging", method="bottleneck", data=train, eval_data=train, eval_every=5)
    with pytest.raises(ValueError, match="LoRA setting"):
        m2.TrainingJob(model="x", task="tagging", method="bottleneck", data=train, recipe=m2.Recipe(rank=8))


def test_a_language_model_job_refuses_tagging_settings():
    with pytest.raises(ValueError, match="trains a tagger"):
        m2.TrainingJob(model="x", data="d.jsonl", method="bottleneck")
    with pytest.raises(ValueError, match="tagging setting"):
        m2.TrainingJob(model="x", data="d.jsonl", recipe=m2.Recipe(keep="best"))
    with pytest.raises(ValueError, match="linear"):
        m2.TrainingJob(model="x", data="d.jsonl", recipe=m2.Recipe(schedule="linear"))


def test_a_tagging_job_with_several_files_is_data(treebank):
    train, dev = treebank
    job = m2.TrainingJob(model="x", task="tagging", method="lora", data=[train, dev], eval_data=dev, epochs=2)
    again = m2.TrainingJob.from_dict(job.to_dict())
    assert again.data == [train, dev] and again.task == "tagging"
