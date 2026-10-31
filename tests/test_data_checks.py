# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The checks of SFT's data in the plan: rendering, length and cut answers, rows of no answer, overlap with the
evaluation data, the origin of the rows, and the loss weighted by it."""

import pytest
from datasets import Dataset

import meepsquared as m2
from meepsquared import _data_checks, _preprocess
from meepsquared.job import TrainingJob

from test_sessions import TEXT8, make_job, make_model

RECIPE = {"effective_batch": 2, "optimizer": "adamw_torch"}
LONG = "the hare runs fast and the sloth sleeps the hare runs fast"
# 10 tokens; 23 tokens with its answer within the first 12; 20 tokens with its answer past the 12th; and
# a user's turn alone, with no answer
CONVERSATIONS = [
    [{"role": "user", "content": "the hare"}, {"role": "assistant", "content": "runs fast"}],
    [{"role": "user", "content": "the"}, {"role": "assistant", "content": "fast"}, {"role": "user", "content": LONG}],
    [{"role": "user", "content": LONG}, {"role": "assistant", "content": "runs fast"}],
    [{"role": "user", "content": "the hare"}],
]


def chat_job(tiny_model, tokenizer, tmp_path, rows, recipe=None, **fields):
    tiny_model.resize_token_embeddings(len(tokenizer))
    return TrainingJob(model=tiny_model, tokenizer=tokenizer, data=Dataset.from_dict({"messages": rows}), steps=1,
                       output=str(tmp_path / "out"),
                       recipe=m2.Recipe(max_length=12, **RECIPE, **(recipe or {})),
                       overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0), **fields)


@pytest.mark.parametrize("template", ["CHATML", "CHATML_MARKED"])
def test_long_examples_cut_answers_and_rows_of_no_answer(tiny_model, chat_tokenizer, tmp_path, template):
    # CHATML: meepsquared's turn markers; CHATML_MARKED: TRL's assistant masks, which refuse a row of no answer
    import conftest

    job = chat_job(tiny_model, chat_tokenizer(getattr(conftest, template)), tmp_path, CONVERSATIONS)
    checks = job.plan()["data_checks"]
    assert checks.value["rows"] == 4
    assert checks.value["longer"] == {"count": 2, "rows": [1, 2]}
    assert checks.value["answers_cut"] == {"count": 1, "rows": [2]}
    assert checks.value["no_answer"] == {"count": 1, "rows": [3]}
    assert checks.value["overlap"] is None and checks.value["synthetic_share"] is None
    assert checks.text == ("4 rows render; 2 longer than 12 tokens, 1 of them with answer tokens cut; 1 without "
                           "answer tokens, left out")
    assert "answers cut: rows 2" in checks.reason and "no answer: rows 3" in checks.reason
    # The row of no answer is left out, and the others are cut to max_length
    trainer = job.trainer()
    assert len(trainer.train_dataset) == 3 and trainer._meepsquared_layout.examples == 3
    assert max(len(ids) for ids in trainer.train_dataset["input_ids"]) == 12


def test_max_cut_share_refuses_more_answers_cut(tiny_model, chat_tokenizer, tmp_path):
    from conftest import CHATML

    with pytest.raises(ValueError, match=r"cuts answer tokens of 1 of 4 examples \(25.0%\), more than max_cut_share "
                                         r"0.2: rows 2"):
        chat_job(tiny_model, chat_tokenizer(CHATML), tmp_path, CONVERSATIONS, {"max_cut_share": 0.2}).plan()
    assert chat_job(make_model(), chat_tokenizer(CHATML), tmp_path / "b", CONVERSATIONS,
                    {"max_cut_share": 0.25}).plan()["data_checks"].value["answers_cut"]["count"] == 1


def test_an_example_that_does_not_render_is_refused(tiny_model, chat_tokenizer, tmp_path):
    from conftest import CHATML

    template = CHATML.replace("{% for m in messages %}", "{% for m in messages %}{% if m['content'] == 'sloth' %}"
                              "{{ raise_exception('the sloth cannot render') }}{% endif %}")
    rows = [CONVERSATIONS[0], [{"role": "user", "content": "sloth"}, {"role": "assistant", "content": "fast"}]]
    with pytest.raises(ValueError, match=r"do not render through the chat template or the tokenizer: data rows 1 "
                                         r"\(.*the sloth cannot render\)"):
        chat_job(tiny_model, chat_tokenizer(template), tmp_path, rows).plan()


def test_an_evaluation_example_of_the_training_data_is_refused(tokenizer, tmp_path):
    eval_data = Dataset.from_dict({"text": ["the hare runs", "the sloth runs"]})  # row 1 is training row 2
    with pytest.raises(ValueError, match=r"1 evaluation examples are training examples: evaluation row 1 is training "
                                         r"row 2 \(the same text\)"):
        make_job(tokenizer, tmp_path / "a", eval_data=eval_data).plan()
    # The same _row id, of another text
    data = TEXT8.add_column("_row", [f"r{i}" for i in range(8)])
    other = Dataset.from_dict({"text": ["the hare runs"], "_row": ["r5"]})
    job = TrainingJob(model=make_model(), tokenizer=tokenizer, data=data, eval_data=other,
                      steps=2, output=str(tmp_path / "b"), recipe=m2.Recipe(**RECIPE),
                      overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0))
    with pytest.raises(ValueError, match=r"evaluation row 0 is training row 5 \(the same _row 'r5'\)"):
        job.plan()
    clean = make_job(tokenizer, tmp_path / "c", eval_data=Dataset.from_dict({"text": ["the hare runs"]})).plan()
    assert clean["data_checks"].value["overlap"] == 0
    assert "no evaluation example is a training example" in clean["data_checks"].text


def origins(kinds):
    return [{"kind": kind, "method": "pivot" if kind == "synthetic" else None} for kind in kinds]


MIXED = TEXT8.add_column("_origin", origins(["real", "synthetic"] * 4)).add_column("_row", list(range(8)))


def mixed_job(tokenizer, tmp_path, data=MIXED, **fields):
    """4 steps of 2 micro-batches of 2 rows: two epochs of the 8 rows."""
    recipe = fields.pop("recipe", {})
    return TrainingJob(model=make_model(), tokenizer=tokenizer, data=data, steps=4, output=str(tmp_path),
                       recipe=m2.Recipe(effective_batch=4, optimizer="adamw_torch", **recipe),
                       overrides=m2.Overrides(micro_batch=2, checkpointed_layers=0), **fields)


def test_the_origin_of_the_rows(tokenizer, tmp_path):
    job = mixed_job(tokenizer, tmp_path / "a")
    checks = job.plan()["data_checks"]
    assert checks.value["synthetic_share"] == 0.5 and checks.text.endswith("50% of the training rows synthetic")
    # Metadata: never in the data the trainer tokenizes and trains
    assert not {"_row", "_origin"} & set(job.trainer().train_dataset.column_names)
    # A synthetic evaluation row is refused
    synthetic = Dataset.from_dict({"text": ["the hare runs"], "_origin": origins(["synthetic"])})
    with pytest.raises(ValueError, match=r"1 evaluation examples are synthetic .*rows 0"):
        mixed_job(tokenizer, tmp_path / "b", eval_data=synthetic).plan()
    # A held-out share is taken from the real rows
    held = mixed_job(tokenizer, tmp_path / "c", eval_share=0.25)
    held.plan()
    trainer = held.trainer()
    assert len(trainer.eval_dataset) == 1 and len(trainer.train_dataset) == 7
    # Too few real rows to hold any out
    one_real = TEXT8.add_column("_origin", origins(["real"] + ["synthetic"] * 7))
    for data, real in ((one_real, 1), (TEXT8.add_column("_origin", origins(["synthetic"] * 8)), 0)):
        with pytest.raises(ValueError, match=rf"eval_share=0.25 holds out real rows only \(the data has _origin\), "
                                             rf"and the data has {real} real rows: too few to hold out any"):
            mixed_job(tokenizer, tmp_path / f"r{real}", data=data, eval_share=0.25).plan()
    # A row without a kind is refused
    unmarked = TEXT8.add_column("_origin", origins(["real"] * 7) + [{"kind": None, "method": None}])
    with pytest.raises(ValueError, match=r"the _origin of rows \[7\]"):
        mixed_job(tokenizer, tmp_path / "d", data=unmarked).plan()


def test_source_weights_weight_each_micro_batch_loss(tokenizer, tmp_path):
    job = mixed_job(tokenizer, tmp_path / "a", recipe={"source_weights": {"synthetic": 0.5}})
    plan = job.plan()
    assert plan["source_weights"].text == "synthetic rows x0.5; every other kind x1"
    trainer = job.trainer()
    kinds = ["real", "synthetic"] * 4
    laid = trainer._meepsquared_layout
    # Each micro-batch holds rows of one kind
    assert all(len({kinds[i] for i in batch}) == 1 for epoch in laid.epochs for batch in epoch)
    loader = trainer.get_train_dataloader()
    seen = set()
    for batch in loader:
        weight = batch[_data_checks.WEIGHT]
        plain = type(trainer).compute_loss(trainer, trainer.model,
                                           {k: v for k, v in batch.items() if k != _data_checks.WEIGHT})
        weighted = trainer.compute_loss(trainer.model, dict(batch))
        assert weighted.item() == pytest.approx(weight * plain.item(), rel=1e-6)
        seen.add(weight)
    assert seen == {0.5, 1.0}
    assert job.train().steps == 4
    # Without _origin, weights are refused
    with pytest.raises(ValueError, match="no _origin column"):
        mixed_job(tokenizer, tmp_path / "b", data=TEXT8, recipe={"source_weights": {"synthetic": 0.5}}).plan()


@pytest.mark.parametrize("recipe, message", [
    ({"max_cut_share": 1.5}, "max_cut_share"),
    ({"source_weights": {"generated": 0.5}}, "source_weights"),
    ({"source_weights": {"synthetic": 0}}, "source_weights"),
])
def test_wrong_data_settings_are_refused(recipe, message):
    with pytest.raises(ValueError, match=message):
        m2.Recipe(**recipe)
    with pytest.raises(ValueError, match="SFT settings"):
        TrainingJob(model="m", data="d", task="grpo", rewards=["json:dumps"],
                    recipe=m2.Recipe(max_cut_share=0.1))


def test_the_normalized_text_and_the_named_rows():
    assert _data_checks.normalized("The  Hare\u00a0runs\n\tFAST") == "the hare runs fast"
    assert _data_checks.rows_text(list(range(7)), None) == "0, 1, 2, 3, 4 and 2 more"
    # Only the first rows are kept; the count says how many there are
    assert _data_checks.rows_text([0, 1, 2, 3, 4], ["a", "b", "c", "d", "e"], 7) == \
        "0 ('a'), 1 ('b'), 2 ('c'), 3 ('d'), 4 ('e') and 2 more"
    merged = _preprocess.merged({"count": 2, "rows": [4, 9], "ids": ["d", "i"]},
                                {"count": 1, "rows": [6], "ids": ["f"]})
    assert merged == {"count": 3, "rows": [4, 6, 9], "ids": ["d", "f", "i"]}
    assert _preprocess.merged({"count": 1, "rows": [4], "ids": None}, {"count": 0, "rows": [], "ids": None}) == \
        {"count": 1, "rows": [4], "ids": None}


def test_the_passes_keep_facts_alone(tokenizer):
    # The loss pass: lengths after the cut, and the booleans of the checks
    batch = {"input_ids": [[3, 4, 5, 6, 7], [3, 4], []], "labels": [[-100, -100, 5, 6, 7], [-100, -100], []]}
    facts = _data_checks.loss_facts(batch, completion_only=False, markers=None, max_length=3)
    assert facts == {"_length": [3, 2, 0], "_empty": [False, False, True], "_has_loss": [True, False, False],
                     "_cut_loss": [True, False, False], "_longer": [True, False, False]}
    # The hash pass: 16 bytes per row, the same for the same text
    hashes = _data_checks.text_hashes({"input_ids": [[3, 4], [3, 4], [5]]}, tokenizer)["_hash"]
    assert len(hashes[0]) == 16 and hashes[0] == hashes[1] != hashes[2]
    # The origin pass
    assert _data_checks.kinds_of({"_origin": [{"kind": "real"}, {"kind": "synthetic"}]}, [0, 1], "data") == \
        {"_kind": ["real", "synthetic"]}
    with pytest.raises(ValueError, match=r"_origin kinds \[\(1, 'made'\)\] are not one of"):
        _data_checks.kinds_of({"_origin": [{"kind": "real"}, {"kind": "made"}]}, [0, 1], "data")
