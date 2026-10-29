# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A training job's evaluation: data given apart or a share held out, every so many steps, its loss counted as the
training's, in the plan, the status file and the result, and continued by a resume."""

import json

import pytest
import torch
from datasets import Dataset, DatasetDict

import meepsquared as m2
from meepsquared import _batches, _sessions, cli
from meepsquared.job import TrainingJob

from test_sessions import STEPS, TEXT8, StopAt, jobs_from_yaml, make_job

# Evaluation examples none of the training data's
EVAL_TEXT = Dataset.from_dict({"text": ["the hare runs", "the sloth sleeps fast", "and the sloth runs fast"]})
CHAT_EVAL = Dataset.from_dict({"messages": [[{"role": "user", "content": "the sloth"},
                                             {"role": "assistant", "content": "runs fast"}]] * 2})
CHAT = Dataset.from_dict({"messages": [[{"role": "user", "content": "the hare"},
                                        {"role": "assistant", "content": "runs fast"}]] * 2})


@pytest.mark.parametrize("fields, message", [
    ({"eval_data": "org/eval", "eval_share": 0.1}, "not both"),
    ({"eval_split": "test"}, "no eval_data is given"),
    ({"eval_share": 0}, "eval_share"),
    ({"eval_share": 0.6}, "eval_share"),
    ({"eval_share": True}, "eval_share"),
    ({"eval_share": "0.1"}, "eval_share"),
    ({"eval_share": 0.1, "eval_every": 0}, "eval_every"),
    ({"eval_every": 10}, "no evaluation data"),
    ({"task": "grpo", "rewards": ["json:dumps"], "eval_share": 0.1}, "SFT settings"),
])
def test_a_wrong_evaluation_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        TrainingJob(model="org/model", data="org/data", **fields)


def test_a_loaded_evaluation_set_is_not_data():
    with pytest.raises(ValueError, match="eval_data is a loaded dataset"):
        TrainingJob(model="org/model", data="org/data", eval_data=TEXT8).to_dict()
    job = TrainingJob(model="org/model", data="org/data", eval_data="org/eval", eval_split="validation",
                      eval_every=50)
    assert job.to_dict()["eval_split"] == "validation" and TrainingJob.from_dict(job.to_dict()) == job


@pytest.mark.parametrize("fields, message", [
    ({"eval_data": TEXT8, "eval_split": "test"}, "a datasets.Dataset, which has no splits"),
    ({"eval_data": DatasetDict({"validation": TEXT8})}, r"required, eval_data has the splits \['validation'\]"),
    ({"eval_data": DatasetDict({"validation": TEXT8}), "eval_split": "test"}, "no split 'test'"),
    ({"eval_data": CHAT}, "eval_data holds conversations"),
])
def test_evaluation_data_that_does_not_fit_is_refused_when_planned(tokenizer, tmp_path, fields, message):
    with pytest.raises(ValueError, match=message):
        make_job(tokenizer, tmp_path / "out", **fields).plan()


def test_a_split_of_a_file_is_refused(tokenizer, tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in EVAL_TEXT["text"]))
    with pytest.raises(ValueError, match="is a file, which has no splits"):
        make_job(tokenizer, tmp_path / "out", eval_data=str(path), eval_split="test").plan()
    plan = make_job(tokenizer, tmp_path / "out2", eval_data=str(path)).plan()
    assert plan["evaluation"].value["rows"] == 3


def test_the_share_held_out_is_seeded_and_disjoint():
    trained, held = _batches.held_out(20, 0.25, 42)
    assert (trained, held) == _batches.held_out(20, 0.25, 42)
    assert len(held) == 5 and not set(trained) & set(held) and sorted(trained + held) == list(range(20))
    assert held != _batches.held_out(20, 0.25, 7)[1]
    with pytest.raises(ValueError, match="leaves none to train on"):
        _batches.held_out(1, 0.5, 42)


def ids(dataset) -> list:
    return sorted(tuple(row["input_ids"]) for row in dataset)


def test_the_share_is_held_out_before_the_layout(tokenizer, tmp_path):
    job = make_job(tokenizer, tmp_path / "a", eval_share=0.25)
    plan = job.plan()
    trainer = job.trainer()
    # 2 of the 8 rows evaluate, the other 6 train, and the micro-batches are laid out from those 6
    assert len(trainer.eval_dataset) == 2 and len(trainer.train_dataset) == 6
    assert trainer._meepsquared_layout.examples == 6
    whole = make_job(tokenizer, tmp_path / "b").trainer().train_dataset
    assert sorted(ids(trainer.eval_dataset) + ids(trainer.train_dataset)) == ids(whole)
    assert plan["data"].text.endswith("8 rows of plain text, 2 of them held out for evaluation")
    evaluation = plan["evaluation"]
    assert evaluation.by == "auto" and evaluation.value == {"data": "a datasets.Dataset, held out with seed 42",
                                                            "rows": 2, "every": 1}
    assert evaluation.text.startswith("2 rows of a datasets.Dataset, held out with seed 42, every 1 steps")
    assert "about 10 evaluations over the" in evaluation.reason
    # The same rows each time the job is planned
    again = make_job(tokenizer, tmp_path / "c", eval_share=0.25).trainer()
    assert ids(again.eval_dataset) == ids(trainer.eval_dataset)
    assert make_job(tokenizer, tmp_path / "d").plan()["evaluation"].text == "none"


def kept(batch, tokenizer) -> list:
    """The tokens of a batch that count in the loss, without end-of-sequence tokens."""
    tokens = [tokenizer.convert_ids_to_tokens(int(i))
              for i, label in zip(batch["input_ids"].flatten(), batch["labels"].flatten()) if label != -100]
    return [token for token in tokens if token != tokenizer.eos_token]


@pytest.mark.parametrize("template", ["CHATML", "CHATML_MARKED"])
def test_the_evaluation_loss_is_masked_as_the_training_loss(tiny_model, chat_tokenizer, tmp_path, template):
    import conftest

    tokenizer = chat_tokenizer(getattr(conftest, template))
    tiny_model.resize_token_embeddings(len(tokenizer))
    job = TrainingJob(model=tiny_model, tokenizer=tokenizer, data=CHAT, eval_data=CHAT_EVAL, steps=1,
                      output=str(tmp_path / "out"), recipe=m2.Recipe(effective_batch=2, optimizer="adamw_torch"),
                      overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0))
    trainer = job.trainer()
    evaluated = [kept(batch, tokenizer) for batch in trainer.get_eval_dataloader()]
    trained = [kept(trainer.data_collator([trainer.train_dataset[i]]), tokenizer)
               for i in range(len(trainer.train_dataset))]
    assert evaluated == trained == [["runs", "fast", "<|im_end|>"]] * 2
    assert trainer.args.prediction_loss_only and trainer.args.per_device_eval_batch_size == 1


def test_the_status_and_the_result_carry_the_evaluations(tokenizer, tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    writes = []
    write = _sessions.write_json

    def recording(path, data):
        if path == status:
            writes.append(json.loads(json.dumps(data)))
        write(path, data)

    monkeypatch.setattr(_sessions, "write_json", recording)
    job = make_job(tokenizer, tmp_path / "out", eval_share=0.25, eval_every=2)
    plan = job.plan()
    evaluation = plan["evaluation"]
    assert evaluation.by == "user" and evaluation.value["every"] == 2 and evaluation.auto_value["every"] == 1
    assert "every 2 steps (3 evaluations)" in evaluation.text
    results = []
    run = job.train

    def train(**kwargs):
        results.append(run(**kwargs))
        return results[-1]

    monkeypatch.setattr(job, "train", train)
    jobs_from_yaml(monkeypatch, [job])
    assert cli.main(["train", "job.yaml", "--status-file", str(status)]) == 0
    evaluations = results[0].evaluations
    assert [e["step"] for e in evaluations] == [2, 4, 6] and all(isinstance(e["eval_loss"], float)
                                                               for e in evaluations)
    assert results[0].to_dict()["evaluations"] == evaluations
    training = [w for w in writes if w["state"] == "training"]
    assert all(w["eval_loss"] is None and w["eval_step"] is None for w in training if w["step"] < 2)
    at_four = [w for w in training if w["step"] == 4][-1]
    assert at_four["eval_step"] == 4 and at_four["eval_loss"] == pytest.approx(evaluations[1]["eval_loss"])
    assert writes[-1]["eval_step"] == 6 and writes[-1]["eval_loss"] == pytest.approx(evaluations[2]["eval_loss"])


def test_the_estimate_counts_the_evaluations(tokenizer, tmp_path):
    plan = make_job(tokenizer, tmp_path / "out", eval_share=0.25, eval_every=2, timed_steps=1).plan()
    duration = plan["duration"]
    assert "with 3 evaluations" in duration.text and "each evaluation 2 forward passes" in duration.reason
    assert plan.estimated_seconds > plan.seconds_per_step * STEPS


def test_a_resumed_training_continues_its_evaluations_on_the_same_rows(tokenizer, tmp_path):
    whole = make_job(tokenizer, tmp_path / "whole", eval_share=0.25, eval_every=2)
    expected = whole.train().evaluations

    stop = tmp_path / "STOP"
    first = make_job(tokenizer, tmp_path / "out", eval_share=0.25, eval_every=2)
    first.trainer().add_callback(StopAt(3, stop))
    assert first.train(stop_file=stop).paused
    stop.unlink()
    # A resume holding out another share is planned otherwise
    other = make_job(tokenizer, tmp_path / "out", eval_share=0.5, eval_every=2)
    with pytest.raises(ValueError, match="evaluation"):
        other.train(resume=True)
    second = make_job(tokenizer, tmp_path / "out", eval_share=0.25, eval_every=2)
    assert ids(second.trainer().eval_dataset) == ids(first.trainer().eval_dataset)
    status = tmp_path / "status.json"
    resumed = second.train(resume=True, status_file=status)
    assert [e["step"] for e in resumed.evaluations] == [2, 4, 6]
    for got, want in zip(resumed.evaluations, expected):
        assert got["eval_loss"] == pytest.approx(want["eval_loss"], rel=1e-5, abs=1e-6)
    assert json.loads(status.read_text())["eval_step"] == 6


def test_a_dataset_dict_split_evaluates(tokenizer, tmp_path):
    job = make_job(tokenizer, tmp_path / "out", eval_data=DatasetDict({"validation": EVAL_TEXT}),
                   eval_split="validation")
    assert job.plan()["evaluation"].value["rows"] == 3 and len(job.trainer().train_dataset) == 8
    assert torch.is_tensor(next(iter(job.trainer().get_eval_dataloader()))["input_ids"])


def test_the_memory_counts_the_evaluation_pass(tokenizer, tmp_path):
    plan = make_job(tokenizer, tmp_path / "out", eval_share=0.25, timed_steps=1).plan()
    assert "and the evaluation pass" in plan["memory"].reason and plan.memory_request_bytes >= plan.peak_host_bytes
    assert "timed forward passes of its largest batch" in plan["duration"].reason
