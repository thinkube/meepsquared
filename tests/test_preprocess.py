# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""SFT's data preprocessed once into a folder keyed by what made it: found again by a job of the same inputs, refused
by a job of other inputs, read memory-mapped by the plan."""

import json
from pathlib import Path

import pytest
from datasets import Dataset

import meepsquared as m2
from meepsquared import _preprocess, cli
from meepsquared.job import TrainingJob

from test_sessions import TEXT8, make_job, make_model

KEYED = {"key_version", "data", "eval_data", "kind", "tokenizer", "answers", "markers", "max_length", "eval_share",
         "seed", "versions"}


def test_the_data_is_built_once_and_found_by_the_same_inputs(tokenizer, tmp_path):
    first = make_job(tokenizer, tmp_path / "a").preprocess(tmp_path / "cache")
    assert not first.found and first.folder == str(tmp_path / "cache" / first.key)
    folder = Path(first.folder)
    assert (folder / "train").is_dir() and not (folder / "eval").exists() and not (folder / "cache").exists()
    assert set(json.loads((folder / "key.json").read_text())) == KEYED
    assert first.report["decision"]["text"] == "8 rows render; 0 longer than 1024 tokens, 0 of them with answer " \
                                               "tokens cut; 0 without answer tokens, left out"
    assert (first.report["rows"], first.report["trained_rows"], first.report["held"]) == (8, 8, 0)
    again = make_job(tokenizer, tmp_path / "b").preprocess(tmp_path / "cache")
    assert again.found and again.key == first.key and again.report == first.report
    # The plan reads the folder, checks it is this job's, and the trainer gets the tokens alone
    job = make_job(tokenizer, tmp_path / "c", preprocessed=first.folder)
    plan = job.plan()
    assert plan["data_checks"].text == first.report["decision"]["text"]
    assert plan["data"].text == "a datasets.Dataset: 8 rows of plain text"
    assert set(job.trainer().train_dataset.column_names) == {"input_ids", "labels"}
    assert job.train().steps == 6


def test_a_plan_without_a_folder_preprocesses_into_its_output(tokenizer, tmp_path):
    job = make_job(tokenizer, tmp_path / "out")
    job.plan()
    folders = list((tmp_path / "out" / "preprocessed").iterdir())
    assert len(folders) == 1 and (folders[0] / "key.json").exists()


def test_other_inputs_make_another_key_and_a_folder_of_other_inputs_is_refused(tokenizer, tmp_path):
    first = make_job(tokenizer, tmp_path / "a").preprocess(tmp_path / "cache")
    shorter = TrainingJob(model=make_model(), tokenizer=tokenizer, data=TEXT8, steps=2, output=str(tmp_path / "b"),
                          recipe=m2.Recipe(effective_batch=2, optimizer="adamw_torch", max_length=3),
                          overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0))
    second = shorter.preprocess(tmp_path / "cache")
    assert not second.found and second.key != first.key
    assert second.report["decision"]["text"].startswith("8 rows render; 3 longer than 3 tokens")
    with pytest.raises(ValueError, match="made from other inputs than this job's: max_length differ"):
        make_job(tokenizer, tmp_path / "c", preprocessed=second.folder).plan()
    with pytest.raises(ValueError, match="is not a preprocessing folder"):
        make_job(tokenizer, tmp_path / "d", preprocessed=str(tmp_path)).plan()
    with pytest.raises(ValueError, match="preprocessed is an SFT setting"):
        TrainingJob(model="m", data="d", task="grpo", rewards=["json:dumps"], preprocessed=str(tmp_path))


def test_the_evaluation_data_is_preprocessed_beside_the_training_data(tokenizer, tmp_path):
    held = make_job(tokenizer, tmp_path / "a", eval_share=0.25).preprocess(tmp_path / "cache")
    assert (held.report["held"], held.report["eval_rows"], held.report["trained_rows"]) == (2, 2, 6)
    assert held.report["eval_source"] == "a datasets.Dataset, held out with seed 42"
    given = make_job(tokenizer, tmp_path / "b",
                     eval_data=Dataset.from_dict({"text": ["the hare runs"]})).preprocess(tmp_path / "cache")
    assert given.key != held.key and (Path(given.folder) / "eval").is_dir()
    assert given.report["eval_rows"] == 1 and given.report["held"] == 0


def test_the_key_is_the_inputs_hashed(tokenizer, tmp_path):
    inputs = _preprocess.inputs({"hash": "x"}, None, tokenizer, "plain text", {}, None, 1024, None)
    assert set(inputs) == KEYED and inputs["seed"] == 42
    assert _preprocess.key_of(inputs) == _preprocess.key_of(json.loads(json.dumps(inputs)))
    assert _preprocess.key_of(inputs) != _preprocess.key_of({**inputs, "max_length": 512})
    assert _preprocess.differences({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == ["b", "c"]
    # A file by its bytes
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "the hare"}\n')
    first = _preprocess.data_identity(str(data), None, None)
    assert set(first) == {"file", "bytes", "hash"}
    data.write_text('{"text": "the sloth"}\n')
    assert _preprocess.data_identity(str(data), None, None)["hash"] != first["hash"]
    # A loaded dataset by its fingerprint; the tokenizer by its vocabulary, template and special tokens
    assert _preprocess.data_identity(TEXT8, None, TEXT8) == {"fingerprint": TEXT8._fingerprint}
    identity = _preprocess.tokenizer_identity(tokenizer)
    assert set(identity) == {"class", "vocabulary", "chat_template", "special_tokens"}
    assert identity["chat_template"] is None and identity["special_tokens"]["pad_token"] == "<pad>"


def test_the_command_preprocesses_a_job_file(tokenizer, tmp_path, capsys):
    model = tmp_path / "model"
    make_model().save_pretrained(model)
    tokenizer.save_pretrained(model)
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"text": text}) + "\n" for text in TEXT8["text"]))
    job = tmp_path / "job.yaml"
    job.write_text(f"model: {model}\ndata: {data}\noutput: {tmp_path / 'out'}\nsteps: 2\n"
                   "recipe:\n  effective_batch: 2\n")
    assert cli.main(["preprocess", str(job), "--cache", str(tmp_path / "cache"), "--json"]) == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(outcome) == {"folder", "key", "found", "report"} and not outcome["found"]
    assert outcome["report"]["rows"] == 8 and outcome["report"]["source"] == str(data)
    assert json.loads((Path(outcome["folder"]) / "key.json").read_text())["data"]["file"] == str(data)
    assert cli.main(["preprocess", str(job), "--cache", str(tmp_path / "cache"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["found"]
