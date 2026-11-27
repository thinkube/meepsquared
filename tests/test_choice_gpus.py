# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A choice job on several GPUs (TrainingJob.gpus): a launch of that many processes, each with its own copy of the
model, the update's questions shared out among them and their gradients summed. Here the processes run on the CPU,
joined by Gloo, as the GPUs' are by NCCL."""

import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

import meepsquared as m2
from meepsquared import _choice_job

from conftest import CHATML
from test_choice import make_job, rows, write

# Four questions an update; the first three train the head alone. Shared out between two processes, the first one
# reads questions 0 and 2 of the first update, both head-only, and has no gradient of the adapter at its step; the
# second one reads 1 and 3, and question 3 trains the adapter.
RECIPE = {"effective_batch": 4, "head_only": 3}


@pytest.fixture
def tokenizer(chat_tokenizer):
    return chat_tokenizer(CHATML)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def trained(result) -> dict:
    return {name: p.detach().clone() for name, p in result._model.trainable().items()}


def logged_losses(trainer) -> list:
    return [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]


def train_in(rank: int, world: int, port: int, tokenizer, folder: str) -> None:
    """One process of a launch of `world`: trains the job and, as process 0, writes what it trained."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))
    folder = Path(folder)
    # The plan times the steps and the sum of the gradients among the processes, which leaves the weights as they were
    job = make_job(tokenizer, folder, output=f"out-{world}", data=str(folder / "train.jsonl"), gpus=world,
                   report_to=[], recipe=RECIPE, timed_steps=1)
    trainer = job.trainer()
    result = job.train()
    if rank == 0:
        torch.save(trained(result), folder / "trained.pt")
        (folder / "losses.json").write_text(json.dumps(logged_losses(trainer)))
        (folder / "plan.json").write_text(json.dumps(job.plan().to_dict(), default=str))
    torch.distributed.destroy_process_group()


def test_two_processes_make_the_updates_one_makes(tokenizer, tmp_path):
    write(tmp_path / "train.jsonl", rows())
    alone = make_job(tokenizer, tmp_path, output="out-1", report_to=[], recipe=RECIPE)
    trainer = alone.trainer()
    result = alone.train()
    assert result.steps == 2
    one = trained(result)
    one_losses = logged_losses(trainer)

    mp.spawn(train_in, args=(2, free_port(), tokenizer, str(tmp_path)), nprocs=2, join=True)
    assert json.loads((tmp_path / "losses.json").read_text()) == pytest.approx(one_losses, rel=1e-4)
    two = torch.load(tmp_path / "trained.pt")
    assert set(two) == set(one)
    # The gradients are summed in another order, and Adam divides each by its own size: a weight whose gradient is
    # near zero moves by a rounding's share of a step. A step moves a weight by up to the learning rate, 1e-2; a
    # question's gradient left out would move the weights by that much.
    for name in one:
        largest = float((two[name] - one[name]).abs().max())
        assert largest < 1e-3, f"{name}: differs by up to {largest:.3g}"
    plan = {d["name"]: d for d in json.loads((tmp_path / "plan.json").read_text())["decisions"]}
    assert plan["gpus"]["value"] == 2 and "2 to each" in plan["gpus"]["text"]
    assert "2 on each of the 2 GPUs at once" in plan["duration"]["reason"]


def test_a_job_on_two_gpus_is_launched_as_two_processes(tokenizer, tmp_path):
    with pytest.raises(ValueError, match="torchrun --nproc_per_node 2"):
        make_job(tokenizer, tmp_path, gpus=2, recipe=RECIPE).plan()
    with pytest.raises(ValueError, match="gpus is a whole number"):
        make_job(tokenizer, tmp_path, gpus=0)
    with pytest.raises(ValueError, match="gpus is a choice setting"):
        m2.TrainingJob(model="m", data="d", gpus=2)


def test_the_update_is_shared_out_evenly(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setattr(_choice_job, "join_launch", lambda gpus: None)
    with pytest.raises(ValueError, match="effective_batch 3 is not shared out evenly among 2 GPUs"):
        make_job(tokenizer, tmp_path, gpus=2, recipe={"effective_batch": 3}).plan()


def test_measurements_on_one_gpu_are_not_reused_for_two(tokenizer, tmp_path):
    earlier = make_job(tokenizer, tmp_path, timed_steps=1, report_to=[], recipe=RECIPE).plan()
    assert earlier["gpus"].text == "one" and earlier.measurements["steps"]["sync_seconds"] == 0.0
    with pytest.raises(m2.MeasureAgain, match="the number of GPUs changed"):
        make_job(tokenizer, tmp_path, gpus=2, report_to=[], recipe=RECIPE).plan(reuse=earlier)
