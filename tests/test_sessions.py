# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A training in sessions: stopped by the stop file or a SIGTERM, resumed exactly, its status in a file, and the
plan's time per step."""

import json
import os
import signal
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback

import meepsquared as m2
from meepsquared import _sessions, cli
from meepsquared.job import TrainingJob

from conftest import WORDS

TEXT8 = Dataset.from_dict({"text": ["the hare runs fast", "and the sloth sleeps", "the sloth runs", "the hare sleeps",
                                    "the hare", "the sloth runs fast", "and the hare sleeps", "the sloth"]})
# 8 examples, one per micro-batch, 2 per update: 4 updates an epoch; 6 steps take two epochs
STEPS = 6


def make_model():
    """The tiny Llama of conftest, the same weights at each call."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(config)
    model._meepsquared = {"padding_free": True, "liger": False}
    return model


def make_job(tokenizer, output, learning_rate=1e-3, **fields):
    return TrainingJob(model=make_model(), tokenizer=tokenizer, data=TEXT8, steps=STEPS, output=str(output),
                       recipe=m2.Recipe(effective_batch=2, optimizer="adamw_torch", schedule="cosine",
                                        learning_rate=learning_rate),
                       overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0), **fields)


class StopAt(TrainerCallback):
    """Asks the training to stop after step `step`: creates the stop file, or sends this process a SIGTERM."""

    def __init__(self, step, stop_file=None):
        self.step, self.stop_file = step, stop_file

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            if self.stop_file is not None:
                Path(self.stop_file).touch()
            else:
                os.kill(os.getpid(), signal.SIGTERM)


def recorded(job):
    """The job's trainer, made to record the input ids of each micro-batch it trains."""
    trainer = job.trainer()
    seen = []
    step = trainer.training_step

    def recording(model, inputs, *args, **kwargs):
        seen.append(inputs["input_ids"].clone())
        return step(model, inputs, *args, **kwargs)

    trainer.training_step = recording
    return trainer, seen


def logged(trainer, key):
    """{step: value} of what the trainer logged under `key`."""
    return {entry["step"]: entry[key] for entry in trainer.state.log_history if key in entry}


def jobs_from_yaml(monkeypatch, jobs):
    """meepsquared's command reads these jobs, in turn, whatever the file it is given."""
    monkeypatch.setattr(TrainingJob, "from_yaml", classmethod(lambda cls, path: jobs.pop(0)))


@pytest.mark.parametrize("stop_at", [2, 4, 5])
def test_a_resumed_training_trains_what_an_uninterrupted_one_does(tokenizer, tmp_path, stop_at):
    # 2: within the first epoch; 4: at its end; 5: within the second
    whole = make_job(tokenizer, tmp_path / "whole")
    trainer, seen_whole = recorded(whole)
    whole.train()

    stop = tmp_path / "STOP"
    first = make_job(tokenizer, tmp_path / "out")
    trainer_first, seen_first = recorded(first)
    trainer_first.add_callback(StopAt(stop_at, stop))
    paused = first.train(stop_file=stop)
    assert paused.paused and paused.steps == stop_at
    checkpoint = Path(paused.checkpoint)
    assert checkpoint == tmp_path / "out" / f"checkpoint-{stop_at}"
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        assert (checkpoint / name).exists(), name

    stop.unlink()
    second = make_job(tokenizer, tmp_path / "out")
    trainer_second, seen_second = recorded(second)
    if stop_at == 5:
        assert second.trainer()._meepsquared_first_epoch == 0
    resumed = second.train(resume=True, stop_file=stop)
    assert not resumed.paused and resumed.steps == STEPS
    assert second.trainer()._meepsquared_first_epoch == stop_at // 4

    # The micro-batches of the two sessions are those of the uninterrupted training, in its order
    together = seen_first + seen_second
    assert len(together) == len(seen_whole) == 2 * STEPS
    assert all(torch.equal(a, b) for a, b in zip(together, seen_whole))
    # The loss and the learning rate of each step after the resume are the uninterrupted training's
    losses, rates = logged(trainer, "loss"), logged(trainer, "learning_rate")
    resumed_losses, resumed_rates = logged(trainer_second, "loss"), logged(trainer_second, "learning_rate")
    assert sorted(resumed_losses) == list(range(1, STEPS + 1))
    for step in range(stop_at + 1, STEPS + 1):
        assert resumed_losses[step] == pytest.approx(losses[step], rel=1e-5, abs=1e-6)
        assert resumed_rates[step] == pytest.approx(rates[step], rel=1e-6)
    # The cosine schedule spans the planned steps across both sessions: it falls to its floor
    assert rates[STEPS] < rates[1]


def test_the_command_stops_with_the_stop_file_and_resumes(tokenizer, tmp_path, monkeypatch):
    output, stop, status = tmp_path / "out", tmp_path / "STOP", tmp_path / "status.json"
    jobs_from_yaml(monkeypatch, [make_job(tokenizer, output), make_job(tokenizer, output)])
    stop.touch()
    args = ["train", "job.yaml", "--status-file", str(status), "--stop-file", str(stop)]
    assert cli.main(args) == cli.PAUSED
    written = json.loads(status.read_text())
    assert written["state"] == "paused" and written["step"] == 1 and written["total_steps"] == STEPS
    assert written["checkpoint"] == str(output / "checkpoint-1") and (output / "checkpoint-1" / "optimizer.pt").exists()
    # A paused training saves no adapter beside its checkpoints
    assert not (output / "adapter_model.safetensors").exists()

    stop.unlink()
    assert cli.main(args + ["--resume"]) == 0
    written = json.loads(status.read_text())
    assert written["state"] == "completed" and written["step"] == STEPS and written["error"] is None
    assert (output / "adapter_model.safetensors").exists()


def test_the_status_file_follows_the_training(tokenizer, tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    writes = []
    write = _sessions.write_json

    def recording(path, data):
        if path == status:
            writes.append(json.loads(json.dumps(data)))
        write(path, data)

    monkeypatch.setattr(_sessions, "write_json", recording)
    jobs_from_yaml(monkeypatch, [make_job(tokenizer, tmp_path / "out")])
    assert cli.main(["train", "job.yaml", "--status-file", str(status)]) == 0

    assert set(writes[0]) == {"state", "step", "total_steps", "loss", "seconds_per_step", "eta_seconds",
                              "checkpoint", "plan", "error", "eval_loss", "eval_step", "updated"}
    assert writes[0]["state"] == "planning" and writes[0]["plan"] is None
    training = [w for w in writes if w["state"] == "training"]
    assert training[0]["step"] == 0 and training[0]["total_steps"] == STEPS
    plan = training[0]["plan"]
    assert {"decisions", "seconds_per_step", "estimated_seconds"} <= set(plan)
    assert [d for d in plan["decisions"] if d["name"] == "length"][0]["value"] == STEPS
    steps = sorted({w["step"] for w in training})
    assert steps == list(range(STEPS + 1))
    after_first = [w for w in training if w["step"] == 1]
    assert after_first[0]["seconds_per_step"] is None and after_first[0]["eta_seconds"] is None
    later = [w for w in training if w["step"] == 3][-1]
    assert later["seconds_per_step"] > 0 and later["eta_seconds"] == pytest.approx(3 * later["seconds_per_step"])
    assert isinstance(later["loss"], float)
    assert all(w["error"] is None for w in training)
    # Transformers saves a checkpoint at the last step, and none before
    assert all(w["checkpoint"] is None for w in training if w["step"] < STEPS)
    assert writes[-1]["state"] == "completed" and writes[-1]["step"] == STEPS
    assert writes[-1]["checkpoint"] == str(tmp_path / "out" / f"checkpoint-{STEPS}")
    assert json.loads(status.read_text()) == writes[-1]
    assert writes[-1]["updated"].endswith("+00:00")


def test_a_sigterm_stops_the_training_with_a_checkpoint(tokenizer, tmp_path, monkeypatch):
    output, status = tmp_path / "out", tmp_path / "status.json"
    job = make_job(tokenizer, output)
    job.trainer().add_callback(StopAt(2))
    jobs_from_yaml(monkeypatch, [job])
    handler = signal.getsignal(signal.SIGTERM)
    assert cli.main(["train", "job.yaml", "--status-file", str(status)]) == cli.PAUSED
    written = json.loads(status.read_text())
    assert written["state"] == "paused" and written["step"] == 2
    assert (output / "checkpoint-2" / "trainer_state.json").exists()
    assert signal.getsignal(signal.SIGTERM) == handler


def test_resume_without_a_checkpoint_is_refused(tokenizer, tmp_path, monkeypatch, capsys):
    output, status = tmp_path / "out", tmp_path / "status.json"
    jobs_from_yaml(monkeypatch, [make_job(tokenizer, output)])
    assert cli.main(["train", "job.yaml", "--status-file", str(status), "--resume"]) == 1
    written = json.loads(status.read_text())
    assert written["state"] == "failed"
    assert written["error"] == (f"FileNotFoundError: resume: the output folder {output} holds no checkpoint to "
                                "resume from")
    assert "FileNotFoundError" in capsys.readouterr().err


def test_a_new_training_over_checkpoints_and_a_resume_planned_otherwise_are_refused(tokenizer, tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    assert make_job(tokenizer, tmp_path / "out").train(stop_file=stop).paused
    with pytest.raises(ValueError, match="holds checkpoints of an earlier training"):
        make_job(tokenizer, tmp_path / "out").train()
    with pytest.raises(ValueError, match="learning_rate: 0.001 then, 0.002 now"):
        make_job(tokenizer, tmp_path / "out", learning_rate=2e-3).train(resume=True)


def test_the_plan_is_not_timed_on_the_cpu_unless_asked(tokenizer, tmp_path):
    plan = make_job(tokenizer, tmp_path / "a").plan()
    duration = plan["duration"]
    assert duration.value is None and duration.text == "not timed" and "not on a CUDA GPU" in duration.reason
    assert plan.seconds_per_step is None and plan.estimated_seconds is None
    data = plan.to_dict()
    assert data["seconds_per_step"] is None and data["estimated_seconds"] is None
    assert make_job(tokenizer, tmp_path / "b", timed_steps=0).plan()["duration"].reason == "timed_steps=0"


def test_timed_steps_state_the_length_and_leave_the_training_as_it_was(tokenizer, tmp_path):
    timed = make_job(tokenizer, tmp_path / "timed", timed_steps=2)
    plan = timed.plan()
    duration = plan["duration"]
    assert plan.seconds_per_step > 0 and duration.value == plan.seconds_per_step
    assert plan.estimated_seconds == pytest.approx(plan.seconds_per_step * STEPS)
    assert duration.text.startswith("about ") and duration.text.endswith(f"for {STEPS} steps")
    assert "the median of 2 optimizer steps timed after 1 to warm up, each 2 forward and backward passes" in \
        duration.reason
    assert "priced on the line through the two at the mean micro-batch of the training" in duration.reason
    data = plan.to_dict()
    assert data["seconds_per_step"] == plan.seconds_per_step and data["estimated_seconds"] == plan.estimated_seconds
    # The timed steps change neither the weights nor the optimizer nor the RNG the training starts from
    timed.train()
    untimed = make_job(tokenizer, tmp_path / "untimed")
    untimed.train()
    timed_losses, untimed_losses = logged(timed.trainer(), "loss"), logged(untimed.trainer(), "loss")
    assert timed_losses == untimed_losses
    # Nor the metrics the trainer logs, which its compute_loss adds to
    assert logged(timed.trainer(), "num_tokens") == logged(untimed.trainer(), "num_tokens")
    assert logged(timed.trainer(), "mean_token_accuracy") == logged(untimed.trainer(), "mean_token_accuracy")


def test_a_step_is_priced_on_the_line_through_the_two_timed_batches():
    # 1 s fixed and 1 ms a token: 2 s at 1,000 tokens, 5 s at 4,000
    assert _sessions.on_the_line(1000, 2.0, 4000, 5.0, 2500) == pytest.approx(3.5)
    assert _sessions.on_the_line(1000, 2.0, 4000, 5.0, 1000) == pytest.approx(2.0)
    # Two batches of the same size: the larger one's time
    assert _sessions.on_the_line(4000, 4.8, 4000, 5.0, 4000) == 5.0


def test_timed_steps_are_a_whole_number_and_for_sft():
    with pytest.raises(ValueError, match="timed_steps"):
        TrainingJob(model="org/model", data="org/data", timed_steps=-1)
    with pytest.raises(ValueError, match="GRPO"):
        TrainingJob(model="org/model", data="org/data", task="grpo", rewards=["json:dumps"], timed_steps=3)
    job = TrainingJob(model="org/model", data="org/data", timed_steps=0)
    assert job.to_dict()["timed_steps"] == 0 and TrainingJob.from_dict(job.to_dict()) == job


def test_status_is_written_whole(tmp_path):
    status = _sessions.Status(tmp_path / "status.json")
    status.update(state="training", step=3, total_steps=10)
    assert json.loads((tmp_path / "status.json").read_text())["step"] == 3
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]
    with pytest.raises(ValueError, match="state must be one of"):
        status.update(state="sleeping")


GIB = 2**30


def test_the_plan_measures_the_memory_over_the_timed_steps(tokenizer, tmp_path):
    plan = make_job(tokenizer, tmp_path / "out", timed_steps=2).plan()
    memory = plan["memory"]
    assert plan.peak_gpu_bytes is None  # the CPU
    assert 0 < plan.peak_host_bytes <= _sessions.peak_resident_bytes()
    request = plan.memory_request_bytes
    assert request % GIB == 0 and plan.peak_host_bytes * 1.1 <= request < plan.peak_host_bytes * 1.1 + GIB
    assert memory.value == {"peak_gpu_bytes": None, "peak_host_bytes": plan.peak_host_bytes,
                            "memory_request_bytes": request}
    assert memory.text.endswith(f"on the host; a job should ask for {request // GIB} GiB (the host's peak, 10% over "
                                "it, rounded up to a whole GiB)")
    assert "peak resident memory since it started, loading included" in memory.reason
    data = plan.to_dict()
    assert (data["peak_gpu_bytes"], data["peak_host_bytes"], data["memory_request_bytes"]) == (
        None, plan.peak_host_bytes, request)


def test_the_memory_is_not_measured_without_timed_steps(tokenizer, tmp_path):
    for fields, why in (({}, "not on a CUDA GPU"), ({"timed_steps": 0}, "timed_steps=0")):
        plan = make_job(tokenizer, tmp_path / str(len(fields)), **fields).plan()
        assert (plan.peak_gpu_bytes, plan.peak_host_bytes, plan.memory_request_bytes) == (None, None, None)
        memory = plan["memory"]
        assert memory.value is None and memory.text == "not measured" and why in memory.reason
        data = plan.to_dict()
        assert data["peak_gpu_bytes"] is None and data["memory_request_bytes"] is None


HOST, GPU = 10 * GIB + 5, 20 * GIB + 7


@pytest.mark.parametrize("holds, peak, how", [
    (None, HOST, "the host's peak; the GPU has memory of its own"),
    (False, HOST + GPU, "the host's peak plus the GPU's"),
    (True, HOST, "the host's peak, which counts the GPU's allocations"),
])
def test_the_memory_request_covers_the_peak_with_a_margin(holds, peak, how):
    timing = _sessions.Timing(1.0, None, GPU, HOST, holds)
    request, why = _sessions.memory_request(timing)
    assert request % GIB == 0 and peak * 1.1 <= request < peak * 1.1 + GIB and why.startswith(how)
    assert why.endswith("10% over it, rounded up to a whole GiB")
