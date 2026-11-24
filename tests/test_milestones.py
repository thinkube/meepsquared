# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Milestones: the adapter kept apart at a share of the steps, the model's answers scored and logged, a resumed
training not making a finished milestone again, and the plan's line."""

import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback

import meepsquared as m2
from meepsquared import _milestones
from meepsquared.job import TrainingJob

from conftest import CHATML

TEXT8 = Dataset.from_dict({"text": ["the hare runs fast", "and the sloth sleeps", "the sloth runs", "the hare sleeps",
                                    "the hare", "the sloth runs fast", "and the hare sleeps", "the sloth"]})
# 8 examples, one per micro-batch, 2 per update: 4 updates an epoch; 6 steps; milestones every half: steps 3 and 6
STEPS = 6
ROWS = [
    {"messages": [{"role": "user", "content": "the hare runs"}], "reference": "the hare runs fast", "group": "hare"},
    {"messages": [{"role": "user", "content": "the hare"}], "reference": "the hare sleeps", "group": "hare",
     "source": "hare"},
    {"messages": [{"role": "user", "content": "the sloth"}], "reference": "the sloth sleeps", "group": "sloth"},
]


def scorer(hypotheses, references):
    """The rows a group has, and how many answers are not empty."""
    return {"rows": len(references), "answered": sum(1 for h in hypotheses if h)}


def make_model(tokenizer):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(config)
    model._meepsquared = {"padding_free": True, "liger": False}
    return model


def make_job(tokenizer, output, **fields):
    return TrainingJob(model=make_model(tokenizer), tokenizer=tokenizer, data=TEXT8, steps=STEPS, output=str(output),
                       recipe=m2.Recipe(effective_batch=2, optimizer="adamw_torch", learning_rate=1e-3),
                       overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0),
                       milestones=0.5, milestone_eval=ROWS, milestone_scorer=scorer, **fields)


class StopAt(TrainerCallback):
    def __init__(self, step, stop_file):
        self.step, self.stop_file = step, stop_file

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            Path(self.stop_file).touch()


def test_the_steps_are_every_share_and_the_last():
    assert _milestones.steps(100, 0.02) == list(range(2, 101, 2))
    assert _milestones.steps(7, 0.5) == [4, 7]
    assert _milestones.steps(3, 0.1) == [1, 2, 3]
    assert _milestones.steps(10, 1) == [10]


def test_a_generation_is_cut_at_its_end_its_cap_and_its_first_loop():
    assert _milestones.cut([5, 6, 2, 7], 10, {2}) == [5, 6]
    assert _milestones.cut([5, 6, 7, 8], 3, {2}) == [5, 6, 7]
    looped = [9] + [5, 6, 7, 8] * 6
    assert _milestones.cut(looped, 100, {2}) == [9, 5, 6, 7, 8]
    assert _milestones.loop_end([5, 6, 7, 8, 9]) is None


def test_the_prompt_is_what_the_training_reads_before_an_answer(chat_tokenizer):
    tokenizer = chat_tokenizer(CHATML)
    assert _milestones.prompt(tokenizer, ROWS[0]) == "<|im_start|>user\nthe hare runs<|im_end|>\n<|im_start|>assistant\n"
    # The cap counts the source when given, else the last user message
    assert _milestones.budget(tokenizer, ROWS[0]) == 2 * 3 + _milestones.NEW_MARGIN
    assert _milestones.budget(tokenizer, ROWS[1]) == 2 * 1 + _milestones.NEW_MARGIN


@pytest.mark.parametrize("row,error", [
    ({"messages": [], "reference": "the hare"}, "'messages' must be a list"),
    ({"messages": [{"role": "user", "content": "the"}, {"role": "assistant", "content": "hare"}],
      "reference": "the hare"}, "end with an answer"),
    ({"messages": [{"role": "user", "content": "the"}]}, "'reference' must be"),
    ({"messages": [{"role": "user", "content": "the"}], "reference": "hare", "group": 3}, "'group' must be"),
])
def test_rows_that_cannot_be_answered_are_refused(row, error):
    with pytest.raises(ValueError, match=error):
        _milestones.rows([row])


def test_the_fields_are_checked(tokenizer):
    with pytest.raises(ValueError, match="above 0 and at most 1"):
        TrainingJob(model="m", data="d.jsonl", milestones=1.5)
    with pytest.raises(ValueError, match="no milestones"):
        TrainingJob(model="m", data="d.jsonl", milestone_eval="e.jsonl", milestone_scorer="s.py:score")
    with pytest.raises(ValueError, match="go together"):
        TrainingJob(model="m", data="d.jsonl", milestones=0.1, milestone_eval="e.jsonl")
    job = TrainingJob(model="m", data="d.jsonl", milestones=0.1, milestone_eval="e.jsonl", milestone_scorer="s.py:score")
    assert TrainingJob.from_dict(job.to_dict()).to_dict() == job.to_dict()


def test_each_milestone_keeps_the_adapter_and_scores_the_answers(chat_tokenizer, tmp_path):
    tokenizer = chat_tokenizer(CHATML)
    status = tmp_path / "status.json"
    job = make_job(tokenizer, tmp_path / "out")
    plan = job.plan()
    assert plan["milestones"].value["steps"] == [3, 6]
    result = job.train(status_file=status)
    assert [m["step"] for m in result.milestones] == [3, 6]
    for step in (3, 6):
        folder = tmp_path / "out" / "milestones" / f"step-{step}"
        for name in ("adapter_config.json", "adapter_model.safetensors", "generations.jsonl", "milestone.json"):
            assert (folder / name).exists(), name
        record = json.loads((folder / "milestone.json").read_text())
        assert record["scores"] == {"hare": {"rows": 2.0, "answered": record["scores"]["hare"]["answered"]},
                                    "sloth": {"rows": 1.0, "answered": record["scores"]["sloth"]["answered"]}}
        assert len((folder / "generations.jsonl").read_text().splitlines()) == len(ROWS)
    logged = [entry for entry in job.trainer().state.log_history if "milestone/hare/rows" in entry]
    assert [entry["step"] for entry in logged] == [3, 6]
    written = json.loads(status.read_text())
    assert [m["step"] for m in written["milestones"]] == [3, 6]
    assert written["next_milestone"] is None and written["milestone_seconds"] is not None


def test_a_resumed_training_does_not_make_a_finished_milestone_again(chat_tokenizer, tmp_path):
    tokenizer = chat_tokenizer(CHATML)
    stop = tmp_path / "STOP"
    first = make_job(tokenizer, tmp_path / "out")
    first.trainer().add_callback(StopAt(4, stop))
    assert first.train(stop_file=stop).paused
    made = (tmp_path / "out" / "milestones" / "step-3" / "milestone.json").read_text()
    assert not (tmp_path / "out" / "milestones" / "step-6").exists()

    stop.unlink()
    second = make_job(tokenizer, tmp_path / "out")
    resumed = second.train(resume=True, stop_file=stop)
    assert [m["step"] for m in resumed.milestones] == [3, 6]
    assert (tmp_path / "out" / "milestones" / "step-3" / "milestone.json").read_text() == made


def test_a_stop_at_a_milestones_step_waits_for_the_milestone(chat_tokenizer, tmp_path):
    tokenizer = chat_tokenizer(CHATML)
    stop = tmp_path / "STOP"
    job = make_job(tokenizer, tmp_path / "out")
    job.trainer().add_callback(StopAt(3, stop))
    result = job.train(stop_file=stop)
    assert result.paused and result.steps == 3
    assert (tmp_path / "out" / "milestones" / "step-3" / "milestone.json").exists()


def test_milestones_at_growing_distances_come_often_at_the_start_and_seldom_later():
    from meepsquared import _milestones

    # The distances grow: 500, 750, 1125, 1687.5, 2531.25, 3796.9, each milestone their sum rounded up
    assert _milestones.steps(11_931, 500 / 11_931, 1.5) == [500, 1250, 2375, 4063, 6594, 10_391, 11_931]
    # Each milestone at least one step after the one before: none twice at a small first distance
    small = _milestones.steps(20, 0.01, 1.2)
    assert small[:3] == [1, 2, 3] and small == sorted(set(small)) and small[-1] == 20
    # Without growth, equal distances as before
    assert _milestones.steps(100, 0.25) == [25, 50, 75, 100]
    assert "1.5 times the one before" in _milestones.spacing_text(10_000, 0.01, 1.5)


def test_milestone_growth_is_above_one_and_needs_milestones():
    import meepsquared as m2

    for bad in (1, 0.5, True, "2"):
        with pytest.raises(ValueError, match="milestone_growth must be above 1"):
            m2.TrainingJob(model="x", data="d.jsonl", milestones=0.01, milestone_growth=bad)
    with pytest.raises(ValueError, match="give milestones"):
        m2.TrainingJob(model="x", data="d.jsonl", milestone_growth=1.5)
    job = m2.TrainingJob(model="x", data="d.jsonl", milestones=0.01, milestone_growth=1.5)
    assert m2.TrainingJob.from_dict(job.to_dict()).milestone_growth == 1.5


def sft_job(**fields):
    import meepsquared as m2

    return m2.TrainingJob(model="m", data="d.jsonl", **{"milestones": 0.5, **fields})


def sft_plan(job):
    from meepsquared.job import SFT_MEASUREMENTS, Decision, Plan

    placement = {"planned_steps": 1000, "start_step": 0, "estimated_without_milestones": 2000.0,
                 "milestone_seconds": 60.0, "milestones_what": "the adapter saved apart, and 40 rows answered",
                 "milestones_why": "", "milestone_value_key": "generation_seconds", "evaluations": 0,
                 "eval_seconds": 0.0, "evaluations_counted": True}
    decisions = [Decision("length", "length", 1000, "1 epoch, 1000 steps"),
                 Decision("duration", "time", 2.0, "old", "timed on 3 steps"),
                 Decision("memory", "memory", {"peak_gpu_bytes": 1}, "peak 21 GiB", "measured"),
                 Decision("milestones", "milestones", {"steps": [500, 1000]}, "2 at steps 500, 1,000")]
    measurements = {"task": "sft", "format": SFT_MEASUREMENTS, "job": job._placement_free(),
                    "placement": placement, "device": "RTX 3090"}
    return Plan(model="m", task="sft", method="lora", device="RTX 3090", free_bytes=None, decisions=decisions,
                seconds_per_step=2.0, estimated_seconds=2120.0, peak_gpu_bytes=1, memory_request_bytes=5,
                measurements=measurements)


def test_an_sft_plan_places_its_milestones_again_without_measuring():
    from meepsquared.job import Plan

    earlier = Plan.from_dict(sft_plan(sft_job()).to_dict())
    revised = sft_job(milestones=0.1, milestone_growth=1.5).plan(reuse=earlier, reused_from="plan job j1")
    assert revised["milestones"].value["steps"] == [100, 250, 475, 813, 1000]
    assert revised.estimated_seconds == 2000.0 + 5 * 60.0
    assert revised["duration"].reason == "timed on 3 steps"
    assert (revised["memory"], revised.memory_request_bytes) == (earlier["memory"], 5)
    assert revised["measurements"].text.startswith("reused from plan job j1, measured on RTX 3090")
    # Reused again, the measurements still name the plan that measured them
    again = sft_job(milestones=None).plan(reuse=revised, reused_from="plan job j2")
    assert again["milestones"].text == "none" and again["measurements"].value == {"from": "plan job j1"}


def test_an_sft_plan_of_another_change_or_without_measured_milestones_measures_again():
    import meepsquared as m2

    earlier = sft_plan(sft_job())
    with pytest.raises(m2.MeasureAgain, match="recipe changed"):
        sft_job(recipe=m2.Recipe(learning_rate=1e-5)).plan(reuse=earlier)
    earlier.measurements["placement"]["milestones_what"] = ""
    with pytest.raises(m2.MeasureAgain, match="placed no milestones"):
        sft_job(milestones=0.1).plan(reuse=earlier)
    earlier.measurements = None
    with pytest.raises(m2.MeasureAgain, match="keeps no measurements"):
        sft_job().plan(reuse=earlier)


def test_a_milestone_is_timed_on_rows_spread_over_the_set():
    from meepsquared import _choice_job

    rows = list(range(1860))
    sample = _choice_job.milestone_sample(rows)
    assert len(sample) == _choice_job.MILESTONE_SAMPLE and sample[0] == 0 and sample[-1] > 1800
    assert _choice_job.milestone_sample(rows[:10]) == rows[:10]
