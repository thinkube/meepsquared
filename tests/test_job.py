# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest
import torch
from datasets import Dataset

import meepsquared as m2
from meepsquared import _batches, _checkpointing, _compile, _moe, cli, job as job_module
from meepsquared.chat_templates import ResponsesOnly
from meepsquared.job import Plan, TrainingJob

GIB = 2**30
TEXT = Dataset.from_dict({"text": ["the hare runs fast", "and the sloth sleeps", "the sloth runs",
                                   "the hare sleeps"]})


TEXT8 = Dataset.from_dict({"text": list(TEXT["text"]) * 2})
TIMES = {"none": 0.030, "selective": 0.033, "whole": 0.040}


def examples(shape) -> int:
    """The examples in a measured batch: the longest ones laid end to end, or the rows."""
    return len(shape.lengths) if shape.lengths is not None else shape.rows


def fake_measure(model, shape):
    """A GPU with 16 GiB, 10 free: a micro-batch of b examples holds b GiB checkpointed, and one layer b / 2 GiB."""
    b = examples(shape)
    return b * GIB, b * GIB // 2, 10 * GIB, 16 * GIB


def fake_figures(model, shape, every, per_layer, compiled):
    """A selective layer holds half of one keeping everything; the times are TIMES."""
    return _checkpointing.Figures(shape, every, per_layer, per_layer // 2, TIMES, 2)


def fake_measurements(monkeypatch, measure=fake_measure, figures=fake_figures):
    monkeypatch.setattr(job_module, "_measure", measure)
    monkeypatch.setattr(job_module, "_measure_figures", figures)


def tiny_job(tiny_model, tokenizer, tmp_path, data=TEXT, **fields):
    return TrainingJob(model=tiny_model, tokenizer=tokenizer, data=data, output=str(tmp_path / "out"), **fields)


def test_the_plan_prints_no_warning_about_a_batch_of_one_row(tiny_model, tokenizer, tmp_path, caplog):
    """TRL warns about padding-free training with one row per batch; the plan's scratch trainer must not trigger it."""
    import logging

    caplog.set_level(logging.WARNING)
    # A micro-batch of 2 rows for the training itself: with 1 row, TRL's warning is about the training and stands
    overrides = m2.Overrides(micro_batch=2, checkpointed_layers=0)
    tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(**CPU_RECIPE), overrides=overrides).plan()
    assert "annihilate" not in caplog.text


def test_the_job_is_data_and_back(tmp_path):
    job = TrainingJob(model="org/model", data="org/data", steps=10,
                      recipe=m2.Recipe(rank=32, learning_rate=2e-4, schedule="cosine", tokens_per_update="auto"),
                      overrides=m2.Overrides(checkpointed_layers=8, compile_layers=False))
    data = job.to_dict()
    assert data == {"model": "org/model", "data": "org/data", "task": "sft", "method": "lora", "steps": 10,
                    "split": "train", "recipe": {"rank": 32, "learning_rate": 2e-4, "schedule": "cosine",
                                                 "tokens_per_update": "auto"},
                    "overrides": {"compile_layers": False, "checkpointed_layers": 8}}
    assert TrainingJob.from_dict(data) == job
    path = tmp_path / "job.yaml"
    path.write_text(job.to_yaml())
    assert TrainingJob.from_yaml(path) == job


def test_yaml_reads_a_learning_rate_with_an_exponent(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text("model: org/model\ndata: org/data\nrecipe:\n  learning_rate: 1e-4\n")
    assert TrainingJob.from_yaml(path).recipe.learning_rate == 1e-4


@pytest.mark.parametrize("fields, message", [
    ({"model": "m", "data": "d", "stpes": 3}, "unknown fields"),
    ({"model": "m", "data": "d", "steps": 3, "epochs": 1}, "not both"),
    ({"model": "m", "data": "d", "task": "grpo"}, "rewards"),
    ({"model": "m", "data": "d", "rewards": ["json:dumps"]}, "for GRPO"),
    ({"model": "m", "data": "d", "method": "full"}, "method"),
    ({"model": "m", "data": "d", "recipe": {"schedule": "linear"}}, "schedule"),
    ({"model": "m", "data": "d", "overrides": {"micro_batch": 0}}, "micro_batch"),
    ({"model": "m", "data": "d", "overrides": {"selective_layers": -1}}, "selective_layers"),
    ({"model": "m", "data": "d", "recipe": {"max_completion_length": 64}}, "GRPO setting"),
    ({"model": "m", "data": "d", "recipe": {"temperature": 0.7}}, "GRPO setting"),
    ({"model": "m", "data": "d", "overrides": {"vllm": True}}, "GRPO setting"),
    ({"model": "m", "data": "d", "task": "grpo", "rewards": ["json:dumps"], "recipe": {"answers_only": True}},
     "SFT setting"),
    ({"model": "m", "data": "d", "recipe": {"experts": True, "dropout": 0.1}}, "dropout"),
    ({"model": "m", "data": "d", "recipe": {"top_p": 1.5}}, "top_p"),
    ({"model": "m", "data": "d", "recipe": {"experts": "yes"}}, "experts"),
    ({"model": "m", "data": "d", "report_to": 3}, "report_to"),
    ({"model": "m", "data": "d", "recipe": {"tokens_per_update": 8192, "effective_batch": 8}},
     "tokens_per_update=8192 and effective_batch=8 both state the batch"),
    ({"model": "m", "data": "d", "recipe": {"tokens_per_update": 0}}, "tokens_per_update"),
    ({"model": "m", "data": "d", "recipe": {"tokens_per_update": "lots"}}, "'auto' or a positive number"),
    ({"model": "m", "data": "d", "task": "grpo", "rewards": ["json:dumps"], "recipe": {"tokens_per_update": 8192}},
     "SFT setting"),
])
def test_a_wrong_job_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        TrainingJob.from_dict(fields)


def test_rewards_by_name(tmp_path):
    import json as json_module

    assert job_module.load_reward("json:dumps") is json_module.dumps
    assert job_module.reward_name(json_module.dumps) == "json:dumps"
    file = tmp_path / "rewards.py"
    file.write_text("def length(completions, **kwargs):\n    return [len(c) for c in completions]\n")
    reward = job_module.load_reward(f"{file}:length")
    assert reward(["ab", "c"]) == [2, 1]
    with pytest.raises(ValueError, match="cannot be written"):
        job_module.reward_name(lambda completions, **kwargs: [0.0])


@pytest.mark.parametrize("parameters, rate, batch", [(8e9, 1e-4, 8), (13.0e9, 1e-4, 8), (20e9, 1e-4, 16),
                                                     (32e9, 5e-5, 16), (70e9, 5e-5, 16)])
def test_defaults_by_model_size(parameters, rate, batch):
    defaults = job_module.size_defaults(int(parameters))
    assert defaults["learning_rate"][0] == rate
    assert defaults["effective_batch"][0] == batch
    assert "Dettmers" in defaults["effective_batch"][1]


def test_warmup_and_steps():
    assert job_module.warmup_steps(10_000) == 100
    assert job_module.warmup_steps(1_000) == 50
    assert job_module.warmup_steps(2) == 0
    assert job_module.planned_steps(7, None, 25, 2) == 7
    # 100 rows in micro-batches of 4: 25 batches, 2 per update: 13 updates an epoch
    assert job_module.batches_per_epoch(100, 4, 1, "sft", 8) == 25
    assert job_module.planned_steps(None, None, 25, 2) == 13
    assert job_module.planned_steps(None, 2, 25, 2) == 26
    # GRPO: 10 prompts of 8 completions, 8 per update
    assert job_module.batches_per_epoch(10, 8, 1, "grpo", 8) == 10
    assert job_module.planned_steps(None, None, 10, 1) == 10


def test_the_learning_rate_scales_with_the_square_root_of_the_batch():
    assert job_module.REFERENCE_TOKENS == 8 * 1024
    assert job_module.scaled_learning_rate(1e-4, 8 * 1024) == 1e-4
    assert job_module.scaled_learning_rate(1e-4, 32 * 1024) == 2e-4
    assert job_module.scaled_learning_rate(1e-4, 2 * 1024) == 5e-5
    assert job_module.scaled_learning_rate(5e-5, 16 * 1024) == 7.07e-5


def test_groups_fill_the_budget_without_splitting_a_conversation():
    assert _batches.groups([3, 3, 3, 3], range(4), 7, padded=False) == [[0, 1], [2, 3]]
    # A budget below one conversation: one conversation per group
    assert _batches.groups([3, 3, 3, 3], range(4), 2, padded=False) == [[0], [1], [2], [3]]
    # In the order given, consecutive ones
    assert _batches.groups([5, 1, 1, 1], [3, 2, 1, 0], 7, padded=False) == [[3, 2, 1], [0]]
    # Padded: every conversation costs the longest, rounded up to the multiple
    assert _batches.groups([4, 1, 1, 1], range(4), 6, padded=True) == [[0], [1, 2, 3]]
    assert _batches.groups([3, 1], range(2), 8, padded=True, multiple=4) == [[0, 1]]
    assert _batches.groups([3, 1], range(2), 7, padded=True, multiple=4) == [[0], [1]]


# One process, packed batches, the Trainer's default seed
ONE = _batches.Loading(seed=42, drop_last=False, padded=False, multiple=1, processes=1, split_batches=False,
                       even_batches=True)


def test_the_order_is_seeded_and_new_each_epoch():
    first = _batches.shuffled(16, 42, 0)
    assert first == _batches.shuffled(16, 42, 0) and sorted(first) == list(range(16))
    assert first != _batches.shuffled(16, 42, 1) and first != _batches.shuffled(16, 43, 0)
    laid = _batches.lay_out([4] * 16, ONE, 1, None, 2, micro=4)
    assert laid == _batches.lay_out([4] * 16, ONE, 1, None, 2, micro=4)
    assert [index for batch in laid.epochs[0] for index in batch] == first
    assert [index for batch in laid.epochs[1] for index in batch] == _batches.shuffled(16, 42, 1)


def test_the_layout_counts_steps_and_epochs_as_the_trainer_does():
    # 6 examples in micro-batches of 2: 3 an epoch; 2 per update: 2 updates an epoch, the last of one
    laid = _batches.lay_out([3] * 6, ONE, 2, None, None, micro=2)
    assert (laid.per_epoch, laid.steps, len(laid.epochs), len(laid.trained)) == (3, 2, 1, 3)
    # 3 steps: a second epoch, cut after its first update
    laid = _batches.lay_out([3] * 6, ONE, 2, 3, None, micro=2)
    assert (laid.steps, len(laid.epochs)) == (3, 2)
    assert list(laid.trained) == list(laid.epochs[0]) + list(laid.epochs[1][:2])
    # 1.5 epochs: ceil(1.5 x 2) = 3 steps over 2 epochs
    laid = _batches.lay_out([3] * 6, ONE, 2, None, 1.5, micro=2)
    assert (laid.steps, len(laid.epochs), len(laid.trained)) == (3, 2, 5)
    # dataloader_drop_last drops the shorter last micro-batch
    laid = _batches.lay_out([3] * 5, dataclasses.replace(ONE, drop_last=True), 1, None, 1, micro=2)
    assert [len(batch) for batch in laid.epochs[0]] == [2, 2]


def test_each_process_trains_its_share_as_accelerate_gives_it():
    from accelerate.data_loader import BatchSamplerShard

    two = dataclasses.replace(ONE, processes=2)
    # 10 examples in 5 micro-batches of 2: Accelerate gives each of 2 processes 3, the last one completed from the
    # start of the epoch
    laid = _batches.lay_out([3] * 10, two, 2, None, 1, micro=2)
    shards = [list(BatchSamplerShard(_batches._Epoch(laid.epochs[0], 2, False), num_processes=2, process_index=p))
              for p in range(2)]
    assert [len(shard) for shard in shards] == [3, 3] and shards[0] == laid.epochs[0][0::2]
    assert laid.per_epoch == 3 and laid.steps == 2
    assert list(laid.trained) == shards[0] + shards[1]
    # A token budget: groups of varying size, which Accelerate deals out in turn
    budget = _batches.lay_out([3, 3, 5, 3, 3, 3], two, 1, None, 1, budget=6)
    assert budget.batch_size is None
    groups = _batches.groups([3, 3, 5, 3, 3, 3], _batches.shuffled(6, 42, 0), 6, padded=False)
    assert list(budget.epochs[0]) == groups
    assert budget.trained[:budget.per_epoch] == tuple(groups[0::2][:budget.per_epoch])


def test_the_batch_sampler_gives_the_epochs_laid_out():
    laid = _batches.lay_out([3] * 6, ONE, 1, None, 2, micro=2)
    sampler = _batches.LaidOutBatchSampler(laid)
    assert len(sampler) == 3 and sampler.batch_size == 2
    assert list(sampler) == laid.epochs[0] and list(sampler) == laid.epochs[1]
    sampler.set_epoch(0)
    assert list(sampler) == laid.epochs[0]
    sampler.set_epoch(2)
    with pytest.raises(RuntimeError, match="epoch 3"):
        list(sampler)
    # A token budget: batch_size None, so that Accelerate takes the batches as of varying size
    assert _batches.LaidOutBatchSampler(_batches.lay_out([3] * 6, ONE, 1, None, 1, budget=6)).batch_size is None


def test_micro_batch_halves_within_the_effective_batch():
    assert job_module.smaller_micro_batch(8, 8) == 4
    assert job_module.smaller_micro_batch(8, 2) == 1
    assert job_module.smaller_micro_batch(12, 12) == 6
    assert job_module.smaller_micro_batch(12, 6) == 3
    assert job_module.smaller_micro_batch(12, 3) == 1


def test_fits_counts_the_inputs_and_one_layer():
    # 10 GiB free of 16: 15% kept back leaves 7.6 GiB
    assert _checkpointing.fits(10 * GIB, int(_checkpointing.MARGIN * 16 * GIB), 4 * GIB, 2 * GIB)
    assert not _checkpointing.fits(10 * GIB, int(_checkpointing.MARGIN * 16 * GIB), 6 * GIB, 2 * GIB)


def test_data_formats():
    assert job_module.data_format(["prompt", "messages", "category"], "sft") == ("conversations (messages)",
                                                                                ["messages"])
    assert job_module.data_format(["prompt", "completion"], "sft") == ("prompt and completion pairs",
                                                                       ["prompt", "completion"])
    assert job_module.data_format(["text", "id"], "sft") == ("plain text", ["text"])
    assert job_module.data_format(["prompt", "answer"], "grpo") == ("prompts", ["prompt", "answer"])
    with pytest.raises(ValueError, match="'messages'"):
        job_module.data_format(["question"], "sft")


def test_plan_decides_without_training(tiny_model, tokenizer, tmp_path, monkeypatch):
    fake_measurements(monkeypatch)
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=2, data=TEXT8)
    plan = job.plan()
    assert plan["effective_batch"].value == 8 and plan["effective_batch"].by == "auto"
    # The default path batches by examples; a token budget is opt-in
    assert "tokens_per_update='auto') batches by tokens instead" in plan["effective_batch"].reason
    with pytest.raises(KeyError):
        plan["tokens_per_update"]
    # 8 examples hold 12 GiB, more than 7.6; 4 hold 6
    assert plan["micro_batch"].value == 4 and plan["accumulation"].value == 2
    # One layer of 4 examples holds 2 GiB: both layers fit, none is checkpointed, none selective
    assert plan["checkpointed_layers"].value == 0 and plan["selective_layers"].value == 0
    # The batch measured is the largest micro-batch the training runs: 2 steps of 2 micro-batches of 4
    # examples, laid end to end, over 2 epochs of the 8 rows
    laid = job.trainer()._meepsquared_layout
    lengths = _batches.lengths(job.trainer().train_dataset, 1024)
    largest = _checkpointing.largest_batch(laid.trained, lengths, padding_free=True)
    longest = _checkpointing.longest_examples(lengths, 4, 1024, padding_free=True)
    assert len(laid.trained) == 4 and largest.text().endswith("(4 examples, end to end)")
    assert (f"one layer for the largest of the 4 micro-batches this job trains, {largest.text()}; the data's 4 "
            f"longest would be {longest.tokens:,}") in plan["checkpointed_layers"].reason
    order = plan["order"]
    assert (order.value, order.by) == (42, "auto")
    assert order.text == "shuffled with seed 42, the micro-batches of 2 epochs laid out ahead"
    assert "1.00 selective" in plan["checkpointed_layers"].reason and "30 / 33 / 40 ms" in plan["checkpointed_layers"].reason
    assert plan["rank"].value == 16 and plan["alpha"].value == 16
    # The stated rate, not scaled
    assert plan["learning_rate"].value == 1e-4 and "sqrt" not in plan["learning_rate"].reason
    assert plan["optimizer"].value == "adamw_8bit"
    assert plan["schedule"].value == "constant" and plan["warmup_steps"].value == 0
    assert all(d.by == "auto" for d in plan.decisions)
    text = str(plan)
    assert text.startswith("meepsquared plan for")
    assert "checkpoint" in text and "0 of 2 layers" in text
    assert "  selective    0 of 2 layers (recompute their cheap operations only" in text
    data = plan.to_dict()
    assert json.loads(json.dumps(data))["decisions"][0]["name"] == "loading"
    trainer = job.trainer()
    assert trainer.state.global_step == 0
    assert trainer.args.per_device_train_batch_size == 4 and trainer.args.gradient_accumulation_steps == 2
    assert trainer.args.learning_rate == 1e-4 and trainer.args.lr_scheduler_type.value == "constant_with_warmup"
    assert job.plan() is plan


def test_a_measured_token_budget_is_opt_in(tiny_model, tokenizer, tmp_path, monkeypatch):
    fake_measurements(monkeypatch)
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=2, recipe=m2.Recipe(tokens_per_update="auto"))
    plan = job.plan()
    # The ceiling of 32 x 1024 tokens: the four short rows of TEXT make one group of 4 examples, which holds
    # 4 + 2 GiB, within 7.6
    budget = plan["tokens_per_update"]
    assert (budget.value, budget.by) == (32768, "auto")
    assert "Recipe(tokens_per_update='auto')" in budget.reason and "up to 32,768 tokens" in budget.reason
    assert "time-to-quality study" in budget.reason
    assert "(4 in the first step)" in budget.text
    with pytest.raises(KeyError):
        plan["effective_batch"]
    assert plan["micro_batch"].value == 32 and plan["accumulation"].value == 1
    assert plan["micro_batch"].text == "up to 32,768 tokens per forward pass (32 x 1024)"
    checkpoint = plan["checkpointed_layers"]
    assert checkpoint.value == 0
    # Measured on the group the training runs, not on 32 rows of 1024 tokens
    assert "the largest of the 2 micro-batches this job trains" in checkpoint.reason
    assert "(4 examples, end to end); 32 rows of 1024 would be 32,768" in checkpoint.reason
    # 1e-4 at 8 x 1024 tokens, times the square root of 32768 / 8192
    rate = plan["learning_rate"]
    assert rate.value == 2e-4 and "sqrt(32,768 / 8,192) = 2" in rate.reason and "square root" in rate.reason
    assert "32,768 tokens per update" in str(plan)
    trainer = job.trainer()
    assert trainer.args.per_device_train_batch_size == 32 and trainer.args.gradient_accumulation_steps == 1
    assert trainer.args.learning_rate == 2e-4 and trainer.args.warmup_steps == 0


def by_tokens_measure(model, shape):
    """A GPU with 16 GiB, 10 free: a batch of t tokens holds t / 2 GiB checkpointed, and one layer t / 4 GiB."""
    return shape.tokens * GIB // 2, shape.tokens * GIB // 4, 10 * GIB, 16 * GIB


def test_a_measured_budget_is_the_largest_whose_groups_fit(tiny_model, tokenizer, tmp_path, monkeypatch):
    fake_measurements(monkeypatch, measure=by_tokens_measure)
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(tokens_per_update="auto", max_length=4))
    plan = job.plan()
    # Each row of TEXT cut to at most 4 tokens. A budget of 4 rows or more groups all four, up to 16 tokens,
    # which hold up to 12 GiB; 2 rows of 4 tokens hold two, up to 8 tokens and 6 GiB, within 7.6
    assert plan["tokens_per_update"].value == 8 and plan["micro_batch"].value == 2
    assert "2 rows of 4 would be 8" in plan["checkpointed_layers"].reason


def test_a_given_rate_is_not_scaled_by_the_budget(tiny_model, tokenizer, tmp_path, monkeypatch):
    fake_measurements(monkeypatch)
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1,
                   recipe=m2.Recipe(tokens_per_update="auto", learning_rate=3e-4))
    rate = job.plan()["learning_rate"]
    assert (rate.value, rate.by, rate.auto_value) == (3e-4, "user", 2e-4)
    assert job.trainer().args.learning_rate == 3e-4


def test_overrides_are_reported_beside_auto(tiny_model, tokenizer, tmp_path, monkeypatch):
    fake_measurements(monkeypatch)
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=2, data=TEXT8,
                   recipe=m2.Recipe(rank=8, schedule="cosine"), overrides=m2.Overrides(checkpointed_layers=1))
    plan = job.plan()
    checkpoint = plan["checkpointed_layers"]
    assert (checkpoint.value, checkpoint.by, checkpoint.auto_value) == (1, "user", 0)
    assert "1 of 2 layers (yours; auto: 0 of 2 layers" in str(plan)
    # A number of layers given by hand leaves the other layers keeping everything
    selective = plan["selective_layers"]
    assert (selective.value, selective.by, selective.auto_value) == (0, "user", 0)
    assert "checkpointed_layers given by hand" in selective.reason
    assert (plan["rank"].value, plan["rank"].by, plan["rank"].auto_value) == (8, "user", 16)
    assert plan["alpha"].value == 8 and plan["alpha"].by == "auto"  # alpha follows the rank
    assert plan["schedule"].by == "user"
    assert [layer.gradient_checkpointing for layer in _checkpointing.layers(job.trainer().model)] == [True, False]
    assert job.trainer().args.lr_scheduler_kwargs == {"min_lr_rate": 0.1}


def tight_measure(model, shape):
    """A GPU with 16 GiB, 10 free: b examples hold b / 2 GiB checkpointed, and one layer holds 2 b GiB."""
    b = examples(shape)
    return b * GIB // 2, 2 * b * GIB, 10 * GIB, 16 * GIB


def test_auto_chooses_selective_when_keeping_everything_does_not_fit(tiny_model, tokenizer, tmp_path, monkeypatch):
    fake_measurements(monkeypatch, measure=tight_measure)
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=2, data=TEXT8)
    plan = job.plan()
    # 8 examples hold 4 + 16 GiB, 4 hold 2 + 8, 2 hold 1 + 4: the first that fits in 7.6
    assert plan["micro_batch"].value == 2 and plan["accumulation"].value == 4
    # Two layers keeping everything hold 8 GiB, too much; one selective (2 GiB) beside one keeping (4) fits
    assert plan["checkpointed_layers"].value == 0 and plan["selective_layers"].value == 1
    assert "  checkpoint   0 of 2 layers (one layer for the largest of the" in str(plan)
    assert "  selective    1 of 2 layers" in str(plan)
    assert _checkpointing.kinds(job.trainer().model) == ["selective", "none"]
    assert job.trainer().args.gradient_checkpointing is False


def test_selective_layers_by_hand_get_the_fewest_whole_beside_them(tiny_model, tokenizer, tmp_path, monkeypatch):
    fake_measurements(monkeypatch, measure=tight_measure)
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=2, data=TEXT8, overrides=m2.Overrides(selective_layers=2))
    plan = job.plan()
    selective, checkpoint = plan["selective_layers"], plan["checkpointed_layers"]
    assert (selective.value, selective.by, selective.auto_value) == (2, "user", 1)
    assert (checkpoint.value, checkpoint.by) == (0, "auto")
    assert checkpoint.reason.startswith("the fewest layers recomputed whole that fit with 2 selective")
    assert _checkpointing.kinds(job.trainer().model) == ["selective", "selective"]


def test_both_layouts_by_hand_need_no_measurement(tiny_model, tokenizer, tmp_path):
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(**CPU_RECIPE),
                   overrides=m2.Overrides(micro_batch=1, checkpointed_layers=1, selective_layers=1))
    plan = job.plan()
    assert plan["checkpointed_layers"].value == 1 and plan["selective_layers"].value == 1
    assert plan["selective_layers"].by == "user" and plan["selective_layers"].auto_text.startswith("not measured")
    assert _checkpointing.kinds(job.trainer().model) == ["whole", "selective"]
    assert job.train().steps == 1


def test_selective_layers_beyond_the_model_or_with_compiled_layers_are_refused(tiny_model, tokenizer, tmp_path):
    import copy

    with pytest.raises(ValueError, match="more than the model's 2 layers"):
        tiny_job(copy.deepcopy(tiny_model), tokenizer, tmp_path, steps=1,
                 overrides=m2.Overrides(micro_batch=1, checkpointed_layers=1, selective_layers=2)).plan()
    with pytest.raises(ValueError, match="compiled"):
        tiny_job(copy.deepcopy(tiny_model), tokenizer, tmp_path, steps=1,
                 overrides=by_hand(selective_layers=1, compile_layers=True)).plan()


def test_auto_mode_on_the_cpu_asks_for_the_batch_by_hand(tiny_model, tokenizer, tmp_path):
    with pytest.raises(RuntimeError, match="micro_batch"):
        tiny_job(tiny_model, tokenizer, tmp_path, steps=1).plan()


def test_a_loaded_model_must_match_the_method(tiny_model, tokenizer, tmp_path):
    with pytest.raises(ValueError, match="qlora"):
        tiny_job(tiny_model, tokenizer, tmp_path, method="qlora", steps=1).plan()


def test_train_runs_and_saves_the_adapter(tiny_model, tokenizer, tmp_path):
    tiny_model._meepsquared = {"padding_free": False, "liger": False}
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=2,
                   recipe=m2.Recipe(effective_batch=2, optimizer="adamw_torch"),
                   overrides=m2.Overrides(micro_batch=1, checkpointed_layers=0))
    plan = job.plan()
    assert plan["accumulation"].value == 2
    assert plan["micro_batch"].auto_text.startswith("not measured")
    # The batch as examples: the stated rate, not scaled
    assert plan["effective_batch"].value == 2 and plan["learning_rate"].value == 1e-4
    result = job.train()
    assert result.steps == 2
    assert isinstance(result.final_loss, float)
    assert result.peak_memory_bytes is None  # trained on the CPU
    assert set(result.to_dict()) == {"steps", "seconds", "final_loss", "peak_memory_bytes", "output", "paused",
                                     "checkpoint", "evaluations", "milestones"}
    # Transformers saves a checkpoint at the last step
    assert result.paused is False and result.checkpoint == str(tmp_path / "out" / "checkpoint-2")
    result.save(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    with pytest.raises(RuntimeError, match="trained already"):
        job.train()


def test_grpo_builds_trls_trainer(tiny_model, tokenizer, tmp_path):
    from trl import GRPOTrainer

    prompts = Dataset.from_dict({"prompt": ["the hare", "the sloth"]})
    job = TrainingJob(model=tiny_model, tokenizer=tokenizer, data=prompts, task="grpo", steps=1,
                      output=str(tmp_path / "out"), rewards=[lambda completions, **kw: [0.0] * len(completions)],
                      recipe=m2.Recipe(effective_batch=2, generations=2, optimizer="adamw_torch"),
                      overrides=m2.Overrides(micro_batch=2, checkpointed_layers=0))
    trainer = job.trainer()
    assert isinstance(trainer, GRPOTrainer)
    assert trainer.args.num_generations == 2
    assert "GRPO" in str(job.plan())


def test_the_command_reads_a_yaml_job(tmp_path, monkeypatch, capsys):
    path = tmp_path / "job.yaml"
    path.write_text("model: org/model\ndata: org/data\nmethod: qlora\nsteps: 5\n")
    seen = []

    def plan(self):
        seen.append(self)
        return Plan(model=self.model, task=self.task, method=self.method, device="the CPU", free_bytes=None,
                    decisions=[])

    monkeypatch.setattr(TrainingJob, "plan", plan)
    assert cli.main(["train", str(path), "--plan-only"]) == 0
    assert seen == [TrainingJob(model="org/model", data="org/data", method="qlora", steps=5)]
    assert "meepsquared plan for org/model, QLoRA" in capsys.readouterr().out
    assert cli.main(["train", str(path), "--plan-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "qlora"


def test_the_new_fields_are_data_and_back():
    job = TrainingJob(model="org/model", data="org/data", task="grpo", rewards=["json:dumps"], report_to="mlflow",
                      recipe=m2.Recipe(experts=True, max_grad_norm=0.1, weight_decay=0.1, generations=8,
                                       max_completion_length=800, temperature=1.0, top_p=0.95, top_k=40,
                                       chat_template_kwargs={"enable_thinking": False}),
                      overrides=m2.Overrides(vllm=True))
    data = job.to_dict()
    assert data["report_to"] == "mlflow" and data["overrides"] == {"vllm": True}
    assert data["recipe"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert TrainingJob.from_dict(data) == job


def test_plan_and_result_are_exported():
    assert m2.Plan is job_module.Plan and m2.Result is job_module.Result
    assert {"Plan", "Result"} <= set(m2.__all__)


def by_hand(**fields):
    """Overrides for the CPU, where auto mode cannot measure, with any others given."""
    return m2.Overrides(micro_batch=1, checkpointed_layers=0, **fields)


CPU_RECIPE = {"effective_batch": 2, "optimizer": "adamw_torch"}


def budget_job(tiny_model, tokenizer, tmp_path, tokens: int, **fields):
    """A job with a token budget on the CPU: forward passes of up to 16 tokens, each conversation 3 to 5."""
    recipe = m2.Recipe(tokens_per_update=tokens, max_length=16, optimizer="adamw_torch")
    return tiny_job(tiny_model, tokenizer, tmp_path, recipe=recipe, **fields)


def conversations_in(batch) -> int:
    """The conversations of a packed batch: one row, the position ids restarting at each conversation."""
    assert batch["position_ids"].shape[0] == 1
    return int((batch["position_ids"] == 0).sum())


def test_a_token_budget_fills_each_batch_with_whole_conversations(tiny_model, tokenizer, tmp_path):
    job = budget_job(tiny_model, tokenizer, tmp_path, 8, steps=2, overrides=by_hand())
    plan = job.plan()
    budget = plan["tokens_per_update"]
    assert (budget.value, budget.by, budget.auto_value) == (8, "user", 8)
    assert budget.auto_text.startswith("8 examples per update")
    assert plan["learning_rate"].value == job_module.scaled_learning_rate(1e-4, 8)
    assert plan["accumulation"].value == 1
    assert plan["micro_batch"].text == "up to 8 tokens per forward pass (1 x 16)"
    assert "tokens per update / (tokens per forward pass x processes)" in str(plan)
    trainer = job.trainer()
    lengths = _batches.lengths(trainer.train_dataset, 16)
    assert len(lengths) == 4 and all(3 <= n <= 5 for n in lengths)
    # The groups of the first epoch's order, seeded by the trainer's seed
    first = _batches.groups(lengths, _batches.shuffled(4, 42, 0), 8, padded=False)
    assert list(trainer._meepsquared_layout.epochs[0]) == first
    assert f"({len(first[0])} in the first step)" in budget.text
    assert plan["length"].value == 2 and trainer.args.warmup_steps == plan["warmup_steps"].value
    loader = trainer.get_train_dataloader()
    batches = list(loader)
    assert len(batches) == len(loader)
    for batch in batches:
        tokens, held = batch["position_ids"].shape[1], conversations_in(batch)
        assert tokens <= 8 or held == 1
    assert sum(conversations_in(batch) for batch in batches) == 4
    result = job.train()
    assert result.steps == 2


def test_a_padded_token_budget_counts_the_padding(tiny_model, tokenizer, tmp_path):
    job = budget_job(tiny_model, tokenizer, tmp_path, 8, steps=1, overrides=by_hand(padding_free=False))
    trainer = job.trainer()
    assert job.plan()["padding_free"].value is False
    held = 0
    for batch in trainer.get_train_dataloader():
        rows, longest = batch["input_ids"].shape
        assert rows * longest <= 8 or rows == 1
        held += rows
    assert held == 4


def test_a_budget_below_one_conversation_is_one_conversation_a_step(tiny_model, tokenizer, tmp_path):
    job = budget_job(tiny_model, tokenizer, tmp_path, 1, steps=1, overrides=by_hand())
    batches = list(job.trainer().get_train_dataloader())
    assert len(batches) == 4 and all(conversations_in(batch) == 1 for batch in batches)
    assert "(1 in the first step" in job.plan()["tokens_per_update"].text


def test_a_budget_over_one_epoch_counts_the_batches_it_makes(tiny_model, tokenizer, tmp_path):
    job = budget_job(tiny_model, tokenizer, tmp_path, 8, epochs=2, overrides=by_hand())
    plan = job.plan()
    trainer = job.trainer()
    first = _batches.groups(_batches.lengths(trainer.train_dataset, 16), _batches.shuffled(4, 42, 0), 8,
                            padded=False)
    assert plan["length"].value == 2 * len(first) and f"2 epochs, {2 * len(first)} steps" in str(plan)


def test_the_budget_is_split_into_forward_passes_that_fit(tiny_model, tokenizer, tmp_path):
    # 40 tokens per update in forward passes of one row of 16: 3 passes of up to 14 tokens
    job = budget_job(tiny_model, tokenizer, tmp_path, 40, steps=1, overrides=by_hand())
    plan = job.plan()
    assert plan["accumulation"].value == 3
    assert plan["micro_batch"].text == "up to 14 tokens per forward pass (1 x 16)"
    assert job.trainer().args.gradient_accumulation_steps == 3


@pytest.mark.parametrize("length, recipe, micro", [
    ({"steps": 3}, {"effective_batch": 4}, 2),
    ({"epochs": 2}, {"effective_batch": 4}, 2),
    ({"steps": 8}, {"tokens_per_update": 8, "max_length": 16}, 1),
])
def test_the_trainer_trains_the_micro_batches_laid_out(tiny_model, tokenizer, tmp_path, length, recipe, micro):
    job = tiny_job(tiny_model, tokenizer, tmp_path, data=TEXT8, recipe=m2.Recipe(optimizer="adamw_torch", **recipe),
                   overrides=m2.Overrides(micro_batch=micro, checkpointed_layers=0), **length)
    trainer = job.trainer()
    laid = trainer._meepsquared_layout
    seen = []
    step = trainer.training_step

    def recording(model, inputs, *args, **kwargs):
        seen.append(inputs["input_ids"].clone())
        return step(model, inputs, *args, **kwargs)

    trainer.training_step = recording
    result = job.train()
    expected = [trainer.data_collator([trainer.train_dataset[i] for i in batch])["input_ids"] for batch in laid.trained]
    assert len(seen) == len(expected) and all(torch.equal(a, b) for a, b in zip(seen, expected))
    assert result.steps == laid.steps == job.plan()["length"].value


def test_a_layout_for_another_loading_is_refused(tiny_model, tokenizer, tmp_path):
    trainer = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(**CPU_RECIPE),
                       overrides=by_hand()).trainer()
    laid = trainer._meepsquared_layout
    with pytest.raises(ValueError, match="laid out for"):
        _batches.install(trainer, dataclasses.replace(laid, loading=dataclasses.replace(laid.loading, seed=7)))


def test_an_auto_budget_on_the_cpu_asks_for_the_batch_by_hand(tiny_model, tokenizer, tmp_path):
    with pytest.raises(RuntimeError, match=r"Recipe\(tokens_per_update=\.\.\.\)"):
        tiny_job(tiny_model, tokenizer, tmp_path, steps=1,
                 recipe=m2.Recipe(tokens_per_update="auto", optimizer="adamw_torch"), overrides=by_hand()).plan()


def test_padding_free_set_by_hand_cuts_the_examples(tiny_model, tokenizer, tmp_path):
    """Overrides(padding_free=True) with a max_length: TRL 1.x refuses that pair in its config, meepsquared cuts instead."""
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(max_length=3, **CPU_RECIPE),
                   overrides=m2.Overrides(padding_free=True, micro_batch=2, checkpointed_layers=0))
    trainer = job.trainer()
    assert trainer.args.padding_free is True and trainer.args.max_length == 3
    assert job.plan()["padding_free"].by == "user"
    batch = trainer.data_collator([trainer.train_dataset[0], trainer.train_dataset[1]])
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]


def test_generating_between_plan_and_train_keeps_the_layers_compiled(tiny_model, tokenizer, tmp_path):
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(**CPU_RECIPE),
                   overrides=by_hand(compile_layers=True))
    assert job.plan()["compile_layers"].value is True
    model = job.trainer().model
    try:
        assert _compile.compiled(model)
        m2.MeepLanguageModel.for_inference(model)
        assert not _compile.compiled(model)
        m2.MeepLanguageModel.for_training(model)
        assert _compile.compiled(model)
        assert [layer.gradient_checkpointing for layer in _checkpointing.layers(model)] == [False, False]
    finally:
        _compile.uncompiled(model)


def test_for_training_compiles_nothing_the_trainer_did_not(tiny_model):
    model = m2.MeepLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False,
                                                  compile_layers=True)
    m2.MeepLanguageModel.for_inference(model)
    m2.MeepLanguageModel.for_training(model)
    assert not _compile.compiled(model)


def test_lora_on_the_experts(tiny_moe, tokenizer, tmp_path):
    job = TrainingJob(model=tiny_moe, tokenizer=tokenizer, data=TEXT, steps=1, output=str(tmp_path / "out"),
                      recipe=m2.Recipe(experts=True, **CPU_RECIPE), overrides=by_hand())
    plan = job.plan()
    experts = plan["experts"]
    assert experts.value is True and experts.by == "user" and experts.auto_value is False
    assert "4 expert stacks (mlp.experts.down_proj, mlp.experts.gate_up_proj)" in experts.text
    assert "routed LoRA" in experts.reason and "not on a CUDA GPU" in experts.reason
    model = job.trainer().model
    assert sum(_moe.routed(module) for module in model.modules()) == 2  # the outer wrapper of each layer's experts
    assert any("experts" in name and "lora_A" in name for name, _ in model.named_parameters())


def test_experts_are_frozen_unless_asked(tiny_moe, tokenizer, tmp_path):
    job = TrainingJob(model=tiny_moe, tokenizer=tokenizer, data=TEXT, steps=1, output=str(tmp_path / "out"),
                      recipe=m2.Recipe(**CPU_RECIPE), overrides=by_hand())
    experts = job.plan()["experts"]
    assert (experts.value, experts.by, experts.text) == (False, "auto", "frozen")
    assert "Recipe(experts=True)" in experts.reason


def test_experts_on_a_dense_model_are_refused(tiny_model, tokenizer, tmp_path):
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(experts=True, **CPU_RECIPE),
                   overrides=by_hand())
    with pytest.raises(ValueError, match="no expert stacks"):
        job.plan()


CHAT = Dataset.from_dict({"messages": [[{"role": "user", "content": "the hare"},
                                        {"role": "assistant", "content": "runs fast"}]] * 2})


def kept_tokens(trainer, tokenizer):
    """The tokens of a collated batch of two examples that count in the loss, without end-of-sequence tokens."""
    batch = trainer.data_collator([trainer.train_dataset[0], trainer.train_dataset[1]])
    kept = [tokenizer.convert_ids_to_tokens(int(i))
            for i, label in zip(batch["input_ids"].flatten(), batch["labels"].flatten()) if label != -100]
    return [token for token in kept if token != tokenizer.eos_token]


def chat_job(tiny_model, tokenizer, tmp_path, **recipe):
    tiny_model.resize_token_embeddings(len(tokenizer))
    return TrainingJob(model=tiny_model, tokenizer=tokenizer, data=CHAT, steps=1, output=str(tmp_path / "out"),
                       recipe=m2.Recipe(**CPU_RECIPE, **recipe), overrides=by_hand())


def test_conversations_train_on_the_answers_with_trls_mask(tiny_model, chat_tokenizer, tmp_path):
    from conftest import CHATML_MARKED

    tokenizer = chat_tokenizer(CHATML_MARKED)
    job = chat_job(tiny_model, tokenizer, tmp_path)
    answers = job.plan()["answers_only"]
    assert answers.value is True and answers.by == "auto"
    assert "TRL's assistant-only loss" in answers.text and "generation" in answers.reason
    trainer = job.trainer()
    assert trainer.args.assistant_only_loss is True
    assert kept_tokens(trainer, tokenizer) == ["runs", "fast", "<|im_end|>"] * 2


def test_conversations_train_on_the_answers_with_meepsquareds_markers(tiny_model, chat_tokenizer, tmp_path):
    from conftest import CHATML

    tokenizer = chat_tokenizer(CHATML)
    job = chat_job(tiny_model, tokenizer, tmp_path)
    answers = job.plan()["answers_only"]
    assert "train_on_responses_only" in answers.text
    assert "'<|im_start|>assistant\\n'" in answers.reason and "'<|im_start|>user\\n'" in answers.reason
    trainer = job.trainer()
    assert not trainer.args.assistant_only_loss
    assert isinstance(trainer.data_collator, ResponsesOnly)
    assert kept_tokens(trainer, tokenizer) == ["runs", "fast", "<|im_end|>"] * 2


def test_conversations_can_train_on_every_token(tiny_model, chat_tokenizer, tmp_path):
    from conftest import CHATML

    tokenizer = chat_tokenizer(CHATML)
    job = chat_job(tiny_model, tokenizer, tmp_path, answers_only=False)
    answers = job.plan()["answers_only"]
    assert (answers.value, answers.by, answers.text) == (False, "user", "every token")
    assert "user" in kept_tokens(job.trainer(), tokenizer)


def test_pairs_train_on_the_completions(tiny_model, tokenizer, tmp_path):
    pairs = Dataset.from_dict({"prompt": ["the hare", "the sloth"], "completion": [" runs fast", " sleeps"]})
    job = TrainingJob(model=tiny_model, tokenizer=tokenizer, data=pairs, steps=1, output=str(tmp_path / "out"),
                      recipe=m2.Recipe(**CPU_RECIPE), overrides=by_hand())
    assert "TRL's completion-only loss" in job.plan()["answers_only"].text
    assert job.trainer().args.completion_only_loss is True


def test_plain_text_trains_on_every_token(tiny_model, tokenizer, tmp_path):
    answers = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(**CPU_RECIPE),
                       overrides=by_hand()).plan()["answers_only"]
    assert (answers.value, answers.text) == (False, "every token")


def test_answers_only_on_plain_text_is_refused(tiny_model, tokenizer, tmp_path):
    with pytest.raises(ValueError, match="plain text"):
        tiny_job(tiny_model, tokenizer, tmp_path, steps=1, recipe=m2.Recipe(answers_only=True, **CPU_RECIPE),
                 overrides=by_hand()).plan()


def test_report_to_reaches_the_trainer(tiny_model, tokenizer, tmp_path):
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, report_to="none", recipe=m2.Recipe(**CPU_RECIPE),
                   overrides=by_hand())
    tracking = job.plan()["report_to"]
    assert (tracking.text, tracking.reason) == ("none", "as given")
    assert job.trainer().args.report_to == []


def test_clipping_and_weight_decay_reach_the_trainer(tiny_model, tokenizer, tmp_path):
    job = tiny_job(tiny_model, tokenizer, tmp_path, steps=1, overrides=by_hand(),
                   recipe=m2.Recipe(max_grad_norm=0.1, weight_decay=0.01, **CPU_RECIPE))
    plan = job.plan()
    assert plan["max_grad_norm"].by == "user" and plan["max_grad_norm"].auto_value == 1.0
    assert job.trainer().args.max_grad_norm == 0.1 and job.trainer().args.weight_decay == 0.01


PROMPTS = Dataset.from_dict({"prompt": ["the hare", "the sloth"]})


def zero_reward(completions, **kwargs):
    return [0.0] * len(completions)


def test_grpo_generation_settings_reach_trls_config(tiny_model, tokenizer, tmp_path):
    job = TrainingJob(model=tiny_model, tokenizer=tokenizer, data=PROMPTS, task="grpo", steps=1,
                      output=str(tmp_path / "out"), rewards=[zero_reward],
                      recipe=m2.Recipe(effective_batch=2, generations=2, optimizer="adamw_torch",
                                       max_completion_length=4, temperature=0.7, top_p=0.9, top_k=5,
                                       chat_template_kwargs={"enable_thinking": False}),
                      overrides=m2.Overrides(micro_batch=2, checkpointed_layers=0))
    plan = job.plan()
    for name in ("max_completion_length", "temperature", "top_p", "top_k", "chat_template_kwargs"):
        assert plan[name].by == "user" and "TRL's default" in plan[name].auto_text
    assert plan["vllm"].value is False and "Transformers' generate" in plan["vllm"].text
    assert "enable_thinking=False" in str(plan)
    args = job.trainer().args
    assert (args.max_completion_length, args.temperature, args.top_p, args.top_k) == (4, 0.7, 0.9, 5)
    assert args.chat_template_kwargs == {"enable_thinking": False}
    assert args.use_vllm is False


def test_a_completion_longer_than_the_example_is_refused(tiny_model, tokenizer, tmp_path):
    job = TrainingJob(model=tiny_model, tokenizer=tokenizer, data=PROMPTS, task="grpo", steps=1,
                      output=str(tmp_path / "out"), rewards=[zero_reward],
                      recipe=m2.Recipe(effective_batch=2, generations=2, max_length=64, max_completion_length=64),
                      overrides=m2.Overrides(micro_batch=2, checkpointed_layers=0))
    with pytest.raises(ValueError, match="no room for the prompt"):
        job.plan()


def test_vllm_without_vllm_is_refused(tiny_model, tokenizer, tmp_path, monkeypatch):
    real_find_spec = job_module.importlib.util.find_spec
    monkeypatch.setattr(job_module.importlib.util, "find_spec",
                        lambda name, *a: None if name == "vllm" else real_find_spec(name, *a))
    job = TrainingJob(model=tiny_model, tokenizer=tokenizer, data=PROMPTS, task="grpo", steps=1,
                      output=str(tmp_path / "out"), rewards=[zero_reward],
                      recipe=m2.Recipe(effective_batch=2, generations=2),
                      overrides=m2.Overrides(micro_batch=2, checkpointed_layers=0, vllm=True))
    with pytest.raises(ValueError, match=r"overrides\.vllm=True"):
        job.plan()


def test_the_plan_says_why_the_attention_and_the_batches(tiny_model, tokenizer, tmp_path, monkeypatch):
    """A model whose heads FlashAttention 2 cannot compute: PyTorch's memory-efficient kernel, packed batches."""
    from meepsquared import MeepLanguageModel

    why = "heads of 512: FlashAttention 2 stops at 256; PyTorch's memory-efficient kernel with packed sequences"

    def loaded(model_name, **kwargs):
        assert "attn_implementation" not in kwargs
        tiny_model.config._attn_implementation = "meepsquared_varlen"
        tiny_model._meepsquared = {"padding_free": True, "padding_free_reason": None, "liger": False,
                              "attention_auto": ("meepsquared_varlen", why)}
        return tiny_model, tokenizer

    monkeypatch.setattr(MeepLanguageModel, "from_pretrained", staticmethod(loaded))
    fake_measurements(monkeypatch)
    plan = TrainingJob(model="org/model", data=TEXT, output=str(tmp_path / "out"), steps=2,
                       overrides=m2.Overrides(liger=False)).plan()
    attention = plan["attention"]
    assert (attention.value, attention.by, attention.reason) == ("meepsquared_varlen", "auto", why)
    batches = plan["padding_free"]
    assert batches.text == "packed, no padding"
    assert batches.reason == "PyTorch's memory-efficient kernel keeps the examples apart (cu_seqlens)"
    assert f"  attention    meepsquared_varlen ({why})" in str(plan)
