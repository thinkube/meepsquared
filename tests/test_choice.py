# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""TrainingJob(task="choice"): candidates after one shared text, scored against a target distribution."""

import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback

import meepsquared as m2
from meepsquared import _choice, _choice_job, cli
from meepsquared.job import TrainingJob
from meepsquared.models import MeepLanguageModel

from conftest import CHATML

PROMPTS = ["the hare ", "the sloth ", "and the hare ", "the sloth sleeps and the hare "]
CANDIDATES = [["runs", "sleeps"], ["runs", "sleeps", "fast"], ["sleeps", "runs fast", "the sloth", "fast"],
              ["runs", "fast"]]
HEAD_NAMES = ["norm.weight", "norm.bias", "scalar.weight", "scalar.bias", "set_project.weight", "set_project.bias",
              "set_attention.in_proj_weight", "set_attention.in_proj_bias", "set_attention.out_proj.weight",
              "set_attention.out_proj.bias", "set_output.weight", "set_output.bias"]


@pytest.fixture
def tokenizer(chat_tokenizer):
    return chat_tokenizer(CHATML)


def make_model(tokenizer):
    """A random two-layer Llama on the CPU over the tokenizer's vocabulary, the same weights at each call, marked as
    meepsquared's from_pretrained marks it."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(config)
    model._meepsquared = {"padding_free": True, "liger": False}
    return model


def rows(count=8, chat=False):
    found = []
    for i in range(count):
        candidates = CANDIDATES[i % len(CANDIDATES)]
        k = len(candidates)
        target = [0.0] * k
        target[i % k] = 0.75
        target[(i + 1) % k] += 0.25
        prompt = PROMPTS[i % len(PROMPTS)]
        if chat:
            prompt = [{"role": "user", "content": "the sloth"}, {"role": "assistant", "content": "sleeps"},
                      {"role": "user", "content": prompt}]
        found.append({"prompt": prompt, "candidates": candidates, "target": target,
                      "right": [j == i % k for j in range(k)], "group": "even" if i % 2 == 0 else "odd"})
    return found


def write(path: Path, found: list) -> str:
    path.write_text("\n".join(json.dumps(r) for r in found) + "\n")
    return str(path)


def chooser_of(tokenizer, kind="set", per_pass=2):
    model = make_model(tokenizer)
    peft = MeepLanguageModel.get_peft_model(model, r=4, lora_alpha=8, lora_dropout=0.0,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                            use_gradient_checkpointing=False, fused_lora=False, compile_layers=False)
    decoder = peft.get_base_model().get_decoder()
    head = _choice.build_head(decoder.config.hidden_size, kind)
    return _choice.Chooser(peft, decoder, head, kind, per_pass, tokenizer.pad_token_id)


def make_job(tokenizer, tmp_path, output="out", data=None, **fields):
    recipe = {"effective_batch": 2, "head_only": 0, "warmup_steps": 1, "learning_rate": 1e-2,
              "head_learning_rate": 1e-2, "rank": 4, "alpha": 8, "dropout": 0.0, **fields.pop("recipe", {})}
    return TrainingJob(model=make_model(tokenizer), tokenizer=tokenizer, task="choice",
                       data=data if data is not None else write(tmp_path / "train.jsonl", rows()),
                       output=str(tmp_path / output), timed_steps=fields.pop("timed_steps", 0),
                       recipe=m2.Recipe(**recipe), **fields)


def logged(trainer, key):
    return {entry["step"]: entry[key] for entry in trainer.state.log_history if key in entry}


def test_a_row_is_the_shared_tokens_then_each_candidates_own(tokenizer):
    for row in rows(4) + rows(4, chat=True):
        encoded = _choice.encode(tokenizer, row)
        for c, candidate in enumerate(row["candidates"]):
            messages = ([{"role": "user", "content": row["prompt"] + candidate}] if isinstance(row["prompt"], str)
                        else row["prompt"][:-1] + [{"role": "user", "content": row["prompt"][-1]["content"]
                                                    + candidate}])
            whole = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            assert encoded["shared"] + encoded["own"][c] == tokenizer(whole, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("row, error", [
    ({"prompt": "the hare", "candidates": ["runs"], "target": [1.0]}, "at least 2"),
    ({"prompt": "the hare", "candidates": ["runs", "fast"], "target": [0.5, 0.6]}, "sums to"),
    ({"prompt": "the hare", "candidates": ["runs", "fast"], "target": [1.5, -0.5]}, "at least 0"),
    ({"prompt": "the hare", "candidates": ["runs", "fast"]}, "'target' is required"),
    ({"prompt": [{"role": "assistant", "content": "x"}], "candidates": ["a", "b"], "target": [1, 0]}, "user message"),
    ({"prompt": "the hare", "candidates": ["runs", "fast"], "target": [1, 0], "right": [True]}, "'right' must be"),
])
def test_rows_that_are_not_questions_are_refused(row, error):
    with pytest.raises(ValueError, match=error):
        _choice.check_row(row, 0, "data", target=True, right=False)


def test_a_milestone_row_needs_its_right_candidates():
    with pytest.raises(ValueError, match="'right' is required"):
        _choice.check_row({"prompt": "the hare", "candidates": ["a", "b"]}, 0, "milestone_eval", target=False,
                          right=True)


def test_the_shared_reading_is_the_full_reading_of_each_candidate(tokenizer):
    chooser = chooser_of(tokenizer, per_pass=2).eval()
    for row in rows(4):
        encoded = _choice.encode(tokenizer, row)
        with torch.no_grad():
            shared = chooser.leaves(encoded["shared"], encoded["own"], grad=False)
            for c, own in enumerate(encoded["own"]):
                whole = chooser.decoder(input_ids=torch.tensor([encoded["shared"] + own])).last_hidden_state[0, -1]
                assert torch.allclose(shared[c], whole, atol=1e-5)


def test_the_checkpointed_reading_gives_the_same_states_and_trains_the_adapter(tokenizer):
    chooser = chooser_of(tokenizer, per_pass=2)
    encoded = _choice.encode(tokenizer, rows(3)[2])
    with torch.no_grad():
        plain = chooser.leaves(encoded["shared"], encoded["own"], grad=False)
    checkpointed = chooser.leaves(encoded["shared"], encoded["own"], grad=True)
    assert torch.allclose(plain, checkpointed, atol=1e-6)
    chooser.question_loss(encoded["shared"], encoded["own"], encoded["target"], grad=True).backward()
    named = dict(chooser.peft.named_parameters())
    assert any(named[n].grad is not None and named[n].grad.abs().sum() > 0
               for n in chooser.adapter_names if "lora_B" in n)
    assert all(p.grad is not None for p in chooser.head.parameters() if p.requires_grad)


def test_the_head_is_named_as_a_saved_state_names_it_and_starts_as_its_scalar(tokenizer):
    head = _choice.build_head(32, "set")
    assert [n for n, _ in head.named_parameters()] == HEAD_NAMES
    leaves = torch.randn(3, 32)
    expected = head["scalar"](head["norm"](leaves)).squeeze(-1)
    assert torch.allclose(_choice.scores(head, leaves), expected)
    assert [n for n, _ in _choice.build_head(32, "scalar").named_parameters()] == HEAD_NAMES[:4]
    chooser = chooser_of(tokenizer)
    names = list(chooser.trainable())
    assert names[-len(HEAD_NAMES):] == ["head." + n for n in HEAD_NAMES]
    assert all(n.startswith("base_model.model.model.layers.") and ".lora_" in n and n.endswith(".default.weight")
               for n in names[:-len(HEAD_NAMES)])


def test_the_ramp_rises_then_falls_to_zero():
    assert _choice_job.ramp(0, 400, 1000) == pytest.approx(1 / 400)
    assert _choice_job.ramp(399, 400, 1000) == pytest.approx(1 - 399 / 1000)
    assert _choice_job.ramp(999, 400, 1000) == pytest.approx(1 / 1000)
    assert _choice_job.ramp(3, 0, 10) == pytest.approx(0.7)


class Adapter(TrainerCallback):
    """The sum of the absolute values of the adapter's B matrices at the end of each step."""

    def __init__(self):
        self.sums = {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        chooser = self.model
        self.sums[state.global_step] = sum(float(p.detach().abs().sum()) for n, p in zip(chooser.adapter_names,
                                                                                 chooser.adapter()) if "lora_B" in n)


def test_a_choice_job_plans_trains_keeps_milestones_and_saves(tokenizer, tmp_path):
    milestone_eval = write(tmp_path / "milestones.jsonl", rows(4))
    job = make_job(tokenizer, tmp_path, milestones=0.5, milestone_eval=milestone_eval,
                   recipe={"head_only": 2})
    plan = job.plan()
    assert plan["length"].value == 4
    assert plan["choice_head"].value == "set" and plan["candidates_per_pass"].value == 4
    assert plan["schedule"].value == "ramped_linear" and plan["head_only"].value == 2
    assert plan["milestones"].value["steps"] == [2, 4]
    trainer = job.trainer()
    watch = Adapter()
    watch.model = trainer.model
    trainer.add_callback(watch)
    result = job.train()
    assert result.steps == 4 and result.final_loss is not None
    # The first two questions, the first step, train the head alone
    assert watch.sums[1] == 0 and watch.sums[2] > 0
    rates = logged(trainer, "learning_rate")
    for step in range(1, 5):
        assert rates[step] == pytest.approx(1e-2 * _choice_job.ramp(step - 1, 1, 4))
    assert [m["step"] for m in result.milestones] == [2, 4]
    scores = result.milestones[-1]["scores"]
    assert set(scores) == {"even", "odd"} and sum(s["of"] for s in scores.values()) == 4
    folder = Path(result.milestones[-1]["folder"])
    for name in ("adapter_model.safetensors", "adapter_config.json", _choice.HEAD_FILE, _choice.CONFIG_FILE,
                 _choice.ANSWERS, "milestone.json"):
        assert (folder / name).is_file(), name
    assert "milestone/even/right" in json.dumps(trainer.state.log_history)
    result.save(tmp_path / "saved")
    assert json.loads((tmp_path / "saved" / _choice.CONFIG_FILE).read_text())["head"] == "set"
    with pytest.raises(ValueError, match="merged=True"):
        result.save(tmp_path / "merged", merged=True)


def state_at(trainer, step: int, examples: int, path: Path):
    """A callback that writes the training's state at `step` as a start_from file."""

    class Dump(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step != step:
                return
            chooser = trainer.model
            optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
            trainable = chooser.trainable()
            saved = {"examples": examples,
                     "trainable": {n: p.detach().clone() for n, p in trainable.items()},
                     "optimizer": {n: {k: optimizer.state[p][k].clone() for k in _choice.ADAM_KEYS}
                                   for n, p in trainable.items() if p in optimizer.state},
                     "milestones": [{"step": 1, "scores": {"all": {"right": 1, "of": 2}}}]}
            torch.save(saved, path)

    return Dump()


class StopAt(TrainerCallback):
    def __init__(self, step, stop_file):
        self.step, self.stop_file = step, stop_file

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            Path(self.stop_file).touch()


def test_start_from_goes_on_at_its_question_and_step(tokenizer, tmp_path):
    data = write(tmp_path / "train.jsonl", rows())
    whole = make_job(tokenizer, tmp_path, "whole", data=data, steps=6)
    trainer = whole.trainer()
    start = tmp_path / "start.pt"
    trainer.add_callback(state_at(trainer, 2, 4, start))
    whole.train()

    job = make_job(tokenizer, tmp_path, "started", data=data, steps=6, start_from=str(start))
    plan = job.plan()
    assert plan["start_from"].value == {"file": str(start), "examples": 4, "step": 2}
    started = job.trainer()
    result = job.train()
    assert result.steps == 6
    losses, rates = logged(trainer, "loss"), logged(trainer, "learning_rate")
    started_losses, started_rates = logged(started, "loss"), logged(started, "learning_rate")
    assert sorted(started_losses) == [3, 4, 5, 6]
    for step in range(3, 7):
        assert started_losses[step] == pytest.approx(losses[step], rel=1e-5, abs=1e-6)
        assert started_rates[step] == pytest.approx(rates[step], rel=1e-6)


def test_a_resumed_session_of_a_started_training_goes_on_from_its_checkpoint(tokenizer, tmp_path):
    data = write(tmp_path / "train.jsonl", rows())
    whole = make_job(tokenizer, tmp_path, "whole", data=data, steps=6)
    trainer = whole.trainer()
    start = tmp_path / "start.pt"
    trainer.add_callback(state_at(trainer, 2, 4, start))
    whole.train()

    uninterrupted = make_job(tokenizer, tmp_path, "uninterrupted", data=data, steps=6, start_from=str(start))
    reference = uninterrupted.trainer()
    uninterrupted.train()

    stop = tmp_path / "STOP"
    first = make_job(tokenizer, tmp_path, "out", data=data, steps=6, start_from=str(start))
    first.trainer().add_callback(StopAt(4, stop))
    paused = first.train(stop_file=stop)
    assert paused.paused and paused.steps == 4
    stop.unlink()

    second = make_job(tokenizer, tmp_path, "out", data=data, steps=6, start_from=str(start))
    resumed = second.train(resume=True, stop_file=stop)
    assert not resumed.paused and resumed.steps == 6
    assert "resumed session" in second.plan()["start_from"].text
    assert second.plan()["start_from"].value == first.plan()["start_from"].value
    losses, rates = logged(reference, "loss"), logged(reference, "learning_rate")
    resumed_trainer = second.trainer()
    resumed_losses, resumed_rates = logged(resumed_trainer, "loss"), logged(resumed_trainer, "learning_rate")
    for step in (5, 6):
        assert resumed_losses[step] == pytest.approx(losses[step], rel=1e-5, abs=1e-6)
        assert resumed_rates[step] == pytest.approx(rates[step], rel=1e-6)
    assert resumed_rates[5] == pytest.approx(1e-2 * _choice_job.ramp(4, 1, 6))


def test_start_from_a_checkpoint_folder_of_another_training_goes_on_at_its_step(tokenizer, tmp_path):
    data = write(tmp_path / "train.jsonl", rows())
    whole = make_job(tokenizer, tmp_path, "whole", data=data, steps=6)
    reference = whole.trainer()
    whole.train()

    stop = tmp_path / "STOP"
    first = make_job(tokenizer, tmp_path, "first", data=data, steps=6)
    first.trainer().add_callback(StopAt(2, stop))
    paused = first.train(stop_file=stop)
    assert paused.paused and paused.steps == 2
    stop.unlink()
    checkpoint = tmp_path / "first" / "checkpoint-2"

    job = make_job(tokenizer, tmp_path, "moved", data=data, steps=6, start_from=str(checkpoint))
    assert job.plan()["start_from"].value == {"file": str(checkpoint), "examples": 4, "step": 2}
    moved = job.trainer()
    assert job.train().steps == 6
    losses, rates = logged(reference, "loss"), logged(reference, "learning_rate")
    moved_losses, moved_rates = logged(moved, "loss"), logged(moved, "learning_rate")
    assert sorted(moved_losses) == [3, 4, 5, 6]
    for step in range(3, 7):
        assert moved_losses[step] == pytest.approx(losses[step], rel=1e-5, abs=1e-6)
        assert moved_rates[step] == pytest.approx(rates[step], rel=1e-6)

    with pytest.raises(ValueError, match="not the same data, length or effective_batch"):
        make_job(tokenizer, tmp_path, "longer", data=data, steps=8, start_from=str(checkpoint)).plan()
    with pytest.raises(FileNotFoundError, match="is not a checkpoint of a choice training"):
        make_job(tokenizer, tmp_path, "none", data=data, steps=6, start_from=str(tmp_path)).plan()


def test_a_start_from_state_of_other_parameters_is_refused(tokenizer, tmp_path):
    start = tmp_path / "start.pt"
    chooser = chooser_of(tokenizer)
    trainable = chooser.trainable()
    name = next(iter(trainable))
    torch.save({"examples": 2, "trainable": {n: p.detach() for n, p in trainable.items() if n != name},
                "optimizer": {}}, start)
    with pytest.raises(ValueError, match="not in the file"):
        make_job(tokenizer, tmp_path, start_from=str(start)).plan()
    torch.save({"examples": 3, "trainable": {n: p.detach() for n, p in trainable.items()}, "optimizer": {}}, start)
    with pytest.raises(ValueError, match="whole optimizer steps"):
        make_job(tokenizer, tmp_path, start_from=str(start)).plan()


def test_the_plan_gives_the_memory_a_job_should_ask_for(tokenizer, tmp_path):
    plan = make_job(tokenizer, tmp_path, timed_steps=1).plan()
    assert plan.memory_request_bytes is not None and plan.memory_request_bytes > 0
    assert plan["memory"].value["memory_request_bytes"] == plan.memory_request_bytes
    assert plan.seconds_per_step is not None


def no_model(monkeypatch):
    """The model loader and the trainer refuse: a plan that reuses measurements builds neither."""
    def refused(*args, **kwargs):
        raise AssertionError("a plan reusing measurements loaded the model or built the trainer")

    monkeypatch.setattr(_choice_job, "_load", refused)
    monkeypatch.setattr(_choice_job.ChoiceTrainer, "__init__", refused)


def test_a_plan_reuses_an_earlier_plans_measurements_without_the_model(tokenizer, tmp_path, monkeypatch):
    milestone_eval = write(tmp_path / "milestones.jsonl", rows(4))
    earlier = make_job(tokenizer, tmp_path, timed_steps=1, milestones=0.5, milestone_eval=milestone_eval,
                       report_to=[]).plan()
    assert earlier["measurements"].text == "measured on the CPU" and earlier.measurements["milestone"] is not None
    no_model(monkeypatch)
    changed = make_job(tokenizer, tmp_path, milestones=0.25, milestone_eval=milestone_eval, report_to=[],
                       recipe={"effective_batch": 4, "learning_rate": 5e-3})
    plan = changed.plan(reuse=earlier.to_dict(), reused_from="plan job j1")
    # The same data's mean question, four of them a step instead of two
    assert plan.seconds_per_step == pytest.approx(2 * earlier.seconds_per_step)
    assert plan["length"].value == 2 and plan["learning_rate"].value == 5e-3
    assert plan["milestones"].value["steps"] == [1, 2]
    assert plan["milestones"].value["seconds"] == pytest.approx(earlier["milestones"].value["seconds"])
    assert (plan.memory_request_bytes, plan["memory"].value) == (earlier.memory_request_bytes, earlier["memory"].value)
    assert plan["weights"] == earlier["weights"] and plan["targets"].text == earlier["targets"].text
    assert [d.name for d in plan.decisions] == [d.name for d in earlier.decisions]
    assert plan["measurements"].value == {"from": "plan job j1"}
    assert plan["measurements"].text.startswith("reused from plan job j1, measured on the CPU: the time of a step")
    again = make_job(tokenizer, tmp_path, report_to=[]).plan(reuse=plan, reused_from="plan job j2")
    assert again["measurements"].value == {"from": "plan job j1"} and again.measurements["milestone"] is not None


def test_a_plan_measures_again_when_what_the_measurements_depend_on_changed(tokenizer, tmp_path, monkeypatch):
    earlier = make_job(tokenizer, tmp_path, timed_steps=1, report_to=[]).plan()
    no_model(monkeypatch)
    with pytest.raises(m2.MeasureAgain, match="the LoRA rank changed"):
        make_job(tokenizer, tmp_path, report_to=[], recipe={"rank": 8}).plan(reuse=earlier)
    with pytest.raises(m2.MeasureAgain, match="the head, the set layers, the candidates per pass changed"):
        make_job(tokenizer, tmp_path, report_to=[],
                 recipe={"choice_head": "scalar", "candidates_per_pass": 2}).plan(reuse=earlier)
    longer = rows()
    longer[3] = {**longer[3], "prompt": "the sloth sleeps and the hare " * 6}
    with pytest.raises(m2.MeasureAgain, match="longest question reads"):
        make_job(tokenizer, tmp_path, data=write(tmp_path / "longer.jsonl", longer),
                 report_to=[]).plan(reuse=earlier)
    with pytest.raises(m2.MeasureAgain, match="timed no milestone"):
        make_job(tokenizer, tmp_path, report_to=[], milestones=0.5,
                 milestone_eval=write(tmp_path / "milestones.jsonl", rows(4))).plan(reuse=earlier)
    # Shorter data is priced on the measured line, its memory bounded by the longer question measured
    shorter = make_job(tokenizer, tmp_path, data=write(tmp_path / "shorter.jsonl", rows()[:2]), report_to=[])
    assert shorter.plan(reuse=earlier).memory_request_bytes == earlier.memory_request_bytes


def test_preprocessing_is_kept_and_read_by_the_plan(tokenizer, tmp_path):
    job = make_job(tokenizer, tmp_path)
    outcome = job.preprocess(tmp_path / "cache")
    assert not outcome.found and outcome.report["rows"] == 8
    # The report has the decision every task's preprocessing reports: Forge shows its text and reason
    assert outcome.report["decision"]["name"] == "data_checks"
    assert outcome.report["decision"]["text"].startswith("8 questions") and outcome.report["decision"]["reason"]
    assert make_job(tokenizer, tmp_path).preprocess(tmp_path / "cache").found
    planned = make_job(tokenizer, tmp_path, preprocessed=outcome.folder).plan()
    assert planned["data"].value == "train.jsonl"
    other = write(tmp_path / "other.jsonl", rows(6))
    with pytest.raises(ValueError, match="other inputs"):
        make_job(tokenizer, tmp_path, data=other, preprocessed=outcome.folder).plan()


def test_the_command_trains_a_choice_job_with_its_status(tokenizer, tmp_path, monkeypatch):
    output, status = tmp_path / "out", tmp_path / "status.json"
    job = make_job(tokenizer, tmp_path, eval_data=write(tmp_path / "eval.jsonl", rows(3)), eval_every=2,
                   milestones=0.5, milestone_eval=write(tmp_path / "milestones.jsonl", rows(4)), report_to=[])
    monkeypatch.setattr(TrainingJob, "from_yaml", classmethod(lambda cls, path: job))
    assert cli.main(["train", "job.yaml", "--status-file", str(status)]) == 0
    written = json.loads(status.read_text())
    assert written["state"] == "completed" and written["step"] == written["total_steps"] == 4
    assert written["eval_loss"] is not None and [m["step"] for m in written["milestones"]] == [2, 4]
    assert (output / _choice.HEAD_FILE).is_file() and (output / "adapter_model.safetensors").is_file()


def test_a_choice_job_refuses_other_tasks_settings_and_others_refuse_its():
    with pytest.raises(ValueError, match="a choice job trains"):
        TrainingJob(model="x", task="choice", method="bottleneck", data="d.jsonl")
    with pytest.raises(ValueError, match="milestone_scorer"):
        TrainingJob(model="x", task="choice", data="d.jsonl", milestones=0.5, milestone_eval="m.jsonl",
                    milestone_scorer="m:f")
    with pytest.raises(ValueError, match="'ramped_linear'"):
        TrainingJob(model="x", task="choice", data="d.jsonl", recipe=m2.Recipe(schedule="cosine"))
    with pytest.raises(ValueError, match="not a choice setting"):
        TrainingJob(model="x", task="choice", data="d.jsonl", recipe=m2.Recipe(tokens_per_update=4096))
    with pytest.raises(ValueError, match="eval_share"):
        TrainingJob(model="x", task="choice", data="d.jsonl", eval_share=0.1)
    with pytest.raises(ValueError, match="choice setting"):
        TrainingJob(model="x", data="d.jsonl", recipe=m2.Recipe(head_only=5))
    with pytest.raises(ValueError, match="choice setting"):
        TrainingJob(model="x", data="d.jsonl", start_from="s.pt")
    with pytest.raises(ValueError, match="choice job's"):
        TrainingJob(model="x", task="tagging", method="lora", data="d.jsonl", recipe=m2.Recipe(schedule="ramped_linear"))
    with pytest.raises(ValueError, match="choice_head"):
        m2.Recipe(choice_head="pairs")


def test_a_choice_job_is_data_and_back():
    job = TrainingJob(model="org/model", task="choice", method="qlora", data=["a.jsonl", "b.jsonl"],
                      start_from="state.pt", milestones=0.25, milestone_eval="key.jsonl",
                      recipe=m2.Recipe(choice_head="scalar", head_learning_rate=1e-3, candidates_per_pass=8,
                                       head_only=500, schedule="ramped_linear"))
    again = TrainingJob.from_dict(job.to_dict())
    assert again.to_dict() == job.to_dict() and again.start_from == "state.pt"


def test_qlora_quantizes_as_the_reference_training_did(monkeypatch):
    from meepsquared import models

    seen = {}

    class Loaded(Exception):
        pass

    def from_pretrained(name, **kwargs):
        seen.update(kwargs)
        raise Loaded

    monkeypatch.setattr(models.MeepLanguageModel, "from_pretrained", staticmethod(from_pretrained))
    job = TrainingJob(model="org/model", task="choice", method="qlora", data="d.jsonl")
    with pytest.raises(Loaded):
        _choice_job._load(job, [])
    assert seen["load_in_4bit"] and not seen["liger"] and seen["attn_implementation"] == "sdpa"
    config = models.four_bit_config(seen["dtype"], seen["double_quant"])
    assert config.load_in_4bit and config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert config.bnb_4bit_use_double_quant is False
    # SFT's QLoRA keeps double quantization
    assert models.four_bit_config(torch.bfloat16).bnb_4bit_use_double_quant is True



def test_milestone_weights_is_a_choice_setting_with_rows_to_answer():
    with pytest.raises(ValueError, match="milestone_weights is a choice setting"):
        TrainingJob(model="x", data="d.jsonl", milestone_weights="qlora")
    with pytest.raises(ValueError, match="milestone_weights is one of"):
        TrainingJob(model="x", task="choice", data="d.jsonl", milestones=0.5, milestone_eval="m.jsonl",
                    milestone_weights="int8")
    with pytest.raises(ValueError, match="give milestones and milestone_eval"):
        TrainingJob(model="x", task="choice", data="d.jsonl", milestone_weights="qlora")
    job = TrainingJob(model="x", task="choice", method="nvfp4", data="d.jsonl", milestones=0.5,
                      milestone_eval="m.jsonl", milestone_weights="qlora")
    assert TrainingJob.from_dict(job.to_dict()).milestone_weights == "qlora"


def test_a_second_chooser_answers_as_the_trained_one_once_its_adapter_and_head_are_copied(tokenizer, tmp_path):
    trained, other = chooser_of(tokenizer), chooser_of(tokenizer)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in trained.trainable().values():
            p.add_(torch.randn_like(p) * 0.3)
    found, _ = _choice.read_encoded(write(tmp_path / "key.jsonl", rows()), tokenizer, "milestone_eval",
                                    target=False, right=True)
    assert _choice.answer(other, found) != _choice.answer(trained, found)
    _choice.copy_trainable(trained, other)
    assert _choice.answer(other, found) == _choice.answer(trained, found)


def test_a_set_head_of_two_layers_starts_as_the_scalar_score_and_mixes_the_candidates_twice():
    torch.manual_seed(0)
    one, two = _choice.build_head(32, "set"), _choice.build_head(32, "set", set_layers=2)
    assert [n for n, _ in one.named_parameters()] == HEAD_NAMES
    names = [n for n, _ in two.named_parameters()]
    assert names[:8] == HEAD_NAMES[:8] and "set_attention_2.in_proj_weight" in names
    assert names[-2:] == ["set_output.weight", "set_output.bias"]
    leaves = torch.randn(5, 32)
    plain = two["scalar"](two["norm"](leaves)).squeeze(-1)
    # The output layer starts at zero: the scores are the scalar score's
    assert torch.allclose(_choice.scores(two, leaves), plain)
    with torch.no_grad():
        two["set_output"].weight.normal_()
        two["set_attention_2"].out_proj.weight.zero_()
        two["set_attention_2"].out_proj.bias.zero_()
    changed = _choice.scores(two, leaves)
    with torch.no_grad():
        two["set_attention_2"].out_proj.weight.normal_()
    assert not torch.allclose(_choice.scores(two, leaves), changed)
    with pytest.raises(ValueError, match="set_layers are the set head's"):
        m2.Recipe(choice_head="scalar", set_layers=2)
    with pytest.raises(ValueError, match="set_layers is a choice setting|recipe.set_layers is a choice setting"):
        TrainingJob(model="x", data="d.jsonl", recipe=m2.Recipe(set_layers=2))


def test_a_saved_head_of_other_layers_is_refused(tokenizer, tmp_path):
    saved = chooser_of(tokenizer)
    _choice.save(saved, tmp_path / "head", {"head": "set"})
    other = chooser_of(tokenizer)
    other.head = _choice.build_head(32, "set", set_layers=2)
    with pytest.raises(ValueError, match="the saved head is not this head"):
        _choice.load(other, tmp_path / "head")


def test_a_row_names_its_none_candidate_by_index():
    row = {**rows(1)[0], "none": 1}
    _choice.check_row(row, 0, "data", True, False, none=True)
    for bad in (2, -1, True, "1"):
        with pytest.raises(ValueError, match="'none' must be the index"):
            _choice.check_row({**row, "none": bad}, 0, "data", True, False)
    with pytest.raises(ValueError, match="'none' is required"):
        _choice.check_row(rows(1)[0], 0, "data", True, False, none=True)


def test_the_none_output_learns_the_none_mass_and_the_rest_the_choice_among_the_others():
    torch.manual_seed(0)
    head = _choice.build_head(32, "set", none_output=True)
    leaves = torch.randn(4, 32)
    z, states = _choice.outputs(head, leaves)
    # It starts at a probability of 0.5
    assert float(torch.sigmoid(_choice.none_logit(head, states, 3))) == pytest.approx(0.5)
    log2 = torch.log(torch.tensor(2.0))
    # All the mass on none: the loss is the none output's alone
    assert float(_choice.choice_loss(head, z, states, [0, 0, 0, 1], 3)) == pytest.approx(float(log2))
    # None of it on none: the none output's loss plus the cross-entropy of the choice among the others
    others = -(torch.tensor([0.5, 0.5, 0.0]) * z[:3].log_softmax(-1)).sum()
    assert float(_choice.choice_loss(head, z, states, [0.5, 0.5, 0, 0], 3)) == pytest.approx(float(log2 + others),
                                                                                         rel=1e-5)
    with pytest.raises(ValueError, match="names no none candidate"):
        _choice.choice_loss(head, z, states, [0.5, 0.5, 0, 0], None)
    with torch.no_grad():
        head["none_output"].bias.fill_(2.0)
    assert _choice.choose(head, z, states, 3)[0] == 3
    with torch.no_grad():
        head["none_output"].bias.fill_(-2.0)
    best, probability = _choice.choose(head, z, states, 3)
    assert best != 3 and probability < 1 - float(torch.sigmoid(torch.tensor(-2.0))) + 1e-6
    every = _choice.probabilities(head, z, states, 3)
    assert sum(every) == pytest.approx(1.0) and every[best] == pytest.approx(probability)
    assert every[3] == pytest.approx(float(torch.sigmoid(_choice.none_logit(head, states, 3))))
    with pytest.raises(ValueError, match="none_output must be True or False"):
        m2.Recipe(none_output="yes")


def test_a_job_with_the_none_output_needs_every_row_to_name_its_none_candidate(tokenizer, tmp_path):
    with pytest.raises(ValueError, match="data row 0: 'none' is required"):
        make_job(tokenizer, tmp_path, recipe={"none_output": True}).plan()
    named = [{**r, "none": len(r["candidates"]) - 1} for r in rows()]
    plan = make_job(tokenizer, tmp_path, output="named", data=write(tmp_path / "named.jsonl", named),
                    recipe={"none_output": True}).plan()
    assert plan["none_output"].value is True


def test_every_candidates_probability_is_the_softmax_and_the_chosen_one_is_its_top():
    torch.manual_seed(0)
    head = _choice.build_head(32, "set")
    z, states = _choice.outputs(head, torch.randn(5, 32))
    every = _choice.probabilities(head, z, states, None)
    best, probability = _choice.choose(head, z, states, None)
    assert len(every) == 5 and sum(every) == pytest.approx(1.0)
    assert max(range(5), key=every.__getitem__) == best and every[best] == pytest.approx(probability)
