# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The choice kind on the causal engine, on a tiny random Llama with a word-level tokenizer and a chat template."""

import json

import pytest
import torch

import meepserve
from meepserve import PartError

HIDDEN, LAYERS = 32, 2
WORDS = ["<s>", "<pad>", "</s>", "<unk>", "user", "model", ":", "|", "the", "hare", "sloth", "runs", "fast", "slow",
         "who", "is", "it", "none", "a", "b", "c", "d"]
TEMPLATE = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} | {% endfor %}"
            "{% if add_generation_prompt %}model :{% endif %}")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
QUESTIONS = [{"prompt": "who runs fast", "candidates": ["the hare", "the sloth", "none"]},
             {"prompt": [{"role": "user", "content": "who is slow"}], "candidates": ["a", "b c", "d", "the sloth", "c"]}]


@pytest.fixture(scope="module")
def causal(tmp_path_factory):
    """The folder of a random two-layer Llama whose tokenizer has a chat template."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    folder = tmp_path_factory.mktemp("causal")
    vocab = {w: i for i, w in enumerate(WORDS)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                        pad_token="<pad>", unk_token="<unk>")
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(folder)
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(vocab), hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
                         num_key_value_heads=2, intermediate_size=64, max_position_embeddings=128,
                         pad_token_id=vocab["<pad>"], bos_token_id=0, eos_token_id=2)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(folder)
    return folder


def write_part(folder, base, seed=1, head="set", set_layers=1, none_output=False, method="lora", weights="bf16",
               per_pass=2, adapter_edit=None, head_edit=None):
    """A choice part in format 1, with a random LoRA adapter (its B not 0, so that it changes the reading) and head."""
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    from meepserve.causal import TOWERS
    from meepserve.choice import build_head

    torch.manual_seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and n.rsplit(".", 1)[-1] in TARGETS and not any(t in n for t in TOWERS)]
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets, task_type="CAUSAL_LM"))
    state = {k: torch.randn_like(v) * 0.2 for k, v in get_peft_model_state_dict(peft).items()}
    if adapter_edit is not None:
        adapter_edit(state)
    save_file(state, str(folder / "adapter_model.safetensors"))
    (folder / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": TARGETS, "use_dora": False,
        "use_rslora": False, "rank_pattern": {}, "alpha_pattern": {}}))
    module = build_head(HIDDEN, head, set_layers, none_output)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.3)
    saved = {n: p.detach().float().contiguous() for n, p in module.named_parameters()}
    if head_edit is not None:
        head_edit(saved)
    save_file(saved, str(folder / "head.safetensors"))
    (folder / "choice.json").write_text(json.dumps({"head": head, "set_layers": set_layers,
                                                    "none_output": none_output, "model": "test/causal",
                                                    "method": method, "candidates_per_pass": per_pass}))
    (folder / "meepserve.json").write_text(json.dumps({"kind": "choice", "format": 1, "base": "test/causal",
                                                       "weights": weights}))
    return folder


def whole_reading(part, question):
    """The probabilities of a question read without the shared cache: each candidate's whole text read alone."""
    from meepserve.choice import encode, outputs, probabilities

    shared, own = encode(part.engine.tokenizer, question["prompt"], question["candidates"])
    with part.engine.reading(part.adapter) as decoder:
        leaves = torch.stack([decoder(input_ids=torch.tensor([shared + o])).last_hidden_state[0, -1] for o in own])
        z, states = outputs(part.head, leaves)
        return probabilities(part.head, z, states, question.get("none"))


def test_answers_as_the_whole_reading(causal, tmp_path):
    part = meepserve.load(write_part(tmp_path / "p", causal), causal, device="cpu")
    answers = part.answer({"model": "x", "questions": QUESTIONS})["answers"]
    assert len(answers) == 2
    for question, answer in zip(QUESTIONS, answers):
        p = answer["probabilities"]
        assert len(p) == len(question["candidates"])
        assert sum(p) == pytest.approx(1.0, abs=1e-5)
        assert answer["chosen"] == max(range(len(p)), key=p.__getitem__)
        assert answer["probability"] == pytest.approx(p[answer["chosen"]])
        assert p == pytest.approx(whole_reading(part, question), abs=2e-2)


def test_the_adapter_changes_the_reading(causal, tmp_path):
    zero = write_part(tmp_path / "zero", causal, seed=1,
                      adapter_edit=lambda s: [v.zero_() for k, v in s.items() if "lora_B" in k])
    live = write_part(tmp_path / "live", causal, seed=1)
    a = meepserve.load(zero, causal, device="cpu").answer({"questions": QUESTIONS})["answers"]
    b = meepserve.load(live, causal, device="cpu").answer({"questions": QUESTIONS})["answers"]
    assert a[0]["probabilities"] != pytest.approx(b[0]["probabilities"], abs=1e-4)


def test_candidates_per_pass_does_not_change_the_answer(causal, tmp_path):
    one = meepserve.load(write_part(tmp_path / "one", causal, per_pass=1), causal, device="cpu")
    many = meepserve.load(write_part(tmp_path / "many", causal, per_pass=8), causal, device="cpu")
    for a, b in zip(one.answer({"questions": QUESTIONS})["answers"], many.answer({"questions": QUESTIONS})["answers"]):
        assert a["probabilities"] == pytest.approx(b["probabilities"], abs=2e-2)


def test_two_parts_share_a_base(causal, tmp_path):
    first, second = write_part(tmp_path / "a", causal, seed=1), write_part(tmp_path / "b", causal, seed=2)
    alone = [meepserve.load(f, causal, device="cpu").answer({"questions": QUESTIONS}) for f in (first, second)]
    engine = meepserve.engine_for(meepserve.read_contract(first), causal, device="cpu")
    a, b = meepserve.load(first, engine=engine), meepserve.load(second, engine=engine)
    assert len(engine.names()) == 2
    for part, expected in ((a, alone[0]), (b, alone[1]), (a, alone[0])):
        for got, want in zip(part.answer({"questions": QUESTIONS})["answers"], expected["answers"]):
            assert got["probabilities"] == pytest.approx(want["probabilities"], abs=1e-5)
    a.close()
    assert len(engine.names()) == 1
    for got, want in zip(b.answer({"questions": QUESTIONS})["answers"], alone[1]["answers"]):
        assert got["probabilities"] == pytest.approx(want["probabilities"], abs=1e-5)
    b.close()
    assert engine.names() == [] and engine.peft is None
    with pytest.raises(PartError, match="closed"):
        b.answer({"questions": QUESTIONS})
    with pytest.raises(PartError, match="closed already"):
        b.close()


def test_none_output(causal, tmp_path):
    part = meepserve.load(write_part(tmp_path / "n", causal, none_output=True), causal, device="cpu")
    question = dict(QUESTIONS[0], none=2)
    answer = part.answer({"questions": [question]})["answers"][0]
    assert sum(answer["probabilities"]) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(PartError, match="'none' is required"):
        part.answer({"questions": [QUESTIONS[0]]})


def test_refusals(causal, tmp_path):
    part = meepserve.load(write_part(tmp_path / "r", causal), causal, device="cpu")
    for request, message in (({"texts": ["a"]}, "not \\['texts'\\]"),
                             ({"questions": []}, "list of questions"),
                             ({"questions": [{"prompt": "who", "candidates": ["a"]}]}, "at least 2"),
                             ({"questions": [{"prompt": "who\x00", "candidates": ["a", "b"]}]}, "x00"),
                             ({"questions": [dict(QUESTIONS[0], none=1)]}, "no none output|has none"),
                             ({"questions": [{"prompt": [{"role": "model", "content": "a"}],
                                              "candidates": ["a", "b"]}]}, "user message")):
        with pytest.raises(PartError, match=message):
            part.answer(request)


def test_load_refusals(causal, tmp_path):
    with pytest.raises(PartError, match="names the weights"):
        meepserve.load(write_part(tmp_path / "w", causal, method="qlora", weights="bf16"), causal, device="cpu")
    with pytest.raises(PartError, match="does not fit the base"):
        meepserve.load(write_part(tmp_path / "k", causal,
                                  adapter_edit=lambda s: s.pop(sorted(s)[0])), causal, device="cpu")
    with pytest.raises(PartError, match="saved head"):
        meepserve.load(write_part(tmp_path / "h", causal,
                                  head_edit=lambda s: s.pop("scalar.bias")), causal, device="cpu")
    folder = write_part(tmp_path / "c", causal)
    contract = json.loads((folder / "meepserve.json").read_text())
    del contract["weights"]
    (folder / "meepserve.json").write_text(json.dumps(contract))
    with pytest.raises(PartError, match="'weights' is required"):
        meepserve.load(folder, causal, device="cpu")


def test_the_contract_names_one_of_the_kinds_engines(causal, tmp_path):
    folder = write_part(tmp_path / "e", causal)
    contract = json.loads((folder / "meepserve.json").read_text())
    contract["engine"] = "tensorrt"
    (folder / "meepserve.json").write_text(json.dumps(contract))
    with pytest.raises(PartError, match="names the engine 'tensorrt'"):
        meepserve.load(folder, causal, device="cpu")
    contract["engine"] = "causal"
    (folder / "meepserve.json").write_text(json.dumps(contract))
    assert meepserve.load(folder, causal, device="cpu").answer({"questions": QUESTIONS})["answers"]


def test_states_and_the_base_alone(causal, tmp_path):
    part = meepserve.load(write_part(tmp_path / "s", causal), causal, device="cpu")
    with_adapter = part.answer({"questions": QUESTIONS[:1], "states": True})["answers"][0]
    alone = part.answer({"questions": QUESTIONS[:1], "states": True, "adapter": False})["answers"][0]
    k = len(QUESTIONS[0]["candidates"])
    assert len(with_adapter["states"]) == k and len(with_adapter["states"][0]) == HIDDEN
    assert len(with_adapter["scores"]) == k
    assert max(abs(x - y) for s, r in zip(with_adapter["states"], alone["states"]) for x, y in zip(s, r)) > 1e-4
    assert part.answer({"questions": QUESTIONS[:1]})["answers"][0]["probabilities"] == pytest.approx(
        with_adapter["probabilities"], abs=1e-6)
    with pytest.raises(PartError, match="true or false"):
        part.answer({"questions": QUESTIONS[:1], "states": "yes"})
