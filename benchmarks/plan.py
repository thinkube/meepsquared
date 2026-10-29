# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""meepsquared's plan for a benchmark's training, written as JSON for every arm to take (sft.py --plan).

    python plan.py --model meta-llama/Llama-3.1-8B-Instruct --method lora --steps 30 --out results/plans/llama-lora.json

The plan is made on the benchmark's own conversations, in the benchmark's
order, for the same number of steps, on the machine the benchmark runs on.
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--source", choices=["huggingface", "thinkube"], default="huggingface")
parser.add_argument("--method", choices=["lora", "qlora", "fp8"], default="lora")
parser.add_argument("--attn", help="meepsquared: the attention implementation, when auto mode's choice is not wanted (sdpa for heads wider than 256)")
parser.add_argument("--data", default="HuggingFaceH4/no_robots")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--rows", type=int, help="the conversations the benchmark trains on (sft.py: effective batch x steps), "
                    "so that the plan lays out the batches the benchmark trains; 64 per step when not given")
parser.add_argument("--tokens-per-update", help="the batch as a token budget: auto (measured) or a number of tokens; "
                    "batched by examples when not given")
parser.add_argument("--out", required=True)
a = parser.parse_args()

import meepsquared as m2  # noqa: E402

from common import chat_dataset, model_location  # noqa: E402

# The rows the benchmark trains on: given, or enough for the effective batch the plan chooses
data = chat_dataset(a.data, a.rows or 64 * a.steps)
given = {"overrides": m2.Overrides(attention=a.attn)} if a.attn else {}
if a.tokens_per_update:
    budget = a.tokens_per_update if a.tokens_per_update == "auto" else int(a.tokens_per_update)
    given["recipe"] = m2.Recipe(tokens_per_update=budget)
job = m2.TrainingJob(model=model_location(a.model, a.source), data=data, method=a.method, steps=a.steps, **given)
plan = job.plan()
print(plan, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({**plan.to_dict(), "benchmark_model": a.model}, indent=1))
print("RESULT plan", a.out, flush=True)
