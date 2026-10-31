# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Under DDP (torchrun, 2 CPU processes on gloo) the trainer is set up for LoRA and trains."""

import json
import os
import subprocess
import sys
from pathlib import Path


def failure(run) -> str:
    """What a failed launch printed: each process's first error, which the launcher's own report hides."""
    lines = run.stderr.splitlines()
    first = next((i for i, line in enumerate(lines) if "Traceback" in line), 0)
    return "\n".join(lines[first:first + 80]) + "\n...\n" + run.stderr[-1500:]


def results_of(folder):
    """What each process wrote in result-<rank>.json."""
    return [json.loads(path.read_text()) for path in sorted(Path(folder).glob("result-*.json"))]


def test_two_processes_train_with_unused_parameter_search_off(tmp_path):
    script = Path(__file__).parent / "ddp" / "train_two_processes.py"
    env = {**os.environ, "OUT": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}
    run = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(script)],
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, failure(run)
    results = results_of(tmp_path)
    assert sorted(r["rank"] for r in results) == [0, 1]
    assert all(r["world"] == 2 and r["find_unused"] is False and r["steps"] == 1 for r in results)


def test_each_process_trains_its_share_of_the_micro_batches_laid_out(tmp_path):
    script = Path(__file__).parent / "ddp" / "train_job_two_processes.py"
    env = {**os.environ, "OUT": str(tmp_path), "CUDA_VISIBLE_DEVICES": "", "MEEPSQUARED_QUIET": "1"}
    run = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(script)],
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, failure(run)
    results = results_of(tmp_path)
    assert sorted(r["rank"] for r in results) == [0, 1]
    # 3 micro-batches an epoch on each process, two epochs, four steps
    assert all(r["same"] and r["batches"] == 6 and r["steps"] == r["planned"] == 4 and r["epochs"] == 2
               and r["processes"] == 2 for r in results)


def test_each_process_resumes_at_its_next_micro_batch(tmp_path):
    script = Path(__file__).parent / "ddp" / "resume_job_two_processes.py"
    env = {**os.environ, "OUT": str(tmp_path), "CUDA_VISIBLE_DEVICES": "", "MEEPSQUARED_QUIET": "1"}
    run = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(script)],
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, failure(run)
    results = results_of(tmp_path)
    assert sorted(r["rank"] for r in results) == [0, 1]
    # Stopped after step 3, within the second epoch: 5 of each process's 6 micro-batches trained, then the last
    assert all(r["same"] and r["paused"] and r["paused_at"] == 3 and (r["first"], r["second"]) == (5, 1)
               and r["steps"] == r["planned"] == 4 for r in results)
