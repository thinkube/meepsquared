# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The meepsquared command: `meepsquared train job.yaml`, also `python -m meepsquared train job.yaml`.

    meepsquared train job.yaml               prints the plan, trains, and saves the adapter in the job's output folder
    meepsquared train job.yaml --plan-only   prints the plan and stops; nothing is trained
    --json                              prints the plan and the result as JSON, for a platform to read
    --status-file STATUS.json           writes the training's state there as it goes (see meepsquared._sessions)
    --stop-file STOP                    when this file exists, stops at the end of the step with a checkpoint
    --resume                            continues from the latest checkpoint in the job's output folder

A SIGTERM stops the training as the stop file does. The exit code: 0, the
training completed; 75, it stopped when asked and saved a checkpoint to
resume from; any other, it failed (the status file says why).

The YAML file holds the fields of a TrainingJob (see meepsquared.job). Relative
paths in it (data, reward files) are read from the current directory.
"""

import argparse
import json
import sys
import traceback
from typing import Optional

# EX_TEMPFAIL of sysexits.h: stopped, to be continued
PAUSED = 75


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="meepsquared", description="Fast LoRA and QLoRA fine-tuning")
    commands = command.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="plan and run a training job written as YAML")
    train.add_argument("job", help="the job's YAML file")
    train.add_argument("--plan-only", action="store_true", help="print the plan and stop; nothing is trained")
    train.add_argument("--json", action="store_true", help="print the plan and the result as JSON")
    train.add_argument("--status-file", help="a JSON file of the training's state, written as it goes")
    train.add_argument("--stop-file", help="when this file exists, stop at the end of the step with a checkpoint "
                                           "(exit code 75)")
    train.add_argument("--resume", action="store_true",
                       help="continue from the latest checkpoint in the job's output folder")
    preprocess = commands.add_parser("preprocess", help="tokenize, check and cut an SFT job's data once, into a "
                                                        "folder keyed by what made it; nothing is trained")
    preprocess.add_argument("job", help="the job's YAML file")
    preprocess.add_argument("--cache", required=True, help="the folder the preprocessed data is kept in, under "
                                                          "its key; found there when made before from the same inputs")
    preprocess.add_argument("--json", action="store_true", help="print the outcome as JSON: folder, key, found, "
                                                                "report")
    return command


def main(argv: Optional[list] = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    if args.command == "preprocess":
        return _preprocess(args)
    if args.plan_only and (args.status_file or args.stop_file or args.resume):
        command.error("--plan-only trains nothing; --status-file, --stop-file and --resume are for a training")
    try:
        return _train(args)
    except Exception as error:
        if args.status_file is not None:
            from meepsquared import _sessions

            if _sessions.main_process():
                _sessions.mark_failed(args.status_file, error)
        traceback.print_exc()
        return 1


def _preprocess(args) -> int:
    from meepsquared.job import TrainingJob

    outcome = TrainingJob.from_yaml(args.job).preprocess(args.cache)
    if args.json:
        print(json.dumps(outcome.to_dict()), flush=True)
    return 0


def _train(args) -> int:
    from meepsquared import _sessions
    from meepsquared.job import TrainingJob

    if args.status_file is not None and _sessions.main_process():
        _sessions.Status(args.status_file).update()
    job = TrainingJob.from_yaml(args.job)
    if args.plan_only:
        plan = job.plan()
        print(json.dumps(plan.to_dict(), indent=2) if args.json else plan, flush=True)
        return 0
    if args.json:
        print(json.dumps({"plan": job.plan().to_dict()}), flush=True)
    result = job.train(resume=args.resume, stop_file=args.stop_file, status_file=args.status_file,
                       stop_on_sigterm=True)
    if not result.paused:
        result.save(result.output)
    if args.json:
        print(json.dumps({"result": result.to_dict()}), flush=True)
    elif result.paused:
        print(f"meepsquared: stopped after step {result.steps} in {result.seconds:.0f} s; checkpoint in "
              f"{result.checkpoint}; continue with --resume", flush=True)
    else:
        loss = f"{result.final_loss:.4f}" if result.final_loss is not None else "none logged"
        print(f"meepsquared: trained {result.steps} steps in {result.seconds:.0f} s, final loss {loss}; "
              f"adapter saved in {result.output}", flush=True)
    return PAUSED if result.paused else 0


if __name__ == "__main__":
    sys.exit(main())
