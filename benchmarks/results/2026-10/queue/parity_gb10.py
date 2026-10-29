"""Parity on tkspark: the held-out loss of TRL (Llama), Unsloth at its default and llebre auto over 200 steps, 2 seeds.

Every arm trains the plan's batches in the plan's order; the held-out conversations are the data's last 400, the same for
every arm. Results to results/round2/parity.jsonl."""
import json, os, re, sys, time, urllib.request

s = json.dumps(json.load(open("/home/thinkube/.claude.json")))
TOKEN = re.search(r"Bearer ([A-Za-z0-9._-]+)", s).group(1)
API = "https://control.thinkube.com/api/v1/jupyter/jobs"
B = "/home/thinkube/thinkube/notebooks/llebre/benchmarks"
R = f"{B}/results"
P = f"{R}/round2/plans"
M = {"llama": ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--method", "lora", "--source", "thinkube"],
     "gemma": ["--model", "google/gemma-4-E4B-it", "--method", "lora", "--source", "thinkube"],
     "qwen": ["--model", "Qwen/Qwen3.5-9B", "--method", "lora", "--source", "thinkube"]}
ATTN = {"llama": "flash_attention_2", "gemma": "auto", "qwen": "auto"}
STEPS = 200


def env(pairs, cmd):
    return ["run_env.py", *[f"{k}={v}" for k, v in pairs.items()], "--", *cmd] if pairs else cmd


RUNS = [(f"plan {m}", "llebre", ["plan.py", *M[m], "--steps", str(STEPS), "--rows", str(8 * STEPS), "--out", f"{P}/{m}-parity.json"])
        for m in M]
for seed in (42, 43):
    for m in M:
        base = [*M[m], "--steps", str(STEPS), "--warmup", "10", "--seed", str(seed), "--plan", f"{P}/{m}-parity.json",
                "--eval-rows", "400", "--eval-every", "50",
                "--results", f"{R}/round2/parity.jsonl", "--progress", f"{R}/round2/progress-gb10.txt"]
        if m == "llama":
            RUNS.append((f"{m} trl seed {seed}", "llebre", ["sft.py", "--arm", "trl", "--label", f"{m}-trl", *base]))
        unsloth_env = {"UNSLOTH_COMPILE_DISABLE": "1"} if m == "qwen" else {}
        RUNS.append((f"{m} unsloth seed {seed}", "unsloth-conv1d",
                     env(unsloth_env, ["sft.py", "--arm", "unsloth", "--label", f"{m}-unsloth", *base])))
        RUNS.append((f"{m} auto seed {seed}", "llebre",
                     ["sft.py", "--arm", "llebre", "--label", f"{m}-auto", "--attn", ATTN[m], "--checkpointing", "plan", *base]))
# Unsloth's newest supported set (Unsloth 2026.10.10, TRL 1.15, transformers 5.17), round 2's Llama speed arm as it ran
RUNS.append(("llama unsloth-new speed", "unsloth-new",
             ["sft.py", "--arm", "unsloth", "--label", "llama-unsloth", *M["llama"], "--steps", "30", "--warmup", "10",
              "--seed", "42", "--plan", f"{P}/llama-lora.json", "--results", f"{R}/round2/unsloth-new.jsonl",
              "--progress", f"{R}/round2/progress-gb10.txt"]))
# TRL on Gemma, the reference its Unsloth arm's evaluation is not (see parity-eval-only), run by name after the queue
RUNS.append(("gemma trl seed 42", "llebre",
             ["sft.py", "--arm", "trl", "--label", "gemma-trl", *M["gemma"], "--steps", str(STEPS), "--warmup", "10",
              "--seed", "42", "--plan", f"{P}/gemma-parity.json", "--eval-rows", "400", "--eval-every", "50",
              "--results", f"{R}/round2/parity.jsonl", "--progress", f"{R}/round2/progress-gb10.txt"]))
# The held-out loss of the untrained Gemma through each arm's evaluation path, run by name after the queue
# (Qwen: the same check, its Unsloth arm with the compile switch its training arms have)
for m in ("gemma", "qwen"):
    for arm, kernel in (("trl", "llebre"), ("unsloth", "unsloth-conv1d"), ("llebre", "llebre")):
        extra = ["--attn", "auto", "--checkpointing", "plan"] if arm == "llebre" else []
        cmd = ["sft.py", "--arm", arm, "--label", f"{m}-{arm}-untrained", *extra, *M[m], "--steps", str(STEPS),
               "--warmup", "10", "--seed", "42", "--plan", f"{P}/{m}-parity.json", "--eval-rows", "400",
               "--eval-only", "--results", f"{R}/round2/parity-eval-only.jsonl"]
        if m == "qwen" and arm == "unsloth":
            cmd = env({"UNSLOTH_COMPILE_DISABLE": "1"}, cmd)
        RUNS.append((f"{m} eval-only {arm}", kernel, cmd))


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d.get("job", d)


def wait(job_id):
    while True:
        try:
            d = call(f"{API}/{job_id}")
            if d["status"] in ("completed", "failed", "cancelled", "error"):
                return d
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "poll error", e, flush=True)
        time.sleep(60)


failed = []
# The first run to do, counted from 0 (the runs before it are done); or the names of the runs to do, in that order.
# WAIT_FOR=<job id>:<name> first waits for a job a stopped queue left running.
ARG = sys.argv[1:]
if ARG and ARG[0].isdigit():
    TODO = [(i, r) for i, r in enumerate(RUNS) if i >= int(ARG[0]) and "eval-only" not in r[0]]
else:
    by_name = {r[0]: (i, r) for i, r in enumerate(RUNS)}
    TODO = [by_name[n] for n in ARG] if ARG else list(enumerate(RUNS))
if os.environ.get("WAIT_FOR"):
    job_id, name = os.environ["WAIT_FOR"].split(":", 1)
    d = wait(job_id)
    print(time.strftime("%H:%M:%S"), name, d["status"], d.get("error") or "", flush=True)
    if d["status"] != "completed":
        failed.append(name)
for index, (name, kernel, cmd) in TODO:
    with open("/home/thinkube/thinkube-ai/notebooks/llebre/benchmarks/results/long-progress.txt", "a") as f:
        f.write(f"=== {time.strftime('%H:%M:%S')} {name}\n")
    d = call(API, {"node": "tkspark", "notebook_path": "llebre/benchmarks/long-run.ipynb", "gpus": 1, "memory_gb": 64, "reserve_memory": False,
                   "kernel_name": kernel, "cell_timeout_seconds": 18000, "parameters": {"cmd": cmd, "cwd": B}})
    print(time.strftime("%H:%M:%S"), f"[{index}] started", name, "job", d["job_id"], "kernel", kernel, flush=True)
    d = wait(d["job_id"])
    print(time.strftime("%H:%M:%S"), f"[{index}]", name, d["status"], d.get("error") or "", flush=True)
    for r in d.get("results") or []:
        for o in r.get("outputs") or []:
            t = "".join(o.get("text", "")) if "text" in o else (o.get("ename", "") + ": " + o.get("evalue", "") if o.get("output_type") == "error" else "")
            for line in t.splitlines():
                if "RESULT" in line or "EXIT" in line or "rror" in line:
                    print("   ", line[:600], flush=True)
    if d["status"] != "completed":
        failed.append(name)
        print("failed:", name, "- the chain goes on", flush=True)
print("chain done; failed:", failed or "none", flush=True)
