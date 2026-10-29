"""Verification of fixes F1 and F2 on tkspark (thinkube-release/llebre/PLAN.md); results to results/diag, never published."""
import json, re, sys, time, urllib.request

s = json.dumps(json.load(open("/home/thinkube/.claude.json")))
TOKEN = re.search(r"Bearer ([A-Za-z0-9._-]+)", s).group(1)
API = "https://control.thinkube.com/api/v1/jupyter/jobs"
B = "/home/thinkube/thinkube/notebooks/llebre/benchmarks"
R = f"{B}/results"
D = f"{R}/diag"


def wrapped(script, args, logs=None, log_file=None):
    """The benchmark script through run_wrapped.py: PyTorch's logs to a file if asked, llebre's allocator decision printed."""
    return [f"{D}/run_wrapped.py", logs or "-", log_file or "-", script, *args]


GEMMA = ["--arm", "llebre", "--label", "llebre-verify-f2", "--model", "google/gemma-4-E4B-it", "--method", "lora", "--source", "thinkube",
         "--attn", "sdpa", "--steps", "30", "--warmup", "10", "--results", f"{D}/verify.jsonl", "--progress", f"{R}/long-progress.txt"]
P = f"{R}/round2/plans"
Q = {"llama": ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--method", "qlora", "--source", "thinkube"],
     "gemma": ["--model", "google/gemma-4-E4B-it", "--method", "qlora", "--source", "thinkube"]}
ATTN = {"llama": "flash_attention_2", "gemma": "auto"}
def env(pairs, cmd):
    return ["run_env.py", *[f"{k}={v}" for k, v in pairs.items()], "--", *cmd] if pairs else cmd


RUNS = [(f"plan {m} qlora", "llebre", 64, False, B, ["plan.py", *Q[m], "--steps", "30", "--rows", "240", "--out", f"{P}/{m}-qlora.json"]) for m in Q]
# QLoRA: llebre auto, and the references, on the plan's training decisions, 3 repeats
for repeat in (1, 2, 3):
    for m in Q:
        base = [*Q[m], "--steps", "30", "--warmup", "10", "--seed", "42", "--plan", f"{P}/{m}-qlora.json",
                "--results", f"{R}/round2/qlora-gb10.jsonl", "--progress", f"{R}/round2/progress-gb10.txt"]
        RUNS.append((f"{m} qlora auto #{repeat}", "llebre", 64, False, B, ["sft.py", "--arm", "llebre", "--label", f"{m}-qlora-auto", "--attn", ATTN[m], "--checkpointing", "plan", *base]))
        RUNS.append((f"{m} qlora trl #{repeat}", "llebre", 64, False, B, ["sft.py", "--arm", "trl", "--label", f"{m}-qlora-trl", *base]))
        RUNS.append((f"{m} qlora unsloth #{repeat}", "unsloth-conv1d", 64, False, B, ["sft.py", "--arm", "unsloth", "--label", f"{m}-qlora-unsloth", *base]))
        RUNS.append((f"{m} qlora unsloth no ckpt #{repeat}", "unsloth-conv1d", 64, False, B, ["sft.py", "--arm", "unsloth", "--label", f"{m}-qlora-unsloth-no-checkpointing", "--checkpointing", "false", *base]))
# Mixture of experts, llebre only: its grouped kernel and PEFT's expert adapters, 3 repeats
for repeat in (1, 2, 3):
    for label, extra in (("moe-llebre", []), ("moe-peft-experts", ["--peft-experts"])):
        RUNS.append((f"{label} #{repeat}", "llebre", 96, False, B, ["moe.py", "--model", "Qwen/Qwen3.6-35B-A3B", "--source", "thinkube",
                     "--steps", "30", "--warmup", "10", "--label", label, *extra,
                     "--results", f"{R}/round2/moe-gb10.jsonl", "--progress", f"{R}/round2/progress-gb10.txt"]))
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
START = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # the first run to do, counted from 0; the runs before it are done
for name, kernel, memory, reserve, cwd, cmd in RUNS[START:]:
    with open(f"/home/thinkube/thinkube-ai/notebooks/llebre/benchmarks/results/long-progress.txt", "a") as f:
        f.write(f"=== {time.strftime('%H:%M:%S')} {name}\n")
    d = call(API, {"node": "tkspark", "notebook_path": "llebre/benchmarks/long-run.ipynb", "gpus": 1, "memory_gb": memory, "reserve_memory": reserve,
                   "kernel_name": kernel, "cell_timeout_seconds": 18000, "parameters": {"cmd": cmd, "cwd": cwd}})
    print(time.strftime("%H:%M:%S"), "started", name, "job", d["job_id"], "kernel", kernel, flush=True)
    d = wait(d["job_id"])
    print(time.strftime("%H:%M:%S"), name, d["status"], d.get("error") or "", flush=True)
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
