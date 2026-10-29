"""The shared evaluation of Unsloth's Qwen adapter, its weights renamed from model.language_model.* (Unsloth loads Qwen 3.5
through its multimodal wrapper) to model.* (the text model the TRL arm loads); the weights are unchanged."""
import json, os, re, sys, time, urllib.request

s = json.dumps(json.load(open("/home/thinkube/.claude.json")))
TOKEN = re.search(r"Bearer ([A-Za-z0-9._-]+)", s).group(1)
API = "https://control.thinkube.com/api/v1/jupyter/jobs"
B = "/home/thinkube/thinkube/notebooks/llebre/benchmarks"
R = f"{B}/results"
P = f"{R}/round2/plans"
A = f"{R}/round2/adapters"
M = {"gemma": ["--model", "google/gemma-4-E4B-it", "--method", "lora", "--source", "thinkube"],
     "qwen": ["--model", "Qwen/Qwen3.5-9B", "--method", "lora", "--source", "thinkube"]}
ATTN = {"gemma": "auto", "qwen": "auto"}
# One epoch: no_robots' 9,500 training conversations, the last 400 held out, 8 per step
STEPS = 1137
EVAL = ["--eval-rows", "400"]


def env(pairs, cmd):
    return ["run_env.py", *[f"{k}={v}" for k, v in pairs.items()], "--", *cmd] if pairs else cmd


def train(m, arm, steps, plan, out, results):
    common = [*M[m], "--steps", str(steps), "--warmup", "10", "--seed", "42", "--plan", plan, *EVAL,
              "--eval-every", "100" if steps > 100 else "10", "--save-adapter", out, "--results", results,
              "--progress", f"{R}/round2/progress-gb10.txt"]
    if arm == "meep":
        return "llebre", ["sft.py", "--arm", "llebre", "--label", f"{m}-meep", "--attn", ATTN[m], "--checkpointing", "plan", *common]
    if arm == "unsloth":
        cmd = ["sft.py", "--arm", "unsloth", "--label", f"{m}-unsloth-fastest", "--checkpointing", "false", *common]
        return "unsloth-conv1d", env({"UNSLOTH_COMPILE_DISABLE": "1"} if m == "qwen" else {}, cmd)
    return "llebre", ["sft.py", "--arm", "trl", "--label", f"{m}-trl", *common]


def shared_eval(m, adapter, label, plan):
    return "llebre", ["sft.py", "--arm", "trl", "--label", label, *M[m], "--steps", str(STEPS), "--warmup", "10",
                      "--seed", "42", "--plan", plan, *EVAL, "--eval-only", "--adapter", adapter]


SMOKE = []
for arm in ("meep", "unsloth"):
    kernel, cmd = train("gemma", arm, 20, f"{P}/gemma-parity.json", f"{A}/smoke-gemma-{arm}", f"{R}/round2/epoch-smoke.jsonl")
    SMOKE.append((f"smoke gemma {arm} 20 steps", kernel, cmd))
for arm in ("meep", "unsloth"):
    kernel, cmd = shared_eval("gemma", f"{A}/smoke-gemma-{arm}", f"smoke-gemma-{arm}-eval-trl", f"{P}/gemma-parity.json")
    SMOKE.append((f"smoke gemma {arm} shared evaluation", kernel, cmd))

RUNS = [(f"plan {m} epoch", "llebre", ["plan.py", *M[m], "--steps", str(STEPS), "--rows", "9100", "--out", f"{P}/{m}-epoch.json"])
        for m in M]
for arm in ("meep", "unsloth", "trl"):
    for m in M:
        kernel, cmd = train(m, arm, STEPS, f"{P}/{m}-epoch.json", f"{A}/{m}-{arm}", f"{R}/round2/epoch.jsonl")
        RUNS.append((f"{m} {arm} epoch", kernel, cmd))
for m in M:
    for arm in ("meep", "unsloth", "trl"):
        kernel, cmd = shared_eval(m, f"{A}/{m}-{arm}", f"{m}-{arm}-eval-trl", f"{P}/{m}-epoch.json")
        RUNS.append((f"{m} {arm} shared evaluation", kernel, cmd))


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d.get("job", d)


def submit(body):
    # A failure that can clear on its own (the gateway down for a minute) is tried again, up to ten minutes
    for attempt in range(20):
        try:
            return call(API, body)
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "submit error, trying again:", e, flush=True)
            time.sleep(30)
    raise SystemExit("Control did not take the job for ten minutes")


def wait(job_id):
    while True:
        try:
            d = call(f"{API}/{job_id}")
            if d["status"] in ("completed", "failed", "cancelled", "error", "lost"):
                return d
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "poll error", e, flush=True)
        time.sleep(60)


def run(name, kernel, cmd):
    d = submit({"node": "tkspark", "notebook_path": "llebre/benchmarks/long-run-stream.ipynb", "gpus": 1, "memory_gb": 64,
                "reserve_memory": False, "kernel_name": kernel, "cell_timeout_seconds": 18000,
                "parameters": {"cmd": cmd, "cwd": B}})
    print(time.strftime("%H:%M:%S"), "started", name, "job", d["job_id"], "kernel", kernel, flush=True)
    d = wait(d["job_id"])
    print(time.strftime("%H:%M:%S"), name, d["status"], d.get("error") or "", flush=True)
    for r in d.get("results") or []:
        for o in r.get("outputs") or []:
            t = "".join(o.get("text", "")) if "text" in o else (o.get("ename", "") + ": " + o.get("evalue", "") if o.get("output_type") == "error" else "")
            for line in t.splitlines():
                if "RESULT" in line or "EXIT" in line or "rror" in line or "adapter" in line:
                    print("   ", line[:600], flush=True)
    return d["status"] == "completed"


RUNS = [("qwen unsloth shared evaluation, renamed", *shared_eval("qwen", f"{A}/qwen-unsloth-renamed", "qwen-unsloth-eval-trl", f"{P}/qwen-epoch.json"))]
for name, kernel, cmd in RUNS:
    ok = run(name, kernel, cmd)
print("chain done; failed:", "none" if ok else name, flush=True)
