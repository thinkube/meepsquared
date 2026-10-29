"""The RTX 3090 part of the measurement round, on tkamd2 (thinkube-release/llebre/PLAN.md section 3): each run a notebook job running a benchmark script.

A failed run is recorded and the round goes on; every result goes to results/round/.
Takes the index of the first run to do (default 0) to resume.
"""
import json, re, sys, time, urllib.request

s = json.dumps(json.load(open("/home/thinkube/.claude.json")))
TOKEN = re.search(r"Bearer ([A-Za-z0-9._-]+)", s).group(1)
API = "https://control.thinkube.com/api/v1/jupyter/jobs"
B = "/home/thinkube/thinkube/notebooks/llebre/benchmarks"
R = f"{B}/results/round"
LOCAL_PROGRESS = "/home/thinkube/thinkube-ai/notebooks/llebre/benchmarks/results/round/progress-3090.txt"

MODELS = {"llama": ("meta-llama/Llama-3.1-8B-Instruct", None), "qwen": ("Qwen/Qwen3.5-9B", None), "gemma": ("google/gemma-4-E4B-it", "sdpa")}


def env(pairs, cmd):
    return ["run_env.py", *[f"{k}={v}" for k, v in pairs.items()], "--", *cmd] if pairs else cmd


RUNS = []
P = f"{R}/../round2/plans"
LOCAL_PROGRESS = "/home/thinkube/thinkube-ai/notebooks/llebre/benchmarks/results/round2/progress-3090-local.txt"
M = {"llama": ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--method", "lora", "--source", "thinkube"],
     "gemma": ["--model", "google/gemma-4-E4B-it", "--method", "lora", "--source", "thinkube"]}
for repeat in (1, 2, 3):
    for m in M:
        base = [*M[m], "--steps", "30", "--warmup", "10", "--seed", "42", "--plan", f"{P}/{m}-lora-3090.json",
                "--results", f"{R}/../round2/speed-3090.jsonl", "--progress", f"{R}/../round2/progress-3090.txt"]
        RUNS.append((f"3090 {m} trl #{repeat}", "llebre", 48, B, 1, ["sft.py", "--arm", "trl", "--label", f"{m}-trl", *base]))
        RUNS.append((f"3090 {m} unsloth #{repeat}", "unsloth-conv1d", 48, B, 1, ["sft.py", "--arm", "unsloth", "--label", f"{m}-unsloth", *base]))

def call(url, body=None, tries=10):
    """One request to Control. A job refused with 422 (its GPUs or memory not free yet, as the last job's server
    goes away) is asked for again a minute later, up to `tries` times; then the refusal, with Control's reason, stops the run."""
    import urllib.error
    for attempt in range(1, tries + 1):
        req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}, method="POST" if body else "GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            return d.get("job", d)
        except urllib.error.HTTPError as e:
            reason = e.read().decode()[:500]
            if e.code != 422 or attempt == tries:
                raise SystemExit(f"Control refused the job ({e.code}): {reason}")
            print(time.strftime("%H:%M:%S"), f"refused (422), attempt {attempt}: {reason}", flush=True)
            time.sleep(60)


def wait(job_id):
    while True:
        try:
            d = call(f"{API}/{job_id}")
            if d["status"] in ("completed", "failed", "cancelled", "error", "lost"):
                return d
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "poll error", e, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(len(RUNS), "runs; starting at", START, flush=True)
    failed = []
    for index, (name, kernel, memory, cwd, gpus, cmd) in enumerate(RUNS[START:], START):
        with open(LOCAL_PROGRESS, "a") as f:
            f.write(f"=== {time.strftime('%H:%M:%S')} [{index}] {name}\n")
        d = call(API, {"node": "tkamd2", "notebook_path": "llebre/benchmarks/long-run.ipynb", "gpus": gpus, "memory_gb": memory,
                       "reserve_memory": False, "kernel_name": kernel, "cell_timeout_seconds": 18000,
                       "parameters": {"cmd": cmd, "cwd": cwd}})
        print(time.strftime("%H:%M:%S"), f"[{index}] started", name, "job", d["job_id"], flush=True)
        d = wait(d["job_id"])
        print(time.strftime("%H:%M:%S"), f"[{index}]", name, d["status"], (d.get("error") or "")[:200], flush=True)
        for r in d.get("results") or []:
            for o in r.get("outputs") or []:
                t = "".join(o.get("text", "")) if "text" in o else (o.get("ename", "") + ": " + o.get("evalue", "") if o.get("output_type") == "error" else "")
                for line in t.splitlines():
                    if line.startswith("RESULT") or "EXIT" in line or "Error" in line:
                        print("   ", line[:300], flush=True)
        if d["status"] != "completed":
            failed.append(f"[{index}] {name}")
    print("round done; failed:", failed or "none", flush=True)
