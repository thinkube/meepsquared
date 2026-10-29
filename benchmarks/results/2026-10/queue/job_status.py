"""Print a notebook job's status; with --output, its cell outputs too. Exit 0 when the job has ended, 1 while it runs."""
import json, re, sys, time, urllib.request

s = json.dumps(json.load(open("/home/thinkube/.claude.json")))
TOKEN = re.search(r"Bearer ([A-Za-z0-9._-]+)", s).group(1)
job = sys.argv[1]
req = urllib.request.Request(f"https://control.thinkube.com/api/v1/jupyter/jobs/{job}", headers={"Authorization": f"Bearer {TOKEN}"})
with urllib.request.urlopen(req, timeout=60) as r:
    d = json.load(r)
d = d.get("job", d)
ended = d["status"] in ("completed", "failed", "cancelled", "error")
print(time.strftime("%H:%M:%S"), d["status"], d.get("error") or "")
if ended and "--output" in sys.argv:
    for r in d.get("results") or []:
        for o in r.get("outputs") or []:
            t = "".join(o.get("text", "")) if "text" in o else (o.get("ename", "") + ": " + o.get("evalue", "") if o.get("output_type") == "error" else "")
            if t:
                print(t[-4000:])
sys.exit(0 if ended else 1)
