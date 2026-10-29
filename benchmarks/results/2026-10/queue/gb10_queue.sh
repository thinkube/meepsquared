#!/bin/sh
# The Spark's queue to the end: Step 2's rest, the references, QLoRA and mixture of experts, Step 3
cd /home/thinkube/thinkube-ai/notebooks/llebre/benchmarks/results/round2/queue
until python3 job_status.py 2758f042-29fa-4458-8609-55db44f0ee91 >/dev/null 2>&1; do sleep 20; done
python3 round2_gb10c.py 17 >> round2_gb10.log 2>&1
python3 ref_gb10c.py >> ref_gb10.log 2>&1
python3 ext_gb10.py >> ext_gb10.log 2>&1
python3 step3_gb10.py >> step3_gb10.log 2>&1
echo "$(date -u +%H:%M) queue done" >> gb10_queue.log
