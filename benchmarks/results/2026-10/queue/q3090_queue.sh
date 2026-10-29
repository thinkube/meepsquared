#!/bin/sh
# The RTX 3090's queue to the end: the references
cd /home/thinkube/thinkube-ai/notebooks/llebre/benchmarks/results/round2/queue
until python3 job_status.py aef333c7-9b2c-4cdd-af98-bd73b09a9ed3 >/dev/null 2>&1; do sleep 20; done
python3 ref_3090.py 1 >> ref_3090.log 2>&1
echo "$(date -u +%H:%M) queue done" >> q3090_queue.log
