#!/bin/sh
# The Spark's queue to the end: the references, QLoRA and mixture of experts, Step 3
cd /home/thinkube/thinkube-ai/notebooks/llebre/benchmarks/results/round2/queue
python3 ref_gb10c.py >> ref_gb10.log 2>&1
python3 ext_gb10.py >> ext_gb10.log 2>&1
python3 step3_gb10.py >> step3_gb10.log 2>&1
echo "$(date -u +%H:%M) queue done" >> gb10_queue.log
