# NeMo AutoModel at meepsquared's SFT settings

NVIDIA NeMo AutoModel on one DGX Spark, with the same data and settings as
`sft.py`: Llama 3.1 8B Instruct in 16-bit, LoRA rank 16 on the seven
projections, 8 conversations of no_robots per step, the first 240 in order,
up to 1,024 tokens each, 30 steps.

| Recipe | What changes |
|---|---|
| `llama3_1_8b_no_robots_lora_spark.yaml` | AutoModel's defaults for these settings |
| `llama3_1_8b_no_robots_lora_spark_compile.yaml` | torch.compile, and the fused linear cross-entropy that does not build the full logits |
| `llama3_1_8b_no_robots_lora_spark_checkpointed.yaml` | activation checkpointing |

Run each recipe in a new container: AutoModel saves a checkpoint at the end
of a run and the next run in the same directory resumes from it.

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN -v "$PWD:/work" -w /opt/Automodel nvcr.io/nvidia/nemo-automodel:26.08 \
  automodel /work/llama3_1_8b_no_robots_lora_spark.yaml
```

The seconds per step are the time from step 10 to step 29, from the
timestamps of AutoModel's step lines. The memory is the largest `mem` of those
lines, `torch.cuda.max_memory_allocated`, the measure `sft.py` reports.
The results are in `../results/automodel.jsonl`.
