# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

# meepsquared on NVIDIA's PyTorch container, for the DGX Spark (GB10) and any NVIDIA GPU.
#
#   docker build -t meepsquared -f Containerfile .
#   docker run --rm -it --gpus all -e HF_TOKEN -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v $PWD:/work -w /work meepsquared meepsquared train job.yaml
#
# podman works the same.
#
# The container's PyTorch, FlashAttention and Triton stay as NVIDIA built them
# for the GPU; pip adds meepsquared at the versions it is tested with: Transformers, PEFT,
# TRL, Liger Kernel, bitsandbytes, and the linear-attention kernels of Qwen 3.5 and 3.6.
#
# PYTORCH_TAG picks the NGC release (26.03 ships PyTorch 2.11). MEEPSQUARED is the pip
# requirement: a release from PyPI by default, or a commit of the repository,
# for example "meepsquared @ git+https://github.com/thinkube/meepsquared@<commit>".
ARG PYTORCH_TAG=26.03-py3
FROM nvcr.io/nvidia/pytorch:${PYTORCH_TAG}
ARG MEEPSQUARED=meepsquared
RUN pip install --no-cache-dir --break-system-packages "${MEEPSQUARED}"
# VLLM: the URL of a vLLM wheel built for this container's PyTorch (tk-vllm-wheels, release v<vllm>-nv<container>,
# v0.22.1-nv26.03 for this tag), for GRPO with vLLM inside the trainer; empty leaves vLLM out. vLLM's dependency
# xgrammar pins Transformers below 5, so Transformers is raised back after it.
ARG VLLM=
RUN if [ -n "${VLLM}" ]; then pip install --no-cache-dir --break-system-packages "${VLLM}" && pip install --no-cache-dir --break-system-packages "transformers>=5.17"; fi
