# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Fast LoRA and QLoRA fine-tuning on Transformers, PEFT and TRL.

    import meepsquared as m2
    job = m2.TrainingJob(model="org/model", data="org/dataset", method="lora", steps=1000)
    print(job.plan())
    job.train().save("out")

The calls code coming from Unsloth uses stay (see meepsquared.models):

    model, tokenizer = m2.MeepLanguageModel.from_pretrained("org/model", load_in_4bit=True)
    model = m2.MeepLanguageModel.get_peft_model(model, r=16)

Importing meepsquared makes TRL's SFTTrainer and GRPOTrainer take their settings
from a meepsquared model, unless their config sets them (see meepsquared._trl), and
makes GRPO's weight updates reach vLLM under their checkpoint names (see meepsquared._vllm).
It also sets PyTorch's CUDA allocator to expandable segments, unless the
allocator is already set (see meepsquared._allocator), and registers the attention
implementation "meepsquared_varlen" with Transformers (see meepsquared._varlen).
"""

from meepsquared import _allocator

_allocator.apply()  # before anything below can start CUDA

from meepsquared import _trl, _varlen, _vllm  # noqa: E402
from meepsquared.job import Overrides, Plan, Recipe, Result, TrainingJob  # noqa: E402
from meepsquared.models import MeepLanguageModel, MeepModel  # noqa: E402
from meepsquared.multi_gpu import fsdp_config  # noqa: E402

__version__ = "0.1.0"
__all__ = ["TrainingJob", "Recipe", "Overrides", "Plan", "Result", "MeepLanguageModel", "MeepModel",
           "fsdp_config", "__version__"]

_trl.patch_trainers()
_varlen.register()
_vllm.patch_vllm_generation()
