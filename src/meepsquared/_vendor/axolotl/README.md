# Vendored: Axolotl's LoRA kernels

The files here are from Axolotl (https://github.com/axolotl-ai-cloud/axolotl),
tag v0.21.0, commit 1f1eb05a391c43e65c1c5ee7ec6ea1b8a841d5d1, directory
src/axolotl/kernels/, under the Apache License 2.0 (LICENSE beside this file).
meepsquared uses them for get_peft_model(fused_lora=True).

| File | Changed |
|---|---|
| lora.py | no |
| quantize.py | no |
| utils.py | no |
| swiglu.py | the import of op_registry made relative |
| geglu.py | the import of op_registry made relative |
| op_registry.py | Python's logging in place of axolotl.utils.logging |

Each changed file says so in its first line.
- `lora.py`: `_base_mm`, the product with the base weight, runs on the FP8 tensor cores where the weight is FP8 (`meepsquared._fp8.scaled_mm`); the forward of `matmul_lora` and the input gradients of `LoRA_MLP`, `LoRA_QKV`, `LoRA_QK` and `LoRA_O` go through it.
- `lora.py`: the forward of `matmul_lora` adds the adapter's product and the biases out of place (`torch.addmm`, `+`) while a selectively recomputed layer computes (`meepsquared._selective.in_place()` false), and in place (`addmm_`, `+=`) as before everywhere else: in a selective layer the base product's output is kept for the backward pass, and a write over it would change what the recompute reads.
