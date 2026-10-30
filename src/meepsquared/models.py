# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""MeepLanguageModel and MeepModel: load a model for training, then add LoRA.

Every part underneath is Transformers, PEFT, TRL, bitsandbytes and Liger Kernel.
"""

import contextlib
import os
import types
from typing import Any, Optional

import torch

from meepsquared import (_banner, _checkpointing, _compile, _fp8, _fused, _linear_attention, _liger, _moe, _save,
                    _selective, _shared_vllm, _split, _unified_memory, _varlen)

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
FLASH_ATTENTION = {"flash_attention_2", "flash_attention_3"}
# Attention implementations that keep the examples of a padding-free batch apart
# through position ids: FlashAttention, and PyTorch's memory-efficient kernel
# with cu_seqlens (see meepsquared._varlen)
PADDING_FREE_ATTENTION = FLASH_ATTENTION | {_varlen.NAME}


def has_linear_attention(config) -> bool:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    return "linear_attention" in (getattr(text, "layer_types", None) or [])


def _trl_passes_boundaries() -> bool:
    """TRL's padding-free collator passes cu_seq_lens from TRL 1.16."""
    import trl

    major, minor = (int(part) for part in trl.__version__.split(".")[:2])
    return (major, minor) >= (1, 16)


# FlashAttention 2 and 3 compute heads of at most this many dimensions
FLASH_ATTENTION_MAX_HEAD_DIM = 256


def _text_config(config):
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def head_widths(config) -> dict:
    """The widths of the model's attention heads, by where the config gives them.

    A config whose layers differ (Gemma 4 in Transformers 5.20 and later: its
    full-attention layers have heads of 512 dimensions, its sliding layers 256)
    gives them per layer, by kind of layer; reading head_dim on the whole config
    is refused there. Otherwise head_dim, and global_head_dim where the config
    has it (Gemma 4 before Transformers 5.20).
    """
    text = _text_config(config)
    if getattr(text, "is_heterogeneous", False):
        kinds = getattr(text, "layer_types", None) or ["decoder"] * text.num_hidden_layers
        widths = {}
        for index, kind in enumerate(kinds):
            width = getattr(text.per_layer_config[index], "head_dim", None)
            if isinstance(width, int):
                name = f"head_dim of the {kind} layers"
                widths[name] = max(widths.get(name, 0), width)
        return widths
    return {name: getattr(text, name) for name in ("head_dim", "global_head_dim")
            if isinstance(getattr(text, name, None), int)}


def check_flash_attention_heads(config, attn_implementation: str) -> None:
    """Refuse FlashAttention for a model with attention heads wider than it computes.

    Gemma 4's full-attention layers have heads of 512 dimensions;
    FlashAttention would fail at the first step.
    """
    if attn_implementation not in FLASH_ATTENTION:
        return
    too_wide = {name: width for name, width in head_widths(config).items() if width > FLASH_ATTENTION_MAX_HEAD_DIM}
    if too_wide:
        raise ValueError(
            f"attn_implementation={attn_implementation!r}: FlashAttention computes heads of at most "
            f"{FLASH_ATTENTION_MAX_HEAD_DIM} dimensions, and this model has {too_wide}. Load it without "
            f"attn_implementation, or with {_varlen.NAME!r}, to train on packed batches with PyTorch's "
            "memory-efficient kernel")


def _flash_attention_2_installed() -> bool:
    from transformers.utils import is_flash_attn_2_available

    return is_flash_attn_2_available()


def choose_attention(config) -> tuple[str, str]:
    """from_pretrained's attention when none is given, and why.

    FlashAttention 2 where it runs the model. Where it does not (heads wider
    than 256, or FlashAttention 2 not installed), PyTorch's memory-efficient
    kernel with packed sequences (see meepsquared._varlen); both train on packed
    batches. Without a CUDA GPU neither runs, and the attention is SDPA.
    """
    if not torch.cuda.is_available():
        return "sdpa", "no CUDA GPU: FlashAttention and PyTorch's memory-efficient kernel run on one"
    widest = max(head_widths(config).values(), default=0)
    if widest > FLASH_ATTENTION_MAX_HEAD_DIM:
        return _varlen.NAME, (f"heads of {widest}: FlashAttention 2 stops at {FLASH_ATTENTION_MAX_HEAD_DIM}; "
                              "PyTorch's memory-efficient kernel with packed sequences")
    if not _flash_attention_2_installed():
        return _varlen.NAME, ("FlashAttention 2 is not installed; PyTorch's memory-efficient kernel with packed "
                              "sequences")
    return "flash_attention_2", "FlashAttention 2 with packed sequences"


def keeps_examples_apart(attn_implementation: str) -> str:
    """How the attention keeps the examples of a packed batch apart, for the plan."""
    if attn_implementation == _varlen.NAME:
        return "PyTorch's memory-efficient kernel keeps the examples apart (cu_seqlens)"
    return f"the attention ({attn_implementation}) keeps the examples apart"


def padding_free_refusal(config, attn_implementation: str) -> Optional[str]:
    """Why the model cannot train on padding-free batches, or None when it can.

    It needs an attention that reads the example boundaries from the position
    ids. Linear-attention layers (the gated DeltaNet layers of Qwen3.5, Qwen3.6
    and Qwen3-Next) take their boundaries only as cu_seq_lens: they need TRL
    1.16 or later, which passes them, and flash-linear-attention, which honours
    them (see meepsquared._linear_attention). Without both, such a model is padded.
    """
    if attn_implementation not in PADDING_FREE_ATTENTION:
        return f"attention is {attn_implementation}"
    if has_linear_attention(config) and not (_trl_passes_boundaries() and _linear_attention.available()):
        return "linear-attention layers need TRL 1.16 or later and flash-linear-attention"
    return None


def padding_free_ok(config, attn_implementation: str) -> bool:
    """Whether the model trains correctly on padding-free batches (see padding_free_refusal)."""
    return padding_free_refusal(config, attn_implementation) is None


def attention_by_part(config, attn_implementation: str):
    """attn_implementation for from_pretrained: by part when the causal-LM class is the whole multimodal model.

    A checkpoint whose causal-LM class holds its vision and audio towers too
    (Gemma 4) gets attn_implementation for its language model, and for each
    tower the attention Transformers picks for it: a tower may support no
    FlashAttention (Gemma 4's audio tower supports eager only), and in text
    training it never runs. A checkpoint that loads as its language model
    alone (Qwen3.5) takes the name as it is.
    """
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    parts = getattr(type(config), "sub_configs", None) or {}
    causal_lm = MODEL_FOR_CAUSAL_LM_MAPPING.get(type(config), None)
    if "text_config" not in parts or causal_lm is None or causal_lm.config_class is not type(config):
        return attn_implementation
    return {**{name: None for name in parts}, "text_config": attn_implementation, "": attn_implementation}


def language_model_targets(model, target_modules):
    """target_modules without the layers of the model's towers.

    A model loaded with its vision and audio towers (Gemma 4) has q_proj,
    v_proj and the others in the towers too; PEFT would put LoRA on them as
    well, and refuses the towers' layer types. The towers are the sub-models
    other than the language model and the models that hold it; when they have
    layers of the target names, the targets become the full names of the
    matching layers outside them. A regular expression is taken as it is.
    """
    from transformers import PreTrainedModel

    if isinstance(target_modules, str):
        return target_modules
    decoder = model.get_decoder()
    modules = list(model.named_modules())
    prefix = next(name for name, module in modules if module is decoder)
    towers = [name for name, module in modules
              if name and isinstance(module, PreTrainedModel) and module is not decoder
              and not prefix.startswith(name + ".")]
    names = set(target_modules)
    matching = [name for name, _ in modules if name.rsplit(".", 1)[-1] in names]
    outside = [name for name in matching if not any(name.startswith(tower + ".") for tower in towers)]
    return target_modules if len(outside) == len(matching) else outside


def load_tokenizer(model_name: str, **kwargs):
    """The tokenizer, without the legacy SentencePiece spaces.

    A SentencePiece tokenizer in legacy mode puts a space after every special
    token: "<|im_start|>user" becomes "<|im_start|> user", so a model would be
    trained on text it never saw. legacy=False tokenizes as written; a
    tokenizer that is not SentencePiece ignores the argument.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = pad_token_for(tokenizer)
    return tokenizer


def pad_token_for(tokenizer) -> str:
    """A pad token for a tokenizer that names none: a padding token of its vocabulary, else its end-of-sequence token.

    Llama 3.1's vocabulary has <|finetune_right_pad_id|>, a token no text
    contains. The end-of-sequence token also ends every turn of a chat, so with
    it as the pad token, a count of the non-padding tokens (Transformers' token
    count of a padding-free batch) misses those, and a padded batch's attention
    mask is what keeps them in the loss.
    """
    padding = [token for token in tokenizer.all_special_tokens + list(tokenizer.get_added_vocab())
               if "pad" in token.lower()]
    if padding:
        return padding[0]
    return tokenizer.eos_token


def _dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is not None:
        return dtype
    if not torch.cuda.is_available():
        raise RuntimeError("meepsquared trains on a CUDA GPU, and torch sees none")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _fused_wanted(model) -> tuple[bool, str]:
    """Whether fused_lora="auto" turns the kernels on, and why not when it does not."""
    if next(model.parameters()).device.type != "cuda":
        return False, "the kernels run on a CUDA GPU"
    if not _fused.available():
        return False, "the kernels need bitsandbytes and Triton, which meepsquared installs"
    supported, activation = _fused.supported_activation(model.config)
    if not supported:
        return False, f"no fused MLP for the {activation} activation of {model.config.model_type}"
    return True, ""


class MeepLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_fp8: bool = False,
        attn_implementation: Optional[str] = None,
        liger: bool = True,
        loss_chunk_budget: int = 32,
        device_map: Any = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        fast_inference: bool = False,
        quant_storage: Optional[torch.dtype] = None,
        shared_vllm: Optional[dict] = None,
        **kwargs,
    ):
        """The model and its tokenizer, ready for get_peft_model.

        dtype: bfloat16 where the GPU supports it, else float16, unless given.
        attn_implementation: Transformers' attention implementation. When not
            given, FlashAttention 2 where it runs the model, else PyTorch's
            memory-efficient kernel with packed sequences ("meepsquared_varlen",
            see meepsquared._varlen), and SDPA without a CUDA GPU (see
            choose_attention). Given, it stands.
        load_in_4bit: QLoRA, NF4 weights with double quantization, computed in dtype.
        load_in_fp8: the frozen linear layers of the decoder held as FP8 (e4m3,
            one scale per tensor) once the adapters are added, their products on
            the FP8 tensor cores (see meepsquared._fp8), inside the fused LoRA
            kernels too; half their memory, and about 1.6 times faster products
            on a GPU of compute capability 8.9 or later. Not with load_in_4bit.
        shared_vllm: vLLM in this process, holding the model's routed expert
            stacks once for both (see meepsquared._shared_vllm): the settings of
            its engine, {"max_lora_rank", "kv_cache_bytes", "max_model_len"}.
            The model then has the engine in model._meepsquared["engine"].
            Not with load_in_4bit; on one GPU.
        quant_storage: the dtype bitsandbytes packs the 4-bit weights in; under
            FSDP2 the model's dtype (torch.bfloat16), so that a layer's weights
            share one dtype to be sharded.
        liger: Liger's kernels and fused loss, which TRL's SFTTrainer applies;
            a model type Liger has no patches for is refused unless liger=False.
        loss_chunk_budget: Liger's chunk_mem_const for the fused loss (see meepsquared._liger).
        device_map: "balanced" or "auto" spread the model's layers over the GPUs
            (see meepsquared._split); otherwise the whole model on the process's GPU: GPU
            LOCAL_RANK under torchrun or accelerate launch, else GPU 0. On a GPU
            that shares the system's memory, a model loaded whole onto it is
            loaded shard by shard (see meepsquared._unified_memory).
        fast_inference: vLLM inside the training process is not offered;
            True is refused. Generation for GRPO through vLLM goes through
            TRL's GRPOConfig(use_vllm=True). With fast_inference False, the
            arguments that only configure that vLLM (gpu_memory_utilization,
            max_lora_rank) have nothing to configure and are not used.
        Prints a banner of what the model will be trained with, unless
        MEEPSQUARED_QUIET=1 is in the environment.
        """
        if fast_inference:
            raise NotImplementedError(
                "fast_inference=True (vLLM inside the training process) is not offered by meepsquared; "
                "for GRPO, generate through vLLM with TRL's GRPOConfig(use_vllm=True)"
            )
        for vllm_only in ("gpu_memory_utilization", "max_lora_rank"):
            kwargs.pop(vllm_only, None)
        if load_in_4bit and load_in_fp8:
            raise ValueError("load_in_4bit and load_in_fp8 are two ways to hold the frozen weights; choose one")
        if load_in_fp8:
            ok, why = _fp8.available()
            if not ok:
                raise RuntimeError(f"load_in_fp8=True: {why}")
        from transformers import AutoConfig, AutoModelForCausalLM

        dtype = _dtype(dtype)
        hub = {"token": token, "trust_remote_code": trust_remote_code}
        quantization = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                                              **({"bnb_4bit_quant_storage": quant_storage} if quant_storage else {}))
        # One process per GPU (torchrun, accelerate launch) puts the model on its own GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = device_map if device_map is not None else {"": local_rank}
        on_one_gpu = isinstance(device_map, dict) and len({str(v) for v in device_map.values()}) == 1
        gpu = next(iter(device_map.values())) if on_one_gpu else None
        gpu_index = int(str(gpu).removeprefix("cuda:")) if on_one_gpu and str(gpu).removeprefix("cuda:").isdigit() else None
        shared = gpu_index is not None and _unified_memory.shares_system_memory(gpu_index)
        # Capped before the weights come: the first allocation to overflow must be this process's error
        allocator_cap = _unified_memory.cap_allocator(gpu_index) if shared else None
        loading = _unified_memory.shards_to_gpu(f"cuda:{gpu_index}") if shared else contextlib.nullcontext()
        if shared_vllm is not None:
            if load_in_4bit:
                raise ValueError("shared_vllm: vLLM shares 16-bit or FP8 expert stacks, not 4-bit ones")
            if gpu_index is None:
                raise ValueError("shared_vllm: the model must be on one GPU")
            _shared_vllm.check_architecture(model_name)
            # The expert stacks are not read: vLLM's take their place once it has loaded
            loading = _shared_vllm.trainer_without_stacks(f"cuda:{gpu_index}")
        model_config = AutoConfig.from_pretrained(model_name, **hub)
        attention_auto, attention_why = choose_attention(model_config)
        if attn_implementation is None:
            attn_implementation = attention_auto
        check_flash_attention_heads(model_config, attn_implementation)
        attention = attention_by_part(model_config, attn_implementation)
        with loading:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, attn_implementation=attention, quantization_config=quantization,
                device_map=device_map, **hub, **kwargs,
            )
        tokenizer = load_tokenizer(model_name, **hub)
        if liger:
            _liger.prepare(model, loss_chunk_budget)
        model.max_seq_length = max_seq_length
        refusal = padding_free_refusal(model.config, attn_implementation)
        padding_free = refusal is None
        if padding_free and has_linear_attention(model.config):
            _linear_attention.patch(model)
        model._meepsquared = {"padding_free": padding_free, "padding_free_reason": refusal, "liger": liger,
                         "split": _split.apply(model), "fp8": load_in_fp8,
                         "attention_auto": (attention_auto, attention_why)}
        model._meepsquared["allocator_cap"] = allocator_cap
        if shared_vllm is not None:
            # Last: once vLLM is imported, Transformers' configuration classes are vLLM's
            engine = _shared_vllm.Engine(model_name, dtype, load_in_fp8, device=torch.device(f"cuda:{gpu_index}"),
                                         **shared_vllm)
            engine.attach(model)
            _shared_vllm.log(engine.memory, "started")
            model._meepsquared["engine"] = engine
        dtype_name = str(dtype).removeprefix("torch.")
        _banner.show(model_type=model.config.model_type, liger=liger, padding_free=padding_free,
                     padding_free_reason=refusal,
                     weights=(f"4-bit NF4 computing in {dtype_name}" if load_in_4bit
                              else f"FP8 e4m3 frozen weights computing in {dtype_name}" if load_in_fp8 else dtype_name))
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        target_modules: Optional[list] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Any = "auto",
        random_state: int = 3407,
        use_rslora: bool = False,
        modules_to_save: Optional[list] = None,
        fused_lora: Any = "auto",
        compile_layers: Any = "auto",
        **kwargs,
    ):
        """LoRA adapters on the model, as a PEFT model.

        The defaults go for throughput and decide for the user: checkpointing
        measured per batch, the fused kernels where the model's MLP has one,
        the layers compiled where that pays.

        use_gradient_checkpointing: which layers compute their forward again in
            the backward pass (non-reentrant), saving memory for time: "auto"
            (or "unsloth", its name elsewhere): when the trainer is built, one
            layer is measured on the largest batch of the training data and the
            cheapest layout that fits the free memory is chosen (see
            meepsquared._checkpointing); "memory" or True for every layer, the least
            memory; "selective" for every layer recomputing only its cheap
            operations, keeping the outputs of its products (see
            meepsquared._selective; refused with compiled layers); False for none; a
            number of layers, or a fraction of them.
        random_state: the seed of the adapters' initial weights.
        fused_lora: Axolotl's fused LoRA kernels on the adapted projections and
            gated MLPs (see meepsquared._fused): "auto" uses them where the kernels
            import and the MLP's activation has one, and says so when not; True
            refuses a model with another activation; False leaves them off.
        compile_layers: each decoder layer compiled as one region with
            torch.compile, when the trainer is built: "auto" where it pays
            (see meepsquared._compile.wanted), True, or False.
        Other keyword arguments go to PEFT's LoraConfig, for example
        target_parameters for the experts of a mixture-of-experts model; LoRA on
        Transformers' expert stacks is then computed on the routed tokens only
        (see meepsquared._moe).

        The model returned also has save_pretrained_merged(directory, tokenizer,
        save_method) and push_to_hub_merged(repo_id, tokenizer, save_method,
        token): save_method "lora" for the adapter alone, "merged_16bit" for
        the base with the adapter merged in (see meepsquared._save).
        """
        from peft import LoraConfig, get_peft_model

        if getattr(model, "_meepsquared", None) is None:
            raise TypeError("get_peft_model takes a model loaded with meepsquared's from_pretrained")
        torch.manual_seed(random_state)
        config = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias,
                            target_modules=language_model_targets(
                                model, target_modules if target_modules is not None else DEFAULT_TARGETS),
                            use_rslora=use_rslora, modules_to_save=modules_to_save, task_type="CAUSAL_LM", **kwargs)
        if compile_layers == "auto":
            compile_layers = (next(model.parameters()).device.type == "cuda"
                              and _compile.wanted(model.config, model.config._attn_implementation))
        # A 4-bit model is not passed through PEFT's prepare_model_for_kbit_training:
        # it turns the norms to float32, which costs memory and speed and changes
        # the dtype every layer computes in
        model.config.use_cache = False
        peft_model = get_peft_model(model, config)
        peft_model._meepsquared_gradient_checkpointing = use_gradient_checkpointing
        peft_model._meepsquared_compile_layers = bool(compile_layers)
        if use_gradient_checkpointing == "selective":
            refusal = _selective.refusal(peft_model, bool(compile_layers))
            if refusal:
                raise ValueError(f"use_gradient_checkpointing='selective': {refusal}")
        # "auto" checkpoints every layer whole until the trainer measures what its batch needs;
        # the layers are laid out once the adapters are on, as the selective policy reads their rank
        _checkpointing.set_checkpointed(peft_model, _checkpointing.recompute(use_gradient_checkpointing,
                                                                             len(_checkpointing.layers(model))))
        peft_model._meepsquared_parameter_targets = bool(kwargs.get("target_parameters"))
        if peft_model._meepsquared_parameter_targets:
            # LoRA on expert stacks, computed on the tokens routed to each expert (see meepsquared._moe)
            _moe.apply(peft_model)
        fp8 = model._meepsquared.get("fp8", False)
        if fp8:
            # Before the fused kernels attach: they read the base layer as it will be
            _fp8.apply(peft_model)
        if fused_lora == "auto":
            fused_lora, why = _fused_wanted(model)
            if not fused_lora and not os.environ.get("MEEPSQUARED_QUIET"):
                print(f"meepsquared: fused LoRA kernels off: {why}", flush=True)
        peft_model._meepsquared_fused_lora = bool(fused_lora)
        if fused_lora:
            _fused.apply(peft_model)
        peft_model.save_pretrained_merged = types.MethodType(_save.save, peft_model)
        peft_model.push_to_hub_merged = types.MethodType(_save.push, peft_model)
        return peft_model

    @staticmethod
    def for_inference(model):
        """Generation mode: no gradient checkpointing, the layers uncompiled, the key-value cache on."""
        model.eval()
        _compile.uncompiled(model)
        if getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing: Any = None):
        """Training mode again, with gradient checkpointing as get_peft_model set it, or as given.

        An "auto" setting keeps the layout the trainer measured, or every
        layer recomputed whole before a trainer measured it. Layers the trainer
        compiled and for_inference uncompiled are compiled again, with the same
        backend.
        """
        model.train()
        _compile.recompile(model)
        setting = (getattr(model, "_meepsquared_gradient_checkpointing", False)
                   if use_gradient_checkpointing is None else use_gradient_checkpointing)
        total = len(_checkpointing.layers(model))
        measured = getattr(model, "_meepsquared_checkpointed_layers", None)
        layout = (measured if setting in _checkpointing.AUTO and measured is not None
                  else _checkpointing.recompute(setting, total))
        _checkpointing.set_checkpointed(model, layout)
        model.config.use_cache = False
        return model


class MeepModel(MeepLanguageModel):
    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        finetune_vision_layers: bool = False,
        target_modules: Optional[list] = None,
        **kwargs,
    ):
        """get_peft_model, with the targets chosen by kind of module.

        Vision layers are not trained by this version; finetune_vision_layers=True
        is refused.
        """
        if finetune_vision_layers:
            raise NotImplementedError("meepsquared does not train vision layers yet")
        if target_modules is None:
            if not finetune_language_layers:
                raise ValueError("nothing to train: finetune_language_layers is False and no target_modules")
            target_modules = (ATTENTION_TARGETS if finetune_attention_modules else []) + (
                MLP_TARGETS if finetune_mlp_modules else [])
            if not target_modules:
                raise ValueError("nothing to train: no attention and no MLP modules chosen")
        return MeepLanguageModel.get_peft_model(model, r=r, target_modules=target_modules, **kwargs)
