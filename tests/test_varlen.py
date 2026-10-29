# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, LlamaConfig, PretrainedConfig

import meepsquared  # noqa: F401  registers the attention
from meepsquared import _compile, _selective, _varlen, models
from meepsquared._varlen import NAME

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the memory-efficient kernel runs on a GPU")


def test_boundaries_from_position_ids():
    # Three examples of 3, 2 and 4 tokens, then two padding tokens: each its own one-token example
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 0]])
    cu_q, cu_k, max_q, max_k = _varlen.boundaries(1, position_ids)
    assert cu_q.tolist() == cu_k.tolist() == [0, 3, 5, 9, 10, 11]
    assert cu_q.dtype == torch.int32
    assert (max_q, max_k) == (4, 4)
    # One example that starts the row and fills it is not packed
    assert _varlen.boundaries(1, torch.arange(6)[None]) is None
    assert _varlen.boundaries(2, torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]])) is None
    assert _varlen.boundaries(1, None) is None


def test_boundaries_as_trl_gives_them():
    cu = torch.tensor([0, 3, 5])
    cu_q, cu_k, max_q, max_k = _varlen.boundaries(1, torch.tensor([[0, 1, 2, 0, 1]]), cu, cu, 3, 3)
    assert cu_q.tolist() == [0, 3, 5] and cu_q.dtype == torch.int32 and (max_q, max_k) == (3, 3)
    with pytest.raises(ValueError, match="max_length_q"):
        _varlen.boundaries(1, None, cu, cu, None, 3)


def test_the_name_is_registered_with_transformers():
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert NAME in ALL_ATTENTION_FUNCTIONS.valid_keys()
    assert ALL_MASK_ATTENTION_FUNCTIONS[NAME] is flash_attention_mask


def _inputs(batch=1, heads=4, kv_heads=2, length=6, width=8, dtype=torch.float32):
    torch.manual_seed(0)
    return tuple(torch.randn(batch, h, length, width, dtype=dtype) for h in (heads, kv_heads, kv_heads))


@pytest.mark.parametrize("kwargs, message", [
    ({"softcap": 30.0}, "soft-caps"),
    ({"s_aux": torch.zeros(4)}, "attention sinks"),
    ({"cache": object()}, "paged cache"),
    ({"sliding_window": 4, "is_causal": False}, "needs a causal mask"),
])
def test_what_the_kernel_cannot_do_is_refused(kwargs, message):
    module = types.SimpleNamespace(is_causal=True)
    with pytest.raises(ValueError, match=message):
        _varlen.attention_forward(module, *_inputs(), None, **kwargs)


def test_a_4d_mask_is_refused():
    module = types.SimpleNamespace(is_causal=True)
    with pytest.raises(ValueError, match="4D attention mask"):
        _varlen.attention_forward(module, *_inputs(), torch.ones(1, 1, 6, 6, dtype=torch.bool))


def test_inputs_the_kernel_does_not_compute_are_refused():
    q, k, v = _inputs(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA GPU"):
        _varlen.check_inputs(q, k, v)
    with pytest.raises(ValueError, match="multiple of 8"):
        _varlen.check_inputs(*_inputs(width=12, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="CUDA GPU"):  # float32 heads align to 4
        _varlen.check_inputs(*_inputs(width=12))
    with pytest.raises(ValueError, match="one dtype"):
        _varlen.check_inputs(q, k.float(), v)
    with pytest.raises(ValueError, match="float16, bfloat16 or float32"):
        _varlen.check_inputs(*_inputs(dtype=torch.float64))
    with pytest.raises(ValueError, match="do not share"):
        _varlen.check_inputs(*_inputs(heads=3, kv_heads=2))
    with pytest.raises(ValueError, match="keys of 16"):
        _varlen.check_inputs(q, torch.randn(1, 2, 6, 16, dtype=torch.bfloat16), v)


def test_repeated_heads_get_their_gradients_summed():
    key = torch.randn(1, 5, 2, 8, requires_grad=True)
    expanded = _varlen._expand(key, 3)
    assert expanded.shape == (1, 5, 6, 8)
    grad = torch.randn_like(expanded)
    expanded.backward(grad)
    assert torch.allclose(key.grad, _varlen._fold(grad, 3))


def _allowed(query_length, key_length, mask_type, window, device="cpu"):
    """Which keys each query sees: as Transformers' causal and sliding-window masks say."""
    shift = key_length - query_length if mask_type == _varlen.CAUSAL_BOTTOM_RIGHT else 0
    i = torch.arange(query_length, device=device)[:, None] + shift
    j = torch.arange(key_length, device=device)[None]
    allowed = (torch.ones(query_length, key_length, dtype=torch.bool, device=device) if mask_type == _varlen.NO_MASK
               else j <= i)
    if window is not None:
        allowed = allowed & (j > i - window)
    return allowed


def _kernel_stand_in(query, key, value, cu_q, cu_k, max_q, max_k, dropout, mask_type, scale, window):
    """What the kernel computes, in PyTorch on the CPU: each sequence on its own.

    Inputs are (batch, length, heads, width), as the kernel takes them.
    """
    group = query.shape[2] // key.shape[2]
    key, value = _varlen._expand(key, group), _varlen._expand(value, group)
    if cu_q is None:
        pieces = [(b, 0, query.shape[1], 0, key.shape[1]) for b in range(query.shape[0])]
    else:
        pieces = [(0, cu_q[i].item(), cu_q[i + 1].item(), cu_k[i].item(), cu_k[i + 1].item())
                  for i in range(len(cu_q) - 1)]
    rows = []
    for b, q0, q1, k0, k1 in pieces:
        q, k, v = (x[b, a:z].transpose(0, 1) for x, a, z in ((query, q0, q1), (key, k0, k1), (value, k0, k1)))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_allowed(q1 - q0, k1 - k0, mask_type, window),
                                             scale=scale)
        rows.append((b, out.transpose(0, 1)))
    if cu_q is None:
        return torch.stack([out for _, out in rows])
    return torch.cat([out for _, out in rows])[None]


def _alone(q, k, v, window=None, scale=None):
    """One sequence, (heads, length, width), with its key heads repeated: the reference."""
    group = q.shape[0] // k.shape[0]
    k, v = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
    allowed = _allowed(q.shape[1], k.shape[1], _varlen.CAUSAL_TOP_LEFT, window, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scale).transpose(0, 1)


@pytest.fixture
def stand_in(monkeypatch):
    """The kernel and its CUDA check replaced, so the attention's bookkeeping runs on the CPU."""
    monkeypatch.setattr(_varlen.EfficientAttention, "apply", _kernel_stand_in)
    monkeypatch.setattr(_varlen, "check_inputs", lambda *inputs: None)


@pytest.mark.parametrize("window", [None, 2])
def test_a_packed_row_keeps_its_examples_apart(stand_in, window):
    module = types.SimpleNamespace(is_causal=True)
    q, k, v = _inputs(length=9)
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    out, weights = _varlen.attention_forward(module, q, k, v, None, scaling=0.5, sliding_window=window,
                                             position_ids=position_ids)
    assert weights is None and out.shape == (1, 9, 4, 8)
    for a, z in ((0, 3), (3, 5), (5, 9)):
        expected = _alone(q[0, :, a:z], k[0, :, a:z], v[0, :, a:z], window, scale=0.5)
        assert torch.allclose(out[0, a:z], expected, atol=1e-6)


@pytest.mark.parametrize("side", ["right", "left"])
def test_a_padded_batch_is_unpadded_and_put_back(stand_in, side):
    module = types.SimpleNamespace(is_causal=True)
    q, k, v = _inputs(batch=2, length=5)
    lengths = [5, 3]
    mask = torch.zeros(2, 5, dtype=torch.bool)
    for row, n in enumerate(lengths):
        if side == "right":
            mask[row, :n] = True
        else:
            mask[row, 5 - n:] = True
    out, _ = _varlen.attention_forward(module, q, k, v, mask)
    assert out.shape == (2, 5, 4, 8)
    for row in range(2):
        kept = mask[row]
        expected = _alone(q[row][:, kept], k[row][:, kept], v[row][:, kept])
        assert torch.allclose(out[row][kept], expected, atol=1e-6)
        assert (out[row][~kept] == 0).all()


def test_rows_without_padding_are_each_one_sequence(stand_in):
    module = types.SimpleNamespace(is_causal=True)
    q, k, v = _inputs(batch=2, length=4)
    out, _ = _varlen.attention_forward(module, q, k, v, None, position_ids=torch.arange(4)[None].expand(2, -1))
    for row in range(2):
        assert torch.allclose(out[row], _alone(q[row], k[row], v[row]), atol=1e-6)


def test_head_widths_by_kind_of_layer():
    assert max(models.head_widths(AutoConfig.for_model("gemma4")).values()) == 512
    assert models.head_widths(PretrainedConfig(head_dim=256, global_head_dim=512)) == {"head_dim": 256,
                                                                                      "global_head_dim": 512}
    with pytest.raises(ValueError, match="512"):
        models.check_flash_attention_heads(AutoConfig.for_model("gemma4"), "flash_attention_2")
    models.check_flash_attention_heads(AutoConfig.for_model("gemma4"), NAME)


def test_the_attention_auto_mode_chooses(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(models, "_flash_attention_2_installed", lambda: True)
    assert models.choose_attention(LlamaConfig()) == ("flash_attention_2", "FlashAttention 2 with packed sequences")
    assert models.choose_attention(PretrainedConfig(head_dim=256, global_head_dim=512)) == (
        NAME, "heads of 512: FlashAttention 2 stops at 256; PyTorch's memory-efficient kernel with packed sequences")
    assert models.choose_attention(AutoConfig.for_model("gemma4"))[0] == NAME
    monkeypatch.setattr(models, "_flash_attention_2_installed", lambda: False)
    name, why = models.choose_attention(LlamaConfig())
    assert name == NAME and why.startswith("FlashAttention 2 is not installed")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert models.choose_attention(LlamaConfig())[0] == "sdpa"


def test_packed_batches_and_the_plan_text():
    assert models.padding_free_ok(LlamaConfig(), NAME)
    assert models.keeps_examples_apart(NAME) == ("PyTorch's memory-efficient kernel keeps the examples apart "
                                                 "(cu_seqlens)")
    assert models.keeps_examples_apart("flash_attention_2") == ("the attention (flash_attention_2) keeps the examples "
                                                                "apart")


def test_compiled_layers_leave_the_attention_outside(monkeypatch):
    assert NAME in _compile.OUTSIDE
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    wanted, why = _compile.decision(LlamaConfig(), NAME)
    assert wanted and why == "the attention runs between two compiled pieces of each layer"
    assert _compile.decision(LlamaConfig(), "flash_attention_2")[0] is False


def test_selective_layers_keep_the_kernels_output():
    assert torch.ops.aten._efficient_attention_forward.default in _selective.attention_ops()
    assert _selective.flash_attention_ops(NAME) == []


def _reference(q, k, v, cu, window, scale):
    """Each sequence alone, in float32, with PyTorch's math attention; inputs (1, total, heads, width)."""
    outs = []
    for a, z in zip(cu[:-1].tolist(), cu[1:].tolist()):
        qs, ks, vs = (x[0, a:z].transpose(0, 1).float() for x in (q, k, v))
        outs.append(_alone(qs, ks, vs, window, scale))
    return torch.cat(outs)[None]


@cuda
@pytest.mark.parametrize("width", [256, 512])
@pytest.mark.parametrize("window", [None, 512])
def test_the_kernel_matches_each_sequence_alone(width, window):
    torch.manual_seed(0)
    lengths = [700, 130, 1024, 9]
    total, heads, kv_heads = sum(lengths), 8, 2
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None].cuda()
    q, k, v = (torch.randn(1, h, total, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for h in (heads, kv_heads, kv_heads))
    grad = torch.randn(1, total, heads, width, device="cuda", dtype=torch.bfloat16)
    module = types.SimpleNamespace(is_causal=True)
    out, _ = _varlen.attention_forward(module, q, k, v, None, scaling=1.0, sliding_window=window,
                                       position_ids=position_ids)
    out.backward(grad)

    qr, kr, vr = (x.detach().float().transpose(1, 2).requires_grad_() for x in (q, k, v))
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist())
    expected = _reference(qr, kr, vr, cu, window, 1.0)
    expected.backward(grad.float())
    assert (out.float() - expected).abs().max().item() < 0.05
    for packed, alone in ((q, qr), (k, kr), (v, vr)):
        reference = alone.grad.transpose(1, 2)
        assert (packed.grad.float() - reference).abs().max().item() < 0.05 * max(1.0, reference.abs().max().item())


@cuda
def test_the_kernel_on_a_padded_batch_matches_each_row_alone():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, h, 64, 512, device="cuda", dtype=torch.bfloat16) for h in (8, 2, 2))
    mask = torch.ones(2, 64, dtype=torch.bool, device="cuda")
    mask[1, 40:] = False
    out, _ = _varlen.attention_forward(types.SimpleNamespace(is_causal=True), q, k, v, mask, scaling=1.0)
    for row, n in ((0, 64), (1, 40)):
        expected = _alone(q[row, :, :n].float(), k[row, :, :n].float(), v[row, :, :n].float(), scale=1.0)
        assert (out[row, :n].float() - expected).abs().max().item() < 0.05
    assert (out[1, 40:] == 0).all()
