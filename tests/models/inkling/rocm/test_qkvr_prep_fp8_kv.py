# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the fp8 KV-cache write-path dtype mismatch.

Prior to this fix, ``_qkvr_qkv_kernel``/``_kv_kernel`` stored the model's
bf16-dtype K/V values directly into a ``torch.uint8``-typed (fp8-storage)
cache buffer with a plain ``tl.store``, which numerically truncates instead
of quantizing -- corrupting every K/V cache entry under
``--kv-cache-dtype=fp8`` and triggering ``HSA_STATUS_ERROR_MEMORY_FAULT``
once garbage bytes are later bit-cast back to e4m3 and dereferenced deep in
the model's MoE expert GEMM (PR #64139, bead vl-see9). This test exercises
only the Triton write kernels in isolation (fast, no full model/engine
needed) and checks that an fp8-typed cache round-trips through a
``float8_e4m3fn`` bitcast-dequant within fp8 precision, rather than landing
on an unrelated numeric value.
"""

import pytest
import torch

from vllm.models.inkling.amd.ops import qkvr_prep


def _make_inputs(*, tokens: int, use_fp8_kv: bool):
    torch.manual_seed(0)
    heads = 16
    kv_heads = 4
    head_dim = 128
    d_rel = 16
    rel_extent = 512
    page_size = 16
    num_blocks = (tokens + page_size - 1) // page_size
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    r_width = heads * d_rel
    device = "cuda"

    qkvr = torch.randn(
        tokens,
        q_width + 2 * kv_width + r_width,
        device=device,
        dtype=torch.bfloat16,
    )
    k_weight = torch.randn(kv_width, 4, device=device, dtype=torch.bfloat16)
    v_weight = torch.randn_like(k_weight)
    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    k_norm_weight = torch.randn_like(q_norm_weight)
    rel_proj = torch.randn(d_rel, rel_extent, device=device, dtype=torch.bfloat16)
    conv_cache = torch.zeros(
        num_blocks,
        kv_heads,
        page_size,
        2 * head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_dtype = torch.uint8 if use_fp8_kv else torch.bfloat16
    key_cache = torch.zeros(
        num_blocks, page_size, kv_heads, head_dim, device=device, dtype=cache_dtype
    )
    value_cache = torch.zeros_like(key_cache)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    seq_idx = torch.zeros(tokens, device=device, dtype=torch.int32)
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    query_start = torch.zeros(tokens, device=device, dtype=torch.int32)
    return (
        qkvr,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        rel_proj,
        1e-6,
        heads,
        kv_heads,
        head_dim,
        d_rel,
        conv_cache,
        key_cache,
        value_cache,
        positions,
        block_table,
        seq_idx,
        slots,
        query_start,
        slots,
        0,
        head_dim,
        page_size,
        None,
    )


@pytest.mark.parametrize("tokens", [9, 128])
@torch.inference_mode()
def test_fp8_kv_cache_write_round_trips_through_e4m3(tokens):
    """The write kernels must bitcast-quantize into ``torch.uint8``-typed
    fp8 caches, matching the already-correct read-side bitcast-dequant in
    ``fa4_rel_attention._cast_kv_tile``. Below ``SMALL_TOKEN_THRESHOLD``
    this exercises ``_qkvr_qkv_kernel``; at/above it, ``_kv_kernel``."""
    args = _make_inputs(tokens=tokens, use_fp8_kv=True)
    qkvr_prep.fused_qkvr_prep(*args)

    key_cache, value_cache = args[12], args[13]
    assert key_cache.dtype == torch.uint8
    assert value_cache.dtype == torch.uint8

    # A valid e4m3 bit pattern dequantizes to a finite value; the pre-fix
    # numeric-truncation bug produced byte values with no such guarantee
    # relative to the bf16 input, since truncation discards the exponent.
    key_fp8 = key_cache.view(torch.float8_e4m3fn).float()
    value_fp8 = value_cache.view(torch.float8_e4m3fn).float()
    assert torch.isfinite(key_fp8).all()
    assert torch.isfinite(value_fp8).all()

    # Compare against the same cache written in bf16 (ground truth,
    # unaffected by the fp8 write path), accounting for e4m3's coarse
    # ~2-3 bit mantissa. _make_inputs reseeds identically, so this produces
    # the same qkvr/weights as the fp8 run above.
    ref_args = _make_inputs(tokens=tokens, use_fp8_kv=False)
    torch.testing.assert_close(ref_args[0], args[0], rtol=0, atol=0)
    qkvr_prep.fused_qkvr_prep(*ref_args)
    ref_key, ref_value = ref_args[12], ref_args[13]

    torch.testing.assert_close(key_fp8, ref_key.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(value_fp8, ref_value.float(), rtol=6e-2, atol=6e-2)


@torch.inference_mode()
def test_bf16_kv_cache_write_unaffected_by_fp8_gate(tokens: int = 9):
    """The ``IS_FP8_KV`` branch must be a no-op for the existing bf16
    (``--kv-cache-dtype=auto``) path -- confirms the fix does not regress
    the already-supported non-fp8 KV cache."""
    args = _make_inputs(tokens=tokens, use_fp8_kv=False)
    key_before = args[12].clone()
    value_before = args[13].clone()

    qkvr_prep.fused_qkvr_prep(*args)

    assert args[12].dtype == torch.bfloat16
    assert not torch.equal(args[12], key_before)
    assert not torch.equal(args[13], value_before)
