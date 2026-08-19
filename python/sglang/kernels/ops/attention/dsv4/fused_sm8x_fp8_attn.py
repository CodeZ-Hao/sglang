"""Fused SM8x SWA attention over the native packed DSV4 FP8 KV cache.

The eager SM8x path dequantizes the selected SWA (+ c4/c128) window into a
bf16 workspace (``dequantize_k_cache_paged``) and then runs
``fused_sm8x_attn`` over it.  For a large prefill chunk the window workspace
forces 512-query chunking, so a 8192-token chunk becomes ~16 x (dequant +
attention) launches per layer.

This module fuses the dequant INTO the flash kernel: each k-tile is decoded
from the packed bytes (448 fp8 nope + 64 bf16 rope + 7 ue8m0 scales per
token) in registers, so there is no materialized bf16 window and no chunking.
One launch handles all queries for a layer.

Register pressure forces the 512-dim dot to run as two 256-dim halves
(``accA``/``accB``); the dequant math mirrors ``dequantize_k_cache_paged``.
"""
from __future__ import annotations

import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE as _DIM_NOPE,
    DIM_ROPE as _DIM_ROPE,
    NOPE_ROPE_BYTES as _NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN as _PADDED_SCALE_PER_TOKEN,
)

# Triton kernels may only read globals that are tl.constexpr.
DIM_NOPE = tl.constexpr(_DIM_NOPE)            # 448
DIM_ROPE = tl.constexpr(_DIM_ROPE)            # 64
NOPE_ROPE_BYTES = tl.constexpr(_NOPE_ROPE_BYTES)  # 576
PADDED_SCALE_PER_TOKEN = tl.constexpr(_PADDED_SCALE_PER_TOKEN)  # 8
DIM_HALF = tl.constexpr((_DIM_NOPE + _DIM_ROPE) // 2)  # 256

_BLOCK_H = 16
_BLOCK_K = 32


@triton.jit
def _fused_sm8x_fp8_attn_kernel(
    q_ptr,
    swa_cache_fp8_ptr,
    swa_cache_bf16_ptr,
    swa_cache_u8_ptr,
    swa_indices_ptr,
    swa_topk_len_ptr,
    extra_cache_fp8_ptr,
    extra_cache_bf16_ptr,
    extra_cache_u8_ptr,
    extra_indices_ptr,
    extra_topk_len_ptr,
    sink_ptr,
    out_ptr,
    num_q,
    swa_topk,
    extra_topk,
    softmax_scale,
    SWA_PAGE_SIZE: tl.constexpr,
    SWA_BYTES_PER_PAGE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    EXTRA_BYTES_PER_PAGE: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_hg = tl.program_id(1)
    h_off = pid_hg * BLOCK_H + tl.arange(0, BLOCK_H)

    d_off = tl.arange(0, DIM_HALF)
    qA = tl.load(
        q_ptr + pid_q * (BLOCK_H * tl.num_programs(1) * (DIM_NOPE + DIM_ROPE))
        + h_off[:, None] * (DIM_NOPE + DIM_ROPE)
        + d_off[None, :]
    )  # (BLOCK_H, 256) bf16
    qB = tl.load(
        q_ptr + pid_q * (BLOCK_H * tl.num_programs(1) * (DIM_NOPE + DIM_ROPE))
        + h_off[:, None] * (DIM_NOPE + DIM_ROPE)
        + (d_off[None, :] + DIM_HALF)
    )

    NEG = -1.0e30
    m_i = tl.full((BLOCK_H,), NEG, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    accA = tl.zeros((BLOCK_H, DIM_HALF), tl.float32)
    accB = tl.zeros((BLOCK_H, DIM_HALF), tl.float32)

    k_off = tl.arange(0, BLOCK_K)

    # ---- SWA window ----
    topk_len = tl.load(swa_topk_len_ptr + pid_q)
    n_swa_blocks = (swa_topk + BLOCK_K - 1) // BLOCK_K
    for kb in range(n_swa_blocks):
        kk = kb * BLOCK_K + k_off
        k_mask = kk < swa_topk
        loc = tl.load(swa_indices_ptr + pid_q * swa_topk + kk, mask=k_mask, other=-1).to(
            tl.int64
        )
        valid = (loc >= 0) & (kk < topk_len)
        loc_safe = tl.where(valid, loc, 0)
        page_idx = loc_safe // SWA_PAGE_SIZE
        in_page = loc_safe % SWA_PAGE_SIZE
        tdata = page_idx * SWA_BYTES_PER_PAGE + in_page * NOPE_ROPE_BYTES
        tscale = page_idx * SWA_BYTES_PER_PAGE + SWA_PAGE_SIZE * NOPE_ROPE_BYTES + in_page * PADDED_SCALE_PER_TOKEN

        m = k_mask[:, None] & valid[:, None]

        # half A: fp8 nope dims 0..255 (tiles 0..3)
        fp8A = tl.load(
            swa_cache_fp8_ptr + tdata[:, None] + d_off[None, :],
            mask=m, other=0.0,
        ).to(tl.float32)
        scA = tl.load(
            swa_cache_u8_ptr + tscale[:, None] + (d_off[None, :] // 64),
            mask=m, other=127,
        ).to(tl.int32)
        kvA = (fp8A * tl.exp2((scA - 127).to(tl.float32))).to(tl.bfloat16)
        sA = tl.dot(qA, tl.trans(kvA))

        # half B: fp8 nope dims 256..447 (tiles 4..6) + bf16 rope 448..511
        nope_b = d_off < (DIM_NOPE - DIM_HALF)  # d_off < 192
        fp8B = tl.load(
            swa_cache_fp8_ptr + (tdata + DIM_HALF)[:, None] + d_off[None, :],
            mask=m & nope_b[None, :], other=0.0,
        ).to(tl.float32)
        scB = tl.load(
            swa_cache_u8_ptr + tscale[:, None] + ((d_off[None, :] + DIM_HALF) // 64),
            mask=m & nope_b[None, :], other=127,
        ).to(tl.int32)
        kvB = (fp8B * tl.exp2((scB - 127).to(tl.float32))).to(tl.bfloat16)
        rope = tl.load(
            swa_cache_bf16_ptr + ((tdata + DIM_NOPE) // 2)[:, None] + (d_off[None, :] - (DIM_NOPE - DIM_HALF)),
            mask=m & ~nope_b[None, :], other=0.0,
        )
        kvB = kvB + rope
        sB = tl.dot(qB, tl.trans(kvB))

        s = (sA + sB) * softmax_scale
        s = tl.where(valid[None, :], s, NEG)
        m_new = tl.maximum(m_i[:, None], s)
        m_new = tl.max(m_new, axis=1)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
        pbf16 = p.to(tl.bfloat16)
        accA = accA * alpha[:, None] + tl.dot(pbf16, kvA)
        accB = accB * alpha[:, None] + tl.dot(pbf16, kvB)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # ---- optional compressed window (c4/c128) ----
    if HAS_EXTRA:
        e_topk_len = tl.load(extra_topk_len_ptr + pid_q)
        n_extra_blocks = (extra_topk + BLOCK_K - 1) // BLOCK_K
        for kb in range(n_extra_blocks):
            kk = kb * BLOCK_K + k_off
            k_mask = kk < extra_topk
            loc = tl.load(extra_indices_ptr + pid_q * extra_topk + kk, mask=k_mask, other=-1).to(
                tl.int64
            )
            valid = (loc >= 0) & (kk < e_topk_len)
            loc_safe = tl.where(valid, loc, 0)
            page_idx = loc_safe // EXTRA_PAGE_SIZE
            in_page = loc_safe % EXTRA_PAGE_SIZE
            tdata = page_idx * EXTRA_BYTES_PER_PAGE + in_page * NOPE_ROPE_BYTES
            tscale = page_idx * EXTRA_BYTES_PER_PAGE + EXTRA_PAGE_SIZE * NOPE_ROPE_BYTES + in_page * PADDED_SCALE_PER_TOKEN

            m = k_mask[:, None] & valid[:, None]
            fp8A = tl.load(
                extra_cache_fp8_ptr + tdata[:, None] + d_off[None, :],
                mask=m, other=0.0,
            ).to(tl.float32)
            scA = tl.load(
                extra_cache_u8_ptr + tscale[:, None] + (d_off[None, :] // 64),
                mask=m, other=127,
            ).to(tl.int32)
            kvA = (fp8A * tl.exp2((scA - 127).to(tl.float32))).to(tl.bfloat16)
            sA = tl.dot(qA, tl.trans(kvA))

            nope_b = d_off < (DIM_NOPE - DIM_HALF)
            fp8B = tl.load(
                extra_cache_fp8_ptr + (tdata + DIM_HALF)[:, None] + d_off[None, :],
                mask=m & nope_b[None, :], other=0.0,
            ).to(tl.float32)
            scB = tl.load(
                extra_cache_u8_ptr + tscale[:, None] + ((d_off[None, :] + DIM_HALF) // 64),
                mask=m & nope_b[None, :], other=127,
            ).to(tl.int32)
            kvB = (fp8B * tl.exp2((scB - 127).to(tl.float32))).to(tl.bfloat16)
            rope = tl.load(
                extra_cache_bf16_ptr + ((tdata + DIM_NOPE) // 2)[:, None] + (d_off[None, :] - (DIM_NOPE - DIM_HALF)),
                mask=m & ~nope_b[None, :], other=0.0,
            )
            kvB = kvB + rope
            sB = tl.dot(qB, tl.trans(kvB))

            s = (sA + sB) * softmax_scale
            s = tl.where(valid[None, :], s, NEG)
            m_new = tl.maximum(m_i[:, None], s)
            m_new = tl.max(m_new, axis=1)
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
            pbf16 = p.to(tl.bfloat16)
            accA = accA * alpha[:, None] + tl.dot(pbf16, kvA)
            accB = accB * alpha[:, None] + tl.dot(pbf16, kvB)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    # ---- sink column ----
    sink = tl.load(sink_ptr + h_off)  # (BLOCK_H,) fp32
    m_new = tl.maximum(m_i, sink)
    alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
    accA = accA * alpha[:, None]
    accB = accB * alpha[:, None]
    l_i = l_i * alpha + tl.exp2((sink - m_new) * 1.4426950408889634)

    outA = accA / l_i[:, None]
    outB = accB / l_i[:, None]
    q_base = pid_q * (BLOCK_H * tl.num_programs(1) * (DIM_NOPE + DIM_ROPE)) + h_off[:, None] * (DIM_NOPE + DIM_ROPE)
    tl.store(out_ptr + q_base + d_off[None, :], outA)
    tl.store(out_ptr + q_base + (d_off[None, :] + DIM_HALF), outB)


def fused_sm8x_fp8_attn(
    q: "torch.Tensor",
    swa_cache: "torch.Tensor",
    swa_indices: "torch.Tensor",
    swa_topk_lengths: "torch.Tensor",
    swa_page_size: int,
    attn_sink: "torch.Tensor",
    softmax_scale: float,
    extra_cache=None,
    extra_indices=None,
    extra_topk_lengths=None,
    extra_page_size: int = 0,
) -> "torch.Tensor":
    """One-launch fused FP8 SWA (+ c4/c128) attention.

    Args:
        q: (num_q, num_heads, 512) bf16.
        swa_cache: (num_pages, swa_bytes_per_page) uint8 packed cache.
        swa_indices: (num_q, swa_topk) int32 token locs.
        swa_topk_lengths: (num_q,) int32.
        swa_page_size: tokens per SWA page (pool swa_window_size).
        attn_sink: (num_heads,) fp32.
        extra_*: optional c4/c128 compressed window inputs.
    """
    import torch

    num_q, num_heads, head_dim = q.shape
    assert head_dim == DIM_NOPE + DIM_ROPE
    assert num_heads % _BLOCK_H == 0

    has_extra = extra_cache is not None and extra_indices is not None
    if has_extra:
        extra_topk = extra_indices.shape[1]
    else:
        extra_topk = 0

    swa_u8 = swa_cache.contiguous().view(torch.uint8)
    swa_bytes_per_page = swa_u8.shape[-1]
    fp8_dtype = swa_u8.view(torch.float8_e4m3fn).dtype
    swa_fp8 = swa_u8.view(fp8_dtype).reshape(-1)
    swa_bf16 = swa_u8.view(torch.bfloat16).reshape(-1)
    swa_u8f = swa_u8.reshape(-1)

    if has_extra:
        extra_u8 = extra_cache.contiguous().view(torch.uint8)
        extra_bytes_per_page = extra_u8.numel() // extra_u8.shape[0]
        extra_fp8 = extra_u8.view(fp8_dtype).reshape(-1)
        extra_bf16 = extra_u8.view(torch.bfloat16).reshape(-1)
        extra_u8f = extra_u8.reshape(-1)
    else:
        extra_fp8 = extra_bf16 = extra_u8f = swa_fp8
        extra_bytes_per_page = swa_bytes_per_page

    swa_topk = swa_indices.shape[1]
    num_head_groups = num_heads // _BLOCK_H
    out = torch.empty_like(q)
    grid = (num_q, num_head_groups)
    _fused_sm8x_fp8_attn_kernel[grid](
        q,
        swa_fp8, swa_bf16, swa_u8f,
        swa_indices,
        swa_topk_lengths,
        extra_fp8, extra_bf16, extra_u8f,
        extra_indices if has_extra else swa_indices,
        extra_topk_lengths if has_extra else swa_topk_lengths,
        attn_sink,
        out,
        num_q,
        swa_topk,
        extra_topk,
        softmax_scale,
        SWA_PAGE_SIZE=swa_page_size,
        SWA_BYTES_PER_PAGE=swa_bytes_per_page,
        EXTRA_PAGE_SIZE=extra_page_size if has_extra else swa_page_size,
        EXTRA_BYTES_PER_PAGE=extra_bytes_per_page,
        HAS_EXTRA=has_extra,
        BLOCK_H=_BLOCK_H,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=1,
    )
    return out
