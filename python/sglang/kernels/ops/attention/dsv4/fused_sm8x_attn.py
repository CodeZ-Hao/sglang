"""Fused SM8x SWA attention kernel.

The eager torch fallback (``_forward_dspark_sm8x_swa``) dequantizes the packed
DSV4 KV window and then runs ~10 small torch ops (matmul, mask, cat, softmax
with sink, output matmul). Each op is a separate kernel launch; on the eager
prefill path that Python/launch overhead dominates. This module replaces the
post-dequant part with a single flash-attention-style Triton kernel:

  given q (num_q, num_heads, head_dim), dequantized kv (num_q, topk, head_dim),
  a validity mask (num_q, topk) and a per-head sink score:

    scores = q @ kv^T * softmax_scale,  masked to -inf where invalid
    probs  = softmax([scores, sink], dim=-1)[..., :-1]
    out    = probs @ kv

The KV is shared across query heads (MQA). DSV4 has 64 query heads; one program
handles a BLOCK_H-head slice of one query so the shared-memory footprint (q/kv
bf16 dot operands + the fp32 accumulator) stays under the SM89 limit. The grid
is (num_q, num_head_groups); the kv tiles are reloaded per head group.
"""
from __future__ import annotations

import triton
import triton.language as tl

# 512 = DIM_NOPE(448) + DIM_ROPE(64); every DSV4 K/V entry has this width.
SM8X_HEAD_DIM = 512

_BLOCK_H = 16
_BLOCK_K = 32
_BLOCK_D = 512


@triton.jit
def _fused_sm8x_attn_kernel(
    q_ptr,
    kv_ptr,
    valid_ptr,
    sink_ptr,
    out_ptr,
    num_q,
    topk,
    softmax_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_hg = tl.program_id(1)
    h_off = pid_hg * BLOCK_H + tl.arange(0, BLOCK_H)
    d_off = tl.arange(0, BLOCK_D)

    # q slice for this head group: (BLOCK_H, BLOCK_D) bf16.
    q = tl.load(
        q_ptr
        + pid_q * (BLOCK_D * tl.num_programs(1) * BLOCK_H)
        + h_off[:, None] * BLOCK_D
        + d_off[None, :]
    )

    NEG = -1.0e30

    m_i = tl.full((BLOCK_H,), NEG, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    n_k_blocks = (topk + BLOCK_K - 1) // BLOCK_K
    for kb in range(n_k_blocks):
        k_off = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_off < topk
        kv = tl.load(
            kv_ptr + pid_q * (topk * BLOCK_D) + k_off[:, None] * BLOCK_D + d_off[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        valid = tl.load(
            valid_ptr + pid_q * topk + k_off,
            mask=k_mask,
            other=0,
        )
        s = tl.dot(q, tl.trans(kv)) * softmax_scale
        s = tl.where(valid[None, :], s, NEG)

        m_new = tl.maximum(m_i[:, None], s)
        m_new = tl.max(m_new, axis=1)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Sink column: a per-head scalar score, then dropped from the output.
    sink = tl.load(sink_ptr + h_off)  # (BLOCK_H,)
    m_new = tl.maximum(m_i, sink)
    alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + tl.exp2((sink - m_new) * 1.4426950408889634)

    out = acc / l_i[:, None]
    tl.store(
        out_ptr
        + pid_q * (BLOCK_D * tl.num_programs(1) * BLOCK_H)
        + h_off[:, None] * BLOCK_D
        + d_off[None, :],
        out,
    )


@triton.jit
def _sm8x_indices_prep_kernel(
    indices_ptr,
    topk_length_ptr,
    valid_ptr,
    safe_ptr,
    topk,
    BLOCK_K: tl.constexpr,
):
    """Compute the SWA/c4/c128 validity mask and safe (clamped) indices in one
    launch, replacing the eager arange/compare/where ops per fallback call."""
    row = tl.program_id(0)
    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < topk
    idx = tl.load(indices_ptr + row * topk + k_off, mask=k_mask, other=-1)
    topk_len = tl.load(topk_length_ptr + row)
    valid = (idx >= 0) & (k_off < topk_len)
    safe = tl.where(valid, idx, 0)
    tl.store(valid_ptr + row * topk + k_off, valid, mask=k_mask)
    tl.store(safe_ptr + row * topk + k_off, safe, mask=k_mask)


def sm8x_indices_prep(
    indices: "torch.Tensor",
    topk_length: "torch.Tensor",
) -> tuple:
    """One-launch valid mask + safe indices for the SM8x fallback windows.

    Args:
        indices: (num_q, topk) int32 raw page/token locations.
        topk_length: (num_q,) int32 per-query valid window length.

    Returns:
        (valid, safe) — (num_q, topk) bool and (num_q, topk) int32.
    """
    import torch

    num_q, topk = indices.shape
    valid = torch.empty((num_q, topk), dtype=torch.bool, device=indices.device)
    safe = torch.empty_like(indices)
    _sm8x_indices_prep_kernel[(num_q,)](
        indices,
        topk_length,
        valid,
        safe,
        topk,
        BLOCK_K=triton.next_power_of_2(topk),
        num_warps=4,
    )
    return valid, safe


def fused_sm8x_attn(
    q: "torch.Tensor",
    kv: "torch.Tensor",
    valid: "torch.Tensor",
    sink: "torch.Tensor",
    softmax_scale: float,
) -> "torch.Tensor":
    """Fused post-dequant SM8x attention.

    Args:
        q: (num_q, num_heads, SM8X_HEAD_DIM) bf16.
        kv: (num_q, topk, SM8X_HEAD_DIM) bf16 (already dequantized).
        valid: (num_q, topk) bool; False entries are masked to -inf.
        sink: (num_heads,) bf16/float per-head sink score.
        softmax_scale: float.

    Returns:
        (num_q, num_heads, SM8X_HEAD_DIM) bf16.
    """
    import torch

    num_q, num_heads, head_dim = q.shape
    topk = kv.shape[1]
    assert head_dim == SM8X_HEAD_DIM
    assert kv.shape[0] == num_q and kv.shape[2] == SM8X_HEAD_DIM
    assert valid.shape == (num_q, topk)
    assert num_heads % _BLOCK_H == 0, (num_heads, _BLOCK_H)

    num_head_groups = num_heads // _BLOCK_H
    out = torch.empty_like(q)
    grid = (num_q, num_head_groups)
    _fused_sm8x_attn_kernel[grid](
        q,
        kv,
        valid,
        sink,
        out,
        num_q,
        topk,
        softmax_scale,
        BLOCK_H=_BLOCK_H,
        BLOCK_K=_BLOCK_K,
        BLOCK_D=_BLOCK_D,
        num_warps=2,
        num_stages=1,
    )
    return out
