"""Fused paged-MQA-logits kernel for the DSV4 C4 indexer.

The torch fallback (``_fp8_paged_mqa_logits_torch_chunk``) materializes the
gathered+dequantized compressed KV window per query-chunk, which forces
adaptive chunking and a workspace that grows with context length (at 200K
context the chunk collapses to 8 queries -> thousands of tiny launches).  This
kernel fuses the whole computation into one launch:

  score[b, i] = relu(sum_d kv[b,i,d] * q[b,d]) * weight[b] * scale[b,i]

The index cache stores one compressed token per page row (page_size=1):
  row = [128 fp8 value bytes][4 byte fp32 scale].
``q`` is used as raw fp8-as-bf16 (matches the fallback, which does
``q_fp8.to(bf16)`` without applying a scale).
"""
from __future__ import annotations

import triton
import triton.language as tl

_BLOCK_P = 64
_HEAD_DIM = 128

# fp8 -> fp32 decode LUT, cached per device. Rebuilding it per call (arange +
# view + to) injects data-generating ops into the graph-captured decode path
# and slows the step; the table is a fixed constant of the fp8 format.
_LUT_CACHE: dict = {}


def _get_fp8_lut(device: str, dtype) -> "torch.Tensor":
    import torch

    key = (device, str(dtype))
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = (
            torch.arange(256, dtype=torch.uint8, device=device)
            .view(dtype)
            .to(torch.float32)
        )
    return _LUT_CACHE[key]


@triton.jit
def _fp8_paged_mqa_logits_fused_kernel(
    q_fp8_ptr,
    kv_fp8_ptr,
    kv_f32_ptr,
    lut_ptr,
    page_table_ptr,
    weight_ptr,
    seq_lens_ptr,
    scores_ptr,
    batch,
    max_pages,
    max_seq_len,
    BLOCK_T: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_pb = tl.program_id(1)

    p_off = pid_pb * BLOCK_P + tl.arange(0, BLOCK_P)
    d_off = tl.arange(0, HEAD_DIM)
    total_tokens = max_pages * BLOCK_T
    p_mask = p_off < total_tokens

    # Query tile for this row: (NUM_HEADS, HEAD_DIM), raw fp8 cast to bf16
    # (matches the fallback, which does q_fp8.to(bf16) without a scale).
    h_off = tl.arange(0, NUM_HEADS)
    q = tl.load(
        q_fp8_ptr + pid_b * (NUM_HEADS * HEAD_DIM) + h_off[:, None] * HEAD_DIM + d_off[None, :]
    ).to(tl.bfloat16)  # (NUM_HEADS, HEAD_DIM)

    # Each page holds BLOCK_T tokens; page layout is separated:
    #   [BLOCK_T*HEAD_DIM fp8 bytes][BLOCK_T*4 scale bytes].
    page = p_off // BLOCK_T
    t_in = p_off % BLOCK_T
    page_idx = tl.load(
        page_table_ptr + pid_b * max_pages + page, mask=p_mask, other=0
    ).to(tl.int64)
    page_idx = tl.maximum(page_idx, 0)
    base = page_idx * (BLOCK_T * (HEAD_DIM + 4))

    # (BLOCK_P, 128) fp8 values + (BLOCK_P,) fp32 scales.
    # The fp8 bytes MUST be decoded (LUT) before the dot: using the raw uint8
    # byte as a bf16 "value" distorts the per-element magnitude non-uniformly
    # (byte 68 for 3.0 vs byte 76 for 6.0), which misranks the top-k pages as
    # more c4 tokens compete at long contexts — the agent tool-call regression.
    # Origin: 修复 2026-08-17 (SGLANG_FUSED_INDEXER=1 fp8 decode).
    kv_bytes = tl.load(
        kv_fp8_ptr + base[:, None] + t_in[:, None] * HEAD_DIM + d_off[None, :],
        mask=p_mask[:, None],
        other=0,
    ).to(tl.int32)  # (BLOCK_P, HEAD_DIM) fp8 bytes
    # Decode via LUT, keep bf16 for the original fast tensor-core dot path.
    kv = tl.load(lut_ptr + kv_bytes).to(tl.bfloat16)  # (BLOCK_P, HEAD_DIM) bf16
    scale = tl.load(
        kv_f32_ptr + (base // 4) + (BLOCK_T * HEAD_DIM) // 4 + t_in, mask=p_mask
    )  # fp32 per-token scale

    # dot = kv @ q^T -> (BLOCK_P, NUM_HEADS); relu + per-head weight, sum heads.
    # bf16 dot (original fast path) with an fp32 head-sum; the bf16 rounding of
    # the LUT-decoded values matches the torch fallback (which also rounds via
    # its bmm output dtype) and preserves the top-k ordering.
    dot = tl.dot(kv, tl.trans(q))  # (BLOCK_P, NUM_HEADS) fp32 accumulate
    dot = tl.maximum(dot.to(tl.bfloat16), 0.0)  # relu in bf16 (matches bmm bf16)
    w = tl.load(weight_ptr + pid_b * NUM_HEADS + h_off)  # (NUM_HEADS,) bf16
    score = tl.sum((dot * w[None, :]).to(tl.float32), axis=1) * scale  # (BLOCK_P,) fp32
    seq_len = tl.load(seq_lens_ptr + pid_b)
    valid = p_mask & (p_off < seq_len)

    tl.store(
        scores_ptr + pid_b * max_seq_len + p_off,
        tl.where(valid, score.to(scores_ptr.dtype.element_ty), 0.0),
        mask=p_mask,
    )


def fused_fp8_paged_mqa_logits(
    q_fp8: "torch.Tensor",
    kvcache_fp8: "torch.Tensor",
    weight: "torch.Tensor",
    seq_lens: "torch.Tensor",
    page_table: "torch.Tensor",
    max_seq_len: int,
    block_size: int = 64,
    head_dim: int = 128,
) -> "torch.Tensor":
    """One-launch replacement for ``_fp8_paged_mqa_logits_torch_chunk``.

    Args: same as the torch fallback dispatcher's chunk body.  ``q_fp8`` is
    (batch, 1, num_heads, head_dim), ``kvcache_fp8`` is a paged cache whose
    rows are ``block_size`` tokens of ``[head_dim fp8][4-byte fp32 scale]``
    each (the DSV4 index cache uses block_size=1, i.e. one token per row).
    ``page_table`` is (batch, max_pages).
    """
    import torch

    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

    batch_size = q_fp8.shape[0]
    max_pages = page_table.shape[1]

    num_heads = q_fp8.shape[2]
    q_flat = q_fp8[:, 0].reshape(batch_size, num_heads, head_dim).contiguous()
    kvcache_flat = kvcache_fp8.reshape(-1)
    # fp32 view for the 4-byte per-token scales.
    kvcache_f32 = kvcache_fp8.view(torch.float32).reshape(-1)
    # fp8 -> fp32 decode LUT (256 entries). The fused kernel looks up the
    # decoded value per element instead of using the raw uint8 byte.
    fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    lut = _get_fp8_lut(q_fp8.device, fp8_dtype)
    scores = torch.zeros((batch_size, max_seq_len), dtype=torch.float32, device=q_fp8.device)
    total_tokens = max_pages * block_size
    grid = (batch_size, triton.cdiv(total_tokens, _BLOCK_P))
    _fp8_paged_mqa_logits_fused_kernel[grid](
        q_flat,
        kvcache_flat,
        kvcache_f32,
        lut,
        page_table,
        weight,
        seq_lens,
        scores,
        batch_size,
        max_pages,
        max_seq_len,
        BLOCK_T=block_size,
        BLOCK_P=_BLOCK_P,
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        num_warps=4,
        num_stages=1,
    )
    return scores
