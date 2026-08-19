"""SM89 lossless MXFP4->FP8 batch dequant for the layerwise GPU-MoE path.

DeepSeek-V4 routed-expert weights are E2M1 FP4 (two nibbles per int8 byte)
plus a per-32-in-dim e8m0fnu power-of-2 scale.  E2M1 is a subset of E4M3, so
dequantizing to FP8 is lossless.  Layout per expert:
  w13: (2048 out, 2048 packed int8) -> fp8 (2048, 4096)
  w13 scale: (2048, 128) e8m0 raw bytes
  w2:  (4096 out, 1024 packed int8) -> fp8 (4096, 2048)
  w2 scale: (4096, 64) e8m0 raw bytes
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mxfp4_to_fp8_kernel(
    w_ptr, s_ptr, out_ptr, N, BLOCK_K: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_k = tl.program_id(1)
    row = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    col = pid_c * 64 + tl.arange(0, 64)
    col_mask = col < N
    byte = tl.load(
        w_ptr + row[:, None] * (N // 2) + (col[None, :] // 2),
        mask=col_mask[None, :], other=0,
    )
    lo = byte & 0x0F
    hi = (byte >> 4) & 0x0F
    n4 = tl.where((col[None, :] % 2) == 0, lo, hi)
    sign = tl.where((n4 & 0x8) != 0, -1.0, 1.0)
    e = ((n4 & 0x6) >> 1).to(tl.float32)
    m = (n4 & 0x1).to(tl.float32)
    is_sub = e == 0.0
    expo = tl.where(is_sub, 0.0, e - 1.0)  # E2M1 subnormal = 0.M * 2^0
    mant = tl.where(is_sub, m * 0.5, 1.0 + m * 0.5)
    val = sign * mant * tl.exp2(expo)
    sc = tl.load(
        s_ptr + row[:, None] * (N // 32) + (col[None, :] // 32),
        mask=col_mask[None, :], other=127,
    ).to(tl.int32)
    scale = tl.exp2((sc - 127.0).to(tl.float32))
    out = (val * scale).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row[:, None] * N + col[None, :], out, mask=col_mask[None, :])


def mxfp4_to_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(K, Np) packed-int8 MXFP4 + (K, N/32) e8m0 -> (K, N) fp8, lossless."""
    K, Np = weight.shape
    N = Np * 2
    if weight.device.type != "cuda":
        weight = weight.cuda()
    if scale.device.type != "cuda":
        scale = scale.cuda()
    out = torch.empty(K, N, dtype=torch.float8_e4m3fn, device=weight.device)
    grid = (triton.cdiv(N, 64), K // 32)
    _mxfp4_to_fp8_kernel[grid](weight, scale.view(torch.uint8), out, N, BLOCK_K=32)
    return out


def mxfp4_batch_to_fp8(weight: torch.Tensor, scale: torch.Tensor, n: int) -> torch.Tensor:
    """Dequant a batch of experts (E, K, Np) + (E, K, N/32) -> (E, K, N) fp8."""
    E = weight.shape[0]
    K, Np = weight.shape[1], weight.shape[2]
    N = Np * 2
    out = torch.empty(E, K, N, dtype=torch.float8_e4m3fn, device=weight.device)
    for e in range(E):
        out[e] = mxfp4_to_fp8(weight[e], scale[e])
    return out
