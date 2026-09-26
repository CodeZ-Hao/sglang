"""Grouped BF16 WO-A projections for decode: a single-token GEMV, a split-K
small-batch GEMM, and the small-batch variant with a fused MXFP8 epilogue."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mxfp8_epilogue import ue8m0_scale


@triton.jit
def _wo_a_bf16_gemv_kernel(X, W, Y, R: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, D)
    x = tl.load(X + group * D + columns).to(tl.float32)
    w = tl.load(
        W + (group * R + rows[:, None]) * D + columns[None, :],
        rows[:, None] < R,
        0,
    ).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + group * R + rows, result, rows < R)


def wo_a_bf16_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for one token."""
    assert x.shape[0] == 1 and x.ndim == weight.ndim == 3
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.is_contiguous() and weight.is_contiguous()
    groups, rows, dim = weight.shape
    assert x.shape[1:] == (groups, dim) and dim == triton.next_power_of_2(dim)
    result = torch.empty((1, groups, rows), dtype=x.dtype, device=x.device)
    # One output row per CTA keeps register use low and exposes enough
    # independent weight loads for single-token decode.
    _wo_a_bf16_gemv_kernel[(rows, groups)](
        x,
        weight,
        result,
        rows,
        dim,
        1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


_WO_A_BT = 64  # output-row tile over R
_WO_A_SPLITS = 8  # split-K factor over D
_WO_A_KTILE = 128  # K step inside one split
# _wo_a_reduce sums a literal `tl.arange(0, 8)` over the split axis, so the two are coupled.
assert _WO_A_SPLITS == 8


@triton.jit
def _wo_a_partial(
    X,
    W,
    P,
    M: tl.constexpr,
    SX: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    KCHUNK: tl.constexpr,
    BT: tl.constexpr,
):
    """One fp32 partial of ``einsum('tgd,grd->tgr')`` for ``M <= 16`` tokens.

    Writes partial ``(SPLITS, M, G, R)`` for :func:`_wo_a_reduce` to sum over the split axis.
    Every stride is that of a CONTIGUOUS ``(G, R, D)`` weight, so the kernel is shape-generic in
    ``G`` and ``R`` (the launch derives both from ``weight.shape``); only the K chunking
    constrains ``D``, to a multiple of ``SPLITS * 128``.  Generalized from the TP4 (2, 1024, 4096)
    literals so the V4.1-Flash (8, 1024, 4096) shape can use it on SM89 too.
    """
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * BT + tl.arange(0, BT)
    k = split * KCHUNK + tl.arange(0, 128)
    acc = tl.zeros((16, BT), tl.float32)
    for i in range(KCHUNK // 128):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * D + offsets[None, :], m[:, None] < M, 0
        )
        w = tl.load(W + (group * R + n[None, :]) * D + offsets[:, None])
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * G + group) * R + n[None, :],
        acc,
        m[:, None] < M,
    )


def _wo_a_partials(x: torch.Tensor, weight: torch.Tensor, m: int) -> torch.Tensor:
    """The ``(SPLITS, m, G, R)`` fp32 partial buffer for a contiguous ``(G, R, D)`` weight."""
    groups, rows, dim = weight.shape
    assert rows % _WO_A_BT == 0, rows
    assert dim % (_WO_A_SPLITS * _WO_A_KTILE) == 0, dim
    partial = torch.empty(
        (_WO_A_SPLITS, m, groups, rows), dtype=torch.float32, device=x.device
    )
    _wo_a_partial[(rows // _WO_A_BT, groups, _WO_A_SPLITS)](
        x,
        weight,
        partial,
        m,
        x.stride(0),
        G=groups,
        R=rows,
        D=dim,
        SPLITS=_WO_A_SPLITS,
        KCHUNK=dim // _WO_A_SPLITS,
        BT=_WO_A_BT,
        num_warps=4,
        num_stages=3,
    )
    return partial


@triton.jit
def _wo_a_reduce(P, Y, E: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    values = tl.load(P + split[:, None] * E + i[None, :], i[None, :] < E, 0)
    tl.store(Y + i, tl.sum(values, 0), i < E)


def wo_a_bf16_small_batch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for a small token batch.
    Partial sums stay in FP32 until the final BF16, token-major store."""
    m = x.shape[0]
    groups, rows, dim = weight.shape
    assert 2 <= m <= 16
    assert x.shape[1:] == (groups, dim)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert weight.is_contiguous()
    assert x.stride(2) == 1 and x.stride(1) == dim and x.stride(0) >= groups * dim
    result = torch.empty((m, groups, rows), dtype=x.dtype, device=x.device)
    partial = _wo_a_partials(x, weight, m)
    total = m * groups * rows
    _wo_a_reduce[(triton.cdiv(total, 256),)](partial, result, total, num_warps=4)
    return result


@triton.jit
def _wo_a_reduce_quant(P, Q, S, M: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    v = tl.load(P + split[:, None] * (M * 2048) + row * 2048 + i[None, :])
    y = tl.sum(v, 0).to(tl.bfloat16).to(tl.float32).reshape((8, 32))
    amax = tl.max(tl.abs(y), 1)
    sf, inv = ue8m0_scale(amax)
    quant = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 2048 + i, quant.reshape((256,)))
    col = tile * 8 + tl.arange(0, 8)
    off = (col // 4) * 512 + row * 16 + col % 4
    tl.store(S + off, sf.to(tl.uint8))
    # Zero only padding rows; valid scale bytes have disjoint writers above.
    for z in tl.static_range(triton.cdiv(8192, M * 8 * 256)):
        s = (row * 8 + tile) * 256 + tl.arange(0, 256) + z * (M * 8 * 256)
        sr = (s % 512) // 16 + ((s % 16) // 4) * 32
        tl.store(S + s, 0, (s < 8192) & (sr >= M))


def _quantize_partial(p):
    m = p.shape[1]
    q = torch.empty((m, 2048), device=p.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(8192, device=p.device, dtype=torch.uint8)
    _wo_a_reduce_quant[(m, 8)](p, q, s, m, num_warps=4)
    return q, s


def wo_a_bf16_small_batch_mxfp8(x: torch.Tensor, weight: torch.Tensor):
    """WO-A with BF16 rounding followed by FlashInfer-compatible MXFP8 quantization.

    The MXFP8 epilogue's scale layout is baked to the TP4 ``(2, 1024, 4096)`` shape, so unlike
    :func:`wo_a_bf16_small_batch` this stays restricted to it.
    """
    m = x.shape[0]
    assert 2 <= m <= 16 and x.shape[1:] == (2, 4096)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    return _quantize_partial(_wo_a_partials(x, weight, m))
