# SPDX-License-Identifier: Apache-2.0
"""Prepared-weight Marlin backend for DeepSeek V4 MXFP4 MoE.

The preparation API is deliberately out-of-place and stream ordered.  A
layerwise loader can therefore target either resident weight buffer without
introducing events or synchronization in this module.  The compute API keeps
all shape-dependent storage in a per-CUDA-stream cache; steady-state forward
contains no allocator, transpose, concat, or routing conversion traffic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.gptq_marlin_repack import mxfp4_marlin_repack
from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm

V4_FP4_GROUP_SIZE = 32
_MARLIN_TILE = 16
_MAX_THREAD_N = 256


@dataclass
class V4MarlinPreparedWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int


@triton.jit
def _swizzle_e8m0_scales_kernel(
    src,
    dst,
    total,
    per_expert,
    n,
    k_groups,
    SRC_IS_E8M0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    expert = offsets // per_expert
    out_index = offsets - expert * per_expert

    # mxfp4_marlin_process_scales first applies the Marlin 8x8 transpose
    # permutation, then swaps the middle two elements of every group of four.
    marlin_index = (out_index // 4) * 4
    lane4 = out_index % 4
    marlin_index += tl.where(lane4 == 1, 2, tl.where(lane4 == 2, 1, lane4))
    tile = marlin_index // 64
    lane64 = marlin_index - tile * 64
    transposed_index = tile * 64 + (lane64 % 8) * 8 + lane64 // 8

    group = transposed_index // n
    col = transposed_index - group * n
    src_index = expert * per_expert + col * k_groups + group
    value = tl.load(src + src_index, mask=mask, other=0)
    if SRC_IS_E8M0:
        bits = value.to(tl.uint8)
    else:
        # V4 scales are exact powers of two.  Rounding is intentional and
        # matches torch's float -> float8_e8m0fnu conversion.
        exponent = tl.floor(tl.log2(value.to(tl.float32)) + 0.5) + 127.0
        exponent = tl.where(value > 0, exponent, 0.0)
        bits = tl.maximum(0.0, tl.minimum(255.0, exponent)).to(tl.uint8)
    tl.store(dst + offsets, bits, mask=mask)


def _swizzle_e8m0_scales(
    src: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    experts = src.shape[0]
    expected = (experts, size_n, size_k // V4_FP4_GROUP_SIZE)
    if tuple(src.shape) != expected:
        raise ValueError(f"expected scales {expected}, got {tuple(src.shape)}")
    output_shape = (experts, size_k // V4_FP4_GROUP_SIZE, size_n)
    if out is None:
        out = torch.empty(output_shape, dtype=torch.float8_e8m0fnu, device=src.device)
    elif tuple(out.shape) != output_shape or out.dtype != torch.float8_e8m0fnu:
        raise ValueError(
            f"scale out must be float8_e8m0fnu {output_shape}, got "
            f"{out.dtype} {tuple(out.shape)}"
        )

    src_is_bits = src.dtype in (torch.int8, torch.uint8, torch.float8_e8m0fnu)
    src_arg = src.view(torch.uint8) if src_is_bits else src
    total = out.numel()
    _swizzle_e8m0_scales_kernel[(triton.cdiv(total, 256),)](
        src_arg,
        out.view(torch.uint8),
        total,
        (size_k // V4_FP4_GROUP_SIZE) * size_n,
        size_n,
        size_k // V4_FP4_GROUP_SIZE,
        SRC_IS_E8M0=src_is_bits,
        BLOCK=256,
    )
    return out


def _prepared_shapes(num_experts: int, hidden_size: int, intermediate_size: int):
    return (
        (num_experts, hidden_size // _MARLIN_TILE, 4 * intermediate_size),
        (num_experts, hidden_size // V4_FP4_GROUP_SIZE, 2 * intermediate_size),
        (num_experts, intermediate_size // _MARLIN_TILE, 2 * hidden_size),
        (num_experts, intermediate_size // V4_FP4_GROUP_SIZE, hidden_size),
    )


def get_v4_mxfp4_marlin_storage_nbytes(
    *, num_experts: int, hidden_size: int, intermediate_size: int
) -> int:
    """Return the bytes required for one prepared V4 MXFP4 layer image.

    Keeping this calculation next to ``allocate_v4_mxfp4_marlin`` lets the
    lazy layerwise-prefill path reserve KV-cache budget without allocating the
    prepared tensors during server startup.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if hidden_size % 64 or intermediate_size % 64:
        raise ValueError(
            "Marlin requires hidden/intermediate multiples of 64, got "
            f"{hidden_size}/{intermediate_size}"
        )

    w13, w13_scale, w2, w2_scale = _prepared_shapes(
        num_experts, hidden_size, intermediate_size
    )
    return (
        math.prod(w13) * torch.tensor([], dtype=torch.int32).element_size()
        + math.prod(w13_scale)
        * torch.tensor([], dtype=torch.float8_e8m0fnu).element_size()
        + math.prod(w2) * torch.tensor([], dtype=torch.int32).element_size()
        + math.prod(w2_scale)
        * torch.tensor([], dtype=torch.float8_e8m0fnu).element_size()
    )


def allocate_v4_mxfp4_marlin(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> V4MarlinPreparedWeights:
    """Allocate stable caller-owned Marlin storage without reading raw weights."""
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if hidden_size % 64 or intermediate_size % 64:
        raise ValueError(
            "Marlin requires hidden/intermediate multiples of 64, got "
            f"{hidden_size}/{intermediate_size}"
        )
    shapes = _prepared_shapes(num_experts, hidden_size, intermediate_size)
    return V4MarlinPreparedWeights(
        w13=torch.empty(shapes[0], dtype=torch.int32, device=device),
        w13_scale=torch.empty(shapes[1], dtype=torch.float8_e8m0fnu, device=device),
        w2=torch.empty(shapes[2], dtype=torch.int32, device=device),
        w2_scale=torch.empty(shapes[3], dtype=torch.float8_e8m0fnu, device=device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )


def prepare_v4_mxfp4_marlin(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    out: Optional[V4MarlinPreparedWeights] = None,
) -> V4MarlinPreparedWeights:
    """Prepare native DSV4 weights on the current CUDA stream.

    ``out`` is optional for ordinary model loading and required by the
    layerwise double-buffer manager to keep storage addresses stable.
    """
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("V4 expert weights must be rank 3")
    experts = w13.shape[0]
    hidden_size = w13.shape[2] * 2
    intermediate_size = w2.shape[2] * 2
    if tuple(w13.shape) != (experts, 2 * intermediate_size, hidden_size // 2):
        raise ValueError(f"inconsistent w13 shape {tuple(w13.shape)}")
    if tuple(w2.shape) != (experts, hidden_size, intermediate_size // 2):
        raise ValueError(f"inconsistent w2 shape {tuple(w2.shape)}")
    if hidden_size % 64 or intermediate_size % 64:
        raise ValueError(
            f"Marlin requires hidden/intermediate multiples of 64, got "
            f"{hidden_size}/{intermediate_size}"
        )

    if out is None:
        out = allocate_v4_mxfp4_marlin(
            num_experts=experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=w13.device,
        )
    else:
        shapes = _prepared_shapes(experts, hidden_size, intermediate_size)
        actual = (out.w13.shape, out.w13_scale.shape, out.w2.shape, out.w2_scale.shape)
        if tuple(map(tuple, actual)) != shapes:
            raise ValueError(f"prepared output shapes {actual} do not match {shapes}")
        if (
            out.hidden_size != hidden_size
            or out.intermediate_size != intermediate_size
            or out.num_experts != experts
        ):
            raise ValueError(
                "prepared output metadata does not match raw weights: "
                f"got E/K/N={out.num_experts}/{out.hidden_size}/"
                f"{out.intermediate_size}, expected "
                f"{experts}/{hidden_size}/{intermediate_size}"
            )

    # Grid-y owns experts, so each matrix is prepared in one stream-ordered
    # launch instead of issuing hundreds of per-expert Python calls.
    mxfp4_marlin_repack(w13, hidden_size, 2 * intermediate_size, out.w13)
    mxfp4_marlin_repack(w2, intermediate_size, hidden_size, out.w2)
    _swizzle_e8m0_scales(
        w13_scale, size_k=hidden_size, size_n=2 * intermediate_size, out=out.w13_scale
    )
    _swizzle_e8m0_scales(
        w2_scale, size_k=intermediate_size, size_n=hidden_size, out=out.w2_scale
    )
    return out


@triton.jit
def _sanitize_topk_kernel(
    ids_in, weights_in, ids_out, weights_out, total, experts, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    ids = tl.load(ids_in + offsets, mask=mask, other=-1).to(tl.int32)
    weights = tl.load(weights_in + offsets, mask=mask, other=0.0).to(tl.float32)
    valid = (ids >= 0) & (ids < experts)
    tl.store(ids_out + offsets, tl.where(valid, ids, 0), mask=mask)
    tl.store(weights_out + offsets, tl.where(valid, weights, 0.0), mask=mask)


@triton.jit
def _swiglu_kernel(
    inp, out, total, n, limit, HAS_LIMIT: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    row = offsets // n
    col = offsets - row * n
    gate = tl.load(inp + row * (2 * n) + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(inp + row * (2 * n) + n + col, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.maximum(-limit, tl.minimum(up, limit))
    value = gate * tl.sigmoid(gate) * up
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _topk_reduce_kernel(
    inp, out, m, k, routed_scale, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (row < m) & (cols < k)
    acc = tl.zeros((BLOCK,), tl.float32)
    for topk_index in range(TOPK):
        acc += tl.load(
            inp + (row * TOPK + topk_index) * k + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(out + row * k + cols, acc * routed_scale, mask=mask)


@triton.jit
def _topk_reduce_gather_kernel(
    inp, out, slot_row, m, k, routed_scale, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    """`_topk_reduce_kernel` for a COMPACTED row set -- see `V4MarlinCompactedRouting`.

    Same arithmetic and the SAME summation order (slot 0..TOPK-1); the only change is
    that each (token, slot) names its row through `slot_row` instead of the fixed
    `token * TOPK + slot` layout, and a -1 slot contributes nothing.  Each row is still
    exactly one expert, so the per-slot dot product is untouched and the result is
    bit-identical to the dense call -- only the row set is smaller.
    """
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (row < m) & (cols < k)
    acc = tl.zeros((BLOCK,), tl.float32)
    for topk_index in range(TOPK):
        sr = tl.load(slot_row + row * TOPK + topk_index, mask=row < m, other=-1).to(
            tl.int32
        )
        acc += tl.load(
            inp + sr * k + cols, mask=mask & (sr >= 0), other=0.0
        ).to(tl.float32)
    tl.store(out + row * k + cols, acc * routed_scale, mask=mask)


class V4MarlinCompactedRouting(msgspec.Struct, frozen=True):
    """A layer's routing restricted to one expert RANGE, as a COMPACT row set.

    Why this exists: `apply_v4_marlin_moe` sizes its two GEMMs by `routed_rows = m *
    topk` whatever the range's expert count, and `_sanitize_topk_kernel` maps every
    out-of-range id to expert 0 with weight 0 **and that row is still computed**, so
    splitting a layer into R ranges costs R x the row work rather than 1x.  Measured
    2026-09-24 (`ut_marlin_range_compact.py`, real shape, rows the only variable):
    98304 rows -> 94.95 ms, 49152 -> 51.13 ms (0.539x), so a compacted 2-range split
    removes the 1.61x inflation that made `PIPELINE_RANGES=2` a net loss on the Marlin
    slot.  With it the layerwise period should become `max(305 ms transfer, 95 + 190 ms
    compute) = 305 ms/layer`, i.e. the transfer fully hidden.

    Rows are enumerated TOKEN-MAJOR, slot order within a token, so `row_token` is
    monotone non-decreasing and every compact array is a plain masked select of the
    flattened `[m, topk]` routing.

    Fields:
        ids:         [routed_rows] int32, range-local expert ids (0 <= id < count)
        weights:     [routed_rows] float32
        row_token:   [routed_rows] int32, the token each row belongs to
        slot_row:    [m, topk] int32, the compact row of (token, slot), or -1
        topk:        the DENSE topk, needed only to size the shared workspace
        routed_rows: rows in this range
    """

    ids: torch.Tensor
    weights: torch.Tensor
    row_token: torch.Tensor
    slot_row: torch.Tensor
    topk: int
    routed_rows: int


def build_compacted_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    start: int,
    count: int,
) -> V4MarlinCompactedRouting:
    """Restrict `[m, topk]` routing to experts in `[start, start + count)`.

    Returns the compact row set for `apply_v4_marlin_moe(compact=...)`.  `slot_row` is the
    only derived quantity the reduce needs, and it doubles as the compaction index: the
    prefix sum of the in-range mask in token-major order IS the row enumeration, so the
    gather and the scatter are the same tensor read two ways.
    """
    m, topk = topk_ids.shape
    mask = (topk_ids >= start) & (topk_ids < start + count)
    flat = mask.reshape(-1)
    slot_row = torch.cumsum(flat.to(torch.int32), dim=0, dtype=torch.int32) - 1
    slot_row = torch.where(flat, slot_row, torch.full_like(slot_row, -1)).reshape(m, topk)
    routed_rows = int(flat.sum().item())
    index = slot_row.reshape(-1)
    keep = index >= 0
    return V4MarlinCompactedRouting(
        ids=(topk_ids.reshape(-1)[keep] - start).to(torch.int32),
        weights=topk_weights.reshape(-1)[keep].to(torch.float32),
        row_token=torch.repeat_interleave(
            torch.arange(m, device=topk_ids.device, dtype=torch.int32),
            mask.sum(dim=1).to(torch.int64),
        ),
        slot_row=slot_row.to(torch.int32),
        topk=topk,
        routed_rows=routed_rows,
    )




def _select_block_size(m: int, topk: int, experts: int) -> int:
    average = m * topk / max(experts, 1)
    for block in (8, 16, 32, 48, 64):
        if average / block < 0.9:
            return block
    return 64


@dataclass
class _V4MarlinWorkspace:
    capacity_m: int
    block_size: int
    ids: torch.Tensor
    weights: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_pad: torch.Tensor
    cumsum: torch.Tensor
    locks: torch.Tensor
    intermediate13: torch.Tensor
    intermediate2: torch.Tensor
    c_tmp: torch.Tensor
    output: torch.Tensor
    empty: torch.Tensor


_WORKSPACES: dict[tuple, _V4MarlinWorkspace] = {}


def release_workspace_cache() -> int:
    """Drop the per-stream workspace cache and return the GiB handed back.

    The workspace is ~1.6 GiB at capacity 16384 (`intermediate13` 960 MiB +
    `intermediate2` 453 MiB + `output` 168 MiB) and it is held by this dict for
    the life of the process, so a chunked prefill keeps the PREVIOUS chunk's
    workspace live while allocating the next one -- the effective requirement is
    double.  That is the DSV4-stage problem the scheduler's per-chunk reclamation
    exists to fix, and holding these tensors here is what defeats it (an
    `empty_cache` cannot free a still-referenced tensor).

    Safe to call mid-flight: a call already inside `apply_v4_marlin_moe` holds its
    own reference to the workspace, so dropping the dict entry only returns the
    memory of anything not currently in use.  The next call re-allocates, which is
    one `torch.empty` of a few hundred MiB.
    """
    if not _WORKSPACES:
        return 0
    freed = 0
    for ws in _WORKSPACES.values():
        seen: set[int] = set()
        for t in (
            ws.ids,
            ws.weights,
            ws.sorted_ids,
            ws.expert_ids,
            ws.num_tokens_post_pad,
            ws.cumsum,
            ws.locks,
            ws.intermediate13,
            ws.intermediate2,
            ws.c_tmp,
            ws.output,
        ):
            if id(t) in seen:
                continue
            seen.add(id(t))
            freed += t.numel() * t.element_size()
    _WORKSPACES.clear()
    return int(freed / 1024**3)


def _routing_capacity(m: int, topk: int, experts: int, block: int) -> int:
    tokens = m * topk
    if tokens < experts + 1:
        return tokens * block
    return tokens + (experts + 1) * (block - 1)


def _get_workspace(
    hidden_states: torch.Tensor,
    weights: V4MarlinPreparedWeights,
    topk: int,
) -> _V4MarlinWorkspace:
    m = hidden_states.shape[0]
    capacity_m = max(1, triton.next_power_of_2(m))
    block = _select_block_size(capacity_m, topk, weights.num_experts)
    stream_id = torch.cuda.current_stream(hidden_states.device).cuda_stream
    key = (
        hidden_states.device.index,
        stream_id,
        hidden_states.dtype,
        weights.hidden_size,
        weights.intermediate_size,
        weights.num_experts,
        topk,
        capacity_m,
        block,
    )
    cached = _WORKSPACES.get(key)
    if cached is not None:
        return cached

    device = hidden_states.device
    k = weights.hidden_size
    n = weights.intermediate_size
    routed = _routing_capacity(capacity_m, topk, weights.num_experts, block)
    route_blocks = triton.cdiv(routed, block)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    locks_size = max(1, min((max(2 * n, k) // 64) * route_blocks, sms * 4))
    c_tmp_first = min(2 * n * routed, sms * 4 * block * _MAX_THREAD_N)
    c_tmp_second = min(k * routed, sms * 4 * block * _MAX_THREAD_N)
    if block == 8:
        c_tmp_first *= 2
        c_tmp_second *= 2
    c_tmp_size = max(1, c_tmp_first, c_tmp_second)
    routed_rows = capacity_m * topk

    cached = _V4MarlinWorkspace(
        capacity_m=capacity_m,
        block_size=block,
        ids=torch.empty((capacity_m, topk), dtype=torch.int32, device=device),
        weights=torch.empty((capacity_m, topk), dtype=torch.float32, device=device),
        sorted_ids=torch.empty(routed, dtype=torch.int32, device=device),
        expert_ids=torch.empty(route_blocks, dtype=torch.int32, device=device),
        num_tokens_post_pad=torch.empty(1, dtype=torch.int32, device=device),
        cumsum=torch.empty(weights.num_experts + 2, dtype=torch.int32, device=device),
        locks=torch.zeros(locks_size, dtype=torch.int32, device=device),
        intermediate13=torch.empty(
            routed_rows * max(2 * n, k), dtype=hidden_states.dtype, device=device
        ),
        intermediate2=torch.empty(
            (routed_rows, n), dtype=hidden_states.dtype, device=device
        ),
        c_tmp=torch.empty(c_tmp_size, dtype=torch.float32, device=device),
        output=torch.empty((capacity_m, k), dtype=hidden_states.dtype, device=device),
        empty=torch.empty(0, dtype=hidden_states.dtype, device=device),
    )
    _WORKSPACES[key] = cached
    return cached


def apply_v4_marlin_moe(
    *,
    hidden_states: torch.Tensor,
    prepared: V4MarlinPreparedWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    swiglu_limit: Optional[float] = None,
    compact: Optional[V4MarlinCompactedRouting] = None,
) -> torch.Tensor:
    """Execute DSV4 MXFP4 MoE with deterministic non-atomic Marlin GEMMs.

    `compact` (optional) replaces the dense `[m, topk]` routing with a range-restricted
    COMPACTED row set, which is what makes an expert-range split cost 1x instead of 2x
    (see `V4MarlinCompactedRouting`).  `topk_ids` / `topk_weights` are then unused -- the
    caller keeps passing them only because it already has them, and the dense `topk` is
    still taken from `compact.topk` to size the shared workspace.
    """
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            f"V4 Marlin requires BF16 activations, got {hidden_states.dtype}"
        )
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    m, k = hidden_states.shape
    if k != prepared.hidden_size:
        raise ValueError(f"hidden size {k} != prepared {prepared.hidden_size}")
    if m == 0 or prepared.num_experts == 0:
        return torch.zeros_like(hidden_states)
    # `compact` replaces the dense `[m, topk]` routing with a RANGE-restricted row set --
    # see `V4MarlinCompactedRouting`.  The workspace is still keyed by the dense topk, so
    # it is sized for the dense row count, which upper-bounds the compact one; only the
    # slices below are narrowed.
    dense_topk = compact.topk if compact is not None else topk_ids.shape[1]
    workspace = _get_workspace(hidden_states, prepared, dense_topk)
    n = prepared.intermediate_size

    if compact is None:
        topk = topk_ids.shape[1]
        routed_rows = m * topk
        ids = workspace.ids[:m]
        routing_weights = workspace.weights[:m]
        _sanitize_topk_kernel[(triton.cdiv(routed_rows, 256),)](
            topk_ids,
            topk_weights,
            ids,
            routing_weights,
            routed_rows,
            prepared.num_experts,
            BLOCK=256,
        )
        gemm1_a = hidden_states
        gemm1_size_m = m
        gemm1_top_k = topk
    else:
        topk = compact.topk
        routed_rows = compact.routed_rows
        ids = workspace.ids.flatten()[:routed_rows]
        routing_weights = workspace.weights.flatten()[:routed_rows]
        # Already range-local and valid: the caller built them from the in-range mask,
        # which is exactly what the sanitizer would have produced, minus the out-of-range
        # rows it would have kept and computed.
        ids.copy_(compact.ids)
        routing_weights.copy_(compact.weights)
        # The GEMM gathers A by ROW (`a[sorted_ids[row] // top_k]`), so a compacted row
        # set needs compacted activations; with `top_k=1` the existing gather then reads
        # straight through.  ~0.5 GiB at chunk 16384 for half a layer's rows, against a
        # ~50 ms GEMM per range.
        gemm1_a = hidden_states.index_select(0, compact.row_token.to(torch.int64))
        gemm1_size_m = routed_rows
        gemm1_top_k = 1

    from sgl_kernel import moe_align_block_size
    from sgl_kernel.scalar_type import scalar_types

    moe_align_block_size(
        ids,
        prepared.num_experts + 1,
        workspace.block_size,
        workspace.sorted_ids,
        workspace.expert_ids,
        workspace.num_tokens_post_pad,
        workspace.cumsum,
        True,
    )

    intermediate1 = workspace.intermediate13[: routed_rows * 2 * n].view(
        routed_rows, 2 * n
    )
    intermediate3 = workspace.intermediate13[: routed_rows * k].view(routed_rows, k)
    intermediate2 = workspace.intermediate2[:routed_rows]
    fp4_type = scalar_types.float4_e2m1f

    moe_wna16_marlin_gemm(
        gemm1_a,
        intermediate1,
        prepared.w13,
        workspace.empty,
        prepared.w13_scale,
        workspace.empty,
        workspace.empty,
        workspace.empty,
        workspace.empty,
        workspace.locks,
        workspace.sorted_ids,
        workspace.expert_ids,
        workspace.num_tokens_post_pad,
        routing_weights,
        moe_block_size=workspace.block_size,
        top_k=gemm1_top_k,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=fp4_type,
        size_m=gemm1_size_m,
        size_n=2 * n,
        size_k=k,
        use_atomic_add=False,
        use_fp32_reduce=True,
        c_tmp_or_none=workspace.c_tmp,
        empty_tensor_or_none=workspace.empty,
        initialize_output=True,
    )

    _swiglu_kernel[(triton.cdiv(routed_rows * n, 256),)](
        intermediate1,
        intermediate2,
        routed_rows * n,
        n,
        0.0 if swiglu_limit is None else swiglu_limit,
        HAS_LIMIT=swiglu_limit is not None,
        BLOCK=256,
    )

    moe_wna16_marlin_gemm(
        intermediate2,
        intermediate3,
        prepared.w2,
        workspace.empty,
        prepared.w2_scale,
        workspace.empty,
        workspace.empty,
        workspace.empty,
        workspace.empty,
        workspace.locks,
        workspace.sorted_ids,
        workspace.expert_ids,
        workspace.num_tokens_post_pad,
        routing_weights,
        moe_block_size=workspace.block_size,
        top_k=1,
        mul_topk_weights=True,
        is_ep=False,
        b_q_type=fp4_type,
        size_m=routed_rows,
        size_n=k,
        size_k=n,
        use_atomic_add=False,
        use_fp32_reduce=True,
        c_tmp_or_none=workspace.c_tmp,
        empty_tensor_or_none=workspace.empty,
        initialize_output=True,
    )

    output = workspace.output[:m]
    if compact is None:
        _topk_reduce_kernel[(m, triton.cdiv(k, 256))](
            intermediate3,
            output,
            m,
            k,
            routed_scaling_factor,
            TOPK=topk,
            BLOCK=256,
        )
    else:
        # Same arithmetic and the same slot order, reading the rows through `slot_row`.
        _topk_reduce_gather_kernel[(m, triton.cdiv(k, 256))](
            intermediate3,
            output,
            compact.slot_row,
            m,
            k,
            routed_scaling_factor,
            TOPK=topk,
            BLOCK=256,
        )
    return output
