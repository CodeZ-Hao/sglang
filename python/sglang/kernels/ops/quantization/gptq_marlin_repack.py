from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Constants matching device::marlin:: in marlin.cuh
_TILE_SIZE = 16


@cache_once
def _jit_gptq_marlin_repack_module() -> Module:
    return load_jit(
        "gptq_marlin_repack",
        cuda_files=["gemm/marlin/gptq_marlin_repack.cuh"],
        cuda_wrappers=[
            ("gptq_marlin_repack", "gptq_marlin_repack"),
            ("mxfp4_marlin_repack", "mxfp4_marlin_repack"),
        ],
    )


@debug_kernel_api
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits

    # Allocate output tensor
    out = torch.empty(
        (size_k // _TILE_SIZE, size_n * _TILE_SIZE // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )

    # JIT architecture detection is based on the current CUDA device. This
    # op can coexist with a remote DSpark draft of a different architecture.
    with torch.cuda.device(b_q_weight.device):
        module = _jit_gptq_marlin_repack_module()
        module.gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits)
    return out


@debug_kernel_api
def mxfp4_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Repack native ``[N, K // 2]`` MXFP4 bytes directly for Marlin.

    The leading expert dimension is optional. Passing ``out`` keeps addresses
    stable for the layerwise prefill slots and avoids an allocator call.
    """
    if b_q_weight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"expected int8/uint8 MXFP4 bytes, got {b_q_weight.dtype}")
    is_batched = b_q_weight.ndim == 3
    expected_input_shape = (
        (b_q_weight.shape[0], size_n, size_k // 2)
        if is_batched
        else (size_n, size_k // 2)
    )
    if tuple(b_q_weight.shape) != expected_input_shape:
        raise ValueError(
            f"expected weight shape {expected_input_shape}, got {tuple(b_q_weight.shape)}"
        )

    b_q_weight = b_q_weight.view(torch.uint8)
    matrix_shape = (size_k // _TILE_SIZE, size_n * _TILE_SIZE // 8)
    expected_shape = (
        (b_q_weight.shape[0], *matrix_shape) if is_batched else matrix_shape
    )
    if out is None:
        out = torch.empty(expected_shape, dtype=torch.int32, device=b_q_weight.device)
    elif tuple(out.shape) != expected_shape or out.dtype != torch.int32:
        raise ValueError(
            f"out must be int32 {expected_shape}, got {out.dtype} {tuple(out.shape)}"
        )

    with torch.cuda.device(b_q_weight.device):
        module = _jit_gptq_marlin_repack_module()
        module.mxfp4_marlin_repack(b_q_weight, out, size_k, size_n)
    return out
