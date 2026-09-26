# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).

Diagnostic / escape-hatch environment variables (KT-DEBUG-ONLY; not for prod):

    SGLANG_KT_HYBRID_TIMING=1
        Per-call wall-time breakdown of submit / mask / gpu / sync / merge
        / cpu_wait stages. Logged at DEBUG for layers (0, 5, 20, 35) on TP0.

    SGLANG_KT_HYBRID_TIMING_DEEP=1
        Insert torch.cuda.synchronize() at each timing stage so DEEP numbers
        reflect real GPU work rather than async-launch return time. Slows
        decode meaningfully; only enable for one-shot triage.

    SGLANG_KT_HYBRID_NO_CPU_STREAM=1
        Collapse the CPU-experts CUDA stream onto the main stream. Useful
        when isolating regressions caused by the multi-stream submit path.

    SGLANG_KT_BYPASS_GPU_MOE=1
        Force GPU-experts apply() to a zero return; routed expert output
        comes purely from the CPU side. "Plan-C" fallback for diagnosing
        whether a regression sits in the GPU MoE path or the merge math.
"""

import bisect
import copy
import ctypes
import gc
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.utils import is_cuda

if is_cuda():
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper, generate_gpu_experts_masks

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)

# Global cache for GPU experts masks (initialized once per session)
_KT_GPU_EXPERTS_MASKS: Optional[torch.Tensor] = None


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        gpu_experts_mask: Boolean tensor of shape [num_experts] indicating which experts are on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        numa_nodes: Optional explicit NUMA node ids for each KT threadpool
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
        gpu_prefill_token_threshold: token threshold for enabling full GPU fallback
        kt_enable_dynamic_expert_update: Enable dynamic GPU expert updates based on runtime statistics
        expert_lora_path: Optional PEFT adapter directory for KT CPU expert LoRA
    """

    layer_idx: int
    gpu_experts_mask: torch.Tensor  # bool tensor of shape [num_experts]
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    numa_nodes: Optional[List[int]] = None
    num_layers: Optional[int] = None
    gpu_prefill_token_threshold: Optional[int] = None
    kt_enable_dynamic_expert_update: bool = False
    expert_lora_path: Optional[str] = None
    # Local addition: explicit checkpoint key prefix (e.g. "mtp.0") for
    # draft/MTP stage weights. None = default "model.layers.{idx}" candidates.
    weight_key_prefix: Optional[str] = None


@dataclass
class KTExpertLoraWeights:
    gate_lora_a: torch.Tensor
    gate_lora_b: torch.Tensor
    up_lora_a: torch.Tensor
    up_lora_b: torch.Tensor
    down_lora_a: torch.Tensor
    down_lora_b: torch.Tensor
    rank: int
    alpha: float


_KT_SFT_METHOD_BY_INFERENCE_METHOD = {
    "AMXBF16": "AMXBF16_SFT",
    "BF16": "AMXBF16_SFT",
    "AMXINT8": "AMXINT8_SFT",
    "AMXINT4": "AMXINT4_SFT",
}


def _map_kt_method_to_sft_method(method: str) -> str:
    normalized = method.upper()
    if normalized in _KT_SFT_METHOD_BY_INFERENCE_METHOD:
        return _KT_SFT_METHOD_BY_INFERENCE_METHOD[normalized]
    raise ValueError(
        f"--kt-expert-lora-path currently supports only AMX/BF16 SFT-compatible "
        f"KT methods {sorted(_KT_SFT_METHOD_BY_INFERENCE_METHOD)}, got {method!r}."
    )


def _load_adapter_config(adapter_path: Path) -> Tuple[Optional[int], float]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"KT expert LoRA adapter config not found: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    rank = config.get("r")
    if rank is not None:
        rank = int(rank)
    alpha = float(config.get("lora_alpha", rank if rank is not None else 1.0))
    return rank, alpha


def _find_adapter_weight_file(adapter_path: Path) -> Path:
    preferred = adapter_path / "adapter_model.safetensors"
    if preferred.is_file():
        return preferred
    candidates = sorted(adapter_path.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No safetensors adapter weights found under {adapter_path}"
        )
    if len(candidates) > 1:
        logger.warning(
            "Multiple safetensors files found under %s; using %s for KT expert LoRA",
            adapter_path,
            candidates[0],
        )
    return candidates[0]


def _get_expert_lora_tensor(
    state_dict: Dict[str, torch.Tensor],
    layer_idx: int,
    expert_idx: int,
    proj_name: str,
    lora_name: str,
) -> torch.Tensor:
    suffixes = [
        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.{lora_name}.weight",
        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.{lora_name}.default.weight",
    ]
    prefixes = ["", "base_model.", "base_model.model.", "base_model.model.model."]
    for suffix in suffixes:
        for prefix in prefixes:
            key = prefix + suffix
            if key in state_dict:
                return state_dict[key]
    for key, tensor in state_dict.items():
        if any(key.endswith(suffix) for suffix in suffixes):
            return tensor
    raise KeyError(
        f"Missing KT expert LoRA tensor for layer={layer_idx}, expert={expert_idx}, "
        f"proj={proj_name}, lora={lora_name}. Expected suffix like {suffixes[0]!r}."
    )


_KT_LORA_STATE_DICT_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def _load_kt_expert_lora_weights(
    adapter_path: str,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> KTExpertLoraWeights:
    adapter_dir = Path(adapter_path)
    rank_from_config, alpha = _load_adapter_config(adapter_dir)
    weight_file = _find_adapter_weight_file(adapter_dir)
    weight_file_str = str(weight_file)
    if weight_file_str not in _KT_LORA_STATE_DICT_CACHE:
        from safetensors.torch import load_file

        _KT_LORA_STATE_DICT_CACHE[weight_file_str] = load_file(
            weight_file_str, device="cpu"
        )
    state_dict = _KT_LORA_STATE_DICT_CACHE[weight_file_str]
    sample = _get_expert_lora_tensor(
        state_dict, layer_idx, 0, "gate_proj", "lora_A"
    )
    rank = rank_from_config or int(sample.shape[0])
    if int(sample.shape[0]) != rank:
        raise ValueError(
            f"KT expert LoRA rank mismatch for layer {layer_idx}: "
            f"adapter_config r={rank}, sample tensor rank={int(sample.shape[0])}"
        )

    # SGLang may set the default torch device to CUDA during model loading.
    # KT SFT CPU kernels consume raw host pointers, so these adapter staging
    # buffers must be explicitly allocated on CPU.
    device = torch.device("cpu")
    gate_lora_a = torch.zeros((num_experts, rank, hidden_size), dtype=dtype, device=device)
    gate_lora_b = torch.zeros((num_experts, moe_intermediate_size, rank), dtype=dtype, device=device)
    up_lora_a = torch.zeros((num_experts, rank, hidden_size), dtype=dtype, device=device)
    up_lora_b = torch.zeros((num_experts, moe_intermediate_size, rank), dtype=dtype, device=device)
    down_lora_a = torch.zeros((num_experts, rank, moe_intermediate_size), dtype=dtype, device=device)
    down_lora_b = torch.zeros((num_experts, hidden_size, rank), dtype=dtype, device=device)

    targets = [
        ("gate_proj", "lora_A", gate_lora_a),
        ("gate_proj", "lora_B", gate_lora_b),
        ("up_proj", "lora_A", up_lora_a),
        ("up_proj", "lora_B", up_lora_b),
        ("down_proj", "lora_A", down_lora_a),
        ("down_proj", "lora_B", down_lora_b),
    ]
    for expert_idx in range(num_experts):
        for proj_name, lora_name, dst in targets:
            tensor = _get_expert_lora_tensor(
                state_dict, layer_idx, expert_idx, proj_name, lora_name
            )
            if tuple(tensor.shape) != tuple(dst[expert_idx].shape):
                raise ValueError(
                    f"KT expert LoRA shape mismatch for layer={layer_idx}, "
                    f"expert={expert_idx}, {proj_name}.{lora_name}: expected "
                    f"{tuple(dst[expert_idx].shape)}, got {tuple(tensor.shape)}"
                )
            dst[expert_idx].copy_(tensor.to(dtype=dtype, device=device))

    return KTExpertLoraWeights(
        gate_lora_a=gate_lora_a.contiguous(),
        gate_lora_b=gate_lora_b.contiguous(),
        up_lora_a=up_lora_a.contiguous(),
        up_lora_b=up_lora_b.contiguous(),
        down_lora_a=down_lora_a.contiguous(),
        down_lora_b=down_lora_b.contiguous(),
        rank=rank,
        alpha=alpha,
    )


# Serialize KT native weight-write tasks against the PD decode channel's KT
# forwards.  The weight-write submit + drain enqueues into the process-global
# CPUInfer singleton (and writes into KExpertsCPUBuffer / host-cache storage);
# holding this lock around a write burst guarantees the native queue is only
# touched by one thread at a time (a concurrent forward would otherwise read
# half-written expert data).  Origin: kt-sglang 耦合 (host-weight cache build).
_KT_CPU_LOCK = threading.Lock()
_SHARED_FULL_CONTEXT = None
_SHARED_STAGING_BUFFER = None  # Global shared staging buffer for all MoE layers
_MXFP4_PREFILL_LAYER_REGISTRY = {}
_MXFP4_LAYERWISE_MANAGERS = {}
_MXFP4_LAYERWISE_DISABLED_REASONS = {}

# --- Per-layer logical-expert route histogram (measurement only) -------------
# The ExpertDistributionRecorder cannot see this path: with --kt-weight-path the
# routed experts are dispatched here, and on this tree the recorder's
# select_experts hook is reached only by the MTP draft (measured 2026-09-22: a
# full decode run logged 2 hook calls, both layer_idx=None, both from
# dspark_draft).  `KT_ROUTE_HIST=<path>` accumulates the routing here instead,
# at the point where the logical (pre-remap) topk ids are still in hand, keyed
# by (phase, layer) with phase split by token count.  The histogram is
# rewritten to <path> every _KT_ROUTE_HIST_DUMP_EVERY notes; creating
# <path>.reset zeroes the accumulators (to separate runs inside one server
# life).  Inert unless the env var is set, and no launcher sets it.
_KT_ROUTE_HIST_PATH = os.environ.get("KT_ROUTE_HIST")
_KT_ROUTE_HIST: Dict[str, torch.Tensor] = {}
_KT_ROUTE_HIST_NOTES = 0
_KT_ROUTE_HIST_DUMP_EVERY = 200
_KT_ROUTE_HIST_DECODE_TOKENS = 16  # num_tokens <= this is a decode step


def kt_route_hist_note(
    layer_idx: int, num_experts: int, num_tokens: int, topk_ids
) -> None:
    """Accumulate one layer's routing into the KT_ROUTE_HIST histogram.

    Keyed by "<phase>/<num_experts>/<layer>" so the MTP draft's 128-expert
    layers stay separate from the target's 384-expert ones.
    """
    global _KT_ROUTE_HIST_NOTES
    if _KT_ROUTE_HIST_PATH is None:
        return
    phase = "decode" if num_tokens <= _KT_ROUTE_HIST_DECODE_TOKENS else "prefill"
    ids = topk_ids.detach().reshape(-1).to(torch.long)
    ids = ids[ids >= 0]
    if ids.numel() == 0:
        return
    bc = torch.bincount(ids.cpu())
    key = f"{phase}/{int(num_experts)}/{int(layer_idx)}"
    acc = _KT_ROUTE_HIST.get(key)
    if acc is None:
        acc = torch.zeros(bc.numel())
        _KT_ROUTE_HIST[key] = acc
    elif acc.numel() < bc.numel():
        padded = torch.zeros(bc.numel())
        padded[: acc.numel()] = acc
        acc = padded
        _KT_ROUTE_HIST[key] = acc
    acc[: bc.numel()] += bc
    _KT_ROUTE_HIST_NOTES += 1
    if _KT_ROUTE_HIST_NOTES % _KT_ROUTE_HIST_DUMP_EVERY == 0:
        _kt_route_hist_dump()


_DISTINCT_STATS: Dict[str, List[float]] = {}
_DISTINCT_NOTES = 0
_DISTINCT_PRINT_EVERY = 200


def kt_distinct_note(num_experts: int, num_tokens: int, topk_ids) -> None:
    """Mean DISTINCT experts per layer per call, gated by KT_DISTINCT_STATS=1.

    The CPU MoE reads one expert's weights per *distinct* expert a step routes
    to, so this -- not the token count -- is what sizes its bytes.  It is also
    the number the step's wall time does not show: the wait for the CPU experts
    is a device-side event on `_sync_done_event`, so it lands in GPU idle time,
    while the host's own `cpu_wait` stays near zero (0.04 ms/layer measured).
    Averaged over layers, split by phase, with the routes/call beside it so the
    amortization per read is visible.
    """
    global _DISTINCT_NOTES
    if not os.environ.get("KT_DISTINCT_STATS") or num_tokens <= 0:
        return
    ids = topk_ids.detach().reshape(-1)
    ids = ids[ids >= 0]
    if ids.numel() == 0:
        return
    counts = torch.bincount(ids.cpu())
    if counts.numel() > num_experts:
        counts = counts[:num_experts]
    distinct = int(counts.count_nonzero())
    phase = "decode" if num_tokens <= _KT_ROUTE_HIST_DECODE_TOKENS else "prefill"
    acc = _DISTINCT_STATS.setdefault(f"{phase}/{int(num_experts)}", [0.0, 0.0, 0.0])
    acc[0] += 1
    acc[1] += distinct
    acc[2] += int(ids.numel())
    _DISTINCT_NOTES += 1
    if _DISTINCT_NOTES % _DISTINCT_PRINT_EVERY == 0:
        parts = " ".join(
            f"{k} n={int(v[0])} distinct/layer={v[1] / v[0]:.2f} routes/layer={v[2] / v[0]:.1f}"
            for k, v in sorted(_DISTINCT_STATS.items())
        )
        print(f"[distinct] {parts}", flush=True)


def _kt_route_hist_dump() -> None:
    reset_marker = Path(_KT_ROUTE_HIST_PATH + ".reset")
    if reset_marker.exists():
        _KT_ROUTE_HIST.clear()
        try:
            reset_marker.unlink()
        except OSError:
            pass
    out = {key: [int(x) for x in counts] for key, counts in _KT_ROUTE_HIST.items()}
    tmp = Path(_KT_ROUTE_HIST_PATH + ".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, _KT_ROUTE_HIST_PATH)

# Eager host-weight cache build (TP1 layerwise prefill).  The KT pack + the
# cudaHostAlloc of a fresh 3.6GB pinned buffer cost ~0.9s/layer on this box, so
# a synchronous per-layer build during the cold first prefill caps prefill at
# ~350 tok/s.  A daemon thread started right after model load builds every
# layer's pinned cache in the background (hidden in the ~3min startup); the
# first layerwise prefill then only does the fast pinned H2D.
_MXFP4_EAGER_CACHE_BUILDER: Optional[threading.Thread] = None
_MXFP4_EAGER_CACHE_LOCK = threading.Lock()
_MXFP4_EAGER_HOST_CACHE: Dict[
    tuple, Dict[int, Optional[Dict[str, torch.Tensor]]]
] = {}
_UNSET = object()  # sentinel: eager cache entry not (yet) recorded


def _is_cuda_oom(exc: Optional[BaseException]) -> bool:
    """True if `exc` is (or wraps, via __cause__/__context__) a CUDA OOM.

    The layerwise manager's TP consensus helpers re-raise forward failures as
    `RuntimeError(...) from original_exc`, so a CUDA OOM surfaces as a wrapped
    RuntimeError rather than a bare torch.cuda.OutOfMemoryError.
    """
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
        if "out of memory" in str(e).lower():
            return True
        e = e.__cause__ or e.__context__
    return False


# Experts per H2D/repack batch for the host-cache layerwise path (baseline
# host staging: raw on CPU, GPU keeps 1 prepared slot + a small staging).
#
# Once the H2D and the repack overlap (two staging buffers + two streams, see
# _repack_from_cache) this only trades staging VRAM against per-batch launch
# count, because the repack cost per expert is independent of the batch size.
# 16 keeps the pair of buffers at 0.47 GiB, which is what the ~8 GiB of
# headroom left after the slots can afford.  Raising it to 32 measured +3.9%
# while the two were still serialized on one stream; 64 did not fit VRAM at all
# (it OOM'd the first layerwise batch on this 32 GB card).
#
# SGLANG_MXFP4_BATCH_EXPERTS overrides it.  The reason to lower it is VRAM, not
# throughput: the staging footprint is 2 x BATCH_EXPERTS experts, so halving the
# batch frees 0.47 GiB, which is what chunk 32768 needs on top of chunk 16384's
# 8.34 GiB of post-slot headroom (measured: 8.10 free at 32768 against a ~8.2
# GiB batch requirement).  A bigger chunk is worth far more than the extra
# launches cost, because the 21 x 7.6 GiB weight transfer is per-CHUNK.
BATCH_EXPERTS = max(1, int(os.environ.get("SGLANG_MXFP4_BATCH_EXPERTS", "16")))
# Expert ranges the prepared slot is split into so the host->GPU transfer of the
# next range overlaps the GPU compute of the current one.  **Measured twice, and
# the default is 1: the split buys nothing.**  The routing compaction that removes
# the GEMM inflation (see below) does work -- R=2 costs 103.8 ms against the dense
# call's 95.6 where the shipped sanitizer split cost 173.7 -- and PR=2 is STILL
# 2-3% slower end to end (three paired probes: 1414.5/1427.1 against
# 1390.2/1365.3), because the split cannot shorten what the period actually is.
#
# What the period is, from the stage events (SGLANG_LAYERWISE_STAGE_DEBUG=1):
# per encoder layer the transfer stream is busy 425-525 ms moving the layer's
# 7.65 GB off the host cache, the grouped GEMM plus its non-MoE neighbours span
# 330-385 ms on the compute stream, and start-to-start `period` comes out 516-550
# ms -- at PR=1 and PR=2 alike.  So **period ~= xfer span + ~50 ms**, the GEMM is
# already hidden behind the H2D, and the split only adds a second range's worth of
# launches and accumulator work.  The lever is the H2D rate, not the schedule:
# 7.65 GB in ~480 ms is 15.9 GB/s on a Gen4 x16 link that does ~26.7.
#
# History: the split was -11% before the compaction (the sanitizer-sized GEMM
# inflated 264 -> 425 ms/layer, reproduced at chunk 32768: 1379 -> 1228) and -2%
# after, i.e. compaction removed the inflation without turning the split into a
# win.  `PIPELINE_RANGES` and the compaction both stay: the compaction is what
# makes the split's cost legible, and the switch stays so the overlap can be
# re-measured if the transfer stops being the critical path.  The other upstream
# alternative -- a second whole-layer prepared slot so no split is needed -- needs
# another 6.72 GiB and does not fit (measured: 0.65 GiB free at chunk 16384 with
# the slots resident).
PIPELINE_RANGES = max(
    1, int(os.environ.get("SGLANG_MXFP4_LAYERWISE_PIPELINE_RANGES", "1"))
)
# Staging buffers per pipeline range, and the experts each holds.  Two per
# range keeps the H2D of one batch overlapped with the Marlin repack of the
# previous one (see the staging_buffers note in the layerwise manager).  The
# expert count is divided rather than the pool grown so the total staging
# footprint matches the 2 x BATCH_EXPERTS it replaces.
STAGING_BUFFERS_PER_RANGE = 2
STAGING_EXPERTS = max(1, BATCH_EXPERTS // PIPELINE_RANGES)
# Free GPU bytes required before the layerwise pipeline auto-re-enables after
# an OOM teardown (prepared 3.19 GiB + staging 0.94 GiB + ~2 GiB margin for a
# short prefill's chunk working memory).  Keeps short layerwise-qualified
# prefills from staying stuck on CPU MoE for the whole session after one
# long-context OOM.
_MXFP4_LAYERWISE_REENABLE_BYTES = int(6.5 * 1024**3)

# Diagnostic only: SGLANG_LAYERWISE_MEM_DEBUG=1 logs the torch allocator's
# free/allocated/reserved/peak around the first few layerwise applies.  This is
# how the per-batch working-memory requirement is measured -- that number is
# what decides whether a second prepared slot (the transfer/compute overlap)
# fits on the card, so it is worth being able to read it directly instead of
# inferring it from an OOM message.  Bounded by default; off in production.
_MXFP4_LAYERWISE_MEM_DEBUG = os.environ.get("SGLANG_LAYERWISE_MEM_DEBUG") == "1"
_MXFP4_LAYERWISE_MEM_DEBUG_LEFT = 32

# Diagnostic only: SGLANG_LAYERWISE_STAGE_DEBUG=1 splits one layerwise layer into
# its device-side stages -- the H2D span on the transfer stream, the trailing
# Marlin repack, the grouped GEMM span on the compute stream, and the layer
# period -- and logs them for the first few layers.  Reading the events is
# deferred to the next layer and skipped unless they have completed, so the
# measurement does not synchronize anything and does not perturb the pipeline it
# is measuring.
_MXFP4_LAYERWISE_STAGE_DEBUG = os.environ.get("SGLANG_LAYERWISE_STAGE_DEBUG") == "1"
_MXFP4_LAYERWISE_STAGE_LEFT = 60

# Whether the intra-layer pipeline is actually active is the one fact every prefill
# measurement depends on, and nothing else records it: `PIPELINE_RANGES` is read from
# the environment at import, and a leg that fails to pass it through silently measures
# the shipped single-range path twice.  Logged once, unconditionally.
_PIPELINE_RANGES_LOGGED = False


def _log_pipeline_ranges(ranges) -> None:
    """Log the effective intra-layer expert split, once per process."""
    global _PIPELINE_RANGES_LOGGED
    if _PIPELINE_RANGES_LOGGED:
        return
    _PIPELINE_RANGES_LOGGED = True
    logger.info(
        "[lwranges] PIPELINE_RANGES=%d ranges=%s staging_experts=%d staging_buffers=%d",
        PIPELINE_RANGES,
        ranges,
        STAGING_EXPERTS,
        STAGING_BUFFERS_PER_RANGE * PIPELINE_RANGES,
    )

# Per-thread slot index used by SharedStagingBuffer.get_slice (0 = main thread,
# 1 = PD-parallel second channel). Set by the channel thread. Local addition.
_STAGING_THREAD_SLOT = threading.local()


class SharedStagingBuffer:
    """Global shared staging buffer for CPU expert input across all MoE layers.

    This avoids allocating a separate staging buffer per layer, which would
    consume significant GPU memory (chunked_prefill_size * hidden_size * N_layers).
    Instead, all layers share a single buffer since MoE layers are processed
    sequentially, not in parallel.

    PD-parallel (local): the main thread's prefill and the second-channel
    decode/prefill both call the KT apply. A plain buffer[:num_tokens] slice
    overlaps between threads and the async H2D copies clobber each other, so
    the buffer is sized for TWO slots and get_slice returns a per-thread
    DISJOINT slice (thread slot 0 = main, 1 = PD channel).
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        from sglang.srt.environ import envs

        self.num_slots = 2 if envs.SGLANG_PD_PARALLEL.get() else 1
        self.buffer = torch.empty(
            (self.num_slots * max_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )
        buffer_size_mb = self.buffer.numel() * self.buffer.element_size() / 1024**2
        logger.info(
            f"[KT] Created shared staging buffer: {buffer_size_mb:.1f} MiB "
            f"(shape={self.buffer.shape}, dtype={dtype}, slots={self.num_slots})"
        )

    def get_slice(self, num_tokens: int) -> torch.Tensor:
        """Get a slice of the buffer for the given number of tokens.

        The returned slice is the calling thread's disjoint slot, so concurrent
        KT submissions from the main and PD-parallel channel threads never
        alias each other's staging region.
        """
        if num_tokens > self.max_tokens:
            raise RuntimeError(
                f"KT batch has {num_tokens} tokens but the shared staging buffer "
                f"was sized for {self.max_tokens}; increase --chunked-prefill-size."
            )
        slot = getattr(_STAGING_THREAD_SLOT, "slot", 0)
        if slot >= self.num_slots:
            raise RuntimeError(
                f"staging slot {slot} requested but buffer has {self.num_slots} "
                "slot(s); slot 1 requires SGLANG_PD_PARALLEL=1"
            )
        start = slot * self.max_tokens
        return self.buffer[start : start + num_tokens]


class KTGraphStateBridge:
    """Persistent buffers for tensors that span a KT breakable-graph seam.

    A CUDA graph records raw addresses.  DSV4 keeps several layer-local tensors
    alive across its MoE call, but those tensors are not arguments to KT's
    ``eager_on_graph`` break and therefore are invisible to the generic bridge.
    Copying them into these owner-retained buffers before the break gives both
    adjacent graph segments stable addresses.  Buffers are shared by all graph
    batch shapes for one layer and sized to the largest shape observed.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}
        # Capture sizes are normally visited largest-first.  Keep an old buffer
        # alive if that invariant ever changes, because an already captured graph
        # may still contain its address.
        self._retired_buffers: list[torch.Tensor] = []

    def copy(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            raise ValueError(f"KT graph bridge {name!r} requires a row dimension")

        buffer = self._buffers.get(name)
        compatible = (
            buffer is not None
            and buffer.device == tensor.device
            and buffer.dtype == tensor.dtype
            and buffer.shape[1:] == tensor.shape[1:]
            and buffer.shape[0] >= tensor.shape[0]
        )
        if not compatible:
            if buffer is not None:
                self._retired_buffers.append(buffer)
            buffer = tensor.new_empty(tensor.shape)
            self._buffers[name] = buffer

        view = buffer[: tensor.shape[0]]
        view.copy_(tensor)
        return view


def get_or_create_shared_staging_buffer(
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SharedStagingBuffer:
    """Get or create the global shared staging buffer."""
    global _SHARED_STAGING_BUFFER
    if _SHARED_STAGING_BUFFER is None:
        _SHARED_STAGING_BUFFER = SharedStagingBuffer(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        )
    return _SHARED_STAGING_BUFFER


def _layer_device(layer: torch.nn.Module) -> torch.device:
    """Device of `layer`, tolerating a layer whose parameters are all gone.

    `KT_PREFILL_RELEASE_RESIDENTS=1` sets a resident layer's raw weight attrs to
    None (testable via `hasattr`, which the layerwise gate depends on), which
    leaves `layer.parameters()` empty and turns a plain `.device` read into a
    StopIteration.  That has now killed a prefill (through `_apply_impl`) and the
    layerwise slot setup, and every caller here only wants a device, so fall back
    to the current CUDA device instead of propagating.
    """
    try:
        return next(layer.parameters()).device
    except StopIteration:
        return torch.device("cuda", torch.cuda.current_device())


class SharedFullContext:
    def __init__(
        self,
        layer: torch.nn.Module,
        init_args: tuple,
        global_num_experts: int,
        moe_runner_config: "MoeRunnerConfig",
        defer_cpu_buffers: bool = False,
    ):
        self._build_layers(layer, init_args, global_num_experts, moe_runner_config)

        # Capture original tensors to support restoration before loading
        self.original_params = {
            name: param for name, param in self.gpu_layer.named_parameters()
        }
        self.original_buffers = {
            name: buf for name, buf in self.gpu_layer.named_buffers()
        }

        # The MXFP4 layerwise manager defers SHM creation until every TP rank
        # has successfully allocated both GPU layer slots.  This keeps an OOM
        # on one rank from stranding peers inside an SHM collective.
        self._cpu_buffers_initialized = False
        if not defer_cpu_buffers:
            self.initialize_cpu_buffers()

        # For M3 MXFP8 layerwise prefill, cache the canonical uint8 ue8m0
        # scale Parameter objects so `_prepare_weight_mxfp8` can rebind to
        # them before each byte-copy. Native MXFP8 path keeps the scale in
        # this layout throughout (no convert to block-fp8).
        if getattr(self, "_is_mxfp8_quant", False):
            self._init_mxfp8_aux()

    def initialize_cpu_buffers(self) -> None:
        if self._cpu_buffers_initialized:
            return
        self._create_cpu_buffers()
        self._cpu_buffers_initialized = True

    def _init_mxfp8_aux(self) -> None:
        """Cache the canonical uint8 ue8m0 [E, N, K//32] scale Parameters.

        Fp8MoEMethod.create_weights (use_mxfp8=True) allocated them; we just
        save references so the shadow gpu_layer can rebind back to the same
        objects across rounds of layerwise prefill.
        """
        layer = self.gpu_layer
        self._w13_scale_mxfp8_param = layer.w13_weight_scale_inv
        self._w2_scale_mxfp8_param = layer.w2_weight_scale_inv

    def _build_layers(self, layer, init_args, global_num_experts, moe_runner_config):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        hidden_size, intermediate_size_per_partition, params_dtype = init_args
        target_device = _layer_device(layer)

        # Create gpu_layer as a shallow copy, then override specific attributes
        self.gpu_layer = copy.copy(layer)
        # Clear module state that shouldn't be shared
        self.gpu_layer._parameters = {}
        self.gpu_layer._buffers = {}
        self.gpu_layer._modules = {}

        # Override expert counts for full GPU execution
        self.gpu_layer.num_experts = global_num_experts
        self.gpu_layer.num_local_experts = global_num_experts
        self.gpu_layer.num_gpu_experts = global_num_experts

        # Create quant_method for gpu_layer
        if self.gpu_layer.quant_config is not None:
            self.gpu_method = self.gpu_layer.quant_config.get_quant_method(
                self.gpu_layer, prefix=""
            )
        else:
            self.gpu_method = UnquantizedFusedMoEMethod(
                self.gpu_layer.use_triton_kernels
            )
        # V4-Flash routed experts are MXFP4-packed (FP4 e2m1 in int8 + ue8m0
        # scales) but quant_config.get_quant_method picks Fp8MoEMethod
        # (V4-Flash uses FP8 for attn / shared experts). The default
        # mxfp4_deepseek pipeline wraps Fp8MoEMethod with DeepSeekMxfp4MoEMethod
        # for V4 routed experts; replicate that wrap here so kt_ep_wrapper's
        # gpu_method correctly handles MXFP4 (shape K = hidden, not hidden/2
        # as the FP8 path assumes), and so the capability-driven dispatch in
        # mxfp4_deepseek.apply (trtllm vs triton_kernels) fires on the
        # wrapped path. Gate by `--kt-method MXFP4` (an explicit user choice
        # that means "this run uses MXFP4 routed experts on the kt path"),
        # NOT by SGLANG_V4_USE_TRITON_KERNELS (which is now a diagnostic
        # override only, see v4_triton_kernels_moe.use_v4_triton_kernels
        # docstring). Origin: kt-sglang 耦合 (V4-Flash routed experts MXFP4
        # detection in kt_ep_wrapper).
        try:
            import os as _os_v4

            from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
            from sglang.srt.layers.quantization.mxfp4_deepseek import (
                DeepSeekMxfp4MoEMethod,
            )
            from sglang.srt.runtime_context import get_exec
            _v4_env = _os_v4.environ.get("SGLANG_V4_USE_TRITON_KERNELS")
            if _v4_env == "1":
                _do_v4_wrap = True
            elif _v4_env == "0":
                _do_v4_wrap = False
            else:
                _do_v4_wrap = (
                    (get_exec().moe.kt_method or "").upper() == "MXFP4"
                )
            if _do_v4_wrap and isinstance(self.gpu_method, Fp8MoEMethod):
                self.gpu_method = DeepSeekMxfp4MoEMethod(self.gpu_method, prefix="")
        except Exception as _v4_tk_wrap_exc:
            logger.warning(
                f"[kt-ep-wrapper] V4-Flash MXFP4 wrap skipped: {_v4_tk_wrap_exc}"
            )
        self.gpu_layer.quant_method = self.gpu_method

        self.gpu_method.create_weights(
            layer=self.gpu_layer,
            num_experts=global_num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
        )

        # Detect quantization type for weight loading based on actually created weights.
        # This is more robust than class-based detection when quant methods are wrapped
        # (e.g., KT wrapper -> compressed-tensors scheme), especially in layerwise prefill.
        # Run once at init.  Freeze the result into _is_* so downstream
        # always sees the original quant type even if a Marlin repack
        # later renames attributes (e.g. _inv → _weight_scale).
        self._detect_quant_type_from_created_weights()
        for _attr in ("is_mxfp4_quant", "is_mxfp8_quant", "is_fp8_quant",
                       "is_fp8_channel_quant", "is_bf16_quant"):
            if hasattr(self, _attr):
                setattr(self, f"_{_attr}", getattr(self, _attr))

        # Move all parameters to target device
        for param in self.gpu_layer.parameters():
            if param.device != target_device:
                param.data = param.data.to(target_device)

        # Save weight/scale shapes after device move so _restore_raw_attrs
        # can re-create tensors on the correct device later.
        self._raw_weight_shapes = {}
        for _sn in self.weight_names:
            if hasattr(self.gpu_layer, _sn):
                _t = getattr(self.gpu_layer, _sn)
                self._raw_weight_shapes[_sn] = (tuple(_t.shape), _t.dtype, _t.device)

        # Create runner config - update both num_experts and num_local_experts for full GPU fallback
        # Set routed_scaling_factor=None to avoid double scaling:
        # - moe_sum_reduce would apply routed_scaling_factor internally
        # - deepseek_v2.py forward_normal also applies routed_scaling_factor for KTEPWrapperMethod
        # By setting it to None here, we ensure it's only applied once in forward_normal
        runner_config = replace(
            moe_runner_config,
            num_experts=global_num_experts,
            num_local_experts=global_num_experts,
            routed_scaling_factor=None,
        )
        self.gpu_layer.moe_runner_config = runner_config
        self.gpu_method.create_moe_runner(self.gpu_layer, runner_config)

    def _get_base_quant_method(self):
        """Unwrap nested quant methods to get the underlying base method.

        Some paths may wrap the real quant method with KT wrappers/schemes.
        """
        method = self.gpu_method
        visited = set()

        while method is not None and id(method) not in visited:
            visited.add(id(method))

            # KT wrapper pattern: method.gpu_method
            nested = getattr(method, "gpu_method", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            # Compressed-tensors scheme pattern: method.scheme
            nested = getattr(method, "scheme", None)
            if nested is not None and nested is not method:
                method = nested
                continue

            break

        return method

    def _detect_quant_type_from_created_weights(self) -> None:
        """Detect quant type from weight attributes created on gpu_layer."""
        layer = self.gpu_layer

        # V4-Flash MXFP4 (must come before FP8 block — both register
        # `w13_weight_scale_inv`, but MXFP4 is FP4 nibble-packed weights with
        # ue8m0 scales rather than FP8 e4m3 weights with FP8 scales). Use the
        # quant method's class name as discriminator to avoid a circular import
        # of DeepSeekMxfp4MoEMethod. Origin: sglang 本身 (V4-Flash full-GPU
        # prefill fallback compat).
        if self.gpu_method.__class__.__name__ == "DeepSeekMxfp4MoEMethod":
            self.is_mxfp4_quant = True
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # INT4 Marlin
        if hasattr(layer, "w13_weight_packed") and hasattr(layer, "w2_weight_packed"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # M3 MXFP8 block (must come before FP8 block — both register
        # `w13_weight_scale_inv`, but MXFP8 stores uint8 ue8m0 scales with
        # block_size=[1,32] while FP8 block uses fp32 scales with [128,128]).
        # The `format_ue8m0` attribute set by Fp8MoEMethod.create_weights
        # when use_mxfp8=True (fp8.py:914) is the canonical discriminator.
        # Origin: kt-sglang 耦合 (M3 MXFP8 layerwise prefill).
        if (
            hasattr(layer, "w13_weight_scale_inv")
            and hasattr(layer, "w2_weight_scale_inv")
            and getattr(layer.w13_weight_scale_inv, "format_ue8m0", False)
        ):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = True
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 block
        if hasattr(layer, "w13_weight_scale_inv") and hasattr(layer, "w2_weight_scale_inv"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = True
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = False
            return

        # FP8 per-channel
        if hasattr(layer, "w13_weight_scale") and hasattr(layer, "w2_weight_scale"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = True
            self.is_bf16_quant = False
            return

        # BF16 / unquantized
        if hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight"):
            self.is_mxfp4_quant = False
            self.is_mxfp8_quant = False
            self.is_fp8_quant = False
            self.is_fp8_channel_quant = False
            self.is_bf16_quant = True
            return

        # Fallback to class-based detection for unknown layouts.
        self.is_mxfp4_quant = False
        self.is_mxfp8_quant = False
        self.is_fp8_quant = self._detect_fp8_quant()
        self.is_fp8_channel_quant = self._detect_fp8_channel_quant()
        self.is_bf16_quant = self._detect_bf16_quant()

    def _detect_fp8_quant(self) -> bool:
        """Detect if the quantization method is FP8 block quant.

        Returns:
            True if FP8 block quant, False otherwise (INT4 Marlin, BF16, etc.)
        """
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        method = self._get_base_quant_method()
        # Check for Fp8MoEMethod with block_quant
        if isinstance(method, Fp8MoEMethod) and getattr(method, "block_quant", False):
            return True

        # Check for CompressedTensorsW8A8Fp8MoEMethod with block_quant
        method_name = method.__class__.__name__
        if "W8A8Fp8" in method_name and getattr(method, "block_quant", False):
            return True

        return False

    def _detect_fp8_channel_quant(self) -> bool:
        """Detect if the quantization method is FP8 per-channel quant.

        Per-channel FP8 differs from block FP8:
        - Per-channel: scale shape is (num_experts, output_dim, 1), weight_scale name
        - Block FP8: scale shape is (num_experts, blocks_n, blocks_k), weight_scale_inv name

        Returns:
            True if FP8 per-channel quant, False otherwise
        """
        try:
            from compressed_tensors.quantization import QuantizationStrategy
        except ImportError:
            return False

        method = self._get_base_quant_method()
        method_name = method.__class__.__name__

        # Check for CompressedTensorsW8A8Fp8MoEMethod with channel strategy
        if "W8A8Fp8" in method_name:
            weight_quant = getattr(method, "weight_quant", None)
            if weight_quant is not None:
                if weight_quant.strategy == QuantizationStrategy.CHANNEL:
                    return True

        return False

    def _detect_bf16_quant(self) -> bool:
        """Detect if the quantization method is BF16/unquantized.

        Returns:
            True if BF16/unquantized, False otherwise (INT4 Marlin, FP8, etc.)
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        method = self._get_base_quant_method()
        # Check for UnquantizedFusedMoEMethod
        if isinstance(method, UnquantizedFusedMoEMethod):
            return True

        return False

    def _resolve_int4_quant_params(self):
        """Resolve INT4 quant params from potentially wrapped quant methods.

        Some quantization paths (e.g., compressed-tensors) expose INT4 metadata on
        the underlying scheme instead of the outer fused method wrapper.
        """
        candidates = []
        seen = set()

        def add_candidate(obj):
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            candidates.append(obj)

        base_method = self._get_base_quant_method()
        add_candidate(self.gpu_method)
        add_candidate(getattr(self.gpu_method, "gpu_method", None))
        add_candidate(getattr(self.gpu_method, "scheme", None))
        add_candidate(base_method)
        add_candidate(getattr(base_method, "scheme", None))
        add_candidate(getattr(self.gpu_layer, "scheme", None))

        required = ("num_bits", "packed_factor", "group_size")
        for candidate in candidates:
            if all(hasattr(candidate, attr) for attr in required):
                return (
                    getattr(candidate, "num_bits"),
                    getattr(candidate, "packed_factor"),
                    getattr(candidate, "group_size"),
                    getattr(candidate, "actorder", None),
                )

        raise AttributeError(
            "Unable to resolve INT4 quantization params: expected attributes "
            "num_bits/packed_factor/group_size on quant method or scheme"
        )

    @property
    def weight_names(self) -> list:
        """Get weight names based on quantization type."""
        if getattr(self, "_is_mxfp4_quant", False):
            # V4-Flash MXFP4 uses the same flat names as FP8 block (w13_weight,
            # w13_weight_scale_inv, w2_weight, w2_weight_scale_inv); the
            # underlying byte payload differs (FP4 nibble + ue8m0 scale) but
            # the staging buffers don't care about content.
            return self.WEIGHT_NAMES_FP8
        if getattr(self, "_is_mxfp8_quant", False):
            # M3 MXFP8 reuses the FP8 block flat names. Byte payload is
            # MXFP8 (fp8 + uint8 ue8m0 [1,32]) — staging buffer dtype/shape
            # follow gpu_layer.w13_weight_scale_inv (uint8) so byte-copy
            # transports the canonical layout. fused_experts_mxfp8 consumes
            # it directly; no convert step.
            return self.WEIGHT_NAMES_FP8
        if self._is_fp8_quant:
            return self.WEIGHT_NAMES_FP8
        elif self._is_fp8_channel_quant:
            return self.WEIGHT_NAMES_FP8_CHANNEL
        elif self._is_bf16_quant:
            return self.WEIGHT_NAMES_BF16
        else:
            return self.WEIGHT_NAMES_INT4

    # Weight names for shared memory buffers (INT4 Marlin format)
    WEIGHT_NAMES_INT4 = [
        "w13_weight_packed",
        "w13_weight_scale",
        "w2_weight_packed",
        "w2_weight_scale",
    ]

    # Weight names for FP8 block quant format
    WEIGHT_NAMES_FP8 = [
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    ]

    # Weight names for FP8 per-channel quant format
    # Per-channel differs from block quant:
    # - Scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
    # - Weight name: w13_weight_scale vs w13_weight_scale_inv
    WEIGHT_NAMES_FP8_CHANNEL = [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ]

    # Weight names for BF16/unquantized format (no scales)
    WEIGHT_NAMES_BF16 = [
        "w13_weight",
        "w2_weight",
    ]

    def _cleanup_cpu_buffers_after_failure(self) -> None:
        """Best-effort symmetric cleanup for a failed SHM setup phase."""

        if torch.cuda.is_available():
            for tensor in getattr(self, "_registered_host_buffers", []):
                try:
                    torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())
                except Exception:
                    pass
        self._registered_host_buffers = []

        # Drop exported torch buffers before closing their SharedMemory maps.
        self.cpu_buffers = {}
        gc.collect()

        for shm in getattr(self, "_opened_shm_refs", {}).values():
            try:
                shm.close()
            except Exception:
                pass
        self._opened_shm_refs = {}

        for shm in getattr(self, "shm_handles", {}).values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            try:
                shm.close()
            except Exception:
                pass
        self.shm_handles = {}

    def _commit_cpu_buffer_phase(
        self, local_error: Optional[Exception], phase: str
    ) -> None:
        """Make every TP rank commit or abort an SHM initialization phase."""

        if _all_tp_ranks_succeeded(local_error is None):
            return

        self._cleanup_cpu_buffers_after_failure()
        message = f"KT shared-memory {phase} failed on at least one TP rank"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _create_cpu_buffers(self):
        """Create CPU buffers in POSIX shared memory and register as pinned memory.

        Uses double buffering (2 experts) to reduce memory usage while maintaining
        pipeline efficiency: write(e+1) || copy(e) only needs 2 buffers.
        """
        self.cpu_buffers = {}
        self.shm_handles: Dict[str, shared_memory.SharedMemory] = {}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}
        self._registered_host_buffers: List[torch.Tensor] = []
        tp_rank = get_tensor_model_parallel_rank()
        num_experts = self.gpu_layer.num_experts

        # No rank may enter a later SHM phase until every peer has completed
        # the current one.  In particular, a local libnuma/SHM/register error
        # must not leave peers waiting forever in a barrier.
        numa_error = None
        try:
            libnuma = ctypes.CDLL("libnuma.so.1")
            if libnuma.numa_available() < 0:
                raise RuntimeError("NUMA is not available on this system")
            libnuma.numa_set_localalloc()
        except Exception as exc:
            numa_error = exc
        self._commit_cpu_buffer_phase(numa_error, "NUMA setup")

        # Generate unique ID on rank 0 and broadcast to all ranks
        if tp_rank == 0:
            self.shm_unique_id = uuid.uuid4().hex[:8]
        else:
            self.shm_unique_id = None
        if dist.is_initialized():
            unique_id_list = [self.shm_unique_id]
            dist.broadcast_object_list(
                unique_id_list,
                src=get_tp_group().first_rank,
                group=get_tp_group().cpu_group,
            )
            self.shm_unique_id = unique_id_list[0]

        allocation_error = None
        try:
            for name in self.weight_names:
                # Size from the frozen snapshot: the full-layer raw may already
                # be freed (see _allocate_mxfp4_slot_storage) and would read as
                # an empty Parameter, collapsing the SHM to 0 bytes.
                _shape, _w_dtype, _w_device = self._raw_weight_shapes[name]
                expert_shape = _shape[1:]
                if (
                    getattr(self, "_is_mxfp4_quant", False)
                    and name in ("w13_weight_scale_inv", "w2_weight_scale_inv")
                ):
                    buf_dtype = torch.bfloat16
                else:
                    buf_dtype = _w_dtype
                element_size = torch.empty((), dtype=buf_dtype).element_size()
                _numel = 1
                for _d in _shape:
                    _numel *= _d
                expert_nbytes = _numel // num_experts * element_size
                double_buf_nbytes = expert_nbytes * 2

                shm_name = f"kt_buf_{name}_r{tp_rank}_{self.shm_unique_id}"
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=double_buf_nbytes
                )
                self.shm_handles[name] = shm

                # Shape: [2, ...expert_shape...]
                cpu_buffer = torch.frombuffer(shm.buf, dtype=buf_dtype).reshape(
                    (2,) + expert_shape
                )

                # Register as pinned memory for fast DMA
                if torch.cuda.is_available():
                    register_result = torch.cuda.cudart().cudaHostRegister(
                        cpu_buffer.data_ptr(), double_buf_nbytes, 0
                    )
                    if int(register_result) != 0:
                        raise RuntimeError(
                            "cudaHostRegister failed for "
                            f"{name} with error code {int(register_result)}"
                        )
                    self._registered_host_buffers.append(cpu_buffer)

                self.cpu_buffers[name] = cpu_buffer
        except Exception as exc:
            allocation_error = exc
        self._commit_cpu_buffer_phase(allocation_error, "allocation")

        pointer_error = None
        try:
            self.all_rank_buffer_ptrs = self._collect_all_rank_buffer_pointers()
            if tp_rank == 0:
                tp_world_size = get_tensor_model_parallel_world_size()
                valid = all(
                    len(ptrs) == tp_world_size and all(ptr > 0 for ptr in ptrs)
                    for ptrs in self.all_rank_buffer_ptrs.values()
                )
                if not valid:
                    raise RuntimeError(
                        "TP0 could not map every rank's shared-memory buffer"
                    )
        except Exception as exc:
            pointer_error = exc
        self._commit_cpu_buffer_phase(pointer_error, "pointer collection")

        # Unlink shared memory after all ranks have collected pointers.
        # The memory remains accessible as long as we hold references via mmap.
        for shm in self.shm_handles.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def _collect_all_rank_buffer_pointers(self) -> Dict[str, List[int]]:
        """Collect CPU buffer pointers from all ranks."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        buffer_names = list(self.cpu_buffers.keys())
        all_rank_ptrs: Dict[str, List[int]] = {name: [] for name in buffer_names}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}

        for rank in range(tp_world_size):
            for name in buffer_names:
                if rank == tp_rank:
                    ptr = self.cpu_buffers[name].data_ptr()
                elif tp_rank == 0:
                    shm_name = f"kt_buf_{name}_r{rank}_{self.shm_unique_id}"
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        self._opened_shm_refs[f"{name}_r{rank}"] = shm
                        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                    except Exception:
                        logger.error(
                            "Rank %d: Failed to open shared memory '%s'",
                            tp_rank,
                            shm_name,
                        )
                        ptr = 0
                else:
                    ptr = 0
                all_rank_ptrs[name].append(ptr)

        return all_rank_ptrs

    def _prepare_weight_int4(self, wrapper):
        """Prepare INT4 Marlin weights by writing from KT, copying to GPU, and postprocessing.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        Postprocessing extracted from CompressedTensorsWNA16MoEMethod.process_weights_after_loading
        in python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_bits, packed_factor, group_size, actorder = (
            self._resolve_int4_quant_params()
        )
        num_experts = layer.num_experts
        device = layer.w13_weight_packed.device

        # Create empty g_idx tensors for non-grouped actorder
        if actorder != "group":
            for name in [
                "w13_weight_g_idx",
                "w2_weight_g_idx",
                "w13_g_idx_sort_indices",
                "w2_g_idx_sort_indices",
            ]:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(
                        torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                        requires_grad=False,
                    ),
                )

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_INT4:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            # Reshape gpu_t to match expert shape for per-expert copy
            expert_shape = cpu_buf.shape[1:]
            gpu_t.set_(gpu_t.view((num_experts,) + expert_shape))
            weight_infos.append((cpu_buf, gpu_t))

        w13_p, w13_s = layer.w13_weight_packed, layer.w13_weight_scale
        w2_p, w2_s = layer.w2_weight_packed, layer.w2_weight_scale
        w13_k, w13_n = w13_p.shape[1] * packed_factor, w13_p.shape[2]
        w2_k, w2_n = w2_p.shape[1] * packed_factor, w2_p.shape[2]
        w2_sk = w2_s.shape[1] * (group_size if group_size != -1 else packed_factor)
        perm = torch.empty(0, dtype=torch.int32, device=device)

        # Tmp buffers for transpose
        tmp_bufs = [
            torch.empty(t.size(1), t.size(2), dtype=t.dtype, device=device)
            for _, t in weight_infos
        ]

        def postprocess_expert(e):
            # Transpose
            for (_, gpu_t), tmp in zip(weight_infos, tmp_bufs):
                d1, d2 = gpu_t.size(1), gpu_t.size(2)
                tmp.copy_(gpu_t[e].reshape(d2, d1).T, non_blocking=True)
                gpu_t[e].copy_(tmp, non_blocking=True)
            # Repack weights
            w13_p[e].copy_(
                gptq_marlin_repack(w13_p[e], perm, w13_k, w13_n, num_bits).view(
                    w13_p[e].shape
                )
            )
            w2_p[e].copy_(
                gptq_marlin_repack(w2_p[e], perm, w2_k, w2_n, num_bits).view(
                    w2_p[e].shape
                )
            )
            # Permute scales
            w13_s[e].copy_(
                marlin_permute_scales(w13_s[e], w13_n, w13_s.shape[2], group_size).view(
                    w13_s[e].shape
                )
            )
            w2_s[e].copy_(
                marlin_permute_scales(w2_s[e], w2_sk, w2_s.shape[2], group_size).view(
                    w2_s[e].shape
                )
            )

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_packed_buf = self.cpu_buffers["w13_weight_packed"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_packed_buf = self.cpu_buffers["w2_weight_packed"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_packed_expert_nbytes = (
                w13_packed_buf.numel() // 2 * w13_packed_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_packed_expert_nbytes = (
                w2_packed_buf.numel() // 2 * w2_packed_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_packed_ptrs = [
                    ptr + slot * w13_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_packed"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_packed_ptrs = [
                    ptr + slot * w2_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_packed"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_packed_ptrs,
                    w13_scale_ptrs,
                    w2_packed_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    # Before writing to slot (e+1)%2, make sure the previous
                    # copy from that slot has completed to avoid overwriting
                    # pinned host memory while DMA is in-flight.
                    if e > 0:
                        events[e - 1].synchronize()
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

        # Reshape to final shape
        w13_p.set_(w13_p.view(num_experts, w13_k // 16, w13_n * (num_bits // 2)))
        w2_p.set_(w2_p.view(num_experts, w2_k // 16, w2_n * (num_bits // 2)))

    def _prepare_weight_fp8(self, wrapper, original_layer=None, gpu_experts_mask=None,
                            logical_to_gpu_index=None):
        """Prepare FP8 block quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 block quant is simpler than INT4 Marlin:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optional DeepGemm ue8m0 conversion is handled after all experts are loaded.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            # ONE host readback for the whole mask.  The per-expert `gpu_experts_mask[e].item()`
            # this replaces is a synchronizing DtoH, so the loop cost `num_experts` pipeline drains
            # per layer -- and this path runs once per layer in the layerwise prefill, i.e. 384 x 40
            # = 15360 drains per chunk (at --kt-num-gpu-experts 0 the mask is all-False, so every
            # one of them is a wasted sync).  Same values, one sync.
            mask_host = gpu_experts_mask.cpu().tolist()
            gpu_expert_ids = [e for e, m in enumerate(mask_host) if m]
            cpu_expert_ids = [e for e in range(num_experts) if not mask_host[e]]
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            # One readback, then index it: `logical_to_gpu_index[e].item()` inside the loop is a
            # synchronizing DtoH per GPU expert.
            gpu_index_host = logical_to_gpu_index.cpu().tolist()
            for e in gpu_expert_ids:
                gpu_idx = gpu_index_host[e]
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale_inv"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale_inv"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    # NOTE: DeepGemm ue8m0 conversion is not used in KT fallback path.
    # The conversion is handled separately in the normal weight loading path.

    def _prepare_weight_mxfp8(self, wrapper, original_layer=None, gpu_experts_mask=None,
                              logical_to_gpu_index=None):
        """Byte-copy M3 MXFP8 weights from CPU staging buffer to GPU for the
        full-GPU layerwise prefill fallback.

        Shadow ``gpu_method`` stays in the MXFP8 view (``use_mxfp8=True``,
        ``weight_block_size=[1, 32]``). ``Fp8MoEMethod.apply`` then routes
        through ``get_triton_quant_info`` -> ``fused_experts_mxfp8``, which
        consumes the uint8 ue8m0 scale directly via ``tl.dot_scaled`` — no
        block-FP8 conversion, no precision loss.

        Origin: kt-sglang 耦合 (M3 MXFP8 layerwise prefill, native MXFP8).
        """
        # Reset shadow to MXFP8 view (idempotent; ensures the Parameter slot
        # points at the canonical uint8 ue8m0 tensor before byte-copy).
        self.gpu_method.use_mxfp8 = True
        self.gpu_method.weight_block_size = [1, 32]
        self.gpu_layer.w13_weight_scale_inv = self._w13_scale_mxfp8_param
        self.gpu_layer.w2_weight_scale_inv = self._w2_scale_mxfp8_param

        # Byte-copy via the FP8 pipeline (uint8 ue8m0 scale + fp8 weight
        # both copied bytewise from kt-kernel CPU staging buffer).
        # original_layer=None disables the GPU shortcut: the real layer's
        # scale slot may have been mutated by Fp8MoEMethod's post-load step
        # on other paths; force the CPU staging route for canonical bytes.
        self._prepare_weight_fp8(
            wrapper,
            original_layer=None,
            gpu_experts_mask=None,
            logical_to_gpu_index=None,
        )

    def _prepare_weight_mxfp4(self, wrapper, original_layer=None, gpu_experts_mask=None,
                              logical_to_gpu_index=None):
        """Prepare V4-Flash MXFP4 weights for the full-GPU prefill fallback.

        V4-Flash MXFP4 routed-experts share flat attribute names with FP8 block
        (`w13_weight` / `w13_weight_scale_inv` / `w2_weight` / `w2_weight_scale_inv`)
        but with different payload semantics: FP4 e2m1 nibble-packed weights +
        ue8m0 per-kgroup scales instead of FP8 e4m3 weights + FP8 scales. The
        staging-buffer byte-copy machinery in `_prepare_weight_fp8` does not
        care about content semantics, so we reuse it as-is for the 144 GPU +
        112 CPU expert load.

        After all 256 experts are filled into `gpu_layer.w13_weight` etc., we
        re-run `gpu_method.process_weights_after_loading(gpu_layer)`, which
        invokes `convert_v4_weights_to_triton_kernels` and stores the swizzled
        result in `gpu_layer._v4_tk_w13` / `_v4_tk_w13_pcg` / `_v4_tk_w2` /
        `_v4_tk_w2_pcg` — exactly what the downstream `gpu_method.apply` →
        `apply_v4_triton_kernels_moe` path expects to read. This requires the
        outer model loader to have skipped the post-swizzle deletes (gated on
        `kt_gpu_prefill_token_threshold > 0` in `mxfp4_deepseek.py`).

        **No caching**: SharedFullContext is a global singleton whose single
        `gpu_layer` holds one layer's swizzled weights at a time. After layer N
        loads, layer N-1's data is overwritten. A boolean or per-layer-set
        cache would be stale when a different layer has since loaded into the
        same gpu_layer. Every load() call must therefore run the full pipeline.

        Origin: sglang 本身 (V4-Flash full-GPU prefill fallback compat).
        """
        # Phase 1+2: byte-copy via the FP8 path (works for FP4-packed bytes).
        self._prepare_weight_fp8(
            wrapper,
            original_layer=original_layer,
            gpu_experts_mask=gpu_experts_mask,
            logical_to_gpu_index=logical_to_gpu_index,
        )

        # Phase 3: re-swizzle the now-256-expert flat tensors into the
        # triton_kernels form `gpu_method.apply` will consume. Ensure all
        # in-flight CPU→GPU copies from Phase 2 are visible first.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.gpu_method.process_weights_after_loading(self.gpu_layer)

    def _prepare_weight_fp8_channel(self, wrapper, original_layer=None, gpu_experts_mask=None,
                                     logical_to_gpu_index=None):
        """Prepare FP8 per-channel quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 per-channel quant differs from FP8 block quant:
        - Per-channel scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
        - Weight name: w13_weight_scale vs w13_weight_scale_inv
        - Both use float8_e4m3fn weights

        Similar to block FP8:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8_CHANNEL:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            # ONE host readback for the whole mask.  The per-expert `gpu_experts_mask[e].item()`
            # this replaces is a synchronizing DtoH, so the loop cost `num_experts` pipeline drains
            # per layer -- and this path runs once per layer in the layerwise prefill, i.e. 384 x 40
            # = 15360 drains per chunk (at --kt-num-gpu-experts 0 the mask is all-False, so every
            # one of them is a wasted sync).  Same values, one sync.
            mask_host = gpu_experts_mask.cpu().tolist()
            gpu_expert_ids = [e for e, m in enumerate(mask_host) if m]
            cpu_expert_ids = [e for e in range(num_experts) if not mask_host[e]]
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            # One readback, then index it: `logical_to_gpu_index[e].item()` inside the loop is a
            # synchronizing DtoH per GPU expert.
            gpu_index_host = logical_to_gpu_index.cpu().tolist()
            for e in gpu_expert_ids:
                gpu_idx = gpu_index_host[e]
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # FP8 per-channel doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _prepare_weight_bf16(self, wrapper, original_layer=None, gpu_experts_mask=None,
                             logical_to_gpu_index=None):
        """Prepare BF16/unquantized weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        BF16/unquantized is similar to FP8 block quant:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)
        - No scales at all (unlike FP8 which has scale_inv)

        The postprocess stage is a no-op for BF16 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: If original_layer and gpu_experts_mask are provided, experts
        already on GPU are copied directly (fast GPU-to-GPU), while CPU experts
        use the KT wrapper pipeline.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_BF16:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((name, cpu_buf, gpu_t))

        # Separate GPU experts (direct copy) from CPU experts (KT transfer)
        gpu_expert_ids = []
        cpu_expert_ids = []
        if gpu_experts_mask is not None and original_layer is not None and logical_to_gpu_index is not None:
            # ONE host readback for the whole mask.  The per-expert `gpu_experts_mask[e].item()`
            # this replaces is a synchronizing DtoH, so the loop cost `num_experts` pipeline drains
            # per layer -- and this path runs once per layer in the layerwise prefill, i.e. 384 x 40
            # = 15360 drains per chunk (at --kt-num-gpu-experts 0 the mask is all-False, so every
            # one of them is a wasted sync).  Same values, one sync.
            mask_host = gpu_experts_mask.cpu().tolist()
            gpu_expert_ids = [e for e, m in enumerate(mask_host) if m]
            cpu_expert_ids = [e for e in range(num_experts) if not mask_host[e]]
        else:
            # Fallback: all experts from CPU
            cpu_expert_ids = list(range(num_experts))

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU) ---
        if gpu_expert_ids:
            # One readback, then index it: `logical_to_gpu_index[e].item()` inside the loop is a
            # synchronizing DtoH per GPU expert.
            gpu_index_host = logical_to_gpu_index.cpu().tolist()
            for e in gpu_expert_ids:
                gpu_idx = gpu_index_host[e]
                for name, _, dst in weight_infos:
                    src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                    dst[e].copy_(src[gpu_idx], non_blocking=True)

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not cpu_expert_ids:
            # All experts are on GPU, nothing more to do
            return

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
            # BF16 doesn't need actual postprocessing (no repack/permute/transpose).
            # This function provides a pipeline synchronization point and
            # can be extended for future BF16-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                # For BF16, we pass empty scale pointer lists (no scales)
                w13_scale_ptrs = [0] * tp_world_size
                w2_scale_ptrs = [0] * tp_world_size
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _restore_raw_attrs(self):
        """Restore ctx.gpu_layer weight/scale attributes to raw (pre-repack) format.

        After Marlin repack, attribute shapes and dtypes change (fp8→int32)
        and _weight_scale_inv is renamed to _weight_scale.  This helper
        reverts those changes so downstream code always sees raw format.
        """
        for _stale in ("w13_weight_scale", "w2_weight_scale"):
            if hasattr(self.gpu_layer, _stale):
                delattr(self.gpu_layer, _stale)
        for _sn, (_shape, _dtype, _device) in self._raw_weight_shapes.items():
            _cur = getattr(self.gpu_layer, _sn, None)
            if _cur is None or _cur.shape != _shape or _cur.dtype != _dtype:
                setattr(self.gpu_layer, _sn,
                        torch.nn.Parameter(
                            torch.empty(_shape, dtype=_dtype, device=_device)))
    def load(self, layer_idx, wrapper, original_layer=None, gpu_experts_mask=None,
             logical_to_gpu_index=None):
        """Load weights from disk to GPU via shared memory.

        Args:
            layer_idx: Layer index in the model
            wrapper: KT wrapper for CPU expert weight loading
            original_layer: Original MoE layer with GPU experts (optional)
            gpu_experts_mask: bool tensor [num_experts], True = on GPU (optional)
            logical_to_gpu_index: int tensor [num_experts], maps logical ID to GPU index (optional)
        """
        for name, param in self.original_params.items():
            setattr(self.gpu_layer, name, param)
        for name, buf in self.original_buffers.items():
            self.gpu_layer.register_buffer(name, buf)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tp_rank = get_tensor_model_parallel_rank()
        t0 = time.perf_counter()

        # Restore raw format before loading new layer's weights
        self._restore_raw_attrs()

        # Select appropriate prepare_weight method based on quantization type
        # FP8/BF16 methods support GPU expert optimization; INT4 uses full CPU pipeline
        if getattr(self, "_is_mxfp4_quant", False):
            # V4-Flash MXFP4: byte-copy via FP8 path + re-swizzle into
            # triton_kernels form. Origin: sglang 本身.
            if _KT_RESIDENTS_RAW_RELEASED:
                # The resident experts' raw slice was given up to the layerwise slot
                # (KT_PREFILL_RELEASE_RESIDENTS).  Phase 1 would read it to copy the
                # residents D2D, so route EVERY expert through the CPU staging path
                # instead -- that is the `original_layer=None` branch, and Phase 2
                # already pulls all 384 experts from the KT host pool.
                self._prepare_weight_mxfp4(wrapper, None, None, None)
            else:
                self._prepare_weight_mxfp4(wrapper, original_layer, gpu_experts_mask,
                                           logical_to_gpu_index)
        elif getattr(self, "_is_mxfp8_quant", False):
            # M3 MXFP8: byte-copy via FP8 path with original_layer=None
            # (Phase 1 shortcut disabled) + Triton MXFP8->block-FP8 convert
            # on shadow gpu_layer so apply() runs the standard block-FP8
            # deep_gemm path. Origin: kt-sglang 耦合 (v2 bridge).
            self._prepare_weight_mxfp8(wrapper, original_layer, gpu_experts_mask,
                                       logical_to_gpu_index)
        elif self._is_fp8_quant:
            # When the inference layer is Marlin-repacked (int32), the
            # raw fp8 context layer can't share GPU→GPU copies.
            # Disable Phase 1 shortcut by passing original_layer=None.
            self._prepare_weight_fp8(wrapper, None, gpu_experts_mask,
                                     logical_to_gpu_index)
        elif self._is_fp8_channel_quant:
            self._prepare_weight_fp8_channel(wrapper, None, gpu_experts_mask,
                                             logical_to_gpu_index)
        elif self._is_bf16_quant:
            self._prepare_weight_bf16(wrapper, original_layer, gpu_experts_mask,
                                      logical_to_gpu_index)
        else:
            # INT4 Marlin format: write(e+1) || copy(e) || postprocess(e-1)
            self._prepare_weight_int4(wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = (time.perf_counter() - t0) * 1000.0

        if tp_rank == 0:
            logger.info(
                "KT layerwise prefill: layer %d prepare weight = %.2f ms",
                layer_idx,
                total_time,
            )


class _Mxfp4PrefillSlot:
    """One complete MXFP4 layer image used by the layerwise prefill pipeline."""

    RAW_NAMES = (
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    )

    def __init__(
        self,
        index: int,
        raw_tensors: Dict[str, torch.Tensor],
        marlin_prepared,
        num_experts: Optional[int] = None,
        *,
        frozen_shapes: Dict[str, tuple],
    ):
        self.index = index
        for name in self.RAW_NAMES:
            setattr(self, name, raw_tensors[name])
        self.marlin_prepared = marlin_prepared

        self.state = "EMPTY"
        self.layer_idx: Optional[int] = None
        self.epoch = -1
        self.has_consumed_event = False
        self.reuse_guard = None

        self.raw_ready_event = torch.cuda.Event()
        self.ready_event = torch.cuda.Event()
        self.consumed_event = torch.cuda.Event()
        # Per-range pipeline fences (see PIPELINE_RANGES).  range_ready_event[r]
        # is recorded on the postprocess stream when range r's Marlin repack
        # lands; the compute stream waits on it before computing that range.
        self.range_ready_event = [
            torch.cuda.Event() for _ in range(PIPELINE_RANGES)
        ]
        # The expert ranges ([start, count) pairs) the cache path pipelines, or
        # None for a single whole-layer range -- the SHM path, which writes the
        # raw slot directly and has no repack to overlap.  Set by _load_slot,
        # together with the host cache and GPU-expert list the later ranges are
        # repacked from in apply().
        self.pipeline_ranges: Optional[List[tuple]] = None
        self.layer_cache: Optional[Dict[str, torch.Tensor]] = None
        self.range_gpu_expert_ids: List[int] = []

        # Sizes come from the context's frozen snapshot, never from the live
        # tensors: with SGLANG_KT_SLOT_FREE_RAW_FIRST the full-layer raw is
        # freed before the manager is built, and its placeholder .shape is (0,)
        # -- reading shape[2] off that was the manager-setup IndexError that
        # killed the first layerwise prefill attempt.
        self.intermediate_size = frozen_shapes["w2_weight"][0][2] * 2
        if num_experts is not None:
            self.num_experts = num_experts
        else:
            self.num_experts = frozen_shapes["w13_weight"][0][0]

    def invalidate(self) -> None:
        # Keep the prepared tensors alive.  Their backing may still be in use on
        # the compute stream; the consumed event protects the next overwrite.
        self.state = "EMPTY"
        self.layer_idx = None
        self.epoch = -1


class _Mxfp4LayerwisePrefillManager:
    """Persistent two-slot MXFP4 layer-to-layer prefill scheduler.

    GPU work is ordered exclusively with stream events.  Python launches the
    current layer first, then performs the successor's KT host write and GPU
    enqueue work while the current layer is already running on the main stream.
    """

    def __init__(
        self,
        context: SharedFullContext,
        signature: tuple,
        slot1_raw_tensors: Optional[Dict[str, torch.Tensor]],
        slot_marlin_prepared: tuple,
    ):
        self.context = context
        self.signature = signature
        self.device = context.gpu_layer.w13_weight.device
        slot0_raw = {
            name: getattr(context.gpu_layer, name).data
            for name in _Mxfp4PrefillSlot.RAW_NAMES
        }
        if slot1_raw_tensors is None:
            # Single-slot mode (VRAM-constrained): no successor prefetch, but
            # host staging still alternates 2 slots for CPU->GPU transfers.
            self.slots = (
                _Mxfp4PrefillSlot(
                    0,
                    slot0_raw,
                    slot_marlin_prepared[0],
                    frozen_shapes=context._raw_weight_shapes,
                ),
            )
        else:
            self.slots = (
                _Mxfp4PrefillSlot(
                    0,
                    slot0_raw,
                    slot_marlin_prepared[0],
                    frozen_shapes=context._raw_weight_shapes,
                ),
                _Mxfp4PrefillSlot(
                    1,
                    slot1_raw_tensors,
                    slot_marlin_prepared[1],
                    frozen_shapes=context._raw_weight_shapes,
                ),
            )
        # Host staging slot count is independent of the layerwise compute
        # slots: keep 2 so CPU->GPU transfers stay pipelined (a single host
        # slot serializes every expert copy -> catastrophic prefill slowdown).
        self.host_slot_free_events = (torch.cuda.Event(), torch.cuda.Event())
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.postprocess_stream = torch.cuda.Stream(device=self.device)
        # Runtime transport control must not use the main stream: it is
        # launched after layer N's compute so layer N+1 transport can overlap
        # that compute.  A one-element NCCL reduction on this dedicated stream
        # is substantially cheaper than a Gloo round-trip per transport phase.
        self.control_stream = torch.cuda.Stream(device=self.device)
        self.host_slot_was_used = [False, False]
        self.host_write_status = torch.ones(
            (1,), dtype=torch.int32, device="cpu"
        )
        self.device_phase_status = torch.ones(
            (1,), dtype=torch.int32, device=self.device
        )

        # Per-layer pinned host-weight cache (TP1 host staging).  Built lazily
        # on the first load of each layer; subsequent batches skip the KT
        # worker entirely and only do H2D + GPU Marlin repack.
        self._host_cache: Dict[int, Optional[Dict[str, torch.Tensor]]] = {}
        self._host_cache_lock = threading.Lock()
        self._cache_dtypes_logged = False
        # Staging buffers for the cache path: the full-layer raw image is NOT
        # materialized on GPU; each batch is H2D'd from the host cache,
        # Marlin-repacked, and the full raw is freed once the host cache covers
        # every layer (baseline host staging: raw on CPU, GPU keeps 1 prepared
        # slot).
        #
        # Each pipeline range owns its own PAIR.  Two buffers per range are what
        # makes the H2D of batch i+1 overlap the Marlin repack of batch i; with
        # one buffer the two strictly serialize (the next copy cannot overwrite
        # it until the repack has consumed it), which measured 12.6 GiB/s of a
        # 23.22 GiB/s pinned peak.
        #
        # The ranges must not SHARE a pair.  A repack is fenced on the compute
        # of its own range (it overwrites that range's prepared storage), so
        # sharing would put `range 1's repack -> staging_free -> range 0's H2D`
        # in one chain, and the transfer stream would transitively wait out the
        # whole previous layer's grouped GEMMs -- exactly the serialization the
        # pipeline exists to remove.  Splitting the batch instead of growing the
        # pool keeps the staging footprint identical to the 2 x BATCH_EXPERTS it
        # replaces, which matters: at chunk 16384 the card has 0.65 GiB free
        # once the slots are resident.
        self._full_raw_freed = False
        # Size staging from the context's frozen shape snapshot, not from the
        # live slot0_raw tensors: `_allocate_mxfp4_slot_storage` frees the
        # full-layer raw BEFORE the prepared alloc (the free pays for the
        # slot), so by manager-construction time those entries are empty
        # Parameters and their .shape would collapse the staging buffers.
        shape_map = context._raw_weight_shapes
        self.staging_buffers: List[Dict[str, torch.Tensor]] = [
            {
                name: torch.empty(
                    (STAGING_EXPERTS, *shape_map[name][0][1:]),
                    dtype=shape_map[name][1],
                    device=self.device,
                )
                for name in _Mxfp4PrefillSlot.RAW_NAMES
            }
            for _ in range(STAGING_BUFFERS_PER_RANGE * PIPELINE_RANGES)
        ]
        # [b] = batch b's H2D is complete (transfer -> postprocess)
        # [b] = batch b's repack has consumed the buffer (postprocess -> transfer)
        self._staging_ready_events = tuple(
            torch.cuda.Event() for _ in self.staging_buffers
        )
        self._staging_free_events = tuple(
            torch.cuda.Event() for _ in self.staging_buffers
        )
        # A buffer's first use has no prior consumer to wait for; every later
        # use must wait for the repack that read the previous occupant.
        self._staging_use_count = [0] * len(self.staging_buffers)
        # range_consumed[r] fences the compute of range r on the main stream.
        # The next layer's repack of range r waits on it, which is what lets
        # repack(L+1, r) run under compute(L, r) without touching range r's
        # buffer early.  `_range_consumed_valid[r]` is False until the first
        # compute records one (and is reset by abort_round).
        self._range_consumed = [torch.cuda.Event() for _ in range(PIPELINE_RANGES)]
        self._range_consumed_valid = [False] * PIPELINE_RANGES
        # Sum buffer for the pipeline ranges; see _range_accumulator.
        self._range_output: Optional[torch.Tensor] = None
        # See _MXFP4_LAYERWISE_STAGE_DEBUG.
        self._stage_marks: Optional[tuple] = None
        self._stage_xfer_start: Optional[torch.cuda.Event] = None
        self._stage_xfer_end: Optional[torch.cuda.Event] = None
        self._stage_fence_end: Optional[torch.cuda.Event] = None
        self._stage_batches: List[torch.cuda.Event] = []
        self._stage_probe_done = False
        self._stage_repack_end: Optional[torch.cuda.Event] = None
        self._stage_last_compute_start: Optional[torch.cuda.Event] = None
        self._stage_streams_logged = False

        self.epoch = -1
        self.last_layer_position: Optional[int] = None
        self.current_slot_index: Optional[int] = None
        self.round_active = False

    @property
    def registry(self):
        return _MXFP4_PREFILL_LAYER_REGISTRY.get(self.signature, {})

    @property
    def layer_order(self) -> List[int]:
        return sorted(self.registry)

    def successor_layer_idx(self, layer_idx: int) -> Optional[int]:
        order = self.layer_order
        pos = bisect.bisect_right(order, layer_idx)
        return order[pos] if pos < len(order) else None

    def abort_round(self) -> None:
        if not self.round_active:
            return
        self.epoch += 1
        self.last_layer_position = None
        self.current_slot_index = None
        self.round_active = False
        for slot in self.slots:
            slot.invalidate()

    def teardown(self) -> None:
        """Free the layerwise GPU footprint (prepared slots + staging).

        Called when the layerwise prefill forward OOMs (e.g. a very long
        context on a VRAM-tight config): the ~4 GiB freed here lets the same
        request fall back to the hybrid CPU MoE path, which shares the GPU for
        attention but keeps MoE weights on CPU.
        """
        # _bind_slot set gpu_layer._v4_marlin_weights to the prepared tensors;
        # clear that alias or the prepared storage survives via the shared
        # gpu_layer even after slot.marlin_prepared is dropped.
        try:
            if self.context is not None:
                gl = self.context.gpu_layer
                for _a in ("_v4_marlin_weights", "_v4_tk_w13", "_v4_tk_w2"):
                    if hasattr(gl, _a):
                        setattr(gl, _a, None)
        except Exception:
            pass
        for slot in self.slots:
            slot.marlin_prepared = None
        self.slots = ()
        try:
            self.staging_buffers.clear()
        except Exception:
            pass
        self._full_raw_freed = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _advance_round(self, layer_idx: int) -> None:
        order = self.layer_order
        pos = bisect.bisect_left(order, layer_idx)
        if pos >= len(order) or order[pos] != layer_idx:
            raise RuntimeError(
                f"MXFP4 layerwise prefill layer {layer_idx} is not registered; "
                f"registered layers are {order}"
            )

        if (
            not self.round_active
            or self.last_layer_position is None
            or pos <= self.last_layer_position
        ):
            self.epoch += 1
            self.round_active = True
            for slot in self.slots:
                if slot.epoch != self.epoch:
                    slot.invalidate()
        self.last_layer_position = pos

    def _find_ready_slot(self, layer_idx: int) -> Optional[_Mxfp4PrefillSlot]:
        for slot in self.slots:
            if (
                slot.state == "READY"
                and slot.layer_idx == layer_idx
                and slot.epoch == self.epoch
            ):
                return slot
        return None

    @staticmethod
    def _record_prepared_backing_on_stream(
        slot: _Mxfp4PrefillSlot, stream: torch.cuda.Stream
    ) -> None:
        """Tell the caching allocator which prepared tensors compute consumes."""

        prepared = slot.marlin_prepared
        if prepared is None:
            return
        if isinstance(prepared, tuple):
            # Strided triton_kernels form: the elements are triton_kernels.Tensor
            # wrappers (or PrecisionConfig, which owns no storage), not torch
            # tensors.  Record whatever underlying storage they expose; nothing to
            # do for a pure alias, since the raw slot tensors are already recorded.
            for item in prepared:
                for attr in ("data", "storage", "base"):
                    inner = getattr(item, attr, None)
                    if isinstance(inner, torch.Tensor):
                        inner.record_stream(stream)
                        break
            return
        for tensor in (
            prepared.w13,
            prepared.w13_scale,
            prepared.w2,
            prepared.w2_scale,
        ):
            tensor.record_stream(stream)

    def _bind_slot(self, slot: _Mxfp4PrefillSlot) -> None:
        layer = self.context.gpu_layer
        prepared = slot.marlin_prepared
        if isinstance(prepared, tuple):
            # Strided triton_kernels form (SGLANG_KT_PREFILL_TK_SLOT=1): zero extra
            # bytes over the raw slot.  The layer's own dispatch keys on these flags,
            # so binding the tk objects is all the compute path needs -- see
            # _allocate_mxfp4_slot_storage for why the Marlin copy is 6.72 GiB.
            w13_swiz, w13_pcg, w2_swiz, w2_pcg = prepared
            layer._v4_tk_w13 = w13_swiz
            layer._v4_tk_w13_pcg = w13_pcg
            layer._v4_tk_w2 = w2_swiz
            layer._v4_tk_w2_pcg = w2_pcg
            layer._v4_tk_intermediate_size = layer.w2_weight.shape[2] * 2
            layer._v4_tk_num_experts = layer.w13_weight.shape[0]
            layer._v4_tk_path = True
            layer._v4_marlin_path = False
            return
        layer._v4_marlin_weights = prepared
        layer._v4_marlin_path = True
        layer._v4_tk_path = False

    def _tp_phase_succeeded(self, local_success: bool) -> bool:
        if (
            not dist.is_initialized()
            or get_tensor_model_parallel_world_size() == 1
        ):
            return local_success
        status = getattr(self, "host_write_status", None)
        if status is None:
            status = torch.ones((1,), dtype=torch.int32, device="cpu")
            self.host_write_status = status
        status.fill_(int(local_success))
        dist.all_reduce(
            status,
            op=dist.ReduceOp.MIN,
            group=get_tp_group().cpu_group,
        )
        return bool(status.item())

    def _commit_tp_runtime_phase(
        self, local_error: Optional[Exception], phase: str
    ) -> None:
        if self._tp_phase_succeeded(local_error is None):
            return
        message = f"MXFP4 {phase} failed on at least one TP rank"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _tp_device_phase_succeeded(self, local_success: bool) -> bool:
        """Fast TP consensus for the per-expert transport control plane.

        Gloo remains the initialization/layer-boundary error plane.  The hot
        loop instead reduces a persistent CUDA scalar on a dedicated control
        stream, so it neither waits for the main compute stream nor allocates.
        Recoverable launch/control errors are propagated here; a poisoned CUDA
        context is process-fatal and is left to the process-group watchdog.
        """

        if (
            not dist.is_initialized()
            or get_tensor_model_parallel_world_size() == 1
        ):
            return local_success
        with torch.cuda.stream(self.control_stream):
            self.device_phase_status.fill_(int(local_success))
            dist.all_reduce(
                self.device_phase_status,
                op=dist.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
            # Keep the blocking scalar read on the stream where Work.wait()
            # inserted NCCL's completion dependency.  Reading after leaving
            # this context would race that dependency on the default stream.
            return bool(self.device_phase_status.item())

    def _commit_tp_device_runtime_phase(
        self, local_error: Optional[Exception], phase: str
    ) -> None:
        if self._tp_device_phase_succeeded(local_error is None):
            return
        message = f"MXFP4 {phase} failed on at least one TP rank"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _submit_host_write(
        self, method, expert_id: int, host_slot: int, sync: bool = True
    ) -> None:
        buffers = self.context.cpu_buffers
        pointers = self.context.all_rank_buffer_ptrs

        def expert_nbytes(name: str) -> int:
            tensor = buffers[name]
            return tensor.numel() // 2 * tensor.element_size()

        offsets = {
            name: host_slot * expert_nbytes(name)
            for name in _Mxfp4PrefillSlot.RAW_NAMES
        }

        def rank_pointers(name: str) -> List[int]:
            return [ptr + offsets[name] for ptr in pointers[name]]

        method.wrapper.submit_write_weight_scale_to_buffer(
            get_tensor_model_parallel_world_size(),
            expert_id,
            rank_pointers("w13_weight"),
            rank_pointers("w13_weight_scale_inv"),
            rank_pointers("w2_weight"),
            rank_pointers("w2_weight_scale_inv"),
        )
        if sync:
            method.wrapper.sync_write_weight_scale_to_buffer()

    @staticmethod
    def _host_cache_dtypes() -> Dict[str, torch.dtype]:
        return {
            "w13_weight": torch.int8,
            "w13_weight_scale_inv": torch.bfloat16,
            "w2_weight": torch.int8,
            "w2_weight_scale_inv": torch.bfloat16,
        }

    def _build_host_cache(
        self, method, layer_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Pack every CPU expert of a layer into host memory ONCE.

        The KT weight-write output is identical for every prefill batch, so
        caching it removes the per-batch CPUInfer write from the layerwise hot
        path; per-batch loads then only do the H2D + GPU Marlin repack.  This
        is the TP1 host staging design used by the local verify-baseline: raw
        weights stay in host memory, the GPU keeps one prepared slot, and the
        per-layer transfer alternates two host staging slots.

        The synchronous (cold-first-prefill) build uses PAGEABLE memory:
        cudaHostAlloc of a fresh 3.6GB costs ~0.7-1.3s on this box and the
        per-layer caches accumulate, so a pinned synchronous build would add
        ~1s to every layer.  The eager background builder
        (maybe_start_mxfp4_host_cache_build) pre-builds PINNED caches at
        startup, so the cold first prefill normally hits those instead.

        Returns None (and records None) when the layer has no CPU experts, the
        world size is > 1 (the SHM collective path handles TP>1), or the
        allocation fails; the caller then falls back to the per-batch packing
        path.
        """
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size != 1:
            self._host_cache[layer_idx] = None
            return None
        if not getattr(self, "_cache_dtypes_logged", False):
            self._cache_dtypes_logged = True
            logger.debug(
                "[SLOTLOAD] cache dtype probe: %s",
                {
                    name: (
                        str(getattr(self.context.gpu_layer, name).dtype),
                        tuple(getattr(self.context.gpu_layer, name).shape),
                    )
                    for name in _Mxfp4PrefillSlot.RAW_NAMES
                },
            )
        pinned = os.environ.get("SGLANG_MXFP4_HOST_CACHE_PINNED", "0") == "1"
        # Sizes come from the layer's construction args, not from gpu_layer's
        # weight shapes: _maybe_free_full_raw replaces the full-layer raw image
        # with a zero-length placeholder once the host cache covers every layer,
        # so the shapes are not a reliable source after the first layer loads.
        # _build_mxfp4_host_cache_entry already treats these args as canonical.
        hidden_size, intermediate_size, _ = method._full_init_args
        layer_cache = _build_mxfp4_host_cache_entry(
            method,
            layer_idx,
            pinned=pinned,
            num_experts=self.context.gpu_layer.num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            include_gpu_experts=os.environ.get(
                "KT_PREFILL_RELEASE_RESIDENTS"
            ) == "1",
        )
        if layer_cache is None:
            self._host_cache[layer_idx] = None
            return None
        num_cpu = sum(
            1
            for e in range(self.context.gpu_layer.num_experts)
            if not bool(method.gpu_experts_mask[e])
        )
        cache_bytes = sum(
            t.numel() * t.element_size() for t in layer_cache.values()
        )
        logger.debug(
            "[SLOTLOAD] layer=%d host-weight cache built: %d CPU experts "
            "(%.1f GiB host, pinned=%s).",
            layer_idx,
            num_cpu,
            cache_bytes / 1024**3,
            pinned,
        )
        self._host_cache[layer_idx] = layer_cache
        return layer_cache

    def _get_host_cache(
        self, method, layer_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        if os.environ.get("SGLANG_MXFP4_HOST_WEIGHT_CACHE", "1") == "0":
            return None
        if layer_idx not in self._host_cache:
            # The eager startup builder may already have this layer's pinned
            # cache ready; take it instead of rebuilding synchronously.
            with _MXFP4_EAGER_CACHE_LOCK:
                eager = _MXFP4_EAGER_HOST_CACHE.get(self.signature, {}).get(
                    layer_idx, _UNSET
                )
            if eager is not _UNSET:
                self._host_cache[layer_idx] = eager
            else:
                with self._host_cache_lock:
                    if layer_idx not in self._host_cache:
                        self._build_host_cache(method, layer_idx)
        return self._host_cache.get(layer_idx)

    def _stage_cpu_from_cache(
        self,
        slot: _Mxfp4PrefillSlot,
        layer_cache: Dict[str, torch.Tensor],
        cpu_expert_ids: List[int],
    ) -> None:
        """H2D CPU experts straight from the host cache.  Must be called with
        the transfer stream current (the per-layer DMA is async and the slot's
        raw fence is recorded by the caller).

        When every expert is CPU resident (the layerwise production config),
        the cache tensor and the slot raw image share the [num_experts, ...]
        layout, so ONE whole-tensor copy per weight does the transfer at full
        H2D bandwidth.  Per-expert copies from pageable host memory stage each
        slice through a bounce buffer and top out around 4-5GB/s (0.8s/layer),
        so the per-expert fallback is reserved for mixed GPU/CPU placements."""
        if not cpu_expert_ids:
            return
        if len(cpu_expert_ids) == slot.num_experts:
            _t = time.perf_counter()
            for name in _Mxfp4PrefillSlot.RAW_NAMES:
                getattr(slot, name).copy_(
                    layer_cache[name], non_blocking=True
                )
            logger.debug(
                "[SLOTLOAD] layer=%d bulk-H2D=%.1fms",
                slot.layer_idx if slot.layer_idx is not None else -1,
                (time.perf_counter() - _t) * 1000,
            )
        else:
            for expert_id in cpu_expert_ids:
                for name in _Mxfp4PrefillSlot.RAW_NAMES:
                    getattr(slot, name)[expert_id].copy_(
                        layer_cache[name][expert_id], non_blocking=True
                    )

    def _prepared_range_view(
        self, slot: _Mxfp4PrefillSlot, start: int, count: int
    ):
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            V4MarlinPreparedWeights,
        )

        prepared = slot.marlin_prepared
        return V4MarlinPreparedWeights(
            w13=prepared.w13[start : start + count],
            w13_scale=prepared.w13_scale[start : start + count],
            w2=prepared.w2[start : start + count],
            w2_scale=prepared.w2_scale[start : start + count],
            hidden_size=prepared.hidden_size,
            intermediate_size=prepared.intermediate_size,
            num_experts=count,
        )

    def _maybe_free_full_raw(self) -> None:
        """Drop the full-layer raw image (slot raw / gpu_layer, ~3.75 GiB).

        Once the host cache covers every registered layer, the cache path
        repacks straight from host memory through the small staging buffer and
        never reads the full raw again; freeing it restores the baseline host
        staging footprint (raw on CPU, GPU keeps 1 prepared + small staging)
        so the production 1M-context config's layerwise prefill fits VRAM.
        """
        if self._full_raw_freed:
            return
        with _MXFP4_EAGER_CACHE_LOCK:
            eager = _MXFP4_EAGER_HOST_CACHE.get(self.signature, {})
            # A layer counts as covered only when a real cache dict exists for
            # it. The eager map records None for a layer whose pack failed and
            # holds no entry at all for one not built yet; both must keep the
            # raw image alive, since the fallback path repacks from it.
            if not all(
                self._host_cache.get(i) is not None
                or eager.get(i) is not None
                for i in self.layer_order
            ):
                return
        self._full_raw_freed = True
        freed_bytes = 0
        for name in _Mxfp4PrefillSlot.RAW_NAMES:
            try:
                t = getattr(self.context.gpu_layer, name)
                freed_bytes += t.numel() * t.element_size()
                setattr(
                    self.context.gpu_layer,
                    name,
                    torch.nn.Parameter(
                        torch.empty(0, device=t.device), requires_grad=False
                    ),
                )
            except Exception:
                pass
            for slot in self.slots:
                setattr(slot, name, None)
            # The context's original_params/original_buffers snapshots also
            # reference the full-layer raw tensors (captured right after
            # create_weights); drop those entries or the ~3.75 GiB stays live.
            try:
                if name in self.context.original_params:
                    del self.context.original_params[name]
            except Exception:
                pass
            try:
                if name in self.context.original_buffers:
                    del self.context.original_buffers[name]
            except Exception:
                pass
        # Force the raw image out of the CUDA caching allocator's live set so
        # the first layerwise forward sees the real free pool (without this the
        # freed raw stays reserved and the production 24K-chunk forward OOMs
        # where the baseline -- which never allocates a full-layer raw -- fits).
        #
        # Only when something was actually freed.  With
        # SGLANG_KT_SLOT_FREE_RAW_FIRST the raw is already gone before this runs,
        # so the loop above frees 0 bytes -- and empty_cache() here would then
        # hand the allocator's whole pool back to the driver right before the
        # server starts serving, making every decode-step allocation a fresh
        # cudaMalloc.
        gc.collect()
        if freed_bytes and torch.cuda.is_available():
            if os.environ.get("SGLANG_KT_NO_EMPTY_CACHE") != "1":
                torch.cuda.empty_cache()
        logger.debug(
            "[SLOTLOAD] freed full-layer raw (%.2f GiB) after host cache "
            "covered all %d layers; layerwise footprint = prepared %.2f GiB "
            "+ staging %.2f GiB.",
            freed_bytes / 1024**3,
            len(self.layer_order),
            sum(
                t.numel() * t.element_size()
                for t in (
                    []
                    if isinstance(self.slots[0].marlin_prepared, tuple)
                    else [
                        self.slots[0].marlin_prepared.w13,
                        self.slots[0].marlin_prepared.w13_scale,
                        self.slots[0].marlin_prepared.w2,
                        self.slots[0].marlin_prepared.w2_scale,
                    ]
                )
            )
            / 1024**3,
            sum(
                t.numel() * t.element_size()
                for buffer in self.staging_buffers
                for t in buffer.values()
            )
            / 1024**3,
        )

    def _stage_probe_source(self, slot, layer_cache: Dict[str, torch.Tensor]) -> None:
        """One-shot: WHICH property of the in-situ copy costs the 25 -> 19.7 GB/s?

        Offline every configuration of this copy reaches 25 GB/s while in situ the same bytes
        move at 19.7, and the offline arms that came back flat are already long: placement
        (either single node, or interleave), allocation mode (cudaHostAlloc vs pageable +
        cudaHostRegister), batch shape, pinned footprint 7-271 GiB, SM load and power draw, the
        real repack chained on the buffer-free event, competing host-memory traffic, and a
        memory-bound instead of compute-bound kernel.

        The first version of this probe copied ONE 720 MiB region three times inside the server
        and read 25.0, which is an upper bound rather than the pipeline's rate: repeating one
        region keeps its page tables warm, while a layer SWEEPS 7.65 GB once.  So this version
        measures all three shapes in the same one-shot, in the server's own context:

          1-region x3   -- the warm case, for continuity with that earlier reading
          sweep         -- every batch of one whole layer into the real staging buffers,
                           exactly as `_repack_from_cache` issues it, nothing else running
          sweep+repack  -- the same with the real `prepare_v4_mxfp4_marlin` chained on each
                           buffer's free event (into a scratch slot, so the live slot is not
                           clobbered), which is the pipeline's actual structure
        """
        if not _MXFP4_LAYERWISE_STAGE_DEBUG or self._stage_probe_done:
            return
        self._stage_probe_done = True
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            V4MarlinPreparedWeights,
            allocate_v4_mxfp4_marlin,
            prepare_v4_mxfp4_marlin,
        )

        def timed(issue) -> float:
            """Seconds the transfer stream spends on `issue`, events on that stream only."""
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.transfer_stream)
            issue()
            end.record(self.transfer_stream)
            end.synchronize()
            return start.elapsed_time(end) / 1e3

        def sweep(with_repack: bool):
            """Issue one layer's worth of batches, the way `_repack_from_cache` does."""
            scratch = (
                allocate_v4_mxfp4_marlin(
                    num_experts=STAGING_EXPERTS,
                    hidden_size=slot.marlin_prepared.hidden_size,
                    intermediate_size=slot.marlin_prepared.intermediate_size,
                    device=self.device,
                )
                if with_repack
                else None
            )
            free = [torch.cuda.Event(), torch.cuda.Event()]
            ready = [torch.cuda.Event(), torch.cuda.Event()]
            used = [False, False]

            def issue() -> None:
                for index, start in enumerate(
                    range(0, slot.num_experts, STAGING_EXPERTS)
                ):
                    count = min(STAGING_EXPERTS, slot.num_experts - start)
                    buf = index % STAGING_BUFFERS_PER_RANGE
                    staging = self.staging_buffers[buf]
                    with torch.cuda.stream(self.transfer_stream):
                        if with_repack and used[buf]:
                            self.transfer_stream.wait_event(free[buf])
                        for name in _Mxfp4PrefillSlot.RAW_NAMES:
                            staging[name][:count].copy_(
                                layer_cache[name][start : start + count],
                                non_blocking=True,
                            )
                        ready[buf].record(self.transfer_stream)
                    if with_repack:
                        with torch.cuda.stream(self.postprocess_stream):
                            self.postprocess_stream.wait_event(ready[buf])
                            prepare_v4_mxfp4_marlin(
                                staging["w13_weight"][:count],
                                staging["w13_weight_scale_inv"][:count],
                                staging["w2_weight"][:count],
                                staging["w2_weight_scale_inv"][:count],
                                out=V4MarlinPreparedWeights(
                                    w13=scratch.w13[:count],
                                    w13_scale=scratch.w13_scale[:count],
                                    w2=scratch.w2[:count],
                                    w2_scale=scratch.w2_scale[:count],
                                    hidden_size=scratch.hidden_size,
                                    intermediate_size=scratch.intermediate_size,
                                    num_experts=count,
                                ),
                            )
                            free[buf].record(self.postprocess_stream)
                            used[buf] = True

            return issue, scratch

        try:
            reps = 3
            source = layer_cache["w13_weight"][:STAGING_EXPERTS]
            warm_bytes = source.numel() * source.element_size() * reps
            warm_dev = torch.empty_like(source, device=self.device)

            def issue_warm() -> None:
                # The stream context is load-bearing: `copy_` outside it lands on the
                # current stream, and the events recorded on the transfer stream would
                # then bracket nothing (measured once as a nonsense 11170 GB/s).
                with torch.cuda.stream(self.transfer_stream):
                    for _ in range(reps):
                        warm_dev.copy_(source, non_blocking=True)

            warm = timed(issue_warm)
            del warm_dev

            sweep_bytes = sum(
                layer_cache[name].numel() * layer_cache[name].element_size()
                for name in _Mxfp4PrefillSlot.RAW_NAMES
            )
            # Three plain sweeps then one chained: the first pass over a layer's pages is
            # the cold one, and the pipeline is a steady-state (repeated) reader, so the
            # SEQUENCE is what says whether 19 GB/s is cold-start or the real rate.
            sweeps = []
            for _ in range(3):
                issue, scratch = sweep(with_repack=False)
                sweeps.append(sweep_bytes / timed(issue) / 1e9)
                del issue, scratch
            issue, scratch = sweep(with_repack=True)
            chained = sweep_bytes / timed(issue) / 1e9
            del issue, scratch

            logger.info(
                "[lwprobe] GB/s  1-region x%d=%.2f  sweeps=%s  sweep+repack=%.2f   "
                "(%.2f GiB per sweep, staging %d experts)",
                reps,
                warm_bytes / warm / 1e9,
                "/".join(f"{s:.2f}" for s in sweeps),
                chained,
                sweep_bytes / 1024**3,
                STAGING_EXPERTS,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic only, never break prefill
            logger.warning("[lwprobe] failed: %s", exc)

    def _repack_from_cache(
        self,
        slot: _Mxfp4PrefillSlot,
        method,
        original_layer: torch.nn.Module,
        layer_cache: Dict[str, torch.Tensor],
        gpu_expert_ids: List[int],
        first_expert: int,
        num_experts: int,
        ready_event: torch.cuda.Event,
        previous_compute_fence: Optional[torch.cuda.Event],
        range_index: int,
    ) -> None:
        """H2D one expert range of the host cache and Marlin-repack it into the
        prepared slot, WITHOUT materializing the full layer raw on GPU.

        For the all-CPU placement each batch is a contiguous slice of the
        cache (one bulk H2D per weight per batch); mixed GPU/CPU placements
        fall back to per-expert copies inside the batch.

        The H2D runs on the transfer stream and the Marlin repack on the
        postprocess stream, with this range's own pair of staging buffers
        alternating per batch, so the copy of batch i+1 overlaps the repack of
        batch i.  Serializing the two on one stream is what held the link at 54%
        of peak (see the staging_buffers note in __init__).

        `previous_compute_fence` is the fence the PREVIOUS reader of this
        range's prepared storage recorded on the compute stream -- the range's
        own grouped GEMM one layer earlier (see PIPELINE_RANGES).  It fences
        the repack only, not the H2D before it, which is the whole point: the
        bytes for the next range keep crossing the link while the current range
        is still being computed.  Completion is published on `ready_event` from
        the postprocess stream; consumers wait on that, not on the transfer
        stream.

        `range_index` selects the staging pair.  It must not be shared between
        ranges: the H2D's wait for a freed staging buffer would then chain
        through the other range's repack and, with it, that range's compute
        fence -- putting the PCIe link back behind the grouped GEMMs.
        """
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            prepare_v4_mxfp4_marlin,
        )

        range_gpu_expert_ids = [
            e
            for e in gpu_expert_ids
            if first_expert <= e < first_expert + num_experts
        ]
        # A released resident raw (KT_PREFILL_RELEASE_RESIDENTS) has no GPU copy
        # to read, so the range must come from the host cache like any CPU
        # expert -- which is why the cache is built over ALL experts when that
        # release is active.  Without this the mixed branch below dereferences
        # `None[gpu_index]` and the whole prefill dies in
        # "MXFP4 cache repack for layer 0 failed".
        _resident_raw = getattr(original_layer, "w13_weight", None)
        all_cpu = (not range_gpu_expert_ids) or _resident_raw is None
        last_expert = first_expert + num_experts
        starts = list(range(first_expert, last_expert, STAGING_EXPERTS))
        buffer_base = range_index * STAGING_BUFFERS_PER_RANGE
        if first_expert == 0:
            self._stage_xfer_start = self._stage_mark(self.transfer_stream)
            self._stage_probe_source(slot, layer_cache)
        if previous_compute_fence is not None:
            with torch.cuda.stream(self.postprocess_stream):
                self.postprocess_stream.wait_event(previous_compute_fence)
            if first_expert == 0:
                # When the last reader of this range's prepared storage finished, as a
                # position in the transfer stream's timeline.  `fence` > `xfer` in the
                # stage report means the repack -- and with it every H2D past the first
                # pair of staging buffers -- was gated by the PREVIOUS layer's GEMM, so
                # the link was idle for that much of the span.
                self._stage_fence_end = self._stage_mark(self.postprocess_stream)
        for index, start in enumerate(starts):
            count = min(STAGING_EXPERTS, last_expert - start)
            buf = buffer_base + index % STAGING_BUFFERS_PER_RANGE
            staging = self.staging_buffers[buf]
            with torch.cuda.stream(self.transfer_stream):
                if _MXFP4_LAYERWISE_STAGE_DEBUG and first_expert == 0:
                    # Paired with the mark below: their difference is the batch's COPY
                    # time, and the gap from the previous pair is the time the transfer
                    # stream spent waiting.  `busy` vs `xfer` separates "the copies are
                    # slow here" from "the link idles".
                    self._stage_batches.append(self._stage_mark(self.transfer_stream))
                if self._staging_use_count[buf] > 0:
                    # This buffer's previous repack must have consumed it.
                    self.transfer_stream.wait_event(self._staging_free_events[buf])
                if all_cpu:
                    for name in _Mxfp4PrefillSlot.RAW_NAMES:
                        staging[name][:count].copy_(
                            layer_cache[name][start : start + count],
                            non_blocking=True,
                        )
                else:
                    # `bool(method.gpu_experts_mask[expert_id])` inside the loop below is a
                    # synchronizing DtoH, so staging one layer cost 384 of them.  Read the mask
                    # once; read the index map lazily, so an all-CPU mask (--kt-num-gpu-experts 0)
                    # pays no extra readback at all.
                    mask_host = method.gpu_experts_mask.cpu().tolist()
                    gpu_index_host = None
                    for i in range(count):
                        expert_id = start + i
                        if mask_host[expert_id]:
                            if gpu_index_host is None:
                                gpu_index_host = (
                                    method.logical_to_gpu_index.cpu().tolist()
                                )
                            gpu_index = gpu_index_host[expert_id]
                            for name in _Mxfp4PrefillSlot.RAW_NAMES:
                                staging[name][i].copy_(
                                    getattr(original_layer, name)[gpu_index],
                                    non_blocking=True,
                                )
                        else:
                            for name in _Mxfp4PrefillSlot.RAW_NAMES:
                                staging[name][i].copy_(
                                    layer_cache[name][expert_id],
                                    non_blocking=True,
                                )
                self._staging_ready_events[buf].record(self.transfer_stream)
                if _MXFP4_LAYERWISE_STAGE_DEBUG and first_expert == 0:
                    self._stage_batches.append(self._stage_mark(self.transfer_stream))
            with torch.cuda.stream(self.postprocess_stream):
                self.postprocess_stream.wait_event(self._staging_ready_events[buf])
                prepare_v4_mxfp4_marlin(
                    staging["w13_weight"][:count],
                    staging["w13_weight_scale_inv"][:count],
                    staging["w2_weight"][:count],
                    staging["w2_weight_scale_inv"][:count],
                    out=self._prepared_range_view(slot, start, count),
                )
                self._staging_free_events[buf].record(self.postprocess_stream)
                self._staging_use_count[buf] += 1
        if last_expert >= slot.num_experts:
            # End of the transfer stream's work for the whole layer: recorded
            # after the last batch's copies, so the span excludes the Marlin
            # repack (which runs on the postprocess stream).
            self._stage_xfer_end = self._stage_mark(self.transfer_stream)
        ready_event.record(self.postprocess_stream)
        if last_expert >= slot.num_experts:
            # Trailing Marlin repack of the layer, on the postprocess stream.
            # A dedicated timing event: `ready_event` is a plain fence and
            # cudaEventElapsedTime rejects mixing the two kinds.
            self._stage_repack_end = self._stage_mark(self.postprocess_stream)
        slot.reuse_guard = "ready"

    def _stage_mark(self, stream) -> Optional[torch.cuda.Event]:
        """Record a timing boundary for SGLANG_LAYERWISE_STAGE_DEBUG, else None."""
        if not _MXFP4_LAYERWISE_STAGE_DEBUG:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def _stage_report_previous(self) -> None:
        """Log the previous layer's device-stage spans.

        Deferred to the next layer and skipped unless every event has completed,
        so the report never synchronizes the pipeline it is measuring.  Whether
        the transfer stream and the compute stream actually overlap is exactly
        the question this answers, so it must not be answered by a probe that
        serializes them.
        """
        global _MXFP4_LAYERWISE_STAGE_LEFT
        prev = self._stage_marks
        fence_end = self._stage_fence_end
        batch_marks = self._stage_batches
        if prev is None:
            return
        self._stage_marks = None
        self._stage_fence_end = None
        self._stage_batches = []
        if not _MXFP4_LAYERWISE_STAGE_DEBUG or _MXFP4_LAYERWISE_STAGE_LEFT <= 0:
            return
        prev_layer, xfer_start, xfer_end, repack_end, compute_start, compute_end = prev
        marks = (xfer_start, xfer_end, repack_end, compute_start, compute_end)
        if any(m is None or not m.query() for m in marks):
            return
        _MXFP4_LAYERWISE_STAGE_LEFT -= 1
        period = (
            self._stage_last_compute_start.elapsed_time(compute_start)
            if self._stage_last_compute_start is not None
            else float("nan")
        )
        self._stage_last_compute_start = compute_start
        fence = (
            xfer_start.elapsed_time(fence_end)
            if fence_end is not None and fence_end.query()
            else float("nan")
        )
        # Per-batch pairs: (start, end) per H2D batch.  `busy` is the sum of the copy
        # spans and `stall` the sum of the waits between them, so `busy` against `xfer`
        # says whether the link is slow or idle.
        busy = stall = max_gap = float("nan")
        if (
            batch_marks
            and len(batch_marks) % 2 == 0
            and xfer_start is not None
            and all(m is not None and m.query() for m in batch_marks)
        ):
            busy = 0.0
            stall = xfer_start.elapsed_time(batch_marks[0])
            max_gap = stall
            for index in range(0, len(batch_marks), 2):
                busy += batch_marks[index].elapsed_time(batch_marks[index + 1])
                if index + 2 < len(batch_marks):
                    gap = batch_marks[index + 1].elapsed_time(batch_marks[index + 2])
                    stall += gap
                    max_gap = max(max_gap, gap)
        logger.info(
            "[lwstage] layer=%d xfer=%.1fms busy=%.1fms stall=%.1fms maxgap=%.1fms "
            "fence=%.1fms repack_tail=%.1fms compute=%.1fms period=%.1fms",
            prev_layer,
            xfer_start.elapsed_time(xfer_end),
            busy,
            stall,
            max_gap,
            fence,
            xfer_end.elapsed_time(repack_end),
            compute_start.elapsed_time(compute_end),
            period,
        )

    def _stage_note_streams(self, main_stream) -> None:
        """Log the stream ids once: a side stream that PyTorch created as a
        *blocking* stream would be implicitly ordered against the default
        stream, which would make the transfer/compute overlap impossible no
        matter how the events are wired."""
        if self._stage_streams_logged:
            return
        self._stage_streams_logged = True
        logger.info(
            "[lwstage] main stream=%d transfer=%d postprocess=%d "
            "(default=%d, non_blocking flags transfer=%s postprocess=%s)",
            main_stream.cuda_stream,
            self.transfer_stream.cuda_stream,
            self.postprocess_stream.cuda_stream,
            torch.cuda.default_stream(self.device).cuda_stream,
            getattr(self.transfer_stream, "is_non_blocking", "n/a"),
            getattr(self.postprocess_stream, "is_non_blocking", "n/a"),
        )

    def _range_accumulator(self, chunk: torch.Tensor) -> torch.Tensor:
        """A stable [m, k] buffer summing the pipeline ranges' partial outputs.

        Every range shares ONE Marlin workspace -- the ranges have the same
        `num_experts`, so the same workspace key -- hence one `output` view that
        each grouped GEMM overwrites.  The accumulation therefore needs a buffer
        of its own.  It is cached so a layer's extra footprint is exactly one
        hidden-state-sized tensor instead of a fresh allocation per layer: the
        card has well under a GiB free once the layerwise slot is resident at
        chunk 16384 (measured 0.65 GiB).
        """
        buf = self._range_output
        if (
            buf is None
            or buf.shape != chunk.shape
            or buf.dtype != chunk.dtype
            or buf.device != chunk.device
        ):
            buf = torch.empty_like(chunk)
            self._range_output = buf
        return buf

    def _pipeline_ranges_for(self, num_experts: int) -> List[tuple]:
        """The expert ranges one layer is split into for the transfer/compute
        overlap, as [start, count) pairs covering [0, num_experts).

        The last range absorbs the remainder so the split always covers every
        expert even when num_experts is not a multiple of PIPELINE_RANGES.
        """
        base = num_experts // PIPELINE_RANGES
        ranges = []
        start = 0
        for index in range(PIPELINE_RANGES):
            count = base if index < PIPELINE_RANGES - 1 else num_experts - start
            ranges.append((start, count))
            start += count
        return ranges

    def _range_fence(
        self, range_index: Optional[int], slot: _Mxfp4PrefillSlot
    ) -> Optional[torch.cuda.Event]:
        """The compute fence this range's prepared storage is protected by.

        Normally the previous layer's compute of the same range, recorded on the
        main stream mid-layer so that the repack can still overlap the previous
        range's GEMM.  Before the first compute of a round there is no per-range
        fence yet, so fall back to the layer-level consumed event, which is
        strictly stronger (it covers the whole previous layer).  `None` means
        the single-range (SHM) path, which has no per-range state.
        """
        if range_index is None:
            return slot.consumed_event if slot.has_consumed_event else None
        if self._range_consumed_valid[range_index]:
            return self._range_consumed[range_index]
        return slot.consumed_event if slot.has_consumed_event else None

    def _consume_range(
        self,
        slot: _Mxfp4PrefillSlot,
        range_index: Optional[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Record that this range's grouped GEMM has been enqueued on `stream`."""
        if range_index is not None:
            self._range_consumed[range_index].record(stream)
            self._range_consumed_valid[range_index] = True
            return
        slot.consumed_event.record(stream)
        slot.has_consumed_event = True

    def _postprocess_slot(self, slot: _Mxfp4PrefillSlot) -> None:
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            prepare_v4_mxfp4_marlin,
        )

        _t_pp = time.perf_counter()
        with torch.cuda.stream(self.postprocess_stream):
            self.postprocess_stream.wait_event(slot.raw_ready_event)
            try:
                # Repack and scale-swizzle into stable caller-owned storage.  The
                # kernel is current-stream ordered and publishes no events; the
                # layerwise scheduler owns the raw/ready/consumed lifecycle.
                #
                # Under SGLANG_KT_PREFILL_TK_SLOT=1 the prepared object is a strided
                # ALIAS of this slot's raw tensors, so the host-written raw IS the
                # compute input and the repack has nothing to write.  That is the
                # point of the mode: it drops the 6.72 GiB prepared copy (and the
                # per-layer repack) to make room for resident GPU experts.
                if not isinstance(slot.marlin_prepared, tuple):
                    prepare_v4_mxfp4_marlin(
                        slot.w13_weight,
                        slot.w13_weight_scale_inv,
                        slot.w2_weight,
                        slot.w2_weight_scale_inv,
                        out=slot.marlin_prepared,
                    )
            finally:
                # Even an exceptional conversion attempt must publish a fence
                # before this slot can be overwritten on a retry.
                try:
                    slot.ready_event.record(self.postprocess_stream)
                    slot.reuse_guard = "ready"
                    if slot.layer_idx is not None:
                        logger.debug(
                            "[SLOTLOAD] layer=%d postprocess=%.1fms",
                            slot.layer_idx,
                            (time.perf_counter() - _t_pp) * 1000,
                        )
                except Exception:
                    self.postprocess_stream.synchronize()
                    slot.reuse_guard = "synchronized"
                    raise

    def _load_slot(
        self,
        slot: _Mxfp4PrefillSlot,
        layer_idx: int,
        method,
        original_layer: torch.nn.Module,
    ) -> None:
        if getattr(slot, "reuse_guard", None) == "poisoned":
            raise RuntimeError(
                f"MXFP4 slot {slot.index} cannot be reused after its CUDA "
                "transfer stream failed to synchronize"
            )
        if slot.state == "LOADING":
            raise RuntimeError(
                f"MXFP4 slot {slot.index} is already loading layer {slot.layer_idx}"
            )

        # A prefetched slot may be invalidated before it is ever consumed.
        # In that case there is no consumed event for this generation, so its
        # ready event (postprocess completion) is the overwrite fence.
        reuse_guard = getattr(slot, "reuse_guard", None)
        if reuse_guard is None and slot.has_consumed_event:
            reuse_guard = "consumed"

        # reuse_guard describes the *current* load generation.  Retain the
        # prior generation only in the local variable above so an exception
        # after partially enqueueing this load cannot mistake an old ready
        # fence for protection of the new DMA.
        slot.reuse_guard = "loading"
        slot.state = "LOADING"
        slot.layer_idx = layer_idx
        slot.epoch = self.epoch

        _t_load_start = time.perf_counter()
        saved_affinity = None
        try:
            setup_error = None
            try:
                weight_infos = [
                    (
                        name,
                        self.context.cpu_buffers[name],
                        getattr(slot, name),
                    )
                    for name in _Mxfp4PrefillSlot.RAW_NAMES
                ]
                gpu_expert_ids = []
                cpu_expert_ids = []
                # This runs once per layer in the layerwise prefill, so the per-expert
                # `.item()` loop it replaces cost 384 synchronizing DtoH per layer (15360 per
                # chunk) -- all of them wasted at --kt-num-gpu-experts 0, where the mask is
                # all-False.  One readback, same values.
                mask_host = method.gpu_experts_mask.cpu().tolist()
                for expert_id in range(slot.num_experts):
                    if mask_host[expert_id]:
                        gpu_expert_ids.append(expert_id)
                    else:
                        cpu_expert_ids.append(expert_id)

                if hasattr(os, "sched_getaffinity"):
                    saved_affinity = os.sched_getaffinity(0)
                    available_cpus = sorted(saved_affinity)
                    if available_cpus:
                        target = available_cpus[
                            -1 - (method.tp_rank % len(available_cpus))
                        ]
                        os.sched_setaffinity(0, {target})
            except Exception as exc:
                setup_error = exc
            self._commit_tp_runtime_phase(
                setup_error, f"transport setup for layer {layer_idx}"
            )

            gpu_copy_error = None
            layer_cache = self._get_host_cache(method, layer_idx)
            try:
                if layer_cache is None:
                    with torch.cuda.stream(self.transfer_stream):
                        if reuse_guard == "consumed":
                            self.transfer_stream.wait_event(slot.consumed_event)
                        elif reuse_guard == "ready":
                            self.transfer_stream.wait_event(slot.ready_event)
                        elif reuse_guard == "raw":
                            self.transfer_stream.wait_event(slot.raw_ready_event)
                        gpu_index_host = method.logical_to_gpu_index.cpu().tolist()
                        for expert_id in gpu_expert_ids:
                            gpu_index = gpu_index_host[expert_id]
                            for name, _, destination in weight_infos:
                                source = getattr(original_layer, name)
                                destination[expert_id].copy_(
                                    source[gpu_index], non_blocking=True
                                )
            except Exception as exc:
                gpu_copy_error = exc
            self._commit_tp_runtime_phase(
                gpu_copy_error, f"GPU expert copy for layer {layer_idx}"
            )

            pending_h2d_error = None
            if layer_cache is not None:
                # Fast path (TP1 host staging): every CPU expert of this layer
                # is already packed in host memory; repack straight into the
                # prepared slot through the small staging buffer (no full-layer
                # raw on GPU, no KT write, no per-expert sync, no SHM
                # collective).  Each range waits on the compute fence of its own
                # storage, which _repack_from_cache applies to the repack rather
                # than to the H2D -- that is what lets the next range's bytes
                # cross the link while the current one is being computed.
                #
                # Only range 0 is issued here.  The later ranges are issued from
                # apply(), after the earlier ranges' grouped GEMMs are enqueued,
                # because that is the only place where the range's compute fence
                # for THIS layer is known (they must not be repacked before the
                # previous layer's compute of the same range has drained).
                slot.pipeline_ranges = self._pipeline_ranges_for(slot.num_experts)
                slot.layer_cache = layer_cache
                slot.range_gpu_expert_ids = gpu_expert_ids
                try:
                    first_start, first_count = slot.pipeline_ranges[0]
                    self._repack_from_cache(
                        slot,
                        method,
                        original_layer,
                        layer_cache,
                        gpu_expert_ids,
                        first_start,
                        first_count,
                        slot.range_ready_event[0],
                        self._range_fence(0, slot),
                        0,
                    )
                    self._maybe_free_full_raw()
                except Exception as exc:
                    pending_h2d_error = exc
            else:
                # SHM path: no host cache for this layer, so there is no repack
                # to overlap and the whole layer is a single range.  Clearing the
                # cache-path state is load-bearing: apply() reads it to decide
                # whether to pipeline, and a stale value from the previous
                # layer's cache-path load would send it down the pipeline.
                slot.pipeline_ranges = None
                slot.layer_cache = None
                slot.range_gpu_expert_ids = []
                for position, expert_id in enumerate(cpu_expert_ids):
                    host_slot = position % 2

                    # Rank 0 writes every rank's SHM.  Every rank must
                    # therefore finish its own DMA before that host slot can
                    # be overwritten.  A prior H2D enqueue failure is sticky
                    # until this common control point, which lets every rank
                    # leave the hot loop in the same collective order instead
                    # of stranding a peer.
                    host_free_error = pending_h2d_error
                    pending_h2d_error = None
                    try:
                        if self.host_slot_was_used[host_slot]:
                            self.host_slot_free_events[host_slot].synchronize()
                    except Exception as exc:
                        if host_free_error is None:
                            host_free_error = exc
                    self._commit_tp_device_runtime_phase(
                        host_free_error,
                        f"host-slot {host_slot} reuse for expert {expert_id}",
                    )

                    write_error = None
                    if method.tp_rank == 0:
                        try:
                            if method.wrapper is None:
                                raise RuntimeError(
                                    "MXFP4 TP0 has no KT wrapper for host "
                                    "weight transport"
                                )
                            self._submit_host_write(
                                method, expert_id, host_slot
                            )
                        except Exception as exc:
                            write_error = exc

                    # The device reduction is both the producer-ready fence
                    # and an error broadcast.  TP0 therefore cannot strand
                    # peer ranks in a later phase if its KT writer fails.
                    self._commit_tp_device_runtime_phase(
                        write_error, f"host write for expert {expert_id}"
                    )

                    host_free_recorded = False
                    try:
                        with torch.cuda.stream(self.transfer_stream):
                            try:
                                for _, cpu_buffer, destination in weight_infos:
                                    destination[expert_id].copy_(
                                        cpu_buffer[host_slot],
                                        non_blocking=True,
                                    )
                            finally:
                                # Once any DMA may have been enqueued, a
                                # peer's failure must not make this rank
                                # forget the local host-slot fence before the
                                # consensus raises.
                                self.host_slot_was_used[host_slot] = True
                                self.host_slot_free_events[host_slot].record(
                                    self.transfer_stream
                                )
                                host_free_recorded = True
                    except Exception as exc:
                        pending_h2d_error = exc
                        if (
                            self.host_slot_was_used[host_slot]
                            and not host_free_recorded
                        ):
                            try:
                                # Event publication itself failed.  A
                                # local-stream sync is the exception-only
                                # safe fallback before this rank reports
                                # failure to its peers.
                                self.transfer_stream.synchronize()
                                self.host_slot_was_used[host_slot] = False
                            except Exception:
                                pass

            # H2D launch errors are reported at the next pre-write consensus.
            # The final expert has no successor, so combine its sticky status
            # with raw-fence publication and commit it once per layer.
            if layer_cache is not None:
                # Cache path: _repack_from_cache already published the ready
                # event and the repack fence; only the error status remains.
                self._commit_tp_runtime_phase(
                    pending_h2d_error,
                    f"cache repack for layer {layer_idx}",
                )
            else:
                raw_ready_error = pending_h2d_error
                try:
                    with torch.cuda.stream(self.transfer_stream):
                        slot.raw_ready_event.record(self.transfer_stream)
                    slot.reuse_guard = "raw"
                except Exception as exc:
                    if raw_ready_error is None:
                        raw_ready_error = exc
                    try:
                        self.transfer_stream.synchronize()
                        slot.reuse_guard = "synchronized"
                    except Exception:
                        pass
                self._commit_tp_runtime_phase(
                    raw_ready_error, f"raw-ready fence for layer {layer_idx}"
                )

                postprocess_error = None
                try:
                    self._postprocess_slot(slot)
                except Exception as exc:
                    postprocess_error = exc
                self._commit_tp_runtime_phase(
                    postprocess_error, f"postprocess for layer {layer_idx}"
                )
            slot.state = "READY"
            if method.tp_rank == 0:
                logger.debug(
                    "[SLOTLOAD] layer=%d load=%.1fms cache=%s",
                    layer_idx,
                    (time.perf_counter() - _t_load_start) * 1000,
                    layer_cache is not None,
                )
        except Exception:
            # Fence any partial transfer generation before a caller can retry
            # and overwrite this slot.  _postprocess_slot upgrades the guard
            # to "ready" when it enqueues any postprocess work.
            fence_error = None
            if slot.reuse_guard not in ("ready", "synchronized"):
                try:
                    with torch.cuda.stream(self.transfer_stream):
                        # A setup failure can happen before the normal
                        # transfer block has consumed the prior generation's
                        # guard.  Preserve that dependency before publishing
                        # a replacement raw fence.
                        if reuse_guard == "consumed":
                            self.transfer_stream.wait_event(slot.consumed_event)
                        elif reuse_guard == "ready":
                            self.transfer_stream.wait_event(slot.ready_event)
                        elif reuse_guard == "raw":
                            self.transfer_stream.wait_event(slot.raw_ready_event)
                        slot.raw_ready_event.record(self.transfer_stream)
                    slot.reuse_guard = "raw"
                except Exception:
                    try:
                        # Event publication failed, so establish the fence
                        # synchronously.  This is exception-only and never
                        # enters the healthy layerwise pipeline.
                        self.transfer_stream.synchronize()
                        slot.reuse_guard = "synchronized"
                    except Exception as exc:
                        # The CUDA context can no longer provide an overwrite
                        # fence.  Keep the slot permanently non-reusable; the
                        # process-group watchdog handles this fatal condition.
                        slot.reuse_guard = "poisoned"
                        fence_error = exc
            slot.invalidate()
            if fence_error is not None:
                raise RuntimeError(
                    f"MXFP4 failed to fence slot {slot.index} after a "
                    "transport error"
                ) from fence_error
            raise
        finally:
            if saved_affinity is not None:
                try:
                    os.sched_setaffinity(0, saved_affinity)
                except Exception:
                    logger.warning(
                        "Failed to restore CPU affinity after MXFP4 transport",
                        exc_info=True,
                    )

    def _acquire(self, layer_idx: int, method, layer: torch.nn.Module):
        self._advance_round(layer_idx)
        ready = self._find_ready_slot(layer_idx)
        if ready is not None:
            return ready, True

        if len(self.slots) == 1:
            slot = self.slots[0]
        elif self.current_slot_index is None:
            slot = self.slots[0]
        else:
            slot = self.slots[1 - self.current_slot_index]
        self._load_slot(slot, layer_idx, method, layer)
        return slot, False

    def _prefetch_successor(self, current: _Mxfp4PrefillSlot) -> None:
        if len(self.slots) == 1:
            # Single-slot mode: no second image to prefetch into.
            return
        successor_idx = self.successor_layer_idx(current.layer_idx)
        if successor_idx is None:
            return
        entry = self.registry.get(successor_idx)
        if entry is None:
            raise RuntimeError(
                f"MXFP4 successor layer {successor_idx} disappeared from registry"
            )
        next_method, next_layer = entry
        target = self.slots[1 - current.index]
        if (
            target.state == "READY"
            and target.layer_idx == successor_idx
            and target.epoch == self.epoch
        ):
            return
        self._load_slot(target, successor_idx, next_method, next_layer)

    def _log_mem_debug(self, label: str, layer_idx: int) -> None:
        """Log this rank's allocator state around a layerwise apply.

        Only meaningful on TP0 (the other ranks would just repeat it), and only
        for the first `_MXFP4_LAYERWISE_MEM_DEBUG_LEFT` applies of the process,
        which is the window that covers the steady-state per-batch working set.
        """
        global _MXFP4_LAYERWISE_MEM_DEBUG_LEFT
        if not _MXFP4_LAYERWISE_MEM_DEBUG or _MXFP4_LAYERWISE_MEM_DEBUG_LEFT <= 0:
            return
        _MXFP4_LAYERWISE_MEM_DEBUG_LEFT -= 1
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        logger.info(
            "[lwmem] %s layer=%d free=%.2f GiB alloc=%.2f reserved=%.2f "
            "peak_alloc=%.2f peak_reserved=%.2f (total=%.2f GiB)",
            label,
            layer_idx,
            free_bytes / 1024**3,
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
            torch.cuda.max_memory_allocated() / 1024**3,
            torch.cuda.max_memory_reserved() / 1024**3,
            total_bytes / 1024**3,
        )

    def apply(self, method, layer, dispatch_output):
        layer_idx = method.kt_config.layer_idx
        self._stage_report_previous()
        self._log_mem_debug("pre-acquire", layer_idx)
        slot, prefetch_hit = self._acquire(layer_idx, method, layer)
        if slot.layer_idx != layer_idx or slot.epoch != self.epoch:
            raise RuntimeError(
                "MXFP4 layerwise prefill acquired stale weights: "
                f"wanted layer={layer_idx}/epoch={self.epoch}, got "
                f"layer={slot.layer_idx}/epoch={slot.epoch}"
            )

        main_stream = None
        result = None
        compute_error = None
        try:
            main_stream = torch.cuda.current_stream(self.device)
            self._stage_note_streams(main_stream)
            self._bind_slot(slot)
            self._record_prepared_backing_on_stream(slot, main_stream)
            # Compute DIRECTLY with apply_v4_marlin_moe (baseline host-staging
            # path), bypassing the DeepSeekMxfp4MoEMethod wrapper's extra topk
            # sanitization / routing workspaces.  The layerwise manager already
            # owns the prepared slot and the standard topk layout, so the direct
            # call is functionally equivalent and keeps the forward's working
            # memory identical to the verify-baseline (the wrapper's extra
            # allocations pushed the production 24K-chunk prefill over the 32GB
            # VRAM limit).
            from sglang.srt.layers.moe.token_dispatcher import (
                StandardCombineInput,
            )
            from sglang.srt.layers.moe.topk import TopKOutputChecker
            from sglang.srt.layers.quantization.v4_marlin_moe import (
                apply_v4_marlin_moe,
                build_compacted_routing,
            )

            topk_output = dispatch_output.topk_output
            if TopKOutputChecker.format_is_standard(topk_output):
                topk_weights = topk_output.topk_weights
                topk_ids = topk_output.topk_ids
            elif TopKOutputChecker.format_is_hash(topk_output):
                topk_weights = topk_output.topk_weights
                topk_ids = topk_output.topk_ids
            else:
                raise NotImplementedError(
                    "MXFP4 layerwise prefill: unsupported topk format "
                    f"{topk_output.format}"
                )
            # One grouped GEMM per pipeline range.  Splitting the layer lets
            # range i+1's H2D run under range i's GEMM; the ranges are strictly
            # ordered on the main stream, so the summary is just their sum.  See
            # PIPELINE_RANGES.  A non-pipelined slot (the SHM path) is one range.
            ranges = slot.pipeline_ranges or [(0, slot.num_experts)]
            pipelined = slot.pipeline_ranges is not None
            _log_pipeline_ranges(ranges)
            compute_start = self._stage_mark(main_stream)
            output = None
            for range_index, (start, count) in enumerate(ranges):
                if not pipelined:
                    ready_event = slot.ready_event
                elif range_index == 0:
                    # Repacked by _load_slot, before this layer's compute.
                    ready_event = slot.range_ready_event[0]
                else:
                    # Repack on the side streams, fenced on the previous layer's
                    # GEMM of this same range, so its bytes cross the link now.
                    self._repack_from_cache(
                        slot,
                        method,
                        layer,
                        slot.layer_cache,
                        slot.range_gpu_expert_ids,
                        start,
                        count,
                        slot.range_ready_event[range_index],
                        self._range_fence(range_index, slot),
                        range_index,
                    )
                    ready_event = slot.range_ready_event[range_index]
                main_stream.wait_event(ready_event)
                # A split layer must COMPACT each range's routing.  Without it
                # `apply_v4_marlin_moe` runs `m * topk` rows per range regardless of how
                # many experts are in it -- the sanitizer maps the out-of-range ids to
                # expert 0 with weight 0 and computes them anyway -- so R ranges cost R x
                # the row work and the split is a net loss (measured -11% at R=2).  With
                # compaction each range runs only its own rows (M=16384, R=2: the GEMM goes
                # 173.7 -> 103.8 ms, 1.816x -> 1.086x dense), which is what lets the H2D
                # hide under the GEMM.
                compact = (
                    build_compacted_routing(topk_ids, topk_weights, start, count)
                    if pipelined and len(ranges) > 1
                    else None
                )
                chunk = apply_v4_marlin_moe(
                    hidden_states=dispatch_output.hidden_states,
                    prepared=self._prepared_range_view(slot, start, count),
                    topk_weights=topk_weights,
                    # The kernel's sanitizer zeroes the weight of every id
                    # outside [0, count), so shifting by `start` is what
                    # restricts this GEMM to its own range of experts.
                    topk_ids=(topk_ids - start) if start else topk_ids,
                    # DSV4 applies routed_scaling_factor after the experts (the
                    # layerwise path is not a fused-RSF backend), matching the
                    # baseline.
                    routed_scaling_factor=1.0,
                    swiglu_limit=method.moe_runner_config.swiglu_limit,
                    compact=compact,
                )
                self._consume_range(
                    slot, range_index if pipelined else None, main_stream
                )
                if output is None:
                    if len(ranges) > 1:
                        output = self._range_accumulator(chunk)
                        output.copy_(chunk)
                    else:
                        output = chunk
                else:
                    output.add_(chunk)
            if _MXFP4_LAYERWISE_STAGE_DEBUG:
                self._stage_marks = (
                    layer_idx,
                    self._stage_xfer_start,
                    self._stage_xfer_end,
                    self._stage_repack_end,
                    compute_start,
                    self._stage_mark(main_stream),
                )
            if pipelined:
                # Keep the layer-level fence meaningful: it now means "every
                # range of this layer is repacked".  The per-range events are
                # what the compute actually waits on.
                slot.ready_event.record(self.postprocess_stream)
            # Debug (2026-08-18): env-gated per-layer output dump for the
            # layerwise-vs-non-layerwise divergence root cause. Saves the MoE
            # apply input/routing/output (bf16, [num_tokens, hidden]) per layer.
            _lw_dump_dir = os.environ.get("SGLANG_LAYERWISE_DUMP_DIR")
            if method.tp_rank == 0 and getattr(self, "_lw_dump_debugged", False) is False:
                self._lw_dump_debugged = True
                logger.warning("[lw-dump] env=%r layer=%s", _lw_dump_dir, layer_idx)
            if _lw_dump_dir:
                os.makedirs(_lw_dump_dir, exist_ok=True)
                torch.save(
                    {
                        "input": dispatch_output.hidden_states.detach().float().cpu(),
                        "topk_ids": topk_ids.detach().cpu(),
                        "topk_weights": topk_weights.detach().float().cpu(),
                        "output": output.detach().float().cpu(),
                    },
                    os.path.join(
                        _lw_dump_dir, f"lw_layer{layer_idx:02d}.pt"
                    ),
                )
            result = StandardCombineInput(hidden_states=output)
        except Exception as exc:
            compute_error = exc
        finally:
            if main_stream is not None:
                try:
                    # Fence even when apply enqueues partial weight-reading
                    # work and then raises.
                    slot.consumed_event.record(main_stream)
                    slot.has_consumed_event = True
                    slot.reuse_guard = "consumed"
                    slot.state = "IN_USE"
                    self.current_slot_index = slot.index
                except Exception as exc:
                    if compute_error is None:
                        compute_error = exc
                    try:
                        main_stream.synchronize()
                        slot.reuse_guard = "synchronized"
                        slot.state = "IN_USE"
                        self.current_slot_index = slot.index
                    except Exception:
                        pass

        # A rank-local Python launch error must be observed by every peer
        # before any successful rank enters successor transport collectives.
        self._commit_tp_runtime_phase(
            compute_error, f"compute launch for layer {layer_idx}"
        )

        # GPU compute is now enqueued.  Host KT writes and successor transfer
        # scheduling can overlap it without requiring an async kt-kernel API.
        self._prefetch_successor(slot)
        self._log_mem_debug("post-prefetch", layer_idx)
        if method.tp_rank == 0:
            logger.debug(
                "KT MXFP4 layerwise prefill: layer=%d epoch=%d slot=%d %s",
                layer_idx,
                self.epoch,
                slot.index,
                "prefetch-hit" if prefetch_hit else "prime",
            )
        return result


def _interleave_host_cache(layer_cache: Dict[str, torch.Tensor]) -> None:
    """Spread the pinned host cache over every NUMA node before pinning.

    The scheduler thread (and everything it spawns) inherits kt-kernel's
    node-0 ``MPOL_BIND``; a ~7 GiB pinned layer allocated from it lands
    entirely on node 0 and the box OOM-kills the scheduler with node 1 free
    (measured 2026-09-21: ``nodemask=0`` on the OOM line). Mirrors the engram
    host table's fix (layers/engram.py::_interleave_across_nodes).
    """
    import ctypes

    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
    except OSError:
        logger.warning(
            "[SLOTLOAD] libnuma unavailable; the pinned host cache follows the "
            "thread's memory policy and may not fit on one node"
        )
        return
    libnuma.numa_num_configured_nodes.restype = ctypes.c_int
    libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
    libnuma.numa_bitmask_setbit.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    libnuma.numa_interleave_memory.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    libnuma.numa_bitmask_free.argtypes = (ctypes.c_void_p,)
    nodes = libnuma.numa_num_configured_nodes()
    if nodes < 2:
        return
    mask = libnuma.numa_allocate_nodemask()
    if not mask:
        return
    try:
        for node in range(nodes):
            libnuma.numa_bitmask_setbit(ctypes.c_void_p(mask), node)
        for t in layer_cache.values():
            libnuma.numa_interleave_memory(
                ctypes.c_void_p(t.data_ptr()),
                ctypes.c_size_t(t.numel() * t.element_size()),
                ctypes.c_void_p(mask),
            )
    finally:
        libnuma.numa_bitmask_free(ctypes.c_void_p(mask))


def _build_mxfp4_host_cache_entry(
    method,
    layer_idx: int,
    *,
    pinned: bool,
    num_experts: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    include_gpu_experts: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Pack every CPU expert of a layer into a host cache tensor dict.

    Standalone (no manager/context needed) so the eager background builder can
    run at startup.  Shapes/dtypes mirror the slot raw image exactly (verified
    against the live gpu_layer): weights int8 [E, ...], scales bf16 (the KT
    write emits bf16; the per-batch H2D upcasts to the slot's fp32 scales).

    Returns None (and records None in the shared eager cache) when the layer
    has no CPU experts, TP>1, allocation fails, or the pack fails; callers
    then fall back to the per-batch packing path.
    """
    if get_tensor_model_parallel_world_size() != 1:
        return None
    if num_experts is None or hidden_size is None or intermediate_size is None:
        if method._full_init_args is None:
            return None
        hidden_size, intermediate_size, _ = method._full_init_args
        num_experts = method.global_num_experts
    host_dtypes = _Mxfp4LayerwisePrefillManager._host_cache_dtypes()
    shapes = {
        "w13_weight": (num_experts, 2 * intermediate_size, hidden_size // 2),
        "w13_weight_scale_inv": (
            num_experts,
            2 * intermediate_size,
            hidden_size // 32,
        ),
        "w2_weight": (num_experts, hidden_size, intermediate_size // 2),
        "w2_weight_scale_inv": (
            num_experts,
            hidden_size,
            intermediate_size // 32,
        ),
    }
    cpu_ids = [
        e
        for e in range(num_experts)
        if include_gpu_experts or not bool(method.gpu_experts_mask[e])
    ]
    if not cpu_ids:
        return None
    try:
        layer_cache = {
            name: torch.empty(
                shapes[name],
                dtype=host_dtypes[name],
                device="cpu",
            )
            for name in _Mxfp4PrefillSlot.RAW_NAMES
        }
    except RuntimeError:
        logger.warning(
            "[SLOTLOAD] layer=%d host-weight cache alloc failed "
            "(%.1f GiB, pinned=%s); falling back to per-batch packing.",
            layer_idx,
            sum(
                shapes[n][0]
                * shapes[n][1]
                * shapes[n][2]
                * host_dtypes[n].itemsize
                for n in _Mxfp4PrefillSlot.RAW_NAMES
            )
            / 1024**3,
            pinned,
        )
        return None
    if method.tp_rank == 0:
        try:
            if hasattr(method.wrapper, "write_weights_to_buffer_batch"):
                with _KT_CPU_LOCK:
                    method.wrapper.write_weights_to_buffer_batch(
                        1,
                        cpu_ids,
                        [
                            int(layer_cache["w13_weight"][e].data_ptr())
                            for e in cpu_ids
                        ],
                        [
                            int(
                                layer_cache["w13_weight_scale_inv"][
                                    e
                                ].data_ptr()
                            )
                            for e in cpu_ids
                        ],
                        [
                            int(layer_cache["w2_weight"][e].data_ptr())
                            for e in cpu_ids
                        ],
                        [
                            int(
                                layer_cache["w2_weight_scale_inv"][e].data_ptr()
                            )
                            for e in cpu_ids
                        ],
                    )
            else:
                with _KT_CPU_LOCK:
                    for expert_id in cpu_ids:
                        pointers = {
                            name: [
                                int(layer_cache[name][expert_id].data_ptr())
                            ]
                            for name in _Mxfp4PrefillSlot.RAW_NAMES
                        }
                        method.wrapper.submit_write_weight_scale_to_buffer(
                            1,
                            expert_id,
                            pointers["w13_weight"],
                            pointers["w13_weight_scale_inv"],
                            pointers["w2_weight"],
                            pointers["w2_weight_scale_inv"],
                        )
                    method.wrapper.sync_write_weight_scale_to_buffer()
        except Exception:
            logger.warning(
                "[SLOTLOAD] layer=%d host-weight cache pack failed; "
                "falling back to per-batch packing.",
                layer_idx,
                exc_info=True,
            )
            return None
    if pinned:
        # Pin AFTER the pack: cudaHostRegister locks the pages that already
        # hold the weights (cost ~2.5% page tables) instead of paying
        # pin_memory=True for a second copy of every layer. The interleave
        # must run before any page is faulted, so it is first.
        _interleave_host_cache(layer_cache)
        for name, tensor in layer_cache.items():
            err = torch.cuda.cudart().cudaHostRegister(
                tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0
            )
            if int(err) != 0:
                logger.warning(
                    "[SLOTLOAD] layer=%d cudaHostRegister(%s) failed (err %s); "
                    "the layer stays pageable.",
                    layer_idx,
                    name,
                    int(err),
                )
                for prev_name, prev in layer_cache.items():
                    if prev_name == name:
                        break
                    torch.cuda.cudart().cudaHostUnregister(prev.data_ptr())
                return None
    return layer_cache


def _release_mxfp4_host_cache(signature: tuple) -> int:
    """Drop the eager pinned host weight cache for a signature; return bytes freed.

    SGLANG_KT_RELEASE_HOST_CACHE=1 calls this right before the layerwise slot
    allocation.  The cache is the layerwise prefill's transfer SOURCE and the
    largest host-side object the run creates (~316 GB for 40 layers, pinned and
    unreclaimable), and with the slot live it is the prime suspect in the
    +40 ms decode step that appears exactly when the slot is resident: a page
    cache of that size on a box already holding the 193 GB engram tables makes
    every allocator/device op pay page-table and reclaim pressure that has
    nothing to do with the attention/MoE math.  libnuma also reports the
    interleave as costing throughput on a single-node-bound scheduler.

    Dropping it costs one re-pack of 0.9 s/layer on the FIRST layerwise prefill
    (the per-batch packing path still works; `_mxfp4_eager_cache_worker`
    rebuilds it too), which is a one-time price against a per-step one.
    """
    if os.environ.get("SGLANG_KT_RELEASE_HOST_CACHE") != "1":
        return 0
    freed = 0
    with _MXFP4_EAGER_CACHE_LOCK:
        layers = _MXFP4_EAGER_HOST_CACHE.pop(signature, {}) or {}
    for layer_cache in layers.values():
        if not layer_cache:
            continue
        for tensor in layer_cache.values():
            nbytes = tensor.numel() * tensor.element_size()
            try:
                torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())
            except Exception:
                pass
            freed += nbytes
        # Pinned pages are unreclaimable until the tensors die; break the refs
        # here (and the manager's own per-layer copies below) so the host
        # allocator actually gets them back.
        layer_cache.clear()
    _KT_HOST_CACHE_RELEASED.add(signature)
    print(
        "[host-free] released pinned host weight cache %.2f GiB (%d layers)"
        % (freed / 1024**3, len(layers)),
        flush=True,
    )
    return freed


_KT_HOST_CACHE_RELEASED: set = set()


def _mxfp4_eager_cache_worker(signature: tuple) -> None:
    """Build every registered layer's pinned host cache in the background.

    Runs once at startup after the model is loaded (all layers registered).
    The cold first layerwise prefill then only does the fast pinned H2D
    instead of paying ~0.9s/layer for pack + cudaHostAlloc synchronously.
    """
    registry = _MXFP4_PREFILL_LAYER_REGISTRY.get(signature, {})
    order = sorted(registry)
    if not order:
        return
    built = 0
    num_cached = 0
    # With KT_PREFILL_RELEASE_RESIDENTS the residents' GPU raw is gone for the
    # whole session, so the layerwise repack has to source them from this cache
    # too (see _repack_from_cache).
    include_gpu_experts = os.environ.get("KT_PREFILL_RELEASE_RESIDENTS") == "1"
    for layer_idx in order:
        entry = registry.get(layer_idx)
        if entry is None:
            continue
        method, layer = entry
        if not _mxfp4_pipeline_requested(method) or method.tp_rank != 0:
            continue
        cache = _build_mxfp4_host_cache_entry(
            method,
            layer_idx,
            pinned=True,
            include_gpu_experts=include_gpu_experts,
        )
        with _MXFP4_EAGER_CACHE_LOCK:
            _MXFP4_EAGER_HOST_CACHE.setdefault(signature, {})[layer_idx] = cache
        if cache is not None:
            built += 1
            # The cached row count, i.e. the same predicate the builder uses --
            # reporting the mask's complement instead understated it by every
            # resident expert whenever include_gpu_experts was on.
            num_cached = len(
                [
                    e
                    for e in range(method.global_num_experts)
                    if include_gpu_experts
                    or not bool(method.gpu_experts_mask[e])
                ]
            )
    # info, not debug: a layer that failed cudaHostRegister returns None and is
    # dropped from `built`, so this count is the only startup evidence that the
    # in-place pin actually took. Runs once, so the line is cheap.
    logger.info(
        "[SLOTLOAD] eager host-weight cache build done: %d/%d layers "
        "(%d experts cached each, pinned).",
        built,
        len(order),
        num_cached,
    )
    # The host side is now covered for every layer, which is the precondition
    # for the layerwise slots to be worth materializing -- and this is the last
    # moment before traffic where VRAM is not yet holding a batch's working
    # memory.  Do it on this thread so the GPU allocator is touched from one
    # place.  See maybe_init_mxfp4_layerwise_slots.
    #
    # Wait for the KV pool first: that allocation is what the slot footprint
    # must not be charged against.  Sizing the pool with the slots already live
    # pushes the required --mem-fraction-static from 0.46 to 0.857 (measured
    # 2026-09-22), because the draft model's load, which runs after this worker
    # starts, measures its own `mem usage` as a delta of process memory and
    # therefore absorbs whatever the slots took.
    if _MXFP4_MEMORY_POOL_READY.wait(timeout=900):
        try:
            maybe_init_mxfp4_layerwise_slots()
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("[SLOTLOAD] eager layerwise slot init failed: %s", exc)


def maybe_start_mxfp4_host_cache_build() -> None:
    """Start the background pinned host-cache builder (idempotent).

    Called after model loading completes; the worker snapshots the layer
    registry (all layers already registered) and packs each layer's CPU
    experts into pinned host memory so the first layerwise prefill is warm.
    """
    global _MXFP4_EAGER_CACHE_BUILDER
    if _MXFP4_EAGER_CACHE_BUILDER is not None:
        return
    if os.environ.get("SGLANG_MXFP4_HOST_WEIGHT_CACHE", "1") == "0":
        return
    if not _MXFP4_PREFILL_LAYER_REGISTRY:
        return
    with _MXFP4_EAGER_CACHE_LOCK:
        if _MXFP4_EAGER_CACHE_BUILDER is not None:
            return
        signatures = list(_MXFP4_PREFILL_LAYER_REGISTRY.keys())
        _MXFP4_EAGER_CACHE_BUILDER = threading.Thread(
            target=_mxfp4_eager_cache_worker_all,
            args=(signatures,),
            name="mxfp4-host-cache-build",
            daemon=True,
        )
        _MXFP4_EAGER_CACHE_BUILDER.start()


def _mxfp4_eager_cache_worker_all(signatures: list) -> None:
    for signature in signatures:
        try:
            _mxfp4_eager_cache_worker(signature)
        except Exception:
            logger.warning(
                "MXFP4 eager host-cache build failed for signature %s",
                signature,
                exc_info=True,
            )


def _mxfp4_pipeline_signature(method, layer: torch.nn.Module) -> tuple:
    device = _layer_device(layer)
    return (
        str(device),
        method.kt_config.weight_path,
        method.kt_config.num_layers,
        method.global_num_experts,
        method._full_init_args,
    )


def _mxfp4_pipeline_requested(method) -> bool:
    return (
        method.gpu_prefill_token_threshold > 0
        and (method.kt_config.method or "").upper() == "MXFP4"
    )


_MXFP4_MEMORY_POOL_READY = threading.Event()


def mark_mxfp4_memory_pool_ready() -> None:
    """Tell the eager layerwise-slot builder that the KV pool is allocated.

    The slot footprint must be charged to whatever is left after the pool, not
    counted as model weights.  See `maybe_init_mxfp4_layerwise_slots` for the
    measured reason (the pool-sizing floor goes 0.46 -> 0.857 otherwise).
    """
    _MXFP4_MEMORY_POOL_READY.set()


def maybe_init_mxfp4_layerwise_slots() -> None:
    """Materialize the layerwise GPU slots while VRAM is still free.

    The slots are allocated lazily today -- from inside the first
    threshold-qualified prefill forward (see
    ``_get_or_initialize_mxfp4_layerwise_manager``, whose caller is the apply
    path).  By then the batch's own working memory is live and the free-VRAM
    reading is at its session minimum, so on a tight config the allocation
    misses by a few percent and the pipeline is disabled for the session: the
    triggering request silently falls through to the hybrid CPU/GPU MoE path,
    and GPU layerwise prefill never runs at all.  Measured 2026-09-22 on
    V4.1/SM89 (--mem-fraction-static 0.55): the first attempt needed 4.22 GiB
    against 4.17 GiB free, on a box that had 15.76 GB free right after pool
    allocation, and the retry loop then flip-flopped disabled/enabled once per
    batch, ~4x slower than plain AMX.

    Nothing about the allocation depends on a request, so do it here instead.
    Same registry walk and same guard as the eager host-cache build, and one
    manager per signature because a manager owns every layer of its signature.

    Best-effort by construction: a failure lands in the same
    ``_MXFP4_LAYERWISE_DISABLED_REASONS`` map the lazy path uses, so the first
    request then behaves exactly as it does today.
    """
    if not torch.cuda.is_available():
        return
    # Only the largest registry gets slots.  Draft/MTP stages register under a
    # separate signature (their `weight_key_prefix`/`_registry_prefix` slip past
    # `_register_mxfp4_prefill_layer`'s filter, which is meant to exclude them)
    # and they only ever see a 6-token decode-proposal batch, so they can never
    # reach a threshold-qualified prefill and their slots are pure waste --
    # measured 2026-09-22: the 3-layer MTP signature took 4.88 GiB and that is
    # what pushed the box from 8.10 GiB free to 3.42 GiB free and into a CUDA
    # OOM on the first prefill (`Tried to allocate 1024 MiB, 77 MiB free`).
    # They keep the lazy path, so a threshold-qualified batch on one of them
    # still initializes on demand.
    if not _MXFP4_PREFILL_LAYER_REGISTRY:
        return
    target_signature = max(
        _MXFP4_PREFILL_LAYER_REGISTRY, key=lambda s: len(_MXFP4_PREFILL_LAYER_REGISTRY[s])
    )
    for signature, registry in list(_MXFP4_PREFILL_LAYER_REGISTRY.items()):
        if signature != target_signature:
            logger.info(
                "[SLOTLOAD] skipping eager layerwise slots for the %d-layer "
                "signature (draft/MTP stage; lazy path retained).",
                len(registry),
            )
            continue
        if (
            not registry
            or signature in _MXFP4_LAYERWISE_MANAGERS
            or signature in _MXFP4_LAYERWISE_DISABLED_REASONS
        ):
            continue
        for layer_idx in sorted(registry):
            method, layer = registry[layer_idx]
            if not _mxfp4_pipeline_requested(method) or method.tp_rank != 0:
                continue
            if not _mxfp4_pipeline_runtime_supported(method, layer):
                continue
            try:
                manager = _get_or_initialize_mxfp4_layerwise_manager(method, layer)
            except Exception as exc:  # pragma: no cover - diagnostic only
                # An exception here means init got PAST the OOM handling (which
                # already records a disabled reason) and raised -- e.g. the
                # manager-setup IndexError.  Leaving the signature live let the
                # FIRST threshold prefill re-run init and raise again, this
                # time fatal inside forward.  Disable instead so every later
                # request takes the hybrid path.
                logger.warning(
                    "[SLOTLOAD] eager layerwise slot init raised: %s; "
                    "disabling the layerwise pipeline for this signature",
                    exc,
                )
                _disable_mxfp4_layerwise_pipeline(
                    signature, f"eager init raised: {exc}"
                )
                break
            if manager is not None:
                # The host cache covers every layer (this runs at the end of the
                # eager build), which is exactly the precondition
                # _maybe_free_full_raw checks, and it is what frees the ~7.91 GiB
                # full-layer raw image.  Dropping it here is what makes the
                # layerwise footprint fit: raw(7.91) + prepared(6.72) = 14.63 GiB
                # against 17.18 GiB left after the 14.29 GiB of model weights
                # leaves no room for a prefill chunk's activations, while
                # prepared alone (6.72) does.
                #
                # It has to happen here rather than on the first layerwise
                # forward (where apply() calls it today): by then the batch's
                # working memory is live, so the free reading is at its session
                # minimum and the allocation that the freed raw would have paid
                # for has already failed.
                manager._maybe_free_full_raw()
                logger.info(
                    "[SLOTLOAD] layerwise GPU slots materialized at startup "
                    "(%d layers on signature), %.2f GiB free after "
                    "(full-layer raw image released).",
                    len(registry),
                    torch.cuda.mem_get_info()[0] / 1024**3,
                )
            # One manager per signature either way; `break` also covers the
            # disabled-reason case, where retrying every layer is pointless.
            break


def _mxfp4_pipeline_backend_supported(method, layer: torch.nn.Module) -> bool:
    if not _mxfp4_pipeline_requested(method) or not torch.cuda.is_available():
        return False
    if method.gpu_method.__class__.__name__ != "DeepSeekMxfp4MoEMethod":
        return False
    # Respect both diagnostic overrides.  The default capability-driven path
    # uses the prepared Marlin backend on Ada and Blackwell consumer GPUs.
    if os.environ.get("SGLANG_V4_USE_TRITON_KERNELS") in ("0", "1"):
        return False
    device = _layer_device(layer)
    return torch.cuda.get_device_capability(device) in ((8, 9), (12, 0))


def _mxfp4_pipeline_runtime_supported(method, layer: torch.nn.Module) -> bool:
    return (
        _mxfp4_pipeline_backend_supported(method, layer)
        and all(
            hasattr(layer, name) for name in _Mxfp4PrefillSlot.RAW_NAMES
        )
    )


def _mxfp4_raw_slot_storage_nbytes(
    *, num_experts: int, hidden_size: int, intermediate_size: int
) -> int:
    """Return the bytes in one full-expert raw MXFP4 slot.

    This mirrors ``DeepSeekMxfp4MoEMethod.create_weights`` without creating
    any tensors.  The resulting value is used only to limit KV-cache sizing;
    the actual full-expert slot remains lazy.
    """
    int8_size = torch.tensor([], dtype=torch.int8).element_size()
    float32_size = torch.tensor([], dtype=torch.float32).element_size()
    return (
        num_experts * (2 * intermediate_size) * (hidden_size // 2) * int8_size
        + num_experts * hidden_size * (intermediate_size // 2) * int8_size
        + num_experts
        * (2 * intermediate_size)
        * (hidden_size // 32)
        * float32_size
        + num_experts
        * hidden_size
        * (intermediate_size // 32)
        * float32_size
    )


def get_mxfp4_layerwise_prefill_reservation_bytes() -> int:
    """Return the unallocated MXFP4 slot capacity needed after a long request.

    KV-cache profiling normally consumes all currently free VRAM.  With
    layerwise slots allocated lazily, that would leave no capacity when the
    first threshold-qualified request arrives.  Account for two raw and two
    prepared slots here, but do not materialize them until that request.
    """
    total_bytes = 0
    for signature, registry in _MXFP4_PREFILL_LAYER_REGISTRY.items():
        if (
            not registry
            or signature in _MXFP4_LAYERWISE_MANAGERS
            or signature in _MXFP4_LAYERWISE_DISABLED_REASONS
        ):
            continue

        first_layer_idx = min(registry)
        method, layer = registry[first_layer_idx]
        if not _mxfp4_pipeline_runtime_supported(method, layer):
            continue

        init_args = getattr(method, "_full_init_args", None)
        num_experts = getattr(method, "global_num_experts", None)
        if init_args is None or num_experts is None:
            continue
        hidden_size, intermediate_size, _ = init_args

        from sglang.srt.layers.quantization.v4_marlin_moe import (
            get_v4_mxfp4_marlin_storage_nbytes,
        )

        raw_slot_bytes = _mxfp4_raw_slot_storage_nbytes(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        prepared_slot_bytes = get_v4_mxfp4_marlin_storage_nbytes(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        total_bytes += 2 * (raw_slot_bytes + prepared_slot_bytes)

    return total_bytes


def _register_mxfp4_prefill_layer(method, layer: torch.nn.Module) -> None:
    # Draft/MTP stages (weight_key_prefix="mtp.{stage}" or a "mtp." layer
    # prefix) only run decode proposals; do not register them for the
    # layerwise GPU prefill, which is owned by the target layers. Local
    # addition; upstream has no KT draft.
    _kp = getattr(getattr(method, "kt_config", None), "weight_key_prefix", None)
    _pfx = getattr(layer, "_registry_prefix", "") or ""
    if _kp is not None or "mtp." in _pfx:
        return
    import os as _os

    if _os.environ.get("SGLANG_DSV4_MXFP4_DBG"):
        print(
            f"[MXFP4-DBG] REGISTER layer={getattr(getattr(method, 'kt_config', None), 'layer_idx', '?')} "
            f"wkp={getattr(getattr(method, 'kt_config', None), 'weight_key_prefix', None)} "
            f"method_id={id(method)} layer_id={id(layer)} method_cls={type(method).__name__}",
            flush=True,
        )
    if not _mxfp4_pipeline_requested(method):
        return
    signature = _mxfp4_pipeline_signature(method, layer)
    method._mxfp4_pipeline_signature = signature
    registry = _MXFP4_PREFILL_LAYER_REGISTRY.setdefault(signature, {})
    layer_idx = method.kt_config.layer_idx
    existing = registry.get(layer_idx)
    if existing is not None:
        if existing[0] is method and existing[1] is layer:
            return  # idempotent re-registration
        # A different instance claims the same layer_idx: this is the
        # local DSPARK draft/MTP stage colliding with the target layers
        # (draft layer_idx 0..2 == target 0..2). Drafts must NOT join the
        # layerwise GPU prefill, so skip instead of raising.
        import os as _os

        if _os.environ.get("SGLANG_DSV4_MXFP4_DBG"):
            print(
                f"[MXFP4-DBG] SKIP-DUP reg layer={layer_idx} "
                f"new_wkp={getattr(getattr(method, 'kt_config', None), 'weight_key_prefix', None)} "
                f"old_method={type(existing[0]).__name__}",
                flush=True,
            )
        return
    registry[layer_idx] = (method, layer)


def _all_tp_ranks_succeeded(local_success: bool) -> bool:
    if not dist.is_initialized() or get_tensor_model_parallel_world_size() == 1:
        return local_success
    status = torch.tensor([int(local_success)], dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return bool(status.item())


def _any_tp_rank_true(local_value: bool) -> bool:
    if not dist.is_initialized() or get_tensor_model_parallel_world_size() == 1:
        return local_value
    status = torch.tensor([int(local_value)], dtype=torch.int32, device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group)
    return bool(status.item())


def _disable_mxfp4_layerwise_pipeline(signature: tuple, reason: str) -> None:
    _MXFP4_LAYERWISE_DISABLED_REASONS[signature] = reason
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if get_tensor_model_parallel_rank() == 0:
        logger.warning(
            "KT MXFP4 layerwise prefill disabled; using hybrid CPU/GPU MoE: %s",
            reason,
        )


def maybe_teardown_mxfp4_layerwise_on_oom() -> None:
    """Free + disable every layerwise manager after a prefill forward OOM.

    Called by the scheduler when a prefill forward raises CUDA OOM.  The OOM
    (attention or MoE workspace) can happen at ANY layer of the forward, so the
    MoE-apply-level catch cannot cover it; tearing the layerwise slots down here
    (prepared + staging, ~4 GiB) and disabling the pipeline lets the scheduler's
    retry take the hybrid CPU MoE path instead of OOMing 4x and failing the
    request with HTTP 500.
    """
    global _MXFP4_LAYERWISE_MANAGERS
    if not _MXFP4_LAYERWISE_MANAGERS:
        return
    for signature in list(_MXFP4_LAYERWISE_MANAGERS.keys()):
        manager = _MXFP4_LAYERWISE_MANAGERS.pop(signature, None)
        if manager is not None:
            try:
                manager.teardown()
            except Exception:
                pass
        _MXFP4_LAYERWISE_DISABLED_REASONS[signature] = (
            "CUDA OOM during prefill forward; fell back to hybrid CPU/GPU MoE"
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.warning(
        "MXFP4 layerwise prefill disabled after CUDA OOM; subsequent prefills "
        "use the hybrid CPU/GPU MoE path."
    )


def _try_mxfp4_initialization(factory):
    """Run an allocation without leaking exception tracebacks to the caller."""

    try:
        return factory(), None
    except torch.cuda.OutOfMemoryError as exc:
        return None, ("oom", str(exc))
    except Exception as exc:
        return None, ("fatal", f"{type(exc).__name__}: {exc}")


# KT_PREFILL_RELEASE_RESIDENTS=1: the registered layers' raw slices are gone -- released
# at LOAD by `mxfp4_deepseek.process_weights_after_loading` (so a large N fits the load
# at all), and again at slot-allocation time by `_release_resident_marlin_copies`.  The
# full-GPU prefill fallback must therefore never take its Phase 1 D2D shortcut for them.
# Seeded from the env rather than only set by the hook: the raw is gone from load onward,
# so the flag has to be in effect for every prefill, including one that runs before slot
# allocation or when the slot is never allocated at all.
_KT_RESIDENTS_RAW_RELEASED = os.environ.get("KT_PREFILL_RELEASE_RESIDENTS") == "1"


def _release_resident_marlin_copies(signature: tuple) -> None:
    global _KT_RESIDENTS_RAW_RELEASED
    """KT_PREFILL_RELEASE_RESIDENTS=1: report the residents' raw release.

    The raw slice is normally already gone by now -- `mxfp4_deepseek` releases it at
    LOAD under this same env, which is what lets a large N fit the load at all.  This
    hook is the idempotent backstop and the measurement point.

    The falsification test for the one claim the prefill-scoped release design rests
    on (see PREFILL_RELEASE_RESIDENTS.md): a release made AFTER the KV/static pool has
    been sized is not absorbed, unlike every load-time release measured so far -- the
    tk-slot's 6.72 GiB, pinning the raw slice's 2.65 GiB and a 0 GiB control all
    landed on the identical "13.44 MiB free".  This runs at slot-allocation time,
    which IS after the pool sizing (the pool is built in kv_cache_configurator, before
    the SLOTLOAD lines), and prints the free memory either side of the drop so the
    claim is decided rather than assumed.

    The RAW slice is what goes, leaving the Marlin copy that decode reads.  The
    fallback's Phase 1 is the raw's only reader, and `_KT_RESIDENTS_RAW_RELEASED`
    (seeded from this same env) sends it down the all-CPU arm instead.

    An earlier version of this hook sat in `_Mxfp4LayerwisePrefillManager.apply`, which
    is circular -- that path only runs once the slot exists, and the slot is exactly
    what fails to allocate without the release.
    """
    if os.environ.get("KT_PREFILL_RELEASE_RESIDENTS") != "1":
        return
    free_before = torch.cuda.mem_get_info()[0] / 1024**3
    alloc_before = torch.cuda.memory_allocated() / 1024**3
    dropped = 0
    registry = _MXFP4_PREFILL_LAYER_REGISTRY.get(signature, {})
    for _layer_idx, entry in registry.items():
        if entry is None:
            continue
        _method, layer = entry
        prep = getattr(layer, "_v4_marlin_weights", None)
        if prep is None or isinstance(prep, tuple):
            continue
        # ONLY the raw slice is released; the Marlin copy stays.  Decode reads
        # `_v4_marlin_weights` and never the raw, so decode is unaffected by this --
        # which is what makes the change a plain load-time one with NO phase scoping
        # and NO rebuild.  The raw is 2.65 GiB at N=3 against the 0.9 GiB the slot was
        # short after the Marlin copies alone were released, and Phase 1 of the
        # full-GPU prefill fallback is its only reader, so
        # `_KT_RESIDENTS_RAW_RELEASED` sends every expert down the CPU-staging path
        # instead (the `original_layer=None` arm of `_prepare_weight_fp8`, i.e. "all
        # experts from CPU"; the KT host pool holds all 384).
        for _name in _Mxfp4PrefillSlot.RAW_NAMES:
            _raw_t = getattr(layer, _name, None)
            if _raw_t is not None:
                dropped += _raw_t.numel() * _raw_t.element_size()
                setattr(layer, _name, None)
    _KT_RESIDENTS_RAW_RELEASED = True
    gc.collect()
    # The flush is what hands these 2.47 GiB back to the driver instead of the
    # caching allocator, so the slot (and every decode step after it) has to
    # grow a fresh pool rather than reuse these blocks.  Gated so the two
    # behaviours can be compared in one session.
    if os.environ.get("SGLANG_KT_NO_EMPTY_CACHE") != "1":
        torch.cuda.empty_cache()
    free_after = torch.cuda.mem_get_info()[0] / 1024**3
    alloc_after = torch.cuda.memory_allocated() / 1024**3
    print(
        "[prefill-free] layers=%d dropped=%.2f GiB  alloc %.3f -> %.3f GiB  "
        "free %.3f -> %.3f GiB"
        % (
            len(registry),
            dropped / 1024**3,
            alloc_before,
            alloc_after,
            free_before,
            free_after,
        ),
        flush=True,
    )


def _allocate_mxfp4_slot_storage(context: SharedFullContext):
    from sglang.srt.layers.quantization.v4_marlin_moe import (
        allocate_v4_mxfp4_marlin,
    )

    slot0_raw = {
        name: getattr(context.gpu_layer, name).data
        for name in _Mxfp4PrefillSlot.RAW_NAMES
    }
    # Single prepared slot, no extra raw copy. The upstream 2-slot ping-pong
    # needs ~7GB (1 raw empty_like + 2 marlin) and OOMs after a 1M-context KV
    # pool on the 32GB card, forcing CPU MoE prefill. Single-slot (~3.2GB)
    # mirrors the local arch's lazy 1-prepared design: the successor prefetch
    # is disabled but the layerwise prefill footprint fits.
    # Sizes/device come from the frozen snapshot, not the live tensors: the
    # Marlin path below frees the full-layer raw, and an OOM retry re-enters
    # this function with empty Attributes -- reading shape[0] off those would
    # allocate a 0-expert slot.
    _w13_shape, _, device = context._raw_weight_shapes["w13_weight"]
    _w2_shape = context._raw_weight_shapes["w2_weight"][0]
    num_experts = _w13_shape[0]
    hidden_size = _w13_shape[2] * 2
    intermediate_size = _w2_shape[2] * 2

    if os.environ.get("SGLANG_KT_PREFILL_TK_SLOT") == "1":
        # The prepared Marlin copy is a byte PERMUTATION -- four tensors, 4.22 + 2.11
        # + 0.27 + 0.14 = 6.72 GiB for a full 384-expert layer -- whereas the
        # triton_kernels form is a STRIDED ALIAS of the same raw tensors (zero extra
        # bytes).  That is the same asymmetry that makes decode's resident slice cost
        # 1.85x in Marlin and 1.00x on the triton path.  Measured 2026-09-23, the
        # layerwise machinery's 7.91 (raw) + 6.72 (prepared) GiB is what leaves only
        # 2.55 GiB of the 31.47 GiB card free, which is why the slot allocation fails
        # at every (N, mfs) with resident experts -- the free memory was invariant to
        # both, 6 configurations and 2 code variants.  Handing the slot the strided
        # form frees those 6.72 GiB and is what lets >=3 resident experts and the
        # layerwise prefill coexist.
        from sglang.srt.layers.quantization.v4_triton_kernels_moe import (
            convert_v4_weights_to_triton_kernels,
        )

        # NOTE the container convention: this returns one entry PER SLOT, and the
        # caller does `slot_marlin_prepared[0]`, so the tk form must be wrapped in a
        # 1-tuple to mirror the Marlin `for _ in range(1)` build -- otherwise the
        # slot receives only w13_swiz and `prepared.w13` fails with
        # "'Tensor' object has no attribute 'w13'".
        return None, (
            convert_v4_weights_to_triton_kernels(
                slot0_raw["w13_weight"],
                slot0_raw["w13_weight_scale_inv"],
                slot0_raw["w2_weight"],
                slot0_raw["w2_weight_scale_inv"],
            ),
        )

    # The full-layer raw (~7.91 GiB at 384 experts, printed by this hook) is
    # only needed for the shapes captured above and for the post-init fallback,
    # and the eager host cache already covers every layer.  Its own free point
    # -- `_maybe_free_full_raw` -- runs only after a SUCCESSFUL manager setup,
    # i.e. after the very allocation that needs the room: with 3 resident
    # experts the card had 3.77 GiB free against the slot's 6.74 GiB, so init
    # failed at every (N, mfs) while residents were untouched.  Free it HERE,
    # before the prepared alloc: 3.77 + 7.91 >= 6.74, residents intact, no
    # release, no rebuild.  Dropping the local `.data` refs first is required
    # or the attribute swap frees 0 bytes (same trap the release probe hit).
    #
    # Gated so the default path stays bit-identical: without the env var the
    # raw survives the attempt exactly as before (and the slot fails the same
    # way it always did at N>0).  Manager/slot construction must not read the
    # live tensors either way -- see _Mxfp4PrefillSlot's frozen_shapes.
    del slot0_raw
    if os.environ.get("SGLANG_KT_SLOT_FREE_RAW_FIRST") == "1":
        freed = 0
        for name in _Mxfp4PrefillSlot.RAW_NAMES:
            t = getattr(context.gpu_layer, name)
            freed += t.numel() * t.element_size()
            setattr(
                context.gpu_layer,
                name,
                torch.nn.Parameter(
                    torch.empty(0, device=t.device), requires_grad=False
                ),
            )
            context.original_params.pop(name, None)
            context.original_buffers.pop(name, None)
        if freed:
            # gc.collect() drops the last references so the blocks reach the
            # caching allocator's free list -- but do NOT empty_cache() after
            # it.  The freed w13_weight block IS the slot's first tensor
            # (384 x 4608 x 2560 int8 = 4.22 GiB, exactly what
            # allocate_v4_mxfp4_marlin asks for first), so the slot is served
            # as a cache hit; handing the blocks back to the driver with
            # empty_cache() would make the slot a fresh cudaMalloc and leave
            # the allocator with no pool for the decode steps that follow.
            gc.collect()
            if os.environ.get("SGLANG_KT_NO_EMPTY_CACHE") != "1":
                torch.cuda.empty_cache()
            print(
                "[slot-free] released full-layer raw %.2f GiB before prepared alloc"
                % (freed / 1024**3),
                flush=True,
            )

    # Single prepared slot, no extra raw copy. The upstream 2-slot ping-pong
    slot_marlin_prepared = tuple(
        allocate_v4_mxfp4_marlin(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        )
        for _ in range(1)
    )
    return None, slot_marlin_prepared


def _initialize_mxfp4_layerwise_pipeline(method, layer: torch.nn.Module) -> None:
    global _SHARED_FULL_CONTEXT

    if not _mxfp4_pipeline_backend_supported(method, layer):
        return
    signature = getattr(method, "_mxfp4_pipeline_signature", None)
    if signature is None:
        signature = _mxfp4_pipeline_signature(method, layer)
        method._mxfp4_pipeline_signature = signature
    if signature in _MXFP4_LAYERWISE_MANAGERS:
        return
    if signature in _MXFP4_LAYERWISE_DISABLED_REASONS:
        return

    context, context_failure = _try_mxfp4_initialization(
        lambda: SharedFullContext(
            layer=layer,
            init_args=method._full_init_args,
            global_num_experts=method.global_num_experts,
            moe_runner_config=method.moe_runner_config,
            defer_cpu_buffers=True,
        )
    )

    if not _all_tp_ranks_succeeded(context_failure is None):
        context = None
        fatal_error = _any_tp_rank_true(
            context_failure is not None and context_failure[0] == "fatal"
        )
        if fatal_error:
            message = "MXFP4 slot 0 initialization failed on at least one TP rank"
            if context_failure is not None:
                message = f"{message}: {context_failure[1]}"
            raise RuntimeError(message)
        context_failure = None
        _disable_mxfp4_layerwise_pipeline(
            signature, "full-layer slot 0 allocation failed on at least one TP rank"
        )
        return
    if context is None or not getattr(context, "_is_mxfp4_quant", False):
        raise RuntimeError("MXFP4 layerwise prefill built a non-MXFP4 full context")

    # Give the slot the residents' Marlin copies when asked (KT_PREFILL_RELEASE_RESIDENTS=1).
    # This is the last moment a release can still influence the slot's allocation, and
    # it is already after the KV pool was sized.
    _release_mxfp4_host_cache(signature)
    _release_resident_marlin_copies(signature)

    slot_storage, slot_failure = _try_mxfp4_initialization(
        lambda: _allocate_mxfp4_slot_storage(context)
    )
    if not _all_tp_ranks_succeeded(slot_failure is None):
        # Free the caching allocator's dead blocks before giving up
        # (mirrors the local arch's gc + empty_cache retry).
        gc.collect()
        torch.cuda.empty_cache()
        slot_storage, slot_failure = _try_mxfp4_initialization(
            lambda: _allocate_mxfp4_slot_storage(context)
        )

    if not _all_tp_ranks_succeeded(slot_failure is None):
        slot_storage = None
        context = None
        fatal_error = _any_tp_rank_true(
            slot_failure is not None and slot_failure[0] == "fatal"
        )
        if fatal_error:
            message = "MXFP4 raw/prepared slot allocation failed on at least one TP rank"
            if slot_failure is not None:
                message = f"{message}: {slot_failure[1]}"
            raise RuntimeError(message)
        _fail_detail = slot_failure[1] if slot_failure is not None else None
        slot_failure = None
        _disable_mxfp4_layerwise_pipeline(
            signature,
            f"full-layer raw/prepared allocation failed on at least one TP rank"
            f"{(' (detail: ' + _fail_detail + ')') if _fail_detail else ''}",
        )
        return
    slot1_raw, slot_marlin_prepared = slot_storage

    manager, manager_failure = _try_mxfp4_initialization(
        lambda: _Mxfp4LayerwisePrefillManager(
            context, signature, slot1_raw, slot_marlin_prepared
        )
    )
    if not _all_tp_ranks_succeeded(manager_failure is None):
        manager = None
        slot_storage = None
        if slot1_raw is not None:
            slot1_raw.clear()
        slot_marlin_prepared = None
        context = None
        fatal_error = _any_tp_rank_true(
            manager_failure is not None and manager_failure[0] == "fatal"
        )
        if fatal_error:
            message = "MXFP4 prepared slot setup failed on at least one TP rank"
            if manager_failure is not None:
                message = f"{message}: {manager_failure[1]}"
            raise RuntimeError(message)
        manager_failure = None
        _disable_mxfp4_layerwise_pipeline(
            signature,
            "persistent MXFP4 prepared slot setup failed on at least one TP rank",
        )
        return

    context.initialize_cpu_buffers()
    _MXFP4_LAYERWISE_MANAGERS[signature] = manager
    if _SHARED_FULL_CONTEXT is None:
        _SHARED_FULL_CONTEXT = context
    if method.tp_rank == 0:
        raw_bytes = sum(
            getattr(slot, name).numel() * getattr(slot, name).element_size()
            for slot in manager.slots
            for name in _Mxfp4PrefillSlot.RAW_NAMES
        )
        prepared_bytes = sum(
            tensor.numel() * tensor.element_size()
            for slot in manager.slots
            for tensor in (
                *(
                    []
                    if isinstance(slot.marlin_prepared, tuple)
                    else [
                        slot.marlin_prepared.w13,
                        slot.marlin_prepared.w13_scale,
                        slot.marlin_prepared.w2,
                        slot.marlin_prepared.w2_scale,
                    ]
                ),
            )
        )
        logger.info(
            "KT MXFP4 layerwise prefill lazily initialized two raw + two Marlin "
            "prepared full-layer slots on %s (raw=%.2f GiB, "
            "prepared=%.2f GiB, total=%.2f GiB)",
            manager.device,
            raw_bytes / 1024**3,
            prepared_bytes / 1024**3,
            (raw_bytes + prepared_bytes) / 1024**3,
        )


def _get_or_initialize_mxfp4_layerwise_manager(
    method, layer: torch.nn.Module
) -> Optional[_Mxfp4LayerwisePrefillManager]:
    """Return the persistent manager, allocating its slots on first use.

    This function is called only from a threshold-qualified prefill forward.
    Each TP rank executes it before the manager enters its transport
    collectives, and `_initialize_mxfp4_layerwise_pipeline` keeps allocation
    failures consistent across ranks.  An OOM records a disabled reason and
    returns ``None`` so the triggering request can use hybrid CPU/GPU MoE.
    """

    signature = getattr(method, "_mxfp4_pipeline_signature", None)
    if signature is None:
        signature = _mxfp4_pipeline_signature(method, layer)
        method._mxfp4_pipeline_signature = signature

    manager = _MXFP4_LAYERWISE_MANAGERS.get(signature)
    if manager is not None or signature in _MXFP4_LAYERWISE_DISABLED_REASONS:
        return manager

    _initialize_mxfp4_layerwise_pipeline(method, layer)
    return _MXFP4_LAYERWISE_MANAGERS.get(signature)


def generate_front_loading_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks by filling layers from first MoE layer onwards.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers (e.g., 1 = every layer, 2 = every other layer)

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    remaining = num_gpu_experts

    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if not is_moe:
            # Dense layer - set all True (bypass KT wrapper)
            masks[layer_idx, :] = True
        elif remaining > 0:
            # MoE layer - allocate GPU experts
            num_for_this_layer = min(remaining, num_experts)
            masks[layer_idx, :num_for_this_layer] = True
            remaining -= num_for_this_layer

    return masks


def generate_uniform_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Generate masks with equal GPU experts per MoE layer.

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Identify MoE layers
    moe_layers = [
        i for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]
    num_moe_layers = len(moe_layers)

    if num_moe_layers == 0:
        return masks

    # Distribute GPU experts evenly
    experts_per_layer = num_gpu_experts // num_moe_layers
    remainder = num_gpu_experts % num_moe_layers

    for idx, layer_idx in enumerate(moe_layers):
        # First 'remainder' layers get one extra expert
        num_for_this_layer = experts_per_layer + (1 if idx < remainder else 0)
        num_for_this_layer = min(num_for_this_layer, num_experts)
        masks[layer_idx, :num_for_this_layer] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def generate_random_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate masks by randomly selecting GPU experts (fixed seed).

    Args:
        num_layers: Total number of layers in the model
        num_experts: Number of experts per layer
        num_gpu_experts: Total number of GPU experts to allocate
        first_k_dense_replace: Layer index where MoE layers start
        moe_layer_freq: Frequency of MoE layers
        seed: Random seed for reproducibility

    Returns:
        Boolean mask tensor of shape [num_layers, num_experts]
    """
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    # Collect all MoE (layer, expert) positions
    moe_positions = []
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if is_moe:
            for expert_idx in range(num_experts):
                moe_positions.append((layer_idx, expert_idx))

    # Randomly select positions
    if len(moe_positions) > 0:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        num_to_select = min(num_gpu_experts, len(moe_positions))
        selected_indices = torch.randperm(len(moe_positions), generator=rng, device='cpu')[:num_to_select]

        for idx in selected_indices:
            layer_idx, expert_idx = moe_positions[idx]
            masks[layer_idx, expert_idx] = True

    # Set non-MoE layers to all True
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
            masks[layer_idx, :] = True

    return masks


def _init_kt_gpu_experts_masks(server_args: "ServerArgs") -> Optional[torch.Tensor]:
    """Initialize GPU experts masks from activation frequency data.

    Args:
        server_args: Global server arguments

    Returns:
        Masks tensor of shape [num_layers, num_experts], or None if KT not configured
    """
    global _KT_GPU_EXPERTS_MASKS

    if _KT_GPU_EXPERTS_MASKS is not None:
        return _KT_GPU_EXPERTS_MASKS

    # Get model config (unwrap VL configs that nest the text model config)
    hf_config = server_args.get_hf_config()

    # fix for kimi-k2.5 models where text_config holds the actual config
    if getattr(hf_config, "text_config", None) is not None:
        hf_config = hf_config.text_config

    num_layers = getattr(hf_config, "num_hidden_layers", None)
    # Try different attribute names for num_experts
    num_experts = getattr(hf_config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(hf_config, "n_routed_experts", None)

    if num_layers is None or num_experts is None:
        logger.warning(
            "Could not determine num_layers or num_experts from model config."
        )
        return None

    # Get first_k_dense_replace to identify which layers are MoE layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0) or 0
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1)

    # NEW (2026-04-29): V4-Flash has hash-MoE layers at the front (num_hash_layers,
    # typically 3) which the HF DeepseekV3Config treats as first_k_dense_replace=3
    # by default. But hash layers DO have routed experts (n_routed_experts=256)
    # — they are NOT dense. Letting generate_uniform_masks set masks[0..2,:] = True
    # for hash layers makes the kt_ep_wrapper send all 256 experts to a GPU MoE
    # that only loaded num_gpu_experts_per_layer worth of weights, triggering
    # the fused_moe Hidden size mismatch assert. Subtract num_hash_layers so
    # hash layers are correctly classified as MoE for mask purposes.
    # Origin: kt-sglang 耦合 (V4-Flash hash-MoE handling in kt_ep_wrapper).
    num_hash_layers = getattr(hf_config, "num_hash_layers", 0) or 0
    if num_hash_layers > 0:
        first_k_dense_replace = max(0, first_k_dense_replace - num_hash_layers)

    # Normalize list-form moe_layer_freq (e.g., MiMo-V2-Flash: [0, 1, 1, ...])
    # to standard (first_k_dense_replace, moe_layer_freq=1) form
    if isinstance(moe_layer_freq, list):
        # Find first MoE layer index from the mask
        first_moe = next((i for i, v in enumerate(moe_layer_freq) if v), 0)
        first_k_dense_replace = max(first_k_dense_replace or 0, first_moe)
        moe_layer_freq = 1

    # Count actual MoE layers
    num_moe_layers = sum(
        1 for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    )
    total_experts = num_moe_layers * num_experts
    logger.debug(
        "[kt-mask] num_layers=%d num_experts=%d first_k_dense_replace=%s (type=%s) "
        "moe_layer_freq=%s (type=%s) computed_num_moe_layers=%d "
        "hf_config_class=%s.%s num_hash_layers=%s n_hash_layers=%s",
        num_layers, num_experts,
        first_k_dense_replace, type(first_k_dense_replace).__name__,
        moe_layer_freq, type(moe_layer_freq).__name__,
        num_moe_layers,
        type(hf_config).__module__, type(hf_config).__name__,
        getattr(hf_config, 'num_hash_layers', '<missing>'),
        getattr(hf_config, 'n_hash_layers', '<missing>'),
    )

    # Determine num_gpu_experts (total across all layers)
    if server_args.kt_gpu_experts_ratio is not None:
        # Use ratio to calculate total GPU experts
        num_gpu_experts = int(total_experts * server_args.kt_gpu_experts_ratio)
        if server_args.kt_num_gpu_experts is not None:
            logger.warning(
                f"--kt-gpu-experts-ratio={server_args.kt_gpu_experts_ratio} is set, "
                f"ignoring --kt-num-gpu-experts={server_args.kt_num_gpu_experts}. "
                f"Actual total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
        else:
            logger.info(
                f"Using kt_gpu_experts_ratio={server_args.kt_gpu_experts_ratio}, "
                f"total GPU experts: {num_gpu_experts} "
                f"(= {total_experts} total experts × {server_args.kt_gpu_experts_ratio})"
            )
    elif server_args.kt_num_gpu_experts is not None:
        # kt_num_gpu_experts is per-layer, multiply by num_moe_layers
        num_gpu_experts = server_args.kt_num_gpu_experts * num_moe_layers
        logger.info(
            f"Using kt_num_gpu_experts={server_args.kt_num_gpu_experts} per layer, "
            f"total GPU experts: {num_gpu_experts} "
            f"(= {server_args.kt_num_gpu_experts} × {num_moe_layers} MoE layers)"
        )
    else:
        logger.warning("Either kt_num_gpu_experts or kt_gpu_experts_ratio is required but not set.")
        return None

    # Get GPU expert placement strategy
    strategy = server_args.kt_expert_placement_strategy

    # Generate masks based on strategy
    tp_rank = get_tensor_model_parallel_rank()

    if strategy == "frequency":
        # `--kt-expert-frequency-file` is the KT-side input: the profile only
        # picks which experts stay resident on the GPU, it never remaps ids (see
        # that flag's help).  Reading the same .pt through
        # `--init-expert-location` also works but has side effects -- that path
        # builds an ExpertLocationMetadata from `logical_count`
        # (`compute_initial_expert_location_metadata` -> `init_by_eplb`) and
        # turns on the logical->physical remap in select_experts -- so it is only
        # a fallback here.
        init_loc = (
            server_args.kt_expert_frequency_file or server_args.init_expert_location
        )
        has_activation_freq = bool(init_loc) and init_loc.endswith(".pt")

        if has_activation_freq:
            logger.info("Loading activation frequency from %s", init_loc)
            loaded_data = torch.load(init_loc, map_location="cpu", weights_only=True)
            # Handle both dict format (from ExpertDistributionRecorder) and raw tensor
            if isinstance(loaded_data, dict):
                if "logical_count" in loaded_data:
                    activation_counts = loaded_data["logical_count"]
                else:
                    raise ValueError(
                        f"Loaded dict does not contain 'logical_count' key. "
                        f"Available keys: {list(loaded_data.keys())}"
                    )
            else:
                activation_counts = loaded_data
            # Expected shape: [buffer_size, num_layers, num_experts]
            if activation_counts.dim() != 3:
                raise ValueError(
                    f"Expected activation counts tensor with 3 dims [buffer_size, num_layers, num_experts], "
                    f"got {activation_counts.dim()} dims with shape {activation_counts.shape}"
                )
            _, file_num_layers, file_num_experts = activation_counts.shape
            if file_num_layers != num_layers:
                raise ValueError(
                    f"Activation counts num_layers ({file_num_layers}) doesn't match "
                    f"model num_layers ({num_layers})"
                )
            if file_num_experts != num_experts:
                raise ValueError(
                    f"Activation counts num_experts ({file_num_experts}) doesn't match "
                    f"model num_experts ({num_experts})"
                )
            # Sum across buffer_size (dim0) to get total activation counts per expert
            activation_freq = activation_counts.sum(dim=0).float()  # [num_layers, num_experts]
            logger.info("Using frequency-based strategy with activation frequency data")
        else:
            # No activation frequency file, use zeros (uniform distribution)
            logger.warning(
                "Using frequency-based strategy WITHOUT activation frequency data "
                "(uniform distribution fallback)"
            )
            activation_freq = torch.zeros(num_layers, num_experts, dtype=torch.float32)
            # For layers that are actually MoE layers, set uniform distribution
            for layer_idx in range(num_layers):
                if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0:
                    activation_freq[layer_idx, :] = 1.0

        # Generate masks on rank 0
        if tp_rank == 0:
            masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts)
            # For non-MoE layers, set all experts to GPU
            for layer_idx in range(num_layers):
                if layer_idx < first_k_dense_replace or layer_idx % moe_layer_freq != 0:
                    masks[layer_idx, :] = True
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "front-loading":
        if tp_rank == 0:
            logger.info("Using front-loading strategy for GPU expert placement")
            masks = generate_front_loading_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "uniform":
        if tp_rank == 0:
            logger.info("Using uniform strategy for GPU expert placement")
            masks = generate_uniform_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    elif strategy == "random":
        if tp_rank == 0:
            logger.info("Using random strategy for GPU expert placement (seed=42)")
            masks = generate_random_masks(
                num_layers, num_experts, num_gpu_experts,
                first_k_dense_replace, moe_layer_freq, seed=42
            )
        else:
            masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    else:
        raise ValueError(f"Unknown kt_expert_placement_strategy: {strategy}")

    if dist.is_initialized():
        dist.broadcast(masks, src=0, group=get_tp_group().cpu_group)

    _KT_GPU_EXPERTS_MASKS = masks

    # Log per-layer GPU expert counts (rank 0 only, MoE layers only)
    if tp_rank == 0:
        per_layer_gpu_experts = masks.sum(dim=1).cpu().tolist()
        for layer_idx, num_gpu in enumerate(per_layer_gpu_experts):
            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and layer_idx % moe_layer_freq == 0
            )
            # Only log for actual MoE layers
            if is_moe_layer:
                logger.info(
                    "KT GPU experts: layer %d (MoE) has %d GPU experts",
                    layer_idx,
                    int(num_gpu),
                )

        # Count total GPU experts only for actual MoE layers
        total_moe_gpu_experts = sum(
            masks[i].sum().item()
            for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        num_moe_layers = sum(
            1 for i in range(num_layers)
            if i >= first_k_dense_replace and i % moe_layer_freq == 0
        )
        logger.info(
            "Generated KT GPU experts masks using '%s' strategy: %d MoE layers (out of %d total layers) x %d experts, "
            "total GPU experts in MoE layers = %d",
            strategy, num_moe_layers, num_layers, num_experts, total_moe_gpu_experts
        )

    return _KT_GPU_EXPERTS_MASKS


def _select_kt_gpu_experts_mask(
    masks: torch.Tensor, layer_idx: int, num_experts: Optional[int]
) -> torch.Tensor:
    """Pick the placement row describing one MoE layer.

    ``masks`` is built from the served model's config, so it only describes that
    model's layers. A draft/MTP stage is a separate, smaller MoE (V4.1: 128
    routed experts against the target's 384) whose stage ids collide with target
    layer ids, so the target's rows do not describe it. Such a layer keeps every
    expert on CPU: the GPU budget is ``kt_num_gpu_experts * num_moe_layers`` over
    the target's layers, and handing a layer GPU experts the budget never
    counted would over-subscribe device memory.
    """
    if num_experts is None or masks.shape[1] == num_experts:
        return masks[layer_idx]
    logger.info(
        "[kt-mask] layer %d has %d experts but the placement table describes %d; "
        "keeping all %d of them on CPU.",
        layer_idx,
        num_experts,
        masks.shape[1],
        num_experts,
    )
    return torch.zeros(num_experts, dtype=torch.bool, device="cpu")


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int, num_experts: Optional[int] = None
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model
        num_experts: Routed experts of the layer being built. Layers whose
            expert count differs from the placement table (draft/MTP stages) get
            an all-CPU mask; see :func:`_select_kt_gpu_experts_mask`.

    Returns:
        KTConfig if KT is configured and not disabled, None otherwise
    """
    # Check if KT EP wrapper is disabled (e.g., for draft models in speculative decoding)
    from sglang.srt.layers.moe.utils import is_kt_ep_wrapper_disabled

    if is_kt_ep_wrapper_disabled():
        return None

    if server_args.kt_weight_path is None:
        return None

    # Get GPU experts masks (initializes if needed)
    masks = _init_kt_gpu_experts_masks(server_args)
    if masks is None:
        return None

    # Get num_layers from model config (unwrap VL configs)
    hf_config = server_args.get_hf_config()
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    num_layers = getattr(hf_config, "num_hidden_layers", None)

    # NOTE: hash-layer skip experiment was tried here (return None when
    # layer_idx < num_hash_layers); it didn't help because the underlying
    # fused_moe shape-mismatch in V4 hash MoE happens with or without KT wrap.
    # Reverted; root cause is in V4 MoE weight layout vs sglang fused_moe.

    # Get mask for this specific layer
    gpu_experts_mask = _select_kt_gpu_experts_mask(masks, layer_idx, num_experts)

    return KTConfig(
        layer_idx=layer_idx,
        gpu_experts_mask=gpu_experts_mask,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        numa_nodes=server_args.kt_numa_nodes,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        gpu_prefill_token_threshold=server_args.kt_gpu_prefill_token_threshold,
        kt_enable_dynamic_expert_update=server_args.kt_enable_dynamic_expert_update,
        expert_lora_path=getattr(server_args, "kt_expert_lora_path", None),
    )


_REMAP_TABLE_CACHE: dict = {}


def _remap_table(
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    """``where(mask, gpu_index, -1)`` as a ``[num_experts]`` lookup table, cached.

    Both inputs are per-layer placement constants, so this only has to be rebuilt when
    one of them is replaced (``data_ptr``) or mutated (``_version``).  The caller's
    lookup then costs a single gather instead of two gathers plus a ``where``, which
    matters because it runs once per MoE layer per decode step on a
    ``[num_tokens, top_k]`` tensor that is tiny -- ``[7, 6]`` at decode, where the
    dispatch of each op costs far more than its work.
    """
    key = (
        gpu_experts_mask.data_ptr(),
        gpu_experts_mask._version,
        logical_to_gpu_index.data_ptr(),
        logical_to_gpu_index._version,
    )
    table = _REMAP_TABLE_CACHE.get(key)
    if table is None:
        table = torch.where(gpu_experts_mask, logical_to_gpu_index, -1).contiguous()
        if len(_REMAP_TABLE_CACHE) > 64:
            _REMAP_TABLE_CACHE.clear()
        _REMAP_TABLE_CACHE[key] = table
    return table


def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    """Mask CPU expert IDs and remap GPU expert IDs to weight indices.

    CPU expert IDs (``gpu_experts_mask=False``) become -1 so the GPU kernel skips them;
    GPU expert IDs become GPU weight indices (0..num_gpu_experts-1).

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        gpu_experts_mask: Boolean tensor of shape [num_experts] where True indicates GPU expert
        logical_to_gpu_index: Int tensor of shape [num_experts] mapping logical ID to GPU index

    Returns:
        Remapped topk_ids tensor with GPU indices for GPU experts, -1 for CPU experts
    """
    return _remap_table(gpu_experts_mask, logical_to_gpu_index)[topk_ids]


def select_top_experts_from_batch(
    topk_ids: torch.Tensor,
    num_experts: int,
    num_gpu_experts: int,
) -> torch.Tensor:
    """Select top N most frequently activated experts from batch routing results.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing logical expert IDs
        num_experts: Total number of experts in the layer
        num_gpu_experts: Number of experts to select for GPU

    Returns:
        Tensor of shape [num_gpu_experts] containing selected expert IDs (sorted)

    Edge cases:
        - If batch has fewer unique experts than num_gpu_experts, fills remaining
          slots with least-activated experts (maintaining determinism)
        - Handles ties by preferring lower expert IDs (deterministic)
    """
    # Count activation frequency for each expert in this batch
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)

    # Flatten topk_ids and count occurrences
    flat_ids = topk_ids.flatten()
    # Filter out invalid IDs (< 0 or >= num_experts)
    valid_mask = (flat_ids >= 0) & (flat_ids < num_experts)
    valid_ids = flat_ids[valid_mask]

    if valid_ids.numel() > 0:
        expert_counts.index_add_(0, valid_ids, torch.ones_like(valid_ids, dtype=torch.int64))

    # Select top num_gpu_experts by frequency
    # For ties, torch.topk with sorted=True will prefer earlier indices (deterministic)
    _, selected_indices = torch.topk(
        expert_counts,
        k=min(num_gpu_experts, num_experts),
        largest=True,
        sorted=True  # Ensures deterministic tie-breaking
    )

    # Sort selected indices for easier debugging and consistent ordering
    selected_experts = selected_indices.sort()[0]

    return selected_experts


def copy_experts_weights_int4(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy INT4 Marlin expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight_packed: Packed INT4 weights for gate+up projection
        - w13_weight_scale: FP16 scales for w13
        - w2_weight_packed: Packed INT4 weights for down projection
        - w2_weight_scale: FP16 scales for w2
    """
    weight_names = ["w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            # In src_layer, expert at logical_id is at index logical_id
            # In dst_layer, we write to gpu_index dst_idx
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 block quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale_inv: FP32 inverse scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale_inv: FP32 inverse scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_fp8_channel(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy FP8 per-channel quant expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: FP8 weights for gate+up projection
        - w13_weight_scale: FP32 per-channel scales for w13
        - w2_weight: FP8 weights for down projection
        - w2_weight_scale: FP32 per-channel scales for w2
    """
    weight_names = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def copy_experts_weights_bf16(
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    selected_experts: torch.Tensor,
) -> None:
    """Copy BF16/unquantized expert weights from source to destination layer.

    Args:
        src_layer: Source layer (temporary full GPU layer) with all experts
        dst_layer: Destination layer (original layer) with subset of experts
        selected_experts: Tensor of logical expert IDs to copy (shape: [num_gpu_experts])

    This copies:
        - w13_weight: BF16 weights for gate+up projection
        - w2_weight: BF16 weights for down projection
    """
    weight_names = ["w13_weight", "w2_weight"]

    # Build mapping: selected logical ID -> dst GPU index
    logical_to_dst_index = {
        int(selected_experts[i].item()): i
        for i in range(len(selected_experts))
    }

    for weight_name in weight_names:
        src_weight = getattr(src_layer, weight_name)  # [global_num_experts, ...]
        dst_weight = getattr(dst_layer, weight_name)  # [num_gpu_experts, ...]

        # Copy each selected expert
        for logical_id, dst_idx in logical_to_dst_index.items():
            dst_weight[dst_idx].copy_(src_weight[logical_id], non_blocking=False)


def update_gpu_expert_mappings(
    selected_experts: torch.Tensor,
    num_experts: int,
    device: torch.device,
):
    """Update GPU expert mapping tables based on newly selected experts.

    Args:
        selected_experts: Tensor of logical expert IDs now on GPU (shape: [num_gpu_experts])
        num_experts: Total number of experts in layer
        device: Target CUDA device for mapping tensors

    Returns:
        Tuple of (gpu_experts_mask, logical_to_gpu_index, gpu_index_to_logical):
            - gpu_experts_mask: CPU bool tensor [num_experts], True = on GPU
            - logical_to_gpu_index: CUDA int32 tensor [num_experts], maps logical -> GPU index
            - gpu_index_to_logical: CPU int32 tensor [num_gpu_experts], reverse mapping
    """
    num_gpu_experts = len(selected_experts)

    # Create new mask (CPU tensor)
    gpu_experts_mask_cpu = torch.zeros(num_experts, dtype=torch.bool, device='cpu')
    gpu_experts_mask_cpu[selected_experts.cpu()] = True

    # Create logical_to_gpu_index (CUDA tensor)
    logical_to_gpu_index = torch.full(
        (num_experts,), -1, dtype=torch.int32, device=device
    )
    for gpu_idx, logical_id in enumerate(selected_experts):
        logical_to_gpu_index[logical_id] = gpu_idx

    # Create gpu_index_to_logical (CPU tensor for weight loading)
    gpu_index_to_logical_cpu = selected_experts.cpu().to(torch.int32)

    return gpu_experts_mask_cpu, logical_to_gpu_index, gpu_index_to_logical_cpu


def update_kt_wrapper_masks(
    wrapper: Optional["KTMoEWrapper"],
    gpu_experts_mask_cpu: torch.Tensor,
) -> None:
    """Update KT wrapper's internal GPU experts mask (rank 0 only).

    Args:
        wrapper: KTMoEWrapper instance (None if not rank 0)
        gpu_experts_mask_cpu: New GPU experts mask to apply

    The wrapper needs updated masks to correctly route tokens to CPU vs GPU experts.
    This is called on rank 0 only since only rank 0 has the wrapper instance.

    CRITICAL: wrapper.gpu_experts_mask is a pinned memory tensor whose pointer is shared
    with C++ code. We MUST use .copy_() to update in-place, not replace the reference.
    """
    if wrapper is None:
        return

    # Update wrapper's internal mask IN-PLACE
    # CRITICAL: The C++ code holds a pointer to this tensor's memory.
    # Replacing the reference would leave C++ pointing to old/freed memory.
    wrapper.gpu_experts_mask.copy_(gpu_experts_mask_cpu)


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (identified by gpu_experts_mask=True) using any quantization method
    - CPU experts (identified by gpu_experts_mask=False) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoEMethod(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, gpu_experts_mask=mask, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    # Tag for quant_method_registry.is_wrapped_method() — set as a class
    # attribute so isinstance-style checks in deepseek_v2 / glm4_moe work
    # without importing this module.
    _quant_wrapper_id = "kt_ep"

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.gpu_experts_mask = kt_config.gpu_experts_mask  # bool tensor [num_experts], on CPU
        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())
        self.kt_expert_lora_path = kt_config.expert_lora_path
        self.kt_expert_lora_enabled = bool(self.kt_expert_lora_path)
        self.kt_expert_lora_weights: Optional[KTExpertLoraWeights] = None
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.kt_expert_lora_enabled:
            if self.num_gpu_experts != 0:
                raise ValueError(
                    "--kt-expert-lora-path first supports CPU experts only. "
                    "Set --kt-num-gpu-experts 0 and do not enable "
                    "--kt-gpu-experts-ratio."
                )
            if kt_config.gpu_prefill_token_threshold:
                raise ValueError(
                    "--kt-expert-lora-path is not compatible with "
                    "--kt-gpu-prefill-token-threshold in the first single-adapter "
                    "implementation."
                )
            if kt_config.kt_enable_dynamic_expert_update:
                raise ValueError(
                    "--kt-expert-lora-path is not compatible with "
                    "--kt-enable-dynamic-expert-update in the first single-adapter "
                    "implementation."
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[kt-wrap-init] tp_rank=%d layer_idx=%s num_gpu_experts=%d "
                "mask_sum=%d mask_shape=%s gpu_method=%s",
                self.tp_rank,
                kt_config.layer_idx,
                self.num_gpu_experts,
                int(self.gpu_experts_mask.sum().item()),
                tuple(self.gpu_experts_mask.shape),
                type(gpu_method).__name__,
            )

        # Mapping tables for non-contiguous GPU expert allocation (CPU tensors)
        # Used by weight_loader to remap expert_id when loading weights
        gpu_expert_indices = torch.where(self.gpu_experts_mask)[0]
        self.logical_to_gpu_index = torch.full(
            (len(self.gpu_experts_mask),), -1, dtype=torch.int32
        )
        self.logical_to_gpu_index[gpu_expert_indices] = torch.arange(
            len(gpu_expert_indices), dtype=torch.int32
        )
        self.gpu_index_to_logical = gpu_expert_indices.to(torch.int32)

        # CUDA tensors for inference (will be set in create_weights)
        self.gpu_experts_mask_cuda = None
        self.logical_to_gpu_index_cuda = None

        self.gpu_prefill_token_threshold = kt_config.gpu_prefill_token_threshold or 0
        self._full_init_args = None
        self.wrapper: Optional[KTMoEWrapper] = None
        # Logical expert id -> KT physical slot (built in
        # process_weights_after_loading from the EPLB map). Pinned so it can
        # be copied to GPU inside a CUDA graph capture.
        self._logical_to_physical_cpu: Optional[torch.Tensor] = None
        # Pre-built GPU copy of the logical->physical map for use inside CUDA
        # graph capture (where host-to-device copies are disallowed).
        self._logical_to_physical_cuda: Optional[torch.Tensor] = None

        # Persistent scalar tensor on GPU used as the "no CPU expert" sentinel
        # in torch.where during CUDA graph capture (host->device copies of
        # scalars are disallowed while capturing).
        self._cpu_sentinel_cuda: Optional[torch.Tensor] = None

        # Dual-stream parallelism: cpu_stream for CPU expert operations,
        # main stream for GPU computation (initialized in create_weights)
        self._cpu_stream: Optional[torch.cuda.Stream] = None
        self._sync_done_event: Optional[torch.cuda.Event] = None  # CPU computation done

        # Persistent buffers for tensors spanning a KT breakable-graph seam.
        # Local addition (DSPARK breakable-cuda-graph support).
        self.graph_state_bridge = KTGraphStateBridge()

        # Shared staging buffer reference (initialized in create_weights, shared across all layers)
        self._shared_staging_buffer: Optional[SharedStagingBuffer] = None
        self._staging_buffer_max_size: int = kt_config.chunked_prefill_size or 8192

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self._full_init_args = (
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
        )

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU weights are indexed by gpu_index (0 to num_gpu_experts-1), not logical expert ID
        # The mapping logical_to_gpu_index is used to remap IDs during weight loading and inference
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # Move mask and mapping tables to GPU for inference
        target_device = _layer_device(layer)
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device=target_device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(device=target_device)

        # Initialize dual-stream for CPU-GPU parallelism (rank 0 only)
        if self.tp_rank == 0:
            self._cpu_stream = torch.cuda.Stream(device=target_device)
            self._sync_done_event = torch.cuda.Event()

            # Get or create shared staging buffer (shared across all MoE layers to save GPU memory)
            self._shared_staging_buffer = get_or_create_shared_staging_buffer(
                max_tokens=self._staging_buffer_max_size,
                hidden_size=hidden_size,
                dtype=params_dtype,
                device=target_device,
            )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts are identified by gpu_experts_mask=False
        if self.tp_rank == 0:
            # SwiGLU activation params for CPU experts. Source of truth is
            # MoeRunnerConfig, populated by the model file from HF config:
            #   - minimax_m3.py forwards config.swiglu_alpha / swiglu_limit
            #     as gemm1_alpha / gemm1_clamp_limit (swiglu_oai path)
            #   - deepseek_v2.py forwards config.swiglu_limit into the
            #     legacy swiglu_limit slot (DSV4 plain-silu clamp path)
            # kt-kernel C++ accepts a single (alpha, limit) pair and
            # disambiguates by alpha != 0 (swiglu_oai vs plain silu).
            _mrc = getattr(layer, "moe_runner_config", None)
            _cfg_alpha = getattr(_mrc, "gemm1_alpha", None) if _mrc is not None else None
            _cfg_clamp = getattr(_mrc, "gemm1_clamp_limit", None) if _mrc is not None else None
            _cfg_swglim = getattr(_mrc, "swiglu_limit", None) if _mrc is not None else None
            _kt_swiglu_alpha = float(_cfg_alpha) if _cfg_alpha is not None else 0.0
            _kt_swiglu_limit = float(
                _cfg_clamp if _cfg_clamp is not None else (_cfg_swglim or 0.0)
            )
            # kt-kernel guards swiglu_limit to MXFP4/MXFP8 only.
            # Zero it out for other methods (AMXINT4, BF16, etc.)
            # so V4-Flash + non-MXFP runs don't crash at init.
            if (self.kt_config.method or "").upper() not in ("MXFP4", "MXFP8"):
                _kt_swiglu_limit = 0.0
                _kt_swiglu_alpha = 0.0
            common_wrapper_kwargs = dict(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=self.gpu_experts_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                numa_nodes=self.kt_config.numa_nodes,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
            )
            if self.kt_expert_lora_enabled:
                if _kt_swiglu_limit != 0.0:
                    raise ValueError(
                        "--kt-expert-lora-path uses KT SFT wrappers, which do not "
                        "support the V4-2604B swiglu_limit path."
                    )
                self.kt_expert_lora_weights = _load_kt_expert_lora_weights(
                    adapter_path=self.kt_expert_lora_path,
                    layer_idx=self.kt_config.layer_idx,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    moe_intermediate_size=intermediate_size_full,
                )
                self.wrapper = KTMoEWrapper(
                    **common_wrapper_kwargs,
                    method=_map_kt_method_to_sft_method(self.kt_config.method),
                    mode="sft",
                    num_gpu_experts=0,
                    lora_rank=self.kt_expert_lora_weights.rank,
                    lora_alpha=self.kt_expert_lora_weights.alpha,
                    max_cache_depth=1,
                )
            else:
                self.wrapper = KTMoEWrapper(
                    **common_wrapper_kwargs,
                    swiglu_limit=_kt_swiglu_limit,
                    swiglu_alpha=_kt_swiglu_alpha,
                    method=self.kt_config.method,
                    max_deferred_experts_per_token=layer_max_deferred,
                )

        # Registration happens during model construction, not on the first
        # request, so layer N can identify and prepare N+1 immediately.
        _register_mxfp4_prefill_layer(self, layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            if (
                metadata is not None
                and getattr(metadata, "physical_to_logical_map_cpu", None) is not None
            ):
                physical_to_logical_map_cpu = (
                    metadata.physical_to_logical_map_cpu[self.kt_config.layer_idx]
                    .contiguous()
                )
                # The EPLB physical->logical map includes shared-expert slots
                # (DSV4 n_shared_experts=1 -> length num_experts+1), but the
                # KT wrapper's expert space is sized to num_experts (routed
                # only). Slice off the trailing shared slot so KT loads the
                # routed experts in physical order matching the topk_ids
                # space submitted to submit_forward.
                if physical_to_logical_map_cpu.numel() > layer.num_experts:
                    physical_to_logical_map_cpu = physical_to_logical_map_cpu[
                        : layer.num_experts
                    ].contiguous()
            else:
                # Fallback for setups without EPLB metadata: identity mapping.
                physical_to_logical_map_cpu = torch.arange(
                    layer.num_experts, dtype=torch.int64, device="cpu"
                )
            # KT loads weights in physical order: KT slot p holds logical
            # expert physical_to_logical_map_cpu[p]. submit_forward consumes
            # topk_ids in that SAME physical space, so the logical routing
            # ids from the dispatcher must be remapped logical -> physical
            # before CPU submission (GPU experts get -1 and are skipped).
            # Build the inverse map: logical expert l -> KT physical slot.
            logical_to_physical = torch.empty(
                layer.num_experts, dtype=torch.int64, device="cpu"
            )
            logical_to_physical[physical_to_logical_map_cpu] = torch.arange(
                layer.num_experts, dtype=torch.int64
            )
            # Pin so the tensor can be copied to GPU inside CUDA graph capture.
            self._logical_to_physical_cpu = logical_to_physical.pin_memory()
            # Pre-build a GPU copy for use during CUDA graph capture, where
            # host-to-device copies are disallowed. The CPU path below will
            # prefer this when available.
            self._logical_to_physical_cuda = logical_to_physical.to(
                device="cuda", non_blocking=True
            )
            # Persistent -1 sentinel on GPU for torch.where inside CUDA graph
            # capture (host->device scalar copies are disallowed while capturing).
            self._cpu_sentinel_cuda = torch.tensor(
                -1, dtype=torch.int64, device="cuda"
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)
            if self.kt_expert_lora_enabled:
                if self.kt_expert_lora_weights is None:
                    raise RuntimeError(
                        "KT expert LoRA is enabled but adapter weights were not loaded."
                    )
                lora = self.kt_expert_lora_weights
                if os.environ.get("SGLANG_KT_EXPERT_LORA_DEBUG") == "1":
                    print(
                        "[KT expert LoRA debug] "
                        f"layer={self.kt_config.layer_idx} "
                        f"rank={lora.rank} alpha={lora.alpha} "
                        f"gate_a={tuple(lora.gate_lora_a.shape)}/{lora.gate_lora_a.dtype}/"
                        f"{lora.gate_lora_a.device}/ptr={lora.gate_lora_a.data_ptr()} "
                        f"gate_b={tuple(lora.gate_lora_b.shape)}/{lora.gate_lora_b.dtype}/"
                        f"{lora.gate_lora_b.device}/ptr={lora.gate_lora_b.data_ptr()} "
                        f"up_a={tuple(lora.up_lora_a.shape)}/{lora.up_lora_a.dtype}/"
                        f"{lora.up_lora_a.device}/ptr={lora.up_lora_a.data_ptr()} "
                        f"up_b={tuple(lora.up_lora_b.shape)}/{lora.up_lora_b.dtype}/"
                        f"{lora.up_lora_b.device}/ptr={lora.up_lora_b.data_ptr()} "
                        f"down_a={tuple(lora.down_lora_a.shape)}/{lora.down_lora_a.dtype}/"
                        f"{lora.down_lora_a.device}/ptr={lora.down_lora_a.data_ptr()} "
                        f"down_b={tuple(lora.down_lora_b.shape)}/{lora.down_lora_b.dtype}/"
                        f"{lora.down_lora_b.device}/ptr={lora.down_lora_b.data_ptr()}",
                        flush=True,
                    )
                self.wrapper.init_lora_weights(
                    lora.gate_lora_a,
                    lora.gate_lora_b,
                    lora.up_lora_a,
                    lora.up_lora_b,
                    lora.down_lora_a,
                    lora.down_lora_b,
                    torch.zeros_like(lora.gate_lora_a),
                    torch.zeros_like(lora.gate_lora_b),
                    torch.zeros_like(lora.up_lora_a),
                    torch.zeros_like(lora.up_lora_b),
                    torch.zeros_like(lora.down_lora_a),
                    torch.zeros_like(lora.down_lora_b),
                )
                logger.info(
                    "Loaded KT expert LoRA for layer %d from %s (rank=%d, alpha=%.3f)",
                    self.kt_config.layer_idx,
                    self.kt_expert_lora_path,
                    lora.rank,
                    lora.alpha,
                )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config

        # Create a separate config for GPU method without routed_scaling_factor.
        # This is because:
        # 1. GPU method's moe_sum_reduce would apply routed_scaling_factor internally
        # 2. KT CPU kernel does NOT apply routed_scaling_factor
        # 3. The combined output (GPU + CPU) would have inconsistent scaling
        # 4. routed_scaling_factor is applied uniformly in deepseek_v2.py forward_normal
        # So we disable it in GPU method to avoid double scaling on GPU part.
        gpu_runner_config = replace(moe_runner_config, routed_scaling_factor=None)
        if self.override_num_local_experts:
            gpu_runner_config = replace(
                gpu_runner_config, num_local_experts=self.num_gpu_experts
            )

        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, gpu_runner_config)

    def _submit_cpu_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        if self.kt_expert_lora_enabled:
            self.wrapper.submit_forward_inference(
                hidden_states,
                topk_ids,
                topk_weights,
                torch.cuda.current_stream(hidden_states.device).cuda_stream,
            )
        else:
            self.wrapper.submit_forward(
                hidden_states,
                topk_ids,
                topk_weights,
                torch.cuda.current_stream(hidden_states.device).cuda_stream,
            )

    def _sync_cpu_forward(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        if self.kt_expert_lora_enabled:
            return self.wrapper.sync_forward_inference(
                torch.cuda.current_stream(ref_tensor.device).cuda_stream,
            )
        return self.wrapper.sync_forward(
            ref_tensor,
            torch.cuda.current_stream(ref_tensor.device).cuda_stream,
        )

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self._submit_cpu_forward(x, topk_ids, topk_weights)

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self._sync_cpu_forward(x)

    def _submit_with_staged_input(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
        staged_hidden_states: torch.Tensor,
    ) -> None:
        """Submit CPU expert computation using staged hidden states.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
            staged_hidden_states: Pre-copied hidden states in staging buffer
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # CPU submission must only contain CPU-resident experts, in KT's
        # physical (load_weights) index space. Mask GPU experts to -1 and
        # remap CPU experts logical -> physical. On the all-CPU (0-expert)
        # path the map is identity so this is a no-op beyond the mask.
        if self.num_gpu_experts > 0 or getattr(self, "_logical_to_physical_cpu", None) is not None:
            # Prefer the pre-built GPU copy during CUDA graph capture (host->device
            # copies are disallowed while capturing). Fall back to the pinned CPU
            # copy for normal execution.
            _ltp_cuda = getattr(self, "_logical_to_physical_cuda", None)
            if _ltp_cuda is not None and _ltp_cuda.device == topk_ids.device:
                _ltp = _ltp_cuda
            else:
                _ltp = getattr(self, "_logical_to_physical_cpu", None)
                if _ltp is None:
                    _ltp = torch.arange(
                        self.gpu_experts_mask.numel(),
                        dtype=torch.int64,
                        device="cpu",
                    )
                _ltp = _ltp.to(device=topk_ids.device)
            topk_ids_long = topk_ids.to(torch.int64)
            safe_ids = topk_ids_long.clamp(
                0, self.gpu_experts_mask.numel() - 1
            )
            on_gpu = self.gpu_experts_mask_cuda[safe_ids]
            _sentinel = getattr(self, "_cpu_sentinel_cuda", None)
            if _sentinel is None or _sentinel.device != topk_ids.device:
                _sentinel = torch.tensor(-1, dtype=torch.int64, device=topk_ids.device)
            cpu_ids = torch.where(
                (topk_ids_long >= 0) & ~on_gpu,
                _ltp[safe_ids],
                _sentinel,
            )
            topk_ids = cpu_ids

        # Submit forward task using staged buffer
        self._submit_cpu_forward(staged_hidden_states, topk_ids, topk_weights)

    def _sync_with_staged_input(
        self, staged_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Synchronize CPU computation using staged hidden states reference.

        Args:
            staged_hidden_states: Staged buffer used in submit

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(staged_hidden_states)

        return self._sync_cpu_forward(staged_hidden_states)

    def _apply_impl(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # Record GPU expert mask for distribution tracking (rank 0 only)
        # Use gpu_experts_mask_cuda which is already on GPU for CUDA graph compatibility
        if self.tp_rank == 0:
            recorder = get_global_expert_distribution_recorder()
            recorder.on_gpu_expert_mask(
                self.kt_config.layer_idx, self.gpu_experts_mask_cuda
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        num_tokens = int(x.shape[0]) if x.dim() > 0 else 0
        _kt_timing = (
            os.environ.get("SGLANG_KT_HYBRID_TIMING") == "1"
            and self.tp_rank == 0
            and getattr(self.kt_config, "layer_idx", None) in (0, 5, 20, 35)
        )
        _kt_t_apply_start = time.perf_counter() if _kt_timing else None
        _kt_t_after_submit = None
        _kt_t_after_mask = None
        _kt_t_after_gpu = None
        _kt_t_after_sync = None
        _kt_t_after_merge = None
        _kt_t_cpu_wait_ms = 0.0
        _kt_t_cxx = None  # end of the C++ submit inside _submit_with_staged_input
        _kt_t_pre_cxx = None  # start of it, so the submit splits into python vs C++

        # Check for full GPU fallback. The full-GPU path's _build_full_context →
        # _prepare_weight_{mxfp4,fp8,fp8_channel,bf16,int4} helpers read flat
        # `w13_weight` / `w13_weight_packed` attributes off `layer`. V4-Flash
        # MXFP4 (triton_kernels path) optionally preserves these when
        # `kt_gpu_prefill_token_threshold > 0` is set (see
        # `mxfp4_deepseek.process_weights_after_loading`); accept either the
        # flat attr or the v4 triton-kernels marker as a hint that the loader
        # can populate the layer. Layouts without either are still skipped to
        # avoid crashing the scheduler. Origin: sglang 本身 (V4-Flash
        # full-GPU prefill fallback compat).
        # 0-expert decode never takes the full-GPU prefill fallback or the
        # MXFP4 layerwise path. Short-circuit so the per-layer decode hot path
        # skips the fallback/mxfp4 plumbing entirely (module lookups, hasattr
        # checks, dict probes). NOTE: only short-circuit when below the prefill
        # threshold — a large 0-expert prefill (num_tokens >= threshold) still
        # takes the full-GPU layerwise path, which is much faster than CPU MoE.
        _zero_gpu = (
            self.num_gpu_experts == 0
            and num_tokens < self.gpu_prefill_token_threshold
        )
        _full_gpu_gate = False
        _mxfp4_requested = False
        _mxfp4_manager = None
        _mxfp4_disabled_after_oom = False
        if not _zero_gpu:
            _full_gpu_fallback_supported = (
                hasattr(layer, "w13_weight")
                or hasattr(layer, "w13_weight_packed")
                or getattr(layer, "_v4_tk_path", False)
            )
            _full_gpu_gate = (
                self.gpu_prefill_token_threshold > 0
                and num_tokens >= self.gpu_prefill_token_threshold
                and _full_gpu_fallback_supported
            )
            _mxfp4_requested = _mxfp4_pipeline_requested(self)
            _mxfp4_signature = getattr(self, "_mxfp4_pipeline_signature", None)
            _mxfp4_manager = (
                _MXFP4_LAYERWISE_MANAGERS.get(_mxfp4_signature)
                if _mxfp4_signature is not None
                else None
            )
            _mxfp4_disabled_after_oom = (
                _mxfp4_signature in _MXFP4_LAYERWISE_DISABLED_REASONS
                if _mxfp4_signature is not None
                else False
            )
            if _mxfp4_disabled_after_oom and _full_gpu_gate:
                # Auto-recover: the OOM teardown freed the layerwise slots, so
                # re-enable the pipeline once the GPU has room for the footprint
                # again.  This keeps a single long-context OOM from disabling
                # GPU layerwise prefill for short requests for the whole
                # session.
                try:
                    _free_bytes = torch.cuda.mem_get_info()[0]
                    if _free_bytes >= _MXFP4_LAYERWISE_REENABLE_BYTES:
                        _MXFP4_LAYERWISE_DISABLED_REASONS.pop(
                            _mxfp4_signature, None
                        )
                        _mxfp4_disabled_after_oom = False
                        logger.warning(
                            "MXFP4 layerwise prefill re-enabled (%.1f GiB "
                            "free).",
                            _free_bytes / 1024**3,
                        )
                except Exception:
                    pass

        if _mxfp4_manager is not None and not _full_gpu_gate:
            _mxfp4_manager.abort_round()

        # Allocate the persistent full-layer slots only when a request actually
        # enters the MXFP4 layerwise path.  Startup reserves only their
        # KV-cache budget, not the tensors themselves; an allocation OOM still
        # records a disabled reason and this same request falls through to the
        # existing hybrid CPU/GPU path below.
        if (
            _full_gpu_gate
            and _mxfp4_requested
            and not _mxfp4_disabled_after_oom
            and _mxfp4_manager is None
            and _mxfp4_pipeline_runtime_supported(self, layer)
        ):
            _mxfp4_manager = _get_or_initialize_mxfp4_layerwise_manager(
                self, layer
            )
            _mxfp4_signature = getattr(self, "_mxfp4_pipeline_signature", None)
            _mxfp4_disabled_after_oom = (
                _mxfp4_signature in _MXFP4_LAYERWISE_DISABLED_REASONS
                if _mxfp4_signature is not None
                else False
            )

        if _full_gpu_gate and not (
            _mxfp4_requested and _mxfp4_disabled_after_oom
        ):
            _oom_fallback = False
            if _mxfp4_pipeline_runtime_supported(self, layer):
                if _mxfp4_manager is None:
                    raise RuntimeError(
                        "MXFP4 layerwise prefill is supported but was not "
                        "initialized during lazy slot allocation"
                    )
                try:
                    return _mxfp4_manager.apply(self, layer, dispatch_output)
                except Exception as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    # A long-context layerwise prefill on a VRAM-tight config
                    # (e.g. the production 1M KV pool + MoE slots + chunk
                    # working memory) can OOM mid-prefill.  Teardown the
                    # layerwise slots (~4 GiB) and fall through to the hybrid
                    # CPU MoE path below so the request completes instead of
                    # failing with HTTP 500 after the scheduler's retries.
                    logger.warning(
                        "[kt-ep] layerwise prefill CUDA OOM; freed layerwise "
                        "slots and fell back to hybrid CPU/GPU MoE"
                    )
                    _mxfp4_manager.teardown()
                    if _mxfp4_signature is not None:
                        _MXFP4_LAYERWISE_MANAGERS.pop(
                            _mxfp4_signature, None
                        )
                        # Disable the layerwise pipeline so the SAME long
                        # request's remaining chunks (and later requests) take
                        # the hybrid CPU MoE path directly instead of re-initing
                        # the ~4 GiB layerwise slots and OOMing per chunk.
                        _disable_mxfp4_layerwise_pipeline(
                            _mxfp4_signature,
                            "CUDA OOM during layerwise prefill; fell back "
                            "to hybrid CPU/GPU MoE",
                        )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _oom_fallback = True

            if not _oom_fallback:
                if _mxfp4_manager is not None:
                    raise RuntimeError(
                        "MXFP4 Marlin layerwise prefill was initialized, but "
                        "the runtime layer has no compatible raw MXFP4 weights"
                    )

                # Non-MXFP4 and unsupported MXFP4 backends retain the
                # existing serialized full-GPU fallback.
                ctx = self._build_full_context(layer)

                # Re-run quant post-processing on the full-expert gpu_layer
                # if supported (e.g. Marlin repack for Fp8MarlinMoEMethod).
                # The ctx.load() call writes raw fp8 weights; downstream
                # apply() expects repacked format.
                # MXFP4's serial load helper already performs its required
                # postprocess.  Running it again double-swizzles the same
                # layer.
                _needs_repack = (
                    hasattr(ctx.gpu_method, "process_weights_after_loading")
                    and not getattr(ctx, "_is_mxfp4_quant", False)
                )
                if _needs_repack:
                    ctx.gpu_method.process_weights_after_loading(ctx.gpu_layer)

                t_compute = time.perf_counter()
                result = ctx.gpu_method.apply(ctx.gpu_layer, dispatch_output)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_time = (time.perf_counter() - t_compute) * 1000.0

                # Dynamic expert update: analyze batch and update GPU experts.
                # MUST run BEFORE _restore_raw_attrs() because on Ampere FP8 the
                # Marlin repack changes weight shapes/dtypes, and
                # _restore_raw_attrs() creates empty tensors (torch.empty) to
                # restore the raw fp8 format, destroying the weight data.
                # Skip for V4-Flash MXFP4 — `_update_gpu_experts_from_batch` →
                # `copy_experts_weights_int4` hardcodes int4 weight names
                # (`w13_weight_packed` etc.) and crashes on MXFP4 layouts. The
                # full-GPU fallback re-loads all 256 experts on every fire anyway,
                # so the dynamic-promote optimization is a no-op for MXFP4. Origin:
                # sglang 本身 (V4-Flash full-GPU prefill fallback compat).
                _mxfp4_skip_dyn_update = getattr(ctx, "_is_mxfp4_quant", False)
                if (self.kt_config.kt_enable_dynamic_expert_update
                        and not _mxfp4_skip_dyn_update):
                    t_update = time.perf_counter()
                    self._update_gpu_experts_from_batch(
                        layer=layer,
                        ctx=ctx,
                        dispatch_output=dispatch_output,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    update_time = (time.perf_counter() - t_update) * 1000.0

                    if self.tp_rank == 0:
                        logger.info(
                            "KT layerwise prefill: layer %d compute = %.2f ms, expert update = %.2f ms",
                            self.kt_config.layer_idx,
                            compute_time,
                            update_time,
                        )
                else:
                    if self.tp_rank == 0:
                        logger.info(
                            "KT layerwise prefill: layer %d compute = %.2f ms",
                            self.kt_config.layer_idx,
                            compute_time,
                        )

                # Restore raw format AFTER dynamic update so the context is
                # clean for the next layer's load() call.
                if _needs_repack:
                    ctx._restore_raw_attrs()

                # Debug (2026-08-18, B): full-GPU fallback per-layer dump
                # (the non-layerwise prefill path), to compare with layerwise.
                _nl_dump_dir = os.environ.get("SGLANG_LAYERWISE_DUMP_DIR")
                if _nl_dump_dir and dispatch_output.hidden_states.shape[0] > 1024:
                    try:
                        os.makedirs(_nl_dump_dir, exist_ok=True)
                        torch.save(
                            {
                                "input": x.detach().float().cpu(),
                                "topk_ids": topk_output.topk_ids.detach().cpu(),
                                "topk_weights": topk_output.topk_weights.detach()
                                .float()
                                .cpu(),
                                "output": result.hidden_states.detach().float().cpu(),
                            },
                            os.path.join(
                                _nl_dump_dir,
                                f"nl_layer{self.kt_config.layer_idx:02d}.pt",
                            ),
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.warning("[nl-dump] layer=%s dump failed: %s",
                                       getattr(self.kt_config, "layer_idx", "?"), exc)

                return result

        # Step 1: Copy hidden_states to staging buffer and submit CPU computation
        # Staging buffer allows GPU computation to proceed without waiting for D2H copy
        staging_buffer = None
        if self.tp_rank == 0 and self._cpu_stream is not None:
            # Use shared staging buffer (shared across all MoE layers to save GPU memory)
            assert self._shared_staging_buffer is not None, "Shared staging buffer not initialized"
            staging_buffer = self._shared_staging_buffer.get_slice(x.shape[0])

            # Copy to staging buffer on main stream
            staging_buffer.copy_(x, non_blocking=True)

            # SGLANG_KT_HYBRID_NO_CPU_STREAM=1 collapses cpu_stream onto main stream.
            _no_cpu_stream = os.environ.get("SGLANG_KT_HYBRID_NO_CPU_STREAM") == "1"
            if not _no_cpu_stream:
                # Fork to cpu_stream (waits for staging copy to complete)
                self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            from contextlib import nullcontext as _ctx_null
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Submit uses staging_buffer, so GPU can modify original x freely
                _kt_t_pre_cxx = time.perf_counter() if _kt_timing else None
                self._submit_with_staged_input(
                    layer, dispatch_output, staging_buffer
                )
                if _kt_t_pre_cxx is not None:
                    _kt_t_cxx = time.perf_counter()
        if _kt_timing:
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
            _kt_t_after_submit = time.perf_counter()

        # Step 2: Prepare GPU computation by masking and remapping expert IDs
        # CPU expert IDs are set to -1; GPU expert IDs are remapped to GPU weight indices
        topk_ids = topk_output.topk_ids
        # Measurement only (KT_ROUTE_HIST): the LOGICAL ids, read before
        # mask_and_remap_expert_ids rewrites CPU experts to -1.  Placed after
        # the CPU submit above, so the bincount overlaps the CPU expert math.
        kt_route_hist_note(
            self.kt_config.layer_idx,
            int(self.gpu_experts_mask.numel()),
            num_tokens,
            topk_ids,
        )
        kt_distinct_note(int(self.gpu_experts_mask.numel()), num_tokens, topk_ids)
        masked_topk_ids = mask_and_remap_expert_ids(
            topk_ids, self.gpu_experts_mask_cuda, self.logical_to_gpu_index_cuda
        )

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )
        if _kt_timing:
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
            _kt_t_after_mask = time.perf_counter()

        # Step 3: Execute GPU expert computation on main stream
        # No wait needed - staging buffer decouples CPU and GPU data access
        # When num_gpu_experts == 0 the gpu_method's weights have shapes that
        # are incompatible with its own apply() (e.g. on SM_120 with V4 Flash
        # where the only routed-expert quant method available, the FP8 fused
        # MoE Triton path, asserts hidden_states.shape[1] == w1.shape[2] -
        # padded_size, which fails because w1 is the empty 0-expert slice).
        # Skip the GPU GEMM entirely and start from zeros; the CPU path then
        # provides 100% of the routed-expert contribution.
        # Origin: kt-sglang 耦合 (sglang/kt_ep_wrapper.py).
        if not getattr(self, "_diag_logged", False) and logger.isEnabledFor(logging.DEBUG):
            self._diag_logged = True
            try:
                _mask_sum = int(self.gpu_experts_mask.sum().item())
            except Exception as e:  # pragma: no cover
                _mask_sum = f"err:{e}"
            logger.debug(
                "[kt-ep-diag] layer=%s num_gpu_experts=%d mask_sum=%s "
                "mask_shape=%s gpu_method=%s",
                getattr(self.kt_config, 'layer_idx', '?'),
                self.num_gpu_experts,
                _mask_sum,
                tuple(self.gpu_experts_mask.shape),
                type(self.gpu_method).__name__,
            )
        # SGLANG_KT_BYPASS_GPU_MOE=1 also short-circuits to zeros, because
        # the kt mask generator returns an all-True (num_gpu_experts ==
        # num_total_experts) per-layer mask in some configurations (e.g. V4
        # Flash + --kt-num-gpu-experts=0), which defeats the
        # num_gpu_experts==0 short-circuit. The env var lets the operator
        # force the bypass without untangling the mask generator.
        if self.num_gpu_experts == 0 or os.environ.get("SGLANG_KT_BYPASS_GPU_MOE") == "1":
            gpu_combine_input = None
            # 0-expert: no GPU routed experts. Skip the zeros_like allocation
            # and add; the merge below takes the CPU output directly (saves
            # one alloc + one add per layer per decode step).
            output = None
            # 2604B sub-mode adds a runtime path-checker assertion in the
            # model (deepseek_v4.py:1169 expects observed == 1 after every
            # MoE forward). The trtllm path bumps it inside its body; the
            # bypass path mirrors that here so the assertion still passes
            # when GPU MoE is short-circuited in favour of CPU experts.
            from sglang.srt.environ import envs as _envs
            if _envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
                from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
                    deepseek_v4_moe_code_path_checker,
                )
                deepseek_v4_moe_code_path_checker.observed += 1
        else:
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
            output = gpu_combine_input.hidden_states
        if _kt_timing:
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
            _kt_t_after_gpu = time.perf_counter()

        # Step 4: Sync CPU results on cpu_stream, then synchronize streams
        if self.tp_rank == 0 and self._cpu_stream is not None:
            _no_cpu_stream = os.environ.get("SGLANG_KT_HYBRID_NO_CPU_STREAM") == "1"
            from contextlib import nullcontext as _ctx_null
            _stream_ctx = _ctx_null() if _no_cpu_stream else torch.cuda.stream(self._cpu_stream)
            with _stream_ctx:
                # Use staging_buffer for sync to get correct buffer reference
                _kt_t_sync_pre = time.perf_counter() if _kt_t_apply_start is not None else None
                cpu_output = self._sync_with_staged_input(staging_buffer)
                if _kt_t_sync_pre is not None:
                    _kt_t_cpu_wait_ms = (time.perf_counter() - _kt_t_sync_pre) * 1000.0
                if not _no_cpu_stream:
                    self._sync_done_event.record(self._cpu_stream)
            if _kt_timing:
                _kt_t_after_sync = time.perf_counter()

            # Main stream waits for cpu_stream to complete before merging results
            if not _no_cpu_stream:
                torch.cuda.current_stream(x.device).wait_event(self._sync_done_event)
            if output is None:
                output = cpu_output
            else:
                output = output + cpu_output
            # Diagnostics: detect NaN/Inf in the merged MoE output (origin: kt-sglang
            # 耦合 decode GPU-expert offload debug). Gated on env to keep the hot
            # path clean.
            if os.environ.get("SGLANG_KT_NAN_DIAG") == "1" and (
                not torch.isfinite(output).all()
            ):
                logger.warning(
                    "[kt-nan] layer=%s num_gpu_experts=%d nonfinite=%d "
                    "gpu_nan=%s cpu_nan=%s",
                    getattr(self.kt_config, "layer_idx", "?"),
                    self.num_gpu_experts,
                    int((~torch.isfinite(output)).sum()),
                    (gpu_combine_input is not None
                     and bool((~torch.isfinite(output)).any())),
                    bool((~torch.isfinite(cpu_output)).any()),
                )
        if _kt_timing:
            _kt_t_after_merge = time.perf_counter()
            # Optional: synchronize GPU at end of apply() to capture true GPU
            # work latency (otherwise gpu_apply Python time only captures
            # kernel-launch CPU overhead, not actual GPU compute). DEEP mode
            # serialises streams so per-stage numbers reflect GPU work, not
            # async launch return.
            if os.environ.get("SGLANG_KT_HYBRID_TIMING_DEEP") == "1":
                torch.cuda.synchronize(x.device)
                _kt_t_after_merge = time.perf_counter()

        if _kt_t_apply_start is not None:
            _kt_total_ms = (_kt_t_after_merge - _kt_t_apply_start) * 1000.0
            _stage_submit_ms = (_kt_t_after_submit - _kt_t_apply_start) * 1000.0
            _stage_mask_ms = (_kt_t_after_mask - _kt_t_after_submit) * 1000.0
            _stage_gpu_ms = (_kt_t_after_gpu - _kt_t_after_mask) * 1000.0
            _stage_sync_ms = (
                (_kt_t_after_sync - _kt_t_after_gpu) * 1000.0
                if _kt_t_after_sync is not None else 0.0
            )
            _stage_merge_ms = (
                (_kt_t_after_merge - _kt_t_after_sync) * 1000.0
                if _kt_t_after_sync is not None
                else (_kt_t_after_merge - _kt_t_after_gpu) * 1000.0
            )
            _cls = type(self)
            if not hasattr(_cls, '_kt_layer_step'):
                _cls._kt_layer_step = {}
            _li = getattr(self.kt_config, 'layer_idx', -1)
            _cls._kt_layer_step[_li] = _cls._kt_layer_step.get(_li, 0) + 1
            _step = _cls._kt_layer_step[_li]
            _sub_cxx_ms = (
                (_kt_t_cxx - _kt_t_pre_cxx) * 1000.0
                if _kt_t_cxx is not None and _kt_t_pre_cxx is not None
                else 0.0
            )
            if _step <= 16 or _step % 16 == 0:
                logger.info(
                    "[kt-time] layer=%s step=%d total=%.2fms submit=%.2f "
                    "sub_py=%.2f sub_cxx=%.2f mask=%.2f gpu=%.2f sync=%.2f "
                    "merge=%.2f cpu_wait=%.2fms num_tokens=%d",
                    _li, _step, _kt_total_ms, _stage_submit_ms,
                    _stage_submit_ms - _sub_cxx_ms, _sub_cxx_ms,
                    _stage_mask_ms, _stage_gpu_ms, _stage_sync_ms,
                    _stage_merge_ms, _kt_t_cpu_wait_ms, num_tokens,
                )
        self._dump_hybrid_output(output, num_tokens, x, topk_output)
        return StandardCombineInput(hidden_states=output)

    def _dump_hybrid_output(self, output, num_tokens, x, topk_output):
        # Debug (2026-08-18, B): env-gated per-layer hybrid (CPU AMX + GPU
        # marlin) MoE output dump, to compare against the layerwise (all-GPU
        # marlin) dumps at common positions. Prefill-only (batch > 1024) to
        # avoid decode-step flooding.
        _nl_dump_dir = os.environ.get("SGLANG_LAYERWISE_DUMP_DIR")
        if _nl_dump_dir and num_tokens > 1024:
            try:
                os.makedirs(_nl_dump_dir, exist_ok=True)
                torch.save(
                    {
                        "input": x.detach().float().cpu(),
                        "topk_ids": topk_output.topk_ids.detach().cpu(),
                        "topk_weights": topk_output.topk_weights.detach()
                        .float()
                        .cpu(),
                        "output": output.detach().float().cpu(),
                    },
                    os.path.join(
                        _nl_dump_dir, f"nl_layer{self.kt_config.layer_idx:02d}.pt"
                    ),
                )
            except Exception:
                pass

    def _apply_capture_stub(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Allocate the static bridge output without invoking KT during capture.

        KT submits host work and CUDA copies at replay time; a breakable
        graph captures the GPU segments around this eager node while
        retaining their launch-overhead benefit. During capture the KT body
        is skipped (contents are never consumed; warmup and replay run the
        real _apply_impl) so no CPU submit / staging copy is recorded into
        the captured graph. Origin: kt-sglang 耦合 (mirrors the local
        kt_ep_wrapper's capture stub on the upstream arch)."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        del layer
        return StandardCombineInput(
            hidden_states=torch.zeros_like(dispatch_output.hidden_states)
        )

    # KT submits host work and CUDA copies at replay time. A breakable graph
    # captures the GPU segments around this eager node while retaining their
    # launch-overhead benefit; a monolithic/full CUDA graph cannot represent it.
    apply = eager_on_graph(True, capture_stub=_apply_capture_stub)(_apply_impl)

    def _update_gpu_experts_from_batch(
        self,
        layer: torch.nn.Module,
        ctx: "SharedFullContext",
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Update original layer's GPU experts based on current batch statistics.

        This method:
        1. Analyzes topk_ids to find most frequently activated experts
        2. Copies selected expert weights from ctx.gpu_layer to layer
        3. Updates all mapping tables (gpu_experts_mask, logical_to_gpu_index, etc.)
        4. Broadcasts changes across TP ranks for consistency

        Args:
            layer: Original MoE layer with subset of GPU experts
            ctx: SharedFullContext containing temporary full GPU layer
            dispatch_output: Current batch dispatch output with routing information
        """
        # Step 1: Select top experts (rank 0 computes, broadcasts to all ranks)
        topk_ids = dispatch_output.topk_output.topk_ids
        device = topk_ids.device

        if self.tp_rank == 0:
            selected_experts = select_top_experts_from_batch(
                topk_ids=topk_ids,
                num_experts=self.global_num_experts,
                num_gpu_experts=self.num_gpu_experts,
            )
        else:
            # Create placeholder on other ranks
            selected_experts = torch.zeros(
                self.num_gpu_experts, dtype=torch.int64, device=device
            )

        # Broadcast selected experts to all ranks for consistent weight updates
        if dist.is_initialized():
            dist.broadcast(selected_experts, src=0, group=get_tp_group().device_group)

        # Step 2: Copy selected expert weights from ctx.gpu_layer to layer.
        # Both are already in inference format: apply() already called
        # process_weights_after_loading() which handles Marlin repack.
        if ctx._is_fp8_quant:
            # Ampere Marlin vs native FP8 block quant: Marlin repack renames
            # w13_weight_scale_inv → w13_weight_scale and changes w13_weight
            # dtype fp8→int32.  Use gpu_method class name (invariant) to pick
            # the right attribute-name list.
            if ctx.gpu_method.__class__.__name__.endswith("MarlinMoEMethod"):
                copy_experts_weights_fp8_channel(
                    src_layer=ctx.gpu_layer, dst_layer=layer,
                    selected_experts=selected_experts,
                )
            else:
                copy_experts_weights_fp8(
                    src_layer=ctx.gpu_layer, dst_layer=layer,
                    selected_experts=selected_experts,
                )
        elif ctx._is_fp8_channel_quant:
            copy_experts_weights_fp8_channel(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        elif ctx._is_bf16_quant:
            copy_experts_weights_bf16(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )
        else:
            copy_experts_weights_int4(
                src_layer=ctx.gpu_layer,
                dst_layer=layer,
                selected_experts=selected_experts,
            )

        # Step 3: Update mapping tables
        gpu_experts_mask_cpu, logical_to_gpu_index_cuda, gpu_index_to_logical_cpu = (
            update_gpu_expert_mappings(
                selected_experts=selected_experts,
                num_experts=self.global_num_experts,
                device=device,
            )
        )

        # Update instance variables (both CPU and CUDA versions)
        # CRITICAL: Use .copy_() for CUDA tensors to maintain same buffer for CUDA graph compatibility
        # CUDA graph captures tensor memory addresses during decode phase, so we must update
        # in-place rather than replacing the tensor reference
        self.gpu_experts_mask = gpu_experts_mask_cpu  # CPU tensor, safe to replace
        self.gpu_experts_mask_cuda.copy_(gpu_experts_mask_cpu)  # In-place update for CUDA graph
        self.logical_to_gpu_index = logical_to_gpu_index_cuda.cpu()  # CPU version for weight loading
        self.logical_to_gpu_index_cuda.copy_(logical_to_gpu_index_cuda)  # In-place update for CUDA graph
        self.gpu_index_to_logical = gpu_index_to_logical_cpu  # CPU tensor, safe to replace

        # Step 4: Update KT wrapper (rank 0 only)
        if self.tp_rank == 0:
            update_kt_wrapper_masks(self.wrapper, gpu_experts_mask_cpu)

        # Log expert changes (rank 0 only)
        if self.tp_rank == 0:
            logger.debug(
                "KT dynamic update: layer %d updated GPU experts to: %s",
                self.kt_config.layer_idx,
                selected_experts.cpu().tolist(),
            )

    def bridge_cuda_graph_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Copy a layer-local tensor into an owner-retained buffer that spans
        the KT breakable-graph seam. Local addition (DSPARK BCG support)."""
        return self.graph_state_bridge.copy(name, tensor)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)

    def _build_full_context(self, layer: torch.nn.Module) -> "SharedFullContext":
        global _SHARED_FULL_CONTEXT

        if _SHARED_FULL_CONTEXT is None:
            _SHARED_FULL_CONTEXT = SharedFullContext(
                layer=layer,
                init_args=self._full_init_args,
                global_num_experts=self.global_num_experts,
                moe_runner_config=self.moe_runner_config,
            )

        _SHARED_FULL_CONTEXT.load(
            layer_idx=self.kt_config.layer_idx,
            wrapper=self.wrapper,
            original_layer=layer,
            gpu_experts_mask=self.gpu_experts_mask,
            logical_to_gpu_index=self.logical_to_gpu_index,
        )
        return _SHARED_FULL_CONTEXT


# ---------------------------------------------------------------------------
# Plugin registration: makes KTEPWrapperMethod available to FusedMoE without
# any base-file import. Activated by importing this module, which happens via
# sglang.srt.models.deepseek_v4 -> auto-discovered by ModelRegistry.
# ---------------------------------------------------------------------------

def _kt_ep_predicate(layer, server_args):
    return create_kt_config_from_server_args(
        server_args, layer.layer_id, layer.num_local_experts
    )


def _kt_ep_factory(layer, gpu_method, kt_config):
    return KTEPWrapperMethod(gpu_method, kt_config)


from sglang.srt.layers.moe.quant_method_registry import register_moe_quant_wrapper

# priority=20 → wraps after mxfp4 (matches PR #38 Phase 3 → outer wrapper)
register_moe_quant_wrapper(
    "kt_ep", _kt_ep_predicate, _kt_ep_factory, priority=20
)
