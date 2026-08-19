from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_FLASHINFER_MULTIARCH_SAMPLING_LOCK = threading.Lock()
_FLASHINFER_MULTIARCH_SAMPLING_READY: set[tuple[str, ...]] = set()

_SUPPORTED_DRAFT_BACKENDS = (
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "trtllm_mha",
    "ascend",
)


def _normalize_cuda_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("gpu-").replace("-", "")


def resolve_speculative_draft_device(device: str) -> int:
    """Resolve a logical CUDA index, ``cuda:N``, or CUDA UUID."""
    raw = str(device).strip()
    index_text = raw[5:] if raw.lower().startswith("cuda:") else raw
    if index_text.isdecimal():
        index = int(index_text)
    else:
        wanted = _normalize_cuda_uuid(raw)
        index = -1
        for candidate in range(torch.cuda.device_count()):
            uuid = getattr(torch.cuda.get_device_properties(candidate), "uuid", None)
            if uuid is not None and _normalize_cuda_uuid(str(uuid)) == wanted:
                index = candidate
                break
        if index < 0:
            raise ValueError(
                "Could not resolve --speculative-draft-device "
                f"{device!r} to a visible CUDA device."
            )

    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            "--speculative-draft-device resolved to logical CUDA index "
            f"{index}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    return index


@contextmanager
def draft_cuda_device_context(gpu_id: int):
    """Make all implicit ``device='cuda'`` draft allocations device-local."""
    with torch.cuda.device(int(gpu_id)):
        yield


def _flashinfer_arch_names(device_ids: tuple[int, ...]) -> tuple[str, ...]:
    """Return FlashInfer's normalized CUDA architecture names."""
    from flashinfer.compilation_context import CompilationContext

    normalized = {
        CompilationContext._normalize_cuda_arch(  # noqa: SLF001
            *torch.cuda.get_device_capability(device_id)
        )
        for device_id in device_ids
    }
    return tuple(f"{major}.{minor}" for major, minor in sorted(normalized))


def ensure_flashinfer_sampling_multiarch(device_ids: tuple[int, ...]) -> None:
    """Load an architecture-complete sampling module for remote drafting.

    FlashInfer 0.6.x caches its sampling module process-wide and prefers an
    installed AOT/JIT-cache binary. If that binary lacks either member of a
    heterogeneous target/draft pair, the first affected sampling primitive
    fails with ``cudaErrorNoKernelImageForDevice``. Force only this module
    through a dedicated JIT directory with every worker architecture enabled.
    """
    arch_names = _flashinfer_arch_names(tuple(sorted(set(device_ids))))
    if len(arch_names) <= 1:
        return

    with _FLASHINFER_MULTIARCH_SAMPLING_LOCK:
        if arch_names in _FLASHINFER_MULTIARCH_SAMPLING_READY:
            return

        import flashinfer
        import flashinfer.sampling as flashinfer_sampling
        from flashinfer.jit import env as flashinfer_jit_env

        arch_key = "_".join(name.replace(".", "") for name in arch_names)
        cache_root = (
            Path(flashinfer_jit_env.FLASHINFER_CACHE_DIR)
            / flashinfer.__version__
            / f"sglang_multiarch_sampling_{arch_key}"
        )
        jit_dir = cache_root / "cached_ops"

        old_arch_list = os.environ.get("FLASHINFER_CUDA_ARCH_LIST")
        old_aot_dir = flashinfer_jit_env.FLASHINFER_AOT_DIR
        old_jit_dir = flashinfer_jit_env.FLASHINFER_JIT_DIR

        logger.info(
            "Preparing FlashInfer sampling kernels for heterogeneous CUDA "
            "architectures: %s.",
            ", ".join(arch_names),
        )
        # Replace a wrapper that may already close over a single-architecture
        # module. FlashInfer's 0.6.x registration decorators are intentionally
        # no-ops, so reconstructing this cached namespace is safe.
        flashinfer_sampling.get_sampling_module.cache_clear()
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = " ".join(arch_names)
        flashinfer_jit_env.FLASHINFER_AOT_DIR = cache_root / "no_aot"
        flashinfer_jit_env.FLASHINFER_JIT_DIR = jit_dir
        try:
            flashinfer_sampling.get_sampling_module()
        except Exception as error:
            flashinfer_sampling.get_sampling_module.cache_clear()
            raise RuntimeError(
                "Failed to build the heterogeneous FlashInfer sampling module "
                f"for {arch_names}. Ensure nvcc from the CUDA toolkit is available."
            ) from error
        finally:
            flashinfer_jit_env.FLASHINFER_AOT_DIR = old_aot_dir
            flashinfer_jit_env.FLASHINFER_JIT_DIR = old_jit_dir
            if old_arch_list is None:
                os.environ.pop("FLASHINFER_CUDA_ARCH_LIST", None)
            else:
                os.environ["FLASHINFER_CUDA_ARCH_LIST"] = old_arch_list

        _FLASHINFER_MULTIARCH_SAMPLING_READY.add(arch_names)
        logger.info(
            "FlashInfer heterogeneous sampling module ready: architectures=%s "
            "cache=%s.",
            ",".join(arch_names),
            jit_dir,
        )


class DraftWorkerBundle(msgspec.Struct, frozen=True):
    draft_worker: TpModelWorker
    draft_model_runner: ModelRunner
    draft_model: torch.nn.Module
    resolved_attention_backend: str


def _resolve_draft_attention_backend_fallback(
    *, draft_server_args: ServerArgs, algo_label: str
) -> str:
    draft_backend = draft_server_args.speculative_draft_attention_backend
    if draft_backend is None:
        draft_backend, _ = draft_server_args.get_attention_backends()
    if draft_backend is None:
        return "triton" if torch.version.hip else "flashinfer"
    if draft_backend not in _SUPPORTED_DRAFT_BACKENDS:
        fallback = "triton" if torch.version.hip else "flashinfer"
        logger.warning(
            "%s draft worker only supports attention_backend in %s for now, "
            "but got %r. Falling back to '%s'.",
            algo_label,
            _SUPPORTED_DRAFT_BACKENDS,
            draft_backend,
            fallback,
        )
        return fallback
    return draft_backend


def build_draft_tp_worker(
    *,
    server_args: ServerArgs,
    gpu_id: int,
    ps: ParallelState,
    nccl_port: int,
    target_model_config: ModelConfig,
    algo_label: str,
    attention_backend_override: Optional[str] = None,
    fp8_gemm_backend_override: Optional[str] = None,
) -> DraftWorkerBundle:
    draft_server_args = deepcopy(server_args)
    # An override names a draft-specific backend the caller has already
    # validated (e.g. a self-drafting architecture); it skips the generic
    # supported-backend fallback below.
    draft_backend = attention_backend_override or (
        _resolve_draft_attention_backend_fallback(
            draft_server_args=draft_server_args, algo_label=algo_label
        )
    )
    draft_cuda_graph_config = draft_server_args.cuda_graph_config
    if (
        server_args.kt_weight_path is not None
        and algo_label != "DSPARK"
        and (Phase.DECODE, "backend")
        not in getattr(server_args, "_cuda_graph_config_locked", set())
        and draft_cuda_graph_config.decode.backend == Backend.BREAKABLE
    ):
        # Target setup automatically changes the default full graph to a
        # breakable graph for KT eager nodes. The DFLASH/EAGLE-style drafts
        # have no KT nodes, so restore their original full-graph fast path
        # without overriding an explicit user graph choice. DSPARK's DeepSeek-
        # V4 draft does run KT nodes (mtp.{stage} experts), so it must stay
        # BREAKABLE.
        draft_cuda_graph_config = replace(
            draft_cuda_graph_config,
            decode=replace(draft_cuda_graph_config.decode, backend=Backend.FULL),
        )
    # Post-resolution ServerArgs rejects bare assignment; route the draft-copy
    # adjustments through the audited mutation point. Keep the resolved value
    # on speculative_draft_attention_backend: downstream draft-worker logic
    # keys on that field (backend selection in _get_attention_backend and the
    # fa4-draft KV dtype override in configure_kv_cache_dtype), so nulling it
    # would silently skip those paths. context_length keeps the draft aligned
    # with the target.
    draft_server_args.override(
        "draft_worker.build",
        skip_tokenizer_init=True,
        speculative_draft_attention_backend=draft_backend,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        attention_backend=draft_backend,
        context_length=target_model_config.context_len,
        cuda_graph_config=draft_cuda_graph_config,
        fp8_gemm_runner_backend=(
            fp8_gemm_backend_override
            if fp8_gemm_backend_override is not None
            else draft_server_args.fp8_gemm_runner_backend
        ),
        # KT offload: the DSpark draft uses the target's kt_weight_path (the
        # draft's mtp.{stage} experts are read from the same checkpoint via
        # weight_key_prefix). Do not null kt_weight_path here, or the draft's
        # KT layers will fail to construct.
        # kt_weight_path=None,
        # KT owns the routed-expert space (256 slots = the GPU mask width); a
        # fused shared-expert slot (257th) breaks the KT mask invariant. Keep
        # the draft's shared expert as a separate GPU-resident module.
        disable_shared_experts_fusion=True,
    )

    # The draft's layers must resolve config from the draft's own bags.
    with get_context().preserve_config():
        get_context().set_server_args(draft_server_args)
        draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            is_draft_worker=True,
        )

    draft_model_runner = draft_worker.model_runner
    draft_worker.draft_runner = draft_model_runner
    return DraftWorkerBundle(
        draft_worker=draft_worker,
        draft_model_runner=draft_model_runner,
        draft_model=draft_model_runner.model,
        resolved_attention_backend=draft_backend,
    )


def make_draft_input_v2(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    bs = int(new_seq_lens.numel())
    device = bonus_tokens.device
    return DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
        topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
        bonus_tokens=bonus_tokens.to(dtype=torch.int64),
        new_seq_lens=new_seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
    )


def make_draft_block_spec_info(
    *,
    draft_token_num: int,
    device: torch.device,
) -> DFlashVerifyInput:
    return DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=int(draft_token_num),
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def make_draft_sampler_capture_hook(draft_sampler):

    def capture_hook(runner, out, forward_batch, num_tokens):
        del runner, num_tokens
        if not isinstance(out, LogitsProcessorOutput) or out.hidden_states is None:
            raise RuntimeError(
                "draft sampler set but the draft forward has no "
                "hidden_states to capture into the graph."
            )
        draft_sampler(out.hidden_states, forward_batch.input_ids)

    return capture_hook


def build_block_pos_offsets(*, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(int(length), device=device, dtype=torch.int64)
