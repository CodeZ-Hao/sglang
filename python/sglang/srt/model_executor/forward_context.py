"""Per-forward-call control context.

Owns ForwardContext — a frozen dataclass holding control configs the model
layer reads at depth via get_forward_context(). The only mandatory field
today is attn_backend; pool refs are derived from attn_backend.*
(every backend caches them at __init__), so a published ForwardContext
is enough to resolve the active pools without a separate global.

ModelRunner._forward_raw publishes a fresh ForwardContext for the
duration of each forward; callers that need a per-call override (PDmux
per-stream backend, frozen-KV MTP draft loop, TBO per-child dispatch) use
dataclasses.replace and wrap the override scope with forward_context().

Distinct from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.TcPiecewiseForwardContext,
which collects compilation-time refs for the piecewise CUDA graph backend.

Concurrency: _current is a contextvars.ContextVar (thread-local). Each
thread holds its own active ForwardContext, so a decode thread can run a
forward concurrently with a prefill thread in the same process without
clobbering the other's context.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs. Read via get_forward_context();
    extend by adding fields here. Frozen so accidental mutation raises at
    write time — use dataclasses.replace for per-call overrides."""

    attn_backend: AttentionBackend


_current: "contextvars.ContextVar[Optional[ForwardContext]]" = contextvars.ContextVar(
    "forward_context_current", default=None
)


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set the active context in THIS thread; return the previous one for
    explicit save/restore. Prefer the forward_context() context manager."""
    prev = _current.get()
    _current.set(ctx)
    return prev


def has_forward_context() -> bool:
    return _current.get() is not None


def get_forward_context() -> ForwardContext:
    ctx = _current.get()
    assert ctx is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return ctx


def get_attn_backend() -> AttentionBackend:
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> KVCache:
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool


@contextmanager
def forward_context(ctx: ForwardContext):
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
