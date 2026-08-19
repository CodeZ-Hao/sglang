"""PD-parallel (second-channel) scheduler mixin.

The overlap scheduler runs ONE batch per iteration with prefill priority, so a
large chunked prefill freezes the decode-mode requests of the running batch for
the whole prefill duration (measured: decode TTFT 0.3s -> ~10-13s while a
30K-50K prefill runs).  This mixin adds an opt-in background thread that steps
work on a second CUDA stream while the main scheduler thread runs the prefill
chunks.

Resource model: prefill chunks at/above ``--kt-gpu-prefill-token-threshold`` use
the GPU-layerwise MoE path (GPU-heavy, CPU idle); decode uses the CPU AMX MoE +
light GPU attention.  The two overlap without starving each other.

The second channel takes work by priority:
  1. decode-mode requests of the running batch (the CPU channel's dominant job);
  2. a CPU (AMX) prefill of the next waiting request while the GPU channel is
     busy with a layerwise prefill batch -- the GPU and CPU prefill channels
     then run concurrently and each request completes prefill staggered in
     FIFO order.

Gated by ``SGLANG_PD_PARALLEL=1`` (default off).  The decode thread reuses the
scheduler's own machinery (``update_running_batch`` / ``process_batch_result``)
and the shared req objects, so when the prefill ends the main thread simply
resumes decode on the already-advanced requests.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import RelayPayload, resolve_forward_inputs
from sglang.srt.mem_cache.common import (
    free_swa_out_of_window_slots,
    maybe_cache_unfinished_req,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


class SchedulerPdParallelMixin:
    """Mixin hooks for PD-parallel decode/prefill on a second channel (opt-in)."""

    def init_pd_parallel(self):
        self.pd_enabled = envs.SGLANG_PD_PARALLEL.get() and self.enable_overlap
        self.pd_active = False
        self._pd_drained = False
        self._pd_gen = 0
        self.pd_thread: Optional[threading.Thread] = None
        self.pd_stop_event = threading.Event()
        self.pd_batch: Optional[ScheduleBatch] = None
        # CPU-channel prefill state. decode (self.pd_batch) always has priority
        # over it; a prefill can be paused (thread stopped) with pd_cpu_req kept
        # and resumed on the next GPU prefill batch.
        self.pd_cpu_req: Optional[Req] = None
        self.pd_cpu_batch: Optional[ScheduleBatch] = None
        # When decode-mode reqs appear while the channel is CPU-prefilling, the
        # main thread records them here; the channel switches to decode at the
        # next chunk boundary (decode has priority, but a mid-forward CPU prefill
        # is not preempted).
        self.pd_decode_pending: Optional[List[Req]] = None
        # The radix node we hold a lock ref on for the CPU-channel req (set on
        # the main thread inside _pd_prepare_cpu_prefill_chunk).
        self._pd_cpu_locked_node = None
        # Set by _pd_prepare_cpu_prefill_chunk when the req has no page-aligned
        # chunk of work left; the channel hands the req back to the scheduler.
        self._pd_cpu_release_pending = False
        # Set by the channel thread after a non-final CPU prefill chunk; the
        # main thread performs the deferred cache_unfinished_req so all radix
        # tree mutations stay single-threaded.
        self._pd_cpu_need_cache = False
        # Reqs that entered via the CPU channel and therefore are NOT listed in
        # the main thread's running_batch; _pd_stop merges them back so no req
        # is orphaned when the channel is torn down.
        self.pd_extra_reqs = set()
        # CPU prefill chunk (tokens). Deliberately SEPARATE from the GPU's
        # --chunked-prefill-size: the CPU AMX channel is far slower than the
        # GPU Marlin channel, so it gets a smaller chunk (default 2048) for
        # finer-grained decode-priority preemption. Must stay below the
        # GPU-layerwise threshold so the MoE routes through the CPU AMX path.
        self._pd_cpu_chunk_tokens = max(
            256, int(os.environ.get("SGLANG_PD_CPU_PREFILL_CHUNK", "2048"))
        )
        self._pd_gpu_threshold = int(
            getattr(
                getattr(self, "server_args", None), "kt_gpu_prefill_token_threshold", 0
            )
        )
        # Set by the channel thread when decode reqs arrive mid-CPU-prefill: the
        # in-progress prefill is handed back to the GPU channel (see
        # _pd_handle_cpu_handoff) and the channel switches to decode immediately.
        self._pd_cpu_handoff = False
        # Lazily-created per-decode-thread CUDA state.
        self.pd_stream = None
        self.pd_copy_stream = None
        self.pd_future_map = None
        self._pd_initialized = False
        # Mirror the main future_map's flags (computed in __init__).
        self._pd_needs_cpu_seq_lens = False
        self._pd_needs_confidence_relay = False
        if self.pd_enabled:
            fm = getattr(self, "future_map", None)
            if fm is not None:
                self._pd_needs_cpu_seq_lens = fm.needs_cpu_seq_lens
                self._pd_needs_confidence_relay = fm.needs_confidence_relay
            logger.info(
                "[PD] PD-parallel second channel ENABLED (SGLANG_PD_PARALLEL=1); "
                "cpu-prefill chunk=%d gpu-threshold=%d.",
                self._pd_cpu_chunk_tokens,
                self._pd_gpu_threshold,
            )

    # ------------------------------------------------------------------
    # lifecycle / lazy init
    # ------------------------------------------------------------------
    def _pd_lazy_init(self) -> None:
        if self._pd_initialized:
            return
        dev_mod = self.device_module
        # DEFAULT stream priority (not higher): a higher-priority second-channel
        # stream can preempt the main thread's GPU-layerwise prefill SLOTLOAD
        # kernels mid-preparation, corrupting the Marlin slot state. Decode is
        # not starved in practice because it is CPU-bound (AMX) with light GPU
        # attention.
        self.pd_stream = dev_mod.Stream()
        self.pd_copy_stream = dev_mod.Stream()
        self.pd_future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=self._pd_needs_cpu_seq_lens,
            needs_confidence_relay=self._pd_needs_confidence_relay,
        )
        # Per-thread attention backend for the CPU-AMX prefill channel is created
        # LAZILY (only when a CPU prefill is about to run) so the decode path
        # never constructs a second DeepseekV4AttnBackend -- its construction
        # captures/allocates CUDA graph state that can interfere with the
        # breakable-graph decode. See _pd_ensure_attn_backend.
        self.pd_attn_backend = None
        self._pd_initialized = True

    def _pd_ensure_attn_backend(self) -> None:
        """Create the per-thread attention backend for the CPU-AMX prefill
        channel. Two concurrent prefill forwards on the SHARED backend race the
        DSV4 compression plan (c_plan.cuh plan_compress_prefill InternalError),
        so the channel publishes its own backend via forward_context. Created
        lazily on the first CPU prefill (NOT at decode start) to avoid
        disturbing the breakable-graph decode's captured state."""
        if self.pd_attn_backend is not None:
            return
        if not envs.SGLANG_PD_CPU_PREFILL.get():
            return
        runner = getattr(getattr(self, "model_worker", None), "model_runner", None)
        if runner is None or getattr(runner, "attn_backend", None) is None:
            return
        try:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )

            self.pd_attn_backend = DeepseekV4AttnBackend(runner)
            logger.info(
                "[PD] created per-thread attention backend for CPU prefill."
            )
        except Exception:
            logger.warning(
                "[PD] could not create per-thread attention backend; "
                "CPU prefill disabled.",
                exc_info=True,
            )
            self.pd_attn_backend = None

    # ------------------------------------------------------------------
    # main-thread hooks (called from event_loop_overlap)
    # ------------------------------------------------------------------
    def _pd_maybe_start(self, batch: Optional[ScheduleBatch]) -> None:
        """Start the decode thread when a prefill batch is about to run while
        the running batch still has decode-mode (already-generating) requests."""
        if not self.pd_enabled:
            return
        if batch is None or not batch.forward_mode.is_extend():
            return
        rb = self.running_batch
        if rb is None or rb.is_empty():
            return
        # Only hand off decode reqs that still own a pool slot; a retracted or
        # just-freed req (req_pool_idx None) must not enter the PD thread.
        decode_reqs = [
            r
            for r in rb.reqs
            if len(r.output_ids) > 0 and r.req_pool_idx is not None
        ]
        if not decode_reqs:
            return
        if self.pd_cpu_req is not None:
            # A CPU prefill request is claimed; decode-mode reqs are recorded
            # by _pd_note_decode_pending so the channel switches to decode at
            # the next chunk boundary (decode-priority handoff).
            return
        if self.pd_active:
            # Channel is decoding; new decode reqs wait for it to drain.
            return
        self._pd_start(decode_reqs)

    def _pd_note_decode_pending(self) -> None:
        """Record decode-mode reqs in ``pd_decode_pending`` so the channel can
        pick them up at the next opportunity. Called every event-loop iteration
        (also when batch is None).

        - CPU channel mid-prefill (pd_cpu_req set): the channel switches to
          decode at the next chunk boundary (decode-priority handoff).
        - Channel decoding (pd_batch set): new decode-mode reqs that joined the
          main thread's running_batch (their prefill completed while the channel
          was busy) are folded into the decode batch so the 4 concurrent
          requests decode TOGETHER instead of serially.
        """
        if not self.pd_enabled:
            return
        if self.pd_cpu_req is None and self.pd_batch is None:
            return
        rb = self.running_batch
        if rb is None or rb.is_empty():
            return
        decode_reqs = [
            r
            for r in rb.reqs
            if len(r.output_ids) > 0 and r.req_pool_idx is not None
        ]
        if not decode_reqs:
            return
        if self.pd_cpu_req is not None:
            self.pd_decode_pending = decode_reqs
            return
        # Channel is decoding: record only the reqs not already in pd_batch.
        existing = {id(r) for r in self.pd_batch.reqs}
        new_reqs = [r for r in decode_reqs if id(r) not in existing]
        if new_reqs:
            self.pd_decode_pending = new_reqs

    def _pd_maybe_start_cpu_prefill(self, batch: Optional[ScheduleBatch]) -> None:
        """Start CPU (AMX) prefill while the GPU channel is busy with a layerwise
        prefill batch. The CPU channel is decode-first, so new claims only fire
        when no decode-mode request needs stepping; a paused CPU prefill request
        is resumed to COMPLETION even when the GPU is idle. Only page-aligned
        chunks are processed; a request whose remaining work is a sub-page tail
        (radix over-match) is handed back to the main scheduler."""
        if not envs.SGLANG_PD_CPU_PREFILL.get():
            return
        if not self.pd_enabled or self.pd_active:
            return
        self._pd_lazy_init()
        self._pd_ensure_attn_backend()
        if self.pd_attn_backend is None:
            # CPU-AMX prefill requires its own attention backend to isolate the
            # DSV4 compression plan from the main thread's prefill.
            return

        req = self.pd_cpu_req
        if req is not None:
            # Resume a paused CPU-channel req. It runs to COMPLETION even if
            # decode reqs appeared meanwhile (decode queues behind the in-flight
            # CPU prefill). Drop it only if it was aborted/finished while paused
            # (e.g. client disconnect), so the channel is not stuck forever. A
            # fresh claim with req_pool_idx still None is NOT invalid (the slot
            # is allocated on first prepare).
            if req.finished():
                logger.info(
                    "[PD] CPU-channel req %s became invalid while paused; "
                    "releasing channel.",
                    req.rid,
                )
                self.pd_cpu_req = None
                self.pd_extra_reqs.discard(req)
                req = None
            else:
                if not self._pd_prepare_cpu_prefill_chunk(req):
                    # Sub-page tail OR KV/SWA OOM: hand the req back to the
                    # main scheduler instead of wedging it on the channel (a
                    # paused req would time out on the client). The main thread
                    # retries with its own OOM backoff if the pool is tight.
                    self._pd_cpu_release_pending = False
                    self._pd_release_cpu_req(req)
                    return
                self._pd_launch_channel_thread()
                return

        # New claim: requires a GPU-layerwise prefill batch on the main thread
        # (the GPU channel is occupied) and no waiting decode work.
        if batch is None or not batch.forward_mode.is_extend():
            return
        if self._pd_gpu_threshold <= 0:
            # No GPU-layerwise channel: every prefill already uses the hybrid
            # (CPU-heavy) path; a dedicated CPU channel adds nothing.
            return
        if batch.extend_num_tokens < self._pd_gpu_threshold:
            # The main batch is NOT GPU-layerwise (GPU not occupied); the CPU
            # channel would race the main thread for the CPU AMX without
            # offloading any GPU work.
            return
        # NOTE: deliberately NOT gated on the GPU batch being "full"
        # (extend >= max_prefill_tokens). The CPU channel works on a waiting
        # request that the GPU channel did not take into THIS batch. Whether the
        # GPU batch is at token capacity is the GPU's concern; the CPU AMX is an
        # independent resource. Once the GPU has its batch (layerwise, above the
        # threshold) and a request is queued outside it, the CPU channel claims
        # it and the two channels prefill concurrently. The CPU chunk
        # (SGLANG_PD_CPU_PREFILL_CHUNK=2048) is deliberately SMALLER than the
        # GPU chunk (--chunked-prefill-size) because the CPU AMX path is slower.
        rb = self.running_batch
        if rb is not None and not rb.is_empty() and any(
            r.req_pool_idx is not None and len(r.output_ids) > 0 for r in rb.reqs
        ):
            # decode has priority on the CPU channel.
            return
        # KV/SWA headroom check: the CPU channel's req is NOT part of the main
        # thread's admission budget. Only claim when the pool can spare at
        # least one page; the chunk itself is adapted to the available KV in
        # _pd_prepare_cpu_prefill_chunk (a tight pool yields a smaller chunk,
        # which still makes forward progress instead of OOM-pausing the channel).
        alloc = self.token_to_kv_pool_allocator
        swa_avail = getattr(alloc, "swa_available_size", None)
        if swa_avail is not None and swa_avail() < self.page_size:
            return
        if alloc.available_size() < self.page_size:
            return
        # Grab the frontmost waiting request that did not join the GPU batch.
        in_batch = {id(r) for r in batch.reqs}
        req = None
        for r in self.waiting_queue:
            if id(r) in in_batch:
                continue
            req = r
            break
        if req is None:
            return
        # SWA/KV-capacity-aware claim: the CPU channel's request ALSO consumes
        # the shared SWA pool (its attention/KV run on the GPU). If the GPU
        # side (running + current batch) already reserves most of the SWA pool,
        # claiming a large request here would push the total over capacity and
        # stall the GPU prefill (e.g. 2x40K exceeding the ~63K SWA pool). Leave
        # the request queued for the GPU to process serially instead.
        try:
            cpu_need = len(req.origin_input_ids) or len(req.full_untruncated_fill_ids)
        except Exception:
            cpu_need = 0
        if cpu_need > 0:
            try:
                # Flush deferred frees first (see the prefill-admission note in
                # scheduler.py): available_size() ignores pending_free_pages,
                # so a just-finished request's KV would make the pool look full
                # and block this claim forever.
                self.token_to_kv_pool_allocator.flush_pending_free()
                swa_free = self.token_to_kv_pool_allocator.available_size()
            except Exception:
                swa_free = None
            if swa_free is not None and cpu_need > swa_free:
                return
        # Prepare BEFORE removing from the queue: if the request has no
        # page-aligned chunk of work (radix over-match), it stays queued for
        # the main thread.
        self._pd_cpu_release_pending = False
        if not self._pd_prepare_cpu_prefill_chunk(req):
            if self._pd_cpu_release_pending:
                # Sub-page tail; leave the req in the waiting queue.
                self._pd_cpu_release_pending = False
            return
        self.waiting_queue.remove(req)
        self.pd_cpu_req = req
        self.pd_extra_reqs.add(req)
        logger.info(
            "[PD] CPU-channel prefill claims req %s (queue depth %d).",
            req.rid,
            len(self.waiting_queue),
        )
        self._pd_launch_channel_thread()

    def _pd_release_cpu_req(self, req: Req) -> None:
        """Hand a CPU-channel req back to the main scheduler (front of the
        waiting queue) when it has no page-aligned chunk of work left. The req
        keeps its req_pool_idx / committed KV; the PrefillAdder reuses the slot
        and the radix cache supplies the already-prefilled prefix."""
        self.pd_cpu_req = None
        self.pd_extra_reqs.discard(req)
        self._pd_cpu_need_cache = False
        self._pd_cpu_locked_node = None
        req.extend_range = None
        if not req.finished():
            self.waiting_queue.insert(0, req)
            logger.info(
                "[PD] CPU channel released req %s (tail) back to the queue.",
                req.rid,
            )

    def _pd_maybe_handoff_cpu_prefill(self) -> None:
        """Main-thread half of the decode-priority handoff. The channel thread
        switched to decode (self.pd_batch) and set ``_pd_cpu_handoff`` at a CPU
        prefill chunk boundary; this releases the in-progress CPU req back to
        the GPU channel (front of the waiting queue, radix-safe cache of the
        completed chunk on the main thread). Called every event-loop iteration
        BEFORE get_next_batch_to_run so the released req is picked up as a GPU
        prefill."""
        if not self.pd_enabled or not self._pd_cpu_handoff:
            return
        self._pd_cpu_handoff = False
        req = self.pd_cpu_req
        if req is not None and not req.finished():
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)
        if req is not None:
            self._pd_release_cpu_req(req)
            logger.info(
                "[PD] handed mid-prefill req %s back to the GPU channel.",
                req.rid,
            )

    def _pd_maybe_stop(self) -> None:
        """Stop the channel thread once the prefill is done (the main thread is
        about to resume decode / idle)."""
        if self.pd_active:
            if self.pd_batch is not None or self.pd_cpu_req is not None:
                # Channel still has work (decode or paused prefill); leave it.
                self.last_batch = None
                return
            self._pd_stop()
            return
        # The channel thread already self-deactivated (its work drained) between
        # the caller's pd_active check and here. The caller popped the prefill
        # result out of result_queue; clear last_batch so the next iteration
        # does not pop from an empty queue (last_batch still points at the
        # prefill batch whose result was just consumed).
        self.last_batch = None

    # ------------------------------------------------------------------
    # channel thread lifecycle
    # ------------------------------------------------------------------
    def _pd_launch_channel_thread(self) -> None:
        thread = self.pd_thread
        if thread is not None and thread.is_alive():
            # A self-drained channel thread may still be inside its finally
            # block; join it so the new session's batch is not clobbered. This
            # only runs when pd_active is False (the old thread is exiting, not
            # mid-forward), so the join is fast.
            thread.join()
            self.pd_thread = None
        self._pd_lazy_init()
        self.pd_stop_event.clear()
        self._pd_gen += 1
        my_gen = self._pd_gen
        self.pd_thread = threading.Thread(
            target=self._pd_second_channel_loop,
            name="pd-channel",
            kwargs={"gen": my_gen},
            daemon=True,
        )
        self.pd_thread.start()
        self.pd_active = True

    def _pd_start(self, decode_reqs: List[Req]) -> None:
        self._pd_lazy_init()
        # Ensure a self-drained (still-exiting) channel thread is fully joined
        # before we touch pd_batch, so its finally cannot race this new batch.
        if self.pd_thread is not None and self.pd_thread.is_alive():
            self.pd_thread.join()
            self.pd_thread = None
        # The decode thread shares the SAME req objects with the main thread's
        # running_batch (advanced in place); the main thread must not STEP them
        # while PD is active (_pd_prepare_block_next_decode blocks decode
        # selection). Keeping them in running_batch lets get_next_batch_to_run
        # merge prefill chunks correctly; _pd_stop rebuilds the batch tensors
        # from the (now-advanced) reqs for a clean resume.
        self.pd_batch = self._pd_build_decode_batch(decode_reqs)
        self.pd_decode_pending = None
        logger.info(
            "[PD] decode thread started: %d decode reqs while prefill runs.",
            len(decode_reqs),
        )
        self._pd_launch_channel_thread()

    def _pd_build_decode_batch(self, reqs: List[Req]) -> ScheduleBatch:
        """Build a decode-mode ScheduleBatch for the channel thread, mirroring
        _build_hisparse_decode_batch / _pd_start's preparation."""
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        self._pd_lazy_init()
        batch = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=None,
            dllm_config=self.dllm_config,
        )
        if not reqs:
            return batch
        device = self.device
        req_pool_indices = [r.req_pool_idx for r in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(
            req_pool_indices, dtype=torch.int64
        )
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=device
        )
        batch.seq_lens_sum = sum(seq_lens)
        # Seed the first-step spec relay so resolve_forward_inputs' gather has
        # a valid future_indices + bonus_tokens (the last generated token per
        # req) before the first PD decode forward runs.
        if batch.spec_info is None:
            from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

            batch.spec_info = DFlashDraftInputV2.create_idle_input(device)
        batch.spec_info.future_indices = batch.req_pool_indices
        last_tokens = torch.tensor(
            [r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1] for r in reqs],
            dtype=torch.int64,
            device=device,
        )
        self.pd_future_map.stash(
            batch.req_pool_indices, RelayPayload(bonus_tokens=last_tokens)
        )
        batch.input_ids = None
        if batch.sampling_info is None:
            from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

            batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
                batch, self.model_config.vocab_size
            )
        return batch

    def _pd_stop(self) -> None:
        self.pd_stop_event.set()
        self._pd_cpu_handoff = False
        thread = self.pd_thread
        if thread is not None:
            thread.join()
            self.pd_thread = None
        self.pd_active = False
        # Reqs that came through the CPU channel are not in running_batch; fold
        # them back so none is orphaned when the channel is torn down. A paused
        # mid-prefill req (pd_cpu_req) stays on the channel (resumed on the next
        # GPU prefill batch); a fully-prefilled req now decodes on the main
        # thread.
        if self.pd_extra_reqs:
            rb = self.running_batch
            pending = [
                r
                for r in self.pd_extra_reqs
                if r is not self.pd_cpu_req and not r.finished()
            ]
            if pending:
                if rb is None or rb.is_empty():
                    rb = self._pd_build_decode_batch(pending)
                else:
                    for req in pending:
                        if req not in rb.reqs:
                            rb.reqs.append(req)
                self.running_batch = rb
            self.pd_extra_reqs = set()
        # The decode thread advanced the shared req objects in place; the main
        # thread's running_batch lists the same reqs but with STALE tensors
        # (seq_lens / req_pool_indices built before PD). Rebuild the batch
        # tensors from the reqs so decode resumes cleanly.
        rb = self.running_batch
        if rb is not None and not rb.is_empty():
            self.running_batch = self._pd_rebuild_running_batch(rb)
        # The rebuild already folded the just-finished prefill req into
        # running_batch; the next get_next_batch_to_run would otherwise re-merge
        # last_batch (the prefill batch) and duplicate those reqs.
        self.last_batch = None
        self.pd_batch = None
        self.pd_cpu_batch = None
        logger.info("[PD] channel thread stopped; resuming main-thread work.")

    def _pd_rebuild_running_batch(self, rb: ScheduleBatch) -> ScheduleBatch:
        """Rebuild a ScheduleBatch for the (advanced) running reqs, mirroring
        _pd_start's decode-batch preparation."""
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        reqs = [r for r in rb.reqs if not r.finished()]
        new_batch = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=None,
            dllm_config=self.dllm_config,
        )
        if not reqs:
            return new_batch
        device = self.device
        req_pool_indices = [r.req_pool_idx for r in reqs]
        new_batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        new_batch.req_pool_indices_cpu = torch.tensor(
            req_pool_indices, dtype=torch.int64
        )
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        new_batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        new_batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        new_batch.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=device
        )
        new_batch.seq_lens_sum = sum(seq_lens)
        if new_batch.spec_info is None:
            from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

            new_batch.spec_info = DFlashDraftInputV2.create_idle_input(self.device)
        new_batch.spec_info.future_indices = new_batch.req_pool_indices
        # Reqs that just finished prefill may not have a decoded token yet;
        # fall back to the last input token for the first-step bonus token.
        last_tokens = torch.tensor(
            [
                r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1]
                for r in reqs
            ],
            dtype=torch.int64,
            device=device,
        )
        self.future_map.stash(
            new_batch.req_pool_indices, RelayPayload(bonus_tokens=last_tokens)
        )
        new_batch.input_ids = None
        if new_batch.sampling_info is None:
            from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

            new_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
                new_batch, self.model_config.vocab_size
            )
        return new_batch

    # ------------------------------------------------------------------
    # CPU-channel prefill
    # ------------------------------------------------------------------
    def _pd_prepare_cpu_prefill_chunk(self, req: Req) -> bool:
        """Prepare one CPU-AMX prefill chunk for ``req`` on the main thread (all
        tree_cache / allocator scheduler-side ops stay single-threaded). Returns
        True if a batch was prepared. Returns False with ``_pd_cpu_release_pending``
        set when the request has no page-aligned chunk of work left (e.g. a radix
        over-match left a sub-page tail) -- the channel must hand it back to the
        main scheduler instead of creating a sub-page prefill batch (the DSV4
        compression metadata requires page-aligned batches)."""
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        self._pd_lazy_init()
        # NOTE: a freshly-claimed waiting req has req_pool_idx == None (the slot
        # is allocated by prepare_for_extend -> alloc_req_slots below). Only
        # reject a req that is already finished.
        if req.finished():
            return False
        # Deferred cache from the channel thread's previous non-final chunk.
        # Runs here on the MAIN thread (before the re-match below) so all radix
        # tree mutations stay single-threaded.
        if self._pd_cpu_need_cache:
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)
            self._pd_cpu_need_cache = False
        req.init_next_round_input(self.tree_cache)
        # NOTE: no inc_lock_ref on the matched node. Locking the CPU-channel
        # req's prefix makes its SWA non-evictable, starving the GPU prefill's
        # admission (the pool's truly-free SWA collapses to < a chunk and the
        # GPU prefill OOMs/rolls back in a busy loop). If the main thread
        # evicts the CPU req's cached prefix while it is paused, the next chunk
        # simply re-prefills from the radix miss (correct, just extra work).
        remaining = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
        if remaining <= 0:
            # Already fully prefilled (e.g. resumed after a final chunk whose
            # result had not been processed yet).
            return False
        chunk = min(remaining, self._pd_cpu_chunk_tokens)
        alloc = self.token_to_kv_pool_allocator
        # SWA window eviction for the CPU-channel request: the chunks this
        # channel prefilled accumulate in the shared SWA pool. Without
        # eviction the pool fills (GPU prefill + CPU prefill), the chunk clamp
        # below collapses to a page, and the channel releases the request as a
        # "tail" after a few chunks -- the CPU prefill contributes almost
        # nothing and the mechanism's overhead makes prefill slower. Evict this
        # request's out-of-window tokens (leave the most recent chunk + window)
        # and flush so the clamp sees the reclaimed pages. The previous chunk's
        # forward has already completed (its result was processed on the channel
        # thread before this prepare runs), and the main thread's GPU prefill
        # reads only its own tokens.
        try:
            if req.kv_committed_len > 0:
                pre_len = req.kv_committed_len - self._pd_cpu_chunk_tokens
                if pre_len > 0:
                    free_swa_out_of_window_slots(
                        req,
                        pre_len,
                        sliding_window_size=self.tree_cache.sliding_window_size,
                        page_size=self.tree_cache.page_size,
                        req_to_token_pool=self.req_to_token_pool,
                        token_to_kv_pool_allocator=alloc,
                        is_chunk_cache=self.tree_cache.is_chunk_cache(),
                    )
        except Exception:
            pass
        try:
            alloc.flush_pending_free()
        except Exception:
            pass
        # Adapt to the momentary KV/SWA availability: the GPU prefill consumes
        # the pool while the CPU channel runs, so a fixed 2K chunk can exceed
        # the free KV and OOM-pause the channel. Clamp to what the pool can
        # spare (leaving a one-page margin) so the channel still makes progress
        # with a smaller chunk under memory pressure.
        swa_avail = getattr(alloc, "swa_available_size", None)
        avail = swa_avail() if swa_avail is not None else alloc.available_size()
        chunk = min(chunk, max(self.page_size, (avail - self.page_size) // self.page_size * self.page_size))
        if self._pd_gpu_threshold > 0 and chunk >= self._pd_gpu_threshold:
            # Keep the chunk on the CPU path (below the GPU-layerwise threshold).
            chunk = (
                max(0, self._pd_gpu_threshold - 1) // self.page_size
            ) * self.page_size
        chunk = chunk // self.page_size * self.page_size
        if chunk < self.page_size:
            # Sub-page tail (e.g. the radix cache already matched almost the
            # whole sequence). The DSV4 compression metadata build requires a
            # page-aligned batch; hand the tail back to the main scheduler.
            self._pd_cpu_release_pending = True
            return False
        start = len(req.prefix_indices)
        req.set_extend_range(start, start + chunk)
        batch = ScheduleBatch.init_new(
            [req],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=None,
            dllm_config=self.dllm_config,
        )
        try:
            # Record the KV allocation / cache-index writes on the channel's
            # own stream so the PD-thread forward on pd_stream sees them (no
            # cross-stream hazard with the main thread's GPU prefill).
            with torch.cuda.stream(self.pd_stream):
                batch.prepare_for_extend()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.warning(
                "[PD] CPU prefill chunk OOM (req %s, %d tokens); pausing channel.",
                req.rid,
                chunk,
            )
            req.extend_range = None
            return False
        batch.contains_last_prefill_chunk = False
        # The main-thread path fills prefill_stats via PrefillStats.from_adder;
        # build a minimal one here so process_batch_result_prefill -> metrics
        # works and the CPU-channel prefill shows up in the batch log.
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )
        from sglang.srt.observability.metrics_collector import QueueCount

        batch.prefill_stats = PrefillStats(
            log_input_tokens=chunk,
            log_hit_tokens=0,
            new_token_ratio=1.0,
            num_running_reqs=QueueCount(total=0),
            num_new_seqs=1,
            num_pending_tokens=0,
        )
        self.pd_cpu_batch = batch
        return True

    def _pd_run_cpu_prefill_step(self, batch: ScheduleBatch):
        """One CPU-AMX prefill chunk on the PD stream. Mirrors run_batch's
        extend path but uses the per-thread stream + future_map."""
        batch.launch_ts = time.monotonic()
        self.pd_future_map.resolve_seq_lens_cpu(batch)
        resolve_forward_inputs(batch, self.pd_future_map)
        with self._forward_isolation(batch, overlap=False):
            future_indices = batch.req_pool_indices
            fwd_kwargs = {}
            if not batch.spec_algorithm.is_none():
                fwd_kwargs["on_publish"] = partial(
                    self.pd_future_map.publish, future_indices
                )
            # The CPU prefill forward shares the main attention backend (like
            # PD decode). A per-thread backend is only used if explicitly
            # created (pd_attn_backend != None) for a prefill+prefill assertion.
            if self.pd_attn_backend is not None:
                ctx_mgr = forward_context(
                    ForwardContext(attn_backend=self.pd_attn_backend)
                )
            else:
                ctx_mgr = contextlib.nullcontext()
            with ctx_mgr:
                batch_result = self.model_worker.forward_batch_generation(
                    batch, **fwd_kwargs
                )
            if batch.spec_algorithm.is_none():
                self.pd_future_map.publish(future_indices, batch.seq_lens + 1)
            self._pd_relay_forward_payload(future_indices, batch_result)
        batch.input_ids = None
        batch_result.copy_done = self.device_module.Event()
        batch_result.copy_to_cpu(
            return_logprob=batch.return_logprob,
            return_hidden_states=batch.return_hidden_states,
        )
        return batch_result

    @staticmethod
    def _pd_cpu_req_prefill_done(req: "Req") -> bool:
        return (
            req.extend_range is not None
            and req.extend_range.end >= len(req.full_untruncated_fill_ids)
        )

    # ------------------------------------------------------------------
    # background channel thread
    # ------------------------------------------------------------------
    def _pd_second_channel_loop(self, gen: int) -> None:
        # Use the PD thread's disjoint staging-buffer slot (KT SharedStagingBuffer
        # get_slice is slot-based; see kt_ep_wrapper.py).
        from sglang.srt.layers.moe.kt_ep_wrapper import _STAGING_THREAD_SLOT

        _STAGING_THREAD_SLOT.slot = 1
        try:
            while not self.pd_stop_event.is_set():
                # Priority 1: decode-mode requests (the CPU channel's dominant
                # job). Decode must not be blocked by CPU prefill.
                if self.pd_batch is not None and not self.pd_batch.is_empty():
                    batch = self.pd_batch
                    # Fold in decode-mode reqs whose prefill completed while the
                    # channel was decoding (recorded by _pd_note_decode_pending),
                    # so the concurrent requests decode TOGETHER (the AMX MoE
                    # amortizes across the batch) instead of serially.
                    if self.pd_decode_pending:
                        pending = self.pd_decode_pending
                        self.pd_decode_pending = None
                        existing = {id(r) for r in batch.reqs}
                        new_reqs = [
                            r for r in pending if id(r) not in existing
                        ]
                        if new_reqs:
                            merged = list(batch.reqs) + new_reqs
                            self.pd_batch = self._pd_build_decode_batch(merged)
                            batch = self.pd_batch
                            logger.info(
                                "[PD] folded %d new decode reqs into the "
                                "decode batch (%d total).",
                                len(new_reqs),
                                len(merged),
                            )
                    with torch.cuda.stream(self.pd_stream):
                        batch = self.update_running_batch(batch)
                    if batch is None or batch.is_empty():
                        self.pd_batch = None
                        continue
                    self.forward_ct += 1
                    batch.forward_iter = self.forward_ct
                    batch.launch_ts = time.monotonic()
                    with torch.cuda.stream(self.pd_stream):
                        result = self._pd_run_decode_step(batch)
                    # Stream output + finish handling on the channel thread. The
                    # output streamer and the KV allocator are internally locked.
                    self.process_batch_result(batch, result)
                    if batch.batch_size() == 0:
                        self.pd_batch = None
                    continue

                # Priority 2: CPU (AMX) prefill.
                if self.pd_cpu_batch is not None:
                    batch = self.pd_cpu_batch
                    req = self.pd_cpu_req
                    self.pd_cpu_batch = None
                    if batch is None or batch.is_empty() or req is None:
                        continue
                    if not self._pd_cpu_req_prefill_done(req):
                        # Mid-chunk: mark it so process_batch_result_prefill
                        # takes the chunked branch (no streaming, no output
                        # token), then cache the produced KV so the next chunk
                        # resumes from the advanced prefix.
                        req.inflight_middle_chunks = 1
                    self.forward_ct += 1
                    batch.forward_iter = self.forward_ct
                    batch.launch_ts = time.monotonic()
                    with torch.cuda.stream(self.pd_stream):
                        result = self._pd_run_cpu_prefill_step(batch)
                    self.process_batch_result(batch, result)
                    if not self._pd_cpu_req_prefill_done(req):
                        # Decode has priority: if decode reqs arrived while this
                        # chunk was running, switch to decode IMMEDIATELY and ask
                        # the main thread (_pd_maybe_handoff_cpu_prefill) to hand
                        # the in-progress CPU prefill back to the GPU channel.
                        if self.pd_decode_pending:
                            self._pd_cpu_need_cache = False
                            pending = self.pd_decode_pending
                            self.pd_decode_pending = None
                            self._pd_cpu_handoff = True
                            self.pd_batch = self._pd_build_decode_batch(pending)
                            continue
                        # Non-final chunk. The produced KV is NOT cached on this
                        # thread: cache_unfinished_req mutates the shared radix
                        # tree, which the main thread also mutates (match/evict/
                        # cache_finished) without a lock. Defer the cache to the
                        # main thread (_pd_prepare_cpu_prefill_chunk) so all tree
                        # mutations stay single-threaded. extend_range is KEPT so
                        # the deferred cache's get_fill_ids() covers the
                        # completed chunk; init_next_round_input in the next
                        # prepare overwrites it.
                        self._pd_cpu_need_cache = True
                        # CPU prefill runs to COMPLETION; decode reqs recorded
                        # in pd_decode_pending are folded in after the request
                        # finishes (no mid-prefill preemption). The main thread
                        # prepares the next chunk.
                        break
                    # Prefill finished: transition to decode, folding any pending
                    # decode reqs in FIRST (they finished prefill earlier and
                    # must not be re-blocked behind this request).
                    self.pd_cpu_req = None
                    pending = self.pd_decode_pending
                    self.pd_decode_pending = None
                    decode_reqs = (list(pending) if pending else []) + [req]
                    self.pd_batch = self._pd_build_decode_batch(decode_reqs)
                    logger.info(
                        "[PD] CPU prefill done for req %s; transitioning to "
                        "decode (%d reqs).",
                        req.rid,
                        len(decode_reqs),
                    )
                    continue
                break
        except Exception:
            logger.error("[PD] channel thread crashed.", exc_info=True)
            # Fail every request this channel owned so the clients get a
            # definitive error instead of hanging forever (the request would
            # otherwise never be returned: it is not in the main scheduler's
            # running batch, and the channel thread just died).
            from http import HTTPStatus

            for _b in (self.pd_batch, self.pd_cpu_batch):
                if _b is None:
                    continue
                for _r in getattr(_b, "reqs", []):
                    if not _r.finished():
                        _r.set_finish_with_abort(
                            error_msg=(
                                "The request failed on the inference channel "
                                "(PD worker error). Please retry."
                            ),
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            err_type="InternalServerError",
                        )
                        try:
                            self.output_streamer.stream_output([_r], False)
                        except Exception:
                            logger.error(
                                "[PD] failed to stream error for req %s.",
                                _r.rid,
                                exc_info=True,
                            )
            self.pd_batch = None
            self.pd_cpu_batch = None
            self.pd_cpu_req = None
        finally:
            # PD owns decode until its reqs DRAIN (or it is stopped); only then
            # hand decode back to the main thread. This avoids the premature
            # stop/restart churn between prefill chunks. The generation guard
            # keeps a stale (self-drained) thread from clobbering state that a
            # new channel session has already started.
            if self._pd_gen == gen:
                self.pd_active = False
                self.pd_batch = None
                self.pd_cpu_batch = None
                self._pd_drained = True

    def _pd_run_decode_step(self, batch: ScheduleBatch):
        """One decode step on the PD stream: resolve inputs, forward, relay to
        the PD future_map, copy result to CPU. Mirrors run_batch's overlap
        spec-v2 path but uses the per-thread stream + future_map."""
        batch.launch_ts = time.monotonic()
        # SWA window eviction on the PD decode thread: the main-thread decode
        # path evicts via prepare_for_decode, but the channel thread skips it,
        # so a decoding request kept its FULL context in the SWA pool and
        # starved the GPU prefill channel (swa 1.00, prefill throttled to tiny
        # chunks, concurrent large requests appear hung).
        try:
            for _r in batch.reqs:
                if _r.kv is not None:
                    _r.decode_batch_idx += 1
                    # Skip the first PD step: the previous forward may still be
                    # reading the swa range. From the 2nd step, tokens >= one
                    # page+window below the current seqlen are dead.
                    if _r.decode_batch_idx >= 2 and _r.seqlen > 0:
                        free_swa_out_of_window_slots(
                            _r,
                            _r.seqlen - 1,
                            sliding_window_size=self.tree_cache.sliding_window_size,
                            page_size=self.tree_cache.page_size,
                            req_to_token_pool=self.req_to_token_pool,
                            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                            is_chunk_cache=self.tree_cache.is_chunk_cache(),
                        )
        except Exception:
            pass
        self.pd_future_map.resolve_seq_lens_cpu(batch)
        # DSPARK schedules its per-step verify budget from the previous step's
        # draft confidence (the scheduler's main-thread prepare_verify_budget
        # hook only runs for batches the main thread forwards). Run it here on
        # the PD thread so decode batches get a real budget instead of None
        # (None degenerates the schedule and collapses the accept rate).
        if self._confidence_budget_prepare is not None:
            self._confidence_budget_prepare(batch, self.pd_future_map)
        resolve_forward_inputs(batch, self.pd_future_map)
        with self._forward_isolation(batch, overlap=False):
            future_indices = batch.req_pool_indices
            fwd_kwargs = {}
            if not batch.spec_algorithm.is_none():
                fwd_kwargs["on_publish"] = partial(
                    self.pd_future_map.publish, future_indices
                )
            batch_result = self.model_worker.forward_batch_generation(
                batch, **fwd_kwargs
            )
            if batch.spec_algorithm.is_none():
                self.pd_future_map.publish(future_indices, batch.seq_lens + 1)
            self._pd_relay_forward_payload(future_indices, batch_result)
            if batch_result.extra_keep_alive_refs:
                # Keep cross-step tensors alive on the PD thread.
                self._pd_keep_alive = getattr(self, "_pd_keep_alive", []) + [
                    batch_result.extra_keep_alive_refs
                ]
        batch.input_ids = None
        if not batch.spec_algorithm.is_none():
            batch.spec_info = batch_result.next_draft_input
            if batch.spec_info is not None:
                batch.spec_info.future_dsa_topk_indices_available = (
                    getattr(batch.spec_info, "dsa_topk_indices", None) is not None
                )
                batch.spec_info.future_indices = future_indices
        if batch_result.new_seq_lens is not None:
            batch.seq_lens = batch_result.new_seq_lens
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        # Result D2H on the PD stream (ordered after the forward).
        batch_result.copy_done = self.device_module.Event()
        batch_result.copy_to_cpu(
            return_logprob=batch.return_logprob,
            return_hidden_states=batch.return_hidden_states,
        )
        return batch_result

    def _pd_relay_forward_payload(
        self, future_indices: torch.Tensor, batch_result
    ) -> None:
        """Stash this step's relay payload for the next step's resolve, on the
        PD thread's own future_map (mirrors scheduler._relay_forward_payload)."""
        if self.spec_algorithm.is_ngram():
            return
        if batch_result.next_draft_input is not None:
            payload = RelayPayload.from_draft_input(batch_result.next_draft_input)
        elif batch_result.has_sampled_token_ids:
            payload = RelayPayload(bonus_tokens=batch_result.next_token_ids)
        else:
            return
        self.pd_future_map.stash(future_indices, payload)

    def _pd_prepare_block_next_decode(self) -> None:
        """Hook: while PD is active OR a CPU-channel prefill request is claimed
        (paused between chunks), get_next_batch_to_run must NOT prepare the
        running batch for decode (the channel owns that work). Returns True to
        block the main-thread decode selection."""
        return self.pd_active or self.pd_cpu_req is not None
