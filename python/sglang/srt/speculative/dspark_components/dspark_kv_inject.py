from contextlib import nullcontext
from threading import RLock
from typing import Optional

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
)
from sglang.srt.layers.quantization.fp8_utils import (
    fp8_gemm_runner_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class TargetHiddenKvInjector:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        model_runner,
        device,
        verify_num_draft_tokens: int,
        block_pos_offsets: torch.Tensor,
        draft_device=None,
        remote_draft: bool = False,
        fp8_gemm_backend: Optional[str] = None,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.model_runner = model_runner
        self.device = device
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self._block_pos_offsets = block_pos_offsets
        self.draft_device = torch.device(draft_device or device)
        self.remote_draft = remote_draft
        self.fp8_gemm_backend = fp8_gemm_backend
        # PD-parallel: prefill (scheduler thread) and decode (background thread)
        # both call inject_target_hidden / inject_ragged, writing the shared
        # draft KV pool concurrently. Serialize the KV write section only.
        self._inject_lock = RLock()

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        req_pool_indices: Optional[torch.Tensor] = None,
        extend_lens: Optional[torch.Tensor] = None,
        uniform_len: Optional[int] = None,
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return
        # PD-parallel: prefill (scheduler thread) and decode (background thread)
        # both write the draft KV pool; serialize the write section.
        with self._inject_lock:
            device = self.draft_device
            context = (
                torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            )
            with context, fp8_gemm_runner_backend_context(self.fp8_gemm_backend):
                cache_loc = cache_loc.to(
                    device=device, dtype=torch.int64, non_blocking=True
                )
                positions = positions.to(
                    device=device, dtype=torch.int64, non_blocking=True
                )
                target_hidden = target_hidden.to(device=device, non_blocking=True)
                n_real = positions.shape[0]
                if target_hidden.shape[0] > n_real:
                    target_hidden = target_hidden[:n_real]
                if cache_loc_2d is not None:
                    cache_loc_2d = cache_loc_2d.to(
                        device=device, dtype=torch.int64, non_blocking=True
                    )
                if commit_lens is not None:
                    commit_lens = commit_lens.to(
                        device=device, dtype=torch.int32, non_blocking=True
                    )
                if req_pool_indices is not None:
                    req_pool_indices = req_pool_indices.to(
                        device=device, dtype=torch.int64, non_blocking=True
                    )
                if extend_lens is not None:
                    extend_lens = torch.as_tensor(
                        extend_lens, dtype=torch.int64, device=device
                    )

                pool = self.draft_model_runner.token_to_kv_pool
                if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
                    self._inject_mla(
                        pool=pool,
                        target_hidden=target_hidden,
                        cache_loc=cache_loc,
                        positions=positions,
                        cache_loc_2d=cache_loc_2d,
                        commit_lens=commit_lens,
                        req_pool_indices=req_pool_indices,
                        extend_lens=extend_lens,
                        uniform_len=uniform_len,
                    )
                    return

                with torch.inference_mode():
                    self.draft_model.write_target_hidden_kv(
                        target_hidden=target_hidden,
                        pool=pool,
                        positions=positions,
                        cache_loc=cache_loc,
                        cache_loc_2d=cache_loc_2d,
                        commit_lens=commit_lens,
                    )

    def _window_inject_mask(
        self,
        *,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        req_pool_indices: Optional[torch.Tensor],
        extend_lens: Optional[torch.Tensor],
        window: int,
    ) -> Optional[torch.Tensor]:
        """Mask of positions inside the draft's sliding window.

        The draft attention is compress_ratio=0 (no c4/c128 global attention):
        it only reads the last `window` positions' SWA KV. The injection
        currently writes the SWA KV for the whole verify batch, so the draft
        SWA pool is sized to the chunk. Restricting the injection to the window
        lets the draft SWA pool shrink to ~window + draft-block per request.
        """
        if cache_loc_2d is not None and commit_lens is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=positions.device).view(1, -1)
            valid = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            pos_2d = positions.view(bs, verify_len)
            last_pos = torch.where(
                valid.view(bs, verify_len),
                pos_2d,
                torch.full_like(pos_2d, -1),
            ).max(dim=1).values
            last_pos = last_pos.clamp(min=0)
            last_pos_exp = last_pos.repeat_interleave(verify_len)
            return valid & (positions >= (last_pos_exp - window))
        if req_pool_indices is not None and extend_lens is not None:
            lens = extend_lens.to(torch.int64)
            offsets = torch.cumsum(lens, 0) - lens
            last_pos = positions[offsets + lens - 1]
            last_pos_exp = last_pos.repeat_interleave(lens)
            return positions >= (last_pos_exp - window)
        return None

    def _inject_mla(
        self,
        *,
        pool,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        req_pool_indices: Optional[torch.Tensor] = None,
        extend_lens: Optional[torch.Tensor] = None,
        uniform_len: Optional[int] = None,
    ) -> None:
        if getattr(pool, "swa_ring_enabled", False):
            # CUDA ring pool: the draft never reads its KV during prefill (its
            # window is only consumed by decode), so only the trailing window
            # per request needs to land in the per-request ring.
            #
            # Decode verify: the draft ring MUST receive target-derived KV for
            # ALL verify positions, not just the accepted prefix. The draft
            # attends over the speculative block positions too (its window
            # reaches prefix+block); if those slots only hold the draft's own
            # store, the draft reads its own previous predictions back (a
            # feedback loop) and the proposals diverge from the target
            # (ring-mode accept ~0 vs flat ~0.29). In flat mode the target's
            # verify store overwrites the shared pool with target KV for every
            # verify position; mirror that here.
            assert req_pool_indices is not None and extend_lens is not None, (
                "ring-mode draft inject requires req_pool_indices + extend_lens"
            )
            n = positions.shape[0]
            window = pool.swa_window_size
            if uniform_len is not None:
                # Uniform block (DSPARK decode): extend_lens == T for every
                # request. Compute the mask entirely on the GPU with an int
                # repeat -- a GPU-repeats repeat_interleave forces a GPU->CPU
                # sync to size the output (drains the queued verify work,
                # ~110ms/step).
                T = int(uniform_len)
                bs = req_pool_indices.shape[0]
                req_slots = req_pool_indices.repeat_interleave(T)
                offs = torch.arange(bs, device=positions.device) * T
                last_pos = positions[offs + (T - 1)]
                chunk_end = last_pos + 1
                token_chunk_end = chunk_end.repeat_interleave(T)
                sel = positions >= (token_chunk_end - window)
            else:
                req_slots = req_pool_indices.repeat_interleave(extend_lens)
                offs = torch.cumsum(extend_lens, 0) - extend_lens
                last_pos = positions[offs + extend_lens - 1]
                chunk_end = last_pos + 1
                token_chunk_end = chunk_end.repeat_interleave(extend_lens)
                sel = positions >= (token_chunk_end - window)
            # NOTE: never mask `sel` by commit_lens here (see comment above).
            # Also avoid any host-side read of a GPU-tensor shape/count below:
            # a `sel_pos.shape[0]` / `.numel()` sync drains the whole queued
            # step (~110ms/step). Empty selections flow through harmlessly.
            sel_pos = positions[sel]
            sel_hidden = target_hidden[sel]
            swa_loc = pool.get_swa_ring_loc(req_slots[sel], sel_pos)
            with torch.inference_mode():
                self.draft_model.write_target_hidden_kv(
                    main_hidden=sel_hidden,
                    swa_loc=swa_loc,
                    positions=sel_pos,
                    pool=pool,
                )
            return

        swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
        mask = None
        if __import__("os").environ.get("SGLANG_DSPARK_DRAFT_WINDOW_INJECT") == "1":
            mask = self._window_inject_mask(
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                req_pool_indices=req_pool_indices,
                extend_lens=extend_lens,
                window=getattr(pool, "swa_window_size", None) or 128,
            )
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
            committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            mask = (
                committed_mask
                if mask is None
                else (mask & committed_mask)
            )
            swa_loc = torch.where(committed_mask, swa_loc, torch.full_like(swa_loc, -1))
        if mask is not None:
            # Sentinel-based selection: keep the full fixed-size arrays and
            # mark non-selected rows with out_loc=-1. The fused write kernel
            # skips out_loc<0 rows, so this is equivalent to the boolean
            # indexing below but avoids the GPU->CPU sync that variable-size
            # indexing forces (~127ms/step on the decode hot path).
            swa_loc = torch.where(mask, swa_loc, torch.full_like(swa_loc, -1))
        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def inject_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        hidden_strided: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> None:
        with self._inject_lock:
            stride = self.verify_num_draft_tokens
            prefix_lens = batch.seq_lens
            hidden = hidden_strided.view(bs, stride, -1)

            pool = self.draft_model_runner.token_to_kv_pool
            if not self.remote_draft and hasattr(
                pool, "set_swa_key_buffer_radix_fused_norm_rope"
            ):
                if hidden_strided.numel() == 0:
                    return
                inject_layout = BuildCommitInjectLayout.execute(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    full_to_swa_mapping=pool.full_to_swa_index_mapping,
                    commit_lens=commit_lens,
                    stride=stride,
                    swa_ring=bool(getattr(pool, "swa_ring_enabled", False)),
                    ring_size=getattr(pool, "swa_ring_size", 0) or 0,
                    ring_base=getattr(pool, "swa_ring_base", 0) or 0,
                )
                with torch.inference_mode():
                    self.draft_model.write_target_hidden_kv(
                        main_hidden=hidden.reshape(-1, hidden.shape[-1]),
                        swa_loc=inject_layout.swa_loc,
                        positions=inject_layout.positions,
                        pool=pool,
                    )
                return

            positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
            verify_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=prefix_lens + stride,
                batch_size=bs,
                draft_token_num=stride,
                device=self.device,
            )
            verify_cache_loc_2d = verify_cache_loc.view(bs, stride)
            self.inject_target_hidden(
                target_hidden=hidden.reshape(-1, hidden.shape[-1]),
                cache_loc=verify_cache_loc,
                cache_loc_2d=verify_cache_loc_2d,
                positions=positions_2d.reshape(-1),
                commit_lens=commit_lens,
                req_pool_indices=batch.req_pool_indices,
                extend_lens=torch.full(
                    (bs,),
                    stride,
                    dtype=torch.int64,
                    device=batch.req_pool_indices.device,
                ),
            )
