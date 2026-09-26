from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )


class CandidateMetadata:
    """Base of an implementation's published state on
    ``DSV4Metadata.candidate_metadata``."""


@dataclass(frozen=True)
class IndexerInputs:
    """One index-source layer's operands on the paged fp4 decode path (one query
    row per request, or per draft token under verify)."""

    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    metadata: PagedIndexerMetadata  # this ratio's lengths, page table and plans
    # [rows] int, one request id per query row, the rows of one request
    # consecutive (verify: its draft tokens); None = every row its own request
    request_ids: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]


def make_candidate_indexer(
    topk_blocks: int, block_size: int
) -> Optional[DeepGemmCandidateIndexer]:
    """The paged fp4 decode path's two-level indexer; None on Hopper, whose decode
    indexer selects through masks inline."""
    if topk_blocks <= 0 or get_platform().device_sm < 100:
        return None
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        raise RuntimeError(
            "the candidate indexer needs DeepGEMM's paged sparse MQA logits "
            "(sgl-deep-gemm >= 0.2.0 with SGLANG_ENABLE_JIT_DEEPGEMM on)"
        )
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )

    return DeepGemmCandidateIndexer(topk_blocks, block_size)


# TODO(candidate): the dense-fp4 prefill (sm100) and the Hopper decode still select
# through these masks inline in the backend; move them behind the protocol too.
@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    # prefill, the dense-fp4 path (sm100): [rows_b, lc_b] bool each.
    request_masks: Optional[List[torch.Tensor]] = None
    # prefill, the torch path: the same selection as `request_masks` but as block ids --
    # `select_candidate_block_ids`'s `blocks`, [rows_b, min(topk_blocks, num_blocks)] int64
    # ascending with the pad `num_blocks` LAST.  The candidate WIDTH is the constant
    # topk_blocks x block_size while the mask spans `lc`, so the ids are O(chunk x topk_blocks)
    # and the mask is O(chunk x lc): at chunk 16384 with lc 81920 that is 268 MiB against
    # 1280 MiB, and the mask's own `torch.cat` is the 1.25 GiB allocation that OOMs a
    # >32768-token prompt at the shipping chunk (task #44).  A consumer rebuilds the mask for
    # one row slice with `expand_candidate_block_ids`.
    request_blocks: Optional[List[torch.Tensor]] = None
    # decode, sm<100 only: the same selection as `mask` but as block ids, so a
    # consumer can gather just the candidate columns instead of scoring the full
    # width and masking after (`SGLANG_OPT_SM89_CANDIDATE_PRUNE_SCORE`).
    # `blocks` is [rows, min(topk_blocks, num_blocks)] int64 ascending with the
    # padding LAST, and `block_lens` is [rows] int64, the number of VALID candidate
    # columns per row.  Within a row, positions increase across the flattened
    # block-by-block layout, so the first `block_lens[b]` columns of that layout are
    # exactly the valid ones -- a prefix, which is what the logits kernel's `lens`
    # argument needs.
    blocks: Optional[torch.Tensor] = None
    block_lens: Optional[torch.Tensor] = None


def published_masks(candidate) -> CandidateMasks:
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: a bool mask over positions keeping the
    topk_blocks best-scoring blocks per query. Unreachable positions are already -inf
    in logits, so an all -inf block means not reachable yet; the block holding the
    query's newest position is always kept."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> tuple:
    """`select_candidate_blocks`'s selection published as block ids rather than a mask,
    so a consumer can gather the candidate columns and score only those. Returns
    `(blocks, block_lens)`: `blocks` is ``[rows, min(topk_blocks, num_blocks)]`` int64
    ASCENDING with the padding at the END (a pad is the value ``num_blocks``, which is also
    its validity flag), and `block_lens` is ``[rows]`` int64, the number of valid candidate
    columns. Within a row the first `block_lens[b]` columns of the flattened block-by-block
    layout are exactly the valid ones -- a prefix, because positions increase inside a block
    and the blocks ascend -- which is the shape the logits kernel's `lens` argument needs.

    The width is `min(topk_blocks, num_blocks)` and NOT `topk_blocks`: padding a short
    sequence's block list out to `topk_blocks` would make the compacted layout WIDER than
    `lmax` itself (16384 columns for a 36-column row) and turn the prune into a large
    pessimisation exactly where the full-width path was already cheap."""
    lens = compress_lens.reshape(-1)
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last[:, None], torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    pad = num_blocks
    blocks = top.indices.masked_fill(top.values == -torch.inf, pad).sort(dim=-1).values

    offsets = torch.arange(block_size, device=logits.device)
    positions = blocks.clamp_max(num_blocks - 1)[:, :, None] * block_size + offsets
    valid = (blocks[:, :, None] < num_blocks) & (positions < lens[:, None, None])
    return blocks, valid.sum(dim=(1, 2))


def expand_candidate_block_ids(
    blocks: torch.Tensor, block_size: int, width: int
) -> torch.Tensor:
    """Rebuild `select_candidate_blocks`'s `[rows, width]` mask from `select_candidate_block_ids`'s
    `blocks`, for ONE row slice: expanding only the slice is what keeps the published state
    O(rows x topk_blocks) instead of O(rows x width).

    The `num_blocks + 1` columns exist so the pads -- whose value is exactly `num_blocks` -- land
    one past the real blocks and write False there.  A pad can therefore never overwrite a
    selected block, and duplicates only ever occur among the pads, which agree.  The selection is
    `top.values > -inf` in `select_candidate_blocks`, i.e. `blocks < num_blocks` here, so the
    result is bitwise the mask `select_candidate_blocks` returns."""
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        blocks.shape[0], num_blocks + 1, dtype=torch.bool, device=blocks.device
    )
    keep.scatter_(1, blocks, blocks < num_blocks)
    return keep[:, :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]
