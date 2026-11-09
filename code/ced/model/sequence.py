"""Per-position bookkeeping shared by every layer: absolute index, document id, document-relative position.

A packed row holds several documents. Every layer keeps documents isolated:
attention is masked to the same document, GDN state and conv history reset at each document
start, and rotary positions restart at 0. The same rules apply when a row is processed in chunks
(prefill + cached steps), so the bookkeeping carries over between chunks in `SeqState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SeqContext:
    idx: torch.Tensor  # (B, L) absolute index in the row (0, 1, 2, ... across chunks)
    doc: torch.Tensor  # (B, L) document id; real documents use ids >= 0
    pos: torch.Tensor  # (B, L) position inside the document (restarts at 0)
    is_doc_start: torch.Tensor  # (B, L) bool
    # Training only, optional: segment boundaries of the flattened (B * L) sequence for fused
    # varlen kernels (fla GDN): every row start and every document start. Precomputed on the host
    # (ced/train/varlen.py) and equal to what is_doc_start implies.
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None  # the same on the host (lets fla skip a device sync)
    # Per-forward memo for objects built from the metadata alone (flex BlockMasks), shared by all
    # layers that see this context.
    memo: dict = field(default_factory=dict)
    # Token Superposition Training's token-scale rotary positions (TSTConfig.token_scale_rope; ced/model/attention.py):
    # the capped-rotary layers rotate at rope_scale x pos and keep the cap in tokens. 1 (the default): every forward.
    rope_scale: int = 1
    # Token Superposition Training's resolution-explicit GDN gates (TSTConfig.gdn_resolution_gates, E): in a bag phase of
    # size s the GDN layers' kernels get g = s x the per-token log decay and, with gdn_beta_power, beta = 1 - (1 - beta)^s
    # (ced/model/gdn.py GatedDeltaNet.gates). 1 (the default): every forward, the gates as they always were.
    gdn_scale: int = 1
    gdn_beta_power: bool = False
    # CUDA graphs (TrainConfig.cuda_graphs; ced/train/cuda_graphs.py): the GDN kernels' chunk index at a fixed capacity,
    # (chunk_indices, chunk_offsets, nt_max) on the device, for the fixed-capacity cu_seqlens above (ced/train/varlen.py
    # fixed_varlen_host; ced/kernels/gdn.py fixed_chunk_index). None (every forward without CUDA graphs): the kernels build
    # the index from cu_seqlens_cpu, as always.
    gdn_chunks: tuple | None = None


@dataclass
class SeqState:
    """What the next chunk needs to know about the positions already processed."""

    next_idx: torch.Tensor  # (B,)
    last_doc: torch.Tensor  # (B,) document id of the last processed position (-1 if none)
    doc_start_idx: torch.Tensor  # (B,) absolute index where that document started


def sequence_context(doc_ids: torch.Tensor, state: SeqState | None) -> tuple[SeqContext, SeqState]:
    """Compute index, position and document-start flags for a chunk, with fixed-shape ops only.

    Contract: document ids increase along a row (across chunks too) and are never reused in the
    same row. Document starts are found from id changes, while attention masks compare ids for
    equality; the two agree only under this contract.
    """
    B, L = doc_ids.shape
    device = doc_ids.device
    if state is None:
        offset = torch.zeros(B, dtype=torch.long, device=device)
        last_doc = torch.full((B,), -1, dtype=torch.long, device=device)
        carried_start = torch.zeros(B, dtype=torch.long, device=device)
    else:
        offset, last_doc, carried_start = state.next_idx, state.last_doc, state.doc_start_idx

    idx = offset[:, None] + torch.arange(L, device=device)[None, :]
    prev_doc = torch.cat([last_doc[:, None], doc_ids[:, :-1]], dim=1)
    is_doc_start = doc_ids != prev_doc
    # Index of the most recent document start at or before each position. -1 means "the document
    # started in an earlier chunk", in which case the carried start index is used.
    start = torch.where(is_doc_start, idx, torch.full_like(idx, -1)).cummax(dim=1).values
    start = torch.where(start < 0, carried_start[:, None], start)
    pos = idx - start

    ctx = SeqContext(idx=idx, doc=doc_ids, pos=pos, is_doc_start=is_doc_start)
    new_state = SeqState(next_idx=idx[:, -1] + 1, last_doc=doc_ids[:, -1], doc_start_idx=start[:, -1])
    return ctx, new_state
