"""Host-side segment boundaries (cu_seqlens) for the fused varlen GDN kernel.

The fused kernel (fla `chunk_gated_delta_rule` with `cu_seqlens`, or ours, GDN kernel "ced") runs one
flattened sequence and restarts the recurrent state at every boundary. Training rows are flattened
row-major, so the boundaries are every row start plus every document start inside a row:

- encoder: positions t of row b with t == 0 or doc[b, t] != doc[b, t - 1];
- decoder slots (Looped-CED v1 only, whose decoder has GDN layers): the slot sequence of each row
  (the first C decoder positions, in order, then the empty slots; ced/model/capacity.py), with a
  boundary at slot 0, at every slot whose document differs from the previous slot's, and at the
  first empty slot.

CED v2 needs the encoder's only (its decoder has no GDN layer). The second-pass control's extra
delta-rule call per GDN layer (pass 2's read) runs over the full sequence, so it reuses the
encoder's boundaries, which pass 1 records (ced/model/gdn.py, GDNPass1).

Computing these on the device would need nonzero (a dynamic shape and a host sync) inside the
training step, which the engineering rules forbid. The data pipeline computes them from the CPU
batch instead. `check_against` verifies them against the model's own is_doc_start flags (tests,
and the trainer's optional debug check).

Every boundary array varlen_for_batch returns has passed ced/train/batch_check.py check_cu_seqlens:
first 0, last the flattened length, strictly increasing, every row start a boundary. The kernels
themselves check none of it. The arrays are unchanged by the check.
"""

from __future__ import annotations

import numpy as np
import torch


def _boundaries(doc: np.ndarray) -> np.ndarray:
    """doc (B, N) -> int32 cu_seqlens over the flattened (B * N) sequence."""
    B, N = doc.shape
    start = np.ones((B, N), dtype=bool)
    start[:, 1:] = doc[:, 1:] != doc[:, :-1]
    starts = np.flatnonzero(start.reshape(-1))
    return np.concatenate([starts, [B * N]]).astype(np.int32)


def slot_docs(doc_ids: np.ndarray, dec_mask: np.ndarray, capacity: int) -> np.ndarray:
    """(B, C) document id of each slot under keep-first-C selection; -1 for empty slots."""
    B, L = doc_ids.shape
    out = np.full((B, capacity), -1, dtype=np.int64)
    for b in range(B):
        kept = np.flatnonzero(dec_mask[b])[:capacity]
        out[b, : len(kept)] = doc_ids[b, kept]
    return out


def varlen_for_batch(doc_ids: torch.Tensor, dec_mask: torch.Tensor | None, capacity: int | None) -> dict:
    """{"enc": cu_seqlens (B*L + boundaries), "dec": cu_seqlens over the slots or None} as CPU int32
    tensors. doc_ids and dec_mask are the CPU batch tensors."""
    from ced.train.batch_check import check_cu_seqlens

    doc = doc_ids.numpy()
    enc = _boundaries(doc)
    check_cu_seqlens(enc, doc.shape[0], doc.shape[1], "enc")  # raises BatchError (batch_check.py); changes nothing
    out = {"enc": torch.from_numpy(enc), "dec": None}
    if dec_mask is not None and capacity is not None:
        dec = _boundaries(slot_docs(doc, dec_mask.numpy(), capacity))
        check_cu_seqlens(dec, doc.shape[0], capacity, "dec")
        out["dec"] = torch.from_numpy(dec)
    return out


def to_device(host: dict, device: torch.device) -> dict:
    """The model's `varlen` argument from varlen_for_batch's host tensors: device copies (non-blocking)
    plus the host copies, every one marked dynamic in its length for torch.compile. Their length (the
    number of segments) changes every batch; without the mark each new length would be a recompile."""
    out = {}
    for key in ("enc", "dec"):
        t = host.get(key)
        if t is None:
            continue
        dev = t.to(device, non_blocking=True)
        for x in (dev, t):
            torch._dynamo.maybe_mark_dynamic(x, 0)
        out[key], out[key + "_cpu"] = dev, t
    return out


def fixed_capacity(rows: int, seq_len: int, docs_per_row: int, chunk: int = 64) -> tuple[int, int]:
    """CUDA graphs (TrainConfig.cuda_graphs; ced/train/cuda_graphs.py): (S, nt_max), the segments and the chunks of chunk
    tokens of the fixed-capacity GDN metadata of a micro-batch of rows x seq_len tokens. S = rows x docs_per_row; a
    micro-batch with n <= S segments has at most ceil(T / chunk) + n chunks (each segment adds at most one partial
    chunk), so nt_max = ceil(T / chunk) + S holds every micro-batch that fits S."""
    S = rows * docs_per_row
    return S, -(-(rows * seq_len) // chunk) + S


def fixed_varlen_host(cu_exact: np.ndarray, rows: int, seq_len: int, docs_per_row: int,
                      chunk: int = 64) -> dict | None:
    """CUDA graphs: a micro-batch's exact boundaries (varlen_for_batch's "enc", checked) at the fixed capacity of
    fixed_capacity, as host int32 arrays: {"cu": (S + 2,), "ci": (nt_max, 2), "co": (S + 2,)}. cu is cu_exact, then the
    token count T repeated up to S segments (empty segments), then T once more (the sentinel segment S, which every
    padding chunk of ci names); ci and co are ced/kernels/gdn.py fixed_chunk_index's. None when the micro-batch has more
    than S segments: it does not fit, and the caller runs it eagerly with its exact boundaries."""
    from ced.kernels.gdn import fixed_chunk_index

    S, nt_max = fixed_capacity(rows, seq_len, docs_per_row, chunk)
    T = rows * seq_len
    n = len(cu_exact) - 1
    assert int(cu_exact[0]) == 0 and int(cu_exact[-1]) == T, (cu_exact[0], cu_exact[-1], T)
    if n > S:
        return None
    cu = np.full(S + 2, T, dtype=np.int32)
    cu[: n + 1] = cu_exact
    ci, co = fixed_chunk_index(cu, nt_max, chunk)
    return {"cu": cu, "ci": ci.numpy(), "co": co.numpy()}


def cu_from_starts(is_start: torch.Tensor) -> torch.Tensor:
    """Reference: cu_seqlens from (B, N) start flags (row starts forced). Uses nonzero: tests only."""
    B, N = is_start.shape
    s = is_start.clone()
    s[:, 0] = True
    starts = torch.nonzero(s.reshape(-1)).squeeze(1)
    return torch.cat([starts, torch.tensor([B * N], device=s.device)]).to(torch.int32)
