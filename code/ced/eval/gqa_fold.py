"""The reference attention reads without repeated key heads.

ced/model/attention.py `attend` and `attend_capped` are the ground truth of the reads on the "reference" backend. For grouped-query attention they expand the keys and values
to the query heads (`repeat_interleave` of k, k_far and v, then a cast to float32): at a decode step over a cache of S positions that is 3 x B x S x Hq x head_dim float32
values written and read per layer, which dwarfs the step's arithmetic (one query position) and bounds how many rows a decode step can carry (8,192 positions, 8 query heads,
head_dim 128, one key head: 100 MB a row and layer, 0.7 GB a row and step over the seven global layers).

The same softmax can be computed on the unexpanded keys: query head h = j * G + g (G = Hq / Hkv) reads key head j, so the G query heads of a key head fold into the query axis,

    q: (B, Lq, Hkv, G, hd) -> (B, Hkv, G * Lq, hd);   scores = q @ k^T with k: (B, Hkv, hd, S);   probs @ v with v: (B, Hkv, S, hd),

a batched matrix multiplication over (B, Hkv) with no expanded tensor. It is the same function: every logit is the same dot product, the mask, the scale, the dtype of the
softmax (at least float32) and the order of the two rotations of the capped read are the reference's. The accumulation order inside a dot product may differ from the
einsum's by a rounding in the last place of float32 (nothing in float64); the two reads agree on random inputs and in the cached generation of tiny
models, token for token.

`folded_reads()` swaps the two functions in ced.model.attention for the duration of a `with` (the model calls them by their module-level names on the reference backend);
nothing about the model, its checkpoints or the training path changes. Not applied to the flex / sdpa backends (they never expand) or to the asentmax normaliser (its own reference).
"""

from __future__ import annotations

import contextlib

import torch

from ced.model import attention as A
from ced.model.norms import compute_dtype


def _fold_q(q: torch.Tensor, hkv: int, dtype) -> torch.Tensor:
    B, Lq, Hq, hd = q.shape
    G = Hq // hkv
    return q.to(dtype).reshape(B, Lq, hkv, G, hd).permute(0, 2, 3, 1, 4).reshape(B, hkv, G * Lq, hd)


def _scores(qf: torch.Tensor, k: torch.Tensor, dtype, B: int, Lq: int, Hq: int, S: int) -> torch.Tensor:
    """(B, Hq, Lq, S): the logits of the folded queries against the unexpanded keys k (B, S, Hkv, hd), before the scale."""
    s = torch.matmul(qf, k.to(dtype).permute(0, 2, 3, 1))  # (B, Hkv, G * Lq, S)
    return s.reshape(B, Hq, Lq, S)


def _read(probs: torch.Tensor, v: torch.Tensor, q: torch.Tensor, dtype) -> torch.Tensor:
    """probs (B, Hq, Lq, S) x v (B, S, Hkv, hd) -> (B, Lq, Hq, hd) in q's dtype, the unexpanded values."""
    B, Lq, Hq, hd = q.shape
    hkv = v.shape[2]
    G = Hq // hkv
    pf = probs.reshape(B, hkv, G * Lq, probs.shape[-1])
    o = torch.matmul(pf, v.to(dtype).permute(0, 2, 1, 3))  # (B, Hkv, G * Lq, hd)
    return o.reshape(B, hkv, G, Lq, hd).permute(0, 3, 1, 2, 4).reshape(B, Lq, Hq, hd).to(q.dtype)


def attend_folded(q: torch.Tensor, keys, allow: torch.Tensor) -> torch.Tensor:
    """ced.model.attention.attend on the unexpanded keys (module docstring)."""
    assert getattr(keys, "k_far", None) is None, A.NOT_CAPPED
    B, Lq, Hq, hd = q.shape
    hkv, S = keys.k.shape[2], keys.k.shape[1]
    dtype = compute_dtype(q)
    scores = _scores(_fold_q(q, hkv, dtype), keys.k, dtype, B, Lq, Hq, S) * hd ** -0.5
    scores = scores.masked_fill(~allow[:, None, :, :], float("-inf"))
    return _read(torch.softmax(scores, dim=-1), keys.v, q, dtype)


def attend_capped_folded(q: torch.Tensor, q_far: torch.Tensor, keys, allow: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
    """ced.model.attention.attend_capped on the unexpanded keys (module docstring): both logit tensors in full, `where` picks per pair, one softmax."""
    assert keys.k_far is not None, "attend_capped needs the unrotated keys (KeySet.k_far)"
    B, Lq, Hq, hd = q.shape
    hkv, S = keys.k.shape[2], keys.k.shape[1]
    dtype = compute_dtype(q)
    near = _scores(_fold_q(q, hkv, dtype), keys.k, dtype, B, Lq, Hq, S) * hd ** -0.5
    farr = _scores(_fold_q(q_far, hkv, dtype), keys.k_far, dtype, B, Lq, Hq, S) * hd ** -0.5
    scores = torch.where(far[:, None, :, :], farr, near)
    scores = scores.masked_fill(~allow[:, None, :, :], float("-inf"))
    return _read(torch.softmax(scores, dim=-1), keys.v, q, dtype)


@contextlib.contextmanager
def folded_reads():
    """Within the block ced.model.attention.attend and attend_capped are the folded reads (the reference backend's calls resolve them by module-level name); restored on exit."""
    saved = (A.attend, A.attend_capped)
    A.attend, A.attend_capped = attend_folded, attend_capped_folded
    try:
        yield
    finally:
        A.attend, A.attend_capped = saved
