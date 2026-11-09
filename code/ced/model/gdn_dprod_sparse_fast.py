"""The fast path of the inserted-row layout of the variants (GDNConfig.dprod.sparse_path "fast"; ced/model/gdn_dprod_sparse.py, the layer that
calls it).

THE ASK. The standing limit: "Kernel work until the B200 arrives (soon): only the clear algorithmic gains (removing duplicated computation and
redundant reads/writes, e.g. one readout per token here), not extreme tuning."

WHAT IT IS. The same function as the plain path, bit for bit, with the data movement of the row layout at its fewest passes. The plain path
(gdn_dprod_sparse.insert_rows / take_tokens) writes the right values but the autograd graph of its slices is wasteful: a slice of a tensor
backs onto "zeros of the whole tensor, then copy the piece in" (SliceBackward), and a q row built as cat(first rows, zeros) then cat(. extra)
is written twice. On the training path that is, per layer and micro-batch, several extra full passes over q and over the kernel's output o
(computed per op, section 3). Here:
- the token axis is cut with ONE `split` (a view; its backward is one cat of the pieces' gradients, no zero-filled full tensor), and the
  group axis with `split` too: the row of a carrying token that is not read (q = 0 on its first row; the output of its first row) is the
  one dropped piece, a (B, n_groups, 1, ...) tensor, whose gradient is a small zeros;
- the q rows are one cat of three pieces (the non-carrying rows, a zero row, the carrying token's q), not two nested cats;
- take_tokens is one cat of two views of the kernel's output.
Every operation is a copy or a view: the values AND the gradients equal the plain path's bit for bit in every dtype (the gradient of a copy is
a copy; the pieces' supports are disjoint, so no addition changes a value), and the shapes depend on Python integers only (the packed-row
phase): no index op, no data-dependent shape. Nothing here is a kernel; nothing is tuned.

WHAT "fast" DOES NOT TOUCH. The chunk kernel ('ced') and the input preparation (qkv_prep) are as they were: the row layout still reaches the
kernel as q, k, v, g, beta of L + n_c rows, and the extra rows still cost what a token costs (the design's "interleave route"). The copies
that remain are the layout's minimum outside the kernels: one write of each of q, k, v, g, beta and one read of o in the forward, the same
gathers in the backward.

"rows" (GDNConfig.dprod.sparse_path "rows"; `prep_rows` below) removes the copies of q, k and v: the fused preparation itself writes the row
layout and reads the kernel's gradients from it (ced/kernels/gdn_prep_rows.py; the Triton kernels ced/kernels/gdn_prep_rows_triton.py move the
addresses of prep_fwd_kernel and prep_da_kernel and change no other statement). The copies of g and beta (small) and of o stay; the gated norm
reading the rows is a kernel item, not built. The same values and gradients again, bit for bit.

The pooling of D's summary is NOT changed: its arithmetic is the plain path's (a masked float32 sum), because a faster form (a batched matrix
product) would run in TF32 under the trainer's `tf32: true` and round x to ten mantissa bits; the fast path is bit-exact by construction.
"""

from __future__ import annotations

import torch

PATHS = ("plain", "fast", "rows")  # GDNConfig.dprod.sparse_path as set; not set (None): "fast" for the variant xmean and kvmean (kvmean since its GPU values check passed), "plain" for the other forms (gdn_dprod_sparse.path_of); "plain" set is every run before the key


@torch.compiler.disable  # never traced: an opaque fused op (its own forward and backward kernels), as ced.model.gdn.qkv_prep
def prep_rows(u: torch.Tensor, doc: torch.Tensor, conv_weight: torch.Tensor, hk: int, hv: int, d: int, k2: torch.Tensor, v2: torch.Tensor,
              stride: int, out_dtype: torch.dtype | None = None):
    """sparse_path "rows": qkv_prep (short conv + SiLU + split + L2 norm + q scale, fused) whose q, k, v are the inserted-row layout's (B, L + L / stride,
    heads, d): the extra rows of k and v are k2, v2 (B, L / stride, heads, d), q is zero on a carrying token's first row and its own on the extra row
    (ced/kernels/gdn_prep_rows.py; the kernels ced/kernels/gdn_prep_rows_triton.py). Equal to insert_rows applied to qkv_prep's outputs, bit for bit."""
    from ced.kernels.gdn_prep_rows import qkv_prep_rows as fused

    return fused(u, doc, conv_weight, hk, hv, d, k2, v2, stride, out_dtype=out_dtype)


def _cut(t: torch.Tensor, sizes: list):
    """t cut along dim 1 into consecutive pieces of the given sizes, as views: one SplitWithSizes node (its backward is one cat of the
    pieces' gradients). One piece covering all of t is t itself (no node at all)."""
    sizes = [n for n in sizes if n]
    if len(sizes) <= 1:
        return [t]
    return list(t.split(sizes, dim=1))


def insert_rows_fast(t: torch.Tensor, extra, rs: list, a_end: int, kind: str = "plain") -> torch.Tensor:
    """gdn_dprod_sparse.insert_rows with the geometry given: rs, a_end = groups_of(L, s, off). t (B, L, ...); extra (B, n_c, ...) the extra
    rows in carrying order (kind "plain"); kind "q": the extra row is the carrying token's own q and its first row is zero. Returns
    (B, L + n_c, ...), equal to insert_rows bit for bit, values and gradients."""
    B, L = t.shape[:2]
    if not rs:
        return t
    rest = t.shape[2:]
    tail = L - a_end
    pieces = _cut(t, [ng * sz for _, ng, sz in rs] + [tail])
    parts, c = [], 0
    for (a, ng, sz), piece in zip(rs, pieces):
        seg = piece.reshape(B, ng, sz, *rest)
        if kind == "q":
            head, last = seg.split([sz - 1, 1], dim=2)
            rows = torch.cat([head, torch.zeros_like(last), last], dim=2)
        else:
            rows = torch.cat([seg, extra[:, c:c + ng].unsqueeze(2)], dim=2)
        c += ng
        parts.append(rows.reshape(B, ng * (sz + 1), *rest))
    if tail:
        parts.append(pieces[-1])
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def take_tokens_fast(o: torch.Tensor, L: int, rs: list, a_end: int) -> torch.Tensor:
    """gdn_dprod_sparse.take_tokens with the geometry given: the token outputs of the rows' outputs o (B, L + n_c, ...): a carrying token's
    output is read on its extra row (its first row has q = 0), every other token's on its own row. Returns (B, L, ...), equal to take_tokens
    bit for bit, values and gradients."""
    B = o.shape[0]
    if not rs:
        return o
    rest = o.shape[2:]
    rows = sum(ng * (sz + 1) for _, ng, sz in rs)
    tail = o.shape[1] - rows
    pieces = _cut(o, [ng * (sz + 1) for _, ng, sz in rs] + [tail])
    parts = []
    for (a, ng, sz), piece in zip(rs, pieces):
        seg = piece.reshape(B, ng, sz + 1, *rest)
        head, _unread, last = seg.split([sz - 1, 1, 1], dim=2)
        parts.append(torch.cat([head, last], dim=2).reshape(B, ng * sz, *rest))
    if tail:
        parts.append(pieces[-1])
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
