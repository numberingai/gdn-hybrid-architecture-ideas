"""Chunked gated delta rule: the GPU-viable form of `ced.model.gdn.delta_rule_recurrence`.

Why this exists. The reference recurrence walks the sequence one token at a time: at L = 8192
that is 8192 sequential steps of a few tiny kernels each, per GDN layer, forward and backward. On
a GPU that is launch-bound. The chunked form below cuts the row into chunks of C tokens, does all
the work *inside* every chunk at once with (C x C) matmuls and one triangular solve, and keeps
only an L / C step scan over chunks for the state. It computes the same function: it is accepted
because `tests/test_gdn_chunked.py` matches it against the recurrence in float64 (outputs, final
state and gradients, to 1e-10 / 1e-9), not because it looks right.

The algorithm is the WY / UT-transform form of the Gated DeltaNet paper (arXiv 2412.06464 §3.3),
as in fla's `naive_chunk_gated_delta_rule` and HF transformers' `torch_chunk_gated_delta_rule`.
Inside a chunk with entry state S_0 (d_k x d_v), positions i, j = 0..C-1 and in-chunk cumulative
log-decay c_i = g_0 + ... + g_i (so gamma_i = exp(c_i)):

    A[i, j] = beta_i (k_i . k_j) exp(c_i - c_j)                    for j < i, else 0
    U       = (I + A)^-1 diag(beta) V                               "new values" (C x d_v)
    W       = (I + A)^-1 diag(beta gamma) K                         "decayed keys" (C x d_k)
    w       = U - W S_0                                             what each token writes
    o_i     = gamma_i S_0^T q_i + sum_{j <= i} exp(c_i - c_j) (q_i . k_j) w_j
    S_C     = gamma_{C-1} S_0 + sum_j exp(c_{C-1} - c_j) k_j w_j^T

Document starts inside a chunk. The recurrence zeroes the state at every document start, and a
packed row has starts at arbitrary positions. fla's varlen kernels align chunks to document
boundaries (cu_seqlens), which gives data-dependent shapes; the HF pure-torch path ignores
document starts and leaks state across documents. Here the chunks stay fixed and documents are
handled by masks. Each position gets a segment number inside its chunk, seg_i = the number of
document starts at or before i in the chunk; segment 0 is the document continuing from the
previous chunk. Then, with no other change to the formulas:
  - every pairwise term (A, and the q . k term of o_i) is kept only when seg_i == seg_j, so no
    token reads or writes across a document boundary;
  - the S_0 terms (gamma_i S_0^T q_i and the W rows) are kept only for segment 0, which is the
    state reset: later segments start from an empty state;
  - the chunk's exit state keeps only the last segment: S_0 survives only if the chunk has no
    start, and only tokens of the last segment write into it.
Because (I + A) is block-diagonal by segment once masked, its inverse is too, so the solve never
mixes documents either. The decay at a document start itself multiplies an empty state, so it
drops out of every kept term (all kept ratios exp(c_i - c_j) have j >= the segment start).

Precision. Document starts are NOT implemented by setting g to a huge negative number at the
start: the cumulative sums c would then be ~1e9 and their differences would lose all float32
precision. g is used as given. The pairwise exponents c_i - c_j are not formed as a difference
either: they are summed directly over the tokens between j and i (Mamba2's "segsum"). With
large decays (|g| ~ 50 per token, |c| ~ 1600 at C = 64) the difference form loses ~1e-4 relative
precision in float32; the direct sum keeps the chunked output as close to float64 as the
recurrence is (measured in the test). Every kept exponent (pairs j <= i, c_i, and the decay to
the chunk end) is <= 0 when g <= 0, so nothing overflows. Masked pairs are replaced by 0 *before*
exp and the result by 0 after it (the double `where`), so even a positive g in a masked pair
cannot put inf or NaN into the forward or the backward pass.

Fixed shapes. The segment numbers come from a cumsum over `is_doc_start`, and every mask is a
comparison, so the op sequence and every tensor shape depend only on (B, L, H, d_k, d_v, C), never
on where the documents start (checked with an aten-op recorder in the test). A row whose length is
not a multiple of C is padded at the end with g = 0, beta = 0 and no document start: padded
positions neither decay nor write, so they leave the state unchanged, and their outputs are cut.

Dtype. The work is done in float32 or wider (float64 stays float64), and results are returned in
the dtype the recurrence would return. Callers in bf16 therefore get a float32 state scan, as fla
does; the recurrence itself would keep the state in bf16.
"""

from __future__ import annotations

import math
from functools import reduce

import torch
import torch.nn.functional as F


def chunked_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    is_doc_start: torch.Tensor,
    S: torch.Tensor,
    chunk_size: int = 64,
    all_states: bool = False,
):
    """Same arguments and results as `delta_rule_recurrence`, computed chunk by chunk.

    q, k: (B, L, H, d_k); v: (B, L, H, d_v); g, beta: (B, L, H); is_doc_start: (B, L) bool;
    S: (B, H, d_k, d_v) state before the first token. Returns outputs (B, L, H, d_v) and the
    final state (B, H, d_k, d_v). q is used as given (the caller has already scaled it).
    `chunk_size` is clamped to L, so a short call (a cached decode step) does not pad to 64.

    all_states (the GDN-side arm CCR's writer on the paths without the fused kernel; ced/model/gdn_arms.py): also
    return the state after every position, (B, L, H, d_k, d_v), as delta_rule_recurrence(..., all_states=True) does:
    inside a chunk S_i = gamma_i S_0 [i in the chunk's first segment] + sum_{j <= i, same segment} exp(c_i - c_j)
    k_j w_j^T (the same w as the output's). False (the default): nothing below changes.
    """
    assert chunk_size >= 1
    B, L, H, dk = q.shape
    dv = v.shape[-1]
    out_dtype = reduce(torch.promote_types, (t.dtype for t in (q, k, v, g, beta, S)))
    work = torch.promote_types(out_dtype, torch.float32)
    C = min(chunk_size, L)
    n = math.ceil(L / C)
    pad = n * C - L

    def per_head(x: torch.Tensor) -> torch.Tensor:
        """(B, L, H, ...) -> (B, H, n, C, ...), padded at the end of the row with zeros."""
        x = x.to(work).transpose(1, 2)  # (B, H, L, ...)
        trailing = x.dim() - 3
        x = F.pad(x, (0, 0) * trailing + (0, pad))
        return x.reshape(B, H, n, C, *x.shape[3:])

    q, k, v, g, beta = map(per_head, (q, k, v, g, beta))  # q, k: (B,H,n,C,dk); v: (B,H,n,C,dv); g, beta: (B,H,n,C)

    # ---- document segments inside each chunk: (B, 1, n, C), shared by all heads
    starts = F.pad(is_doc_start.to(torch.long), (0, pad)).reshape(B, 1, n, C)
    seg = starts.cumsum(-1)  # 0 = the document carried in from the previous chunk
    same_seg = seg[..., :, None] == seg[..., None, :]  # (B, 1, n, C, C)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()  # j <= i
    pair_ok = same_seg & causal  # j <= i and the same document
    in_first = (seg == 0).to(work)  # sees the incoming state S_0
    in_last = (seg == seg[..., -1:]).to(work)  # writes into the outgoing state
    carry_through = (seg[..., -1] == 0).to(work)  # (B, 1, n): no start in the chunk, S_0 survives

    # ---- decays, all exponents <= 0 where kept
    zero = torch.zeros((), dtype=work, device=q.device)
    c = g.cumsum(-1)  # (B, H, n, C) in-chunk cumulative log-decay
    # pair_exp[..., i, j] = c_i - c_j = g_{j+1} + ... + g_i for j < i, and 0 for j >= i. Summed
    # directly (a cumsum down the columns of the strictly lower part of g_i), not as a difference
    # of two cumulative sums, which cancels badly in float32 once |c| is large (Mamba2's segsum).
    strictly_lower = causal.tril(-1)
    pair_exp = torch.where(strictly_lower, g[..., :, None].expand(*g.shape, C), zero).cumsum(-2)
    pair_decay = torch.where(pair_ok, torch.exp(torch.where(pair_ok, pair_exp, zero)), zero)  # (B,H,n,C,C)
    decay_from_start = torch.exp(c)  # gamma_i
    decay_to_end = torch.exp(pair_exp[..., -1, :])  # gamma_{C-1} / gamma_i

    # ---- intra-chunk work for all chunks at once: UT transform, one triangular solve
    k_beta = k * beta[..., None]
    A = (k_beta @ k.transpose(-1, -2)) * torch.where(strictly_lower, pair_decay, zero)  # j < i, same document
    eye = torch.eye(C, dtype=work, device=q.device)
    rhs = torch.cat([v * beta[..., None], k_beta * (decay_from_start * in_first)[..., None]], dim=-1)
    solved = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
    U, W = solved.split([dv, dk], dim=-1)  # U: (B,H,n,C,dv), W: (B,H,n,C,dk)

    intra = (q @ k.transpose(-1, -2)) * pair_decay  # (B,H,n,C,C): q_i . k_j, j <= i, same document
    q_from_state = q * (decay_from_start * in_first)[..., None]
    k_to_end = k * (decay_to_end * in_last)[..., None]
    state_decay = (decay_from_start[..., -1] * carry_through)[..., None, None]  # (B,H,n,1,1)

    # ---- the only sequential part: one step per chunk.
    # The chunk axis is unbound once rather than indexed as X[:, :, i] inside the loop: the backward
    # of each X[:, :, i] zero-fills and accumulates a full-size gradient, which is O(n^2) in the
    # number of chunks (measured: 8.7x the time for 4x the length); unbind's backward is one stack.
    per_chunk = (t.unbind(2) for t in (U, W, q_from_state, intra, k_to_end, state_decay))
    state = S.to(work)
    outputs = []
    if all_states:  # (CCR's writer) the state after every position of each chunk, from the same w
        states = []
        extra = zip((decay_from_start * in_first).unbind(2), pair_decay.unbind(2), k.unbind(2))
    for U_i, W_i, q_i, intra_i, k_i, decay_i in zip(*per_chunk):
        written = U_i - W_i @ state  # (B, H, C, dv): w = U - W S_0
        outputs.append(q_i @ state + intra_i @ written)
        if all_states:
            from_start_i, pair_i, k_raw_i = next(extra)
            states.append(from_start_i[..., None, None] * state[:, :, None]
                          + torch.einsum("bhij,bhjk,bhjv->bhikv", pair_i, k_raw_i, written))
        state = state * decay_i + k_i.transpose(-1, -2) @ written

    o = torch.stack(outputs, dim=2).reshape(B, H, n * C, dv)[:, :, :L].transpose(1, 2)
    if all_states:
        st = torch.stack(states, dim=2).reshape(B, H, n * C, dk, dv)[:, :, :L].transpose(1, 2)
        return o.to(out_dtype), state.to(out_dtype), st.to(out_dtype)
    return o.to(out_dtype), state.to(out_dtype)
