"""Our Triton kernels for the chunked gated delta rule (training forward and backward, empty initial
state, varlen segments, grouped value heads). Orchestrated by ced/kernels/gdn.py.

Derived from flash-linear-attention (fla-core 0.5.2, MIT licence; Copyright (c) 2023-2026 Songlin Yang,
Yu Zhang, Zhiyuan Li and contributors, https://github.com/fla-org/flash-linear-attention): the
per-value-head arithmetic of each kernel below is fla's (chunk_local_cumsum, the fused K K^T + triangular
solve, recompute_w_u, the chunk-state recurrences, chunk_fwd_o, chunk_bwd_dv_local, chunk_bwd_dqkwg and
prepare_wy_repr_bwd), with the same dot precisions. What differs is the organisation:

- programs per key head where the key-side work is shared (the solve's K K^T with GP = 2, the output's
  q k^T, dv_local's k q^T, dqkwg): those products are computed once per key head instead of once per
  value head, and dq and dk are summed over the group in float32 registers instead of being written per
  value head and reduced by PyTorch kernels; w/u and the WY backward run per value head (GP = 1, fewer
  live tiles), their dk part summed by dk_sum_kernel in fla's order;
- the WY backward also finishes dg (dqkwg's part + its own, then the chunk-local reverse cumsum);
- fixed block sizes, warp counts and heads per program, chosen by measurement at our shapes on a MIG 1g
  slice (no run-time autotuning: the same kernels run in every process);
- an accumulation dtype `ACC` (float32, or float64 when the inputs are float64): the float64 path
  checks the algebra against the float64 reference to rounding, independent of precision.

Also here: the GDN input preparation (short conv within documents, SiLU, L2 norm), forward and
backward (prep_*_kernel; ced/kernels/gdn.py: qkv_prep).

F3, the compensated cumulative sum (GDNConfig.decay_cumsum "compensated"). Every kernel that reads gc takes a constexpr
F3 and a second pointer gl. F3 False (the default of every run, gl None): each `if F3:` branch is dropped at compile time
and the kernel is the code above it always was, statement for statement (tests/test_gdn_f3.py checks the syntax trees
against main's). F3 True: cumsum_fwd_kernel stores the within-chunk cumulative sum of g in NATURAL units as a float pair
(hi, lo) from a sequential TwoSum (Knuth) scan, hi the running float sum and lo the running sum of its exact rounding
errors, and every reader forms its exponent from the pair: a decay between two tokens of a chunk as exp2(((hi_i - hi_j) +
(lo_i - lo_j)) * RCP_LN2), in that order, and a decay from the chunk start as exp2((hi_i + lo_i) * RCP_LN2). The pair's
error is O(64^2 2^-48) of the chunk's sum of |g| instead of the plain sum's O(2^-24) (ced/kernels/gdn_f3_ref.py: the
bound and a float32 emulation of the scan).

Layouts (batch 1, flattened varlen): q, k (1, T, H, K); v, u, v_new, o (1, T, HV, V); w (1, T, HV, K);
g, gc, beta (1, T, HV); A (1, T, HV, BT); h, dh (1, NT, HV, K, V). Value head j reads key head
j // G. Chunks never cross a segment boundary (cu_seqlens; chunk_indices lists (segment, chunk in
segment) per global chunk; chunk_offsets the first global chunk of each segment).

C, the input preparation in the chunk kernels' prologues (GDNConfig.edges "fused"). Every kernel that reads q, k or v
takes a constexpr PRO and, after its other arguments, the in_proj_qkv output `qkv` (B*L, C), the document ids `doc`,
the conv weight `cw`, the per-row L2-norm factor `rs` (B*L, 2H) of prep_rs_kernel, the row length L, C, the conv taps
KC and the preparation's accumulation dtype PACC. PRO False (the default of every launch, the new arguments None): each
`if PRO:` branch is dropped at compile time and the kernel is the code above it always was, statement for statement
(tests/test_gdn_edges.py checks the syntax trees against main's). PRO True: each load of q, k or v is replaced by
_pro_tile, which forms the same values from qkv with qkv_prep's forward arithmetic (prep_fwd_kernel: the conv taps in
the same order, SiLU, the L2 norm's factor from prep_rs_kernel, which is prep_fwd_kernel's reduction statement for
statement, and the q scale), rounded to the kernel's input dtype exactly where prep_fwd_kernel stores; rows outside the
segment are 0 as the masked loads give. q, k and v are then never written. The q, k, v pointers of a PRO launch are
1-element dummies of the input dtype: only their dtype is read (and pointer offsets formed); nothing loads them.
"""

from __future__ import annotations

import os

import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.4426950408889634)  # g is used in log2 units inside the kernels, as in fla
# fla's in-chunk triangular solve uses TF32 dots on TF32-capable GPUs; the float32 correctness tests
# set CED_FLA_SOLVE_PRECISION=tf32x3 for fla's kernel (tests/gpu/conftest.py) and for ours.
SOLVE_PRECISION = os.environ.get("CED_FLA_SOLVE_PRECISION", "tf32")
# C (PRO): the q scale of the input preparation, d^-0.5 for the head width 128 every kernel here takes (the wrappers
# assert K = V = 128); qkv_prep passes prep_fwd_kernel the same Python float, d**-0.5.
PRO_Q_SCALE = tl.constexpr(128 ** -0.5)


@triton.jit
def _chunk_bounds(cu_seqlens, chunk_indices, i_tg):
    """(bos, T, i_t): the segment's first token, its length, the chunk's index inside it."""
    i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    return bos, eos - bos, i_t


@triton.jit
def _f3_pair(hi_i, hi_j, lo_i, lo_j):
    """F3: the decay exp(c_i - c_j) between two tokens of a chunk from the compensated pairs (natural units), the exponent
    formed as (hi_i - hi_j) + (lo_i - lo_j) in that order (hi_i - hi_j is exact when the two are within a factor 2)."""
    return tl.math.exp2(((hi_i - hi_j) + (lo_i - lo_j)) * RCP_LN2)


@triton.jit
def _f3_one(hi, lo):
    """F3: the decay exp(c_i) from the chunk start (one compensated pair, natural units)."""
    return tl.math.exp2((hi + lo) * RCP_LN2)


# ------------------------------------------------------------------------------------------------
# Forward


@triton.jit(do_not_specialize=["T"])
def cumsum_fwd_kernel(g, gc, gl, cu_seqlens, chunk_indices, T, HV: tl.constexpr, BT: tl.constexpr, ACC: tl.constexpr,
                      F3: tl.constexpr):
    """gc = chunk-local cumulative sum of g, times 1 / ln 2 (fla's chunk_local_cumsum, scale RCP_LN2).
    F3: (gc, gl) = (hi, lo), the compensated sum in natural units (no 1 / ln 2): one sequential TwoSum scan over the
    chunk's positions, hi_t = fl(hi_{t-1} + g_t), lo_t = fl(lo_{t-1} + r_t) with r_t = hi_{t-1} + g_t - hi_t exactly
    (Knuth's TwoSum: six adds, no branch, no magnitude condition). The positions past the segment's end add 0. The
    chunk is held in registers; position t is read as a masked sum (one nonzero term: exact) and written with a select,
    so nothing is reordered: ced/kernels/gdn_f3_ref.py compensated_chunk_cumsum is the same arithmetic in PyTorch."""
    i_tg, i_hv = tl.program_id(0), tl.program_id(1)
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p = (bos + o_t).to(tl.int64) * HV + i_hv
    b_s = tl.load(g + p, mask=m_t, other=0.0).to(ACC)
    if F3:
        o_i = tl.arange(0, BT)
        b_hi = tl.zeros([BT], dtype=ACC)
        b_lo = tl.zeros([BT], dtype=ACC)
        s = tl.sum(b_hi, axis=0)  # 0 in ACC: the running float sum
        e = tl.sum(b_lo, axis=0)  # 0 in ACC: the running sum of the exact rounding errors
        for t in range(BT):
            x = tl.sum(tl.where(o_i == t, b_s, 0.0), axis=0)
            s_new = s + x
            bb = s_new - s
            err = (s - (s_new - bb)) + (x - bb)
            s = s_new
            e = e + err
            b_hi = tl.where(o_i == t, s, b_hi)
            b_lo = tl.where(o_i == t, e, b_lo)
        tl.store(gc + p, b_hi.to(gc.dtype.element_ty), mask=m_t)
        tl.store(gl + p, b_lo.to(gl.dtype.element_ty), mask=m_t)
    else:
        b_o = tl.cumsum(b_s, axis=0) * RCP_LN2
        tl.store(gc + p, b_o.to(gc.dtype.element_ty), mask=m_t)


@triton.jit(do_not_specialize=["T"])
def intra_fwd_kernel(k, gc, gl, beta, A, cu_seqlens, chunk_indices, T,
                     H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
                     BC: tl.constexpr, BK: tl.constexpr, GP: tl.constexpr, ACC: tl.constexpr, SOLVE_PREC: tl.constexpr,
                     F3: tl.constexpr, qkv=None, doc=None, cw=None, rs=None, L: tl.constexpr = None, C: tl.constexpr = None,
                     KC: tl.constexpr = None, PACC: tl.constexpr = None, PRO: tl.constexpr = False):
    """fla's fused K K^T + triangular solve (chunk_gated_delta_rule_fwd_kkt_solve_kernel) per (chunk, GP
    value heads of one key head): K K^T once; per value head A = (I + beta K K^T * decay, strictly
    lower)^-1 by fla's 16 x 16 block forward substitution and block merge. Stores A in k's dtype (only
    its lower triangle: readers mask the rest). GP = 1 is fla's grid."""
    HV: tl.constexpr = H * G
    i_tg, i_p = tl.program_id(0), tl.program_id(1)
    i_h = (i_p * GP) // G
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    i_tc0 = i_t * BT
    i_tc1 = i_tc0 + BC
    i_tc2 = i_tc0 + 2 * BC
    i_tc3 = i_tc0 + 3 * BC
    k += (bos * H + i_h).to(tl.int64) * K
    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # K K^T: the 10 lower-triangular 16 x 16 blocks, shared by the group's value heads
    r00 = tl.zeros([BC, BC], dtype=ACC)
    r11 = tl.zeros([BC, BC], dtype=ACC)
    r22 = tl.zeros([BC, BC], dtype=ACC)
    r33 = tl.zeros([BC, BC], dtype=ACC)
    r10 = tl.zeros([BC, BC], dtype=ACC)
    r20 = tl.zeros([BC, BC], dtype=ACC)
    r21 = tl.zeros([BC, BC], dtype=ACC)
    r30 = tl.zeros([BC, BC], dtype=ACC)
    r31 = tl.zeros([BC, BC], dtype=ACC)
    r32 = tl.zeros([BC, BC], dtype=ACC)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        if PRO:  # k of rows i_tc0 .. i_tc0 + 15, channels o_k of key head i_h (BK divides K: m_k is all true)
            b_k0 = _pro_tile(qkv, doc, cw, rs, bos + i_tc0 + o_i, m_tc0, H * K + i_h * K + o_k, H + i_h, L, C, KC, 2 * H,
                             True, False, PACC, k.dtype.element_ty)
        else:
            b_k0 = tl.load(k + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_tc0[:, None] & m_k[None, :], other=0.0)
        r00 += tl.dot(b_k0, tl.trans(b_k0))
        if i_tc1 < T:
            if PRO:
                b_k1 = _pro_tile(qkv, doc, cw, rs, bos + i_tc1 + o_i, m_tc1, H * K + i_h * K + o_k, H + i_h, L, C, KC,
                                 2 * H, True, False, PACC, k.dtype.element_ty)
            else:
                b_k1 = tl.load(k + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_tc1[:, None] & m_k[None, :], other=0.0)
            r11 += tl.dot(b_k1, tl.trans(b_k1))
            r10 += tl.dot(b_k1, tl.trans(b_k0))
            if i_tc2 < T:
                if PRO:
                    b_k2 = _pro_tile(qkv, doc, cw, rs, bos + i_tc2 + o_i, m_tc2, H * K + i_h * K + o_k, H + i_h, L, C, KC,
                                     2 * H, True, False, PACC, k.dtype.element_ty)
                else:
                    b_k2 = tl.load(k + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_tc2[:, None] & m_k[None, :], other=0.0)
                r22 += tl.dot(b_k2, tl.trans(b_k2))
                r20 += tl.dot(b_k2, tl.trans(b_k0))
                r21 += tl.dot(b_k2, tl.trans(b_k1))
                if i_tc3 < T:
                    if PRO:
                        b_k3 = _pro_tile(qkv, doc, cw, rs, bos + i_tc3 + o_i, m_tc3, H * K + i_h * K + o_k, H + i_h, L, C,
                                         KC, 2 * H, True, False, PACC, k.dtype.element_ty)
                    else:
                        b_k3 = tl.load(k + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :], mask=m_tc3[:, None] & m_k[None, :], other=0.0)
                    r33 += tl.dot(b_k3, tl.trans(b_k3))
                    r30 += tl.dot(b_k3, tl.trans(b_k0))
                    r31 += tl.dot(b_k3, tl.trans(b_k1))
                    r32 += tl.dot(b_k3, tl.trans(b_k2))

    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    for j in tl.static_range(GP):
        i_hv = i_p * GP + j
        pg = gc + bos.to(tl.int64) * HV + i_hv
        pb = beta + bos.to(tl.int64) * HV + i_hv
        b_g0 = tl.load(pg + (i_tc0 + o_i) * HV, mask=m_tc0, other=0.0).to(ACC)
        b_g1 = tl.load(pg + (i_tc1 + o_i) * HV, mask=m_tc1, other=0.0).to(ACC)
        b_g2 = tl.load(pg + (i_tc2 + o_i) * HV, mask=m_tc2, other=0.0).to(ACC)
        b_g3 = tl.load(pg + (i_tc3 + o_i) * HV, mask=m_tc3, other=0.0).to(ACC)
        b_b0 = tl.load(pb + (i_tc0 + o_i) * HV, mask=m_tc0, other=0.0).to(ACC)
        b_b1 = tl.load(pb + (i_tc1 + o_i) * HV, mask=m_tc1, other=0.0).to(ACC)
        b_b2 = tl.load(pb + (i_tc2 + o_i) * HV, mask=m_tc2, other=0.0).to(ACC)
        b_b3 = tl.load(pb + (i_tc3 + o_i) * HV, mask=m_tc3, other=0.0).to(ACC)

        # gate (masked so that out-of-range rows never produce 0 * inf) and beta
        if F3:
            pl = gl + bos.to(tl.int64) * HV + i_hv
            b_l0 = tl.load(pl + (i_tc0 + o_i) * HV, mask=m_tc0, other=0.0).to(ACC)
            b_l1 = tl.load(pl + (i_tc1 + o_i) * HV, mask=m_tc1, other=0.0).to(ACC)
            b_l2 = tl.load(pl + (i_tc2 + o_i) * HV, mask=m_tc2, other=0.0).to(ACC)
            b_l3 = tl.load(pl + (i_tc3 + o_i) * HV, mask=m_tc3, other=0.0).to(ACC)
            a00 = r00 * tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], _f3_pair(b_g0[:, None], b_g0[None, :], b_l0[:, None], b_l0[None, :]), 0.0)
            a11 = r11 * tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], _f3_pair(b_g1[:, None], b_g1[None, :], b_l1[:, None], b_l1[None, :]), 0.0)
            a22 = r22 * tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], _f3_pair(b_g2[:, None], b_g2[None, :], b_l2[:, None], b_l2[None, :]), 0.0)
            a33 = r33 * tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], _f3_pair(b_g3[:, None], b_g3[None, :], b_l3[:, None], b_l3[None, :]), 0.0)
            a10 = r10 * tl.where(m_tc1[:, None] & m_tc0[None, :], _f3_pair(b_g1[:, None], b_g0[None, :], b_l1[:, None], b_l0[None, :]), 0.0)
            a20 = r20 * tl.where(m_tc2[:, None] & m_tc0[None, :], _f3_pair(b_g2[:, None], b_g0[None, :], b_l2[:, None], b_l0[None, :]), 0.0)
            a21 = r21 * tl.where(m_tc2[:, None] & m_tc1[None, :], _f3_pair(b_g2[:, None], b_g1[None, :], b_l2[:, None], b_l1[None, :]), 0.0)
            a30 = r30 * tl.where(m_tc3[:, None] & m_tc0[None, :], _f3_pair(b_g3[:, None], b_g0[None, :], b_l3[:, None], b_l0[None, :]), 0.0)
            a31 = r31 * tl.where(m_tc3[:, None] & m_tc1[None, :], _f3_pair(b_g3[:, None], b_g1[None, :], b_l3[:, None], b_l1[None, :]), 0.0)
            a32 = r32 * tl.where(m_tc3[:, None] & m_tc2[None, :], _f3_pair(b_g3[:, None], b_g2[None, :], b_l3[:, None], b_l2[None, :]), 0.0)
        else:
            a00 = r00 * tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], tl.math.exp2(b_g0[:, None] - b_g0[None, :]), 0.0)
            a11 = r11 * tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], tl.math.exp2(b_g1[:, None] - b_g1[None, :]), 0.0)
            a22 = r22 * tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], tl.math.exp2(b_g2[:, None] - b_g2[None, :]), 0.0)
            a33 = r33 * tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], tl.math.exp2(b_g3[:, None] - b_g3[None, :]), 0.0)
            a10 = r10 * tl.where(m_tc1[:, None] & m_tc0[None, :], tl.math.exp2(b_g1[:, None] - b_g0[None, :]), 0.0)
            a20 = r20 * tl.where(m_tc2[:, None] & m_tc0[None, :], tl.math.exp2(b_g2[:, None] - b_g0[None, :]), 0.0)
            a21 = r21 * tl.where(m_tc2[:, None] & m_tc1[None, :], tl.math.exp2(b_g2[:, None] - b_g1[None, :]), 0.0)
            a30 = r30 * tl.where(m_tc3[:, None] & m_tc0[None, :], tl.math.exp2(b_g3[:, None] - b_g0[None, :]), 0.0)
            a31 = r31 * tl.where(m_tc3[:, None] & m_tc1[None, :], tl.math.exp2(b_g3[:, None] - b_g1[None, :]), 0.0)
            a32 = r32 * tl.where(m_tc3[:, None] & m_tc2[None, :], tl.math.exp2(b_g3[:, None] - b_g2[None, :]), 0.0)
        a00 = a00 * b_b0[:, None]
        a11 = a11 * b_b1[:, None]
        a22 = a22 * b_b2[:, None]
        a33 = a33 * b_b3[:, None]
        a10 = a10 * b_b1[:, None]
        a20 = a20 * b_b2[:, None]
        a21 = a21 * b_b2[:, None]
        a30 = a30 * b_b3[:, None]
        a31 = a31 * b_b3[:, None]
        a32 = a32 * b_b3[:, None]

        # forward substitution on the diagonal blocks: (I + A_cc)^-1
        i00 = -a00
        i11 = -a11
        i22 = -a22
        i33 = -a33
        for i in range(2, min(BC, T - i_tc0)):
            b_a = tl.sum(tl.where((o_i == i)[:, None], -a00, 0.0), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a = b_a + tl.sum(b_a[:, None] * i00, 0)
            i00 = tl.where((o_i == i)[:, None], b_a, i00)
        for i in range(2, min(BC, T - i_tc1)):
            b_a = tl.sum(tl.where((o_i == i)[:, None], -a11, 0.0), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a = b_a + tl.sum(b_a[:, None] * i11, 0)
            i11 = tl.where((o_i == i)[:, None], b_a, i11)
        for i in range(2, min(BC, T - i_tc2)):
            b_a = tl.sum(tl.where((o_i == i)[:, None], -a22, 0.0), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a = b_a + tl.sum(b_a[:, None] * i22, 0)
            i22 = tl.where((o_i == i)[:, None], b_a, i22)
        for i in range(2, min(BC, T - i_tc3)):
            b_a = tl.sum(tl.where((o_i == i)[:, None], -a33, 0.0), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a = b_a + tl.sum(b_a[:, None] * i33, 0)
            i33 = tl.where((o_i == i)[:, None], b_a, i33)
        i00 += m_I
        i11 += m_I
        i22 += m_I
        i33 += m_I

        # block merge: the full inverse
        i10 = -tl.dot(tl.dot(i11, a10, input_precision=SOLVE_PREC), i00, input_precision=SOLVE_PREC)
        i21 = -tl.dot(tl.dot(i22, a21, input_precision=SOLVE_PREC), i11, input_precision=SOLVE_PREC)
        i32 = -tl.dot(tl.dot(i33, a32, input_precision=SOLVE_PREC), i22, input_precision=SOLVE_PREC)
        i20 = -tl.dot(i22, tl.dot(a20, i00, input_precision=SOLVE_PREC) + tl.dot(a21, i10, input_precision=SOLVE_PREC),
                      input_precision=SOLVE_PREC)
        i31 = -tl.dot(i33, tl.dot(a31, i11, input_precision=SOLVE_PREC) + tl.dot(a32, i21, input_precision=SOLVE_PREC),
                      input_precision=SOLVE_PREC)
        i30 = -tl.dot(i33, tl.dot(a30, i00, input_precision=SOLVE_PREC) + tl.dot(a31, i10, input_precision=SOLVE_PREC)
                      + tl.dot(a32, i20, input_precision=SOLVE_PREC), input_precision=SOLVE_PREC)

        # the inverse in A's dtype (fla stores it and reads it back for w and u)
        dt = A.dtype.element_ty
        e00 = i00.to(dt)
        e10 = i10.to(dt)
        e11 = i11.to(dt)
        e20 = i20.to(dt)
        e21 = i21.to(dt)
        e22 = i22.to(dt)
        e30 = i30.to(dt)
        e31 = i31.to(dt)
        e32 = i32.to(dt)
        e33 = i33.to(dt)
        pA = A + (bos.to(tl.int64) * HV + i_hv) * BT
        tl.store(pA + (i_tc0 + o_i)[:, None] * (HV * BT) + o_i[None, :], e00, mask=m_tc0[:, None])
        tl.store(pA + (i_tc1 + o_i)[:, None] * (HV * BT) + o_i[None, :], e10, mask=m_tc1[:, None])
        tl.store(pA + (i_tc1 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :], e11, mask=m_tc1[:, None])
        tl.store(pA + (i_tc2 + o_i)[:, None] * (HV * BT) + o_i[None, :], e20, mask=m_tc2[:, None])
        tl.store(pA + (i_tc2 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :], e21, mask=m_tc2[:, None])
        tl.store(pA + (i_tc2 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :], e22, mask=m_tc2[:, None])
        tl.store(pA + (i_tc3 + o_i)[:, None] * (HV * BT) + o_i[None, :], e30, mask=m_tc3[:, None])
        tl.store(pA + (i_tc3 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :], e31, mask=m_tc3[:, None])
        tl.store(pA + (i_tc3 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :], e32, mask=m_tc3[:, None])
        tl.store(pA + (i_tc3 + o_i)[:, None] * (HV * BT) + (3 * BC + o_i)[None, :], e33, mask=m_tc3[:, None])




@triton.jit(do_not_specialize=["T"])
def wu_fwd_kernel(k, v, gc, gl, beta, A, w, u, cu_seqlens, chunk_indices, T,
                  H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                  GP: tl.constexpr, U_PREC: tl.constexpr, F3: tl.constexpr, qkv=None, doc=None, cw=None, rs=None,
                  L: tl.constexpr = None, C: tl.constexpr = None, KC: tl.constexpr = None, PACC: tl.constexpr = None,
                  PRO: tl.constexpr = False):
    """w = A (k beta e^g) and u = A (v beta) (fla's recompute_w_u_fwd_kernel, 64-column blocks) per
    (chunk, GP value heads of one key head). The forward and the backward's recompute both use this
    kernel, so the recomputed values are the forward's."""
    HV: tl.constexpr = H * G
    i_tg, i_p = tl.program_id(0), tl.program_id(1)
    i_h = (i_p * GP) // G
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    tk = bos.to(tl.int64) + o_t
    for j in tl.static_range(GP):
        i_hv = i_p * GP + j
        b_b = tl.load(beta + tk * HV + i_hv, mask=m_t, other=0.0)
        b_A = tl.load(A + tk[:, None] * (HV * BT) + i_hv * BT + o_A[None, :],
                      mask=m_t[:, None] & (o_A[None, :] <= o_A[:, None]), other=0.0)
        for i_v in tl.static_range(V // 64):
            o_v = i_v * 64 + tl.arange(0, 64)
            p_v = tk[:, None] * (HV * V) + i_hv * V + o_v[None, :]
            if PRO:  # v of value head i_hv, channels o_v: no norm
                b_v = _pro_tile(qkv, doc, cw, rs, tk, m_t, 2 * H * K + i_hv * V + o_v, 0, L, C, KC, 2 * H, False, False,
                                PACC, v.dtype.element_ty)
            else:
                b_v = tl.load(v + p_v, mask=m_t[:, None], other=0.0)
            b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
            b_u = tl.dot(b_A, b_vb, input_precision=U_PREC)
            tl.store(u + p_v, b_u.to(u.dtype.element_ty), mask=m_t[:, None])
        if F3:
            b_g = _f3_one(tl.load(gc + tk * HV + i_hv, mask=m_t, other=0.0), tl.load(gl + tk * HV + i_hv, mask=m_t, other=0.0))
        else:
            b_g = tl.math.exp2(tl.load(gc + tk * HV + i_hv, mask=m_t, other=0.0))
        for i_k in tl.static_range(K // 64):
            o_k = i_k * 64 + tl.arange(0, 64)
            if PRO:
                b_k = _pro_tile(qkv, doc, cw, rs, tk, m_t, H * K + i_h * K + o_k, H + i_h, L, C, KC, 2 * H, True, False,
                                PACC, k.dtype.element_ty)
            else:
                b_k = tl.load(k + tk[:, None] * (H * K) + i_h * K + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_kb = b_k * b_b[:, None]
            b_kb *= b_g[:, None]
            b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
            tl.store(w + tk[:, None] * (HV * K) + i_hv * K + o_k[None, :], b_w.to(w.dtype.element_ty), mask=m_t[:, None])


@triton.jit(do_not_specialize=["T"])
def state_fwd_kernel(k, u, w, v_new, gc, gl, h, cu_seqlens, chunk_offsets, T,
                     H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                     BV: tl.constexpr, ACC: tl.constexpr, F3: tl.constexpr, qkv=None, doc=None, cw=None, rs=None,
                     L: tl.constexpr = None, C: tl.constexpr = None, KC: tl.constexpr = None, PACC: tl.constexpr = None,
                     PRO: tl.constexpr = False):
    """The sequential chunk-state recurrence (fla's chunk_gated_delta_rule_fwd_kernel_h_blockdim64 for
    K = 128, no initial or final state): per (V block, segment, value head), h_c is stored before
    chunk c is absorbed, v_new = u - w h_c, h_{c+1} = e^{g_last} h_c + k^T (v_new * e^{g_last - g})."""
    HV: tl.constexpr = H * G
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    b_h1 = tl.zeros([64, BV], dtype=ACC)
    b_h2 = tl.zeros([64, BV], dtype=ACC)
    h += (boh.to(tl.int64) * HV + i_hv) * K * V
    u += (bos.to(tl.int64) * HV + i_hv) * V
    v_new += (bos.to(tl.int64) * HV + i_hv) * V
    k += (bos.to(tl.int64) * H + i_hv // G) * K
    w += (bos.to(tl.int64) * HV + i_hv) * K
    gc += bos.to(tl.int64) * HV + i_hv
    if F3:
        gl += bos.to(tl.int64) * HV + i_hv
    o_v = i_v * BV + tl.arange(0, BV)
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + o_k1
    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h = h + i_t * (HV * K * V)
        tl.store(p_h + o_k1[:, None] * V + o_v[None, :], b_h1.to(h.dtype.element_ty))
        tl.store(p_h + o_k2[:, None] * V + o_v[None, :], b_h2.to(h.dtype.element_ty))
        b_w = tl.load(w + o_t[:, None] * (HV * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        b_w = tl.load(w + o_t[:, None] * (HV * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
        b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        b_v = tl.load(u + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0) - b_v
        tl.store(v_new + o_t[:, None] * (HV * V) + o_v[None, :], b_v.to(v_new.dtype.element_ty), mask=m_t[:, None])
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(gc + last_idx * HV).to(ACC)
        b_g = tl.load(gc + o_t * HV, mask=m_t, other=0.0).to(ACC)
        if F3:
            b_l_last = tl.load(gl + last_idx * HV).to(ACC)
            b_l = tl.load(gl + o_t * HV, mask=m_t, other=0.0).to(ACC)
            b_v = b_v * tl.where(m_t, _f3_pair(b_g_last, b_g, b_l_last, b_l), 0)[:, None]
            b_g_last = _f3_one(b_g_last, b_l_last)
        else:
            b_v = b_v * tl.where(m_t, tl.math.exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.math.exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_v = b_v.to(k.dtype.element_ty)
        if PRO:  # k^T of key head i_hv // G, the first 64 channels, then the second
            b_k = tl.trans(_pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, H * K + (i_hv // G) * K + o_k1, H + i_hv // G, L, C,
                                     KC, 2 * H, True, False, PACC, k.dtype.element_ty))
        else:
            b_k = tl.load(k + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v)
        if PRO:
            b_k = tl.trans(_pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, H * K + (i_hv // G) * K + o_k2, H + i_hv // G, L, C,
                                     KC, 2 * H, True, False, PACC, k.dtype.element_ty))
        else:
            b_k = tl.load(k + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
        b_h2 += tl.dot(b_k, b_v)


@triton.jit(do_not_specialize=["T"])
def out_fwd_kernel(q, k, v_new, h, gc, gl, o, cu_seqlens, chunk_indices, T,
                   H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                   BV: tl.constexpr, ACC: tl.constexpr, F3: tl.constexpr, qkv=None, doc=None, cw=None, rs=None,
                   L: tl.constexpr = None, C: tl.constexpr = None, KC: tl.constexpr = None, PACC: tl.constexpr = None,
                   PRO: tl.constexpr = False):
    """o = (q h) e^g + (q k^T * decay, causal) v_new (fla's chunk_fwd_kernel_o, scale 1), per
    (chunk, key head): q k^T once for the group."""
    HV: tl.constexpr = H * G
    i_tg, i_h = tl.program_id(0), tl.program_id(1)
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_K = tl.arange(0, K)
    q += (bos.to(tl.int64) * H + i_h) * K
    k += (bos.to(tl.int64) * H + i_h) * K
    if PRO:  # q (scaled) and k^T of key head i_h, every channel
        b_q = _pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, i_h * K + o_K, i_h, L, C, KC, 2 * H, True, True, PACC,
                        q.dtype.element_ty)
        b_kt = tl.trans(_pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, H * K + i_h * K + o_K, H + i_h, L, C, KC, 2 * H, True,
                                  False, PACC, k.dtype.element_ty))
    else:
        b_q = tl.load(q + o_t[:, None] * (H * K) + o_K[None, :], mask=m_t[:, None], other=0.0)
        b_kt = tl.load(k + o_K[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
    b_qk = tl.dot(b_q, b_kt)
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    for j in tl.static_range(G):
        i_hv = i_h * G + j
        b_g = tl.load(gc + (bos.to(tl.int64) + o_t) * HV + i_hv, mask=m_t, other=0.0).to(ACC)
        if F3:
            b_l = tl.load(gl + (bos.to(tl.int64) + o_t) * HV + i_hv, mask=m_t, other=0.0).to(ACC)
            b_A = tl.where(m_A, b_qk * _f3_pair(b_g[:, None], b_g[None, :], b_l[:, None], b_l[None, :]), 0)
            b_ge = _f3_one(b_g, b_l)[:, None]
        else:
            b_A = tl.where(m_A, b_qk * tl.math.exp2(b_g[:, None] - b_g[None, :]), 0)
            b_ge = tl.math.exp2(b_g)[:, None]
        p_h = h + (i_tg.to(tl.int64) * HV + i_hv) * K * V
        for i_v in tl.static_range(V // BV):
            o_v = i_v * BV + tl.arange(0, BV)
            b_h = tl.load(p_h + o_K[:, None] * V + o_v[None, :])
            p_v = (bos.to(tl.int64) + o_t[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
            b_v = tl.load(v_new + p_v, mask=m_t[:, None], other=0.0)
            b_o = tl.dot(b_q, b_h) * b_ge
            b_o = b_o + tl.dot(b_A.to(b_v.dtype), b_v)
            tl.store(o + p_v, b_o.to(o.dtype.element_ty), mask=m_t[:, None])


# ------------------------------------------------------------------------------------------------
# Backward


@triton.jit(do_not_specialize=["T"])
def dv_local_kernel(q, k, gc, gl, do, dv, cu_seqlens, chunk_indices, T,
                    H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                    BV: tl.constexpr, ACC: tl.constexpr, F3: tl.constexpr, qkv=None, doc=None, cw=None, rs=None,
                    L: tl.constexpr = None, C: tl.constexpr = None, KC: tl.constexpr = None, PACC: tl.constexpr = None,
                    PRO: tl.constexpr = False):
    """dv_local = (k q^T * decay, anti-causal) do (fla's chunk_bwd_kernel_dv_local, scale 1), k q^T
    once per key head."""
    HV: tl.constexpr = H * G
    i_tg, i_h = tl.program_id(0), tl.program_id(1)
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_K = tl.arange(0, K)
    q += (bos.to(tl.int64) * H + i_h) * K
    k += (bos.to(tl.int64) * H + i_h) * K
    if PRO:  # k and q^T (scaled) of key head i_h, every channel
        b_k = _pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, H * K + i_h * K + o_K, H + i_h, L, C, KC, 2 * H, True, False,
                        PACC, k.dtype.element_ty)
        b_qt = tl.trans(_pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, i_h * K + o_K, i_h, L, C, KC, 2 * H, True, True, PACC,
                                  q.dtype.element_ty))
    else:
        b_k = tl.load(k + o_t[:, None] * (H * K) + o_K[None, :], mask=m_t[:, None], other=0.0)
        b_qt = tl.load(q + o_K[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
    b_kq = tl.dot(b_k, b_qt)
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    for j in tl.static_range(G):
        i_hv = i_h * G + j
        b_g = tl.load(gc + (bos.to(tl.int64) + o_t) * HV + i_hv, mask=m_t, other=0.0).to(ACC)
        if F3:
            b_l = tl.load(gl + (bos.to(tl.int64) + o_t) * HV + i_hv, mask=m_t, other=0.0).to(ACC)
            b_A = tl.where(m_A, b_kq * _f3_pair(b_g[None, :], b_g[:, None], b_l[None, :], b_l[:, None]), 0).to(do.dtype.element_ty)
        else:
            b_A = tl.where(m_A, b_kq * tl.math.exp2(b_g[None, :] - b_g[:, None]), 0).to(do.dtype.element_ty)
        for i_v in tl.static_range(V // BV):
            o_v = i_v * BV + tl.arange(0, BV)
            p = (bos.to(tl.int64) + o_t[:, None]) * (HV * V) + i_hv * V + o_v[None, :]
            b_do = tl.load(do + p, mask=m_t[:, None], other=0.0)
            tl.store(dv + p, tl.dot(b_A, b_do).to(dv.dtype.element_ty), mask=m_t[:, None])


@triton.jit(do_not_specialize=["T"])
def state_bwd_kernel(q, k, w, gc, gl, do, dv, dv2, dh, dh_ext, dv_ext, cu_seqlens, chunk_offsets, T,
                     H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                     BV: tl.constexpr, ACC: tl.constexpr, EXT: tl.constexpr, F3: tl.constexpr, qkv=None, doc=None,
                     cw=None, rs=None, L: tl.constexpr = None, C: tl.constexpr = None, KC: tl.constexpr = None,
                     PACC: tl.constexpr = None, PRO: tl.constexpr = False):
    """The sequential state-gradient recurrence, chunks in reverse (fla's
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 for K = 128, no final-state gradient, scale 1):
    dh_c is stored, dv2 = k dh_c e^{g_last - g} + dv_local, dh_{c-1} = e^{g_last} dh_c + (q e^g)^T do - w^T dv2.

    EXT (control (c)'s second-pass read): the chunk states h and v_new were also read by someone else,
    whose gradients arrive as dh_ext (1, NT, HV, K, V; dL/dh at chunk c's start) and dv_ext (1, T, HV, V;
    dL/dv_new): dv2 also adds dv_ext, and dh_ext[c] joins the running sum after chunk c, so the recurrence
    runs once with both sources. EXT False: the code above, unchanged."""
    HV: tl.constexpr = H * G
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    b_dh1 = tl.zeros([64, BV], dtype=ACC)
    b_dh2 = tl.zeros([64, BV], dtype=ACC)
    q += (bos.to(tl.int64) * H + i_hv // G) * K
    k += (bos.to(tl.int64) * H + i_hv // G) * K
    w += (bos.to(tl.int64) * HV + i_hv) * K
    do += (bos.to(tl.int64) * HV + i_hv) * V
    dv += (bos.to(tl.int64) * HV + i_hv) * V
    dv2 += (bos.to(tl.int64) * HV + i_hv) * V
    dh += (boh.to(tl.int64) * HV + i_hv) * K * V
    if EXT:
        dh_ext += (boh.to(tl.int64) * HV + i_hv) * K * V
        dv_ext += (bos.to(tl.int64) * HV + i_hv) * V
    gc += bos.to(tl.int64) * HV + i_hv
    if F3:
        gl += bos.to(tl.int64) * HV + i_hv
    o_v = i_v * BV + tl.arange(0, BV)
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + o_k1
    for i_s in range(NT):  # chunks in reverse; an ascending loop index lets Triton pipeline the loads
        i_t = NT - 1 - i_s
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_dh = dh + i_t * (HV * K * V)
        tl.store(p_dh + o_k1[:, None] * V + o_v[None, :], b_dh1.to(dh.dtype.element_ty))
        tl.store(p_dh + o_k2[:, None] * V + o_v[None, :], b_dh2.to(dh.dtype.element_ty))
        last_idx = min((i_t + 1) * BT, T) - 1
        bg_last = tl.load(gc + last_idx * HV).to(ACC)
        b_g = tl.load(gc + o_t * HV, mask=m_t, other=0.0).to(ACC)
        if F3:
            bl_last = tl.load(gl + last_idx * HV).to(ACC)
            b_l = tl.load(gl + o_t * HV, mask=m_t, other=0.0).to(ACC)
            bg_last_exp = _f3_one(bg_last, bl_last)
            b_g_exp = _f3_one(b_g, b_l)
        else:
            bg_last_exp = tl.math.exp2(bg_last)
            b_g_exp = tl.math.exp2(b_g)
        b_do = tl.load(do + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0)
        if PRO:  # k of key head i_hv // G, the first 64 channels, then the second
            b_k = _pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, H * K + (i_hv // G) * K + o_k1, H + i_hv // G, L, C, KC, 2 * H,
                            True, False, PACC, k.dtype.element_ty)
        else:
            b_k = tl.load(k + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))
        if PRO:
            b_k = _pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, H * K + (i_hv // G) * K + o_k2, H + i_hv // G, L, C, KC, 2 * H,
                            True, False, PACC, k.dtype.element_ty)
        else:
            b_k = tl.load(k + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
        b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
        if F3:
            b_dv *= tl.where(m_t, _f3_pair(bg_last, b_g, bl_last, b_l), 0)[:, None]
        else:
            b_dv *= tl.where(m_t, tl.math.exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(dv + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0)
        if EXT:
            b_dv += tl.load(dv_ext + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0).to(ACC)
        tl.store(dv2 + o_t[:, None] * (HV * V) + o_v[None, :], b_dv.to(dv2.dtype.element_ty), mask=m_t[:, None])
        b_w = tl.load(w + o_k1[:, None] + o_t[None, :] * (HV * K), mask=m_t[None, :], other=0.0)
        if PRO:  # q^T (scaled) of key head i_hv // G, the first 64 channels, then the second
            b_q = tl.trans(_pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, (i_hv // G) * K + o_k1, i_hv // G, L, C, KC, 2 * H,
                                     True, True, PACC, q.dtype.element_ty))
        else:
            b_q = tl.load(q + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
        b_dh1 *= bg_last_exp
        b_q = b_q * b_g_exp[None, :]
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))
        b_w = tl.load(w + o_k2[:, None] + o_t[None, :] * (HV * K), mask=m_t[None, :], other=0.0)
        if PRO:
            b_q = tl.trans(_pro_tile(qkv, doc, cw, rs, bos + o_t, m_t, (i_hv // G) * K + o_k2, i_hv // G, L, C, KC, 2 * H,
                                     True, True, PACC, q.dtype.element_ty))
        else:
            b_q = tl.load(q + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
        b_dh2 *= bg_last_exp
        b_q = b_q * b_g_exp[None, :]
        b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))
        if EXT:
            p_dhx = dh_ext + i_t * (HV * K * V)
            b_dh1 += tl.load(p_dhx + o_k1[:, None] * V + o_v[None, :]).to(ACC)
            b_dh2 += tl.load(p_dhx + o_k2[:, None] * V + o_v[None, :]).to(ACC)


@triton.jit(do_not_specialize=["T"])
def dqkwg_kernel(q, k, v_new, gc, gl, h, do, dh, dv, dq, dk_part, dw, dg_part, cu_seqlens, chunk_indices, T,
                 H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, BV: tl.constexpr, ACC: tl.constexpr, F3: tl.constexpr, qkv=None, doc=None, cw=None,
                 rs=None, L: tl.constexpr = None, C: tl.constexpr = None, KC: tl.constexpr = None,
                 PACC: tl.constexpr = None, PRO: tl.constexpr = False):
    """fla's chunk_bwd_kernel_dqkwg (scale 1) per (chunk, key head): K blocks in turn, and for each the
    group's value heads in turn, so only one K block's accumulators are live; dq (final) and the
    dqkwg part of dk are summed over the group in registers. Per value head: dw (stored negated, as
    fla) and the dqkwg part of dg (before the reverse cumsum)."""
    HV: tl.constexpr = H * G
    i_tg, i_h = tl.program_id(0), tl.program_id(1)
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    tk = bos.to(tl.int64) + o_t
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    last = min(i_t * BT + BT, T) - 1
    # per value head of the group: dg (over K blocks) and dg_last (over K blocks)
    b_dg0 = tl.zeros([BT], dtype=ACC)
    b_dg1 = tl.zeros([BT], dtype=ACC)
    b_dgl0 = tl.zeros([1], dtype=ACC)
    b_dgl1 = tl.zeros([1], dtype=ACC)
    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        if PRO:  # q (scaled) and k of key head i_h, channels o_k
            b_q = _pro_tile(qkv, doc, cw, rs, tk, m_t, i_h * K + o_k, i_h, L, C, KC, 2 * H, True, True, PACC,
                            q.dtype.element_ty)
            b_k = _pro_tile(qkv, doc, cw, rs, tk, m_t, H * K + i_h * K + o_k, H + i_h, L, C, KC, 2 * H, True, False, PACC,
                            k.dtype.element_ty)
        else:
            b_q = tl.load(q + tk[:, None] * (H * K) + i_h * K + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_k = tl.load(k + tk[:, None] * (H * K) + i_h * K + o_k[None, :], mask=m_t[:, None], other=0.0)
        b_dq_sum = tl.zeros([BT, BK], dtype=ACC)
        b_dk_sum = tl.zeros([BT, BK], dtype=ACC)
        for j in tl.static_range(G):
            i_hv = i_h * G + j
            p_h = h + (i_tg.to(tl.int64) * HV + i_hv) * K * V
            p_dh = dh + (i_tg.to(tl.int64) * HV + i_hv) * K * V
            b_dgl = tl.zeros([1], dtype=ACC)
            b_dq = tl.zeros([BT, BK], dtype=ACC)
            b_dk = tl.zeros([BT, BK], dtype=ACC)
            b_ds = tl.zeros([BT, BT], dtype=ACC)
            b_dw = tl.zeros([BT, BK], dtype=ACC)
            for i_v in range(V // BV):
                o_v = i_v * BV + tl.arange(0, BV)
                p_tv = tk[:, None] * (HV * V) + i_hv * V + o_v[None, :]
                b_v = tl.load(v_new + p_tv, mask=m_t[:, None], other=0.0)
                b_do = tl.load(do + p_tv, mask=m_t[:, None], other=0.0)
                b_dv = tl.load(dv + p_tv, mask=m_t[:, None], other=0.0)
                b_hb = tl.load(p_h + o_v[:, None] + o_k[None, :] * V)
                b_dhb = tl.load(p_dh + o_v[:, None] + o_k[None, :] * V)
                b_dgl += tl.sum(b_hb * b_dhb)
                b_ds += tl.dot(b_do, tl.trans(b_v))
                b_dq += tl.dot(b_do, b_hb.to(b_do.dtype))
                b_dk += tl.dot(b_v, b_dhb.to(b_v.dtype))
                b_dw += tl.dot(b_dv.to(b_v.dtype), b_hb.to(b_v.dtype))
            tl.store(dw + tk[:, None] * (HV * K) + i_hv * K + o_k[None, :], -b_dw.to(dw.dtype.element_ty), mask=m_t[:, None])
            b_g = tl.load(gc + tk * HV + i_hv, mask=m_t, other=0.0).to(ACC)
            b_g_last = tl.load(gc + (bos.to(tl.int64) + last) * HV + i_hv).to(ACC)
            if F3:
                b_l = tl.load(gl + tk * HV + i_hv, mask=m_t, other=0.0).to(ACC)
                b_l_last = tl.load(gl + (bos.to(tl.int64) + last) * HV + i_hv).to(ACC)
                b_dgl *= _f3_one(b_g_last, b_l_last)
                b_dq = b_dq * _f3_one(b_g, b_l)[:, None]
                b_dk = b_dk * tl.where(m_t, _f3_pair(b_g_last, b_g, b_l_last, b_l), 0)[:, None]
                b_dgl += tl.sum(b_dk * b_k)
                b_ds = tl.where(m_A, b_ds * _f3_pair(b_g[:, None], b_g[None, :], b_l[:, None], b_l[None, :]), 0)
            else:
                b_dgl *= tl.math.exp2(b_g_last)
                b_dq = b_dq * tl.math.exp2(b_g)[:, None]
                b_dk = b_dk * tl.where(m_t, tl.math.exp2(-b_g + b_g_last), 0)[:, None]
                b_dgl += tl.sum(b_dk * b_k)
                b_ds = tl.where(m_A, b_ds * tl.math.exp2(b_g[:, None] - b_g[None, :]), 0)
            b_ds = b_ds.to(b_k.dtype)
            b_dq += tl.dot(b_ds, b_k)
            b_dk += tl.dot(tl.trans(b_ds), b_q)
            b_dgk = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
            if j == 0:
                b_dg0 += b_dgk
                b_dgl0 += b_dgl
            else:
                b_dg1 += b_dgk
                b_dgl1 += b_dgl
            b_dq_sum += b_dq
            b_dk_sum += b_dk
        tl.store(dq + tk[:, None] * (H * K) + i_h * K + o_k[None, :], b_dq_sum.to(dq.dtype.element_ty), mask=m_t[:, None])
        tl.store(dk_part + tk[:, None] * (H * K) + i_h * K + o_k[None, :], b_dk_sum.to(dk_part.dtype.element_ty),
                 mask=m_t[:, None])
    b_dg0 = tl.where(o_t < last, b_dg0, b_dg0 + b_dgl0)
    tl.store(dg_part + tk * HV + i_h * G, b_dg0.to(dg_part.dtype.element_ty), mask=m_t)
    if G > 1:
        b_dg1 = tl.where(o_t < last, b_dg1, b_dg1 + b_dgl1)
        tl.store(dg_part + tk * HV + i_h * G + 1, b_dg1.to(dg_part.dtype.element_ty), mask=m_t)


@triton.jit(do_not_specialize=["T"])
def wy_bwd_kernel(k, v, beta, gc, gl, A, dw, du, dg_part, dk_wy, dv, db, dg, cu_seqlens, chunk_indices, T,
                  H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                  GP: tl.constexpr, ACC: tl.constexpr, F3: tl.constexpr, qkv=None, doc=None, cw=None, rs=None,
                  L: tl.constexpr = None, C: tl.constexpr = None, KC: tl.constexpr = None, PACC: tl.constexpr = None,
                  PRO: tl.constexpr = False):
    """fla's prepare_wy_repr_bwd_kernel per (chunk, GP value heads of one key head), K = V = 128 in
    64-column blocks, fused with the end of the backward: the WY part of dk per value head (dk_wy; the
    group sum and dqkwg's part are added by dk_sum_kernel); dg = dqkwg's part + the WY part, then
    fla's chunk-local reverse cumsum. The first dk term (A^T dw * beta e^g) is recomputed in the second
    pass instead of being stored and re-read. K K^T once per program."""
    HV: tl.constexpr = H * G
    i_tg, i_p = tl.program_id(0), tl.program_id(1)
    i_h = (i_p * GP) // G
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_A = tl.arange(0, BT)
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    tk = bos.to(tl.int64) + o_t  # token index
    p_k = k + tk[:, None] * (H * K) + i_h * K
    b_kk = tl.zeros([BT, BT], dtype=ACC)  # K K^T
    for i_k in range(K // 64):
        o_k = i_k * 64 + tl.arange(0, 64)
        if PRO:  # k of key head i_h, channels o_k (here and in the two passes below)
            b_k = _pro_tile(qkv, doc, cw, rs, tk, m_t, H * K + i_h * K + o_k, H + i_h, L, C, KC, 2 * H, True, False, PACC,
                            k.dtype.element_ty)
        else:
            b_k = tl.load(p_k + o_k[None, :], mask=m_t[:, None], other=0.0)
        b_kk += tl.dot(b_k, tl.trans(b_k))
    for j in tl.static_range(GP):
        i_hv = i_p * GP + j
        b_b = tl.load(beta + tk * HV + i_hv, mask=m_t, other=0.0)
        b_db = tl.zeros([BT], dtype=ACC)
        # A^T; only the lower triangle of A is ever written (the buffer is not zero-filled)
        b_A = tl.load(A + (bos.to(tl.int64) * HV + i_hv) * BT + o_A[:, None] + o_t[None, :] * (HV * BT),
                      mask=(o_A[:, None] <= o_A[None, :]) & m_t[None, :], other=0.0)
        b_dA = tl.zeros([BT, BT], dtype=ACC)
        b_g = tl.load(gc + tk * HV + i_hv, mask=m_t, other=0.0).to(ACC)
        if F3:
            b_l = tl.load(gl + tk * HV + i_hv, mask=m_t, other=0.0).to(ACC)
            b_g_exp = _f3_one(b_g, b_l)
        else:
            b_g_exp = tl.math.exp2(b_g)
        b_dg = tl.zeros([BT], dtype=ACC)
        p_dw = dw + tk[:, None] * (HV * K) + i_hv * K
        # pass 1 over K: dA from dw, and the db / dg terms of A^T dw
        for i_k in range(K // 64):
            o_k = i_k * 64 + tl.arange(0, 64)
            if PRO:
                b_k = _pro_tile(qkv, doc, cw, rs, tk, m_t, H * K + i_h * K + o_k, H + i_h, L, C, KC, 2 * H, True, False,
                                PACC, k.dtype.element_ty)
            else:
                b_k = tl.load(p_k + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_dw = tl.load(p_dw + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_kbg = b_k * (b_b * b_g_exp)[:, None]
            b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
            b_dkbg = tl.dot(b_A, b_dw)
            b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
            b_dg += tl.sum(b_dkbg * b_kbg, 1)
        # over V: dA from du, and dv
        for i_v in range(V // 64):
            o_v = i_v * 64 + tl.arange(0, 64)
            p_tv = tk[:, None] * (HV * V) + i_hv * V + o_v[None, :]
            if PRO:  # v of value head i_hv, channels o_v: no norm
                b_v = _pro_tile(qkv, doc, cw, rs, tk, m_t, 2 * H * K + i_hv * V + o_v, 0, L, C, KC, 2 * H, False, False,
                                PACC, v.dtype.element_ty)
            else:
                b_v = tl.load(v + p_tv, mask=m_t[:, None], other=0.0)
            b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
            b_du = tl.load(du + p_tv, mask=m_t[:, None], other=0.0)
            b_dA += tl.dot(b_du, tl.trans(b_vb))
            b_dvb = tl.dot(b_A, b_du)
            b_dv = b_dvb * b_b[:, None]
            b_db += tl.sum(b_dvb * b_v, 1)
            tl.store(dv + p_tv, b_dv.to(dv.dtype.element_ty), mask=m_t[:, None])
        b_dA = tl.where(m_A, b_dA, 0)
        b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
        b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
        if F3:
            b_dA *= _f3_pair(b_g[:, None], b_g[None, :], b_l[:, None], b_l[None, :])
        else:
            b_dA *= tl.math.exp2(b_g[:, None] - b_g[None, :])
        b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)
        # pass 2 over K: the rest of dk and db
        for i_k in range(K // 64):
            o_k = i_k * 64 + tl.arange(0, 64)
            if PRO:
                b_k = _pro_tile(qkv, doc, cw, rs, tk, m_t, H * K + i_h * K + o_k, H + i_h, L, C, KC, 2 * H, True, False,
                                PACC, k.dtype.element_ty)
            else:
                b_k = tl.load(p_k + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_kb = b_k * b_b[:, None]
            b_dkb = tl.dot(b_dA, b_k)
            b_db += tl.sum(b_dkb * b_k, 1)
            b_t = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb).to(b_dA.dtype), b_dA))
            b_dw = tl.load(p_dw + o_k[None, :], mask=m_t[:, None], other=0.0)
            b_t += tl.dot(b_A, b_dw) * (b_g_exp * b_b)[:, None]
            tl.store(dk_wy + tk[:, None] * (HV * K) + i_hv * K + o_k[None, :], b_t.to(dk_wy.dtype.element_ty),
                     mask=m_t[:, None])
        tl.store(db + tk * HV + i_hv, b_db.to(db.dtype.element_ty), mask=m_t)
        b_AdA = b_dA * (b_kk * b_b[:, None])
        b_dg += tl.sum(b_AdA, 1) - tl.sum(b_AdA, 0)
        # dg: dqkwg's part + this, then the chunk-local reverse cumsum
        b_s = tl.load(dg_part + tk * HV + i_hv, mask=m_t, other=0.0).to(ACC)
        b_s = tl.where(m_t, b_s + b_dg, 0)
        b_o = tl.cumsum(b_s, axis=0)
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z + b_s
        tl.store(dg + tk * HV + i_hv, b_o.to(dg.dtype.element_ty), mask=m_t)


@triton.jit
def dk_sum_kernel(dk_part, dk_wy, dk, TOTAL, H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
                  ACC: tl.constexpr):
    """dk = dqkwg's part (summed over the group) + the WY parts summed over the group (fla's order)."""
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
    m_t = o_t < TOTAL
    o_k = tl.arange(0, K)
    b_w = tl.zeros([BT, K], dtype=ACC)
    for j in tl.static_range(G):
        b_w += tl.load(dk_wy + o_t[:, None] * (H * G * K) + (i_h * G + j) * K + o_k[None, :], mask=m_t[:, None], other=0.0).to(ACC)
    p = o_t[:, None] * (H * K) + i_h * K + o_k[None, :]
    b_d = tl.load(dk_part + p, mask=m_t[:, None], other=0.0).to(ACC) + b_w
    tl.store(dk + p, b_d.to(dk.dtype.element_ty), mask=m_t[:, None])


# ------------------------------------------------------------------------------------------------
# Control (c)'s second-pass read: o2 = (q2 h) e^g +
# (q2 k^T * decay, causal) v_new over pass 1's chunk states h and v_new (out_fwd_kernel with q = q2), and
# its backward (with dv_local_kernel for dL/dv_new). Pass 1's own backward (state_bwd_kernel with EXT)
# takes dh_ext and dv_ext from there.


@triton.jit(do_not_specialize=["T"])
def read_dqkg_kernel(q, k, v_new, gc, gl, h, do, dq, dk, dg, dh, cu_seqlens, chunk_indices, T,
                     H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr, ACC: tl.constexpr, F3: tl.constexpr):
    """The read's backward except dL/dv_new (dv_local_kernel with q = q2 gives that), per (chunk, key head):
    dqkwg_kernel's output part (no state write, so no dw and no state-gradient terms): dq = (do h^T) e^g +
    ds k, dk = ds^T q with ds = (do v_new^T * decay, causal), dq and dk summed over the group in registers;
    per value head dg = the chunk-local reverse cumsum of sum(dq q) - sum(dk k) (G <= 2), and
    dh_c = (q e^g)^T do (state_bwd_kernel's source term, its arithmetic): dL/dh at the chunk's start."""
    HV: tl.constexpr = H * G
    i_tg, i_h = tl.program_id(0), tl.program_id(1)
    bos, T, i_t = _chunk_bounds(cu_seqlens, chunk_indices, i_tg)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    tk = bos.to(tl.int64) + o_t
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_dg0 = tl.zeros([BT], dtype=ACC)
    b_dg1 = tl.zeros([BT], dtype=ACC)
    for i_k in range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        b_q = tl.load(q + tk[:, None] * (H * K) + i_h * K + o_k[None, :], mask=m_t[:, None], other=0.0)
        b_k = tl.load(k + tk[:, None] * (H * K) + i_h * K + o_k[None, :], mask=m_t[:, None], other=0.0)
        b_dq_sum = tl.zeros([BT, BK], dtype=ACC)
        b_dk_sum = tl.zeros([BT, BK], dtype=ACC)
        for j in tl.static_range(G):
            i_hv = i_h * G + j
            p_h = h + (i_tg.to(tl.int64) * HV + i_hv) * K * V
            p_dh = dh + (i_tg.to(tl.int64) * HV + i_hv) * K * V
            b_g = tl.load(gc + tk * HV + i_hv, mask=m_t, other=0.0).to(ACC)
            if F3:
                b_l = tl.load(gl + tk * HV + i_hv, mask=m_t, other=0.0).to(ACC)
                b_qg = b_q * _f3_one(b_g, b_l)[:, None]
            else:
                b_qg = b_q * tl.math.exp2(b_g)[:, None]
            b_dq = tl.zeros([BT, BK], dtype=ACC)
            b_ds = tl.zeros([BT, BT], dtype=ACC)
            for i_v in range(V // BV):
                o_v = i_v * BV + tl.arange(0, BV)
                p_tv = tk[:, None] * (HV * V) + i_hv * V + o_v[None, :]
                b_v = tl.load(v_new + p_tv, mask=m_t[:, None], other=0.0)
                b_do = tl.load(do + p_tv, mask=m_t[:, None], other=0.0)
                b_hb = tl.load(p_h + o_v[:, None] + o_k[None, :] * V)
                b_ds += tl.dot(b_do, tl.trans(b_v))
                b_dq += tl.dot(b_do, b_hb.to(b_do.dtype))
                b_dh = tl.dot(tl.trans(b_qg), b_do.to(b_qg.dtype))
                tl.store(p_dh + o_k[:, None] * V + o_v[None, :], b_dh.to(dh.dtype.element_ty))
            if F3:
                b_dq = b_dq * _f3_one(b_g, b_l)[:, None]
                b_ds = tl.where(m_A, b_ds * _f3_pair(b_g[:, None], b_g[None, :], b_l[:, None], b_l[None, :]), 0)
            else:
                b_dq = b_dq * tl.math.exp2(b_g)[:, None]
                b_ds = tl.where(m_A, b_ds * tl.math.exp2(b_g[:, None] - b_g[None, :]), 0)
            b_ds = b_ds.to(b_k.dtype)
            b_dq += tl.dot(b_ds, b_k)
            b_dk = tl.dot(tl.trans(b_ds), b_q)
            b_dgk = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
            if j == 0:
                b_dg0 += b_dgk
            else:
                b_dg1 += b_dgk
            b_dq_sum += b_dq
            b_dk_sum += b_dk
        tl.store(dq + tk[:, None] * (H * K) + i_h * K + o_k[None, :], b_dq_sum.to(dq.dtype.element_ty), mask=m_t[:, None])
        tl.store(dk + tk[:, None] * (H * K) + i_h * K + o_k[None, :], b_dk_sum.to(dk.dtype.element_ty), mask=m_t[:, None])
    # g enters as the chunk-local cumulative sum: dg = reverse cumsum (wy_bwd_kernel's arithmetic)
    b_s = tl.where(m_t, b_dg0, 0)
    b_o = -tl.cumsum(b_s, axis=0) + tl.sum(b_s, axis=0) + b_s
    tl.store(dg + tk * HV + i_h * G, b_o.to(dg.dtype.element_ty), mask=m_t)
    if G > 1:
        b_s = tl.where(m_t, b_dg1, 0)
        b_o = -tl.cumsum(b_s, axis=0) + tl.sum(b_s, axis=0) + b_s
        tl.store(dg + tk * HV + i_h * G + 1, b_o.to(dg.dtype.element_ty), mask=m_t)


# ------------------------------------------------------------------------------------------------
# GDN input preparation: causal depthwise short conv within documents, SiLU, split into q, k, v,
# L2 norm of q and k, q scaled (ced/model/gdn.py, the training path), forward and backward.


@triton.jit
def _prep_row(x, doc, w, o_r, ok_r, pos_r, o_c, d_r, C: tl.constexpr, KC: tl.constexpr, ACC: tl.constexpr):
    """Pre-activation conv output a[r, c] = sum_j x[r - j, c] w[c, KC-1-j] over taps whose input is in
    row r's document (and row), plus the per-tap inputs (for the weight gradient)."""
    a = tl.zeros([o_r.shape[0], o_c.shape[0]], dtype=ACC)
    for j in tl.static_range(KC):
        src = o_r - j
        ok = ok_r & (pos_r >= j)
        if j > 0:
            ok = ok & (tl.load(doc + src, mask=ok, other=-1) == d_r)
        xj = tl.load(x + src.to(tl.int64)[:, None] * C + o_c[None, :], mask=ok[:, None], other=0.0).to(ACC)
        wj = tl.load(w + o_c * KC + (KC - 1 - j)).to(ACC)
        a += xj * wj[None, :]
    return a


@triton.jit
def _pro_tile(qkv, doc, cw, rs, tok, m_t, o_c, slot, L: tl.constexpr, C: tl.constexpr, KC: tl.constexpr,
              RS_N: tl.constexpr, NORM: tl.constexpr, QS: tl.constexpr, ACC: tl.constexpr, OUT: tl.constexpr):
    """C's prologue (PRO): q, k or v at tokens `tok` (absolute indices of the flattened rows; rows m_t) and channels o_c
    of the in_proj_qkv output qkv (B*L, C), as qkv_prep's forward stores it (prep_fwd_kernel, statement for statement):
    the conv taps within the token's document and row (_prep_row, the same helper and order), SiLU, and for q and k
    (NORM) the L2 norm's factor rs[tok, slot] that prep_rs_kernel stored (prep_fwd_kernel's rsqrt of its row sum),
    then q's scale (QS); rounded to OUT (the kernel's input dtype) as prep_fwd_kernel's store rounds. Rows outside m_t
    have every tap masked: a = 0, so the tile's row is exactly 0, the value of the masked load it replaces."""
    pos = tok % L
    d_r = tl.load(doc + tok, mask=m_t, other=-1)
    a = _prep_row(qkv, doc, cw, tok, m_t, pos, o_c, d_r, C, KC, ACC)
    y = a * tl.sigmoid(a)
    if NORM:  # prep_fwd_kernel: n = y * rsqrt(...)[:, None], then n * Q_SCALE for q
        y = y * tl.load(rs + tok.to(tl.int64) * RS_N + slot, mask=m_t, other=0.0).to(ACC)[:, None]
        if QS:
            y = y * tl.full([1], PRO_Q_SCALE, ACC)
    return y.to(OUT)


@triton.jit(do_not_specialize=["TOTAL", "L"])
def prep_rs_kernel(x, doc, w, rs, TOTAL, L, C: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, KC: tl.constexpr,
                   BT: tl.constexpr, ACC: tl.constexpr, EPS: tl.constexpr):
    """C (GDNConfig.edges "fused"): the L2 norm's per-row factor of q and k, rs[t, i_c] = rsqrt(sum(y^2) + eps) for
    head slot i_c (q heads, then k heads), stored in ACC. prep_fwd_kernel's statements up to its rsqrt, unchanged, with
    the same grid over blocks of BT tokens and the same launch options (the wrappers pass prep_fwd_kernel's), so the
    row sum is formed as qkv_prep forms it; the chunk kernels' prologues (_pro_tile) multiply by it."""
    i_t, i_c = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < TOTAL
    pos = o_t % L
    d0 = tl.load(doc + o_t, mask=m_t, other=-1)
    o_c = i_c * D + tl.arange(0, D)
    a = _prep_row(x, doc, w, o_t, m_t, pos, o_c, d0, C, KC, ACC)
    y = a * tl.sigmoid(a)
    r = tl.math.rsqrt(tl.sum(y * y, 1) + tl.full([1], EPS, ACC))
    tl.store(rs + o_t.to(tl.int64) * (2 * HK) + i_c, r.to(rs.dtype.element_ty), mask=m_t)


@triton.jit(do_not_specialize=["TOTAL", "L"])
def prep_fwd_kernel(x, doc, w, q, k, v, TOTAL, L, C: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
                    D: tl.constexpr, KC: tl.constexpr, BT: tl.constexpr, ACC: tl.constexpr,
                    Q_SCALE: tl.constexpr, EPS: tl.constexpr):
    """Per (block of BT tokens, one 128-channel head slot: q heads, then k heads, then v heads)."""
    i_t, i_c = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < TOTAL
    pos = o_t % L
    d0 = tl.load(doc + o_t, mask=m_t, other=-1)
    o_c = i_c * D + tl.arange(0, D)
    a = _prep_row(x, doc, w, o_t, m_t, pos, o_c, d0, C, KC, ACC)
    y = a * tl.sigmoid(a)
    o_d = tl.arange(0, D)
    if i_c < 2 * HK:
        n = y * tl.math.rsqrt(tl.sum(y * y, 1) + tl.full([1], EPS, ACC))[:, None]
        if i_c < HK:
            n = n * tl.full([1], Q_SCALE, ACC)
            tl.store(q + (o_t.to(tl.int64)[:, None] * HK + i_c) * D + o_d[None, :], n.to(q.dtype.element_ty), mask=m_t[:, None])
        else:
            tl.store(k + (o_t.to(tl.int64)[:, None] * HK + i_c - HK) * D + o_d[None, :], n.to(k.dtype.element_ty),
                     mask=m_t[:, None])
    else:
        tl.store(v + (o_t.to(tl.int64)[:, None] * HV + i_c - 2 * HK) * D + o_d[None, :], y.to(v.dtype.element_ty),
                 mask=m_t[:, None])


@triton.jit
def _prep_da(x, doc, w, dq, dk, dv, o_r, ok_r, pos_r, o_c, d_r, i_c, C: tl.constexpr, KC: tl.constexpr,
             HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, ACC: tl.constexpr, Q_SCALE: tl.constexpr,
             EPS: tl.constexpr):
    """dL/da at rows o_r (zero where not ok_r): the conv output's gradient through SiLU and, for q / k,
    the L2 norm (and q's scale)."""
    a = _prep_row(x, doc, w, o_r, ok_r, pos_r, o_c, d_r, C, KC, ACC)
    s = tl.sigmoid(a)
    y = a * s
    o_d = tl.arange(0, D)
    if i_c < 2 * HK:
        if i_c < HK:
            dn = tl.load(dq + (o_r.to(tl.int64)[:, None] * HK + i_c) * D + o_d[None, :], mask=ok_r[:, None], other=0.0).to(ACC)
            dn = dn * tl.full([1], Q_SCALE, ACC)
        else:
            dn = tl.load(dk + (o_r.to(tl.int64)[:, None] * HK + i_c - HK) * D + o_d[None, :], mask=ok_r[:, None],
                         other=0.0).to(ACC)
        rs = tl.math.rsqrt(tl.sum(y * y, 1) + tl.full([1], EPS, ACC))
        n = y * rs[:, None]
        dy = rs[:, None] * (dn - n * tl.sum(dn * n, 1)[:, None])
    else:
        dy = tl.load(dv + (o_r.to(tl.int64)[:, None] * HV + i_c - 2 * HK) * D + o_d[None, :], mask=ok_r[:, None],
                     other=0.0).to(ACC)
    da = dy * (s + y * (1 - s))
    return tl.where(ok_r[:, None], da, 0)


@triton.jit(do_not_specialize=["TOTAL", "L"])
def prep_da_kernel(x, doc, w, dq, dk, dv, da, dw_part, TOTAL, L, C: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
                   D: tl.constexpr, KC: tl.constexpr, BT: tl.constexpr, ACC: tl.constexpr,
                   Q_SCALE: tl.constexpr, EPS: tl.constexpr):
    """Backward, part 1, per (block of BT tokens, head slot): da = dL/d(conv output) through SiLU and
    the L2 norm (stored), and this block's partial sums of the conv weight gradient,
    dw_part[i_t, KC-1-j, c] = sum_t da[t, c] x[t - j, c] over taps within t's document."""
    i_t, i_c = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < TOTAL
    pos = o_t % L
    d0 = tl.load(doc + o_t, mask=m_t, other=-1)
    o_c = i_c * D + tl.arange(0, D)
    b_da = _prep_da(x, doc, w, dq, dk, dv, o_t, m_t, pos, o_c, d0, i_c, C, KC, HK, HV, D, ACC, Q_SCALE, EPS)
    tl.store(da + o_t.to(tl.int64)[:, None] * C + o_c[None, :], b_da.to(da.dtype.element_ty), mask=m_t[:, None])
    for j in tl.static_range(KC):
        src = o_t - j
        ok = m_t & (pos >= j)
        if j > 0:
            ok = ok & (tl.load(doc + src, mask=ok, other=-1) == d0)
        xj = tl.load(x + src.to(tl.int64)[:, None] * C + o_c[None, :], mask=ok[:, None], other=0.0).to(ACC)
        tl.store(dw_part + (i_t.to(tl.int64) * KC + (KC - 1 - j)) * C + o_c, tl.sum(b_da * xj, 0))


@triton.jit(do_not_specialize=["TOTAL", "L"])
def prep_dx_kernel(da, doc, w, dx, TOTAL, L, C: tl.constexpr, D: tl.constexpr, KC: tl.constexpr, BT: tl.constexpr,
                   ACC: tl.constexpr):
    """Backward, part 2: dx[t, c] = sum_j da[t + j, c] w[c, KC-1-j] over rows t + j in t's document (and row)."""
    i_t, i_c = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < TOTAL
    pos = o_t % L
    d0 = tl.load(doc + o_t, mask=m_t, other=-1)
    o_c = i_c * D + tl.arange(0, D)
    b_dx = tl.zeros([BT, D], dtype=ACC)
    for j in tl.static_range(KC):
        o_r = o_t + j
        ok = m_t & (pos + j < L)
        if j > 0:
            ok = ok & (tl.load(doc + o_r, mask=ok, other=-2) == d0)
        b_da = tl.load(da + o_r.to(tl.int64)[:, None] * C + o_c[None, :], mask=ok[:, None], other=0.0).to(ACC)
        wj = tl.load(w + o_c * KC + (KC - 1 - j)).to(ACC)
        b_dx += b_da * wj[None, :]
    tl.store(dx + o_t.to(tl.int64)[:, None] * C + o_c[None, :], b_dx.to(dx.dtype.element_ty), mask=m_t[:, None])
