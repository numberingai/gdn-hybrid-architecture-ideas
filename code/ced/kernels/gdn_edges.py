"""C: the GDN layer's input preparation fused into our chunk kernels (GDNConfig.edges "fused").

What "fused" computes: exactly what kernel "ced" with its fused input preparation computes today (ced.kernels.gdn.qkv_prep
followed by ced.kernels.gdn.ced_delta_rule), forward and backward, with these differences in where values live:

- q, k and v are never written. A small kernel (gdn_triton.prep_rs_kernel: qkv_prep's forward statements up to its
  rsqrt, same grid and launch options) stores the L2 norm's per-row factor of q and k, (B*L, 2 H_k) floats; every chunk
  kernel that read q, k or v forms them in a prologue from the in_proj_qkv output u with qkv_prep's arithmetic (the conv
  taps in order, SiLU, the factor, the q scale) and rounds them where qkv_prep stores (gdn_triton.py, PRO). Only u, the
  factor, and what our kernel keeps anyway are saved for the backward (not q, k, v: 2 H_k + H_v heads x 128 values a
  token fewer).
- The backward's input-gradient chain is today's kernels, launched as qkv_prep's backward launches them: the chunk
  kernels' dq, dk, dv (their prologues read u), then prep_da_kernel (da in float32, as today), prep_dx_kernel, and the
  conv weight's gradient summed over blocks by the same PyTorch call.
- The varlen metadata is formed on the device from cu_seqlens at a capacity fixed by the shapes (NT_cap = ceil(B L /
  64) + N chunks for N segments; programs past the last chunk see an empty segment and store nothing): no host read, no
  .item(), grids from shapes only. Today's path reads cu_seqlens_cpu on the host.
- Two torch.library custom ops (ced::gdn_edges_fwd, ced::gdn_edges_bwd) with fake implementations and registered
  autograd, in place of the two torch.compiler.disable functions (qkv_prep, ced_delta_rule): the layer can compile as
  one graph. Everything that touches values (casts, sums) stays inside the ops, as inside today's disabled functions,
  so Inductor generates nothing new around them.

Bit for bit today's path by construction, except one point a CPU cannot settle: prep_rs_kernel's row sum is
prep_fwd_kernel's statement in a kernel with the same tile and launch options, and Triton's reduction order follows the
layout its compiler assigns. The GPU check (tests/gpu/test_gdn_edges_gpu.py) compares every output and gradient bit for
bit with today's path; under the interpreter (numpy) the order is the same by construction.

Not here (not bit for bit today's by construction): the gates (their backward ends in sums over every token,
Inductor's order today), the gated output norm (Inductor's row and weight-gradient reductions), the SiLU / L2-norm
backward in the chunk kernels' epilogues (two row reductions in another layout, and the conv weight's block sums),
in_proj_a and in_proj_b as one GEMM (one bf16 rounding of dx where today has two).

Scope: the one-pass training path (GatedDeltaNet.forward without keep, valid, state or the GDN-side arms), impl
"triton", keep "all". Elsewhere the model keeps today's path.

On a CPU without the Triton interpreter the ops run a PyTorch reference (prep_reference + the chunked delta rule), and
their saved outputs are placeholders (the reference keeps g itself where the kernel keeps its cumulative sum): it serves
the CPU tests of the ops' registration, autograd and compile tracing, not training.
"""

from __future__ import annotations

import os
import weakref

import torch
import torch.nn.functional as F

from ced.kernels import gdn as G_

CHUNK = G_.CHUNK
D_HEAD = 128  # our kernels' head width (K = V = 128)
# The last boundaries' chunk metadata: (weak reference to the cu_seqlens tensor, its version, T, (cu_ext, ci, co)). Every
# GDN layer of a forward gets the same cu_seqlens object, so the metadata is formed once a forward (cached_chunk_meta).
_META_CACHE: list = []
# How often the Triton implementation ran in this process, forward and backward (the GPU run's training
# legs need positive evidence that "fused" ran the fused kernels, not a silent fallback to today's path). With
# CED_EDGES_REPORT=1 the first call each way prints one line ("EDGES-RAN forward" / "EDGES-RAN backward"); otherwise
# nothing is printed. Counted in the ops' implementations, which run eagerly even when the op is traced.
RAN = {"forward": 0, "backward": 0}


def _ran(way: str, u: torch.Tensor, hk: int, hv: int) -> None:
    RAN[way] += 1
    if RAN[way] == 1 and os.environ.get("CED_EDGES_REPORT") == "1":
        print(f"EDGES-RAN {way} (first call: u {tuple(u.shape)} {u.dtype}, {hk} key / {hv} value heads, {u.device})",
              flush=True)


def _use_triton(x: torch.Tensor) -> bool:
    return x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


def _dtypes(u_dtype: torch.dtype, cw_dtype: torch.dtype, out_bf16: bool):
    """(comp, out, prep accumulation dtype, chunk accumulation dtype) as today's path forms them: qkv_prep computes in
    promote(u, conv_weight) and stores in out (bf16 for GDNConfig.input_dtype "bfloat16", else comp); our kernel
    accumulates in float64 for float64 inputs, else float32 (acc_dtypes(k))."""
    comp = torch.promote_types(u_dtype, cw_dtype)
    out = torch.bfloat16 if out_bf16 else comp
    prep_acc = torch.float64 if comp == torch.float64 else torch.float32
    acc = torch.float64 if out == torch.float64 else torch.float32
    return comp, out, prep_acc, acc


def chunk_capacity(T: int, n_segments: int) -> int:
    """An upper bound on the number of chunks of 64 over n_segments segments of T tokens in all (each segment adds at
    most one partial chunk): the grids of the chunk kernels, from shapes only."""
    return (T + CHUNK - 1) // CHUNK + n_segments


def chunk_meta(cu_seqlens: torch.Tensor, T: int):
    """The varlen metadata of our kernels on cu_seqlens' device, with fixed shapes and no host read.

    Returns (cu_ext (N + 2,) int32: the boundaries with one empty segment [T, T) appended; ci (NT_cap, 2) int32: (segment,
    chunk in segment) of every global chunk, chunks in ced.kernels.gdn.chunk_index's order, then (N, 0) (the empty
    segment) past the last chunk; co (N + 1,) int32: each segment's first global chunk, co[N] the chunk count; NT_cap).
    For the first NT entries ci equals chunk_index's (tests/test_gdn_edges.py)."""
    cu = cu_seqlens.to(torch.int32)
    N = cu.shape[0] - 1
    counts = (cu[1:] - cu[:-1] + (CHUNK - 1)) // CHUNK
    co = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0, dtype=torch.int32)])
    cap = chunk_capacity(T, N)
    c = torch.arange(cap, device=cu.device, dtype=torch.int32)
    seg = torch.searchsorted(co[1:].contiguous(), c, right=True, out_int32=True)  # N past the last chunk
    idx = torch.where(seg < N, c - co[seg], torch.zeros_like(c))
    ci = torch.stack([seg, idx], 1).contiguous()
    cu_ext = torch.cat([cu, cu[-1:]])
    return cu_ext, ci, co, cap


def cached_chunk_meta(cu_seqlens: torch.Tensor, T: int):
    """chunk_meta(cu_seqlens, T)[:3], formed once for a boundaries tensor: every GDN layer of a forward gets the same
    cu_seqlens object, so the 16 or so small launches run once a forward instead of once a layer (432 a 4B micro-batch,
    about 9 ms of host time at the profile's 20.9 us a launch). Keyed on the tensor itself (a weak reference, so a freed
    tensor never matches), its version counter and T. Outside traced code only: under torch.compile the metadata is
    formed in the graph (fused_delta_rule), where its shapes follow cu_seqlens' dynamic length."""
    if _META_CACHE:
        ref, version, t, meta = _META_CACHE[0]
        if ref() is cu_seqlens and version == cu_seqlens._version and t == T:
            return meta
    meta = chunk_meta(cu_seqlens, T)[:3]
    _META_CACHE[:] = [(weakref.ref(cu_seqlens), cu_seqlens._version, T, meta)]
    return meta


def prep_reference(u: torch.Tensor, doc: torch.Tensor, cw: torch.Tensor, hk: int, hv: int, d: int):
    """qkv_prep's function in PyTorch (the CPU reference of the ops): causal depthwise conv within documents and rows from
    an empty history, SiLU, split, L2 norm of q and k (eps 1e-6), q scaled by d^-0.5; in promote(u, cw)'s dtype."""
    B, L, C = u.shape
    KC = cw.shape[1]
    win = F.pad(u, (0, 0, KC - 1, 0)).unfold(1, KC, 1)  # (B, L, C, KC): entry i is position t - (KC-1) + i
    same = F.pad(doc, (KC - 1, 0), value=-1).unfold(1, KC, 1) == doc[..., None]
    a = F.silu((win * same[:, :, None, :].to(u.dtype) * cw).sum(-1))
    q, k, v = (t.reshape(B, L, -1, d) for t in a.split([hk * d, hk * d, hv * d], dim=-1))
    unit = lambda t: t * torch.rsqrt(t.pow(2).sum(-1, keepdim=True) + 1e-6)  # noqa: E731
    return unit(q) * d**-0.5, unit(k), v


def _reference(u, doc, cw, g, beta, cu_seqlens, hk, hv, out):
    """The ops' CPU reference: o (1, T, hv, d) in out from prep_reference and the chunked delta rule over the flattened
    rows (state reset at every boundary of cu_seqlens; value head j reads key head j // (hv / hk))."""
    from ced.model.gdn_chunked import chunked_delta_rule

    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    q, k, v = (t.to(out).reshape(1, T, -1, d) for t in prep_reference(u, doc, cw, hk, hv, d))
    starts = torch.zeros(1, T, dtype=torch.bool, device=u.device)
    starts[0, cu_seqlens[:-1].long()] = True
    rep = hv // hk
    o, _ = chunked_delta_rule(q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2), v, g.to(q.dtype),
                              beta.to(q.dtype), starts, q.new_zeros(1, hv, d, d), CHUNK)
    return o.to(out).contiguous()


def _pro_args(u, doc, cw, rs, prep_acc_tl):
    """The prologue's launch arguments (gdn_triton.py, PRO)."""
    B, L, C = u.shape
    return dict(qkv=u, doc=doc, cw=cw, rs=rs, L=L, C=C, KC=cw.shape[1], PACC=prep_acc_tl, PRO=True)


# ------------------------------------------------------------------------------------------------
# The Triton implementation (a GPU, or the interpreter): today's launches (ced/kernels/gdn.py, GatedDeltaRule and
# _QKVPrep), with q, k, v replaced by 1-element dummies of their dtype and the prologue's arguments added.


def _fwd_triton(u, doc, cw, g, beta, cu, ci, co, hk: int, hv: int, out_bf16: bool, f3: bool):
    import triton.language as tl

    from ced.kernels import gdn_triton as K_

    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    H, HV, K, V = hk, hv, d, d
    assert d == D_HEAD and C == (2 * hk + hv) * d and HV in (H, 2 * H), (u.shape, hk, hv)
    assert g.shape == beta.shape == (1, T, HV) and doc.shape == (B, L), (g.shape, beta.shape, doc.shape)
    G = HV // H
    comp, out, prep_acc, acc = _dtypes(u.dtype, cw.dtype, out_bf16)
    assert beta.dtype == out, (beta.dtype, out)
    PACC = tl.float64 if prep_acc == torch.float64 else tl.float32
    ACC = tl.float64 if acc == torch.float64 else tl.float32
    u, doc, cw = u.contiguous(), doc.contiguous(), cw.contiguous()
    _ran("forward", u, hk, hv)
    dev = u.device
    # the L2 norm's per-row factor of q and k: qkv_prep's forward grid and launch options
    pcf = G_.prep_config_for(out)["fwd"]
    rs = torch.empty(T, 2 * hk, dtype=prep_acc, device=dev)
    K_.prep_rs_kernel[((T + pcf["BT"] - 1) // pcf["BT"], 2 * hk)](u, doc, cw, rs, T, L, C=C, HK=hk, D=d, KC=cw.shape[1],
                                                                   ACC=PACC, EPS=1e-6, **pcf)
    pro = _pro_args(u, doc, cw, rs, PACC)
    qd = torch.empty(1, dtype=out, device=dev)  # the dummies of q, k and v: their dtype is read, never their memory
    kd, vd = qd, qd
    cap, N = ci.shape[0], co.shape[0] - 1
    g = g.to(acc).contiguous()
    beta = beta.contiguous()
    cf = G_.config_for(out)
    # today's GatedDeltaRule.forward, launch for launch (grids over the capacity)
    gc = torch.empty_like(g)
    gl = torch.empty_like(g) if f3 else None
    G_._launch(K_.cumsum_fwd_kernel, (cap, HV), cf, "cumsum", g, gc, gl, cu, ci, T, HV=HV, BT=CHUNK, ACC=ACC, F3=bool(f3))
    F3 = gl is not None
    A = torch.empty(1, T, HV, CHUNK, dtype=out, device=dev)
    G_._launch(K_.intra_fwd_kernel, (cap, HV // G_._gp(cf, "intra", G)), cf, "intra", kd, gc, gl, beta, A, cu, ci, T,
               H=H, G=G, K=K, BT=CHUNK, BC=16, ACC=ACC, SOLVE_PREC=K_.SOLVE_PRECISION, F3=F3, **pro)
    w, u_ = _wu(K_, cf, kd, vd, gc, gl, beta, A, cu, ci, cap, T, H, G, out, pro)
    h, v_new = _state_fwd(K_, cf, kd, w, u_, gc, gl, cu, co, cap, N, T, H, G, ACC, out, pro)
    o = torch.empty_like(v_new)
    G_._launch(K_.out_fwd_kernel, (cap, H), cf, "out", qd, kd, v_new, h, gc, gl, o, cu, ci, T,
               H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3, **pro)
    return o, rs, gc, (gl if F3 else g.new_empty(0)), A, w, h, v_new


def _wu(K_, cf, kd, vd, gc, gl, beta, A, cu, ci, cap, T, H, G, out, pro):
    """today's recompute_wu: w and u from A (the forward, and the backward's recompute)."""
    HV = H * G
    w = torch.empty(1, T, HV, D_HEAD, dtype=out, device=A.device)
    u_ = torch.empty(1, T, HV, D_HEAD, dtype=out, device=A.device)
    u_prec = G_.CONFIG["wu"].get("u_prec", "ieee") if out == torch.float32 else "tf32"
    G_._launch(K_.wu_fwd_kernel, (cap, H * G // G_._gp(cf, "wu", G)), cf, "wu", kd, vd, gc, gl, beta, A, w, u_, cu, ci, T,
               H=H, G=G, K=D_HEAD, V=D_HEAD, BT=CHUNK, U_PREC=u_prec, F3=gl is not None, **pro)
    return w, u_


def _state_fwd(K_, cf, kd, w, u_, gc, gl, cu, co, cap, N, T, H, G, ACC, out, pro):
    """today's state_fwd: the chunk states h (at the capacity; chunks past the last are never written) and v_new."""
    HV = H * G
    h = torch.empty(1, cap, HV, D_HEAD, D_HEAD, dtype=out, device=w.device)
    v_new = torch.empty_like(u_)
    BV = cf["state_fwd"]["BV"]
    G_._launch(K_.state_fwd_kernel, (D_HEAD // BV, N * HV), cf, "state_fwd", kd, u_, w, v_new, gc, gl, h, cu, co, T,
               H=H, G=G, K=D_HEAD, V=D_HEAD, BT=CHUNK, ACC=ACC, F3=gl is not None, **pro)
    return h, v_new


def _bwd_triton(do, u, doc, cw, beta, rs, gc, gl, A, w, h, v_new, cu, ci, co, hk: int, hv: int, out_bf16: bool):
    import triton.language as tl

    from ced.kernels import gdn_triton as K_

    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    H, HV, K, V = hk, hv, d, d
    G = HV // H
    comp, out, prep_acc, acc = _dtypes(u.dtype, cw.dtype, out_bf16)
    PACC = tl.float64 if prep_acc == torch.float64 else tl.float32
    ACC = tl.float64 if acc == torch.float64 else tl.float32
    gl = gl if gl.numel() else None
    F3 = gl is not None
    cap, N = ci.shape[0], co.shape[0] - 1
    u, doc, cw = u.contiguous(), doc.contiguous(), cw.contiguous()
    _ran("backward", u, hk, hv)
    dev = u.device
    pro = _pro_args(u, doc, cw, rs, PACC)
    qd = torch.empty(1, dtype=out, device=dev)
    kd, vd = qd, qd
    cf = G_.config_for(out)
    # today's GatedDeltaRule.backward (keep "all", no second-pass read: EXT False), launch for launch
    do = torch.zeros(1, T, HV, V, dtype=out, device=dev) if do is None else do.contiguous()
    dv_local = torch.empty_like(do)
    G_._launch(K_.dv_local_kernel, (cap, H), cf, "dv_local", qd, kd, gc, gl, do, dv_local, cu, ci, T,
               H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3, **pro)
    dh = torch.empty(1, cap, HV, K, V, dtype=out, device=dev)
    dv2 = torch.empty_like(dv_local)
    BV = cf["state_bwd"]["BV"]
    G_._launch(K_.state_bwd_kernel, (V // BV, N * HV), cf, "state_bwd", qd, kd, w, gc, gl, do, dv_local, dv2, dh,
               dh, dv_local, cu, co, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, EXT=False, F3=F3, **pro)
    dq = torch.empty(1, T, H, K, dtype=out, device=dev)
    dk_part = torch.empty(1, T, H, K, dtype=acc, device=dev)
    dw = torch.empty_like(w)
    dg_part = torch.empty(1, T, HV, dtype=acc, device=dev)
    G_._launch(K_.dqkwg_kernel, (cap, H), cf, "dqkwg", qd, kd, v_new, gc, gl, h, do, dh, dv2, dq, dk_part, dw, dg_part,
               cu, ci, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3, **pro)
    dk_wy = torch.empty(1, T, HV, K, dtype=acc, device=dev)
    dv = torch.empty(1, T, HV, V, dtype=out, device=dev)
    db = torch.empty_like(beta)
    dg = torch.empty(1, T, HV, dtype=acc, device=dev)
    G_._launch(K_.wy_bwd_kernel, (cap, HV // G_._gp(cf, "wy", G)), cf, "wy", kd, vd, beta, gc, gl, A, dw, dv2, dg_part,
               dk_wy, dv, db, dg, cu, ci, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3, **pro)
    dk = torch.empty(1, T, H, K, dtype=out, device=dev)
    BTs = cf["dk_sum"]["BT"]
    G_._launch(K_.dk_sum_kernel, ((T + BTs - 1) // BTs, H), cf, "dk_sum", dk_part, dk_wy, dk, T, H=H, G=G, K=K, ACC=ACC)
    # today's _QKVPrep.backward, launch for launch (dq, dk, dv in the input dtype, read into float32; da in float32)
    cfgs = G_.prep_config_for(out)
    cfg = cfgs["da"]
    nb = (T + cfg["BT"] - 1) // cfg["BT"]
    da = torch.empty(B, L, C, dtype=prep_acc, device=dev)
    dw_part = torch.empty(nb, cw.shape[1], C, dtype=prep_acc, device=dev)
    K_.prep_da_kernel[(nb, 2 * hk + hv)](u, doc, cw, dq, dk, dv, da, dw_part, T, L, C=C, HK=hk, HV=hv, D=d,
                                         KC=cw.shape[1], ACC=PACC, Q_SCALE=d**-0.5, EPS=1e-6, **cfg)
    du = torch.empty_like(u)
    cfg = cfgs["dx"]
    K_.prep_dx_kernel[((T + cfg["BT"] - 1) // cfg["BT"], C // d)](da, doc, cw, du, T, L, C=C, D=d, KC=cw.shape[1],
                                                                   ACC=PACC, **cfg)
    dcw = dw_part.sum(0).t().to(cw.dtype).contiguous()
    return du, dcw, dg.to(acc), db


# ------------------------------------------------------------------------------------------------
# The custom ops


@torch.library.custom_op("ced::gdn_edges_fwd", mutates_args=())
def gdn_edges_fwd(u: torch.Tensor, doc: torch.Tensor, cw: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                  cu: torch.Tensor, ci: torch.Tensor, co: torch.Tensor, hk: int, hv: int, out_bf16: bool, f3: bool
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor]:
    """(o, rs, gc, gl, A, w, h, v_new): o (1, B*L, hv, 128) the delta rule's output in the input dtype; the rest is what
    the backward reads (gl empty without F3). u (B, L, C) the in_proj_qkv output, doc (B, L) int64, cw (C, KC), g
    (1, B*L, hv) float32 (float64 with float64 inputs), beta (1, B*L, hv) in the input dtype; cu, ci, co the chunk
    metadata of chunk_meta (one computation serves every GDN layer of a forward: cached_chunk_meta)."""
    if _use_triton(u):
        return _fwd_triton(u, doc, cw, g, beta, cu, ci, co, hk, hv, out_bf16, f3)
    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    comp, out, prep_acc, acc = _dtypes(u.dtype, cw.dtype, out_bf16)
    o = _reference(u, doc, cw, g, beta, cu[:-1], hk, hv, out)
    z = lambda *s, dt=out: torch.zeros(*s, dtype=dt, device=u.device)  # noqa: E731
    return (o, z(T, 2 * hk, dt=prep_acc), g.to(acc).clone(), (z(1, T, hv, dt=acc) if f3 else z(0, dt=acc)),
            z(1, T, hv, CHUNK), z(1, T, hv, d), z(1, ci.shape[0], hv, d, d), z(1, T, hv, d))


@gdn_edges_fwd.register_fake
def _(u, doc, cw, g, beta, cu, ci, co, hk, hv, out_bf16, f3):
    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    comp, out, prep_acc, acc = _dtypes(u.dtype, cw.dtype, out_bf16)
    e = lambda *s, dt=out: u.new_empty(*s, dtype=dt)  # noqa: E731
    return (e(1, T, hv, d), e(T, 2 * hk, dt=prep_acc), e(1, T, hv, dt=acc), (e(1, T, hv, dt=acc) if f3 else e(0, dt=acc)),
            e(1, T, hv, CHUNK), e(1, T, hv, d), e(1, ci.shape[0], hv, d, d), e(1, T, hv, d))


@torch.library.custom_op("ced::gdn_edges_bwd", mutates_args=())
def gdn_edges_bwd(do: torch.Tensor, u: torch.Tensor, doc: torch.Tensor, cw: torch.Tensor, beta: torch.Tensor,
                  rs: torch.Tensor, gc: torch.Tensor, gl: torch.Tensor, A: torch.Tensor, w: torch.Tensor, h: torch.Tensor,
                  v_new: torch.Tensor, cu: torch.Tensor, ci: torch.Tensor, co: torch.Tensor, hk: int, hv: int,
                  out_bf16: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(du, dcw, dg, dbeta): the gradients wrt u (B, L, C), cw, g (1, B*L, hv) and beta, as today's path gives them."""
    if _use_triton(u):
        return _bwd_triton(do, u, doc, cw, beta, rs, gc, gl, A, w, h, v_new, cu, ci, co, hk, hv, out_bf16)
    # the CPU reference: the vector-Jacobian product of the forward again (gc holds g itself there); torch.func, since
    # autograd's dispatch keys are below a custom op's implementation
    comp, out, prep_acc, acc = _dtypes(u.dtype, cw.dtype, out_bf16)
    fn = lambda u_, cw_, g_, b_: _reference(u_, doc, cw_, g_, b_, cu[:-1], hk, hv, out)  # noqa: E731
    o, vjp = torch.func.vjp(fn, u, cw, gc, beta)
    return tuple(t.contiguous() for t in vjp(do.to(o.dtype)))


@gdn_edges_bwd.register_fake
def _(do, u, doc, cw, beta, rs, gc, gl, A, w, h, v_new, cu, ci, co, hk, hv, out_bf16):
    comp, out, prep_acc, acc = _dtypes(u.dtype, cw.dtype, out_bf16)
    return u.new_empty(u.shape), cw.new_empty(cw.shape), gc.new_empty(gc.shape, dtype=acc), beta.new_empty(beta.shape)


def _setup_context(ctx, inputs, output):
    u, doc, cw, g, beta, cu, ci, co, hk, hv, out_bf16, f3 = inputs
    o, rs, gc, gl, A, w, h, v_new = output
    ctx.save_for_backward(u, doc, cw, beta, rs, gc, gl, A, w, h, v_new, cu, ci, co)
    ctx.meta = (hk, hv, out_bf16)
    ctx.mark_non_differentiable(rs, gc, gl, A, w, h, v_new)


def _backward(ctx, do, *unused):
    hk, hv, out_bf16 = ctx.meta
    du, dcw, dg, db = torch.ops.ced.gdn_edges_bwd(do, *ctx.saved_tensors, hk, hv, out_bf16)
    return du, None, dcw, dg, db, None, None, None, None, None, None, None


gdn_edges_fwd.register_autograd(_backward, setup_context=_setup_context)


def check_process() -> None:
    """edges "fused" runs our Triton kernels (impl "triton") with keep "all"; the per-process A/B switches
    CED_GDN_IMPL and CED_GDN_KEEP must not ask for another (ced/kernels/gdn.py)."""
    impl = os.environ.get("CED_GDN_IMPL") or G_.DEFAULT_IMPL
    keep = os.environ.get("CED_GDN_KEEP") or G_.DEFAULT_KEEP
    if impl != "triton" or keep != "all":
        raise ValueError(f"GDNConfig.edges 'fused' runs our Triton kernels with keep 'all'; this process has impl {impl!r}, "
                         f"keep {keep!r} (CED_GDN_IMPL / CED_GDN_KEEP)")


def fused_delta_rule(u: torch.Tensor, doc: torch.Tensor, conv_weight: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                     cu_seqlens: torch.Tensor, hk: int, hv: int, d: int, out_dtype: torch.dtype | None = None,
                     f3: bool = False) -> torch.Tensor:
    """qkv_prep(u, doc, conv_weight, hk, hv, d, out_dtype) followed by the delta rule of kernel "ced" (the model's
    ced_delta_rule wrapper with these q, k, v), in one call whose q, k, v are never written. u (B, L, C) the in_proj_qkv
    output; doc (B, L) document ids; g, beta (B, L, hv), g in float32 (the wrapper casts as ced_delta_rule's does);
    cu_seqlens the varlen boundaries on u's device (no host copy is read). Returns o (B, L, hv, d) in the kernel's input
    dtype. Impl "triton" and keep "all" only (CED_GDN_IMPL / CED_GDN_KEEP; read outside a traced graph, and by the model
    at construction). The chunk metadata of these boundaries is formed once for every GDN layer of a forward
    (cached_chunk_meta), or in the graph under torch.compile."""
    if not torch.compiler.is_compiling():
        check_process()
    if not isinstance(f3, bool):
        raise TypeError(f"f3 must be a bool, not {f3!r}")
    B, L, C = u.shape
    assert d == D_HEAD and C == (2 * hk + hv) * d, (u.shape, hk, hv, d)
    assert out_dtype in (None, torch.bfloat16), out_dtype
    out_bf16 = out_dtype == torch.bfloat16
    comp, out, prep_acc, acc = _dtypes(u.dtype, conv_weight.dtype, out_bf16)
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    meta = chunk_meta(cu_seqlens, B * L)[:3] if torch.compiler.is_compiling() else cached_chunk_meta(cu_seqlens, B * L)
    o = gdn_edges_fwd(u, doc, conv_weight, flat(g).to(acc), flat(beta).to(out), *meta, hk, hv, out_bf16, f3)[0]
    return o.reshape(B, L, hv, d)
